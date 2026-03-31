# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "starvla",
#     "torch>=2.0",
#     "torchvision>=0.17",
#     "transformers>=4.57,<5",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "kernels>=0.11.0",
#     "qwen-vl-utils",
#     "omegaconf",
#     "rich",
#     "diffusers",
#     "timm",
#     "einops",
#     "scipy",
#     "huggingface-hub",
#     "tiktoken",
#     "torchcodec",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# starvla = { git = "https://github.com/ginwind/VLA-JEPA.git", rev = "2a16b6c4ed1f81a854d776f2760c48017e3bc49a" }
#
# [tool.uv]
# exclude-newer = "2026-03-30T00:00:00Z"
# ///
"""VLA-JEPA model server.

VLA-JEPA is published as a starVLA fork. The released Hugging Face repo stores
checkpoint variants in subdirectories (for example ``LIBERO/`` and
``SimplerEnv/``), each with a ``config.yaml``, ``dataset_statistics.json``, and
``checkpoints/*.pt``.

This server accepts either:
- a direct local checkpoint file, or
- a local/Hugging Face repo directory plus ``checkpoint_variant``.

Released configs also embed local training paths for the base Qwen-VL and
V-JEPA encoders, so we patch those to public Hugging Face IDs at load time.
"""

from __future__ import annotations

import ast
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

_SUPPORTED_CHECKPOINT_SUFFIXES = (".pt", ".safetensors")
_FRAMEWORK_ALIASES = {
    # Some upstream configs use older or informal names for the released model.
    "QwenJEVLA": "VLA_JEPA",
    "VLA-JEPA": "VLA_JEPA",
}
_BASE_VLM_BY_BASENAME = {
    "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
}
_BASE_ENCODER_BY_BASENAME = {
    "vjepa2-vitl-fpc64-256": "facebook/vjepa2-vitl-fpc64-256",
}
_DENORM_MODES = {"quantile", "minmax"}


class _AdaptiveEnsembler:
    """Match the official VLA-JEPA SimplerEnv overlap-ensemble behavior."""

    def __init__(self, pred_action_horizon: int, adaptive_ensemble_alpha: float = 0.0) -> None:
        self.pred_action_horizon = pred_action_horizon
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_history: deque[np.ndarray] = deque(maxlen=self.pred_action_horizon)

    def reset(self) -> None:
        self.action_history.clear()

    def ensemble_action(self, cur_action: np.ndarray) -> np.ndarray:
        self.action_history.append(np.asarray(cur_action, dtype=np.float32))
        num_actions = len(self.action_history)

        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for i, pred_actions in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        ref = curr_act_preds[num_actions - 1, :]
        dot_product = np.sum(curr_act_preds * ref, axis=1)
        norm_previous_pred = np.linalg.norm(curr_act_preds, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * curr_act_preds, axis=0).astype(np.float32)


def _load_observation_params(observation_params: str | dict[str, Any] | None) -> dict[str, Any]:
    if observation_params is None:
        return {}
    if isinstance(observation_params, str):
        try:
            return json.loads(observation_params)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(observation_params)
            if not isinstance(parsed, dict):
                raise ValueError(f"observation_params must decode to a dict, got {type(parsed).__name__}")
            return dict(parsed)
    return dict(observation_params)


def _looks_like_local_reference(path_str: str) -> bool:
    return path_str.startswith(("./", "../", "/", "~")) or path_str.count("/") >= 2


def _resolve_upstream_model_path(path_str: str, *, mapping: dict[str, str], fallback: str, label: str) -> str:
    basename = Path(path_str).name
    if basename in mapping:
        resolved = mapping[basename]
        logger.info("Resolved upstream %s path %r -> %s", label, path_str, resolved)
        return resolved
    logger.warning("Unknown upstream %s path %r; falling back to %s", label, path_str, fallback)
    return fallback


def _find_checkpoint_file(directory: Path, checkpoint_filename: str | None = None) -> Path:
    search_dirs = [directory / "checkpoints", directory]

    if checkpoint_filename is not None:
        for base in search_dirs:
            candidate = base / checkpoint_filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"checkpoint_filename={checkpoint_filename!r} was not found under {directory} or {directory / 'checkpoints'}"
        )

    candidates: list[Path] = []
    for base in search_dirs:
        if base.is_dir():
            candidates.extend(sorted(p for p in base.iterdir() if p.suffix in _SUPPORTED_CHECKPOINT_SUFFIXES))
    if candidates:
        return candidates[-1]

    variant_dirs = [p.name for p in directory.iterdir()] if directory.is_dir() else []
    raise FileNotFoundError(
        f"No checkpoint files found under {directory}. "
        f"If this is a multi-variant repo, pass checkpoint_variant. Contents: {variant_dirs}"
    )


def _resolve_checkpoint_path(
    checkpoint: str,
    *,
    checkpoint_variant: str | None = None,
    checkpoint_filename: str | None = None,
) -> str:
    path = Path(checkpoint).expanduser()
    if path.is_file() and path.suffix in _SUPPORTED_CHECKPOINT_SUFFIXES:
        return str(path)

    if path.is_dir():
        root = path / checkpoint_variant if checkpoint_variant else path
        return str(_find_checkpoint_file(root, checkpoint_filename=checkpoint_filename))

    from huggingface_hub import snapshot_download

    allow_patterns = [f"{checkpoint_variant}/**"] if checkpoint_variant else None
    logger.info("Downloading VLA-JEPA checkpoint repo %s (variant=%s)", checkpoint, checkpoint_variant or "<root>")
    local_dir = Path(snapshot_download(checkpoint, allow_patterns=allow_patterns))
    root = local_dir / checkpoint_variant if checkpoint_variant else local_dir
    return str(_find_checkpoint_file(root, checkpoint_filename=checkpoint_filename))


def _select_action_stats(norm_stats: dict[str, Any], unnorm_key: str | None) -> tuple[str, dict[str, Any]]:
    if unnorm_key is None:
        if len(norm_stats) != 1:
            raise ValueError(
                f"Model was trained on multiple datasets; pass unnorm_key from: {list(norm_stats.keys())}"
            )
        unnorm_key = next(iter(norm_stats))

    if unnorm_key not in norm_stats:
        raise ValueError(f"unnorm_key={unnorm_key!r} not found; available keys: {list(norm_stats.keys())}")
    return unnorm_key, norm_stats[unnorm_key]["action"]


def _resolve_denorm_range(action_stats: dict[str, Any], denorm_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if denorm_mode == "minmax":
        low_key, high_key = "min", "max"
        fallback_low_key, fallback_high_key = "q01", "q99"
    elif denorm_mode == "quantile":
        low_key, high_key = "q01", "q99"
        fallback_low_key, fallback_high_key = "min", "max"
    else:
        raise ValueError(f"Unsupported denorm_mode={denorm_mode!r}; expected one of {sorted(_DENORM_MODES)}")

    action_low = np.asarray(action_stats.get(low_key, action_stats.get(fallback_low_key)), dtype=np.float32)
    action_high = np.asarray(action_stats.get(high_key, action_stats.get(fallback_high_key)), dtype=np.float32)
    return action_low, action_high


def _postprocess_actions(
    normalized_actions: np.ndarray,
    action_stats: dict[str, Any],
    *,
    denorm_mode: str,
    invert_gripper: bool,
) -> np.ndarray:
    """Unnormalize actions with the checkpoint's recorded statistics.

    Assumption: released VLA-JEPA checkpoints follow the starVLA convention
    where the gripper channel is binary after unnormalization (0=close, 1=open).
    When ``invert_gripper`` is enabled, we convert that to the harness' common
    ``+1=close, -1=open`` convention.
    """

    actions = np.asarray(normalized_actions, dtype=np.float32).copy()
    actions = np.clip(actions, -1.0, 1.0)

    if actions.ndim != 2:
        raise ValueError(f"Expected normalized action chunk with shape [T, D], got {actions.shape}")

    if actions.shape[1] > 6:
        actions[:, 6] = np.where(actions[:, 6] < 0.5, 0.0, 1.0)

    action_low, action_high = _resolve_denorm_range(action_stats, denorm_mode)
    mask = np.asarray(action_stats.get("mask", np.ones_like(action_low, dtype=bool)), dtype=bool)
    actions = np.where(mask, 0.5 * (actions + 1.0) * (action_high - action_low) + action_low, actions)

    if invert_gripper and actions.shape[1] > 6:
        actions[:, 6] = 1.0 - 2.0 * actions[:, 6]

    return actions.astype(np.float32, copy=False)


class VLAJEPAModelServer(PredictModelServer):
    """Model server for the released VLA-JEPA checkpoints."""

    def __init__(
        self,
        checkpoint: str,
        *,
        checkpoint_variant: str | None = None,
        checkpoint_filename: str | None = None,
        unnorm_key: str | None = None,
        denorm_mode: str = "quantile",
        image_size: int = 224,
        invert_gripper: bool = True,
        adaptive_ensemble_horizon: int | None = None,
        adaptive_ensemble_alpha: float = 0.0,
        use_bf16: bool = False,
        observation_params: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.checkpoint_variant = checkpoint_variant
        self.checkpoint_filename = checkpoint_filename
        self.unnorm_key = unnorm_key
        if denorm_mode not in _DENORM_MODES:
            raise ValueError(f"Unsupported denorm_mode={denorm_mode!r}; expected one of {sorted(_DENORM_MODES)}")
        self.denorm_mode = denorm_mode
        self.image_size = image_size
        self.invert_gripper = invert_gripper
        if adaptive_ensemble_horizon is not None and adaptive_ensemble_horizon < 1:
            raise ValueError("adaptive_ensemble_horizon must be >= 1 when set")
        self.adaptive_ensemble_horizon = adaptive_ensemble_horizon
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.use_bf16 = use_bf16
        self._observation_params = _load_observation_params(observation_params)
        self._model = None
        self._action_stats: dict[str, Any] | None = None
        self._adaptive_ensemblers: dict[str, _AdaptiveEnsembler] = {}

    def get_observation_params(self) -> dict[str, Any]:
        return dict(self._observation_params)

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        self._adaptive_ensemblers.pop(ctx.session_id, None)
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self._adaptive_ensemblers.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        import transformers as tfm
        from starVLA.model.framework.base_framework import baseframework

        checkpoint_path = _resolve_checkpoint_path(
            self.checkpoint,
            checkpoint_variant=self.checkpoint_variant,
            checkpoint_filename=self.checkpoint_filename,
        )
        logger.info("Loading VLA-JEPA checkpoint from %s", checkpoint_path)

        patches: list[tuple[Any, str, Any]] = []

        def _patch_from_pretrained(cls_to_patch: Any) -> None:
            if cls_to_patch is None:
                return
            original = cls_to_patch.from_pretrained.__func__

            @classmethod
            def _patched(cls, *args: Any, **kwargs: Any) -> Any:
                if kwargs.get("attn_implementation") == "flash_attention_2":
                    kwargs["attn_implementation"] = "kernels-community/flash-attn2"
                return original(cls, *args, **kwargs)

            patches.append((cls_to_patch, "from_pretrained", classmethod(original)))
            cls_to_patch.from_pretrained = _patched

        _patch_from_pretrained(getattr(tfm, "Qwen2_5_VLForConditionalGeneration", None))
        _patch_from_pretrained(getattr(tfm, "Qwen3VLForConditionalGeneration", None))

        import importlib

        import starVLA.model.framework as fw_mod
        import starVLA.model.framework.base_framework as bf_mod

        importlib.import_module("starVLA.model.framework.VLA_JEPA")

        original_build_framework = fw_mod.build_framework

        def _patched_build_framework(cfg: Any) -> Any:
            framework_name = getattr(cfg.framework, "name", None) or getattr(cfg.framework, "framework_py", None)
            if framework_name in _FRAMEWORK_ALIASES:
                cfg.framework.name = _FRAMEWORK_ALIASES[framework_name]
                if hasattr(cfg.framework, "framework_py"):
                    cfg.framework.framework_py = _FRAMEWORK_ALIASES[framework_name]

            base_vlm = str(getattr(cfg.framework.qwenvl, "base_vlm", ""))
            if base_vlm and not Path(base_vlm).expanduser().exists():
                if _looks_like_local_reference(base_vlm) or Path(base_vlm).name in _BASE_VLM_BY_BASENAME:
                    cfg.framework.qwenvl.base_vlm = _resolve_upstream_model_path(
                        base_vlm,
                        mapping=_BASE_VLM_BY_BASENAME,
                        fallback="Qwen/Qwen3-VL-2B-Instruct",
                        label="base_vlm",
                    )

            base_encoder = str(getattr(cfg.framework.vj2_model, "base_encoder", ""))
            if base_encoder and not Path(base_encoder).expanduser().exists():
                if _looks_like_local_reference(base_encoder) or Path(base_encoder).name in _BASE_ENCODER_BY_BASENAME:
                    cfg.framework.vj2_model.base_encoder = _resolve_upstream_model_path(
                        base_encoder,
                        mapping=_BASE_ENCODER_BY_BASENAME,
                        fallback="facebook/vjepa2-vitl-fpc64-256",
                        label="base_encoder",
                    )

            return original_build_framework(cfg)

        patches.append((fw_mod, "build_framework", original_build_framework))
        fw_mod.build_framework = _patched_build_framework
        patches.append((bf_mod, "build_framework", original_build_framework))
        bf_mod.build_framework = _patched_build_framework

        try:
            self._model = baseframework.from_pretrained(checkpoint_path)
        finally:
            for obj, attr, original in reversed(patches):
                setattr(obj, attr, original)

        if self.use_bf16:
            self._model = self._model.to(torch.bfloat16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device).eval()

        norm_stats = self._model.norm_stats
        resolved_unnorm_key, self._action_stats = _select_action_stats(norm_stats, self.unnorm_key)
        logger.info(
            "VLA-JEPA model loaded on %s (variant=%s, unnorm_key=%s)",
            device,
            self.checkpoint_variant or "<root>",
            resolved_unnorm_key,
        )

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        from PIL import Image as PILImage

        self._load_model()
        assert self._model is not None
        assert self._action_stats is not None

        def _to_pil(image: Any) -> PILImage.Image:
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            pil_image = pil_image.convert("RGB")
            if pil_image.size != (self.image_size, self.image_size):
                pil_image = pil_image.resize((self.image_size, self.image_size), PILImage.Resampling.BILINEAR)
            return pil_image

        batch_images: list[list[PILImage.Image]] = []
        instructions: list[str] = []
        states: list[np.ndarray] = []
        saw_state = False
        saw_missing_state = False

        for obs in obs_batch:
            images_source = obs.get("images", {})
            if isinstance(images_source, dict):
                pil_images = [_to_pil(image) for image in images_source.values()]
            else:
                pil_images = [_to_pil(images_source)]
            batch_images.append(pil_images)
            instructions.append(str(obs.get("task_description", "")))

            raw_state = obs.get("states", obs.get("state"))
            if raw_state is None:
                saw_missing_state = True
                continue

            saw_state = True
            state = np.asarray(raw_state, dtype=np.float32).flatten().reshape(1, -1)
            states.append(state)

        if saw_state and saw_missing_state:
            raise ValueError("Mixed state availability within one VLA-JEPA batch is not supported")

        model_output = self._model.predict_action(
            batch_images=batch_images,
            instructions=instructions,
            state=states if saw_state else None,
        )
        normalized_actions_batch = np.asarray(model_output["normalized_actions"], dtype=np.float32)

        outputs: list[Action] = []
        for normalized_actions, ctx in zip(normalized_actions_batch, ctx_batch, strict=True):
            actions = _postprocess_actions(
                normalized_actions,
                self._action_stats,
                denorm_mode=self.denorm_mode,
                invert_gripper=self.invert_gripper,
            )
            if self.adaptive_ensemble_horizon is not None:
                ensembler = self._adaptive_ensemblers.get(ctx.session_id)
                if ensembler is None:
                    ensembler = _AdaptiveEnsembler(
                        pred_action_horizon=self.adaptive_ensemble_horizon,
                        adaptive_ensemble_alpha=self.adaptive_ensemble_alpha,
                    )
                    self._adaptive_ensemblers[ctx.session_id] = ensembler
                outputs.append({"actions": ensembler.ensemble_action(actions)})
            else:
                outputs.append({"actions": actions})
        return outputs


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(VLAJEPAModelServer)
