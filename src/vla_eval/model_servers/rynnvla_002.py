# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "transformers==4.43.0",
#     "accelerate>=0.33",
#     "huggingface-hub>=0.23.4",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "sentencepiece==0.1.99",
#     "safetensors>=0.4.2",
#     "pyyaml>=6.0",
#     "requests>=2.32.3",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""RynnVLA-002 model server.

RynnVLA-002 is a Chameleon-based VLA from Alibaba DAMO Academy. The
reference repo is not packaged as an installable library for the modules used
by inference (``model`` / ``data``), so this server lazily clones the pinned
reference repo, downloads the upstream Chameleon tokenizer assets on first use,
and then imports the reference modules from that checkout.

The reference evaluation keeps action chunks *inside* the policy loop so image
history advances on every environment step even while actions are buffered.
This server mirrors that behavior with its own per-session action queue instead
of relying on ``PredictModelServer``'s chunk buffer.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

_RYNNVLA_REPO_URL = "https://github.com/alibaba-damo-academy/RynnVLA-002.git"
_RYNNVLA_REPO_REV = "e548ccc2977ba4fc06013de18192fa6245e454a4"
_CHAMELEON_TOKENIZER_HUB = "seliny2/Chameleon_7B_mGPT"
_LUMINA_TOKENIZER_HUB = "Alpha-VLLM/Lumina-mGPT-7B-768"
_CHAMELEON_TOKENIZER_FILES = {
    "original_tokenizers/text_tokenizer.json": "text_tokenizer.json",
    "original_tokenizers/vqgan.yaml": "vqgan.yaml",
    "original_tokenizers/vqgan.ckpt": "vqgan.ckpt",
}
_LIBERO_MIN_ACTION = np.asarray(
    [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0],
    dtype=np.float32,
)
_LIBERO_MAX_ACTION = np.asarray(
    [0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0],
    dtype=np.float32,
)


@dataclass(frozen=True)
class _HistorySpec:
    frames: int
    use_wrist: bool
    use_state: bool


@dataclass
class _Frame:
    front: Any
    wrist: Any | None = None


@dataclass
class _SessionState:
    history: deque[_Frame]
    pending_actions: deque[np.ndarray] = field(default_factory=deque)


@contextlib.contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _parse_history_spec(history_spec: str) -> _HistorySpec:
    parts = history_spec.split("_")
    if len(parts) < 5 or parts[0] != "his" or parts[-1] != "state":
        raise ValueError(f"Unsupported history_spec: {history_spec!r}")
    try:
        frames = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"history_spec has non-integer frame count: {history_spec!r}") from exc
    if frames < 1:
        raise ValueError(f"history_spec frame count must be >= 1: {history_spec!r}")

    state_prefix = parts[-2]
    if state_prefix not in {"w", "wo"}:
        raise ValueError(f"history_spec state suffix must be 'w_state' or 'wo_state': {history_spec!r}")

    view_tokens = parts[2:-2]
    return _HistorySpec(
        frames=frames,
        use_wrist="wrist" in view_tokens,
        use_state=state_prefix == "w",
    )


def _select_image(images: dict[str, Any], preferred_key: str | None, fallback_index: int) -> Any:
    if preferred_key and preferred_key in images:
        return images[preferred_key]
    values = list(images.values())
    if preferred_key and values:
        logger.debug("Image key %r missing; falling back to image index %d", preferred_key, fallback_index)
    if 0 <= fallback_index < len(values):
        return values[fallback_index]
    available = sorted(images) if isinstance(images, dict) else []
    raise KeyError(f"Could not find image {preferred_key!r}; available keys={available}")


def _to_pil_rgb(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


def _unnormalize_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    actions = actions[..., : len(_LIBERO_MIN_ACTION)]
    scale = _LIBERO_MAX_ACTION - _LIBERO_MIN_ACTION + 1e-8
    return ((actions + 1.0) / 2.0) * scale + _LIBERO_MIN_ACTION


def _build_request_item(
    *,
    task_description: str,
    history_spec: _HistorySpec,
    history: deque[_Frame],
    current_front: Any,
    current_wrist: Any | None,
    state: np.ndarray | None,
) -> dict[str, Any]:
    images: list[Any] = []
    past_frames = list(history)[-(history_spec.frames - 1) :] if history_spec.frames > 1 else []
    for frame in past_frames:
        images.append(frame.front)
        if history_spec.use_wrist:
            if frame.wrist is None:
                raise ValueError("history_spec requires wrist images, but a history frame is missing one")
            images.append(frame.wrist)

    images.append(current_front)
    if history_spec.use_wrist:
        if current_wrist is None:
            raise ValueError("history_spec requires wrist images, but the current observation has none")
        images.append(current_wrist)

    human_prompt = f"What action should the robot take to {task_description}?"
    if history_spec.use_state:
        human_prompt += "<|state|>"
    human_prompt += "<|image|>" * len(images)

    item: dict[str, Any] = {
        "conversations": [{"from": "human", "value": human_prompt}],
        "image": images,
        "action": [],
    }
    if history_spec.use_state:
        if state is None:
            raise ValueError("history_spec requires proprio state, but the observation has none")
        item["state"] = np.asarray(state, dtype=np.float32)
    return item


class RynnVLA002ModelServer(PredictModelServer):
    """RynnVLA-002 VLA model server for LIBERO-style observations."""

    def __init__(
        self,
        model_hub: str = "Alibaba-DAMO-Academy/RynnVLA-002",
        model_subfolder: str = "VLA_model_256/libero_goal",
        history_spec: str = "his_2_third_view_wrist_w_state",
        tokenizer_model: str = _LUMINA_TOKENIZER_HUB,
        chameleon_tokenizer_hub: str = _CHAMELEON_TOKENIZER_HUB,
        reference_repo_dir: str | None = None,
        repo_cache_dir: str | None = None,
        repo_url: str = _RYNNVLA_REPO_URL,
        repo_rev: str = _RYNNVLA_REPO_REV,
        front_image_key: str | None = "agentview",
        wrist_image_key: str | None = "wrist",
        *,
        chunk_size: int | None = None,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        # RynnVLA keeps chunking internal so history advances on every step.
        super().__init__(chunk_size=1, action_ensemble=action_ensemble, **kwargs)
        self.model_hub = model_hub
        self.model_subfolder = model_subfolder
        self.history_spec = _parse_history_spec(history_spec)
        self.tokenizer_model = tokenizer_model
        self.chameleon_tokenizer_hub = chameleon_tokenizer_hub
        self.reference_repo_dir = reference_repo_dir
        self.repo_cache_dir = repo_cache_dir
        self.repo_url = repo_url
        self.repo_rev = repo_rev
        self.front_image_key = front_image_key
        self.wrist_image_key = wrist_image_key
        self.action_horizon = chunk_size

        self._reference_repo_path: Path | None = None
        self._device = None
        self._model = None
        self._item_processor = None
        self._generation_config = None
        self._sessions: dict[str, _SessionState] = {}

    def _make_session_state(self) -> _SessionState:
        return _SessionState(history=deque(maxlen=max(self.history_spec.frames - 1, 0)))

    def _session_state(self, session_id: str) -> _SessionState:
        return self._sessions.setdefault(session_id, self._make_session_state())

    def get_observation_params(self) -> dict[str, Any]:
        return {
            "send_wrist_image": self.history_spec.use_wrist,
            "send_state": self.history_spec.use_state,
        }

    def _resolve_reference_repo(self) -> Path:
        if self._reference_repo_path is not None:
            return self._reference_repo_path

        if self.reference_repo_dir is not None:
            path = Path(self.reference_repo_dir).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"reference_repo_dir does not exist: {path}")
            self._reference_repo_path = path
            return path

        git = shutil.which("git")
        if git is None:
            raise RuntimeError("git is required to fetch the RynnVLA-002 reference repo")

        cache_root = Path(self.repo_cache_dir or "~/.cache/vla-eval/reference-repos").expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        target = cache_root / f"rynnvla-002-{self.repo_rev[:12]}"
        if target.exists():
            self._reference_repo_path = target
            return target

        tmp_dir = Path(tempfile.mkdtemp(prefix="rynnvla-002-", dir=cache_root))
        try:
            subprocess.run([git, "init"], cwd=tmp_dir, check=True, capture_output=True, text=True)
            subprocess.run(
                [git, "remote", "add", "origin", self.repo_url], cwd=tmp_dir, check=True, capture_output=True
            )
            subprocess.run(
                [git, "fetch", "--depth", "1", "origin", self.repo_rev],
                cwd=tmp_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([git, "checkout", "FETCH_HEAD"], cwd=tmp_dir, check=True, capture_output=True, text=True)
            try:
                tmp_dir.rename(target)
            except FileExistsError:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        self._reference_repo_path = target
        return target

    def _ensure_reference_imports(self) -> Path:
        repo_root = self._resolve_reference_repo()
        code_roots = [repo_root / "rynnvla-002", repo_root]
        for root in code_roots:
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
        return repo_root

    def _ensure_chameleon_tokenizer_assets(self, repo_root: Path) -> None:
        from huggingface_hub import hf_hub_download

        target_dir = repo_root / "rynnvla-002" / "ckpts" / "chameleon" / "tokenizer"
        target_dir.mkdir(parents=True, exist_ok=True)

        for hub_name, local_name in _CHAMELEON_TOKENIZER_FILES.items():
            target_path = target_dir / local_name
            if target_path.exists():
                continue
            source = Path(hf_hub_download(self.chameleon_tokenizer_hub, filename=hub_name))
            shutil.copy2(source, target_path)

    def _load_model(self) -> None:
        if self._model is not None:
            return

        repo_root = self._ensure_reference_imports()
        self._ensure_chameleon_tokenizer_assets(repo_root)

        import importlib

        import torch
        from transformers import GenerationConfig

        model_module = importlib.import_module("model")
        item_processor_module = importlib.import_module("data.pre_tokenize_action_state")
        model_cls = model_module.ChameleonXLLMXForConditionalGeneration_ck_action_head
        item_processor_cls = item_processor_module.ItemProcessor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": False,
        }
        if self.model_subfolder:
            model_kwargs["subfolder"] = self.model_subfolder

        if self._device.type == "cuda":
            try:
                import flash_attn  # noqa: F401
            except Exception:
                model_kwargs["attn_implementation"] = "sdpa"
            else:
                model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["attn_implementation"] = "sdpa"

        logger.info(
            "Loading RynnVLA-002 from %s%s on %s",
            self.model_hub,
            f"/{self.model_subfolder}" if self.model_subfolder else "",
            self._device,
        )
        self._model = model_cls.from_pretrained(self.model_hub, **model_kwargs).to(self._device).eval()
        if self.action_horizon is None:
            self.action_horizon = int(getattr(self._model.config, "time_horizon", 1))

        item_processor_cwd = repo_root / "rynnvla-002" / "evals_libero"
        with _temporary_cwd(item_processor_cwd):
            self._item_processor = item_processor_cls(tokenizer=self.tokenizer_model, target_size=256)

        self._generation_config = GenerationConfig(
            max_new_tokens=1,
            max_length=int(self._model.config.max_position_embeddings),
            temperature=1.0,
            top_k=None,
            do_sample=False,
            eos_token_id=[8710],
        )
        logger.info(
            "RynnVLA-002 loaded (history=%s, action_horizon=%s, wrist_image=%s, state=%s)",
            self.history_spec.frames,
            self.action_horizon,
            self.history_spec.use_wrist,
            self.history_spec.use_state,
        )

    def _predict_chunk(self, item: dict[str, Any]) -> np.ndarray:
        import torch

        self._load_model()
        assert self._model is not None
        assert self._item_processor is not None
        assert self._generation_config is not None
        assert self._device is not None

        tokens = self._item_processor.process_item(item, training_mode=False)
        tokens.append(10004)
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=self._device).unsqueeze(0)

        with torch.inference_mode():
            predicted = self._model.generate_action_head(input_ids, self._generation_config)

        actions = predicted.detach().float().cpu().numpy().astype(np.float32)
        actions = _unnormalize_actions(actions)
        if self.action_horizon is not None:
            actions = actions[: self.action_horizon]
        return actions

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        session = self._session_state(ctx.session_id)
        images = obs.get("images", {})
        if not isinstance(images, dict) or not images:
            raise ValueError("RynnVLA-002 requires observation images")

        front = _to_pil_rgb(_select_image(images, self.front_image_key, fallback_index=0))
        wrist = None
        if self.history_spec.use_wrist:
            wrist = _to_pil_rgb(_select_image(images, self.wrist_image_key, fallback_index=1))

        if not session.pending_actions:
            state = obs.get("states", obs.get("state"))
            item = _build_request_item(
                task_description=str(obs.get("task_description", "")),
                history_spec=self.history_spec,
                history=session.history,
                current_front=front,
                current_wrist=wrist,
                state=None if state is None else np.asarray(state, dtype=np.float32),
            )
            for action in self._predict_chunk(item):
                session.pending_actions.append(np.asarray(action, dtype=np.float32))
            if not session.pending_actions:
                raise RuntimeError("RynnVLA-002 returned an empty action chunk")

        action = session.pending_actions.popleft()
        session.history.append(_Frame(front=front, wrist=wrist))
        return {"actions": np.asarray(action, dtype=np.float32)}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        self._sessions[ctx.session_id] = self._make_session_state()
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self._sessions.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(RynnVLA002ModelServer)
