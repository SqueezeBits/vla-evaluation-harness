# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.1",
#     "transformers>=4.40,<5",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate>=0.30",
#     "diffusers>=0.27",
#     "timm>=1.0",
#     "peft>=0.11",
#     "huggingface-hub>=0.23",
#     "sentencepiece>=0.1.99",
#     "einops",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-03-31T00:00:00Z"
# ///
"""FRAPPE model server.

FRAPPE is published as a GitHub repo plus Hugging Face checkpoint repos. The
source repo is not packaged on PyPI, so this server pins a source checkout,
adds it to ``sys.path`` lazily, patches one upstream Hub-loading bug, and runs
inference directly in-process.

The released RoboTwin checkpoints are action-chunking policies that condition on
six images (three cameras across two timesteps) plus the current 14-D dual-arm
joint state. The harness still sends one observation per environment step, so
this server keeps its own per-session observation window and action buffer
instead of relying on ``PredictModelServer`` chunk buffering.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

_DEFAULT_SOURCE_REPO = "https://github.com/OpenHelix-Team/frappe.git"
_DEFAULT_SOURCE_REF = "bc97186b9fe0c88ad1b450e1c571ac6ffcdb9c11"
_DEFAULT_CHECKPOINT = "hhhJB/frappe-robotwin-handover-mic"
_DEFAULT_TEXT_ENCODER = "google/t5-v1_1-xxl"
_DEFAULT_VISION_ENCODER = "google/siglip-so400m-patch14-384"
_DEFAULT_CAMERA_KEYS = ("head_camera", "right_camera", "left_camera")
_DEFAULT_LEFT_ARM_DIM = 6
_DEFAULT_RIGHT_ARM_DIM = 6
_DEFAULT_CONTROL_FREQUENCY = 25.0
_DEFAULT_RESIZE_WIDTH = 640
_DEFAULT_RESIZE_HEIGHT = 480


@dataclass
class _ObservationFrame:
    images: list[np.ndarray | None]
    joint_state: np.ndarray


@dataclass
class _SessionState:
    instruction: str = ""
    lang_embeddings: Any | None = None
    lang_attn_mask: Any | None = None
    observation_window: deque[_ObservationFrame] = field(default_factory=lambda: deque(maxlen=2))
    buffered_actions: deque[np.ndarray] = field(default_factory=deque)


@dataclass
class _AcceleratorShim:
    device: Any
    is_main_process: bool = True


def _parse_camera_keys(camera_keys: str | None) -> tuple[str, ...]:
    if camera_keys is None or not camera_keys.strip():
        return _DEFAULT_CAMERA_KEYS
    try:
        parsed = json.loads(camera_keys)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(camera_keys)
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError("camera_keys must decode to a non-empty list of camera names")
    return tuple(str(key) for key in parsed)


def _prepend_sys_path(path: Path) -> None:
    raw = str(path)
    if raw not in sys.path:
        sys.path.insert(0, raw)


def _resolve_cache_dir(cache_dir: str | None) -> Path:
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
    return Path.home().expanduser() / ".cache" / "vla-eval"


def _ensure_source_checkout(
    source_repo: str,
    source_ref: str,
    *,
    repo_dir: str | None,
    cache_dir: Path,
) -> Path:
    if repo_dir:
        path = Path(repo_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"repo_dir does not exist: {path}")
        return path

    source_root = cache_dir / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    checkout_dir = source_root / f"frappe-{source_ref[:12]}"

    if not (checkout_dir / ".git").exists():
        logger.info("Cloning FRAPPE source repo %s into %s", source_repo, checkout_dir)
        subprocess.run(["git", "clone", source_repo, str(checkout_dir)], check=True, capture_output=True, text=True)

    current_ref = None
    try:
        current_ref = subprocess.run(
            ["git", "-C", str(checkout_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        logger.warning("Failed to read current FRAPPE checkout ref for %s", checkout_dir)

    if current_ref != source_ref:
        logger.info("Checking out FRAPPE source ref %s", source_ref)
        subprocess.run(
            ["git", "-C", str(checkout_dir), "fetch", "origin", source_ref],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout_dir), "checkout", source_ref],
            check=True,
            capture_output=True,
            text=True,
        )

    return checkout_dir


def _resolve_checkpoint_dir(checkpoint: str, *, cache_dir: Path) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        parent = path.parent
        if (parent / "config.json").is_file():
            return parent.resolve()
        raise ValueError(
            f"checkpoint file {path} is missing a sibling config.json; "
            "pass a Hugging Face repo ID or a local FRAPPE checkpoint directory"
        )
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise ValueError(f"checkpoint directory {path} does not contain config.json")
        return path.resolve()

    from huggingface_hub import snapshot_download

    logger.info("Downloading FRAPPE checkpoint repo %s", checkpoint)
    local_dir = Path(snapshot_download(checkpoint, cache_dir=str(cache_dir / "hf")))
    if not (local_dir / "config.json").is_file():
        raise FileNotFoundError(f"Downloaded FRAPPE repo {checkpoint} has no config.json")
    return local_dir


def _pad_or_trim(vec: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).flatten()
    if arr.size == size:
        return arr
    if arr.size > size:
        return arr[:size]
    return np.pad(arr, (0, size - arr.size)).astype(np.float32)


def _to_rgb_array(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        arr = image
    else:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            arr = np.asarray(image.convert("RGB"))
        else:
            arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class _FRAPPEPolicy:
    def __init__(
        self,
        *,
        source_dir: Path,
        checkpoint_dir: Path,
        text_encoder: str,
        vision_encoder: str,
        left_arm_dim: int,
        right_arm_dim: int,
        control_frequency: float,
        use_bf16: bool,
        compile_experts: bool,
    ) -> None:
        self.source_dir = source_dir
        self.checkpoint_dir = checkpoint_dir
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.left_arm_dim = left_arm_dim
        self.right_arm_dim = right_arm_dim
        self.control_frequency = control_frequency
        self.use_bf16 = use_bf16
        self.compile_experts = compile_experts

        self._loaded = False
        self._device = None
        self._dtype = None
        self._policy = None
        self._vision_model = None
        self._image_processor = None
        self._text_embedder = None
        self._state_indices: list[int] | None = None
        self._state_token_dim = 128
        self._image_aspect_ratio = "pad"

    @property
    def state_dim(self) -> int:
        return self.left_arm_dim + self.right_arm_dim + 2

    def _load(self) -> None:
        if self._loaded:
            return

        import torch
        import torch.nn as nn
        import yaml
        from timm.models.vision_transformer import Mlp, RmsNorm

        if not torch.cuda.is_available():
            raise RuntimeError("FRAPPE inference requires CUDA; no GPU was detected")

        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16 if self.use_bf16 else torch.float32

        _prepend_sys_path(self.source_dir)
        _prepend_sys_path(self.source_dir / "models")

        from configs.state_vec import STATE_VEC_IDX_MAPPING
        from models.inference_runner import GateNetwork, MOEExpert, MOERDTRunner
        from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
        from models.multimodal_encoder.t5_encoder import T5Embedder

        class _PatchedMOERDTRunner(MOERDTRunner):
            def __init__(
                self,
                *,
                expert_configs: list[dict[str, Any]],
                action_dim: int,
                pred_horizon: int,
                lang_token_dim: int,
                img_token_dim: int,
                state_token_dim: int,
                max_lang_cond_len: int,
                img_cond_len: int,
                lang_pos_embed_config: Any = None,
                img_pos_embed_config: Any = None,
                dtype: Any = torch.bfloat16,
                resolution: int = 256,
                accelerator: Any = None,
                gate_hidden_dim: int = 256,
                use_lora: bool = False,
                lora_config: dict[str, Any] | None = None,
            ) -> None:
                super(MOERDTRunner, self).__init__()
                if accelerator is None:
                    raise ValueError("accelerator must be provided")

                self.accelerator = accelerator
                self.num_experts = len(expert_configs)
                self.pred_horizon = pred_horizon
                self.action_dim = action_dim
                self.max_lang_cond_len = max_lang_cond_len
                self.img_cond_len = img_cond_len
                self._expert_configs = expert_configs

                self.experts = nn.ModuleList()
                for i, expert_config in enumerate(expert_configs):
                    if accelerator.is_main_process:
                        logger.info(
                            "Create FRAPPE expert %d: enc_type=%s learnable_tokens=%s",
                            i + 1,
                            expert_config.get("enc_type"),
                            expert_config.get("learnable_tokens"),
                        )

                    common_kwargs = dict(
                        enc_type=expert_config.get("enc_type"),
                        resolution=resolution,
                        accelerator=accelerator,
                        learnable_tokens=expert_config.get("learnable_tokens"),
                        action_dim=action_dim,
                        pred_horizon=pred_horizon,
                        config=expert_config["config"],
                        lang_token_dim=lang_token_dim,
                        img_token_dim=img_token_dim,
                        state_token_dim=state_token_dim,
                        max_lang_cond_len=max_lang_cond_len,
                        img_cond_len=img_cond_len,
                        lang_pos_embed_config=lang_pos_embed_config,
                        img_pos_embed_config=img_pos_embed_config,
                        dtype=dtype,
                        device_id=None,
                    )

                    checkpoint_path = expert_config.get("checkpoint_path")
                    if checkpoint_path and os.path.exists(checkpoint_path):
                        expert = MOEExpert.from_pretrained(checkpoint_path, **common_kwargs)
                    else:
                        expert = MOEExpert(**common_kwargs)

                    if use_lora and lora_config:
                        expert.apply_lora(
                            lora_r=lora_config["r"],
                            lora_alpha=lora_config["alpha"],
                            lora_dropout=lora_config["dropout"],
                            target_modules=lora_config["target_modules"],
                        )
                    expert.eval()
                    self.experts.append(expert)

                self.gate_network = GateNetwork(
                    hidden_size=2048,
                    pred_horizon=pred_horizon,
                    num_experts=self.num_experts,
                    temperature=1.0,
                )
                self.streams = [torch.cuda.Stream() for _ in range(self.num_experts)]

                class _FusionFinalLayer(nn.Module):
                    def __init__(self, hidden_size: int, out_channels: int) -> None:
                        super().__init__()
                        self.norm_fusion = RmsNorm(hidden_size, eps=1e-6)

                        def approx_gelu() -> nn.GELU:
                            return nn.GELU(approximate="tanh")

                        self.ffn_fusion = Mlp(
                            in_features=hidden_size,
                            hidden_features=hidden_size,
                            out_features=out_channels,
                            act_layer=approx_gelu,
                            drop=0,
                        )

                    def forward(self, x: Any) -> Any:
                        return self.ffn_fusion(self.norm_fusion(x))

                self.fusion_final_layer = _FusionFinalLayer(2048, action_dim)
                self.gate_network = self.gate_network.to(accelerator.device)
                self.label_smoothing_epsilon = 0.1

        checkpoint_cfg = json.loads((self.checkpoint_dir / "config.json").read_text())
        base_cfg = yaml.safe_load((self.source_dir / "configs" / "base.yaml").read_text())
        self._state_token_dim = int(base_cfg["model"]["state_token_dim"])
        self._image_aspect_ratio = str(base_cfg["dataset"].get("image_aspect_ratio", "pad"))

        self._state_indices = [
            *[STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(self.left_arm_dim)],
            STATE_VEC_IDX_MAPPING["left_gripper_open"],
            *[STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(self.right_arm_dim)],
            STATE_VEC_IDX_MAPPING["right_gripper_open"],
        ]

        self._text_embedder = T5Embedder(
            from_pretrained=self.text_encoder,
            model_max_length=int(checkpoint_cfg.get("max_lang_cond_len", 1024)),
            device=str(self._device),
        )
        self._vision_model = SiglipVisionTower(vision_tower=self.vision_encoder, args=None)
        self._image_processor = self._vision_model.image_processor
        self._vision_model = self._vision_model.to(self._device, dtype=self._dtype).eval()

        accelerator = _AcceleratorShim(device=self._device)
        self._policy = _PatchedMOERDTRunner.from_pretrained(
            str(self.checkpoint_dir),
            expert_configs=checkpoint_cfg["expert_configs"],
            action_dim=int(checkpoint_cfg["action_dim"]),
            pred_horizon=int(checkpoint_cfg["pred_horizon"]),
            lang_token_dim=int(checkpoint_cfg["lang_token_dim"]),
            img_token_dim=int(checkpoint_cfg["img_token_dim"]),
            state_token_dim=int(checkpoint_cfg["state_token_dim"]),
            max_lang_cond_len=int(checkpoint_cfg["max_lang_cond_len"]),
            img_cond_len=int(checkpoint_cfg["img_cond_len"]),
            lang_pos_embed_config=checkpoint_cfg.get("lang_pos_embed_config"),
            img_pos_embed_config=checkpoint_cfg.get("img_pos_embed_config"),
            dtype=self._dtype,
            resolution=int(checkpoint_cfg.get("resolution", 256)),
            accelerator=accelerator,
            gate_hidden_dim=int(checkpoint_cfg.get("gate_hidden_dim", 256)),
            use_lora=bool(checkpoint_cfg.get("use_lora", False)),
            lora_config=checkpoint_cfg.get("lora_config"),
        )

        for expert in self._policy.experts:
            if hasattr(expert.model, "merge_and_unload"):
                expert.model = expert.model.merge_and_unload()
            if self.compile_experts and hasattr(torch, "compile"):
                expert.model = torch.compile(expert.model)

        self._policy = self._policy.to(self._device, dtype=self._dtype).eval()
        self._loaded = True

    def encode_instruction(self, instruction: str) -> tuple[Any, Any]:
        self._load()
        assert self._text_embedder is not None
        return self._text_embedder.get_text_embeddings([instruction])

    def _format_joint_to_state(self, joints: Any) -> tuple[Any, Any]:
        import torch

        assert self._state_indices is not None
        batch, steps, _ = joints.shape
        state = torch.zeros((batch, steps, self._state_token_dim), device=joints.device, dtype=joints.dtype)
        state[:, :, self._state_indices] = joints

        state_mask = torch.zeros((batch, self._state_token_dim), device=joints.device, dtype=joints.dtype)
        state_mask[:, self._state_indices] = 1
        return state, state_mask

    def _unformat_action_to_joint(self, action: Any) -> Any:
        assert self._state_indices is not None
        return action[:, :, self._state_indices]

    def predict_chunk(
        self,
        *,
        images: list[np.ndarray | None],
        joint_state: np.ndarray,
        lang_embeddings: Any,
        lang_attn_mask: Any,
    ) -> np.ndarray:
        import torch
        from PIL import Image as PILImage

        self._load()
        assert self._image_processor is not None
        assert self._policy is not None
        assert self._vision_model is not None
        assert self._device is not None
        assert self._dtype is not None

        image_processor = self._image_processor

        background_color = np.array(
            [int(x * 255) for x in image_processor.image_mean],
            dtype=np.uint8,
        ).reshape(1, 1, 3)
        background_image = (
            np.ones(
                (
                    self._image_processor.size["height"],
                    self._image_processor.size["width"],
                    3,
                ),
                dtype=np.uint8,
            )
            * background_color
        )

        def _expand_to_square(image: PILImage.Image) -> PILImage.Image:
            width, height = image.size
            if width == height:
                return image
            fill = tuple(int(x * 255) for x in image_processor.image_mean)
            if width > height:
                result = PILImage.new(image.mode, (width, width), fill)
                result.paste(image, (0, (width - height) // 2))
                return result
            result = PILImage.new(image.mode, (height, height), fill)
            result.paste(image, ((height - width) // 2, 0))
            return result

        image_tensors = []
        for image in images:
            if image is None:
                pil = PILImage.fromarray(background_image)
            else:
                pil = PILImage.fromarray(image).convert("RGB")
            if self._image_aspect_ratio == "pad":
                pil = _expand_to_square(pil)
            image_tensors.append(image_processor.preprocess(pil, return_tensors="pt")["pixel_values"][0])

        image_tensor = torch.stack(image_tensors, dim=0).to(self._device, dtype=self._dtype)
        image_embeds = self._vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self._vision_model.hidden_size).unsqueeze(0)

        joints = torch.as_tensor(joint_state, device=self._device, dtype=self._dtype).reshape(1, 1, -1)
        state_tokens, state_mask = self._format_joint_to_state(joints)
        ctrl_freqs = torch.tensor([self.control_frequency], device=self._device, dtype=torch.float32)

        trajectory = self._policy.predict_action(
            lang_tokens=lang_embeddings.to(self._device, dtype=self._dtype),
            lang_attn_mask=lang_attn_mask.to(self._device),
            img_tokens=image_embeds,
            state_tokens=state_tokens[:, -1:, :],
            action_mask=state_mask.unsqueeze(1).to(self._device, dtype=self._dtype),
            ctrl_freqs=ctrl_freqs,
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)
        return trajectory.squeeze(0).cpu().numpy()


class FRAPPEModelServer(PredictModelServer):
    """FRAPPE RoboTwin model server.

    User-facing ``chunk_size`` refers to the policy's predicted action horizon,
    not the base class' chunk buffer. The server keeps its own action buffer so
    it can still update FRAPPE's two-frame observation history on every step.
    """

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        *,
        source_repo: str = _DEFAULT_SOURCE_REPO,
        source_ref: str = _DEFAULT_SOURCE_REF,
        repo_dir: str | None = None,
        cache_dir: str | None = None,
        text_encoder: str = _DEFAULT_TEXT_ENCODER,
        vision_encoder: str = _DEFAULT_VISION_ENCODER,
        camera_keys: str | None = None,
        left_arm_dim: int = _DEFAULT_LEFT_ARM_DIM,
        right_arm_dim: int = _DEFAULT_RIGHT_ARM_DIM,
        control_frequency: float = _DEFAULT_CONTROL_FREQUENCY,
        chunk_size: int = 32,
        resize_width: int = _DEFAULT_RESIZE_WIDTH,
        resize_height: int = _DEFAULT_RESIZE_HEIGHT,
        apply_jpeg_reencode: bool = True,
        jpeg_quality: int = 95,
        use_bf16: bool = True,
        compile_experts: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=1, **kwargs)
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if left_arm_dim < 1 or right_arm_dim < 1:
            raise ValueError("left_arm_dim and right_arm_dim must be >= 1")

        self.checkpoint = checkpoint
        self.source_repo = source_repo
        self.source_ref = source_ref
        self.repo_dir = repo_dir
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.camera_keys = _parse_camera_keys(camera_keys)
        self.left_arm_dim = left_arm_dim
        self.right_arm_dim = right_arm_dim
        self.control_frequency = control_frequency
        self.action_chunk_size = chunk_size
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.apply_jpeg_reencode = apply_jpeg_reencode
        self.jpeg_quality = jpeg_quality
        self.use_bf16 = use_bf16
        self.compile_experts = compile_experts
        self.state_dim = left_arm_dim + right_arm_dim + 2

        self._sessions: dict[str, _SessionState] = {}
        self._policy: _FRAPPEPolicy | None = None

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_state": True}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        self._sessions[ctx.session_id] = self._new_session_state()
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self._sessions.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)

    def _new_session_state(self) -> _SessionState:
        state = _SessionState()
        state.observation_window.append(
            _ObservationFrame(
                images=[None for _ in self.camera_keys],
                joint_state=np.zeros(self.state_dim, dtype=np.float32),
            )
        )
        return state

    def _blank_image(self) -> np.ndarray:
        return np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8)

    def _process_image(self, image: np.ndarray | None) -> np.ndarray:
        from PIL import Image as PILImage

        arr = self._blank_image() if image is None else image
        pil = PILImage.fromarray(arr).convert("RGB")
        if pil.size != (self.resize_width, self.resize_height):
            pil = pil.resize((self.resize_width, self.resize_height), PILImage.Resampling.BILINEAR)
        if self.apply_jpeg_reencode:
            with BytesIO() as buf:
                pil.save(buf, format="JPEG", quality=self.jpeg_quality)
                buf.seek(0)
                pil = PILImage.open(buf).convert("RGB")
        return np.asarray(pil)

    def _extract_camera_arrays(self, obs: Observation) -> list[np.ndarray]:
        images_source = obs.get("images", {})
        if isinstance(images_source, dict):
            items = [(name, _to_rgb_array(value)) for name, value in images_source.items()]
        elif images_source is None:
            items = []
        else:
            items = [("image", _to_rgb_array(images_source))]

        selected: list[np.ndarray | None] = []
        used_names: set[str] = set()
        item_map = {name: value for name, value in items}

        for key in self.camera_keys:
            if key in item_map and item_map[key] is not None:
                selected.append(item_map[key])
                used_names.add(key)
            else:
                selected.append(None)

        fallbacks = [value for name, value in items if name not in used_names and value is not None]
        for i, image in enumerate(selected):
            if image is None and fallbacks:
                selected[i] = fallbacks.pop(0)

        last_real = next((image for image in reversed(selected) if image is not None), None)
        finalized: list[np.ndarray] = []
        for image in selected:
            finalized.append(self._process_image(image if image is not None else last_real))
        return finalized

    def _extract_joint_state(self, obs: Observation) -> np.ndarray:
        raw_state = obs.get("joint_state", obs.get("states", obs.get("state")))
        if raw_state is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        return _pad_or_trim(np.asarray(raw_state, dtype=np.float32), self.state_dim)

    def _load_policy(self) -> None:
        if self._policy is not None:
            return

        source_dir = _ensure_source_checkout(
            self.source_repo,
            self.source_ref,
            repo_dir=self.repo_dir,
            cache_dir=self.cache_dir,
        )
        checkpoint_dir = _resolve_checkpoint_dir(self.checkpoint, cache_dir=self.cache_dir)
        self._policy = _FRAPPEPolicy(
            source_dir=source_dir,
            checkpoint_dir=checkpoint_dir,
            text_encoder=self.text_encoder,
            vision_encoder=self.vision_encoder,
            left_arm_dim=self.left_arm_dim,
            right_arm_dim=self.right_arm_dim,
            control_frequency=self.control_frequency,
            use_bf16=self.use_bf16,
            compile_experts=self.compile_experts,
        )

    def _predict_chunk(self, session: _SessionState) -> np.ndarray:
        self._load_policy()
        assert self._policy is not None

        instruction = session.instruction or ""
        if session.lang_embeddings is None:
            session.lang_embeddings, session.lang_attn_mask = self._policy.encode_instruction(instruction)

        window = list(session.observation_window)
        if len(window) == 1:
            window = [window[0], window[0]]
        prev_frame, curr_frame = window[-2], window[-1]
        images = [*prev_frame.images, *curr_frame.images]
        actions = self._policy.predict_chunk(
            images=images,
            joint_state=curr_frame.joint_state,
            lang_embeddings=session.lang_embeddings,
            lang_attn_mask=session.lang_attn_mask,
        )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[0] < 1:
            raise ValueError(f"FRAPPE predict_chunk must return [T, D], got {actions.shape}")
        return actions[: self.action_chunk_size]

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        session = self._sessions.setdefault(ctx.session_id, self._new_session_state())

        instruction = str(obs.get("task_description", "") or "").strip()
        if instruction and instruction != session.instruction:
            session.instruction = instruction
            session.lang_embeddings = None
            session.lang_attn_mask = None

        images: list[np.ndarray | None] = list(self._extract_camera_arrays(obs))
        frame = _ObservationFrame(
            images=images,
            joint_state=self._extract_joint_state(obs),
        )
        session.observation_window.append(frame)

        if session.buffered_actions:
            return {"actions": session.buffered_actions.popleft()}

        actions = self._predict_chunk(session)
        for action in actions[1:]:
            session.buffered_actions.append(np.asarray(action, dtype=np.float32))
        return {"actions": np.asarray(actions[0], dtype=np.float32)}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(FRAPPEModelServer)
