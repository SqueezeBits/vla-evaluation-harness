# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "simvla @ git+https://github.com/SqueezeBits/simvla.git@dd52bd2d716a93f11d675e0e10c3a4d9fcf1db8f",
#     "torch>=2.2",
#     "torchvision>=0.17",
#     "transformers>=4.57,<4.60",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "safetensors",
#     "scipy>=1.11",
#     "einops",
#     "timm",
#     "accelerate",
#     "peft",
#     "fastapi",
#     "uvicorn",
#     "json_numpy",
#     "websockets",
#     "opencv-python"
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# ///
"""WMVLA model server — Baseline Latent World Model VLA with future tokens.

WMVLA extends SimVLA with learnable future tokens that participate in the
action head's attention. At inference, the future tokens are present in the
sequence but their outputs are discarded — only action predictions are used.

Standalone server that inherits directly from PredictModelServer.
Dependencies are installed via ``pip install git+...`` from the simvla package.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

_SIMVLA_PKG_URL = "git+https://github.com/SqueezeBits/simvla.git"
_SIMVLA_PKG_REV = "dd52bd2d716a93f11d675e0e10c3a4d9fcf1db8f"


class WMVLAModelServer(PredictModelServer):
    """WMVLA (Baseline World Model VLA) model server.

    Loads a BaselineWMVLA checkpoint via the simvla package and runs
    inference with two camera views (primary + wrist), proprioceptive
    state, and a language instruction. Produces chunked 7D actions.

    Args:
        checkpoint: Path or HuggingFace repo ID for the WMVLA checkpoint.
        norm_stats: Explicit path to normalization statistics JSON file.
        smolvlm_model: SmolVLM base model path or HuggingFace repo ID.
        pkg_url: Git URL for the simvla package.
        pkg_rev: Git revision to pin for the simvla package.
        image_size: Input image resolution (square).
        chunk_size: Number of actions per inference call.
        action_ensemble: Strategy for blending overlapping action chunks.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        norm_stats: str | None = None,
        norm_stats_subdir: str = "norm_stats",
        norm_stats_filename: str = "libero_norm.json",
        smolvlm_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        pkg_url: str = _SIMVLA_PKG_URL,
        pkg_rev: str = _SIMVLA_PKG_REV,
        image_size: int = 384,
        chunk_size: int = 10,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.checkpoint = checkpoint
        self.norm_stats = norm_stats
        self.norm_stats_subdir = norm_stats_subdir
        self.norm_stats_filename = norm_stats_filename
        self.smolvlm_model = smolvlm_model
        self.pkg_url = pkg_url
        self.pkg_rev = pkg_rev
        self.image_size = image_size
        self._model = None
        self._processor = None
        self._device = None
        self._transform = None

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_CLOSE_POS}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": STATE_EEF_POS_AA_GRIP, "language": LANGUAGE}

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def _ensure_simvla_package(self) -> None:
        """Install the simvla package if not already importable."""
        try:
            importlib.import_module("simvla")
            return
        except ModuleNotFoundError:
            pass

        install_spec = f"{self.pkg_url}@{self.pkg_rev}"
        logger.info("Installing simvla package: pip install %s", install_spec)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", install_spec],
            stdout=subprocess.DEVNULL,
        )

    def _resolve_norm_stats(self) -> Path | None:
        """Resolve normalization statistics file.

        Resolution order:
        1. Explicit ``norm_stats`` path (if provided and exists).
        2. Bundled norm stats from the installed simvla package.
        """
        # 1. Explicit path
        if self.norm_stats is not None:
            p = Path(self.norm_stats).expanduser().resolve()
            if p.is_file():
                return p
            logger.warning("Explicit norm_stats path does not exist: %s — trying package data", p)

        # 2. Package data from installed simvla
        try:
            import importlib.resources as pkg_resources

            ref = pkg_resources.files("simvla") / self.norm_stats_subdir / self.norm_stats_filename
            with pkg_resources.as_file(ref) as pkg_path:
                if pkg_path.is_file():
                    return Path(pkg_path)
        except Exception:
            logger.debug("Could not resolve norm_stats from simvla package data", exc_info=True)

        return None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from torchvision import transforms

        self._ensure_simvla_package()

        # The file simvla.py in this directory shadows the installed simvla
        # package.  Temporarily remove paths that contain it so Python finds
        # the real package.
        _hidden: list[tuple[int, str]] = []
        for _i in range(len(sys.path) - 1, -1, -1):
            _candidate = Path(sys.path[_i]) / "simvla.py"
            if _candidate.is_file():
                _hidden.append((_i, sys.path.pop(_i)))
        # Invalidate any cached "simvla" module entry from the wrong location
        if "simvla" in sys.modules:
            del sys.modules["simvla"]
        try:
            from simvla.models.modeling_wmvla import BaselineWMVLA
            from simvla.models.processing_smolvlm_vla import SmolVLMVLAProcessor
        finally:
            for _i, _p in _hidden:
                sys.path.insert(_i, _p)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading WMVLA from %s on %s", self.checkpoint, self._device)

        self._model = BaselineWMVLA.from_pretrained(self.checkpoint)
        self._model = self._model.to(self._device)
        self._model.eval()

        self._processor = SmolVLMVLAProcessor.from_pretrained(self.smolvlm_model)

        norm_stats_path = self._resolve_norm_stats()
        if norm_stats_path is not None:
            logger.info("Loading norm stats from: %s", norm_stats_path)
            self._model.action_space.load_norm_stats(str(norm_stats_path))
        else:
            logger.warning("No norm_stats found — actions may be unnormalized")

        self._transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        logger.info("WMVLA model loaded (image_size=%d, chunk_size=%s)", self.image_size, self.chunk_size)

    def _preprocess_images(self, image0: np.ndarray, image1: np.ndarray) -> tuple[Any, Any]:
        """Preprocess two camera views into model input tensors."""
        import torch
        from PIL import Image as PILImage

        img0 = PILImage.fromarray(image0.astype(np.uint8)).convert("RGB")
        img1 = PILImage.fromarray(image1.astype(np.uint8)).convert("RGB")

        assert self._transform is not None
        img0_t = self._transform(img0)
        img1_t = self._transform(img1)

        # Pad to 3 views (model expects all views stacked)
        padding = torch.zeros_like(img0_t)
        images = torch.stack([img0_t, img1_t, padding], dim=0)
        image_mask = torch.tensor([[True, True, False]])

        return images.unsqueeze(0), image_mask

    def _extract_images(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Extract primary and wrist images from an observation."""
        images_dict = obs.get("images", {})
        image_keys = list(images_dict.keys()) if isinstance(images_dict, dict) else []
        if len(image_keys) >= 2:
            image0 = np.asarray(images_dict[image_keys[0]], dtype=np.uint8)
            image1 = np.asarray(images_dict[image_keys[1]], dtype=np.uint8)
        elif len(image_keys) == 1:
            image0 = np.asarray(images_dict[image_keys[0]], dtype=np.uint8)
            image1 = np.zeros_like(image0)
        else:
            image0 = (
                np.asarray(images_dict, dtype=np.uint8)
                if not isinstance(images_dict, dict)
                else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            )
            image1 = np.zeros_like(image0)
        return image0, image1

    def _extract_state(self, obs: Observation) -> np.ndarray:
        """Extract and pad/truncate 8D proprioceptive state."""
        state = np.asarray(obs.get("state", obs.get("states", np.zeros(8))), dtype=np.float32).flatten()
        if len(state) < 8:
            state = np.pad(state, (0, 8 - len(state)))
        return state[:8]

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        import torch

        self._load_model()
        assert self._model is not None
        assert self._processor is not None

        batch_size = len(obs_batch)
        all_images = []
        all_masks = []
        all_proprios = []
        prompts = []

        for obs in obs_batch:
            image0, image1 = self._extract_images(obs)
            state = self._extract_state(obs)
            prompts.append(obs.get("task_description", ""))

            images, image_mask = self._preprocess_images(image0, image1)
            all_images.append(images)
            all_masks.append(image_mask)
            all_proprios.append(torch.tensor(state, dtype=torch.float32))

        # Stack into batched tensors
        images_batch = torch.cat(all_images, dim=0).to(self._device)  # (B, 3, C, H, W)
        masks_batch = torch.cat(all_masks, dim=0).to(self._device)  # (B, 3)
        proprio_batch = torch.stack(all_proprios, dim=0).to(self._device)  # (B, 8)

        lang = self._processor.encode_language(prompts)
        lang = {k: v.to(self._device) for k, v in lang.items()}

        with torch.no_grad():
            actions = self._model.generate_actions(
                input_ids=lang["input_ids"],
                image_input=images_batch,
                image_mask=masks_batch,
                proprio=proprio_batch,
                steps=self.chunk_size or 10,
            )

        actions = actions.cpu().numpy()  # (B, chunk_size, 7)
        return [{"actions": actions[i]} for i in range(batch_size)]


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(WMVLAModelServer)
