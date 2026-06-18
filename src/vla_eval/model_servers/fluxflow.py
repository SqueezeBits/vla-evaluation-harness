# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vla-eval",
#     "fluxflow[eval,raft,depth]",
# ]
# # The `depth` extra (xformers + depth-anything-3) powers the DA3 depth extractor
# # the depth IDM input modes (depth_flow / source_depth_flow) need. DA3 is heavy
# # — it drags in open3d + pycolmap — so it bloats this script's venv even for
# # non-depth configs, but it resolves cleanly against the rest of the stack
# # (numpy 2.x, torch 2.9.1/cu128); the older depth-anything-3 numpy<2 pin that
# # once forced a separate venv no longer applies. require_imports still raises a
# # clear hint if a depth dep somehow goes missing.
#
# # uv reads [tool.uv] overrides/sources/indexes only from the resolution root.
# # For a PEP 723 inline script that root is THIS block, not fluxflow's own
# # pyproject.toml, so the torch-stack overrides that make fluxflow installable
# # must be replicated here (lerobot 0.3.3 pins torchvision<0.23, which collides
# # with the torchvision 0.24.1 / cu128 stack fluxflow needs). Keep in sync with
# # fluxflow/pyproject.toml [tool.uv].
# [tool.uv]
# override-dependencies = [
#     "torch==2.9.1",
#     "torchvision==0.24.1",
#     "torchcodec>=0.9,<0.10",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# fluxflow = { path = "/workspace/code/fluxflow", editable = true }
# torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
# torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
# torchcodec = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""FluxFlow model server — Flux2 (optical-flow generator) + IDM (action decoder).

Wraps the combined ``FluxFlowPolicy`` (fluxflow repo) as a ``PredictModelServer``
so the harness LIBERO benchmark can drive it step-by-step. Flux2 generates a
future optical-flow image from the current scene + text (+ optional past-flow
and wrist-cam references); the IDM decodes an action chunk from (source image,
generated flow, proprio). Modeled on ``simvla.py`` / ``selfflow.py``.

The IDM may also consume extra input streams beyond the generated flow,
selected by its trained ``input_mode`` (read from the IDM config.json):
  - wrist modes (wrist_flow / source_wrist_flow): the current wrist-cam frame
    feeds the IDM directly. The server requests the wrist camera and passes its
    tensor to ``predict_from_flow``.
  - depth modes (depth_flow / source_depth_flow): a monocular depth model (DA3)
    runs on the agentview, and the same depth Representation training used
    (``depth_repr`` + kwargs) colormaps it into the depth image the IDM consumes.
    Only the current-frame ``depth_colormap`` is deployable; depth_delta /
    depth_stack need a future-frame depth that doesn't exist at rollout time.

When ``use_wrist_ref`` is set the server additionally passes the wrist frame to
Flux2 as an extra reference image (Image 3 with past-flow on, else Image 2),
filling the prompt template's ``{wrist_clause}`` — matching the Flux LoRA's
``--use_wrist_ref`` training recipe. This Flux conditioning flag is independent
of the IDM's wrist input mode: either one requests the wrist camera. Both must
agree with how flux_ckpt / the IDM were trained.

Convention (verified to match SimVLA/selfflow, which trained on the same LeRobot
LIBERO dataset):
  - proprio: 8-D ``[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]`` (axis-angle).
  - action: 7-D ``[Δxyz(3), Δaxis-angle(3), gripper]``, gripper +1=close.
  - action/proprio normalization is baked into the IDM checkpoint, so the server
    passes raw proprio through and returns raw actions (no separate stats file).
  - images arrive already 180°-flipped to upright by the benchmark, the same
    orientation fluxflow trained on, so no extra flip here.

``flux_ckpt`` may be a local LoRA/full-ft dir OR a HF Hub repo id; the base
weights load from ``base_model`` and the LoRA is detected + fused in
(``flux_mode=auto``), exactly as ``eval_policy.py`` does.

Launch via the harness:

    vla-eval serve --config configs/model_servers/fluxflow/libero.yaml
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import sys
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.image_utils import hconcat_uint8
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


@contextlib.contextmanager
def _unshadow_fluxflow() -> Iterator[None]:
    """Let ``import fluxflow`` resolve to the installed package, not this script.

    This single-file server is itself named ``fluxflow.py`` and is run directly,
    so its directory lands on ``sys.path[0]`` and shadows the real ``fluxflow``
    package (``import fluxflow`` finds the script -> "not a package"). Pop any
    path entry that holds a ``fluxflow.py`` for the duration of the import, drop
    a stale non-package ``sys.modules`` entry, then restore. Mirrors the guard in
    selfflow.py.
    """
    hidden: list[tuple[int, str]] = []
    for i in range(len(sys.path) - 1, -1, -1):
        if (Path(sys.path[i]) / "fluxflow.py").is_file():
            hidden.append((i, sys.path.pop(i)))
    # If "fluxflow" is bound to this script (a plain module, no __path__), drop it
    # so the real package can be imported; leave a real package import cached.
    mod = sys.modules.get("fluxflow")
    if mod is not None and not hasattr(mod, "__path__"):
        del sys.modules["fluxflow"]
    try:
        yield
    finally:
        for i, p in hidden:
            sys.path.insert(i, p)


# Flux2 only generates good flow for the prompt distribution its LoRA trained on,
# so rollout MUST wrap the LIBERO task with the same template the Flux run used.
#
# PROMPT_V2_TEMPLATE — the legacy configs/prompted_v2.yaml template (kept for
# models trained on it). Imperative "Given ... generate ..." framing.
PROMPT_V2_TEMPLATE = (
    "Given the current robot scene and an optical flow visualization of its "
    "previous {num_actions} actions, generate an optical flow visualization "
    "for the next {num_actions} actions performing the following task: "
    "{instruction}"
)

# PROMPT_V3_UV_TEMPLATE — configs/prompted_v3_uv.yaml verbatim: BFL-guide-aligned
# caption framing, references named by number (Image 1 = source, Image 2 =
# past-flow, Image 3 = wrist), <robot_flow> trigger token, uv colormap sentence.
# This is the default since the current models train on it (flow_repr=uv).
# {wrist_clause} is filled by _build_prompt (empty when use_wrist_ref is off).
# For the videojam encoding swap in the colormap sentence from prompted_v3_videojam.
PROMPT_V3_UV_TEMPLATE = (
    "<robot_flow> A dense optical-flow visualization of a robot's egocentric "
    "scene, rendered as a uv colormap where horizontal motion drives the red "
    "channel, vertical motion drives the green channel, and stationary regions "
    "appear mid-gray. Image 1 is the current observation. Image 2 is the same "
    "kind of optical-flow visualization for the previous {num_actions} frames, "
    "provided as a motion-history reference.{wrist_clause} The flow field shows "
    "the next {num_actions} frames of motion as the robot performs the task: "
    '"{instruction}".'
)

# PROMPT_V3_VIDEOJAM_TEMPLATE — configs/prompted_v3_videojam.yaml verbatim: same
# BFL-guide framing as v3_uv but with the HSV (videojam) colormap sentence. Pair
# with flow_repr=videojam so the colormap description matches the target.
PROMPT_V3_VIDEOJAM_TEMPLATE = (
    "<robot_flow> A dense optical-flow visualization of a robot's egocentric "
    "scene, rendered as a HSV flow-direction colormap where hue encodes motion "
    "angle, saturation encodes magnitude, and stationary regions are white. "
    "Image 1 is the current observation. Image 2 is the same kind of optical-flow "
    "visualization for the previous {num_actions} frames, provided as a "
    "motion-history reference.{wrist_clause} The flow field shows the next "
    '{num_actions} frames of motion as the robot performs the task: "{instruction}".'
)

# Named presets so a config can select a template by short alias
# (prompt_template: v3_uv) instead of pasting the full multi-line string into
# YAML. Any value not in this dict is treated as a literal template string, so
# configs that inline a custom template keep working unchanged.
PROMPT_PRESETS = {
    "v2": PROMPT_V2_TEMPLATE,
    "v3_uv": PROMPT_V3_UV_TEMPLATE,
    "v3_videojam": PROMPT_V3_VIDEOJAM_TEMPLATE,
}


def _read_idm_config(idm_dir: str) -> dict:
    """The IDM checkpoint's config dict (fields nested under 'cfg'), from a local
    checkpoint dir OR a HF Hub repo id.

    The policy's ``IDM.from_pretrained`` already accepts either form; this mirrors
    it so the server's pre-load reads (num_actions / input_mode) also work for a
    Hub id. ``Path(...).read_text`` is a local-only read, so fall back to
    ``hf_hub_download`` when config.json isn't on disk (i.e. idm_dir is a repo id).
    """
    local = Path(idm_dir) / "config.json"
    if local.is_file():
        text = local.read_text()
    else:
        from huggingface_hub import hf_hub_download

        text = Path(hf_hub_download(repo_id=idm_dir, filename="config.json")).read_text()
    data = json.loads(text)
    return data.get("cfg", data)


def _read_num_actions(idm_dir: str) -> int:
    """Read num_actions from the IDM checkpoint config (local dir or Hub repo id)."""
    return int(_read_idm_config(idm_dir)["num_actions"])


def _read_input_mode(idm_dir: str) -> str:
    """Read input_mode from the IDM checkpoint config (local dir or Hub repo id).

    Needed at construction time — before the heavy policy load — so the server
    can declare whether it must request the wrist camera (a wrist IDM input mode)
    and build a depth extractor (a depth IDM input mode). Mirrors the IDMConfig
    default for configs predating the field.
    """
    return str(_read_idm_config(idm_dir).get("input_mode", "source_flow"))


def _libero_suite_max_steps(suite: str) -> int | None:
    """Canonical rollout horizon for a LIBERO suite, used as the append_progress
    denominator so the progress fraction matches training.

    Pulled from the benchmark's ``MAX_STEP_MAPPING`` (single source of truth)
    rather than copied here. The import is lazy and cheap: the benchmark module
    only needs numpy+PIL at import time (the heavy LIBERO deps load lazily inside
    the benchmark), and vla-eval is always installed in this server's env.
    Returns ``None`` for an unknown suite or if the import fails.
    """
    try:
        from vla_eval.benchmarks.libero.benchmark import MAX_STEP_MAPPING
    except Exception:  # pragma: no cover - defensive; import is expected to work
        logger.warning("append_progress: could not import MAX_STEP_MAPPING", exc_info=True)
        return None
    return MAX_STEP_MAPPING.get(suite)


class FluxFlowModelServer(PredictModelServer):
    """Flux2 + IDM policy server. Produces chunked 7-D LIBERO actions per step."""

    def __init__(
        self,
        idm_dir: str,
        flux_ckpt: str,
        *,
        base_model: str = "black-forest-labs/FLUX.2-klein-base-4B",
        flux_mode: str = "auto",
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: int = 0,
        prompt_template: str = "v3_uv",
        append_progress: bool = False,
        rollout_max_steps: int | None = None,
        use_wrist_ref: bool = False,
        past_flow: bool = True,
        past_flow_source: str = "gt",
        flow_model: str = "raft",
        flow_repr: str = "uv",
        flow_sigma: float = 0.15,
        flow_scale: float = 0.0,
        depth_model: str = "da3",
        depth_model_name: str = "da3metric-large",
        depth_hf_repo: str | None = None,
        depth_compute_res: int = 504,
        depth_store_res: int = 256,
        depth_focal_px: float = 0.0,
        depth_metric_scale: float = 300.0,
        depth_repr: str = "depth_colormap",
        depth_cmap: str = "turbo",
        depth_power: float = 1.0,
        depth_invert: bool = True,
        depth_clip_pct: tuple[float, float] = (2.0, 98.0),
        raft_ckpt: str | None = None,
        raft_dir: str | None = None,
        raft_iters: int = 20,
        raft_small: bool = False,
        waft_cfg: str = "config/a1/tar-c-t.json",
        waft_ckpt: str | None = None,
        waft_dir: str | None = None,
        waft_da_ckpt: str | None = None,
        device: str = "cuda",
        action_ensemble: str = "newest",
        emit_flow: bool = True,
        flow_image_key: str = "predicted_flow",
        overlay_prompt: bool = True,
        caption_band_frac: float = 0.35,
        overlay_arrows: bool = True,
        arrow_grid: int = 20,
        arrow_scale: float = 1.0,
        arrow_min_frac: float = 0.01,
        flow_overlay: bool = True,
        flow_overlay_alpha: float = 0.5,
        flow_overlay_min_frac: float = 0.01,
        emit_inputs: bool = True,
        concat_inputs: bool = True,
        input_image_key: str = "model_input",
        past_flow_key: str = "past_flow",
        wrist_input_key: str = "wrist_input",
        depth_input_key: str = "depth_input",
        **kwargs: Any,
    ) -> None:
        # chunk_size MUST equal the training horizon and be known before the
        # first observation (PredictModelServer needs it to build the chunk
        # buffer). Derive it from the IDM checkpoint. Drop any stray chunk_size
        # forwarded by the auto-argparse MRO walk so we don't double-pass it.
        kwargs.pop("chunk_size", None)
        num_actions = _read_num_actions(idm_dir)
        super().__init__(chunk_size=num_actions, action_ensemble=action_ensemble, **kwargs)
        self.num_actions = num_actions

        # IDM input mode (read from config.json before the heavy load) decides
        # which streams the IDM itself consumes — independent of the Flux2
        # conditioning flags (use_wrist_ref / past_flow). These mirror
        # IDMConfig.uses_source / uses_wrist / uses_depth exactly. The 'no source'
        # modes (wrist_flow / depth_flow) drop the agentview from the IDM input
        # even though it still conditions Flux2 as Image 1.
        self.idm_input_mode = _read_input_mode(idm_dir)
        self.idm_uses_source = self.idm_input_mode in ("source_flow", "source_wrist_flow", "source_depth_flow")
        self.idm_uses_wrist = self.idm_input_mode in ("wrist_flow", "source_wrist_flow")
        self.idm_uses_depth = self.idm_input_mode in ("depth_flow", "source_depth_flow")

        self.idm_dir = idm_dir
        self.flux_ckpt = flux_ckpt
        self.base_model = base_model
        self.flux_mode = flux_mode
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        # Resolve a named preset (v2 / v3_uv / v3_videojam) to its template; any
        # other value is taken as a literal format string. MUST match the Flux
        # LoRA's training template, so the rollout prompt distribution matches.
        self.prompt_template = PROMPT_PRESETS.get(prompt_template, prompt_template)
        self.append_progress = append_progress
        self.rollout_max_steps = rollout_max_steps
        # Feed the wrist-cam frame at the current timestep as an extra Flux2
        # reference (Image N after source/past-flow), filling the template's
        # {wrist_clause}. Must match the Flux LoRA's --use_wrist_ref setting.
        self.use_wrist_ref = use_wrist_ref

        self.past_flow = past_flow
        if past_flow_source not in ("gt", "pred"):
            raise ValueError(f"past_flow_source must be 'gt' or 'pred', got {past_flow_source!r}")
        self.past_flow_source = past_flow_source
        self.flow_model_name = flow_model
        self.flow_repr = flow_repr
        self.flow_sigma = flow_sigma
        self.flow_scale = flow_scale
        self.raft_ckpt = raft_ckpt or str(Path.home() / ".cache/simvla/raft/checkpoints/raft-things.pth")
        self.raft_dir = raft_dir or str(Path.home() / ".cache/simvla/raft")
        self.raft_iters = raft_iters
        self.raft_small = raft_small
        self.waft_cfg = waft_cfg
        self.waft_ckpt = waft_ckpt or str(Path.home() / ".cache/simvla/waft/checkpoints/waft-a1.pth")
        self.waft_dir = waft_dir or str(Path.home() / ".cache/simvla/waft")
        self.waft_da_ckpt = waft_da_ckpt
        # Depth IDM stream (depth_flow / source_depth_flow): a monocular depth
        # model (DA3) run on the agentview at depth_compute_res, then the same
        # depth Representation training used (depth_repr + its kwargs) to colormap
        # the raw depth into the [-1, 1] image the IDM consumes. Must match the
        # --depth_repr / --depth_cmap / --depth_power the IDM trained on. Only
        # the current-frame depth_colormap is deployable online — depth_delta /
        # depth_stack also need t+num_actions depth, which doesn't exist at
        # rollout time — so _load_model rejects them.
        self.depth_model_name_kind = depth_model
        self.depth_model_name = depth_model_name
        self.depth_hf_repo = depth_hf_repo
        self.depth_compute_res = int(depth_compute_res)
        # Resolution the precompute cache stored depth at (scripts/precompute_depth.py
        # --store_res). Training colormaps the depth AFTER downsampling the raw
        # metric depth to this res, so rollout must downsample the raw depth here
        # too — before the colormap — for the IDM depth image to match training
        # (the turbo LUT is nonlinear and robust_normalize is resolution-sensitive,
        # so downsample-then-colormap != colormap-then-resize). Keep equal to the
        # IDM image_size used at train time (256 for the exp configs).
        self.depth_store_res = int(depth_store_res)
        self.depth_focal_px = depth_focal_px
        self.depth_metric_scale = depth_metric_scale
        self.depth_repr = depth_repr
        self.depth_cmap = depth_cmap
        self.depth_power = depth_power
        self.depth_invert = depth_invert
        self.depth_clip_pct = tuple(depth_clip_pct)
        self.device = device
        # When True, predict() attaches the Flux2-generated flow image to each
        # action response under extra_images[flow_image_key]; the harness records
        # it as a separate video stream when --save-traj is on, so the predicted
        # flow can be viewed alongside the rollout. Set false for max throughput.
        self.emit_flow = emit_flow
        self.flow_image_key = flow_image_key
        # When emitting flow, render the templated prompt in a caption band
        # appended BELOW the visualization (a COPY only — never the flow cached
        # for past-flow conditioning), so the prompt never occludes the flow.
        self.overlay_prompt = overlay_prompt
        # Caption band height as a fraction of the flow image height. Fixed (not
        # derived from the per-frame line count) so every frame keeps identical
        # dimensions even as the wrapped prompt shifts with the progress %, which
        # the video encoder requires; the font auto-fits the text into the band.
        self.caption_band_frac = caption_band_frac
        # Quiver overlay: decode the RGB flow image back to (u, v) pixel
        # displacements and draw a grid of direction arrows onto the emitted
        # visualization (again a COPY only). arrow_grid ~= arrows along the short
        # edge; arrow_scale stretches the drawn length; arrow_min_frac hides cells
        # whose mean displacement is below that fraction of the image diagonal.
        self.overlay_arrows = overlay_arrows
        self.arrow_grid = arrow_grid
        self.arrow_scale = arrow_scale
        self.arrow_min_frac = arrow_min_frac
        # Add a second column to the predicted-flow video: the generated flow
        # alpha-composited onto the agentview observation. Stationary (gray) flow
        # regions stay fully transparent so the scene shows through; regions with
        # valid motion (decoded |displacement| >= flow_overlay_min_frac of the
        # image diagonal) are blended at flow_overlay_alpha. Lets the predicted
        # motion be read against the real scene alongside the raw flow.
        self.flow_overlay = flow_overlay
        self.flow_overlay_alpha = flow_overlay_alpha
        self.flow_overlay_min_frac = flow_overlay_min_frac
        # When True, also record the model's *inputs* as extra video streams: the
        # preprocessed observation actually fed to the model (input_image_key) and
        # the past-flow reference used for conditioning (past_flow_key, only when
        # past_flow is on). Recorded raw (no arrows/prompt) so they show exactly
        # what the model received. Set false for max throughput.
        self.emit_inputs = emit_inputs
        # When True (default), every input modality the model consumes this step
        # (preprocessed observation + wrist + past-flow + depth) is stitched
        # left-to-right into a single strip recorded under input_image_key, so
        # --save-traj produces one "model input" video instead of one directory
        # per modality. Set false to keep each modality as its own video stream.
        self.concat_inputs = concat_inputs
        self.input_image_key = input_image_key
        self.past_flow_key = past_flow_key
        self.wrist_input_key = wrist_input_key
        self.depth_input_key = depth_input_key

        self._policy = None
        self._flow_model = None
        # Depth model + Representation transform, built in _load_model only when
        # the IDM consumes a depth stream.
        self._depth_model = None
        self._depth_rep = None
        # Per-session past-flow history, keyed by the absolute env step each entry
        # was produced at: session_id -> deque[(step, {"src": tensor[1,3,S,S] [-1,1]
        # | "flow_pil": PIL})]. The past-flow reference is the entry from
        # ``num_actions`` steps ago, so the conditioning window stays a fixed
        # ``num_actions``-step interval no matter how often we replan
        # (``execute_size`` < ``num_actions`` for overlapping chunks). Cleared each
        # episode so the first ``num_actions`` steps whiteout.
        self._past: dict[str, deque[tuple[int, dict[str, Any]]]] = {}
        # Per-session append_progress denominator, resolved from the LIBERO suite
        # at episode start (MAX_STEP_MAPPING[suite]). rollout_max_steps overrides it.
        self._session_max_steps: dict[str, int] = {}

    # ---- spec declarations -------------------------------------------------
    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    @property
    def _needs_wrist(self) -> bool:
        """The wrist camera is required when it conditions Flux2 (use_wrist_ref)
        or feeds the IDM directly (a wrist input mode) — either is enough."""
        return self.use_wrist_ref or self.idm_uses_wrist

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {
            "agentview": IMAGE_RGB,
            "state": STATE_EEF_POS_AA_GRIP,
            "language": LANGUAGE,
        }
        if self._needs_wrist:
            spec["wrist"] = IMAGE_RGB
        return spec

    def get_observation_params(self) -> dict[str, Any]:
        # agentview + proprio always; wrist only when conditioning on it or
        # feeding it to the IDM (the benchmark adds obs["images"]["wrist"] when
        # send_wrist_image is True). Depth needs no extra camera — it is computed
        # monocularly from the agentview.
        return {"send_wrist_image": self._needs_wrist, "send_state": True}

    # ---- loading -----------------------------------------------------------
    def _build_flow_extractor(self):
        """Mirror eval_policy.build_flow_extractor."""
        with _unshadow_fluxflow():
            from fluxflow.flow_extractors import load_flow_model

        flow_kw: dict = {"device": self.device, "scale": self.flow_scale}
        if self.flow_model_name == "waft":
            flow_kw.update(
                cfg=self.waft_cfg,
                ckpt=self.waft_ckpt,
                waft_dir=self.waft_dir,
                depth_anything_ckpt=self.waft_da_ckpt,
            )
        elif self.flow_model_name == "raft":
            flow_kw.update(
                ckpt=self.raft_ckpt,
                raft_dir=self.raft_dir,
                iters=self.raft_iters,
                small=self.raft_small,
            )
        return load_flow_model(self.flow_model_name, **flow_kw)

    def _build_depth_extractor(self):
        """Build the monocular depth model + depth Representation (mirrors
        train_idm's depth path). Only the current-frame ``depth_colormap`` is
        deployable online; depth_delta / depth_stack additionally need the depth
        at t+num_actions, which doesn't exist at rollout time."""
        if self.depth_repr != "depth_colormap":
            raise ValueError(
                f"depth_repr={self.depth_repr!r} reads a future-frame depth "
                "(depth at t+num_actions) that is unavailable at rollout time; "
                "only 'depth_colormap' (current-frame depth) is deployable. "
                "Retrain the IDM with --depth_repr depth_colormap to evaluate it."
            )
        with _unshadow_fluxflow():
            from fluxflow.depth_extractors import load_depth_model
            from fluxflow.representations import build_representation

        depth_kw: dict = {"device": self.device}
        if self.depth_model_name_kind == "da3":
            depth_kw.update(
                model_name=self.depth_model_name,
                hf_repo=self.depth_hf_repo,
                metric_scale=self.depth_metric_scale,
                focal_px=self.depth_focal_px,
            )
        model = load_depth_model(self.depth_model_name_kind, **depth_kw)
        rep = build_representation(
            {
                "type": self.depth_repr,
                "cmap": self.depth_cmap,
                "power": self.depth_power,
                "invert": self.depth_invert,
                "clip_pct": self.depth_clip_pct,
                "out_key": "depth_image",
            }
        )
        return model, rep

    def _load_model(self) -> None:
        """Build the policy (+ flow extractor). Called eagerly by run_server at
        startup (and idempotently guarded), so the heavy load happens before the
        benchmark connects rather than on the first observation."""
        if self._policy is not None:
            return
        with _unshadow_fluxflow():
            from fluxflow.policy import FluxFlowPolicy

        logger.info(
            "Loading FluxFlowPolicy (idm=%s, flux=%s, mode=%s) on %s",
            self.idm_dir,
            self.flux_ckpt,
            self.flux_mode,
            self.device,
        )
        self._policy = FluxFlowPolicy.from_checkpoints(
            idm_dir=self.idm_dir,
            flux_ckpt=self.flux_ckpt,
            base_model=self.base_model,
            flux_mode=self.flux_mode,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
            device=self.device,
        )
        # Trust the checkpoint over the CLI for the chunk length (parity with
        # eval_policy, which derives num_actions from the IDM config).
        ck = int(self._policy.idm.cfg.num_actions)
        if ck != self.num_actions:
            logger.warning(
                "num_actions from config.json (%d) != idm.cfg (%d); using %d",
                self.num_actions,
                ck,
                ck,
            )
            self.num_actions = ck
            self.chunk_size = ck
        if self.past_flow and self.past_flow_source == "gt":
            logger.info(
                "Building flow extractor for GT past-flow: %s -> %s (sigma=%s)",
                self.flow_model_name,
                self.flow_repr,
                self.flow_sigma,
            )
            self._flow_model = self._build_flow_extractor()
        # Cross-check the Flux conditioning flag against the trained IDM: a wrist
        # IDM forces the wrist camera on regardless of use_wrist_ref.
        if self.idm_uses_wrist and not self.use_wrist_ref:
            logger.info(
                "IDM input_mode=%s consumes the wrist camera (feeding it to the "
                "IDM, not Flux2); requesting the wrist image.",
                self.idm_input_mode,
            )
        if self.idm_uses_depth:
            logger.info(
                "Building depth extractor for the IDM depth stream: %s/%s -> %s "
                "(cmap=%s, power=%s, compute_res=%d, store_res=%d)",
                self.depth_model_name_kind,
                self.depth_model_name,
                self.depth_repr,
                self.depth_cmap,
                self.depth_power,
                self.depth_compute_res,
                self.depth_store_res,
            )
            self._depth_model, self._depth_rep = self._build_depth_extractor()
        logger.info(
            "FluxFlow server ready (chunk_size=%d, image_size=%d, past_flow=%s/%s, "
            "wrist_ref=%s, idm_input_mode=%s, append_progress=%s)",
            self.num_actions,
            self._policy.image_size,
            self.past_flow,
            self.past_flow_source if self.past_flow else "off",
            f"on (Image {3 if self.past_flow else 2})" if self.use_wrist_ref else "off",
            self.idm_input_mode,
            self.append_progress,
        )

    # ---- observation parsing ----------------------------------------------
    def _extract_primary_image(self, obs: Observation) -> np.ndarray:
        """First (agentview) camera as uint8 HWC RGB."""
        images = obs.get("images", {})
        if isinstance(images, dict) and images:
            key = "agentview" if "agentview" in images else next(iter(images))
            return np.asarray(images[key], dtype=np.uint8)
        if not isinstance(images, dict):
            return np.asarray(images, dtype=np.uint8)
        raise ValueError("observation has no images")

    def _extract_wrist_image(self, obs: Observation) -> np.ndarray:
        """Wrist camera as uint8 HWC RGB (already upright from the benchmark)."""
        images = obs.get("images", {})
        if isinstance(images, dict) and "wrist" in images:
            return np.asarray(images["wrist"], dtype=np.uint8)
        raise ValueError(
            "use_wrist_ref is on but the observation has no 'wrist' image; "
            "ensure get_observation_params requested send_wrist_image=True"
        )

    def _extract_state(self, obs: Observation) -> np.ndarray:
        """Raw 8-D proprio (axis-angle); pad/truncate like simvla/selfflow."""
        state = np.asarray(obs.get("state", obs.get("states", np.zeros(8))), dtype=np.float32).flatten()
        if len(state) < 8:
            state = np.pad(state, (0, 8 - len(state)))
        return state[:8]

    def _build_prompt(self, instruction: str, ctx: SessionContext) -> str:
        instr = instruction
        if self.append_progress:
            # Denominator priority: explicit rollout_max_steps override, else the
            # suite's MAX_STEP_MAPPING value resolved at episode start.
            max_steps = self.rollout_max_steps or self._session_max_steps.get(ctx.session_id)
            if max_steps and max_steps > 1:
                pct = int(round(100 * ctx.step / (max_steps - 1)))
                pct = max(0, min(100, pct))
            else:
                pct = 0
            instr = f"{instr.rstrip('.')}. The task is {pct}% complete."
        return self.prompt_template.format(
            instruction=instr,
            num_actions=self.num_actions,
            wrist_clause=self._wrist_clause(),
        )

    def _wrist_clause(self) -> str:
        """The '{wrist_clause}' substitution, mirroring the dataset's
        _format_instruction: an 'Image N is a wrist-camera view ...' sentence
        when wrist conditioning is on (N = 3 with past-flow, else 2), empty
        otherwise. Templates without the placeholder ignore the kwarg, so this
        is safe for the v2 template too.
        """
        if not self.use_wrist_ref:
            return ""
        wrist_idx = 3 if self.past_flow else 2
        return f" Image {wrist_idx} is a wrist-camera view of the same scene at the current timestep."

    # ---- past-flow conditioning -------------------------------------------
    def _whiteout_pil(self):
        """No-motion fill at episode start: uv -> mid-gray (0.0), videojam -> white."""
        import torch

        s = self._policy.image_size
        val = 0.0 if self.flow_repr == "uv" else 1.0
        fill = torch.full((3, s, s), val, dtype=torch.float32)
        return self._policy._flow_to_pil(fill)

    def _past_ref(self, sid: str, step: int) -> dict[str, Any] | None:
        """Cached entry from exactly ``num_actions`` steps ago, or ``None``.

        The past-flow window must stay a fixed ``num_actions``-step interval to
        match training, independent of ``execute_size`` (overlapping replan runs
        ``predict()`` more often than every ``num_actions`` steps). We therefore
        reference the entry whose step == ``step - num_actions`` rather than the
        most recent one. When ``execute_size`` does not divide ``num_actions``
        that exact step was never a replan step (``predict()`` never ran there),
        so no entry exists and the caller whiteouts.
        """
        hist = self._past.get(sid)
        if not hist:
            return None
        target = step - self.num_actions
        for s, entry in hist:
            if s == target:
                return entry
        return None

    def _build_past_flow_pil(self, sid: str, src, step: int):
        """Past-flow reference PIL for the current step (or whiteout at episode start)."""
        with _unshadow_fluxflow():
            from fluxflow.flow_extractors import extract_flow_rgb

        prev = self._past_ref(sid, step)
        if self.past_flow_source == "pred":
            if prev is None or prev.get("flow_pil") is None:
                return self._whiteout_pil()
            return prev["flow_pil"]
        # gt: real flow(obs[t-num_actions] -> current_obs) via the extractor, as in training.
        if prev is None or prev.get("src") is None:
            return self._whiteout_pil()
        # extract_flow_rgb/RAFT require inputs on the flow model's device (unlike
        # predict_from_flow, which moves src internally); _preprocess_image yields CPU.
        dev = self._flow_model.device
        flow_rgb = extract_flow_rgb(
            self._flow_model,
            prev["src"].to(dev),
            src.to(dev),
            sigma=self.flow_sigma,
            mode=self.flow_repr,
        )  # [1,3,S,S] in [-1,1]
        return self._policy._flow_to_pil(flow_rgb[0])

    # ---- depth conditioning -----------------------------------------------
    def _build_depth_image(self, img: np.ndarray):
        """Colormapped depth image for the current agentview frame.

        Runs the monocular depth model on the agentview at depth_compute_res
        (DA3 requires sides divisible by its ViT patch size), then the depth
        Representation training used. Returns a ``[1, 3, S, S]`` tensor in
        ``[-1, 1]`` at the IDM image_size — the depth stream predict_from_flow
        feeds the IDM. ``img`` is the uint8 HWC RGB agentview.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image as _Image

        with _unshadow_fluxflow():
            from fluxflow.depth_extractors import extract_depth_raw

        res = self.depth_compute_res
        pil = _Image.fromarray(img.astype(np.uint8), mode="RGB").resize((res, res), _Image.BICUBIC)
        rgb = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)
        # HWC[0,1] -> [1,3,H,W] in [-1,1] on the depth model's device (the
        # extract_depth_raw / depth_extractors input convention).
        x = (rgb.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(self._depth_model.device)
        depth_raw = extract_depth_raw(self._depth_model, x)  # [1, H, W] metric units
        # Downsample the RAW metric depth to the cache's store_res BEFORE the
        # colormap, mirroring training (precompute_depth.py downsample_depth ->
        # DepthColormap at store_res). The colormap (turbo LUT) is nonlinear and
        # robust_normalize's clip percentiles are resolution-sensitive, so
        # colormapping at compute_res and resizing the RGB afterward would NOT
        # match downsample-then-colormap. downsample_depth uses bilinear; depth is
        # an intensive scalar, so values are resampled, not rescaled.
        s = self.depth_store_res
        if depth_raw.shape[-2:] != (s, s):
            depth_raw = F.interpolate(
                depth_raw.unsqueeze(1), size=(s, s), mode="bilinear", align_corners=False
            ).squeeze(1)  # [1, s, s]
        # Representation maps raw depth -> [3, s, s] in [-1,1]; predict_from_flow
        # resizes to the IDM image_size (a no-op when store_res == image_size), so
        # the [-1,1] tensor is fed directly.
        depth_rgb = self._depth_rep({"depth": depth_raw[0].cpu()})  # [3, s, s]
        return depth_rgb.unsqueeze(0)  # [1, 3, s, s] in [-1,1]

    # ---- inference ---------------------------------------------------------
    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()  # idempotent; run_server pre-loads at startup
        policy = self._policy
        sid = ctx.session_id

        img = self._extract_primary_image(obs)  # uint8 HWC
        state = self._extract_state(obs)  # raw 8-D
        instruction = obs.get("task_description", obs.get("language", ""))
        prompt = self._build_prompt(instruction, ctx)

        # src is the clean [-1,1] CHW frame (no jitter at inference): used both as
        # IDM source and as the flow-extractor "current" frame.
        pil, src = policy._preprocess_image(img)

        past_flow_pil = None
        if self.past_flow:
            past_flow_pil = self._build_past_flow_pil(sid, src, ctx.step)

        # Wrist-cam frame at the current timestep. _preprocess_ref yields both a
        # flux_image_size PIL (the Flux2 reference, Image N after source/past-flow)
        # and a [1,3,S,S] [-1,1] IDM tensor; un-jittered, matching training's
        # wrist_image path. Decode once and route each to its consumer: the PIL to
        # Flux2 only when use_wrist_ref, the tensor to the IDM only for a wrist
        # input mode (the two flags are independent).
        wrist_flux_pil = None
        wrist_src = None
        if self._needs_wrist:
            wrist_flux_pil, wrist_src = policy._preprocess_ref(self._extract_wrist_image(obs))
        flux_wrist_pil = wrist_flux_pil if self.use_wrist_ref else None
        idm_wrist = wrist_src if self.idm_uses_wrist else None

        # Depth stream for the IDM (depth_flow / source_depth_flow): monocular
        # depth on the agentview -> colormapped depth image. Depth never
        # conditions Flux2, so it is computed independently of the references.
        depth_src = None
        if self.idm_uses_depth:
            depth_src = self._build_depth_image(img)

        flow = policy._generate_flow(
            [pil],
            [prompt],
            policy.seed,
            past_flow_pil=past_flow_pil,
            wrist_pil=flux_wrist_pil,
        )  # [1,3,S,S] in [-1,1]

        actions = policy.predict_from_flow(src, flow, state, wrist=idm_wrist, depth=depth_src)[0]  # (num_actions, 7)

        # Render the generated flow to a PIL once; reused for both the 'pred'
        # past-flow cache and the emitted visualization (avoids a double render).
        flow_pil = None
        if self.past_flow_source == "pred" or self.emit_flow:
            flow_pil = policy._flow_to_pil(flow[0].float().cpu())

        # Cache (keyed by absolute step) for a future replan's past-flow reference,
        # read back num_actions steps later by _past_ref. maxlen retains at least
        # the full num_actions-step window: one entry per replan, plus slack for
        # execute_size=1 (a replan every step).
        entry: dict[str, Any] = {}
        if self.past_flow_source == "gt":
            entry["src"] = src.detach()
        else:
            entry["flow_pil"] = flow_pil
        hist = self._past.get(sid)
        if hist is None:
            stride = self.execute_size or self.num_actions
            hist = deque(maxlen=self.num_actions // max(1, stride) + 2)
            self._past[sid] = hist
        hist.append((ctx.step, entry))

        result: Action = {"actions": np.asarray(actions, dtype=np.float32)}
        extra = self._build_extra_images(
            pil=pil,
            past_flow_pil=past_flow_pil,
            wrist_pil=wrist_flux_pil,
            depth_chw=None if depth_src is None else depth_src[0],
            flow_chw=flow[0],
            flow_pil=flow_pil,
            prompt=prompt,
        )
        if extra:
            result["extra_images"] = extra
        return result

    def _build_extra_images(
        self,
        *,
        pil: Any,
        past_flow_pil: Any,
        wrist_pil: Any,
        depth_chw: Any,
        flow_chw: Any,
        flow_pil: Any,
        prompt: str,
    ) -> dict[str, np.ndarray]:
        """Assemble the extra video streams (uint8 HWC RGB) for one replan step.

        The harness chunk buffer re-emits these for every sub-step of the chunk,
        keeping them aligned with the rollout frames. Streams:
          - predicted flow (``flow_image_key``): the generated flow, with optional
            arrows + prompt banner burned on (always on COPIES, so the *cached*
            ``flow_pil`` used for past-flow conditioning stays pixel-clean). This
            is the model *output* and always stays its own stream. With
            ``flow_overlay`` a second column is appended: the flow
            alpha-composited onto the agentview observation.
          - model input (``input_image_key``): the model *inputs* for this step.
            With ``concat_inputs`` (default) every input modality the model
            consumes — preprocessed observation (only when the IDM uses source;
            the 'no source' modes drop it), wrist, past-flow, depth — is stitched
            left-to-right into one strip so a single video shows the full input.
            With it off each modality is recorded as its own stream instead:
              - past flow (``past_flow_key``): the conditioning reference
                (whiteout on the first replan), recorded raw.
              - wrist (``wrist_input_key``): the wrist-cam reference, recorded raw
                (only when the wrist camera is requested).
              - depth (``depth_input_key``): the colormapped depth image fed to
                the IDM, recorded raw (only for a depth input mode).
        """
        extra: dict[str, np.ndarray] = {}
        if self.emit_flow and flow_pil is not None:
            from PIL import Image

            # Decode the flow field once; both the arrow and overlay columns reuse it.
            uv = self._decode_flow_uv(flow_chw) if (self.overlay_arrows or self.flow_overlay) else None
            left = flow_pil
            if self.overlay_arrows:
                left = self._overlay_flow_arrows(left, uv)
            cols = [np.asarray(left.convert("RGB"), dtype=np.uint8)]
            if self.flow_overlay:
                # Right column: predicted flow alpha-composited onto the agentview.
                cols.append(self._overlay_flow_on_obs(pil, flow_pil, uv))
            vis_arr = hconcat_uint8(cols)
            if self.overlay_prompt:
                vis_arr = np.asarray(
                    self._overlay_prompt_text(Image.fromarray(vis_arr, "RGB"), prompt), dtype=np.uint8
                )
            extra[self.flow_image_key] = vis_arr
        if self.emit_inputs:
            # Ordered model-input modalities, left-to-right in the strip.
            # np.asarray copies, so the cached PILs are never mutated.
            streams: list[tuple[str, np.ndarray]] = []
            # source (agentview): only when the IDM consumes it. The 'no source'
            # modes (wrist_flow / depth_flow) drop the agentview from the IDM
            # input, so it must not appear in the model-input strip even though
            # it still conditions Flux2 (visible via the predicted-flow overlay).
            if self.idm_uses_source:
                streams.append((self.input_image_key, np.asarray(pil.convert("RGB"), dtype=np.uint8)))
            if wrist_pil is not None:
                streams.append((self.wrist_input_key, np.asarray(wrist_pil.convert("RGB"), dtype=np.uint8)))
            if past_flow_pil is not None:
                streams.append((self.past_flow_key, np.asarray(past_flow_pil.convert("RGB"), dtype=np.uint8)))
            if depth_chw is not None:
                # [3,H,W] in [-1,1] -> uint8 HWC RGB.
                d = depth_chw.detach().float().cpu().numpy()
                d = ((d + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                streams.append((self.depth_input_key, np.transpose(d, (1, 2, 0))))

            # streams is non-empty for every valid mode (a no-source mode always
            # has wrist or depth), but guard anyway so an input-less config can't
            # crash hconcat.
            if streams:
                if self.concat_inputs:
                    extra[self.input_image_key] = hconcat_uint8([frame for _, frame in streams])
                else:
                    extra.update(dict(streams))
        return extra

    # ---- visualization -----------------------------------------------------
    @staticmethod
    def _wrap_text(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
        """Greedy word-wrap *text* to lines no wider than *max_width* pixels."""
        lines: list[str] = []
        current = ""
        for word in text.split():
            trial = word if not current else f"{current} {word}"
            if current and draw.textlength(trial, font=font) > max_width:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
        return lines

    def _overlay_prompt_text(self, image: Any, text: str) -> Any:
        """Return a copy of *image* with *text* rendered in a caption band appended
        *below* it, so the prompt never occludes the flow content.

        The band height is a fixed fraction of the image height
        (``caption_band_frac``), so every frame keeps identical dimensions even as
        the wrapped prompt shifts with the progress percentage — the video encoder
        requires uniform frame size. The largest font (down to a floor) whose
        wrapped text fits the band is used. Never mutates *image* (the caller may
        have cached it for conditioning); returns an unmodified copy when *text*
        is empty.
        """
        from PIL import Image, ImageDraw, ImageFont

        img = image.convert("RGB")
        if not text:
            return img.copy()
        w, h = img.size
        margin = max(4, w // 64)
        band_h = max(2 * margin + 10, round(h * self.caption_band_frac))
        inner_w = max(1, w - 2 * margin)
        inner_h = band_h - 2 * margin

        scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))

        # Largest font (down to a floor) whose wrapped text fits the fixed band.
        font: Any = ImageFont.load_default()
        lines: list[str] = [text]
        line_h = inner_h
        for font_size in range(max(11, w // 36), 7, -1):
            try:
                font = ImageFont.load_default(size=font_size)  # scalable default (Pillow >= 10.1)
            except TypeError:  # pragma: no cover - very old Pillow
                font = ImageFont.load_default()
            lines = self._wrap_text(scratch, text, font, inner_w)
            bbox = scratch.textbbox((0, 0), "Ayg", font=font)
            line_h = (bbox[3] - bbox[1]) + max(2, font_size // 5)
            if line_h * len(lines) <= inner_h:
                break

        # Image on top, opaque dark caption band beneath it.
        out = Image.new("RGB", (w, h + band_h), (0, 0, 0))
        out.paste(img, (0, 0))

        draw = ImageDraw.Draw(out)
        y = h + margin
        for line in lines:
            if y + line_h > h + band_h:
                break  # safety: never spill past the band
            draw.text((margin, y), line, fill=(255, 255, 255), font=font)
            y += line_h
        return out

    # ---- arrow (quiver) overlay -------------------------------------------
    def _decode_flow_uv(self, flow_rgb: Any) -> tuple[np.ndarray, np.ndarray]:
        """Invert the RGB flow encoding back to ``(u, v)`` pixel displacements.

        *flow_rgb* is the ``[3, S, S]`` flow image in ``[-1, 1]`` (exactly what
        ``policy._generate_flow`` returns). Reuses fluxflow's own inverse so the
        recovered field matches what the IDM consumes: ``uv`` decodes the raw
        ``[u, v]`` channels, ``videojam`` decodes the HSV colormap. Returns two
        ``[S, S]`` float arrays in pixel units of the flow image.
        """
        with _unshadow_fluxflow():
            from fluxflow.flow_codec import uv_rgb_to_flow, videojam_rgb_to_flow

        rgb01 = ((flow_rgb.detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(0)  # [1,3,S,S]
        decode = uv_rgb_to_flow if self.flow_repr == "uv" else videojam_rgb_to_flow
        flow = decode(rgb01, sigma=self.flow_sigma)[0]  # [2, S, S], pixel units
        return flow[0].numpy(), flow[1].numpy()

    @staticmethod
    def _draw_arrow(draw: Any, x0: float, y0: float, x1: float, y1: float, head: float) -> None:
        """Draw a single arrow (shaft + two barbs) with a dark underlay for contrast."""
        white, black = (255, 255, 255), (0, 0, 0)
        shaft = [(x0, y0), (x1, y1)]
        draw.line(shaft, fill=black, width=3)
        draw.line(shaft, fill=white, width=1)
        ang = math.atan2(y1 - y0, x1 - x0)
        for da in (math.radians(150), -math.radians(150)):
            hx = x1 + head * math.cos(ang + da)
            hy = y1 + head * math.sin(ang + da)
            barb = [(x1, y1), (hx, hy)]
            draw.line(barb, fill=black, width=3)
            draw.line(barb, fill=white, width=1)

    def _overlay_flow_arrows(self, image: Any, uv: tuple[np.ndarray, np.ndarray]) -> Any:
        """Return a COPY of *image* with a grid of flow-direction arrows drawn on.

        *uv* is the decoded ``(u, v)`` displacement field; this averages it over
        each grid cell and draws one arrow per cell whose mean displacement
        exceeds ``arrow_min_frac`` of the image diagonal. Never mutates *image*
        (the caller may have cached it for past-flow conditioning).
        """
        from PIL import ImageDraw

        img = image.convert("RGB").copy()
        u, v = uv
        h, w = u.shape
        n = max(1, int(self.arrow_grid))
        step = max(1, min(h, w) // n)
        half = step // 2
        min_mag = max(0.0, self.arrow_min_frac) * math.hypot(h, w)
        head = max(2.0, step * 0.35)

        draw = ImageDraw.Draw(img)
        for cy in range(half, h, step):
            for cx in range(half, w, step):
                du = float(u[max(0, cy - half) : cy + half + 1, max(0, cx - half) : cx + half + 1].mean())
                dv = float(v[max(0, cy - half) : cy + half + 1, max(0, cx - half) : cx + half + 1].mean())
                if math.hypot(du, dv) < min_mag:
                    continue
                self._draw_arrow(draw, cx, cy, cx + du * self.arrow_scale, cy + dv * self.arrow_scale, head)
        return img

    def _overlay_flow_on_obs(self, obs_pil: Any, flow_pil: Any, uv: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Alpha-composite the predicted flow onto the agentview observation.

        *uv* is the decoded ``(u, v)`` displacement field. Stationary (gray) flow
        regions — ``|displacement|`` below ``flow_overlay_min_frac`` of the image
        diagonal — stay fully transparent so the observation shows through;
        regions with valid motion are blended at ``flow_overlay_alpha``. Returns a
        uint8 HWC RGB array sized to the flow field. Never mutates the cached
        *flow_pil*/*obs_pil*.
        """
        from PIL import Image

        u, v = uv
        h, w = u.shape
        obs = np.asarray(obs_pil.convert("RGB").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
        flow = np.asarray(flow_pil.convert("RGB").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)

        mag = np.hypot(u, v)
        min_mag = max(0.0, self.flow_overlay_min_frac) * math.hypot(h, w)
        alpha = np.where(mag >= min_mag, float(self.flow_overlay_alpha), 0.0).astype(np.float32)[..., None]

        out = obs * (1.0 - alpha) + flow * alpha
        return out.clip(0, 255).astype(np.uint8)

    # ---- episode lifecycle -------------------------------------------------
    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        self._past.pop(ctx.session_id, None)
        self._session_max_steps.pop(ctx.session_id, None)
        # Resolve the suite's rollout horizon for append_progress (override-able
        # by rollout_max_steps). The benchmark sends "suite" in the task payload.
        if self.append_progress and self.rollout_max_steps is None:
            suite = (config.get("task") or {}).get("suite") if isinstance(config, dict) else None
            if suite:
                max_steps = _libero_suite_max_steps(suite)
                if max_steps:
                    self._session_max_steps[ctx.session_id] = max_steps
                else:
                    logger.warning(
                        "append_progress: no MAX_STEP_MAPPING entry for suite %r; "
                        "progress will report 0%% until rollout_max_steps is set",
                        suite,
                    )
            else:
                logger.warning("append_progress: episode start carried no 'suite'; cannot resolve max steps")
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self._past.pop(ctx.session_id, None)
        self._session_max_steps.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(FluxFlowModelServer)
