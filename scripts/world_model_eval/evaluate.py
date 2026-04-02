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
#     "matplotlib",
#     "scikit-learn",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../..", editable = true }
# starvla = { git = "https://github.com/ginwind/VLA-JEPA.git", rev = "2a16b6c4ed1f81a854d776f2760c48017e3bc49a" }
#
# [tool.uv]
# exclude-newer = "2026-03-30T00:00:00Z"
# ///
"""Offline evaluation of VLA-JEPA world model future predictions.

Phase 2 of RFC-0008: loads observations and action_tokens recorded during
rollout (Phase 1), runs the VJEPA2 encoder to get ground-truth features,
runs the world model predictor to get predicted features, and computes
feature-space metrics.

Usage::

    python scripts/world_model_eval/evaluate.py \
        --record_dir /path/to/recorded_data \
        --checkpoint ginwind/VLA-JEPA \
        --checkpoint_variant LIBERO \
        --output_dir /path/to/results

The script produces:
  - ``metrics.json``: per-episode and aggregate metrics
  - ``temporal_curve.png``: prediction quality vs. horizon
  - ``tsne.png``: t-SNE of predicted vs ground-truth features (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_episode(episode_dir: Path) -> dict[str, Any]:
    """Load observations and inference data from a recorded episode.

    The recording format uses two aggregated files per episode:
    - ``obs.npz``: images at every step, keyed as ``{view}_{step:04d}``,
      plus ``num_steps`` and ``view_names`` metadata arrays.
    - ``inference.npz``: action_tokens and normalized_actions at inference
      steps, keyed as ``action_tokens_{step:04d}`` etc., plus
      ``inference_steps`` array.

    Returns a dict with ``metadata``, ``result``, ``observations`` (list of
    per-step image dicts), and ``inferences`` (dict mapping step index to
    action_tokens/normalized_actions).
    """
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    result_path = episode_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}

    # Load observations
    obs_path = episode_dir / "obs.npz"
    observations: list[dict[str, np.ndarray]] = []
    if obs_path.exists():
        obs_archive = np.load(obs_path, allow_pickle=True)
        num_steps = int(obs_archive["num_steps"])
        view_names = list(obs_archive["view_names"])
        for step in range(num_steps):
            frame: dict[str, np.ndarray] = {}
            for view in view_names:
                key = f"{view}_{step:04d}"
                if key in obs_archive:
                    frame[view] = obs_archive[key]
            observations.append(frame)

    # Load inference data
    inferences: dict[int, dict[str, np.ndarray]] = {}
    inf_path = episode_dir / "inference.npz"
    if inf_path.exists():
        inf_archive = np.load(inf_path, allow_pickle=True)
        for step_idx in inf_archive["inference_steps"]:
            step_idx = int(step_idx)
            inferences[step_idx] = {
                "action_tokens": inf_archive[f"action_tokens_{step_idx:04d}"],
                "normalized_actions": inf_archive[f"normalized_actions_{step_idx:04d}"],
            }

    return {"metadata": metadata, "result": result, "observations": observations, "inferences": inferences}


# ---------------------------------------------------------------------------
# Model loading (reuses VLA-JEPA harness patching logic)
# ---------------------------------------------------------------------------


def load_vla_jepa_model(
    checkpoint: str,
    *,
    checkpoint_variant: str | None = None,
    use_bf16: bool = True,
    device: str | None = None,
) -> Any:
    """Load VLA-JEPA model with all submodules (vj_encoder, vj_predictor, etc).

    Applies the same upstream-path patching as ``VLAJEPAModelServer``.
    """
    import importlib

    import torch
    import transformers as tfm

    # Re-use the harness helpers for checkpoint resolution and path patching.
    from vla_eval.model_servers.vla_jepa import (
        _BASE_ENCODER_BY_BASENAME,
        _BASE_VLM_BY_BASENAME,
        _FRAMEWORK_ALIASES,
        _looks_like_local_reference,
        _resolve_checkpoint_path,
        _resolve_upstream_model_path,
    )

    from starVLA.model.framework.base_framework import baseframework

    checkpoint_path = _resolve_checkpoint_path(checkpoint, checkpoint_variant=checkpoint_variant)
    logger.info("Loading VLA-JEPA from %s", checkpoint_path)

    patches: list[tuple[Any, str, Any]] = []

    def _patch_from_pretrained(cls_to_patch: Any) -> None:
        if cls_to_patch is None:
            return
        original = cls_to_patch.from_pretrained.__func__

        @classmethod
        def _patched(cls: Any, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("attn_implementation") == "flash_attention_2":
                kwargs["attn_implementation"] = "kernels-community/flash-attn2"
            return original(cls, *args, **kwargs)

        patches.append((cls_to_patch, "from_pretrained", classmethod(original)))
        cls_to_patch.from_pretrained = _patched

    _patch_from_pretrained(getattr(tfm, "Qwen2_5_VLForConditionalGeneration", None))
    _patch_from_pretrained(getattr(tfm, "Qwen3VLForConditionalGeneration", None))

    import starVLA.model.framework as fw_mod
    import starVLA.model.framework.base_framework as bf_mod

    importlib.import_module("starVLA.model.framework.VLA_JEPA")
    original_build = fw_mod.build_framework

    def _patched_build(cfg: Any) -> Any:
        fw_name = getattr(cfg.framework, "name", None) or getattr(cfg.framework, "framework_py", None)
        if fw_name in _FRAMEWORK_ALIASES:
            cfg.framework.name = _FRAMEWORK_ALIASES[fw_name]
            if hasattr(cfg.framework, "framework_py"):
                cfg.framework.framework_py = _FRAMEWORK_ALIASES[fw_name]

        base_vlm = str(getattr(cfg.framework.qwenvl, "base_vlm", ""))
        if base_vlm and not Path(base_vlm).expanduser().exists():
            if _looks_like_local_reference(base_vlm) or Path(base_vlm).name in _BASE_VLM_BY_BASENAME:
                cfg.framework.qwenvl.base_vlm = _resolve_upstream_model_path(
                    base_vlm, mapping=_BASE_VLM_BY_BASENAME, fallback="Qwen/Qwen3-VL-2B-Instruct", label="base_vlm"
                )

        base_enc = str(getattr(cfg.framework.vj2_model, "base_encoder", ""))
        if base_enc and not Path(base_enc).expanduser().exists():
            if _looks_like_local_reference(base_enc) or Path(base_enc).name in _BASE_ENCODER_BY_BASENAME:
                cfg.framework.vj2_model.base_encoder = _resolve_upstream_model_path(
                    base_enc,
                    mapping=_BASE_ENCODER_BY_BASENAME,
                    fallback="facebook/vjepa2-vitl-fpc64-256",
                    label="base_encoder",
                )

        return original_build(cfg)

    patches.append((fw_mod, "build_framework", original_build))
    fw_mod.build_framework = _patched_build
    patches.append((bf_mod, "build_framework", original_build))
    bf_mod.build_framework = _patched_build

    try:
        model = baseframework.from_pretrained(checkpoint_path)
    finally:
        for obj, attr, orig in reversed(patches):
            setattr(obj, attr, orig)

    if use_bf16:
        model = model.to(torch.bfloat16)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    logger.info("Model loaded on %s", device)
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def encode_video_features(
    model: Any,
    frames: list[dict[str, np.ndarray]],
    *,
    views: list[str] | None = None,
) -> Any:
    """Encode a sequence of observation frames with VJEPA2.

    Args:
        model: VLA_JEPA model instance.
        frames: list of T observation dicts, each containing
            image arrays keyed by view name (e.g. ``agentview``, ``wrist``).
            Each array is ``[H, W, 3]`` uint8.
        views: ordered list of view names. If ``None``, auto-detected from
            the first frame's keys.

    Returns:
        video_embeddings tensor of shape
        ``[1, T_reduced * dim_per_frame, V * embed_dim]``.
    """
    import torch

    if views is None:
        views = sorted(frames[0].keys())

    V = len(views)
    T = len(frames)

    # Build [V, T, C, H, W] video tensor
    video_np = np.stack(
        [[frames[t][v] for t in range(T)] for v in views],
        axis=0,
    )  # [V, T, H, W, 3]
    video_np = video_np.transpose(0, 1, 4, 2, 3)  # [V, T, C, H, W]

    # Process each view through vj_processor
    device = next(model.vj_encoder.parameters()).device
    input_videos = []
    for v_idx in range(V):
        processed = model.vj_processor(videos=video_np[v_idx], return_tensors="pt")
        input_videos.append(processed["pixel_values_videos"].to(device))
    input_videos_t = torch.cat(input_videos, dim=0)  # [V, T, C, H, W]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        video_embeddings = model.vj_encoder.get_vision_features(pixel_values_videos=input_videos_t)
        # Concat multi-view features along embedding dimension
        video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)

    return video_embeddings  # [1, T_reduced * dim_per_frame, V * embed_dim]


def predict_and_compare(
    model: Any,
    video_embeddings: Any,
    action_tokens: Any,
    num_frames: int,
) -> dict[str, np.ndarray]:
    """Run the world model predictor and compare with ground-truth features.

    Args:
        model: VLA_JEPA model instance.
        video_embeddings: output of ``encode_video_features``,
            shape ``[1, T_reduced * dim_per_frame, V * embed_dim]``.
        action_tokens: numpy array of shape ``[num_action_tokens, hidden_dim]``
            from recording (float16). These are the QwenVL hidden states at
            ``<|action_{}|>`` positions.
        num_frames: number of raw frames in the window (before tubelet compression).

    Returns:
        dict with keys ``predicted``, ``ground_truth`` (numpy arrays),
        and per-timestep ``cosine_sim``, ``l2_dist``.
    """
    import torch

    device = next(model.vj_predictor.parameters()).device
    tubelet_size = model.vj_encoder.config.tubelet_size
    T_reduced = num_frames // tubelet_size

    # Split into input_states (first T_reduced-1 steps) and gt_states (last T_reduced-1 steps)
    dim_per_step = video_embeddings.shape[1] // T_reduced
    input_states = video_embeddings[:, : dim_per_step * (T_reduced - 1), :]
    gt_states = video_embeddings[:, dim_per_step:, :]

    # Prepare action_tokens
    action_tokens_t = torch.from_numpy(action_tokens.astype(np.float32)).unsqueeze(0).to(device)
    if model.use_bf16 if hasattr(model, "use_bf16") else next(model.vj_predictor.parameters()).dtype == torch.bfloat16:
        action_tokens_t = action_tokens_t.to(torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        predicted_states = model.vj_predictor(input_states, action_tokens_t)

    # Convert to numpy for metrics
    pred_np = predicted_states.float().cpu().numpy()[0]  # [T_reduced-1 * dim_per_step, embed_dim]
    gt_np = gt_states.float().cpu().numpy()[0]

    # Per-timestep metrics
    n_steps = T_reduced - 1
    cosine_sims = []
    l2_dists = []
    rand_cosine_sims = []
    rand_l2_dists = []

    # Build shuffled GT for random baseline: permute step order
    rng = np.random.default_rng()
    shuffled_indices = rng.permutation(n_steps)

    for step_idx in range(n_steps):
        start = step_idx * dim_per_step
        end = (step_idx + 1) * dim_per_step
        pred_step = pred_np[start:end].flatten()
        gt_step = gt_np[start:end].flatten()

        # Cosine similarity
        dot = np.dot(pred_step, gt_step)
        norm_pred = np.linalg.norm(pred_step)
        norm_gt = np.linalg.norm(gt_step)
        cos_sim = dot / (norm_pred * norm_gt + 1e-8)
        cosine_sims.append(float(cos_sim))

        # L2 distance
        l2 = float(np.linalg.norm(pred_step - gt_step))
        l2_dists.append(l2)

        # Random baseline: compare prediction against a different step's GT
        rand_idx = int(shuffled_indices[step_idx])
        rand_start = rand_idx * dim_per_step
        rand_end = (rand_idx + 1) * dim_per_step
        rand_gt_step = gt_np[rand_start:rand_end].flatten()

        rand_dot = np.dot(pred_step, rand_gt_step)
        rand_norm_gt = np.linalg.norm(rand_gt_step)
        rand_cosine_sims.append(float(rand_dot / (norm_pred * rand_norm_gt + 1e-8)))
        rand_l2_dists.append(float(np.linalg.norm(pred_step - rand_gt_step)))

    return {
        "predicted": pred_np,
        "ground_truth": gt_np,
        "cosine_sim": np.array(cosine_sims),
        "l2_dist": np.array(l2_dists),
        "rand_cosine_sim": np.array(rand_cosine_sims),
        "rand_l2_dist": np.array(rand_l2_dists),
        "dim_per_step": dim_per_step,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def compute_r_squared(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute R-squared between predicted and ground-truth feature vectors."""
    ss_res = np.sum((ground_truth - predicted) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def build_gt_feature_bank(
    model: Any,
    observations: list[dict[str, np.ndarray]],
    view_names: list[str],
    num_frames: int,
) -> np.ndarray:
    """Encode all episode frames with VJEPA2 and return per-temporal-patch features.

    Slides a window of ``num_frames`` across the episode with stride=1,
    extracting the **middle** temporal patch's feature from each window to
    get a consistent representation for every frame position.

    For frames near the start/end that can't be the middle of a full window,
    we use the earliest/latest available window.

    Returns an array of shape ``[num_frames_in_episode, dim_per_patch * embed_dim]``.
    """

    tubelet_size = model.vj_encoder.config.tubelet_size
    T_reduced = num_frames // tubelet_size
    n_obs = len(observations)

    # Encode all possible windows with stride = num_frames (non-overlapping for speed)
    # and collect per-temporal-patch features
    all_patch_features: dict[int, np.ndarray] = {}  # frame_idx -> feature

    for win_start in range(0, n_obs - num_frames + 1, tubelet_size):
        window_obs = observations[win_start : win_start + num_frames]
        if any(len(f) != len(view_names) for f in window_obs):
            continue

        video_embeddings = encode_video_features(model, window_obs, views=view_names)
        emb_np = video_embeddings.float().cpu().numpy()[0]  # [T_reduced * dim_per_patch, embed_dim]
        dim_per_patch = emb_np.shape[0] // T_reduced

        # Extract feature for each temporal patch in this window
        for t_idx in range(T_reduced):
            # Each temporal patch covers `tubelet_size` frames
            frame_idx = win_start + t_idx * tubelet_size
            if frame_idx not in all_patch_features:
                feat = emb_np[t_idx * dim_per_patch : (t_idx + 1) * dim_per_patch].flatten()
                all_patch_features[frame_idx] = feat

    if not all_patch_features:
        return np.array([])

    # Build ordered bank
    sorted_indices = sorted(all_patch_features.keys())
    bank = np.stack([all_patch_features[i] for i in sorted_indices])
    logger.info("  GT feature bank: %d entries, dim=%d", len(bank), bank.shape[1])
    return bank


def compute_retrieval_metrics(
    predicted_features: np.ndarray,
    gt_bank: np.ndarray,
    target_indices: list[int],
) -> dict[str, float]:
    """Compute retrieval metrics for predicted features against a GT bank.

    Args:
        predicted_features: ``[N, D]`` array of predicted next-state features.
        gt_bank: ``[M, D]`` array of GT features (the retrieval database).
        target_indices: length-N list, where ``target_indices[i]`` is the
            index into ``gt_bank`` of the correct next frame for prediction i.

    Returns:
        dict with ``mean_rank``, ``median_rank``, ``recall_at_1``,
        ``recall_at_5``, ``mrr`` (mean reciprocal rank).
    """
    # Normalize for cosine similarity
    pred_norm = predicted_features / (np.linalg.norm(predicted_features, axis=1, keepdims=True) + 1e-8)
    bank_norm = gt_bank / (np.linalg.norm(gt_bank, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity: [N, M]
    similarities = pred_norm @ bank_norm.T

    ranks = []
    for i, target_idx in enumerate(target_indices):
        if target_idx < 0 or target_idx >= len(gt_bank):
            continue
        # Rank = number of entries with higher similarity than the target
        target_sim = similarities[i, target_idx]
        rank = int(np.sum(similarities[i] > target_sim))  # 0-indexed, 0 = best
        ranks.append(rank)

    if not ranks:
        return {
            "mean_rank": float("nan"),
            "median_rank": float("nan"),
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
        }

    ranks_arr = np.array(ranks)
    return {
        "mean_rank": float(np.mean(ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
        "recall_at_1": float(np.mean(ranks_arr < 1)),
        "recall_at_5": float(np.mean(ranks_arr < 5)),
        "mrr": float(np.mean(1.0 / (ranks_arr + 1))),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_temporal_curve(
    per_step_cosine: list[list[float]],
    per_step_l2: list[list[float]],
    output_path: Path,
    *,
    rand_cosine: list[list[float]] | None = None,
    rand_l2: list[list[float]] | None = None,
) -> None:
    """Plot prediction quality vs. prediction horizon with optional random baseline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _to_matrix(data: list[list[float]]) -> np.ndarray:
        max_steps = max(len(s) for s in data)
        matrix = np.full((len(data), max_steps), np.nan)
        for i, vals in enumerate(data):
            matrix[i, : len(vals)] = vals
        return matrix

    cosine_matrix = _to_matrix(per_step_cosine)
    l2_matrix = _to_matrix(per_step_l2)
    max_steps = cosine_matrix.shape[1]
    steps = np.arange(1, max_steps + 1)

    cos_mean = np.nanmean(cosine_matrix, axis=0)
    cos_std = np.nanstd(cosine_matrix, axis=0)
    l2_mean = np.nanmean(l2_matrix, axis=0)
    l2_std = np.nanstd(l2_matrix, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Cosine similarity
    ax1.plot(steps, cos_mean, "b-o", markersize=4, label="Predicted vs. GT")
    ax1.fill_between(steps, cos_mean - cos_std, cos_mean + cos_std, color="blue", alpha=0.15)
    if rand_cosine:
        rand_cos_matrix = _to_matrix(rand_cosine)
        rand_cos_mean = np.nanmean(rand_cos_matrix, axis=0)
        rand_cos_std = np.nanstd(rand_cos_matrix, axis=0)
        ax1.plot(steps, rand_cos_mean, "gray", linestyle="--", marker="x", markersize=4, label="Random baseline")
        ax1.fill_between(steps, rand_cos_mean - rand_cos_std, rand_cos_mean + rand_cos_std, color="gray", alpha=0.1)
    ax1.set_xlabel("Prediction Horizon (steps)")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Feature Prediction Quality vs. Horizon")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # L2 distance
    ax2.plot(steps, l2_mean, "r-o", markersize=4, label="Predicted vs. GT")
    ax2.fill_between(steps, l2_mean - l2_std, l2_mean + l2_std, color="red", alpha=0.15)
    if rand_l2:
        rand_l2_matrix = _to_matrix(rand_l2)
        rand_l2_mean = np.nanmean(rand_l2_matrix, axis=0)
        rand_l2_std = np.nanstd(rand_l2_matrix, axis=0)
        ax2.plot(steps, rand_l2_mean, "gray", linestyle="--", marker="x", markersize=4, label="Random baseline")
        ax2.fill_between(steps, rand_l2_mean - rand_l2_std, rand_l2_mean + rand_l2_std, color="gray", alpha=0.1)
    ax2.set_xlabel("Prediction Horizon (steps)")
    ax2.set_ylabel("L2 Distance")
    ax2.set_title("Feature Prediction Error vs. Horizon")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved temporal curve to %s", output_path)


def plot_tsne(
    all_predicted: list[np.ndarray],
    all_gt: list[np.ndarray],
    output_path: Path,
    *,
    max_samples: int = 2000,
) -> None:
    """Plot t-SNE of predicted vs. ground-truth features."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    pred_flat = np.concatenate(all_predicted, axis=0)
    gt_flat = np.concatenate(all_gt, axis=0)

    # Subsample paired points together to preserve correspondence
    n = pred_flat.shape[0]
    if n > max_samples:
        idx = np.random.default_rng(42).choice(n, max_samples, replace=False)
        pred_flat = pred_flat[idx]
        gt_flat = gt_flat[idx]

    n_pairs = len(pred_flat)
    # Stack predicted then GT so pairs are at indices [i] and [n_pairs + i]
    combined = np.concatenate([pred_flat, gt_flat], axis=0)

    logger.info("Running t-SNE on %d samples (%d pairs)...", len(combined), n_pairs)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined) // 4))
    embedded = tsne.fit_transform(combined)

    pred_embed = embedded[:n_pairs]
    gt_embed = embedded[n_pairs:]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw lines connecting each predicted point to its GT counterpart
    for i in range(n_pairs):
        ax.plot(
            [gt_embed[i, 0], pred_embed[i, 0]],
            [gt_embed[i, 1], pred_embed[i, 1]],
            c="gray",
            alpha=0.15,
            linewidth=0.5,
        )

    ax.scatter(gt_embed[:, 0], gt_embed[:, 1], c="blue", alpha=0.4, s=10, label="Ground Truth")
    ax.scatter(pred_embed[:, 0], pred_embed[:, 1], c="red", alpha=0.4, s=10, label="Predicted")
    ax.legend()
    ax.set_title("t-SNE: Predicted vs. Ground-Truth VJEPA2 Features")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved t-SNE to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLA-JEPA world model predictions (RFC-0008 Phase 2)")
    parser.add_argument("--record_dir", type=str, required=True, help="Directory with recorded rollout data")
    parser.add_argument("--checkpoint", type=str, required=True, help="VLA-JEPA checkpoint (HF repo or local path)")
    parser.add_argument("--checkpoint_variant", type=str, default=None, help="Checkpoint variant (e.g. LIBERO)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for results")
    parser.add_argument("--num_frames", type=int, default=8, help="Frames per sliding window (must match training)")
    parser.add_argument("--window_stride", type=int, default=4, help="Stride between sliding windows")
    parser.add_argument("--use_bf16", action="store_true", default=True, help="Use bfloat16 (default: True)")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bfloat16")
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE visualization")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit number of episodes to process")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    args = parser.parse_args()

    if args.no_bf16:
        args.use_bf16 = False

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    record_dir = Path(args.record_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover episodes
    episodes = sorted(
        [d for d in record_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()],
        key=lambda p: p.name,
    )
    if not episodes:
        logger.error("No episodes found in %s", record_dir)
        sys.exit(1)

    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    logger.info("Found %d episodes to evaluate", len(episodes))

    # Load model
    model = load_vla_jepa_model(
        args.checkpoint,
        checkpoint_variant=args.checkpoint_variant,
        use_bf16=args.use_bf16,
        device=args.device,
    )

    tubelet_size = model.vj_encoder.config.tubelet_size
    T_reduced = args.num_frames // tubelet_size
    logger.info("num_frames=%d, tubelet_size=%d, T_reduced=%d", args.num_frames, tubelet_size, T_reduced)

    # Evaluate each episode
    all_episode_metrics: list[dict[str, Any]] = []
    all_per_step_cosine: list[list[float]] = []
    all_per_step_l2: list[list[float]] = []
    all_rand_cosine: list[list[float]] = []
    all_rand_l2: list[list[float]] = []
    all_predicted_features: list[np.ndarray] = []
    all_gt_features: list[np.ndarray] = []

    for ep_idx, episode_dir in enumerate(episodes):
        logger.info("[%d/%d] Processing %s", ep_idx + 1, len(episodes), episode_dir.name)
        episode = load_episode(episode_dir)
        observations = episode["observations"]
        inferences = episode["inferences"]

        if len(observations) < args.num_frames:
            logger.warning(
                "Episode %s has only %d observations, need %d — skipping",
                episode_dir.name,
                len(observations),
                args.num_frames,
            )
            continue

        # Determine available views from first observation
        view_names = sorted(observations[0].keys())
        if not view_names:
            logger.warning("Episode %s has no image data — skipping", episode_dir.name)
            continue

        episode_cosine: list[list[float]] = []
        episode_l2: list[list[float]] = []
        episode_rand_cosine: list[list[float]] = []
        episode_rand_l2: list[list[float]] = []
        episode_r2: list[float] = []

        # Build GT feature bank for retrieval evaluation
        logger.info("  Building GT feature bank (%d observations, %d views)...", len(observations), len(view_names))
        gt_bank = build_gt_feature_bank(model, observations, view_names, args.num_frames)
        logger.info("  GT bank: %d entries", len(gt_bank))
        # Map from frame index to bank index
        tubelet_size = model.vj_encoder.config.tubelet_size
        bank_frame_indices = sorted(
            set(
                win_start + t * tubelet_size
                for win_start in range(0, len(observations) - args.num_frames + 1, tubelet_size)
                for t in range(args.num_frames // tubelet_size)
            )
        )
        frame_to_bank: dict[int, int] = {f: i for i, f in enumerate(bank_frame_indices)}
        episode_predicted_for_retrieval: list[np.ndarray] = []
        episode_retrieval_targets: list[int] = []

        # Slide window across episode observations (consecutive frames)
        n_windows = max(0, (len(observations) - args.num_frames) // args.window_stride + 1)
        logger.info("  Sliding %d windows (stride=%d)...", n_windows, args.window_stride)
        for win_start in range(0, len(observations) - args.num_frames + 1, args.window_stride):
            win_end = win_start + args.num_frames
            window_obs = observations[win_start:win_end]

            # Observations are already keyed by view name
            frames = window_obs

            # Skip if any frame is missing a view
            if any(len(f) != len(view_names) for f in frames):
                continue

            # Find the closest inference step at or before this window start
            # to get the action_tokens that condition the world model
            inference_steps = sorted(inferences.keys())
            closest_inf = None
            for s in inference_steps:
                if s <= win_start:
                    closest_inf = s
            if closest_inf is None:
                # No inference before this window — use the first available
                if inference_steps:
                    closest_inf = inference_steps[0]
                else:
                    continue

            action_tokens = inferences[closest_inf]["action_tokens"]

            # Encode video features
            video_embeddings = encode_video_features(model, frames, views=view_names)

            # Predict and compare
            result = predict_and_compare(model, video_embeddings, action_tokens, args.num_frames)

            episode_cosine.append(result["cosine_sim"].tolist())
            episode_l2.append(result["l2_dist"].tolist())
            episode_rand_cosine.append(result["rand_cosine_sim"].tolist())
            episode_rand_l2.append(result["rand_l2_dist"].tolist())
            episode_r2.append(compute_r_squared(result["predicted"], result["ground_truth"]))

            # Collect predicted features for retrieval
            logger.debug("  Window [%d:%d] cos=%.4f l2=%.4f", win_start, win_end,
                         float(np.mean(result["cosine_sim"])), float(np.mean(result["l2_dist"])))
            if len(gt_bank) > 0:
                dim = result["dim_per_step"]
                n = result["n_steps"]
                for i in range(n):
                    pred_feat = result["predicted"][i * dim : (i + 1) * dim].flatten()
                    episode_predicted_for_retrieval.append(pred_feat)
                    # Target: the GT frame this prediction corresponds to
                    # Prediction i predicts temporal patch (i+1) in the window,
                    # which maps to frame win_start + (i+1) * tubelet_size
                    target_frame = win_start + (i + 1) * tubelet_size
                    episode_retrieval_targets.append(frame_to_bank.get(target_frame, -1))

            if args.tsne:
                dim = result["dim_per_step"]
                n = result["n_steps"]
                for i in range(n):
                    all_predicted_features.append(result["predicted"][i * dim : (i + 1) * dim])
                    all_gt_features.append(result["ground_truth"][i * dim : (i + 1) * dim])

        if not episode_cosine:
            logger.warning("  No valid windows for episode %s — skipping", episode_dir.name)
            continue

        logger.info("  Collected %d windows, %d retrieval predictions", len(episode_cosine), len(episode_predicted_for_retrieval))

        # Compute retrieval metrics for this episode
        retrieval_metrics: dict[str, float] = {}
        if episode_predicted_for_retrieval and len(gt_bank) > 0:
            retrieval_metrics = compute_retrieval_metrics(
                np.stack(episode_predicted_for_retrieval),
                gt_bank,
                episode_retrieval_targets,
            )

        # Aggregate window metrics for this episode
        all_cosine_flat = [c for window in episode_cosine for c in window]
        all_l2_flat = [v for window in episode_l2 for v in window]

        ep_metrics: dict[str, Any] = {
            "episode_id": episode_dir.name,
            "task": episode["metadata"].get("task", {}).get("name", ""),
            "success": episode.get("result", {}).get("result", {}).get("metrics", {}).get("success", None),
            "num_windows": len(episode_cosine),
            "cosine_sim_mean": float(np.mean(all_cosine_flat)),
            "cosine_sim_std": float(np.std(all_cosine_flat)),
            "l2_dist_mean": float(np.mean(all_l2_flat)),
            "l2_dist_std": float(np.std(all_l2_flat)),
            "r2_mean": float(np.mean(episode_r2)),
        }
        if retrieval_metrics:
            ep_metrics["retrieval"] = retrieval_metrics
        all_episode_metrics.append(ep_metrics)
        all_per_step_cosine.extend(episode_cosine)
        all_per_step_l2.extend(episode_l2)
        all_rand_cosine.extend(episode_rand_cosine)
        all_rand_l2.extend(episode_rand_l2)

        retrieval_str = ""
        if retrieval_metrics:
            retrieval_str = f" R@1={retrieval_metrics['recall_at_1']:.3f} R@5={retrieval_metrics['recall_at_5']:.3f} MRR={retrieval_metrics['mrr']:.3f} rank={retrieval_metrics['mean_rank']:.1f}/{len(gt_bank)}"
        logger.info(
            "  %s: cos=%.4f l2=%.4f r2=%.4f (%d windows)%s",
            episode_dir.name[:8],
            ep_metrics["cosine_sim_mean"],
            ep_metrics["l2_dist_mean"],
            ep_metrics["r2_mean"],
            ep_metrics["num_windows"],
            retrieval_str,
        )

    if not all_episode_metrics:
        logger.error("No episodes produced valid metrics")
        sys.exit(1)

    logger.info(
        "Evaluation complete: %d episodes, %d total windows. Aggregating...",
        len(all_episode_metrics),
        sum(e["num_windows"] for e in all_episode_metrics),
    )

    # Aggregate metrics
    aggregate: dict[str, Any] = {
        "num_episodes": len(all_episode_metrics),
        "num_windows": sum(e["num_windows"] for e in all_episode_metrics),
        "cosine_sim_mean": float(np.mean([e["cosine_sim_mean"] for e in all_episode_metrics])),
        "cosine_sim_std": float(np.std([e["cosine_sim_mean"] for e in all_episode_metrics])),
        "l2_dist_mean": float(np.mean([e["l2_dist_mean"] for e in all_episode_metrics])),
        "l2_dist_std": float(np.std([e["l2_dist_mean"] for e in all_episode_metrics])),
        "r2_mean": float(np.mean([e["r2_mean"] for e in all_episode_metrics])),
    }

    # Aggregate retrieval metrics
    eps_with_retrieval = [e for e in all_episode_metrics if "retrieval" in e]
    if eps_with_retrieval:
        aggregate["retrieval"] = {
            key: float(np.mean([e["retrieval"][key] for e in eps_with_retrieval]))
            for key in ["mean_rank", "median_rank", "recall_at_1", "recall_at_5", "mrr"]
        }

    output = {
        "config": {
            "checkpoint": args.checkpoint,
            "checkpoint_variant": args.checkpoint_variant,
            "num_frames": args.num_frames,
            "window_stride": args.window_stride,
            "tubelet_size": tubelet_size,
        },
        "aggregate": aggregate,
        "episodes": all_episode_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2))
    logger.info("Saved metrics to %s", metrics_path)
    retrieval_log = ""
    if "retrieval" in aggregate:
        r = aggregate["retrieval"]
        retrieval_log = (
            f", R@1={r['recall_at_1']:.3f}, R@5={r['recall_at_5']:.3f}, MRR={r['mrr']:.3f}, rank={r['mean_rank']:.1f}"
        )
    logger.info(
        "Aggregate: cos=%.4f (+-%.4f), l2=%.4f (+-%.4f), r2=%.4f%s",
        aggregate["cosine_sim_mean"],
        aggregate["cosine_sim_std"],
        aggregate["l2_dist_mean"],
        aggregate["l2_dist_std"],
        aggregate["r2_mean"],
        retrieval_log,
    )

    # Visualizations
    if all_per_step_cosine:
        plot_temporal_curve(
            all_per_step_cosine,
            all_per_step_l2,
            output_dir / "temporal_curve.png",
            rand_cosine=all_rand_cosine,
            rand_l2=all_rand_l2,
        )

    if args.tsne and all_predicted_features:
        plot_tsne(all_predicted_features, all_gt_features, output_dir / "tsne.png")

    logger.info("Done. Results in %s", output_dir)


if __name__ == "__main__":
    main()
