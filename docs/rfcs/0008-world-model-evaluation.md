# RFC-0008: World Model Future Prediction Evaluation

- **Author:** @junhalee
- **Status:** Proposed
- **Type:** Standards Track
- **Created:** 2026-04-01
- **Requires:** RFC-0003, RFC-0007
- **Superseded-By:** —

## Summary

Add infrastructure for evaluating the **future prediction quality** of VLA models that include a world model component (e.g., VLA-JEPA, CoW-VLA). These models learn to predict future observations in feature space during training, but this capability is not exercised during standard action-only evaluation. This RFC introduces a two-phase pipeline: (1) recording observations and model internals during benchmark rollout, and (2) offline feature extraction and world model evaluation.

## Problem

Current evaluation measures only **task success rate** — whether the robot completes the task. This tells us nothing about the quality of the model's internal world model, which predicts future visual states in a learned feature space.

Understanding world model quality is valuable because:

- It reveals whether the model has learned meaningful dynamics (even when actions fail).
- It enables comparison across models that use different world model architectures.
- It can diagnose failure modes: a model may predict actions correctly but have a poor world model, or vice versa.

The challenge is that these models predict futures in **feature space** (not pixel space), using different teacher encoders:

| Model | Teacher Encoder | World Model Module |
|-------|----------------|-------------------|
| VLA-JEPA | VJEPA2 (`facebook/vjepa2-vitl-fpc64-256`) | `VisionTransformerPredictorAC` |
| CoW-VLA | VidTwin | TBD |

Each model must be evaluated against its own teacher encoder's feature space.

## Background: VLA-JEPA Architecture

VLA-JEPA has five main submodules:

```
VLA_JEPA
  ├── qwen_vl_interface    # QwenVL vision-language encoder
  ├── action_model         # FlowmatchingActionHead (DiT-based action prediction)
  ├── vj_encoder           # VJEPA2 base encoder (frozen, from facebook/vjepa2-vitl-fpc64-256)
  ├── vj_processor         # VJEPA2 video preprocessor
  └── vj_predictor         # VisionTransformerPredictorAC (world model)
```

### Training data flow

During training, the world model is exercised:

```
Single image ──→ QwenVL ──→ action_tokens [B, num_action_tokens, H]
                                    │
Multi-frame video ──→ VJEPA2 encoder ──→ video_embeddings
                         │                      │
                    input_states           gt_states
                    (frames 1..T-1)        (frames 2..T)
                         │                      │
                         └──→ vj_predictor ──→ predicted_states
                                                │
                                        L1_loss(predicted, gt)
```

Key details:
- **`action_tokens`**: Hidden states at `<|action_{}|>` special token positions in QwenVL output. These encode the model's intended action and serve as conditioning for the world model.
- **`embodied_action_tokens`**: Hidden states at `<|embodied_action|>` positions. These feed the `action_model` (FlowmatchingActionHead) to produce continuous actions. **Not used by the world model.**
- VJEPA2 processes multi-view video with temporal tubelet patches. With `num_frames=8` and `tubelet_size=2`, this yields `T=4` temporal tokens.
- The world model uses causal attention: frame t can only attend to frames <= t.

### Inference data flow (current)

During inference (`predict_action()`), the world model is **not used**:

```
Single image ──→ QwenVL ──→ embodied_action_tokens ──→ action_model ──→ normalized_actions
```

The `action_tokens` needed by the world model exist in the prompt but are not extracted.

## Design

### Phase 1: Recording During Rollout

#### Changes to `VLAJEPAModelServer`

Add an optional `--record_dir` flag. When set, the server records per-step data during evaluation:

```python
class VLAJEPAModelServer(PredictModelServer):
    def __init__(self, ..., record_dir: str | None = None, **kwargs):
        ...
        self.record_dir = record_dir
```

#### What to record

At each step of `predict_batch()`, save:

| Field | Shape | Description |
|-------|-------|-------------|
| `agentview_image` | `[H, W, 3]` uint8 | Raw agentview observation |
| `wrist_image` | `[H, W, 3]` uint8 | Raw wrist observation (if available) |
| `action_tokens` | `[num_tokens, hidden_dim]` float16 | QwenVL hidden states at `<\|action_{}\|>` positions |
| `normalized_actions` | `[chunk_size, action_dim]` float32 | Predicted normalized actions |
| `task_description` | str | Task instruction text |

#### How to extract `action_tokens`

Modify `predict_batch()` to also extract `action_tokens` from QwenVL hidden states, mirroring what training does:

```python
# Already computed in predict_action():
#   embodied_action_indices → embodied_action_tokens → action_model

# Additionally extract (same as training forward()):
action_indices = torch.isin(
    qwen_inputs['input_ids'],
    torch.tensor(self.action_token_ids, device=...)
)
action_tokens = last_hidden[action_indices].view(B, -1, H)
```

This requires either:
1. Modifying `predict_action()` in starVLA to also return `action_tokens`, or
2. Overriding the inference path in `VLAJEPAModelServer` to extract them directly.

Option (2) is preferred to avoid forking the upstream library. We can access `self._model.action_token_ids` and run QwenVL ourselves, then call `self._model.action_model.predict_action()` separately.

#### Storage format

```
{record_dir}/
  metadata.json              # model config, checkpoint, benchmark info
  {episode_id}/
    metadata.json             # task_description, num_steps, success
    step_0000.npz             # agentview_image, wrist_image, action_tokens, normalized_actions
    step_0001.npz
    ...
```

Using `.npz` (numpy compressed) for efficient storage of mixed-type per-step data. Images stored as uint8, action_tokens as float16 to save space.

#### Performance considerations

- Recording adds I/O overhead. Use a background thread for writes to avoid blocking inference.
- Float16 for action_tokens reduces storage by 2x with negligible precision loss.
- Estimated storage: ~200KB/step (256x256 image + action_tokens), ~50MB/episode (250 steps).

### Phase 2: Offline Feature Extraction and Evaluation

A standalone script that loads recorded data and evaluates world model predictions.

#### Pipeline

```
For each episode:
  1. Load observation images [step_0, step_1, ..., step_N]
  2. Construct sliding windows of T consecutive frames
  3. For each window [t, t+1, ..., t+T-1]:
     a. Preprocess frames with vj_processor
     b. Encode with vj_encoder → video_embeddings [T//tubelet, dim_per_frame, V*embed]
     c. Split: input_states (frames 1..T-1), gt_states (frames 2..T)
     d. Load action_tokens for steps in this window
     e. Run vj_predictor(input_states, action_tokens) → predicted_states
     f. Compute metrics: cosine_sim(predicted, gt), L2(predicted, gt)
  4. Aggregate metrics across windows and episodes
```

#### Metrics

| Metric | Description |
|--------|-------------|
| **Cosine similarity** | Measures directional alignment of predicted vs ground-truth features |
| **L2 distance** | Measures magnitude of prediction error |
| **R-squared (R²)** | Proportion of variance in GT features explained by predictions |
| **CKA** | Centered Kernel Alignment — representation similarity invariant to rotation/scaling |

All metrics computed per-frame-in-window to analyze how prediction quality degrades over the prediction horizon.

#### Visualization

- **Temporal degradation curves**: Metric vs. prediction horizon (1-step, 2-step, ..., T-1 step ahead)
- **t-SNE / PCA**: Predicted vs. ground-truth features in 2D, colored by timestep
- **Per-episode quality**: Heatmap of prediction quality across steps, useful for identifying when the world model loses track
- **Clustering analysis**: Whether predicted features cluster with their corresponding GT features

#### Script interface

```bash
# Phase 2: Extract features and evaluate
python scripts/world_model_eval/extract_and_evaluate.py \
  --record_dir /path/to/recorded_data \
  --checkpoint ginwind/VLA-JEPA \
  --checkpoint_variant LIBERO \
  --output_dir /path/to/results \
  --num_frames 8 \
  --metrics cosine l2 r2 cka \
  --visualize tsne temporal_curve

# Phase 2 can also be split:
python scripts/world_model_eval/extract_features.py ...   # VJEPA2 encode only
python scripts/world_model_eval/evaluate.py ...           # metrics + viz only
```

### Phase 3: Analysis Tooling (Future)

- Cross-model comparison dashboards (VLA-JEPA vs CoW-VLA)
- Correlation analysis: world model quality vs. task success rate
- Per-task breakdown: which tasks have better/worse world model predictions

## File Structure

```
# Modified
src/vla_eval/model_servers/vla_jepa.py    # Add --record_dir flag and recording logic

# New
scripts/world_model_eval/
  __init__.py
  extract_and_evaluate.py                  # Main entry point for Phase 2
  extract_features.py                      # VJEPA2 feature extraction
  evaluate.py                              # Metrics computation
  visualize.py                             # t-SNE, curves, heatmaps
  utils.py                                 # Data loading, storage helpers
```

## Implementation Order

1. **Phase 1**: Modify `VLAJEPAModelServer` to support `--record_dir` with background I/O.
2. **Phase 2**: Build `extract_and_evaluate.py` — load recorded data, run VJEPA2 encoder + world model, compute metrics.
3. **Phase 2 viz**: Add t-SNE, temporal curves, and clustering visualizations.
4. **Phase 3**: Extend to CoW-VLA (VidTwin encoder, different world model architecture).

## Open Questions

1. **Window stride**: Should sliding windows overlap (stride=1) or be non-overlapping (stride=T)? Overlapping gives more data points but is slower.
2. **Multi-view handling**: VLA-JEPA trains with 2 views (agentview + wrist). Should we record and evaluate both, or just agentview?
3. **Frame rate mismatch**: Training videos may have different frame rates than rollout steps. How should we handle temporal alignment?
4. **CoW-VLA specifics**: The CoW-VLA world model architecture and VidTwin integration need separate analysis before extending this framework to it.

## Alternatives Considered

### A. Run world model during inference (online evaluation)

Instead of recording and evaluating offline, run the world model at each inference step. Rejected because:
- During inference, only single frames are available, but VJEPA2 expects multi-frame video input.
- Would significantly slow down evaluation (VJEPA2 encoding + world model forward pass per step).
- Cannot compute ground-truth features for future frames until those frames are observed.

### B. Pixel-space evaluation

Decode predicted features back to pixels and use image quality metrics (SSIM, LPIPS, FVD). Rejected because:
- VLA-JEPA and CoW-VLA predict in feature space, not pixel space.
- No decoder is trained or available for these feature spaces.
- Feature-space metrics are more appropriate for the learned representations.
