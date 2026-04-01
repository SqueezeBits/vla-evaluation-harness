# Isaac-GR00T — Reproduction Report

NVIDIA's generalist robot foundation model. [GitHub](https://github.com/NVIDIA/Isaac-GR00T) |
[Paper](https://arxiv.org/abs/2503.14734) | 2B params.

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO | **94.9%** | 97.0% | Approximate (−2.1pp) |
| SimplerEnv WidowX | **37.5%** | 57.1%† | Partial (−19.6pp)‡ |
| SimplerEnv Google Robot | — | 67.7%†† | Not yet evaluated |
| CALVIN | — | — | No checkpoint |

† 4-task subset avg (Spoon 64.5, Carrot 65.5, Eggplant 93.0, Block 5.5). Full 7-task avg = 62.1%.
†† Non-standard 6-task set.

### LIBERO

| | |
|---|---|
| **Checkpoint** | `0xAnkitSingh/GR00T-N1.6-LIBERO` (community) |
| **Server config** | [`configs/model_servers/groot/libero.yaml`](../../configs/model_servers/groot/libero.yaml) |
| **Benchmark config** | [`configs/libero_all.yaml`](../../configs/libero_all.yaml) |
| **Results** | [`data/groot-libero/`](data/groot-libero/) |

| Suite | Reproduced | Reported |
|-------|:----------:|:--------:|
| Spatial | 96.6% | 97.65% |
| Object | 98.4% | 98.45% |
| Goal | 96.8% | 97.50% |
| Long | 87.8% | 94.35% |
| **Average** | **94.9%** | **97.0%** |

Pipeline audit: All items match. No discrepancies in pipeline.
- −2.1pp gap likely due to community checkpoint vs official NVIDIA fine-tuning.
- Community checkpoint: `0xAnkitSingh/GR00T-N1.6-LIBERO`, `embodiment_tag=LIBERO_PANDA`.
- Official NVIDIA does not release LIBERO checkpoints — only training recipe (20K steps, batch 640).
- `invert_gripper=True`: maps model output [0,1] (0=close) → [-1,+1] (+1=close) via `1.0 - 2.0 * x`.
- `chunk_size=16`.

### SimplerEnv — WidowX VM

| | |
|---|---|
| **Checkpoint** | `nvidia/GR00T-N1.6-bridge` (official NVIDIA) |
| **Server config** | [`configs/model_servers/groot/simpler_widowx.yaml`](../../configs/model_servers/groot/simpler_widowx.yaml) |
| **Benchmark config** | [`configs/simpler_all_tasks.yaml`](../../configs/simpler_all_tasks.yaml) |
| **Results** | [`data/groot-simpler/`](data/groot-simpler/) |

Status: **Partial** (several pipeline gaps identified; some fixed, two remaining).

Reported (4-task subset): Spoon 64.5%, Carrot 65.5%, Eggplant 93.0%, Block 5.5% = **57.1% avg**.
(Full 7-task avg = 62.1%, but includes non-standard open/close drawer tasks.)

Reproduced (4-task, no overlay, 504 steps): Spoon **75.0%**, Carrot **75.0%**, Block 0.0%, Eggplant 0.0% = **37.5% avg**.

Pipeline audit — all critical items fixed:
1. **State/proprio not sent** (CRITICAL) → **FIXED**: benchmark computes base-relative EE pose from `base_pose + tcp_pose`. Bridge rotation applied in groot.py.
2. **Bridge rotation correction** (CRITICAL) → **FIXED**: `default_rot = [[0,0,1],[0,1,0],[-1,0,0]]` applied to state euler conversion.
3. **accumulate_success** (MEDIUM) → **FIXED**: OR-accumulate success across episode, matching official `rollout_policy.py`.
4. **max_episode_steps** (HIGH) → **FIXED**: 504 steps (matching official `MultiStepConfig` default).
5. **chunk_size** → `chunk_size=1` matching official SimplerEnv README `n_action_steps=1`. (`chunk_size=8` tested but worse — 27% vs 75%).
6. **Gripper polarity** → Resolved: GR00T bridge [0,1] maps correctly without `invert_gripper`.

Remaining gap: PutEggplant 0% (official 93%) — sink camera env without rgb_overlay. CALVIN: no checkpoint.

‡ PutSpoon (75.0%) and PutCarrot (75.0%) both exceed official (64.5%, 65.5%). PutEggplant suffers without overlay (sink camera env).

### SimplerEnv — Google Robot

| | |
|---|---|
| **Checkpoint** | `nvidia/GR00T-N1.6-fractal` (official NVIDIA) |
| **Server config** | [`configs/model_servers/groot/simpler_google_robot.yaml`](../../configs/model_servers/groot/simpler_google_robot.yaml) |
| **Benchmark config** | [`configs/simpler_all_tasks.yaml`](../../configs/simpler_all_tasks.yaml) |
| **Results** | — (not yet evaluated) |

Status: Not yet evaluated.
Checkpoint: `nvidia/GR00T-N1.6-fractal`, `embodiment_tag=OXE_GOOGLE`.
Reported (6-task): 67.7% avg. Requires sticky gripper (15 repeats) — not yet implemented.

## Configuration Notes

- N1 paper ≠ N1.5 ≠ N1.6 — versions differ significantly. All numbers are **N1.6**.
- `Gr00tPolicy` handles normalization, tokenization, and action decoding internally.
- `embodiment_tag` must be set per benchmark: `LIBERO_PANDA`, `OXE_WIDOWX`, `OXE_GOOGLE`.
- Action chunk size = 16 (but official SimplerEnv eval uses n_action_steps=8).