# Reproductions

Systematic verification that vla-eval reproduces published VLA model scores across codebases and benchmarks.

## Summary

| Codebase | Model | LIBERO | CALVIN | SE WidowX |
|----------|-------|:------:|:------:|:---------:|
| [openpi](openpi.md) | Pi0.5 | **97.7%** (96.9%) Reproduced | — | — |
| [Dexbotic](dexbotic.md) | DB-CogACT | **94.7%** (94.9%) Reproduced | **4.02** (4.06) Reproduced | WIP (69.5%) |
| [Isaac-GR00T](groot.md) | GR00T N1.6 | **94.9%** (97.0%) Approximate | — | WIP (57.1%†) |
| [X-VLA](xvla.md) | X-VLA-0.9B | **97.4%** (98.1%) Reproduced | **4.30** (4.43) Reproduced | WIP (95.8%) |
| [StarVLA](starvla.md) | QwenGR00T | — | — | — |

Format: **reproduced** (reported) verdict. — = no checkpoint or not yet evaluated.
† 4-task subset; full 7-task avg = 62.1%.

## Verdict Criteria

Based on binomial 95% CI for 500 episodes per suite (±1.9pp at p=0.95):

- **Reproduced**: within 95% CI of reported score.
- **Approximate**: outside CI but ≤5pp gap.
- **Partial**: >5pp gap, or known systematic issue.
- **Not reproduced**: fundamental pipeline mismatch or failure.

## Files

| File | Contents |
|------|----------|
| [dexbotic.md](dexbotic.md) | DB-CogACT — reported + reproduced + audit (3/3 benchmarks) |
| [xvla.md](xvla.md) | X-VLA — reported + reproduced + audit (3/3 benchmarks) |
| [groot.md](groot.md) | GR00T N1.6 — reported + reproduced + audit (2/3 benchmarks) |
| [openpi.md](openpi.md) | Pi0/Pi0.5 — reported + reproduced + audit (1/3 benchmarks) |
| [starvla.md](starvla.md) | StarVLA — reported scores + prior attempts (0/3 reproduced) |
| [common-pitfalls.md](common-pitfalls.md) | Reproduction pitfalls taxonomy (rotation, gripper, state, etc.) |
| [running-guide.md](running-guide.md) | How to run evaluations + supply/demand data |
| [`data/`](data/) | Raw result JSONs per codebase×benchmark |
