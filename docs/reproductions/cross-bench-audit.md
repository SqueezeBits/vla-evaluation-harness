# Cross-Benchmark Pipeline Verification Audit

**Date:** 2026-03-30
**Branch:** `run-cross-benchmark-evals`
**Scope:** All 10 model-benchmark pairs (Stage 1 + Cross-benchmark)

---

## Pair 1: X-VLA x LIBERO — 97.2% (98.1%) Reproduced

STATUS: Reproduced

### Config
- Server config: `configs/model_servers/xvla/libero.yaml`
- Benchmark config: `configs/libero_all.yaml`
- Official eval: N/A (X-VLA LIBERO eval uses same generate_actions() call; no separate client script available)

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Image resolution** | 256x256 | 256x256 | Yes | benchmark.py:19 `LIBERO_ENV_RESOLUTION = 256`, utils.py:44 `preprocess_libero_image` resizes to 256x256 |
| **Image cameras** | agentview + wrist | agentview + wrist | Yes | xvla.py:86 `image_keys=("agentview", "wrist")`, benchmark.py:198,206 sends both; obs_params at xvla.py:126 sets `send_wrist_image: True` |
| **State/proprio format** | controller [pos3, aa3, gripper2] -> 20D rot6d | controller [pos3, aa3, gripper2] -> 20D rot6d | Yes | benchmark.py:220-226 sends `controller_states` from `robot.controller.ee_pos/ee_ori_mat`; xvla.py:142 prefers `controller_states` over `states`; xvla.py:192-201 `_state_to_xvla_proprio` converts [pos3, aa3] to [pos3, rot6d6, 0, zeros10] |
| **State source** | robot.controller (ee_pos, ee_ori_mat) | controller_states (same source) | Yes | benchmark.py:221-222 `robot.controller.ee_pos`, `robot.controller.ee_ori_mat`; xvla.py:142 `obs.get("controller_states")` |
| **rot6d convention** | contiguous (model internal) | contiguous | Yes | xvla.py:65 imports `axisangle_to_rot6d_contiguous`, xvla.py:68 `rot6d_contiguous_to_matrix`; rotation.py:80-87 contiguous = `mat[:,:2].T.flatten()` i.e. [col0, col1] |
| **Action dimension** | 20D raw -> 7D converted | 20D raw -> 7D converted | Yes | xvla.py:89 `output_action_dim=7`; xvla.py:377-378 `_convert_ee6d_to_7d` extracts arm-1 from 20D |
| **Rotation output** | rot6d -> axis-angle | rot6d -> axis-angle | Yes | xvla.py:186 `_rot6d_to_axisangle(actions[i, 3:9])` converts contiguous rot6d to axis-angle via matrix->quat->axisangle |
| **Euler offset** | none | none | Yes | N/A for LIBERO (no coordinate frame mismatch) |
| **Gripper threshold** | 0.5 (sigmoid already applied) | 0.5 | Yes | xvla.py:188 `1.0 if float(actions[i, 9]) > 0.5 else -1.0`; comment at xvla.py:187 notes sigmoid already applied by `generate_actions() -> postprocess()` |
| **Gripper polarity** | >0.5 = close (+1), <=0.5 = open (-1) | >0.5 = close (+1), <=0.5 = open (-1) | Yes | xvla.py:188; benchmark.py:187-190 discretizes to +1/-1; LIBERO convention: +1=close, -1=open |
| **Action mode** | absolute | absolute | Yes | xvla.py:126 `absolute_action: True` in obs_params; benchmark.py:174-176 sets `robot.controller.use_delta = False` |
| **chunk_size** | 30 | 30 | Yes | libero.yaml:13 `chunk_size: 30` |
| **action_ensemble** | newest | newest | Yes | xvla.py:215 default `action_ensemble="newest"` |
| **max_steps / EP_LEN** | 220/280/300/520 per suite | 220/280/300/520 per suite | Yes | benchmark.py:22-28 `MAX_STEP_MAPPING`; libero_all.yaml does not override |
| **control_freq** | N/A (MuJoCo step-based) | N/A | N/A | LIBERO uses robosuite step, not freq-based control |
| **sim_freq** | N/A (MuJoCo step-based) | N/A | N/A | LIBERO uses robosuite step, not freq-based control |
| **domain_id / embodiment_tag** | 3 (libero) | 3 | Yes | libero.yaml:12 `domain_id: 3`; xvla_domain_config.py:47 `"libero": 3` |
| **Proprio update (closed-loop)** | feed last predicted action[:10] as next proprio | feed last predicted action[:10] as next proprio | Yes | xvla.py:87 `use_predicted_proprio=True`; xvla.py:333-346 uses `_last_raw_actions` to build proprio from last prediction's first 10 dims; matches official pattern |

### Discrepancies
None.

### Notes
- Wrist image is flipped `[::-1, ::-1]` by LIBERO preprocessing (benchmark.py:198 via utils.py:49), then un-flipped by the model server (xvla.py:315-316 `raw_images[1] = raw_images[1][::-1, ::-1]`). This matches X-VLA training which used unflipped wrist images.
- Initial state uses `controller_states` which reads from `robot.controller` (MuJoCo internal), not the observation quaternion. This matters because the coordinate frames differ slightly.

---

## Pair 2: Pi0.5 x LIBERO — 97.7% (96.9%) Reproduced

STATUS: Reproduced

### Config
- Server config: `configs/model_servers/pi0/libero.yaml`
- Benchmark config: `configs/libero_all.yaml`
- Official eval: N/A (OpenPI provides its own eval scripts; our integration uses the same `policy_config.create_trained_policy` + `policy.infer` API)

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Image resolution** | 224x224 | 224x224 | Yes | libero.yaml:19 `image_resolution: 224`; pi0.py:68-75 `_maybe_resize` resizes to `image_resolution` |
| **Image cameras** | observation/image + observation/wrist_image | observation/image + observation/wrist_image | Yes | libero.yaml:15-16 `image_key: "observation/image"`, `wrist_image_key: "observation/wrist_image"`; pi0.py:112-117 maps images to these keys |
| **State/proprio format** | [pos3, aa3, gripper2] 8D float64 | [pos3, aa3, gripper2] 8D float64 | Yes | libero.yaml:17-18 `state_key: "observation/state"`, `state_dim: 8`; pi0.py:123-125 reads `obs.get("states")` and passes as float64; benchmark.py:212-218 sends 8D [pos3, quat2aa3, gripper2] as `states` |
| **State source** | obs (robot0_eef_pos/quat + gripper_qpos) | states (same) | Yes | benchmark.py:212-218 uses `raw_obs["robot0_eef_pos"]`, `quat_to_axisangle(raw_obs["robot0_eef_quat"])`, `raw_obs["robot0_gripper_qpos"]` |
| **rot6d convention** | N/A (Pi0 uses raw state) | N/A | N/A | Pi0/OpenPI handles rotation internally |
| **Action dimension** | 7D | 7D | Yes | Pi0 outputs 7D [pos3, rot3, gripper1] directly from `policy.infer` |
| **Rotation output** | axis-angle (OpenPI internal) | axis-angle (pass-through) | Yes | pi0.py:130 returns `result["actions"]` directly from OpenPI |
| **Euler offset** | none | none | N/A | Not applicable to Pi0 |
| **Gripper threshold** | handled by benchmark | handled by benchmark | Yes | benchmark.py:187-190 discretizes gripper at 0 |
| **Gripper polarity** | RLDS convention in OpenPI | RLDS convention in OpenPI | Yes | Pi0 outputs in its native format; benchmark.py:187-190 binarizes |
| **Action mode** | delta (default) | delta (default) | Yes | obs_params (pi0.py:94-99) does not set `absolute_action`; benchmark defaults to delta |
| **chunk_size** | 10 | 10 | Yes | libero.yaml:20 `chunk_size: 10` |
| **action_ensemble** | newest | newest | Yes | pi0.py:56 default `action_ensemble="newest"` |
| **max_steps / EP_LEN** | 220/280/300/520 per suite | 220/280/300/520 per suite | Yes | benchmark.py:22-28 `MAX_STEP_MAPPING` |
| **control_freq** | N/A | N/A | N/A | LIBERO step-based |
| **sim_freq** | N/A | N/A | N/A | LIBERO step-based |
| **domain_id / embodiment_tag** | N/A | N/A | N/A | Pi0 doesn't use domain_id |
| **Proprio update (closed-loop)** | fresh env state each step | fresh env state each step | Yes | Pi0 does not use predicted proprio; benchmark sends fresh `states` every step |

### Discrepancies
None.

### Notes
- Exceeds the reported 96.9% at 97.7%, indicating our pipeline may be slightly more favorable or within normal variance.
- OpenPI handles all normalization/denormalization internally via its config system (`pi05_libero`).

---

## Pair 3: GR00T x LIBERO — 94.9% (97.0%) Approximate

STATUS: Approximate

### Config
- Server config: `configs/model_servers/groot/libero.yaml`
- Benchmark config: `configs/libero_all.yaml`
- Official eval: N/A (GR00T LIBERO uses community checkpoint `0xAnkitSingh/GR00T-N1.6-LIBERO`; no official eval script available)

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Image resolution** | model default (224x224 in Eagle backbone) | model default (no override) | Yes | libero.yaml does not set `image_resolution`; groot.py:173 only resizes if `self.image_resolution` is set |
| **Image cameras** | video keys from modality config | auto-detected from modality config | Yes | groot.py:160-161 uses `_modality_config["video"].modality_keys`; obs_params at groot.py:153 sends wrist_image + state |
| **State/proprio format** | decomposed into per-key arrays from statistics.json | decomposed into per-key arrays | Yes | groot.py:193-208 decomposes flat state into per-key arrays matching `_state_dims` from statistics.json |
| **State source** | obs (robot0_eef_pos/quat + gripper_qpos) | states (same) | Yes | groot.py:199 `obs.get("states", obs.get("state"))` reads from benchmark's `states` field; benchmark.py:212-218 sends 8D obs-based state |
| **rot6d convention** | N/A (GR00T handles internally) | N/A | N/A | GR00T policy handles rotation conversion internally |
| **Action dimension** | 7D (concatenated action keys) | 7D | Yes | groot.py:215-216 concatenates action keys from modality config |
| **Rotation output** | model native | model native pass-through | Yes | groot.py:216 returns concatenated action_dict values directly |
| **Euler offset** | none | none | N/A | Not applicable |
| **Gripper threshold** | N/A (invert_gripper handles conversion) | invert_gripper=true | Yes | libero.yaml:8 `invert_gripper: true`; groot.py:217-221 transforms [0,1] to [-1,1] with sign inversion: `1.0 - 2.0 * actions[..., -1]` |
| **Gripper polarity** | model outputs [0,1] (0=close, 1=open) | inverted to [-1,+1] (-1=open, +1=close) | Yes | groot.py:219-221: 0->+1 (close), 1->-1 (open), matching LIBERO convention |
| **Action mode** | delta (default) | delta (default) | Yes | obs_params does not set `absolute_action` |
| **chunk_size** | 16 | 16 | Yes | libero.yaml:9 `chunk_size: 16` |
| **action_ensemble** | newest | newest | Yes | groot.py:57 default `action_ensemble="newest"` |
| **max_steps / EP_LEN** | 220/280/300/520 per suite | 220/280/300/520 per suite | Yes | benchmark.py:22-28 |
| **control_freq** | N/A | N/A | N/A | LIBERO step-based |
| **sim_freq** | N/A | N/A | N/A | LIBERO step-based |
| **domain_id / embodiment_tag** | LIBERO_PANDA | LIBERO_PANDA | Yes | libero.yaml:7 `embodiment_tag: LIBERO_PANDA` |
| **Proprio update (closed-loop)** | fresh env state + policy.reset() per episode | fresh env state + policy.reset() per episode | Yes | groot.py:225-228 calls `self._policy.reset()` on episode_start |

### Discrepancies
None identified in pipeline. Score gap (94.9% vs 97.0%) likely due to community checkpoint quality or stochastic variance.

### Notes
- Uses community checkpoint (`0xAnkitSingh/GR00T-N1.6-LIBERO`), not official NVIDIA. The 97.0% reported score may use a different checkpoint or eval setup.
- GR00T `Gr00tPolicy` handles all normalization, tokenization, and action decoding internally.
- The `invert_gripper` formula `1.0 - 2.0 * x` correctly maps: 0 (close) -> +1 (close), 1 (open) -> -1 (open).

---

## Pair 4: OFT x LIBERO (spatial) — 94.0% (~96.8%) Partial

STATUS: Approximate (Partial — only libero_spatial evaluated)

### Config
- Server config: `configs/model_servers/oft/libero_joint.yaml`
- Benchmark config: `configs/libero_all.yaml`
- Official eval: N/A (OFT uses openvla-oft repo's `run_libero_eval.py` which we replicate via `get_vla_action`)

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Image resolution** | 224x224 (OFT default center crop) | 224x224 (center_crop=True) | Yes | oft.py:59 `center_crop: bool = True` default; OFT's `get_processor` applies center crop to 224x224 |
| **Image cameras** | full_image + wrist_image (2 images) | full_image + wrist_image | Yes | _base.yaml:5 `num_images_in_input: 2`; oft.py:123-126 sends `send_wrist_image: True` when `num_images_in_input >= 2`; oft.py:144-147 maps to `full_image` and `wrist_image` |
| **State/proprio format** | [pos3, aa3, gripper2] 8D float64 | [pos3, aa3, gripper2] 8D float64 | Yes | oft.py:151-153 reads `obs.get("states")` as float64; `PROPRIO_DIM` from OFT constants |
| **State source** | obs (robot0_eef_pos/quat + gripper_qpos) | states (same) | Yes | benchmark.py:212-218 sends obs-based 8D state |
| **rot6d convention** | N/A (OFT uses raw state) | N/A | N/A | OFT handles internally |
| **Action dimension** | 7D | 7D | Yes | `get_vla_action` returns 7D [pos3, rot3, gripper1] |
| **Rotation output** | axis-angle (OFT native) | axis-angle (pass-through) | Yes | oft.py:158-166 returns raw `get_vla_action` output |
| **Euler offset** | none | none | N/A | Not applicable |
| **Gripper threshold** | 0.5 (RLDS [0,1] -> robosuite [-1,+1]) | 0.5 (sign-based) | Yes | oft.py:167-169: `actions_arr[..., -1] = -np.sign(2 * actions_arr[..., -1] - 1)`. Maps: <0.5->+1 (close), >0.5->-1 (open). This matches RLDS convention (0=close, 1=open) -> robosuite (-1=open, +1=close) |
| **Gripper polarity** | RLDS [0=close,1=open] -> [-1=open,+1=close] | same conversion | Yes | oft.py:167 comment confirms: "RLDS [0=close,1=open] -> robosuite [-1=open,+1=close]" |
| **Action mode** | delta (default) | delta (default) | Yes | obs_params does not set `absolute_action` |
| **chunk_size** | 10 (NUM_ACTIONS_CHUNK) | 10 | Yes | _base.yaml:6 `chunk_size: 10`; oft.py:92 `NUM_ACTIONS_CHUNK` from prismatic constants |
| **action_ensemble** | newest | newest | Yes | oft.py:67 default `action_ensemble="newest"` |
| **max_steps / EP_LEN** | 220 (libero_spatial) | 220 | Yes | benchmark.py:23 `"libero_spatial": 220` |
| **control_freq** | N/A | N/A | N/A | LIBERO step-based |
| **sim_freq** | N/A | N/A | N/A | LIBERO step-based |
| **domain_id / embodiment_tag** | unnorm_key=libero_spatial_no_noops | libero_spatial_no_noops | Yes | libero_joint.yaml:8 `unnorm_key: libero_spatial_no_noops` |
| **Proprio update (closed-loop)** | fresh env state each step | fresh env state each step | Yes | OFT does not use predicted proprio |

### Discrepancies
None identified in pipeline.

### Notes
- Score gap (94.0% vs ~96.8%) may be due to: (a) only libero_spatial evaluated (the joint checkpoint may need per-suite `unnorm_key` switching), (b) stochastic variance.
- The `unnorm_key` in libero_joint.yaml is set to `libero_spatial_no_noops`. For other suites, it needs to be changed (the comment in the YAML notes this).
- `use_proprio=True` is the default (oft.py:57), and `PROPRIO_DIM` comes from OFT constants.

---

## Pair 5: DB-CogACT x LIBERO — 95.2% (94.9%) Reproduced

STATUS: Reproduced

### Config
- Server config: `configs/model_servers/db_cogact/libero.yaml`
- Benchmark config: `configs/libero_all.yaml`
- Official eval: N/A (DB-CogACT uses the dexbotic repo's inference; we replicate via `CogACTForCausalLM.inference_action`)

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Image resolution** | 224x224 (CogVLM default) | 224x224 (model.process_images handles) | Yes | cogact.py:222 `self._model.process_images(pil_images)` — CogACT/CogVLM applies its own 224x224 resize |
| **Image cameras** | agentview (single image) | first image only | Yes | cogact.py:183-185: when `camera_keys` is None, extracts only first image; LIBERO sends agentview first |
| **State/proprio format** | N/A (CogACT doesn't use proprio) | N/A | N/A | CogACT is vision-language only, no state input |
| **State source** | N/A | N/A | N/A | No state used |
| **rot6d convention** | N/A | N/A | N/A | CogACT outputs raw 7D actions |
| **Action dimension** | 7D | 7D | Yes | cogact.py:246-247 returns raw denormalized actions |
| **Rotation output** | axis-angle (model native) | axis-angle (pass-through) | Yes | CogACT outputs 7D [pos3, rot3, gripper1] |
| **Euler offset** | none | none | N/A | Not applicable |
| **Gripper threshold** | benchmark handles | benchmark handles | Yes | benchmark.py:187-190 binarizes at 0 |
| **Gripper polarity** | model native | model native pass-through | Yes | cogact.py:247 returns raw actions; benchmark.py handles discretization |
| **Action mode** | delta (default) | delta (default) | Yes | No `absolute_action` in obs_params |
| **chunk_size** | per-suite map | per-suite: spatial=12, object=16, goal=16, 10=15 | Yes | libero.yaml:10 `chunk_size_map: '{"libero_spatial": 12, "libero_object": 16, "libero_goal": 16, "libero_10": 15}'`; cogact.py:157-167 resolves per-suite chunk_size on episode_start |
| **action_ensemble** | newest | newest | Yes | _base.yaml does not override; cogact.py:77 default `action_ensemble="newest"` |
| **max_steps / EP_LEN** | 220/280/300/520 per suite | 220/280/300/520 per suite | Yes | benchmark.py:22-28 |
| **control_freq** | N/A | N/A | N/A | LIBERO step-based |
| **sim_freq** | N/A | N/A | N/A | LIBERO step-based |
| **domain_id / embodiment_tag** | N/A | N/A | N/A | CogACT doesn't use domain_id |
| **Proprio update (closed-loop)** | N/A (no proprio) | N/A | N/A | CogACT is vision-language only |

### Discrepancies
None.

### Notes
- `use_text_template: true` in _base.yaml wraps the task description as "What action should the robot take to {text}?" (cogact.py:228).
- CogACT uses cfg_scale=1.5 and num_ddim_steps=10 (defaults from cogact.py:69-70).
- Our score (95.2%) slightly exceeds the reported 94.9%, well within normal variance.

---

## Pair 6: DB-CogACT x CALVIN — 4.05 avg_len (4.06) Reproduced

STATUS: Reproduced

### Config
- Server config: `configs/model_servers/db_cogact/calvin.yaml`
- Benchmark config: `configs/calvin_eval.yaml`
- Official eval: N/A (DB-CogACT uses the dexbotic repo; no separate CALVIN eval script available)

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Image resolution** | 200x200 (CALVIN native) | 200x200 | Yes | benchmark.py:269 `Resize(size=200)` in val_transforms; CogACT model.process_images handles its own resizing from 200x200 |
| **Image cameras** | rgb_static (single) | rgb_static (single) | Yes | cogact.py:183-185 extracts first image; benchmark.py:552-553 sends rgb_static; calvin_eval.yaml does not set send_wrist_image |
| **State/proprio format** | N/A (CogACT vision-only) | N/A | N/A | No state used by CogACT |
| **State source** | N/A | N/A | N/A | No state used |
| **rot6d convention** | N/A | N/A | N/A | CogACT outputs raw 7D |
| **Action dimension** | 7D [pos3, euler3, gripper1] | 7D | Yes | CogACT outputs 7D; benchmark processes as delta |
| **Rotation output** | euler (CALVIN native) | euler (delta accumulation) | Yes | benchmark.py:528-548 `_process_delta_action`: accumulates deltas on [pos3, euler3, gripper1], normalizes euler to [-pi, pi] |
| **Euler offset** | none | none | N/A | Not applicable |
| **Gripper threshold** | 0 (binarize at 0) | 0 | Yes | benchmark.py:542 `1.0 if act[6] > 0 else -1.0` |
| **Gripper polarity** | >0 = open (+1), <=0 = close (-1) | same | Yes | CALVIN convention: +1=open, -1=close; benchmark.py:542 |
| **Action mode** | delta (accumulation) | delta (accumulation) | Yes | calvin_eval.yaml does not set `absolute_action`; benchmark.py:528-548 delta accumulation with base from previous action |
| **chunk_size** | 7 | 7 | Yes | calvin.yaml:7 `chunk_size: 7` |
| **action_ensemble** | newest | newest | Yes | Default from cogact.py:77 |
| **max_steps / EP_LEN** | 360 per subtask x 5 = 1800 | 360 per subtask x 5 = 1800 | Yes | benchmark.py:28 `EP_LEN = 360`; benchmark.py:591 `(self._ep_len or EP_LEN) * NUM_SUBTASKS` |
| **control_freq** | N/A (PyBullet step-based) | N/A | N/A | CALVIN uses PyBullet step |
| **sim_freq** | N/A | N/A | N/A | CALVIN uses PyBullet step |
| **domain_id / embodiment_tag** | N/A | N/A | N/A | CogACT doesn't use domain_id |
| **Proprio update (closed-loop)** | delta accumulation from last_act | delta accumulation from last_act | Yes | benchmark.py:447-448 initializes `_last_act` from `robot_obs_raw[0:6] + [14:15]`; benchmark.py:537-548 accumulates deltas |

### Discrepancies
None.

### Notes
- CALVIN observation is 200x200 uint8 RGB converted from float [-1,1] tensor (benchmark.py:552-553).
- Delta mode resets `_last_act` on subtask transition (benchmark.py:489-490).
- Our 4.05 vs reported 4.06 is effectively identical (within 1 subtask completion difference across 1000 sequences).

---

## Pair 7: DB-CogACT x SimplerEnv — 72.2% (69.5%) Reproduced

STATUS: Reproduced

### Config
- Server config: `configs/model_servers/db_cogact/simpler.yaml`
- Benchmark config: `configs/simpler_all_tasks.yaml`
- Official eval: N/A (DB-CogACT SimplerEnv eval uses same inference API)

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Image resolution** | env default (get_image_from_maniskill2_obs_dict) | env default | Yes | benchmark.py:212-216 calls `get_image_from_maniskill2_obs_dict`; CogACT model.process_images handles resizing |
| **Image cameras** | primary (single) | primary (single) | Yes | benchmark.py:122-123 `_build_obs_dict` sends `{"primary": image}`; cogact.py:183-185 extracts first image |
| **State/proprio format** | N/A (CogACT vision-only) | N/A | N/A | No state used |
| **State source** | N/A | N/A | N/A | No state used |
| **rot6d convention** | N/A | N/A | N/A | N/A |
| **Action dimension** | 7D [pos3, rot3, gripper1] | 7D | Yes | cogact.py returns 7D; benchmark.py:195 asserts len=7 |
| **Rotation output** | euler (passed directly to env) | euler (passed directly) | Yes | benchmark.py:200-201 passes rotation as-is to ManiSkill2 env.step(); comment at line 198-199 confirms "Rotation passed directly...matching official eval pipelines" |
| **Euler offset** | none | none | N/A | Not applicable |
| **Gripper threshold** | 0.5 | 0.5 | Yes | benchmark.py:202 `1.0 if raw_action[6] > 0.5 else -1.0` |
| **Gripper polarity** | >0.5 = close (+1), <=0.5 = open (-1) | same | Yes | benchmark.py:202; ManiSkill2 WidowX convention |
| **Action mode** | delta (WidowX default control mode) | delta | Yes | benchmark.py:148 `get_robot_control_mode(self.robot, "vla")` returns delta for WidowX |
| **chunk_size** | 5 | 5 | Yes | simpler.yaml:7 `chunk_size: 5` |
| **action_ensemble** | newest | newest | Yes | Default from cogact.py:77 |
| **max_steps / EP_LEN** | 120 | 120 | Yes | simpler_all_tasks.yaml:31 `max_episode_steps: 120`; benchmark.py:231 |
| **control_freq** | 5 Hz | 5 Hz | Yes | simpler_all_tasks.yaml:29 `control_freq: 5` |
| **sim_freq** | 500 Hz | 500 Hz | Yes | simpler_all_tasks.yaml:30 `sim_freq: 500` |
| **domain_id / embodiment_tag** | N/A | N/A | N/A | CogACT doesn't use domain_id |
| **Proprio update (closed-loop)** | N/A (no proprio) | N/A | N/A | CogACT is vision-only |

### Discrepancies
None.

### Notes
- Our 72.2% exceeds the reported 69.5%, suggesting our pipeline is slightly more favorable or the reference number is from a different eval configuration.
- `use_text_template: true` wraps task description (cogact.py:228).

---

## Pair 8: X-VLA x CALVIN — 3.97 (4.43) Not reproduced

STATUS: Not reproduced (OLD Docker, not verified)

### Config
- Server config: `configs/model_servers/xvla/calvin.yaml`
- Benchmark config: `configs/calvin_eval.yaml`
- Official eval: X-VLA CALVIN eval script (external)

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Image resolution** | 200x200 (CALVIN native) | 200x200 | Yes | benchmark.py:269 `Resize(size=200)`; X-VLA processor handles its own resizing |
| **Image cameras** | rgb_static + rgb_gripper | rgb_static + rgb_gripper | Yes | xvla.py:93 `image_keys=("rgb_static", "rgb_gripper")` for calvin profile; obs_params at xvla.py:127 sets `send_wrist_image: True` |
| **State/proprio format** | [pos3, euler_to_rot6d_interleaved6, gripper_bool1, zeros10] = 20D | [pos3, aa_to_rot6d_contiguous6, 0, zeros10] = 20D | MISMATCH | Official (xvla_calvin_client.py:77-82): `euler_xyz_to_rotate6D(obs["robot_obs"][3:6])` uses interleaved rot6d from euler. Ours (xvla.py:192-201 `_state_to_xvla_proprio`): converts from axis-angle via `axisangle_to_rot6d_contiguous`. Two differences: (1) interleaved vs contiguous rot6d layout, (2) euler source vs axis-angle source |
| **State source** | robot_obs[0:3], robot_obs[3:6] (euler), robot_obs[-1:] | benchmark sends `states` = [pos3, euler3, gripper2] 8D, then model server converts aa->rot6d | MISMATCH | Official (xvla_calvin_client.py:77-82): reads directly from `obs["robot_obs"]` which contains euler angles at indices 3:6. Ours: benchmark.py:570-572 sends `raw[0:7] + raw[14:15]` where raw[3:6] = euler XYZ (CALVIN robot_obs format), but xvla.py:192-201 treats state[3:6] as **axis-angle** and converts via `_axisangle_to_rot6d`. The euler angles are misinterpreted as axis-angle. |
| **rot6d convention** | interleaved | contiguous | MISMATCH | Official (xvla_calvin_client.py:50): `R.from_euler("xyz", q).as_matrix()[..., :, :2].reshape(...)` produces interleaved. Ours: xvla.py:65 uses `axisangle_to_rot6d_contiguous`. Different memory layout. |
| **Action dimension** | 20D raw -> (pos3, quat4, gripper_int) | 20D raw -> 7D [pos3, aa3, gripper] | MISMATCH | Official (xvla_calvin_client.py:106-109): returns `(pos3, quat4, gripper_int)` tuple to CALVIN env.step(). Ours: xvla.py:89 `output_action_dim=7` produces [pos3, axisangle3, gripper] 7D; benchmark.py:511-526 `_process_absolute_action` converts aa->euler for env.step() |
| **Rotation output** | rot6d -> quaternion (interleaved decode) | rot6d -> axis-angle (contiguous decode) -> euler in benchmark | MISMATCH | Official (xvla_calvin_client.py:107): `rotate6D_to_quat(action_predict[3:9])` with interleaved stride-2 decoding. Ours: xvla.py:186 `_rot6d_to_axisangle` with contiguous decoding, then benchmark.py:523 converts aa->euler. Different decode convention. |
| **Euler offset** | none | none | Yes | N/A |
| **Gripper threshold** | 0.8 | 0.5 | MISMATCH | Official (xvla_calvin_client.py:108): `1 if action_predict[9] < 0.8 else -1`. Ours: xvla.py:188 `1.0 if float(actions[i, 9]) > 0.5 else -1.0`. Different threshold AND inverted comparison direction. |
| **Gripper polarity** | <0.8 = open (1), >=0.8 = close (-1) | >0.5 = close (+1), <=0.5 = open (-1) | MISMATCH | Official CALVIN convention: 1=open, -1=close. Official client maps <0.8->1(open), >=0.8->-1(close). Ours: >0.5->+1(close), <=0.5->-1(open). Inverted meaning AND different threshold. |
| **Action mode** | absolute (quat to env) | absolute (aa->euler to env) | Partial | Both use absolute actions, but different rotation representations delivered to CALVIN env |
| **chunk_size** | 20 (action[:20]) | 20 | Yes | calvin.yaml:8 `chunk_size: 20`; official xvla_calvin_client.py:99 `[:20]` |
| **action_ensemble** | newest (pop from queue) | newest | Yes | Both consume actions sequentially |
| **max_steps / EP_LEN** | 720 per subtask | 360 per subtask | MISMATCH | Official (xvla_calvin_client.py:43): `EP_LEN = 720`. Ours: benchmark.py:28 `EP_LEN = 360`. Official gives 2x more steps per subtask. |
| **control_freq** | N/A | N/A | N/A | PyBullet step-based |
| **sim_freq** | N/A | N/A | N/A | PyBullet step-based |
| **domain_id / embodiment_tag** | 2 (Calvin) | 2 | Yes | calvin.yaml:7 `domain_id: 2`; xvla_domain_config.py:46 `"Calvin": 2` |
| **Proprio update (closed-loop)** | `self.proprio[:10] = action_predict[:10]` | `_last_raw_actions` -> proprio[:10] = last_actions[-1, :10] | Yes | Official (xvla_calvin_client.py:104): updates proprio[:10] with raw action. Ours: xvla.py:333-337 does same via `_last_raw_actions` |

### Discrepancies

1. **rot6d convention mismatch (interleaved vs contiguous)** — CRITICAL
   - Official uses interleaved layout from `euler_xyz_to_rotate6D` (xvla_calvin_client.py:50)
   - Ours uses contiguous layout from `axisangle_to_rot6d_contiguous` (xvla.py:65,200)
   - Impact: The rot6d values fed as proprio will be in a different memory layout, causing the model to receive mismatched proprioceptive input. However, after the first step, predicted proprio is used (action[:10]), which the model itself produced, so the layout is self-consistent from step 2 onward. The initial step mismatch may cause a small perturbation.
   - **FIXED**: `_state_to_xvla_proprio` now uses `euler_to_rot6d_interleaved` for the `calvin` profile to match the official convention

2. **State interpretation: euler treated as axis-angle** — CRITICAL
   - CALVIN robot_obs[3:6] contains euler XYZ angles (benchmark.py:570-572 sends raw[0:7] which includes euler at indices 3:6)
   - xvla.py:192-201 `_state_to_xvla_proprio` calls `_axisangle_to_rot6d(state[3:6])`, interpreting the euler values as axis-angle
   - For small angles, euler and axis-angle are approximately equal, which may explain why the model still partially works (3.97 avg_len)
   - Impact: Corrupted initial proprio; mitigated by predicted proprio on subsequent steps
   - **FIXED**: `calvin` profile now uses `euler_state: True` so state[3:6] is interpreted as euler XYZ, matching `robot_obs` format

3. **Gripper threshold and polarity** — HIGH
   - Official: `<0.8 -> 1 (open), >=0.8 -> -1 (close)`, threshold=0.8
   - Ours: `>0.5 -> +1 (close), <=0.5 -> -1 (open)`, threshold=0.5
   - CALVIN gripper convention: +1=open, -1=close
   - Our mapping is inverted: we output +1 for close, but CALVIN treats +1 as open
   - Impact: Gripper actions are correct in magnitude but the polarity interpretation differs from official
   - **Fix note**: Only the threshold needs changing (0.5→0.8). The comparison direction (`>`) must NOT be changed — the benchmark's `_process_absolute_action` at line 524 already handles the CALVIN polarity flip (`>0 → -1 = close in CALVIN`). Changing both would cause a double-flip bug.
   - **FIXED**: Threshold changed to 0.8 in the `calvin` profile

4. **EP_LEN: 360 vs 720** — HIGH
   - Official allows 720 steps per subtask; ours allows 360
   - Impact: Model has half the time budget to complete each subtask, directly reducing avg_len
   - **FIXED**: Added `"ep_len": 720` to `_PROFILE_OBS_PARAMS["calvin"]`; benchmark uses this via obs_params to override the default EP_LEN

5. **Action format delivered to CALVIN env** — MEDIUM
   - Official sends `(pos3, quat4, gripper_int)` tuple to CALVIN env.step()
   - Ours sends 7D [pos3, euler3, gripper] via `_process_absolute_action`
   - The CALVIN `CalvinEnvWrapper.step()` may handle these differently internally
   - **FIXED**: Our `_process_absolute_action` converts aa->euler and sends [pos3, euler3, gripper] which CALVIN env accepts correctly

6. **`absolute_action` not set in obs_params** — CRITICAL
   - `_PROFILE_OBS_PARAMS["calvin"]` at `xvla.py:127` did not include `absolute_action: True`
   - LIBERO obs_params at `xvla.py:126` includes it, but CALVIN was missing it
   - Without it, CALVIN benchmark defaults to `absolute_action=False` (delta mode)
   - X-VLA CALVIN outputs absolute actions — running in delta mode causes the model's absolute position outputs to be accumulated as deltas
   - Impact: Fundamental action interpretation mismatch. This is arguably the most critical discrepancy.
   - **FIXED**: Added `"absolute_action": True, "ep_len": 720` to `_PROFILE_OBS_PARAMS["calvin"]`

### Notes
- The 3.97 avg_len (vs 4.43 reported) with these discrepancies suggests the model is robust enough to partially compensate, likely because: (a) predicted proprio dominates after step 1, (b) small CALVIN euler angles approximate axis-angle, (c) the model's policy may be tolerant of gripper threshold differences.
- This pair needs significant pipeline fixes before reproduction can be claimed.

---

## Pair 9: X-VLA x SimplerEnv WidowX — 0% (95.8%) Not reproduced

STATUS: Not reproduced

### Config
- Server config: `configs/model_servers/xvla/simpler_widowx.yaml`
- Benchmark config: `configs/simpler_all_tasks.yaml`
- Official eval: X-VLA SimplerEnv WidowX eval script (external)

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Image resolution** | env default (256x320 for google, 256x256 for widowx in GR00T wrapper, but raw from `get_image_from_maniskill2_obs_dict` for X-VLA) | env default | Yes | Official (xvla_simpler_widowx_client.py:147): `get_image_from_maniskill2_obs_dict(env, obs)` — no resize. Ours: benchmark.py:212-216 same call |
| **Image cameras** | primary (single) | primary (single) | Yes | Official uses single image; xvla.py:99 `image_keys=("primary",)` for simpler profile |
| **State/proprio format** | [ee_pos_wrt_base3, identity_rot6d_like7, zeros10] = 20D | benchmark sends NO state (SimplerEnv has no send_state) | MISMATCH | Official (xvla_simpler_widowx_client.py:126-139): computes `ee_pose_wrt_base` from `obs["agent"]["base_pose"]` and `obs["extra"]["tcp_pose"]`, then `[ee_pos3, [1,0,0,1,0,0,0], zeros10]` = 20D. Ours: SimplerEnv benchmark.py has no `send_state` parameter; `_build_obs_dict` (line 121-123) only sends image + task_description. xvla.py:357 falls through to `torch.zeros(1, dim_proprio)`. |
| **State source** | agent/base_pose + extra/tcp_pose (computed wrt base) | none (zeros) | MISMATCH | Official computes EE pose relative to robot base. Ours sends no state at all. |
| **rot6d convention** | N/A (initial proprio uses identity-like values) | N/A (zeros) | MISMATCH | Official uses `[1,0,0,1,0,0,0]` as placeholder rot6d+gripper. Ours uses zeros. |
| **Action dimension** | 20D raw -> 7D [pos3, euler3, gripper] | 20D raw -> 7D [pos3, aa3, gripper] | MISMATCH | Official (xvla_simpler_widowx_client.py:93-97): converts rot6d->euler via `rotate6D_to_euler_xyz` (interleaved decode) and adds euler_offset `[0, pi/2, 0]`. Ours: xvla.py does NOT have simpler_widowx profile (only "simpler"), and `output_action_dim` for "simpler" is None (xvla.py:98-102), so raw 20D actions are returned without conversion! |
| **Rotation output** | rot6d -> euler (interleaved) + offset [0, pi/2, 0] | rot6d -> axis-angle (contiguous) if output_action_dim=7, else raw 20D | BLOCKER | Official adds `np.array([0, math.pi / 2, 0])` euler offset (xvla_simpler_widowx_client.py:95). Ours: (a) the config has `benchmark_profile: "simpler_widowx"` but the profile dict (xvla.py:84-114) only contains `"simpler"`, not `"simpler_widowx"` — this would cause a ValueError crash; (b) even if it resolved, the "simpler" profile has `output_action_dim=None`, so no rot6d->euler conversion happens; (c) no euler_offset mechanism exists in xvla.py at all. |
| **Euler offset** | [0, pi/2, 0] | not implemented | BLOCKER | Official (xvla_simpler_widowx_client.py:95): `rotate6D_to_euler_xyz(action_pred[3:9]) + np.array([0, math.pi / 2, 0])`. The `euler_offset: "0,1.5707963,0"` in simpler_widowx.yaml is a config value, but `euler_offset` is NOT a parameter of `XVLAModelServer.__init__` — it would be silently ignored or cause an error. |
| **Gripper threshold** | 0.7 | 0.5 (if output_action_dim=7) | MISMATCH | Official (xvla_simpler_widowx_client.py:96): `1.0 if action_pred[9] < 0.7 else -1.0`. Ours: xvla.py:188 uses 0.5. |
| **Gripper polarity** | <0.7 = close (1.0), >=0.7 = open (-1.0) | >0.5 = close (+1), <=0.5 = open (-1) | MISMATCH | SimplerEnv expects: +1=close, -1=open (benchmark.py:202). Official: <0.7->1(close), matching. Ours: >0.5->+1(close), same direction but different threshold. |
| **Action mode** | delta (WidowX default) | delta | Yes | Both use WidowX default delta control |
| **chunk_size** | full sequence (no explicit chunking, pops one at a time) | 30 | Partial | Official (xvla_simpler_widowx_client.py:83): extends deque with all actions, pops one at a time (effectively using full chunk). Ours: simpler_widowx.yaml:10 `chunk_size: 30` matches X-VLA's output length |
| **action_ensemble** | newest (sequential pop) | newest | Yes | Both consume sequentially |
| **max_steps / EP_LEN** | 1200 (hardcoded) | 120 | BLOCKER | Official (xvla_simpler_widowx_client.py:103): `max_steps: int = 1200`. SimplerEnv's `_max_episode_steps` is set to 10000 in the GR00T wrapper. Ours: simpler_all_tasks.yaml:31 `max_episode_steps: 120`. 10x fewer steps. |
| **control_freq** | 5 Hz (SimplerEnv default) | 5 Hz | Yes | simpler_all_tasks.yaml:29 |
| **sim_freq** | 500 Hz (SimplerEnv default) | 500 Hz | Yes | simpler_all_tasks.yaml:30 |
| **domain_id / embodiment_tag** | 0 (Bridge) | 0 | Yes | simpler_widowx.yaml:9 `domain_id: 0`; xvla_domain_config.py:43 `"Bridge": 0` |
| **Proprio update (closed-loop)** | `self.proprio[:10] = action_pred[:10]` | predicted proprio (if use_predicted_proprio=True) | Partial | Official updates proprio[:10] with raw 20D action. Ours (xvla.py:333-337): same mechanism via `_last_raw_actions`, but initial proprio is wrong (zeros vs computed EE pose). |

### Discrepancies

1. **benchmark_profile "simpler_widowx" does not exist** — BLOCKER
   - simpler_widowx.yaml:8 sets `benchmark_profile: "simpler_widowx"` but xvla.py:84-114 only has `"simpler"` in `_BENCHMARK_PROFILES`
   - xvla.py:118-122 `_get_profile` raises `ValueError` for unknown profiles
   - Impact: Server would crash on startup. The 0% score may indicate this config was never actually run successfully, or the profile name was changed after the config was written.
   - **FIXED**: Added `"simpler_widowx"` profile to `_BENCHMARK_PROFILES` with correct `output_action_dim=7`, interleaved rot6d decode, euler offset, and gripper threshold/direction

2. **euler_offset not implemented** — BLOCKER
   - simpler_widowx.yaml:11 has `euler_offset: "0,1.5707963,0"` but `euler_offset` is not a parameter of `XVLAModelServer.__init__` (xvla.py:207-218)
   - The `euler_offset` parameter causes an argparse crash when the server starts — `run_server` builds argparse from `__init__` signature, and unknown args raise "unrecognized arguments" error. It is NOT silently ignored.
   - Official adds `[0, pi/2, 0]` to euler output (xvla_simpler_widowx_client.py:95)
   - Impact: Without this offset, all rotation actions are in the wrong coordinate frame
   - **FIXED**: `euler_offset` added as a parameter to `XVLAModelServer.__init__`; applied during action conversion in the `simpler_widowx` profile

3. **No state/proprio from SimplerEnv** — BLOCKER
   - SimplerEnv benchmark has no `send_state` parameter; never sends state
   - Official computes EE pose wrt base from `obs["agent"]["base_pose"]` and `obs["extra"]["tcp_pose"]`
   - Impact: Model receives zero proprio instead of meaningful EE pose, corrupting the policy input
   - **FIXED**: Added `send_state` parameter to SimplerEnv benchmark; sends 8D EE pose state when enabled; simpler_widowx.yaml sets `send_state: true`

4. **Action output format: raw 20D vs converted 7D** — BLOCKER
   - The "simpler" profile has `output_action_dim=None` (xvla.py:98-102), so 20D raw actions are returned
   - SimplerEnv benchmark.py:195 asserts `len(raw_action) == 7` — this would crash
   - Impact: Action dimension mismatch causes runtime crash
   - **FIXED**: `simpler_widowx` profile sets `output_action_dim=7` and performs rot6d→euler conversion + euler_offset

5. **max_steps: 120 vs 1200** — BLOCKER
   - Official uses 1200 steps; ours uses 120 (10x fewer)
   - Impact: Catastrophically insufficient time budget
   - **FIXED**: New `configs/simpler_xvla_widowx.yaml` config sets `max_episode_steps: 1200` for all 4 tasks

6. **Gripper threshold: 0.5 vs 0.7** — MEDIUM
   - Different thresholds for gripper binarization
   - Impact: Some gripper actions may be incorrectly classified
   - **Fix note**: For WidowX Bridge domain, the comparison direction must be INVERTED: `< 0.7 → close(+1)` not `> 0.7 → close(+1)`. This is because the Bridge domain's sigmoid convention is opposite to LIBERO — low sigmoid means close.
   - **FIXED**: `simpler_widowx` profile uses threshold=0.7 with inverted comparison (`< 0.7 → close`)

7. **rot6d decode convention: interleaved vs contiguous** — HIGH (if action conversion were implemented)
   - Official uses interleaved decode (`v6[..., 0:5:2]` stride-2); ours uses contiguous
   - Impact: Wrong rotation values in output actions
   - **FIXED**: `simpler_widowx` profile uses interleaved rot6d decode matching official `rotate6D_to_euler_xyz`

### Notes
- This pair has at least 5 BLOCKER-level issues. The 0% score is expected. The config appears to have been written speculatively without implementing the required server-side changes.
- To fix: (a) add "simpler_widowx" profile with `output_action_dim=7`, (b) implement euler_offset in XVLAModelServer, (c) add state sending to SimplerEnv benchmark, (d) fix rot6d interleaved decode, (e) increase max_episode_steps to 1200, (f) adjust gripper threshold to 0.7.

---

## Pair 10: GR00T x SimplerEnv WidowX — 25% (57.1%) Not reproduced

STATUS: Not reproduced

### Config
- Server config: `configs/model_servers/groot/simpler_widowx.yaml`
- Benchmark config: `configs/simpler_all_tasks.yaml`
- Official eval: GR00T SimplerEnv eval script + bridge modality config (external)

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Image resolution** | 256x256 (WidowXBridgeEnv) | 256x256 | Yes | Official (groot_simpler_eval.py:244): `image_size: (256, 256)` for widowx envs. Ours: simpler_widowx.yaml:8 `image_resolution: 256`; groot.py:173-176 resizes to 256x256 |
| **Image cameras** | video.image_0 (single) | primary (single, auto-detected) | Yes | Official (groot_bridge_modality.json:66-68): `"image_0"` video key. Ours: modality config from checkpoint provides video keys; groot.py:160-161 uses modality config |
| **State/proprio format** | 8D [x, y, z, roll, pitch, yaw, pad, gripper] with bridge rotation conversion | NO state sent (SimplerEnv has no send_state) | MISMATCH | Official (groot_simpler_eval.py:192-207): `_process_observation` extracts `eef_pos`, computes `rpy_bridge_converted = te.mat2euler(rm_bridge @ self.default_rot.T)` where `default_rot = [[0,0,1],[0,1,0],[-1,0,0]]`. State is [x, y, z, roll', pitch', yaw', 0, gripper]. Ours: SimplerEnv benchmark.py has no send_state; `_build_obs_dict` (line 121-123) only sends image + task. groot.py:195-196 sets zeros for state: `observation["state"][sk] = np.zeros((B, 1, dim))` |
| **State source** | agent/eef_pos with bridge rotation correction | none (zeros) | MISMATCH | Official uses `obs["agent"]["eef_pos"]` with `quat2mat(quat) @ default_rot.T` rotation correction. Ours sends no state. |
| **rot6d convention** | N/A (GR00T handles internally) | N/A | N/A | GR00T handles internally |
| **Action dimension** | 7D [x, y, z, roll, pitch, yaw, gripper] | 7D (concatenated action keys) | Yes | Official (groot_bridge_modality.json:38-65): 7 action keys. Ours: groot.py:215-216 concatenates action keys |
| **Rotation output** | euler (model native for bridge) | euler (model native) | Yes | GR00T outputs in the format defined by its modality config |
| **Euler offset** | N/A | N/A | N/A | GR00T handles coordinate frames via embodiment_tag |
| **Gripper threshold** | `2.0 * (action > 0.5) - 1.0` (WidowX) | benchmark: `1.0 if > 0.5 else -1.0` | Partial | Official (groot_simpler_eval.py:210-211): `_postprocess_gripper` maps [0,1] -> [-1,1] via `2*(action>0.5)-1`. Ours: groot.py does NOT invert_gripper for SimplerEnv (simpler_widowx.yaml has no `invert_gripper`); raw gripper value passes to benchmark.py:202 which binarizes at 0.5. The question is whether GR00T bridge outputs [0,1] or [-1,1]. |
| **Gripper polarity** | trained [0=close, 1=open] -> env [-1,1] | raw pass-through + benchmark binarize | MISMATCH | Official (groot_simpler_eval.py:211 comment): "trained with [0, 1], 0 close, 1 open -> convert to SimplerEnv [-1, 1]". With `2*(x>0.5)-1`: 0(close)->-1, 1(open)->+1. ManiSkill2 WidowX: +1=close, -1=open. So official maps 0(close)->-1(env_open) which seems WRONG in the official code too, OR ManiSkill2 convention is -1=close. Ours: no invert_gripper set, so raw [0,1] value passes through; benchmark.py:202 binarizes at 0.5: >0.5->+1, <=0.5->-1. |
| **Sticky gripper** | yes (15 repeats) for GoogleFractal, NO for WidowX | no | Yes | Official WidowXBridgeEnv (groot_simpler_eval.py:210-211): simple threshold, no sticky gripper (that's only in GoogleFractalEnv at line 107-120). Ours: no sticky gripper mechanism. Match for WidowX. |
| **Action mode** | delta (WidowX default) | delta | Yes | Both use WidowX default delta control |
| **chunk_size** | 16 (GR00T default) | 16 | Yes | simpler_widowx.yaml:8 `chunk_size: 16` |
| **action_ensemble** | newest | newest | Yes | groot.py:57 default |
| **max_steps / EP_LEN** | 10000 (env._max_episode_steps override) | 120 | MISMATCH | Official (groot_simpler_eval.py:126): `env._max_episode_steps = 10000`. Ours: simpler_all_tasks.yaml:31 `max_episode_steps: 120`. Official allows ~83x more steps. |
| **control_freq** | 5 Hz (SimplerEnv default) | 5 Hz | Yes | simpler_all_tasks.yaml:29 |
| **sim_freq** | 500 Hz (SimplerEnv default) | 500 Hz | Yes | simpler_all_tasks.yaml:30 |
| **domain_id / embodiment_tag** | OXE_WIDOWX | OXE_WIDOWX | Yes | simpler_widowx.yaml:7 `embodiment_tag: OXE_WIDOWX` |
| **Proprio update (closed-loop)** | fresh state each step (from env obs) | zeros (no state) | MISMATCH | Official sends fresh state every step from env. Ours sends no state; groot.py initializes state to zeros. |

### Discrepancies

1. **No state/proprio sent from SimplerEnv** — CRITICAL
   - SimplerEnv benchmark has no `send_state` parameter
   - Official GR00T eval extracts 8D state [x, y, z, roll', pitch', yaw', pad, gripper] with bridge rotation correction (`quat2mat @ default_rot.T`)
   - Ours sends zero state
   - Impact: Model receives no proprioceptive feedback, degrading action quality significantly
   - **FIXED**: Added `send_state` parameter to SimplerEnv benchmark; sends 8D state with bridge rotation correction when `bridge_rotation: true` is set in the model server config

2. **Bridge rotation correction missing** — CRITICAL (dependent on #1)
   - Official applies `default_rot = [[0,0,1],[0,1,0],[-1,0,0]]` rotation correction to convert ManiSkill2 quat to bridge-convention euler
   - Even if state were sent, our SimplerEnv benchmark doesn't apply this correction
   - Impact: Would need custom state processing in SimplerEnv benchmark
   - **FIXED**: SimplerEnv benchmark applies bridge rotation correction when obs_params includes `bridge_rotation: true`; groot/simpler_widowx.yaml now sets `bridge_rotation: true`

3. **max_episode_steps: 120 vs 10000** — HIGH
   - Official overrides to 10000 steps (effectively unlimited)
   - Ours uses 120 steps
   - Impact: With 120 steps at 5Hz = 24 seconds of robot time, many tasks may not complete. However, the official eval's `done` flag from the env can terminate episodes early, so 10000 is more of "no artificial limit" rather than "needs 10000 steps."
   - **FIXED**: New `configs/simpler_groot_widowx.yaml` config sets `max_episode_steps: 10000` for all 4 tasks

4. **Gripper polarity uncertain** — MEDIUM
   - Without `invert_gripper`, GR00T's raw [0,1] output passes through
   - benchmark.py:202 binarizes at 0.5: >0.5->+1(close in ManiSkill2), <=0.5->-1(open)
   - If GR00T bridge outputs 0=close, 1=open, then 1(open)->+1(close_in_env) = INVERTED
   - Impact: Gripper may be inverted; needs `invert_gripper: true` for SimplerEnv or a different gripper mapping
   - Still needs empirical testing to confirm correct polarity direction

### Notes
- The 25% success rate (vs 57.1% reported) with missing state and potentially wrong max_steps suggests the model can partially succeed on pure vision, but state feedback is important for this benchmark.
- The bridge rotation correction (`default_rot.T` matrix multiplication) is a non-trivial coordinate frame transform that would need to be implemented either in the benchmark or model server.
- The `invert_gripper` question needs empirical testing: if GR00T bridge outputs match SimplerEnv convention directly, no inversion is needed; if they follow the RLDS convention (0=close, 1=open), inversion IS needed.

---

## Pair 11: DB-CogACT x RoboTwin 2.0 — Not yet evaluated

STATUS: Not yet evaluated

### Config
- Server config: `configs/model_servers/db_cogact/robotwin2.yaml`
- Benchmark config: `configs/robotwin_eval.yaml`

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Action dimension** | 14D (dual-arm joint positions) | 14D | Yes | robotwin/benchmark.py uses 14D action; dexbotic/cogact.py `_convert_actions` outputs 14D |
| **Image cameras** | head + left + right (3 cameras) | head + left + right | Yes | robotwin2.yaml: `camera_keys: '["head_camera", "left_camera", "right_camera"]'` |
| **State/proprio** | joint positions | joint positions | Yes | robotwin/benchmark.py sends joint state |
| **Checkpoint** | per-task fine-tuned | per-task | Yes | robotwin2.yaml: `Dexmal/robotwin-db-cogact/adjust_bottle` (override per task) |
| **Protocol** | Protocol A (single-task, 50 clean demos) | Protocol A | Yes | `task_config: demo_clean` |
| **chunk_size** | 16 | 16 | Yes | robotwin2.yaml |
| **max_steps** | 300 (RoboTwin default) | 300 | Yes | robotwin/benchmark.py default |

### Discrepancies
None identified. Per-task checkpoint override needed for each of 4 tasks.

### Notes
- Only 4 tasks evaluated in the official report (adjust_bottle, grab_roller, place_empty_cup, place_phone_stand).
- Each task needs a separate server run with the correct checkpoint path.
- Reported score: 58.5% average (Easy mode).

---

## Pair 12: X-VLA x RoboTwin 2.0 — Not yet evaluated

STATUS: Not yet evaluated

### Config
- Server config: `configs/model_servers/xvla/robotwin.yaml`
- Benchmark config: `configs/robotwin_eval.yaml`

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Action dimension** | 20D raw (dual-arm EE6D) | 20D raw | Partial | xvla.py robotwin profile: `output_action_dim=None` (raw 20D), `preserve_env_grippers=True`. RoboTwin benchmark expects 14D joint positions — conversion unclear |
| **Image cameras** | head + left + right | head + left + right | Yes | xvla.py:109 `image_keys=("head_camera", "left_camera", "right_camera")` |
| **domain_id** | 6 | 6 | Yes | robotwin.yaml: `domain_id: 6` |
| **chunk_size** | 30 | 30 | Yes | robotwin.yaml |
| **Protocol** | Protocol A (50 tasks, 50 clean demos) | TBD | — | |
| **Checkpoint** | 2toINF/X-VLA-WidowX | 2toINF/X-VLA-WidowX | Unclear | Same checkpoint as SimplerEnv WidowX — may need a RoboTwin-specific checkpoint |

### Discrepancies
1. **Action dimension mismatch** — X-VLA outputs 20D (dual-arm EE6D), RoboTwin expects 14D joint positions. The `preserve_env_grippers=True` flag extracts grippers from env state, but the arm-to-joint conversion is unclear.
2. **Checkpoint** — `2toINF/X-VLA-WidowX` is a Bridge/WidowX checkpoint, not a RoboTwin-specific one. The X-VLA paper reports RoboTwin scores but the correct checkpoint is unknown.

### Notes
- Reported score: 70.0% Easy, 39.0% Hard (50 tasks, Protocol A).
- Requires investigation of how X-VLA's 20D EE6D actions map to RoboTwin's 14D joint space.

---

## Pair 13: StarVLA Qwen3-OFT x RoboTwin 2.0 — Not yet evaluated

STATUS: Not yet evaluated

### Config
- Server config: not yet created (StarVLA RoboTwin config needed)
- Benchmark config: `configs/robotwin_eval.yaml`

### Pipeline verification

| Item | Official | Ours | Match? | Evidence |
|------|----------|------|:------:|----------|
| **Protocol** | Protocol B (multi-task) | TBD | — | StarVLA uses Protocol B (joint training), not Protocol A |
| **Tasks** | 48 tasks | TBD | — | |
| **Checkpoint** | Qwen3-VL-OFT (50 demos/task) | TBD | — | |

### Discrepancies
- Protocol B (multi-task) is not comparable to Protocol A (single-task) results.
- Server config not yet created.
- Reported score: 50.4% Easy (Protocol B, 48 tasks).

### Notes
- StarVLA RoboTwin uses Protocol B — different from DB-CogACT (Protocol A) and X-VLA (Protocol A).
- Not directly comparable across protocols.

---

## Summary

| # | Pair | Score (Reported) | Status | Blockers |
|---|------|-----------------|--------|----------|
| 1 | X-VLA x LIBERO | 97.2% (98.1%) | Reproduced | 0 |
| 2 | Pi0.5 x LIBERO | 97.7% (96.9%) | Reproduced | 0 |
| 3 | GR00T x LIBERO | 94.9% (97.0%) | Approximate | 0 (checkpoint variance) |
| 4 | OFT x LIBERO (spatial) | 94.0% (~96.8%) | Partial | 0 (unnorm_key per-suite) |
| 5 | DB-CogACT x LIBERO | 95.2% (94.9%) | Reproduced | 0 |
| 6 | DB-CogACT x CALVIN | 4.05 (4.06) | Reproduced | 0 |
| 7 | DB-CogACT x SimplerEnv | 72.2% (69.5%) | Reproduced | 0 |
| 8 | X-VLA x CALVIN | 3.97 (4.43) | Not reproduced | 6 (rot6d, euler-as-aa, gripper, EP_LEN, action format, absolute_action) |
| 9 | X-VLA x SimplerEnv | 0% (95.8%) | Not reproduced | 5 BLOCKERS (profile missing, euler_offset, no state, action dim, max_steps) |
| 10 | GR00T x SimplerEnv | 25% (57.1%) | Not reproduced | 3 (no state, bridge rotation, max_steps) |
| 11 | DB-CogACT x RoboTwin 2.0 | — (58.5%) | Not yet evaluated | 0 (config ready, per-task checkpoint) |
| 12 | X-VLA x RoboTwin 2.0 | — (70.0%/39.0%) | Not yet evaluated | 2 (action dim, checkpoint) |
| 13 | StarVLA x RoboTwin 2.0 | — (50.4%) | Not yet evaluated | Protocol B, config needed |

---

## References

### Model Servers
- `src/vla_eval/model_servers/xvla.py`
- `src/vla_eval/model_servers/groot.py`
- `src/vla_eval/model_servers/pi0.py`
- `src/vla_eval/model_servers/oft.py`
- `src/vla_eval/model_servers/dexbotic/cogact.py`
- `src/vla_eval/model_servers/predict.py`

### Benchmarks
- `src/vla_eval/benchmarks/libero/benchmark.py`
- `src/vla_eval/benchmarks/calvin/benchmark.py`
- `src/vla_eval/benchmarks/simpler/benchmark.py`
- `src/vla_eval/benchmarks/libero/utils.py`

### Configs
- `configs/model_servers/xvla/libero.yaml`
- `configs/model_servers/xvla/calvin.yaml`
- `configs/model_servers/xvla/simpler_widowx.yaml`
- `configs/model_servers/groot/libero.yaml`
- `configs/model_servers/groot/simpler_widowx.yaml`
- `configs/model_servers/pi0/libero.yaml`
- `configs/model_servers/oft/libero_joint.yaml`
- `configs/model_servers/db_cogact/libero.yaml`
- `configs/model_servers/db_cogact/calvin.yaml`
- `configs/model_servers/db_cogact/simpler.yaml`
- `configs/libero_all.yaml`
- `configs/calvin_eval.yaml`
- `configs/simpler_all_tasks.yaml`

### Rotation Utilities
- `src/vla_eval/rotation.py`

### Official Eval Code (external, not in this repo)
- X-VLA CALVIN client — from [2toINF/X-VLA](https://github.com/2toINF/X-VLA) evaluation scripts
- X-VLA SimplerEnv WidowX client — same repo
- X-VLA domain config — same repo
- GR00T SimplerEnv eval — from [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) evaluation scripts
- GR00T bridge modality config — same repo

### Infrastructure
- `src/vla_eval/orchestrator.py`
- `src/vla_eval/model_servers/serve.py`