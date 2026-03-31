"""Debug script: compare official vs our SimplerEnv setup for X-VLA.

Run inside the SimplerEnv Docker container with --dev mode:
  docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
    -v $(pwd)/src:/workspace/src \
    -v $(pwd)/experiments:/workspace/experiments \
    --entrypoint conda \
    ghcr.io/allenai/vla-evaluation-harness/simpler:latest \
    run -n simpler python3 /workspace/experiments/debug_simpler_xvla.py
"""

from __future__ import annotations

import numpy as np
import simpler_env
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def make_env_official(task: str, episode_id: int = 0):
    """Create env the way the official X-VLA eval does."""
    env = simpler_env.make(task)
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": episode_id}})
    return env, obs


def make_env_ours(env_name: str, episode_id: int = 0):
    """Create env the way our benchmark does."""
    from transforms3d.euler import euler2quat
    from sapien.core import Pose

    control_mode = get_robot_control_mode("widowx", "vla")
    env = build_maniskill2_env(
        env_name,
        obs_mode="rgbd",
        robot="widowx",
        scene_name="bridge_table_1_v1",
        control_freq=5,
        sim_freq=500,
        max_episode_steps=1200,
        control_mode=control_mode,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path="/app/simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
    )

    # Robot init (matches our benchmark config)
    r, p, y = 0.0, 0.0, 0.0
    rpy_quat = euler2quat(r, p, y)
    center_quat = [0, 0, 0, 1]
    init_quat = (Pose(q=rpy_quat) * Pose(q=center_quat)).q

    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([0.147, 0.028]),
            "init_rot_quat": init_quat,
        },
        "obj_init_options": {"episode_id": episode_id},
    }
    obs, _ = env.reset(options=env_reset_options)
    return env, obs


def extract_state(obs):
    """Extract state info from obs."""
    agent = obs.get("agent", {})
    extra = obs.get("extra", {})
    base_pose = agent.get("base_pose")
    tcp_pose = extra.get("tcp_pose")
    qpos = agent.get("qpos")
    return {
        "base_pose": np.asarray(base_pose) if base_pose is not None else None,
        "tcp_pose": np.asarray(tcp_pose) if tcp_pose is not None else None,
        "qpos": np.asarray(qpos) if qpos is not None else None,
    }


def compare_envs():
    task = "widowx_spoon_on_towel"
    env_name = "PutSpoonOnTableClothInScene-v0"

    print("=" * 60)
    print("Creating OFFICIAL env...")
    env1, obs1 = make_env_official(task, episode_id=0)
    state1 = extract_state(obs1)
    img1 = get_image_from_maniskill2_obs_dict(env1, obs1)

    print("\nCreating OUR env...")
    env2, obs2 = make_env_ours(env_name, episode_id=0)
    state2 = extract_state(obs2)
    img2 = get_image_from_maniskill2_obs_dict(env2, obs2)

    print("\n" + "=" * 60)
    print("INITIAL STATE COMPARISON")
    print("=" * 60)

    print(f"\nbase_pose (official): {state1['base_pose']}")
    print(f"base_pose (ours):     {state2['base_pose']}")
    if state1["base_pose"] is not None and state2["base_pose"] is not None:
        print(f"  diff: {np.abs(state1['base_pose'] - state2['base_pose']).max():.6f}")

    print(f"\ntcp_pose (official): {state1['tcp_pose']}")
    print(f"tcp_pose (ours):     {state2['tcp_pose']}")
    if state1["tcp_pose"] is not None and state2["tcp_pose"] is not None:
        print(f"  diff: {np.abs(state1['tcp_pose'] - state2['tcp_pose']).max():.6f}")

    print(f"\nqpos (official): {state1['qpos']}")
    print(f"qpos (ours):     {state2['qpos']}")
    if state1["qpos"] is not None and state2["qpos"] is not None:
        print(f"  diff: {np.abs(state1['qpos'] - state2['qpos']).max():.6f}")

    print(f"\nimage shape (official): {img1.shape}, mean: {img1.mean():.1f}")
    print(f"image shape (ours):     {img2.shape}, mean: {img2.mean():.1f}")
    print(f"  pixel diff mean: {np.abs(img1.astype(float) - img2.astype(float)).mean():.1f}")

    # Test stepping with zero action
    print("\n" + "=" * 60)
    print("ZERO ACTION TEST (5 steps)")
    print("=" * 60)
    zero_action = np.zeros(7, dtype=np.float32)
    for step in range(5):
        obs1, r1, d1, t1, i1 = env1.step(zero_action)
        obs2, r2, d2, t2, i2 = env2.step(zero_action)
        s1, s2 = extract_state(obs1), extract_state(obs2)
        tcp_diff = np.abs(s1["tcp_pose"] - s2["tcp_pose"]).max() if s1["tcp_pose"] is not None else -1
        print(f"  step {step}: tcp_diff={tcp_diff:.6f} done1={d1} done2={d2}")

    # Test stepping with a typical X-VLA action
    print("\n" + "=" * 60)
    print("TYPICAL ACTION TEST")
    print("=" * 60)
    # Typical action: small pos delta + rotation with euler_offset
    typical_action = np.array([0.05, -0.01, 0.02, -0.1, 1.41, -0.08, 1.0], dtype=np.float32)
    print(f"Action: {typical_action}")
    obs1t, r1t, d1t, _, _ = env1.step(typical_action)
    obs2t, r2t, d2t, _, _ = env2.step(typical_action)
    s1t, s2t = extract_state(obs1t), extract_state(obs2t)
    print(f"  tcp_pose (official): {s1t['tcp_pose']}")
    print(f"  tcp_pose (ours):     {s2t['tcp_pose']}")
    if s1t["tcp_pose"] is not None and s2t["tcp_pose"] is not None:
        print(f"  diff: {np.abs(s1t['tcp_pose'] - s2t['tcp_pose']).max():.6f}")

    print(f"\n  action_space (official): {env1.action_space}")
    print(f"  action_space (ours):     {env2.action_space}")
    print(f"  max_steps (official): {getattr(env1, '_max_episode_steps', 'N/A')}")
    print(f"  max_steps (ours):     {getattr(env2, '_max_episode_steps', 'N/A')}")

    env1.close()
    env2.close()
    print("\nDone.")


if __name__ == "__main__":
    compare_envs()
