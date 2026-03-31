"""End-to-end comparison: run 1 episode using official eval logic with our model server via WebSocket.

Usage: Start model server first, then run inside SimplerEnv Docker:
  docker run --rm --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
    -v $(pwd)/src:/workspace/src \
    -v $(pwd)/experiments:/workspace/experiments \
    --entrypoint conda \
    ghcr.io/allenai/vla-evaluation-harness/simpler:latest \
    run -n simpler python3 /workspace/experiments/debug_simpler_xvla_e2e.py --server ws://localhost:8003
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import numpy as np

sys.path.insert(0, "/workspace/src")

from vla_eval.connection import Connection
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import simpler_env


async def run_episode(server_url: str):
    task = "widowx_spoon_on_towel"

    # Create env (official way)
    env = simpler_env.make(task)
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": 0}})
    instruction = env.get_language_instruction()
    print(f"Task: {task}, Instruction: {instruction}")

    # Extract initial image
    image = get_image_from_maniskill2_obs_dict(env, obs)
    print(f"Image shape: {image.shape}, mean: {image.mean():.1f}")

    # Compute initial base-relative EE pose (like official eval)
    bp = obs["agent"]["base_pose"]
    tp = obs["extra"]["tcp_pose"]
    print(f"base_pose: {bp}")
    print(f"tcp_pose: {tp}")

    # Connect to model server
    conn = Connection(server_url)
    await conn.connect(benchmark="debug")
    print(f"Connected. Server info: {conn.server_info.get('observation_params', {})}")

    # Build observation dict (like our benchmark sends)
    obs_dict = {
        "images": {"primary": image},
        "task_description": instruction,
        "base_pose": np.asarray(bp, dtype=np.float32),
        "tcp_pose": np.asarray(tp, dtype=np.float32),
    }

    # Start episode
    await conn.start_episode({"name": task})

    # Run 20 steps and print actions
    for step in range(20):
        action_resp = await conn.act(obs_dict)
        actions = np.asarray(action_resp.get("actions", []))
        if step < 5:
            a = actions[0] if actions.ndim > 1 else actions
            print(f"  step {step}: pos={a[:3].round(4)} rot={a[3:6].round(4)} grip={a[6]:.2f}")

        # Step env
        env_action = actions[0] if actions.ndim > 1 else actions
        obs, reward, done, truncated, info = env.step(env_action)
        image = get_image_from_maniskill2_obs_dict(env, obs)

        # Update obs for next step
        obs_dict = {
            "images": {"primary": image},
            "task_description": instruction,
            "base_pose": np.asarray(obs["agent"]["base_pose"], dtype=np.float32),
            "tcp_pose": np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32),
        }

        if done:
            print(f"SUCCESS at step {step}!")
            break

    print(f"Episode done: done={done}, steps={step + 1}")
    await conn.close()
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="ws://localhost:8003")
    args = parser.parse_args()
    asyncio.run(run_episode(args.server))


if __name__ == "__main__":
    main()
