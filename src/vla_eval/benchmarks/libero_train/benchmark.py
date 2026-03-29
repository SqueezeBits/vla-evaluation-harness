"""LIBERO training-environment benchmark for failure trajectory collection.

Spawns the exact same training environments as the official LIBERO dataset,
using init_states from the HDF5 files. This pairs with the noop-filtered
LeRobot training data so that model-generated (failure) trajectories can be
matched 1:1 with the original (success) training episodes.

Requires a precomputed mapping JSON (from ``scripts/compute_libero_mapping.py``)
that links each noop LeRobot episode to its original HDF5 demo index.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.benchmarks.libero.utils import preprocess_libero_image
from vla_eval.rotation import matrix_to_quat, quat_to_axisangle
from vla_eval.types import Action, EpisodeResult, Observation, Task

# EGL for headless rendering
os.environ.setdefault("EGL_PLATFORM", "device")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# Max steps per suite (same as evaluation benchmark)
MAX_STEP_MAPPING = {
    "libero_spatial": 220,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_10": 520,
    "libero_90": 400,
}


class LIBEROTrainBenchmark(StepBenchmark):
    """LIBERO training-environment benchmark for failure trajectory collection.

    Unlike the evaluation benchmark which uses LIBERO's built-in benchmark
    suite for init_states, this benchmark loads init_states directly from
    the official HDF5 dataset files. A precomputed mapping JSON links each
    noop-filtered LeRobot episode to its corresponding HDF5 demo.

    Each task returned by ``get_tasks()`` represents one specific training
    episode. Use ``episodes_per_task: 1`` in the config.

    Args:
        suite: LIBERO suite name (e.g. "libero_spatial", "libero_10").
        mapping_file: Path to the JSON mapping file from compute_libero_mapping.py.
        seed: Random seed for environment initialization.
        num_steps_wait: Dummy action steps at episode start (default 10).
        send_wrist_image: Include wrist camera image in observations.
        send_state: Include proprioceptive state in observations.
        absolute_action: Use absolute (world-frame) actions instead of delta.
        max_steps: Override the default suite-specific max step count.
        max_tasks: Limit the number of tasks (episodes) to run. None = all.
    """

    def __init__(
        self,
        suite: str = "libero_spatial",
        mapping_file: str = "",
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
        absolute_action: bool = False,
        max_steps: int | None = None,
        max_tasks: int | None = None,
    ) -> None:
        super().__init__()
        self.suite = suite
        self.mapping_file = mapping_file
        self.seed = seed
        self.num_steps_wait = num_steps_wait
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self.absolute_action = absolute_action
        self._max_steps = max_steps
        self._max_tasks = max_tasks
        self._env = None
        self._current_bddl: str | None = None
        self._mapping: dict | None = None
        self._init_state_cache: dict[str, dict[int, np.ndarray]] = {}

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        self._init_state_cache.clear()

    def _load_mapping(self) -> dict:
        """Load and cache the episode mapping JSON."""
        if self._mapping is not None:
            return self._mapping
        if not self.mapping_file:
            raise ValueError("mapping_file is required for LIBEROTrainBenchmark")
        with open(self.mapping_file) as f:
            self._mapping = json.load(f)
        return self._mapping

    def _get_init_state(self, hdf5_file: str, demo_idx: int) -> np.ndarray:
        """Load an init_state from HDF5, with caching per file."""
        if hdf5_file not in self._init_state_cache:
            self._init_state_cache[hdf5_file] = {}
        cache = self._init_state_cache[hdf5_file]
        if demo_idx not in cache:
            with h5py.File(hdf5_file, "r") as f:
                cache[demo_idx] = f["data"][f"demo_{demo_idx}"]["states"][0].copy()
        return cache[demo_idx]

    def _get_bddl_file(self, task_name: str) -> str:
        """Resolve the BDDL file for a task by searching LIBERO's task registry."""
        self._init_libero()
        from libero.libero import benchmark as libero_benchmark

        benchmark_dict = libero_benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.suite]()

        for task_id in range(task_suite.n_tasks):
            task_obj = task_suite.get_task(task_id)
            if task_obj.language == task_name:
                from libero.libero import get_libero_path

                return str(Path(get_libero_path("bddl_files")) / task_obj.problem_folder / task_obj.bddl_file)

        raise ValueError(f"Task '{task_name}' not found in LIBERO {self.suite} suite")

    def _init_libero(self) -> None:
        """Lazily patch torch.load for LIBERO compatibility."""
        import functools

        import torch

        if not hasattr(torch.load, "_patched_for_libero"):
            _original_torch_load = torch.load

            @functools.wraps(_original_torch_load)
            def _patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _original_torch_load(*args, **kwargs)

            _patched_load._patched_for_libero = True  # type: ignore[attr-defined]
            torch.load = _patched_load

    def get_tasks(self) -> list[Task]:
        """Return one task per mapped training episode.

        Each task dict includes metadata needed to spawn the exact training
        environment: task name, HDF5 file path, demo index, and BDDL file.
        """
        mapping = self._load_mapping()
        tasks: list[Task] = []

        for task_name, task_info in sorted(mapping["tasks"].items()):
            hdf5_file = task_info["hdf5_file"]
            bddl_file = self._get_bddl_file(task_name)

            for ep in task_info["episodes"]:
                tasks.append(
                    {
                        "name": task_name,
                        "suite": self.suite,
                        "hdf5_file": hdf5_file,
                        "hdf5_demo_idx": ep["hdf5_demo_idx"],
                        "lerobot_episode_idx": ep["lerobot_episode_idx"],
                        "bddl_file": bddl_file,
                        "num_train_actions": ep["num_actions"],
                    }
                )

        if self._max_tasks is not None:
            tasks = tasks[: self._max_tasks]

        return tasks

    def reset(self, task: Task) -> Any:
        from libero.libero.envs import OffScreenRenderEnv

        bddl_file = task["bddl_file"]
        hdf5_file = task["hdf5_file"]
        hdf5_demo_idx = task["hdf5_demo_idx"]

        # Only create a new env when the BDDL (scene) changes
        if self._env is None or self._current_bddl != bddl_file:
            if self._env is not None:
                self._env.close()

            env_args = {
                "bddl_file_name": bddl_file,
                "camera_heights": LIBERO_ENV_RESOLUTION,
                "camera_widths": LIBERO_ENV_RESOLUTION,
            }
            env = OffScreenRenderEnv(**env_args)
            env.seed(self.seed)
            self._env = env
            self._current_bddl = bddl_file

        # Reset env before setting init state
        self._env.reset()

        # Load init_state from HDF5 and set it
        init_state = self._get_init_state(hdf5_file, hdf5_demo_idx)
        obs = self._env.set_init_state(init_state)

        # Run dummy action wait steps
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)

        # Switch to absolute action mode after settling
        if self.absolute_action:
            for robot in self._env.robots:
                robot.controller.use_delta = False

        return obs

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.tolist()
        assert len(raw_action) == 7, f"Action dimension mismatch: got {len(raw_action)}, expected 7"

        # Discretize gripper
        if raw_action[-1] < 0:
            gripper = -1.0
        else:
            gripper = 1.0
        processed_action = raw_action[:-1] + [gripper]

        assert self._env is not None
        obs, reward, done, info = self._env.step(processed_action)
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        img = preprocess_libero_image(raw_obs["agentview_image"], LIBERO_ENV_RESOLUTION)

        obs_dict: dict[str, Any] = {
            "images": {"agentview": img},
            "task_description": task["name"],
        }

        if self.send_wrist_image:
            wrist = preprocess_libero_image(raw_obs["robot0_eye_in_hand_image"], LIBERO_ENV_RESOLUTION)
            obs_dict["images"]["wrist"] = wrist

        if self.send_state:
            obs_dict["states"] = np.concatenate(
                [
                    raw_obs["robot0_eef_pos"],
                    quat_to_axisangle(raw_obs["robot0_eef_quat"]),
                    raw_obs["robot0_gripper_qpos"],
                ]
            )
            assert self._env is not None
            robot = self._env.robots[0]
            ee_pos = np.asarray(robot.controller.ee_pos, dtype=np.float32)
            ee_ori_mat = np.asarray(robot.controller.ee_ori_mat, dtype=np.float32)
            ee_aa = quat_to_axisangle(matrix_to_quat(ee_ori_mat))
            obs_dict["controller_states"] = np.concatenate(
                [ee_pos, ee_aa, np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)]
            )

        return obs_dict

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.done}

    def get_metadata(self) -> dict[str, Any]:
        mapping = self._load_mapping()
        total_episodes = sum(len(t["episodes"]) for t in mapping["tasks"].values())
        return {
            "max_steps": self._max_steps or MAX_STEP_MAPPING.get(self.suite, 300),
            "max_episodes_per_task": 1,
            "suite": self.suite,
            "total_training_episodes": total_episodes,
        }

    def render(self) -> np.ndarray | None:
        try:
            assert self._env is not None
            return self._env.render()
        except Exception:
            return None
