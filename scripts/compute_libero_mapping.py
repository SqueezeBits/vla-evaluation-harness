#!/usr/bin/env python3
"""Compute episode mapping between LeRobot noop-filtered datasets and official LIBERO HDF5.

For each suite (spatial, object, goal, 10), this script:
1. Reads the LeRobot noop dataset to get per-episode action sequences grouped by task.
2. Reads the official LIBERO HDF5 demos, applies the same noop filter, and collects
   filtered action sequences for each of the 50 demos per task.
3. Matches LeRobot episodes to HDF5 demos using action-sequence similarity.
4. Outputs a JSON mapping file per suite.

Usage:
    uv run python scripts/compute_libero_mapping.py \
        --official-dir /path/to/libero_official \
        --noop-dir /path/to/data \
        --output-dir /path/to/output

Output JSON structure (per suite):
    {
        "suite": "libero_spatial",
        "tasks": {
            "<task_name>": {
                "hdf5_file": "<path>",
                "episodes": [
                    {"lerobot_episode_idx": 5, "hdf5_demo_idx": 0, "num_actions": 120},
                    ...
                ]
            }
        }
    }
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# Scene prefix pattern for libero_10/90 HDF5 filenames
SCENE_PREFIX_RE = re.compile(r"^[A-Z]+(?:_[A-Z]+)*_SCENE\d+_")


def is_noop(action: np.ndarray, prev_action: np.ndarray | None = None, threshold: float = 1e-4) -> bool:
    """Check if an action is a no-op (matches the official LIBERO preprocessing logic)."""
    if prev_action is None:
        return float(np.linalg.norm(action[:-1])) < threshold
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return float(np.linalg.norm(action[:-1])) < threshold and gripper_action == prev_gripper_action


def filter_noop_actions(actions: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Apply noop filtering to a sequence of actions.

    Returns the filtered actions and the original indices that were kept.
    """
    filtered = []
    kept_indices = []
    prev_action = None
    for i, action in enumerate(actions):
        if is_noop(action, prev_action):
            continue
        filtered.append(action)
        kept_indices.append(i)
        prev_action = action
    if len(filtered) == 0:
        return np.empty((0, actions.shape[1]), dtype=actions.dtype), []
    return np.stack(filtered), kept_indices


def hdf5_filename_to_task_name(filename: str, suite: str) -> str:
    """Convert HDF5 filename to task description string."""
    name = filename.replace("_demo.hdf5", "")
    # Strip scene prefix for libero_10/90
    if suite in ("libero_10", "libero_90"):
        name = SCENE_PREFIX_RE.sub("", name)
    return name.replace("_", " ")


def load_lerobot_episodes(noop_dataset_dir: Path) -> dict[str, list[dict]]:
    """Load LeRobot noop dataset episodes grouped by task name.

    Returns:
        {task_name: [{"episode_idx": int, "actions": np.ndarray}, ...]}
        sorted by episode_idx within each task.
    """
    meta_dir = noop_dataset_dir / "meta"
    data_dir = noop_dataset_dir / "data"

    # Load task mapping
    tasks_map: dict[int, str] = {}
    with open(meta_dir / "tasks.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            tasks_map[entry["task_index"]] = entry["task"]

    # Load episode metadata
    episodes_meta: list[dict] = []
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            episodes_meta.append(json.loads(line))

    # Group episodes by task
    task_episodes: dict[str, list[dict]] = {}
    for ep_meta in episodes_meta:
        ep_idx = ep_meta["episode_index"]
        # episodes.jsonl has "tasks" (list of task descriptions) or "task_index"
        if "task_index" in ep_meta:
            task_name = tasks_map[ep_meta["task_index"]]
        elif "tasks" in ep_meta:
            task_name = ep_meta["tasks"][0] if isinstance(ep_meta["tasks"], list) else ep_meta["tasks"]
        else:
            raise ValueError(f"Cannot determine task for episode {ep_idx}")

        # Read parquet for this episode
        chunk_idx = ep_idx // 1000
        parquet_path = data_dir / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        table = pq.read_table(parquet_path, columns=["action"])
        actions = np.array(table.column("action").to_pylist(), dtype=np.float64)

        if task_name not in task_episodes:
            task_episodes[task_name] = []
        task_episodes[task_name].append({"episode_idx": ep_idx, "actions": actions})

    # Sort by episode_idx within each task
    for task_name in task_episodes:
        task_episodes[task_name].sort(key=lambda x: x["episode_idx"])

    return task_episodes


def load_hdf5_demos(hdf5_path: Path) -> list[dict]:
    """Load all demos from an HDF5 file, apply noop filtering.

    Returns list of {"demo_idx": int, "filtered_actions": np.ndarray, "init_state": np.ndarray}
    """
    demos = []
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        num_demos = len([k for k in data.keys() if k.startswith("demo_")])
        for i in range(num_demos):
            demo = data[f"demo_{i}"]
            actions = demo["actions"][()]
            init_state = demo["states"][0]
            filtered_actions, _ = filter_noop_actions(actions)
            demos.append(
                {
                    "demo_idx": i,
                    "filtered_actions": filtered_actions,
                    "init_state": init_state,
                    "num_original_actions": len(actions),
                }
            )
    return demos


def match_episodes(
    lerobot_episodes: list[dict],
    hdf5_demos: list[dict],
    task_name: str,
) -> list[dict]:
    """Match LeRobot episodes to HDF5 demos by action sequence similarity.

    Strategy:
    1. First try sequential matching: HDF5 demos with >0 filtered actions, in order,
       should correspond to LeRobot episodes in order (since preprocessing is sequential).
    2. Verify by comparing action sequences (first few actions + lengths).
    3. Fall back to brute-force matching if sequential doesn't work.
    """
    # Get HDF5 demos that survive filtering (non-empty after noop removal)
    surviving_demos = [d for d in hdf5_demos if len(d["filtered_actions"]) > 0]

    if len(surviving_demos) != len(lerobot_episodes):
        print(
            f"  WARNING [{task_name}]: {len(surviving_demos)} surviving HDF5 demos "
            f"vs {len(lerobot_episodes)} LeRobot episodes. Using brute-force matching."
        )
        return _brute_force_match(lerobot_episodes, hdf5_demos, task_name)

    # Try sequential matching with verification
    matches = []
    all_verified = True
    for lr_ep, hdf5_demo in zip(lerobot_episodes, surviving_demos):
        lr_actions = lr_ep["actions"]
        hdf5_actions = hdf5_demo["filtered_actions"]

        # Verify: lengths must match and first few actions must be close
        if len(lr_actions) != len(hdf5_actions):
            all_verified = False
            break

        # Compare first 5 actions (allowing small float tolerance)
        n_compare = min(5, len(lr_actions))
        if not np.allclose(lr_actions[:n_compare], hdf5_actions[:n_compare], atol=1e-3):
            all_verified = False
            break

        matches.append(
            {
                "lerobot_episode_idx": lr_ep["episode_idx"],
                "hdf5_demo_idx": hdf5_demo["demo_idx"],
                "num_actions": len(lr_actions),
            }
        )

    if all_verified:
        return matches

    print(f"  WARNING [{task_name}]: Sequential matching failed verification. Using brute-force.")
    return _brute_force_match(lerobot_episodes, hdf5_demos, task_name)


def _brute_force_match(
    lerobot_episodes: list[dict],
    hdf5_demos: list[dict],
    task_name: str,
) -> list[dict]:
    """Brute-force match by finding the closest HDF5 demo for each LeRobot episode."""
    matches = []
    used_demos: set[int] = set()

    for lr_ep in lerobot_episodes:
        lr_actions = lr_ep["actions"]
        best_demo_idx = -1
        best_score = float("inf")

        for demo in hdf5_demos:
            if demo["demo_idx"] in used_demos:
                continue
            hdf5_actions = demo["filtered_actions"]
            if len(hdf5_actions) == 0:
                continue

            # Length difference as primary score
            len_diff = abs(len(lr_actions) - len(hdf5_actions))
            if len_diff > 5:  # Allow small tolerance for edge cases
                continue

            # Compare overlapping action values
            n_compare = min(len(lr_actions), len(hdf5_actions), 10)
            action_diff = float(np.mean(np.abs(lr_actions[:n_compare] - hdf5_actions[:n_compare])))
            score = len_diff * 100 + action_diff

            if score < best_score:
                best_score = score
                best_demo_idx = demo["demo_idx"]

        if best_demo_idx == -1:
            print(f"    ERROR [{task_name}]: No match for LeRobot episode {lr_ep['episode_idx']}")
            continue

        used_demos.add(best_demo_idx)
        matches.append(
            {
                "lerobot_episode_idx": lr_ep["episode_idx"],
                "hdf5_demo_idx": best_demo_idx,
                "num_actions": len(lr_actions),
            }
        )

    return matches


def compute_mapping_for_suite(
    suite: str,
    official_dir: Path,
    noop_dataset_dir: Path,
) -> dict:
    """Compute the full mapping for one LIBERO suite."""
    print(f"\n{'=' * 60}")
    print(f"Processing suite: {suite}")
    print(f"{'=' * 60}")

    # Load LeRobot episodes
    print(f"  Loading LeRobot noop dataset from {noop_dataset_dir}...")
    lerobot_episodes = load_lerobot_episodes(noop_dataset_dir)
    print(f"  Found {sum(len(v) for v in lerobot_episodes.values())} episodes across {len(lerobot_episodes)} tasks")

    # Map HDF5 files to task names
    hdf5_dir = official_dir / suite
    hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))
    hdf5_task_map: dict[str, Path] = {}
    for hdf5_file in hdf5_files:
        task_name = hdf5_filename_to_task_name(hdf5_file.name, suite)
        hdf5_task_map[task_name] = hdf5_file

    print(f"  Found {len(hdf5_files)} HDF5 files")

    # Match tasks between LeRobot and HDF5
    result: dict[str, dict] = {}
    total_matched = 0
    total_episodes = 0

    for task_name, lr_eps in sorted(lerobot_episodes.items()):
        total_episodes += len(lr_eps)

        if task_name not in hdf5_task_map:
            print(f"  WARNING: Task '{task_name}' not found in HDF5 files")
            # Try fuzzy match
            for hdf5_name in hdf5_task_map:
                if task_name.lower() == hdf5_name.lower():
                    print(f"    -> Fuzzy matched to '{hdf5_name}'")
                    task_name_key = hdf5_name
                    break
            else:
                print("    -> SKIPPING (no match found)")
                continue
        else:
            task_name_key = task_name

        hdf5_path = hdf5_task_map[task_name_key]
        print(f"  Processing: {task_name} ({len(lr_eps)} episodes)")

        # Load and filter HDF5 demos
        hdf5_demos = load_hdf5_demos(hdf5_path)

        # Match
        matches = match_episodes(lr_eps, hdf5_demos, task_name)
        total_matched += len(matches)

        result[task_name] = {
            "hdf5_file": str(hdf5_path),
            "episodes": matches,
        }

    print(f"\n  Summary: {total_matched}/{total_episodes} episodes matched")
    return {"suite": suite, "tasks": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LIBERO episode mapping (noop ↔ official)")
    parser.add_argument(
        "--official-dir",
        type=Path,
        default=Path("/home/data1/junhalee/data/libero_official"),
        help="Path to official LIBERO HDF5 dataset directory",
    )
    parser.add_argument(
        "--noop-dir",
        type=Path,
        default=Path("/home/data1/junhalee/data"),
        help="Parent directory containing libero_*_no_noops_* datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/data1/junhalee/data/libero_mappings"),
        help="Output directory for mapping JSON files",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=SUITES,
        choices=SUITES,
        help="Suites to process",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover noop dataset directories
    suite_to_noop_dir: dict[str, Path] = {}
    for suite in args.suites:
        # Extract the subset name from suite (e.g., "libero_spatial" -> "spatial")
        subset = suite.replace("libero_", "")
        # Find matching noop directory
        candidates = list(args.noop_dir.glob(f"libero_{subset}_no_noop*"))
        if not candidates:
            print(f"WARNING: No noop dataset found for {suite} in {args.noop_dir}")
            continue
        suite_to_noop_dir[suite] = candidates[0]
        print(f"Found noop dataset for {suite}: {candidates[0].name}")

    # Process each suite
    for suite in args.suites:
        if suite not in suite_to_noop_dir:
            continue

        mapping = compute_mapping_for_suite(suite, args.official_dir, suite_to_noop_dir[suite])

        output_path = args.output_dir / f"{suite}_mapping.json"
        with open(output_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"\n  Saved mapping to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
