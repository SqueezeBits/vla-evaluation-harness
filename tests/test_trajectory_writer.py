"""Tests for TrajectoryWriter (LeRobot v2.1 format)."""

from __future__ import annotations

import json
import shutil
from unittest.mock import MagicMock

import numpy as np
import pytest

from vla_eval.config import TrajectoryConfig
from vla_eval.results.trajectory_writer import TrajectoryWriter


@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary directory for trajectory output."""
    d = tmp_path / "trajectories"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_obs(
    *,
    cameras: dict[str, tuple[int, int]] | None = None,
    state_dim: int = 7,
) -> dict:
    """Create a synthetic observation dict."""
    if cameras is None:
        cameras = {"agentview": (256, 256), "wristview": (128, 128)}
    images = {name: np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for name, (h, w) in cameras.items()}
    return {"images": images, "states": np.random.randn(state_dim).astype(np.float32)}


def _make_action(action_dim: int = 7) -> dict:
    return {"actions": np.random.randn(action_dim).astype(np.float32)}


# -- Basic lifecycle tests -------------------------------------------------


def test_single_episode_parquet(output_dir):
    """A single episode writes a valid parquet file with correct schema."""
    pytest.importorskip("pyarrow.parquet")
    import pyarrow.parquet as pq

    n_steps = 5
    with TrajectoryWriter(output_dir, fps=10) as writer:
        writer.start_episode("pick up the block")
        for _ in range(n_steps):
            writer.add_step(_make_obs(cameras={}), _make_action())
        writer.end_episode()

    parquet_path = output_dir / "data" / "chunk-000" / "episode_000000.parquet"
    assert parquet_path.exists()

    table = pq.read_table(parquet_path)
    assert len(table) == n_steps

    # Required columns
    for col in ("action", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        assert col in table.column_names

    # Check index values
    assert table.column("frame_index").to_pylist() == list(range(n_steps))
    assert table.column("episode_index").to_pylist() == [0] * n_steps


def test_state_recorded_in_parquet(output_dir):
    """Observation states are written to the parquet file."""
    pytest.importorskip("pyarrow.parquet")
    import pyarrow.parquet as pq

    with TrajectoryWriter(output_dir, fps=10) as writer:
        writer.start_episode("test states")
        for _ in range(3):
            writer.add_step(_make_obs(cameras={}, state_dim=4), _make_action())
        writer.end_episode()

    table = pq.read_table(output_dir / "data" / "chunk-000" / "episode_000000.parquet")
    assert "observation.state" in table.column_names
    states = table.column("observation.state").to_pylist()
    assert len(states[0]) == 4


def test_video_files_created(output_dir):
    """MP4 video files are created for each camera viewpoint."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    cameras = {"agentview": (64, 64), "wristview": (64, 64)}
    with TrajectoryWriter(output_dir, fps=10) as writer:
        writer.start_episode("video test")
        for _ in range(5):
            writer.add_step(_make_obs(cameras=cameras), _make_action())
        writer.end_episode()

    for cam_name in cameras:
        video_path = output_dir / "videos" / "chunk-000" / f"observation.images.{cam_name}" / "episode_000000.mp4"
        assert video_path.exists(), f"Missing video for {cam_name}"
        assert video_path.stat().st_size > 0


def test_multiple_episodes_metadata(output_dir):
    """Multiple episodes produce correct metadata files."""
    with TrajectoryWriter(output_dir, fps=10) as writer:
        for i in range(3):
            task = "task A" if i < 2 else "task B"
            writer.start_episode(task)
            for _ in range(4):
                writer.add_step(_make_obs(cameras={}), _make_action())
            writer.end_episode(metadata={"success": i % 2 == 0})

    # info.json
    info_path = output_dir / "meta" / "info.json"
    assert info_path.exists()
    info = json.loads(info_path.read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 12
    assert info["total_tasks"] == 2
    assert info["fps"] == 10

    # episodes.jsonl
    episodes_path = output_dir / "meta" / "episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    assert len(episodes) == 3
    assert episodes[0]["length"] == 4
    assert episodes[2]["tasks"] == ["task B"]

    # tasks.jsonl
    tasks_path = output_dir / "meta" / "tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    assert len(tasks) == 2
    assert {t["task"] for t in tasks} == {"task A", "task B"}

    # episodes_stats.jsonl
    stats_path = output_dir / "meta" / "episodes_stats.jsonl"
    stats = [json.loads(line) for line in stats_path.read_text().splitlines()]
    assert len(stats) == 3
    assert "action" in stats[0]["stats"]


def test_info_json_features(output_dir):
    """info.json features section describes actions, states, and cameras."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    cameras = {"agentview": (64, 64)}
    with TrajectoryWriter(output_dir, fps=30, robot_type="panda") as writer:
        writer.start_episode("feature test")
        for _ in range(3):
            writer.add_step(_make_obs(cameras=cameras, state_dim=7), _make_action(action_dim=7))
        writer.end_episode()

    info = json.loads((output_dir / "meta" / "info.json").read_text())
    assert info["robot_type"] == "panda"

    assert "action" in info["features"]
    assert info["features"]["action"]["shape"] == [7]

    assert "observation.state" in info["features"]
    assert info["features"]["observation.state"]["shape"] == [7]

    assert "observation.images.agentview" in info["features"]
    vid_feat = info["features"]["observation.images.agentview"]
    assert vid_feat["dtype"] == "video"
    assert vid_feat["shape"] == [64, 64, 3]
    assert vid_feat["info"]["video.fps"] == 30.0


def test_global_index_spans_episodes(output_dir):
    """Global frame index is continuous across episodes."""
    pytest.importorskip("pyarrow.parquet")
    import pyarrow.parquet as pq

    with TrajectoryWriter(output_dir, fps=10) as writer:
        writer.start_episode("ep0")
        for _ in range(3):
            writer.add_step(_make_obs(cameras={}), _make_action())
        writer.end_episode()

        writer.start_episode("ep1")
        for _ in range(2):
            writer.add_step(_make_obs(cameras={}), _make_action())
        writer.end_episode()

    t0 = pq.read_table(output_dir / "data" / "chunk-000" / "episode_000000.parquet")
    t1 = pq.read_table(output_dir / "data" / "chunk-000" / "episode_000001.parquet")

    assert t0.column("index").to_pylist() == [0, 1, 2]
    assert t1.column("index").to_pylist() == [3, 4]


def test_timestamps(output_dir):
    """Timestamps are correctly spaced according to fps."""
    pytest.importorskip("pyarrow.parquet")
    import pyarrow.parquet as pq

    with TrajectoryWriter(output_dir, fps=20) as writer:
        writer.start_episode("ts test")
        for _ in range(4):
            writer.add_step(_make_obs(cameras={}), _make_action())
        writer.end_episode()

    table = pq.read_table(output_dir / "data" / "chunk-000" / "episode_000000.parquet")
    timestamps = table.column("timestamp").to_pylist()
    assert timestamps == pytest.approx([0.0, 0.05, 0.10, 0.15])


def test_context_manager_closes(output_dir):
    """Using as a context manager writes metadata on exit."""
    with TrajectoryWriter(output_dir, fps=10) as writer:
        writer.start_episode("ctx test")
        for _ in range(2):
            writer.add_step(_make_obs(cameras={}), _make_action())
        writer.end_episode()

    assert (output_dir / "meta" / "info.json").exists()
    assert (output_dir / "meta" / "episodes.jsonl").exists()


# -- Error handling --------------------------------------------------------


def test_add_step_without_episode_raises(output_dir):
    writer = TrajectoryWriter(output_dir, fps=10)
    with pytest.raises(RuntimeError, match="No active episode"):
        writer.add_step(_make_obs(), _make_action())


def test_start_episode_twice_raises(output_dir):
    writer = TrajectoryWriter(output_dir, fps=10)
    writer.start_episode("first")
    with pytest.raises(RuntimeError, match="Previous episode not ended"):
        writer.start_episode("second")


def test_end_episode_without_start_raises(output_dir):
    writer = TrajectoryWriter(output_dir, fps=10)
    with pytest.raises(RuntimeError, match="No active episode"):
        writer.end_episode()


def test_empty_episode_skipped(output_dir):
    """An episode with no steps is silently skipped."""
    with TrajectoryWriter(output_dir, fps=10) as writer:
        writer.start_episode("empty ep")
        writer.end_episode()

        writer.start_episode("real ep")
        for _ in range(2):
            writer.add_step(_make_obs(cameras={}), _make_action())
        writer.end_episode()

    info = json.loads((output_dir / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1


def test_chunk_boundary(output_dir):
    """Episodes beyond chunks_size go into the next chunk directory."""
    pytest.importorskip("pyarrow.parquet")

    with TrajectoryWriter(output_dir, fps=10, chunks_size=2) as writer:
        for i in range(3):
            writer.start_episode(f"task {i}")
            writer.add_step(_make_obs(cameras={}), _make_action())
            writer.end_episode()

    # Episode 0, 1 in chunk-000; episode 2 in chunk-001
    assert (output_dir / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert (output_dir / "data" / "chunk-000" / "episode_000001.parquet").exists()
    assert (output_dir / "data" / "chunk-001" / "episode_000002.parquet").exists()


# -- JSON serializability ---------------------------------------------------


def _check_json_native(obj, path="root"):
    """Recursively assert all values are native Python types (no numpy)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), f"Non-string key at {path}: {type(k)}"
            _check_json_native(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_json_native(v, f"{path}[{i}]")
    else:
        assert isinstance(obj, (str, int, float, bool, type(None))), (
            f"Non-JSON-native type at {path}: {type(obj).__name__} = {obj!r}"
        )


def test_all_json_output_is_natively_serializable(output_dir):
    """All JSON/JSONL files must contain only native Python types (no numpy).

    This includes camera info in info.json (whose shape values originate from
    numpy ndarray.shape) and user-supplied metadata that may contain numpy scalars.
    """
    with TrajectoryWriter(output_dir, fps=10, robot_type="panda") as writer:
        # Manually register camera info to exercise the video-feature branch in info.json
        # without requiring ffmpeg. Values use numpy ints to mimic ndarray.shape.
        writer._camera_info["agentview"] = {"height": np.int64(64), "width": np.int64(64), "channels": np.int64(3)}

        for i in range(2):
            writer.start_episode(f"task {i}")
            for _ in range(3):
                writer.add_step(_make_obs(cameras={}, state_dim=7), _make_action(action_dim=7))
            writer.end_episode(metadata={"success": True, "reward": np.float32(1.0)})

    # Check info.json
    info = json.loads((output_dir / "meta" / "info.json").read_text())
    _check_json_native(info, "info.json")

    # Check episodes.jsonl
    for line in (output_dir / "meta" / "episodes.jsonl").read_text().splitlines():
        record = json.loads(line)
        _check_json_native(record, "episodes.jsonl")

    # Check tasks.jsonl
    for line in (output_dir / "meta" / "tasks.jsonl").read_text().splitlines():
        record = json.loads(line)
        _check_json_native(record, "tasks.jsonl")

    # Check episodes_stats.jsonl
    for line in (output_dir / "meta" / "episodes_stats.jsonl").read_text().splitlines():
        record = json.loads(line)
        _check_json_native(record, "episodes_stats.jsonl")


# -- Config tests ----------------------------------------------------------


def test_trajectory_config_default():
    """Absent trajectory section yields disabled config."""
    cfg = TrajectoryConfig.from_dict(None)
    assert cfg.enabled is False


def test_trajectory_config_present_defaults_disabled():
    """Presence of trajectory section alone does not enable recording."""
    cfg = TrajectoryConfig.from_dict({"output_dir": "/tmp/traj"})
    assert cfg.enabled is False
    assert cfg.output_dir == "/tmp/traj"
    assert cfg.fps == 10


def test_trajectory_config_explicit_enable():
    """Setting enabled: true enables recording."""
    cfg = TrajectoryConfig.from_dict({"enabled": True, "output_dir": "/tmp/traj"})
    assert cfg.enabled is True


def test_trajectory_config_explicit_disable():
    """Setting enabled: false disables even when section is present."""
    cfg = TrajectoryConfig.from_dict({"enabled": False, "output_dir": "/tmp/traj"})
    assert cfg.enabled is False


def test_trajectory_config_all_fields():
    """All config fields are parsed correctly."""
    cfg = TrajectoryConfig.from_dict(
        {
            "output_dir": "/data/out",
            "fps": 30,
            "video_codec": "libx264",
            "robot_type": "panda",
            "image_keys": ["agentview", "wristview"],
            "chunks_size": 500,
        }
    )
    assert cfg.fps == 30
    assert cfg.video_codec == "libx264"
    assert cfg.robot_type == "panda"
    assert cfg.image_keys == ["agentview", "wristview"]
    assert cfg.chunks_size == 500


# -- Runner integration tests ----------------------------------------------


@pytest.mark.anyio
async def test_sync_runner_calls_trajectory_writer():
    """SyncEpisodeRunner calls trajectory writer methods in order."""
    pytest.importorskip("pyarrow.parquet")

    from tests.conftest import StubBenchmark, start_echo_server, stop_server
    from vla_eval.connection import Connection
    from vla_eval.runners.sync_runner import SyncEpisodeRunner

    port = _find_free_port()
    server_task = await start_echo_server(port)

    try:
        benchmark = StubBenchmark(done_at_step=3)
        runner = SyncEpisodeRunner()
        task = {"name": "test_task"}

        writer = MagicMock()
        async with Connection(f"ws://127.0.0.1:{port}") as conn:
            await runner.run_episode(benchmark, task, conn, max_steps=50, trajectory_writer=writer)

        writer.start_episode.assert_called_once_with(task_description="test_task")
        assert writer.add_step.call_count == 3  # done_at_step=3
        writer.end_episode.assert_called_once()
        # Check metadata contains success
        end_meta = writer.end_episode.call_args[1]["metadata"]
        assert end_meta["success"] is True
    finally:
        await stop_server(server_task)


@pytest.mark.anyio
async def test_sync_runner_no_writer():
    """SyncEpisodeRunner works normally when trajectory_writer is None."""
    from tests.conftest import StubBenchmark, start_echo_server, stop_server
    from vla_eval.connection import Connection
    from vla_eval.runners.sync_runner import SyncEpisodeRunner

    port = _find_free_port()
    server_task = await start_echo_server(port)

    try:
        benchmark = StubBenchmark(done_at_step=3)
        runner = SyncEpisodeRunner()
        task = {"name": "test_task"}

        async with Connection(f"ws://127.0.0.1:{port}") as conn:
            result = await runner.run_episode(benchmark, task, conn, max_steps=50, trajectory_writer=None)

        assert result["metrics"]["success"] is True
        assert result["steps"] == 3
    finally:
        await stop_server(server_task)


@pytest.mark.anyio
async def test_sync_runner_writer_error_does_not_crash():
    """If trajectory writer raises, the episode still completes."""
    from tests.conftest import StubBenchmark, start_echo_server, stop_server
    from vla_eval.connection import Connection
    from vla_eval.runners.sync_runner import SyncEpisodeRunner

    port = _find_free_port()
    server_task = await start_echo_server(port)

    try:
        benchmark = StubBenchmark(done_at_step=3)
        runner = SyncEpisodeRunner()
        task = {"name": "test_task"}

        writer = MagicMock()
        writer.add_step.side_effect = OSError("disk full")

        async with Connection(f"ws://127.0.0.1:{port}") as conn:
            result = await runner.run_episode(benchmark, task, conn, max_steps=50, trajectory_writer=writer)

        # Episode should still succeed despite writer failure
        assert result["metrics"]["success"] is True
        assert result["steps"] == 3
    finally:
        await stop_server(server_task)


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
