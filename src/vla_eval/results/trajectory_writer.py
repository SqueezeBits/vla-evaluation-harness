"""TrajectoryWriter: stores rollout data in HuggingFace LeRobot v2.1 format.

Produces a dataset with:
- Per-episode parquet files (actions, states, timestamps, indices)
- Per-episode MP4 videos for each camera viewpoint
- Metadata files: info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl

Usage::

    writer = TrajectoryWriter(output_dir="./trajectories", fps=10)

    writer.start_episode(task_description="pick up the red block")
    for obs, action in rollout:
        writer.add_step(observation=obs, action=action)
    writer.end_episode(metadata={"success": True})

    writer.close()  # finalizes metadata
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CODEBASE_VERSION = "v2.1"
_DEFAULT_CHUNKS_SIZE = 1000

# Video encoding defaults (LeRobot convention)
_DEFAULT_VIDEO_CODEC = "libsvtav1"
_FALLBACK_VIDEO_CODEC = "libx264"
_DEFAULT_PIX_FMT = "yuv420p"


def _codec_name_for_metadata(codec: str) -> str:
    """Map ffmpeg encoder name to the short codec name stored in metadata."""
    return {"libsvtav1": "av1", "libx264": "h264", "libx265": "h265"}.get(codec, codec)


def _check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    import shutil

    return shutil.which("ffmpeg") is not None


def _check_ffmpeg_codec(codec: str) -> str:
    """Return *codec* if ffmpeg supports it, otherwise fall back to libx264."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if codec in result.stdout:
            return codec
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if codec != _FALLBACK_VIDEO_CODEC:
        logger.warning("ffmpeg encoder %s unavailable, falling back to %s", codec, _FALLBACK_VIDEO_CODEC)
    return _FALLBACK_VIDEO_CODEC


def _encode_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int,
    codec: str,
    pix_fmt: str = _DEFAULT_PIX_FMT,
) -> None:
    """Encode a list of uint8 HWC frames to an MP4 file using ffmpeg."""
    if not frames:
        return
    h, w, _c = frames[0].shape
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:",
        "-vcodec",
        codec,
        "-pix_fmt",
        pix_fmt,
    ]
    # Codec-specific flags
    if codec == "libsvtav1":
        cmd += ["-crf", "30", "-g", "2", "-preset", "12"]
    elif codec == "libx264":
        cmd += ["-crf", "18", "-g", "2", "-preset", "fast"]

    cmd.append(str(output_path))

    raw = b"".join(f.tobytes() for f in frames)
    proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode(errors='replace')}")


class _EpisodeBuffer:
    """Accumulates data for a single episode before flushing to disk."""

    __slots__ = (
        "episode_index",
        "task_description",
        "task_index",
        "image_buffers",
        "actions",
        "states",
        "metadata",
    )

    def __init__(self, episode_index: int, task_description: str, task_index: int) -> None:
        self.episode_index = episode_index
        self.task_description = task_description
        self.task_index = task_index
        self.image_buffers: dict[str, list[np.ndarray]] = {}
        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray | None] = []
        self.metadata: dict[str, Any] = {}


class TrajectoryWriter:
    """Writes rollout trajectories in HuggingFace LeRobot v2.1 dataset format.

    Parameters
    ----------
    output_dir:
        Root directory for the dataset.
    fps:
        Frame rate of the dataset (controls video FPS and timestamp spacing).
    chunks_size:
        Number of episodes per chunk directory (default 1000).
    video_codec:
        FFmpeg video encoder name. Falls back to libx264 if unavailable.
    robot_type:
        Optional robot identifier stored in metadata.
    image_keys:
        If provided, only these camera names are recorded. ``None`` means
        auto-detect from the first observation.
    state_key:
        Key in the observation dict for the proprioceptive state vector.
        Defaults to ``"states"``; also checks ``"state"`` as fallback.
    """

    def __init__(
        self,
        output_dir: str | Path,
        fps: int = 10,
        chunks_size: int = _DEFAULT_CHUNKS_SIZE,
        video_codec: str = _DEFAULT_VIDEO_CODEC,
        robot_type: str | None = None,
        image_keys: list[str] | None = None,
        state_key: str = "states",
    ) -> None:
        self._root = Path(output_dir)
        self._fps = fps
        self._chunks_size = chunks_size
        self._has_ffmpeg = _check_ffmpeg()
        self._video_codec = _check_ffmpeg_codec(video_codec) if self._has_ffmpeg else video_codec
        self._robot_type = robot_type
        self._image_keys = image_keys
        self._state_key = state_key

        self._root.mkdir(parents=True, exist_ok=True)

        # Bookkeeping
        self._current: _EpisodeBuffer | None = None
        self._global_frame_index = 0
        self._next_episode_index = 0
        self._tasks: dict[str, int] = {}  # task_description -> task_index

        # Accumulated episode metadata (written at close)
        self._episode_records: list[dict[str, Any]] = []
        self._episode_stats: list[dict[str, Any]] = []

        # Schema info (populated from first episode)
        self._action_dim: int | None = None
        self._state_dim: int | None = None
        self._camera_info: dict[str, dict[str, Any]] = {}  # cam_name -> {height, width, channels}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_episode(self, task_description: str, episode_index: int | None = None) -> int:
        """Begin recording a new episode.

        Parameters
        ----------
        task_description:
            Natural-language task instruction for this episode.
        episode_index:
            Explicit episode index. If ``None``, auto-increments.

        Returns
        -------
        int
            The episode index assigned to this episode.
        """
        if self._current is not None:
            raise RuntimeError("Previous episode not ended. Call end_episode() first.")

        ep_idx = episode_index if episode_index is not None else self._next_episode_index
        self._next_episode_index = max(self._next_episode_index, ep_idx + 1)

        # Register task
        if task_description not in self._tasks:
            self._tasks[task_description] = len(self._tasks)

        self._current = _EpisodeBuffer(
            episode_index=ep_idx,
            task_description=task_description,
            task_index=self._tasks[task_description],
        )
        return ep_idx

    def add_step(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        """Record one timestep of observation and action.

        Parameters
        ----------
        observation:
            Dict with ``"images"`` (dict of camera_name → uint8 HWC array),
            optional state vector under the configured ``state_key``.
        action:
            Dict with ``"actions"`` key containing the action vector.
        """
        buf = self._current
        if buf is None:
            raise RuntimeError("No active episode. Call start_episode() first.")

        # --- Images ---
        images = observation.get("images", {})
        if self._image_keys is not None:
            images = {k: images[k] for k in self._image_keys if k in images}

        for cam_name, frame in images.items():
            if cam_name not in buf.image_buffers:
                buf.image_buffers[cam_name] = []
            buf.image_buffers[cam_name].append(np.asarray(frame, dtype=np.uint8))

            # Record camera info on first frame
            if cam_name not in self._camera_info:
                h, w, c = frame.shape
                self._camera_info[cam_name] = {"height": h, "width": w, "channels": c}

        # --- Action ---
        act = np.asarray(action["actions"], dtype=np.float32)
        if act.ndim == 2:
            # Action chunk: store each sub-step individually
            for a in act:
                buf.actions.append(a)
        else:
            buf.actions.append(act)
            if self._action_dim is None:
                self._action_dim = act.shape[0]

        # --- State ---
        state = observation.get(self._state_key)
        if state is None:
            state = observation.get("state")
        if state is not None:
            state = np.asarray(state, dtype=np.float32)
            buf.states.append(state)
            if self._state_dim is None:
                self._state_dim = state.shape[0]
        else:
            buf.states.append(None)

    def end_episode(self, metadata: dict[str, Any] | None = None) -> None:
        """Finalize the current episode: encode videos, write parquet and episode metadata.

        Parameters
        ----------
        metadata:
            Optional extra metadata (e.g. ``{"success": True}``) stored alongside
            the episode record.
        """
        buf = self._current
        if buf is None:
            raise RuntimeError("No active episode. Call start_episode() first.")

        if metadata:
            buf.metadata = metadata

        n_frames = len(buf.actions)
        if n_frames == 0:
            logger.warning("Episode %d has no steps, skipping.", buf.episode_index)
            self._current = None
            return

        chunk_idx = buf.episode_index // self._chunks_size

        # --- Write videos ---
        if buf.image_buffers:
            if not self._has_ffmpeg:
                raise RuntimeError(
                    "ffmpeg is required to encode trajectory videos but was not found on PATH. "
                    "Install it: sudo apt install ffmpeg (Ubuntu/Debian) or brew install ffmpeg (macOS)."
                )
            for cam_name, frames in buf.image_buffers.items():
                video_key = f"observation.images.{cam_name}"
                video_path = (
                    self._root
                    / "videos"
                    / f"chunk-{chunk_idx:03d}"
                    / video_key
                    / f"episode_{buf.episode_index:06d}.mp4"
                )
                _encode_video(frames, video_path, self._fps, self._video_codec)

        # --- Write parquet ---
        self._write_parquet(buf, chunk_idx, n_frames)

        # --- Accumulate episode metadata ---
        self._episode_records.append(
            {
                "episode_index": buf.episode_index,
                "tasks": [buf.task_description],
                "length": n_frames,
                **({"metadata": buf.metadata} if buf.metadata else {}),
            }
        )

        # --- Compute and store per-episode stats ---
        self._episode_stats.append(self._compute_episode_stats(buf, n_frames))

        self._global_frame_index += n_frames
        self._current = None

    def close(self) -> None:
        """Write all metadata files (info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl)."""
        if self._current is not None:
            logger.warning("Closing with an unfinished episode; discarding episode %d.", self._current.episode_index)
            self._current = None

        meta_dir = self._root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        self._write_info_json(meta_dir)
        self._write_jsonl(meta_dir / "episodes.jsonl", self._episode_records)
        self._write_jsonl(
            meta_dir / "tasks.jsonl",
            [{"task_index": idx, "task": desc} for desc, idx in sorted(self._tasks.items(), key=lambda x: x[1])],
        )
        self._write_jsonl(meta_dir / "episodes_stats.jsonl", self._episode_stats)

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_parquet(self, buf: _EpisodeBuffer, chunk_idx: int, n_frames: int) -> None:
        """Write a single-episode parquet file."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_path = self._root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{buf.episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        columns: dict[str, Any] = {}

        # Actions
        actions = np.stack(buf.actions[:n_frames])
        columns["action"] = [row.tolist() for row in actions]

        # States
        has_states = any(s is not None for s in buf.states[:n_frames])
        if has_states:
            state_arrays = []
            for s in buf.states[:n_frames]:
                if s is not None:
                    state_arrays.append(s)
                else:
                    dim = self._state_dim or 0
                    state_arrays.append(np.zeros(dim, dtype=np.float32))
            states = np.stack(state_arrays)
            columns["observation.state"] = [row.tolist() for row in states]

        # Mandatory index columns
        columns["timestamp"] = [i / self._fps for i in range(n_frames)]
        columns["frame_index"] = list(range(n_frames))
        columns["episode_index"] = [buf.episode_index] * n_frames
        columns["index"] = list(range(self._global_frame_index, self._global_frame_index + n_frames))
        columns["task_index"] = [buf.task_index] * n_frames

        # Build Arrow schema
        fields = []
        action_dim = actions.shape[1]
        fields.append(pa.field("action", pa.list_(pa.float32(), action_dim)))
        if has_states:
            state_dim = states.shape[1]
            fields.append(pa.field("observation.state", pa.list_(pa.float32(), state_dim)))
        fields.append(pa.field("timestamp", pa.float32()))
        fields.append(pa.field("frame_index", pa.int64()))
        fields.append(pa.field("episode_index", pa.int64()))
        fields.append(pa.field("index", pa.int64()))
        fields.append(pa.field("task_index", pa.int64()))

        schema = pa.schema(fields)
        table = pa.table(columns, schema=schema)
        pq.write_table(table, parquet_path)

    def _compute_episode_stats(self, buf: _EpisodeBuffer, n_frames: int) -> dict[str, Any]:
        """Compute per-episode normalization stats for numeric features."""
        stats: dict[str, Any] = {}

        actions = np.stack(buf.actions[:n_frames])
        stats["action"] = _array_stats(actions)

        has_states = any(s is not None for s in buf.states[:n_frames])
        if has_states:
            state_arrays = [s for s in buf.states[:n_frames] if s is not None]
            if state_arrays:
                states = np.stack(state_arrays)
                stats["observation.state"] = _array_stats(states)

        timestamps = np.array([i / self._fps for i in range(n_frames)], dtype=np.float32)
        stats["timestamp"] = _array_stats(timestamps.reshape(-1, 1))

        return {"episode_index": buf.episode_index, "stats": stats}

    def _write_info_json(self, meta_dir: Path) -> None:
        """Write the top-level info.json metadata file."""
        total_episodes = len(self._episode_records)
        total_frames = sum(ep["length"] for ep in self._episode_records)
        total_videos = total_episodes * len(self._camera_info)
        total_chunks = (max((ep["episode_index"] for ep in self._episode_records), default=0) // self._chunks_size) + 1

        features: dict[str, Any] = {}

        # Action feature
        if self._action_dim is not None:
            features["action"] = {
                "dtype": "float32",
                "shape": [self._action_dim],
                "names": None,
            }

        # State feature
        if self._state_dim is not None:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": [self._state_dim],
                "names": None,
            }

        # Video features
        codec_meta = _codec_name_for_metadata(self._video_codec)
        for cam_name, cam in self._camera_info.items():
            video_key = f"observation.images.{cam_name}"
            features[video_key] = {
                "dtype": "video",
                "shape": [cam["height"], cam["width"], cam["channels"]],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": float(self._fps),
                    "video.height": cam["height"],
                    "video.width": cam["width"],
                    "video.channels": cam["channels"],
                    "video.codec": codec_meta,
                    "video.pix_fmt": _DEFAULT_PIX_FMT,
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

        # Mandatory index features
        for name, dtype in [
            ("timestamp", "float32"),
            ("frame_index", "int64"),
            ("episode_index", "int64"),
            ("index", "int64"),
            ("task_index", "int64"),
        ]:
            features[name] = {"dtype": dtype, "shape": [1], "names": None}

        info = {
            "codebase_version": _CODEBASE_VERSION,
            "robot_type": self._robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": total_videos,
            "total_chunks": total_chunks,
            "chunks_size": self._chunks_size,
            "fps": self._fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }

        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        """Write a list of dicts as newline-delimited JSON."""
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=_json_default) + "\n")


def _array_stats(arr: np.ndarray) -> dict[str, list[float]]:
    """Compute min/max/mean/std/count stats for a 2-D array (frames × features)."""
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _json_default(obj: object) -> Any:
    """JSON serializer fallback for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # type: ignore[no-matching-overload]
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
