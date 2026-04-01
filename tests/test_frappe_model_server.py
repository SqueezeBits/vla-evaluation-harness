from __future__ import annotations

from pathlib import Path

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.frappe import FRAPPEModelServer, _resolve_checkpoint_dir


class _FakeFRAPPEServer(FRAPPEModelServer):
    def __init__(self) -> None:
        super().__init__(
            checkpoint="dummy",
            chunk_size=3,
            resize_width=8,
            resize_height=8,
            apply_jpeg_reencode=False,
        )
        self.windows: list[list[float]] = []

    def _predict_chunk(self, session):
        self.windows.append([float(frame.joint_state[0]) for frame in session.observation_window])
        value = float(session.observation_window[-1].joint_state[0])
        return np.array([[value * 10 + 1], [value * 10 + 2], [value * 10 + 3]], dtype=np.float32)


def _make_obs(value: float) -> dict[str, object]:
    image = np.full((8, 8, 3), int(value), dtype=np.uint8)
    return {
        "images": {
            "head_camera": image,
            "right_camera": image,
            "left_camera": image,
        },
        "task_description": "handover mic",
        "joint_state": np.full(14, value, dtype=np.float32),
    }


def test_extract_camera_arrays_falls_back_to_available_images() -> None:
    server = FRAPPEModelServer(
        checkpoint="dummy",
        resize_width=8,
        resize_height=8,
        apply_jpeg_reencode=False,
    )
    obs = {
        "images": {
            "head_camera": np.zeros((8, 8, 3), dtype=np.uint8),
            "left_camera": np.full((8, 8, 3), 7, dtype=np.uint8),
        }
    }

    images = server._extract_camera_arrays(obs)

    assert len(images) == 3
    assert int(images[0][0, 0, 0]) == 0
    assert int(images[1][0, 0, 0]) == 7
    assert int(images[2][0, 0, 0]) == 7


def test_predict_updates_history_while_serving_buffered_actions() -> None:
    server = _FakeFRAPPEServer()
    ctx = SessionContext("session-frappe", "episode-frappe")

    actions = [server.predict(_make_obs(value), ctx)["actions"] for value in (1.0, 2.0, 3.0, 4.0)]

    np.testing.assert_allclose(actions[0], np.array([11.0], dtype=np.float32))
    np.testing.assert_allclose(actions[1], np.array([12.0], dtype=np.float32))
    np.testing.assert_allclose(actions[2], np.array([13.0], dtype=np.float32))
    np.testing.assert_allclose(actions[3], np.array([41.0], dtype=np.float32))
    assert server.windows == [[0.0, 1.0], [3.0, 4.0]]


def test_resolve_checkpoint_dir_accepts_weight_file_parent(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "frappe-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}")
    weights = checkpoint_dir / "pytorch_model.bin"
    weights.write_bytes(b"weights")

    resolved_dir = _resolve_checkpoint_dir(str(checkpoint_dir), cache_dir=tmp_path)
    resolved_file_parent = _resolve_checkpoint_dir(str(weights), cache_dir=tmp_path)

    assert resolved_dir == checkpoint_dir
    assert resolved_file_parent == checkpoint_dir
