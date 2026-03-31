from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vla_eval.cli.config_loader import load_config
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.vla_jepa import (
    _load_observation_params,
    _postprocess_actions,
    _resolve_checkpoint_path,
    VLAJEPAModelServer,
)


@pytest.mark.parametrize("checkpoint_filename", [None, "VLA-JEPA-LIBERO.pt"])
def test_resolve_checkpoint_path_from_local_variant_dir(tmp_path: Path, checkpoint_filename: str | None) -> None:
    variant_dir = tmp_path / "LIBERO"
    checkpoint_dir = variant_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    expected = checkpoint_dir / "VLA-JEPA-LIBERO.pt"
    expected.write_bytes(b"fake")
    (variant_dir / "config.yaml").write_text("framework: {}\n")
    (variant_dir / "dataset_statistics.json").write_text("{}\n")

    resolved = _resolve_checkpoint_path(
        str(tmp_path),
        checkpoint_variant="LIBERO",
        checkpoint_filename=checkpoint_filename,
    )

    assert resolved == str(expected)


def test_postprocess_actions_unnormalizes_and_flips_gripper() -> None:
    normalized = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    stats = {
        "q01": np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
        "q99": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "mask": np.array([True, True, True, True, True, True, False]),
    }

    actions = _postprocess_actions(normalized, stats, denorm_mode="quantile", invert_gripper=True)

    np.testing.assert_allclose(actions[0, :6], np.zeros(6, dtype=np.float32))
    assert actions[0, 6] == pytest.approx(-1.0)


def test_postprocess_actions_uses_minmax_for_libero_mode() -> None:
    normalized = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    stats = {
        "q01": np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
        "q99": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "min": np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0], dtype=np.float32),
        "max": np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0], dtype=np.float32),
        "mask": np.array([True, True, True, True, True, True, False]),
    }

    quantile_actions = _postprocess_actions(normalized, stats, denorm_mode="quantile", invert_gripper=False)
    minmax_actions = _postprocess_actions(normalized, stats, denorm_mode="minmax", invert_gripper=False)

    assert quantile_actions[0, 0] == pytest.approx(1.0)
    assert minmax_actions[0, 0] == pytest.approx(2.0)


def test_load_observation_params_accepts_python_repr_strings() -> None:
    params = _load_observation_params("{'send_wrist_image': True, 'send_state': True}")

    assert params == {"send_wrist_image": True, "send_state": True}


class _RecordingModel:
    def __init__(self) -> None:
        self.batch_images = None

    def predict_action(self, *, batch_images, instructions, state):  # noqa: ANN001
        self.batch_images = batch_images
        return {"normalized_actions": np.zeros((len(batch_images), 1, 7), dtype=np.float32)}


class _SequencedModel:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = outputs
        self.calls = 0

    def predict_action(self, *, batch_images, instructions, state):  # noqa: ANN001
        output = self.outputs[self.calls]
        self.calls += 1
        return {"normalized_actions": output[None, ...]}


def test_predict_batch_resizes_images_to_default_224() -> None:
    server = VLAJEPAModelServer(checkpoint="dummy")
    server._model = _RecordingModel()
    server._action_stats = {
        "q01": np.full(7, -1.0, dtype=np.float32),
        "q99": np.full(7, 1.0, dtype=np.float32),
        "mask": np.array([True, True, True, True, True, True, False]),
    }

    obs = {
        "images": {"primary": np.zeros((256, 320, 3), dtype=np.uint8)},
        "task_description": "test task",
    }
    ctx = SessionContext("session-resize", "episode-resize")

    server.predict_batch([obs], [ctx])

    assert server._model.batch_images is not None
    assert server._model.batch_images[0][0].size == (224, 224)


def test_predict_batch_applies_adaptive_ensemble_for_simpler_style_chunks() -> None:
    chunk_a = np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    chunk_b = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    server = VLAJEPAModelServer(
        checkpoint="dummy",
        adaptive_ensemble_horizon=7,
        adaptive_ensemble_alpha=0.0,
        invert_gripper=False,
    )
    server._model = _SequencedModel([chunk_a, chunk_b])
    server._action_stats = {
        "q01": np.full(7, -1.0, dtype=np.float32),
        "q99": np.full(7, 1.0, dtype=np.float32),
        "mask": np.array([True, True, True, True, True, True, False]),
    }

    obs = {"images": {"primary": np.zeros((224, 224, 3), dtype=np.uint8)}, "task_description": "test task"}
    ctx = SessionContext("session-a", "episode-a")

    first = server.predict_batch([obs], [ctx])[0]["actions"]
    second = server.predict_batch([obs], [ctx])[0]["actions"]

    np.testing.assert_allclose(first, chunk_a[0], atol=1e-6)
    np.testing.assert_allclose(second, np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)


def test_vla_jepa_configs_resolve_variant_specific_runtime_overrides() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    simpler = load_config(str(repo_root / "configs/model_servers/vla_jepa/simpler.yaml"))
    libero = load_config(str(repo_root / "configs/model_servers/vla_jepa/libero.yaml"))

    assert simpler["args"]["chunk_size"] == 1
    assert simpler["args"]["invert_gripper"] is False
    assert simpler["args"]["adaptive_ensemble_horizon"] == 7
    assert simpler["args"]["adaptive_ensemble_alpha"] == pytest.approx(0.1)
    assert libero["args"]["chunk_size"] == 7
    assert libero["args"]["denorm_mode"] == "minmax"
