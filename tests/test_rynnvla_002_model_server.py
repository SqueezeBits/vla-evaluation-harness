from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path

import numpy as np

from vla_eval.cli.config_loader import load_config
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.rynnvla_002 import (
    RynnVLA002ModelServer,
    _Frame,
    _HistorySpec,
    _build_request_item,
    _parse_history_spec,
    _unnormalize_actions,
)


def test_parse_history_spec_detects_wrist_and_state() -> None:
    spec = _parse_history_spec("his_2_third_view_wrist_w_state")

    assert spec == _HistorySpec(frames=2, use_wrist=True, use_state=True)


def test_build_request_item_uses_previous_and_current_images() -> None:
    spec = _HistorySpec(frames=2, use_wrist=True, use_state=True)
    history = deque([_Frame(front="front_prev", wrist="wrist_prev")])

    item = _build_request_item(
        task_description="open drawer",
        history_spec=spec,
        history=history,
        current_front="front_now",
        current_wrist="wrist_now",
        state=np.arange(8, dtype=np.float32),
    )

    assert (
        item["conversations"][0]["value"]
        == "What action should the robot take to open drawer?<|state|><|image|><|image|><|image|><|image|>"
    )
    assert item["image"] == ["front_prev", "wrist_prev", "front_now", "wrist_now"]
    np.testing.assert_array_equal(item["state"], np.arange(8, dtype=np.float32))


def test_unnormalize_actions_matches_libero_minmax() -> None:
    actions = np.asarray([[-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    unnormalized = _unnormalize_actions(actions)

    expected = np.asarray(
        [[-0.9375, 0.0, 0.9375, 0.05303571, 0.0, 0.00535715, -1.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(unnormalized, expected, atol=1e-6)


def test_predict_uses_internal_chunk_cache_and_advances_history(monkeypatch) -> None:
    server = RynnVLA002ModelServer(chunk_size=2)
    calls: list[int] = []

    def _predict_chunk(item):
        calls.append(len(item["image"]))
        return np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(server, "_predict_chunk", _predict_chunk)

    ctx = SessionContext(session_id="session-a", episode_id="episode-a")
    obs0 = {
        "images": {
            "agentview": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist": np.ones((2, 2, 3), dtype=np.uint8),
        },
        "states": np.zeros(8, dtype=np.float32),
        "task_description": "open drawer",
    }
    obs1 = {
        "images": {
            "agentview": np.full((2, 2, 3), 2, dtype=np.uint8),
            "wrist": np.full((2, 2, 3), 3, dtype=np.uint8),
        },
        "states": np.ones(8, dtype=np.float32),
        "task_description": "open drawer",
    }
    obs2 = {
        "images": {
            "agentview": np.full((2, 2, 3), 4, dtype=np.uint8),
            "wrist": np.full((2, 2, 3), 5, dtype=np.uint8),
        },
        "states": np.full(8, 2.0, dtype=np.float32),
        "task_description": "open drawer",
    }

    first = server.predict(obs0, ctx)["actions"]
    second = server.predict(obs1, ctx)["actions"]
    third = server.predict(obs2, ctx)["actions"]

    np.testing.assert_array_equal(first, np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(second, np.asarray([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(third, np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert calls == [2, 4]
    assert len(server._sessions[ctx.session_id].history) == 1


def test_episode_start_resets_cached_session_state() -> None:
    server = RynnVLA002ModelServer(chunk_size=2)
    ctx = SessionContext(session_id="session-reset", episode_id="episode-reset")
    session = server._session_state(ctx.session_id)
    session.pending_actions.append(np.ones(7, dtype=np.float32))

    asyncio.run(server.on_episode_start({}, ctx))

    assert len(server._sessions[ctx.session_id].pending_actions) == 0
    assert len(server._sessions[ctx.session_id].history) == 0


def test_rynnvla_configs_resolve_suite_specific_checkpoints() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    goal = load_config(str(repo_root / "configs/model_servers/rynnvla_002/libero_goal.yaml"))
    spatial = load_config(str(repo_root / "configs/model_servers/rynnvla_002/libero_spatial.yaml"))

    assert goal["args"]["model_subfolder"] == "VLA_model_256/libero_goal"
    assert goal["args"]["chunk_size"] == 5
    assert spatial["args"]["model_subfolder"] == "VLA_model_256/libero_spatial"
    assert spatial["args"]["chunk_size"] == 10
