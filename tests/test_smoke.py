from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_snn_prototype.config import SimConfig
from embodied_snn_prototype.simulate import run_closed_loop, summarize_episode


def test_closed_loop_smoke() -> None:
    config = SimConfig(steps=25)
    result = run_closed_loop(config)
    metrics = summarize_episode(result, config)

    assert result.trajectory.shape == (25, 5)
    assert result.sensory_history.shape == (25, 4)
    assert result.action_history.shape == (25, 4)
    assert result.rate_history.shape[0] == 25
    assert result.final_dust >= 0.0
    assert metrics.energy_proxy >= 0.0
    assert -3.0 <= metrics.objective <= 3.0


def test_structured_plastic_mode_smoke() -> None:
    config = SimConfig(steps=20, connectivity_mode="structured_plastic", neural_noise_std=0.01)
    result = run_closed_loop(config)
    metrics = summarize_episode(result, config)

    assert result.action_history.shape == (20, 4)
    assert float(result.action_history[:, 0].max()) <= 1.0
    assert float(result.action_history[:, 0].min()) >= 0.0
    assert float(result.action_history[:, 1].max()) <= 1.0
    assert float(result.action_history[:, 1].min()) >= -1.0
    assert metrics.energy_proxy >= 0.0