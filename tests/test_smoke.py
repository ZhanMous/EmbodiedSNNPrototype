from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_snn_prototype.config import SimConfig
from embodied_snn_prototype.simulate import run_closed_loop


def test_closed_loop_smoke() -> None:
    config = SimConfig(steps=25)
    result = run_closed_loop(config)

    assert result.trajectory.shape == (25, 5)
    assert result.sensory_history.shape == (25, 4)
    assert result.action_history.shape == (25, 4)
    assert result.rate_history.shape[0] == 25
    assert result.final_dust >= 0.0


def test_plastic_readout_records_learning_history() -> None:
    config = SimConfig(steps=20, connectivity_mode="plastic_readout", plasticity_lr=0.02, plasticity_decay=0.002)
    result = run_closed_loop(config)

    assert result.reward_history.shape == (20,)
    assert result.readout_history.shape[0] == 21
    assert result.readout_history.shape[1:] == (5, 5)
    assert np.all(result.readout_history >= 0.0)
    assert np.max(result.readout_history) <= config.readout_weight_max + 1e-9