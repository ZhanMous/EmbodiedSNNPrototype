from __future__ import annotations

from pathlib import Path
import sys


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