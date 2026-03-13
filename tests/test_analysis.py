from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_snn_prototype.analysis import run_connectivity_study
from embodied_snn_prototype.config import SimConfig


def test_connectivity_study_outputs(tmp_path: Path) -> None:
    base = SimConfig(steps=30, neural_noise_std=0.0)
    records, summary = run_connectivity_study(
        base,
        seeds=[7, 9],
        modes=["structured", "no_recurrence"],
        output_dir=tmp_path,
    )

    assert len(records) == 4
    assert len(summary) == 2
    assert (tmp_path / "benchmark_runs.csv").exists()
    assert (tmp_path / "benchmark_summary.csv").exists()
    assert (tmp_path / "benchmark_report.md").exists()
    assert (tmp_path / "objective_by_mode.png").exists()
    assert (tmp_path / "objective_faceted_by_mode.png").exists()
    assert (tmp_path / "food_vs_energy.png").exists()


def test_objective_differs_by_mode(tmp_path: Path) -> None:
    base = SimConfig(steps=35, neural_noise_std=0.0)
    _, summary = run_connectivity_study(
        base,
        seeds=[7, 11, 19],
        modes=["structured", "random_sparse", "no_recurrence"],
        output_dir=tmp_path,
    )

    objective_vals = [round(float(r["objective_mean"]), 6) for r in summary]
    assert len(set(objective_vals)) > 1


def test_sweep_grid_expands_runs(tmp_path: Path) -> None:
    base = SimConfig(steps=20, neural_noise_std=0.0, rewiring_prob=0.1)
    records, summary = run_connectivity_study(
        base,
        seeds=[3, 5],
        modes=["structured", "structured_plastic"],
        noise_values=[0.0, 0.03],
        rewiring_values=[0.05, 0.15],
        hunger_values=[0.35, 0.43],
        plasticity_lr_values=[0.01, 0.02],
        plasticity_decay_values=[0.005, 0.01],
        output_dir=tmp_path,
    )

    assert len(records) == 128
    assert len(summary) == 64
    assert all("neural_noise_std" in row for row in summary)
    assert all("rewiring_prob" in row for row in summary)
    assert all("hunger_drive" in row for row in summary)
    assert all("plasticity_lr" in row for row in summary)
    assert all("plasticity_decay" in row for row in summary)
