from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_snn_prototype.analysis import run_connectivity_study
from embodied_snn_prototype.config import SimConfig


def _write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_mode_comparison(path: Path, summary: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(summary, key=lambda row: str(row["mode"]))
    labels = [str(row["mode"]) for row in ordered]
    objective = [float(row["objective_mean"]) for row in ordered]
    objective_err = [float(row["objective_std"]) for row in ordered]
    food = [float(row["food_eaten_mean"]) for row in ordered]
    dust = [float(row["final_dust_mean"]) for row in ordered]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax1.bar(x - width / 2, objective, width=width, yerr=objective_err, capsize=4, label="objective_mean")
    ax1.bar(x + width / 2, food, width=width, label="food_eaten_mean")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Objective / Food")
    ax1.set_title("Fixed vs Plastic Readout")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, dust, color="firebrick", marker="o", linewidth=1.8, label="final_dust_mean")
    ax2.set_ylabel("Final dust")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed vs reward-modulated plastic readout")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "plasticity_compare"))
    parser.add_argument("--steps", type=int, default=220)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    base_cfg = SimConfig(steps=args.steps, neural_noise_std=0.0, rewiring_prob=0.0)

    _, summary = run_connectivity_study(
        base_cfg,
        seeds=[3, 7, 11, 19, 23],
        modes=["structured", "plastic_readout"],
        output_dir=output_dir,
        hunger_values=[0.45, 0.55, 0.65],
            plasticity_lr_values=[0.001, 0.005, 0.015],
        plasticity_decay_values=[0.002, 0.01],
    )

    best_by_mode: dict[str, dict[str, float | str]] = {}
    for row in summary:
        mode = str(row["mode"])
        if mode not in best_by_mode:
            best_by_mode[mode] = row

    best_rows = [best_by_mode[mode] for mode in sorted(best_by_mode)]
    summary_csv = output_dir / "best_readout_modes.csv"
    plot_path = output_dir / "fixed_vs_plastic.png"
    _write_summary_csv(summary_csv, best_rows)
    _plot_mode_comparison(plot_path, best_rows)

    print("Readout plasticity comparison complete")
    print(f"Saved best summary: {summary_csv}")
    print(f"Saved comparison plot: {plot_path}")
    for row in best_rows:
        print(
            "mode={mode} objective={objective_mean:.4f} food={food_eaten_mean:.4f} dust={final_dust_mean:.4f} hunger={hunger_drive:.3f} lr={plasticity_lr:.3f} decay={plasticity_decay:.3f}".format(
                **{k: float(v) if isinstance(v, (int, float)) else v for k, v in row.items()}
            )
        )


if __name__ == "__main__":
    main()
