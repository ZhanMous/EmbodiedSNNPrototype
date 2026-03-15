from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_snn_prototype.config import SimConfig
from embodied_snn_prototype.simulate import run_closed_loop


def _read_sweep_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _run_grid(
    base: SimConfig,
    tau_mem_values: list[float],
    hunger_values: list[float],
    sync_values: list[float],
    seeds: list[int],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    total = len(tau_mem_values) * len(hunger_values) * len(sync_values) * len(seeds)
    done = 0

    for tau_mem in tau_mem_values:
        for hunger in hunger_values:
            for sync in sync_values:
                for seed in seeds:
                    cfg = replace(
                        base,
                        tau_mem=float(tau_mem),
                        hunger_drive=float(hunger),
                        brain_body_sync_ms=float(sync),
                        seed=int(seed),
                    )
                    result = run_closed_loop(cfg)
                    rows.append(
                        {
                            "tau_mem": float(tau_mem),
                            "hunger_drive": float(hunger),
                            "brain_body_sync_ms": float(sync),
                            "seed": int(seed),
                            "food_eaten": float(result.food_eaten),
                            "final_dust": float(result.final_dust),
                            "final_x": float(result.trajectory[-1, 0]),
                            "final_y": float(result.trajectory[-1, 1]),
                        }
                    )
                    done += 1
                    if done % 10 == 0 or done == total:
                        print(f"Progress: {done}/{total}")

    return rows


def _write_metrics_csv(rows: list[dict[str, float]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "tau_mem",
        "hunger_drive",
        "brain_body_sync_ms",
        "seed",
        "food_eaten",
        "final_dust",
        "final_x",
        "final_y",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_food_heatmaps(
    rows: list[dict[str, float]],
    tau_mem_values: list[float],
    hunger_values: list[float],
    sync_values: list[float],
    path: str | Path,
) -> None:
    # One heatmap per brain-body sync value. Axes: tau_mem x hunger_drive.
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tau_to_i = {float(v): i for i, v in enumerate(tau_mem_values)}
    hunger_to_j = {float(v): j for j, v in enumerate(hunger_values)}

    fig, axes = plt.subplots(1, len(sync_values), figsize=(4.5 * len(sync_values), 4), squeeze=False)
    axes_1d = axes[0]

    vmin = min(float(r["food_eaten"]) for r in rows)
    vmax = max(float(r["food_eaten"]) for r in rows)

    for ax, sync in zip(axes_1d, sync_values):
        values = np.zeros((len(tau_mem_values), len(hunger_values)), dtype=float)
        counts = np.zeros((len(tau_mem_values), len(hunger_values)), dtype=int)

        for row in rows:
            if float(row["brain_body_sync_ms"]) != float(sync):
                continue
            i = tau_to_i[float(row["tau_mem"])]
            j = hunger_to_j[float(row["hunger_drive"])]
            values[i, j] += float(row["food_eaten"])
            counts[i, j] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_values = np.divide(values, counts, where=counts > 0)

        im = ax.imshow(mean_values, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"sync={sync} ms")
        ax.set_xlabel("hunger_drive")
        ax.set_ylabel("tau_mem")
        ax.set_xticks(np.arange(len(hunger_values)))
        ax.set_xticklabels([f"{v:.2f}" for v in hunger_values])
        ax.set_yticks(np.arange(len(tau_mem_values)))
        ax.set_yticklabels([f"{v:.1f}" for v in tau_mem_values])

        for i in range(len(tau_mem_values)):
            for j in range(len(hunger_values)):
                ax.text(j, i, f"{mean_values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=axes_1d, fraction=0.025, pad=0.04)
    cbar.set_label("mean food_eaten")
    fig.suptitle("Food-eaten heatmaps by tau_mem and hunger_drive", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parameter sweep for embodied SNN prototype")
    parser.add_argument("--sweep-config", type=str, default=str(ROOT / "configs" / "sweep.yaml"))
    args = parser.parse_args()

    cfg = _read_sweep_yaml(args.sweep_config)
    base_cfg_path = ROOT / cfg.get("base_config", "configs/quick.yaml")
    base_cfg = SimConfig.from_yaml(base_cfg_path)

    sweep = cfg.get("sweep", {})
    tau_mem_values = [float(v) for v in sweep.get("tau_mem", [12.0, 20.0, 28.0])]
    hunger_values = [float(v) for v in sweep.get("hunger_drive", [0.40, 0.55, 0.70])]
    sync_values = [float(v) for v in sweep.get("brain_body_sync_ms", [10.0, 15.0, 20.0])]
    seeds = [int(v) for v in sweep.get("seeds", [1, 2, 3, 4, 5])]

    outputs = cfg.get("outputs", {})
    csv_path = ROOT / outputs.get("csv", "outputs/sweeps/metrics.csv")
    heatmap_path = ROOT / outputs.get("heatmap", "outputs/sweeps/food_eaten_heatmap.png")

    rows = _run_grid(base_cfg, tau_mem_values, hunger_values, sync_values, seeds)
    _write_metrics_csv(rows, csv_path)
    _plot_food_heatmaps(rows, tau_mem_values, hunger_values, sync_values, heatmap_path)

    print("Sweep complete")
    print(f"Saved metrics: {csv_path}")
    print(f"Saved heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
