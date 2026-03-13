from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SimConfig
from .simulate import summarize_episode, run_closed_loop


def run_connectivity_study(
    base_config: SimConfig,
    seeds: list[int],
    modes: list[str],
    output_dir: str | Path,
    noise_values: list[float] | None = None,
    rewiring_values: list[float] | None = None,
    hunger_values: list[float] | None = None,
    plasticity_lr_values: list[float] | None = None,
    plasticity_decay_values: list[float] | None = None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | str]]]:
    records: list[dict[str, float | int | str]] = []
    noise_grid = noise_values or [float(base_config.neural_noise_std)]
    rewiring_grid = rewiring_values or [float(base_config.rewiring_prob)]
    hunger_grid = hunger_values or [float(base_config.hunger_drive)]
    plasticity_lr_grid = plasticity_lr_values or [float(base_config.plasticity_lr)]
    plasticity_decay_grid = plasticity_decay_values or [float(base_config.plasticity_decay)]

    for mode in modes:
        for noise in noise_grid:
            for rewiring in rewiring_grid:
                for hunger in hunger_grid:
                    for plasticity_lr in plasticity_lr_grid:
                        for plasticity_decay in plasticity_decay_grid:
                            for seed in seeds:
                                cfg = SimConfig(**asdict(base_config))
                                cfg.seed = int(seed)
                                cfg.connectivity_mode = mode
                                cfg.neural_noise_std = float(noise)
                                cfg.rewiring_prob = float(rewiring)
                                cfg.hunger_drive = float(hunger)
                                cfg.plasticity_lr = float(plasticity_lr)
                                cfg.plasticity_decay = float(plasticity_decay)
                                result = run_closed_loop(cfg)
                                metrics = summarize_episode(result, cfg)
                                row: dict[str, float | int | str] = {
                                    "mode": mode,
                                    "seed": seed,
                                    "steps": cfg.steps,
                                    "neural_noise_std": cfg.neural_noise_std,
                                    "rewiring_prob": cfg.rewiring_prob,
                                    "hunger_drive": cfg.hunger_drive,
                                    "plasticity_lr": cfg.plasticity_lr,
                                    "plasticity_decay": cfg.plasticity_decay,
                                    **asdict(metrics),
                                }
                                records.append(row)

    summary = _aggregate_records(records)
    _write_outputs(records, summary, output_dir)
    return records, summary


def _aggregate_records(records: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    metric_names = [
        "food_eaten",
        "final_dust",
        "near_food_ratio",
        "mean_forward",
        "mean_abs_turn",
        "mean_groom",
        "spike_rate_mean",
        "energy_proxy",
        "objective",
    ]

    by_mode: dict[tuple[str, float, float, float, float, float], list[dict[str, float | int | str]]] = {}
    for row in records:
        key = (
            str(row["mode"]),
            float(row.get("neural_noise_std", 0.0)),
            float(row.get("rewiring_prob", 0.0)),
            float(row.get("hunger_drive", 0.0)),
            float(row.get("plasticity_lr", 0.0)),
            float(row.get("plasticity_decay", 0.0)),
        )
        by_mode.setdefault(key, []).append(row)

    summary: list[dict[str, float | str]] = []
    for (mode, noise, rewiring, hunger, plasticity_lr, plasticity_decay), rows in by_mode.items():
        line: dict[str, float | str] = {
            "mode": mode,
            "neural_noise_std": noise,
            "rewiring_prob": rewiring,
            "hunger_drive": hunger,
            "plasticity_lr": plasticity_lr,
            "plasticity_decay": plasticity_decay,
            "n": float(len(rows)),
        }
        for metric in metric_names:
            vals = [float(r[metric]) for r in rows]
            line[f"{metric}_mean"] = float(mean(vals))
            line[f"{metric}_std"] = float(stdev(vals) if len(vals) > 1 else 0.0)
        summary.append(line)

    summary.sort(key=lambda x: float(x["objective_mean"]), reverse=True)
    return summary


def _write_outputs(
    records: list[dict[str, float | int | str]],
    summary: list[dict[str, float | str]],
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs_csv = out / "benchmark_runs.csv"
    summary_csv = out / "benchmark_summary.csv"
    report_md = out / "benchmark_report.md"
    bar_png = out / "objective_by_mode.png"
    facet_png = out / "objective_faceted_by_mode.png"
    scatter_png = out / "food_vs_energy.png"

    _write_csv(runs_csv, records)
    _write_csv(summary_csv, summary)
    _write_report(report_md, summary)
    _plot_objective_bar(bar_png, summary)
    _plot_objective_facets(facet_png, summary)
    _plot_food_energy_scatter(scatter_png, summary)


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: list[dict[str, float | str]]) -> None:
    lines: list[str] = []
    lines.append("# EmbodiedSNNPrototype Benchmark Report")
    lines.append("")
    lines.append("## Objective Ranking")
    lines.append("")
    lines.append("| mode | noise | rewiring | hunger | p_lr | p_decay | objective_mean | objective_std | food_eaten_mean | energy_proxy_mean | near_food_ratio_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for row in summary:
        normalized = {k: float(v) if isinstance(v, (int, float)) else v for k, v in row.items()}
        lines.append(
            "| {mode} | {neural_noise_std:.3f} | {rewiring_prob:.3f} | {hunger_drive:.3f} | {plasticity_lr:.3f} | {plasticity_decay:.3f} | {objective_mean:.4f} | {objective_std:.4f} | {food_eaten_mean:.4f} | {energy_proxy_mean:.4f} | {near_food_ratio_mean:.4f} |".format(
                **normalized
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- objective = food + 0.35*near_food - 0.25*dust - energy_proxy")
    lines.append("- energy_proxy combines spike activity and action magnitudes")
    lines.append("- objective_faceted_by_mode.png shows best objective vs hunger for each mode")
    lines.append("- Use benchmark_runs.csv for seed-level analysis")

    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_objective_bar(path: Path, summary: list[dict[str, float | str]]) -> None:
    modes = [
        (
            f"{r['mode']}\n"
            f"n={float(r['neural_noise_std']):.2f}, r={float(r['rewiring_prob']):.2f}\n"
            f"h={float(r['hunger_drive']):.2f}, lr={float(r['plasticity_lr']):.2f}, d={float(r['plasticity_decay']):.2f}"
        )
        for r in summary
    ]
    values = [float(r["objective_mean"]) for r in summary]
    errs = [float(r["objective_std"]) for r in summary]

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    ax.bar(modes, values, yerr=errs, capsize=4)
    ax.set_title("Objective by Connectivity Mode")
    ax.set_ylabel("objective_mean")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=30)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_objective_facets(path: Path, summary: list[dict[str, float | str]]) -> None:
    modes = sorted({str(r["mode"]) for r in summary})
    cols = 2
    rows = max(1, int(np.ceil(len(modes) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(11.2, 4.2 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for i, mode in enumerate(modes):
        ax = axes_flat[i]
        mode_rows = [r for r in summary if str(r["mode"]) == mode]
        hunger_vals = sorted({float(r["hunger_drive"]) for r in mode_rows})

        x_vals: list[float] = []
        y_vals: list[float] = []
        for hunger in hunger_vals:
            candidates = [float(r["objective_mean"]) for r in mode_rows if float(r["hunger_drive"]) == hunger]
            if not candidates:
                continue
            x_vals.append(hunger)
            y_vals.append(max(candidates))

        if x_vals:
            ax.plot(x_vals, y_vals, marker="o", linewidth=1.8)
        ax.set_title(f"{mode}: best objective vs hunger")
        ax.set_xlabel("hunger_drive")
        ax.set_ylabel("best objective_mean")
        ax.grid(alpha=0.3)

    for j in range(len(modes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_food_energy_scatter(path: Path, summary: list[dict[str, float | str]]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    for row in summary:
        food = float(row["food_eaten_mean"])
        energy = float(row["energy_proxy_mean"])
        mode = str(row["mode"])
        noise = float(row["neural_noise_std"])
        rewiring = float(row["rewiring_prob"])
        hunger = float(row["hunger_drive"])
        plasticity_lr = float(row["plasticity_lr"])
        plasticity_decay = float(row["plasticity_decay"])
        ax.scatter(energy, food, s=70)
        ax.text(
            energy + 0.001,
            food + 0.001,
            f"{mode}\nn={noise:.2f}, r={rewiring:.2f}\nh={hunger:.2f}, lr={plasticity_lr:.2f}, d={plasticity_decay:.2f}",
            fontsize=7,
        )

    ax.set_title("Food vs Energy Proxy")
    ax.set_xlabel("energy_proxy_mean")
    ax.set_ylabel("food_eaten_mean")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)
