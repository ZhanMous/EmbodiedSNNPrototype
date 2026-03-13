from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def load_study_config(
    path: str | Path,
) -> tuple[Any, list[int], list[str], list[float], list[float], list[float], list[float], list[float], Path]:
    from embodied_snn_prototype.config import SimConfig

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sim_raw = data.get("sim", {})
    study_raw = data.get("study", {})

    if "food_position" in sim_raw:
        sim_raw["food_position"] = tuple(sim_raw["food_position"])

    sim_config = SimConfig(**sim_raw)
    seeds = [int(s) for s in study_raw.get("seeds", [7, 11, 19, 23])]
    modes = [str(m) for m in study_raw.get("modes", ["structured", "structured_rewired", "random_sparse", "no_recurrence", "structured_plastic"])]
    noise_values = [float(v) for v in study_raw.get("neural_noise_std_sweep", [sim_config.neural_noise_std])]
    rewiring_values = [float(v) for v in study_raw.get("rewiring_prob_sweep", [sim_config.rewiring_prob])]
    hunger_values = [float(v) for v in study_raw.get("hunger_drive_sweep", [sim_config.hunger_drive])]
    plasticity_lr_values = [float(v) for v in study_raw.get("plasticity_lr_sweep", [sim_config.plasticity_lr])]
    plasticity_decay_values = [float(v) for v in study_raw.get("plasticity_decay_sweep", [sim_config.plasticity_decay])]
    output_dir = ROOT / str(study_raw.get("output_dir", "outputs/benchmark"))
    return (
        sim_config,
        seeds,
        modes,
        noise_values,
        rewiring_values,
        hunger_values,
        plasticity_lr_values,
        plasticity_decay_values,
        output_dir,
    )


def main() -> None:
    from embodied_snn_prototype.analysis import run_connectivity_study

    parser = argparse.ArgumentParser(description="Run EmbodiedSNN connectivity benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "benchmark.yaml"),
        help="YAML file containing sim + study fields",
    )
    args = parser.parse_args()

    (
        sim_cfg,
        seeds,
        modes,
        noise_values,
        rewiring_values,
        hunger_values,
        plasticity_lr_values,
        plasticity_decay_values,
        out_dir,
    ) = load_study_config(args.config)
    records, summary = run_connectivity_study(
        sim_cfg,
        seeds=seeds,
        modes=modes,
        noise_values=noise_values,
        rewiring_values=rewiring_values,
        hunger_values=hunger_values,
        plasticity_lr_values=plasticity_lr_values,
        plasticity_decay_values=plasticity_decay_values,
        output_dir=out_dir,
    )

    print("Embodied connectivity benchmark complete")
    print(f"Runs: {len(records)}")
    print(f"Modes: {', '.join(modes)}")
    print(f"Seeds: {', '.join(str(s) for s in seeds)}")
    print(f"Noise sweep: {', '.join(f'{v:.3f}' for v in noise_values)}")
    print(f"Rewiring sweep: {', '.join(f'{v:.3f}' for v in rewiring_values)}")
    print(f"Hunger sweep: {', '.join(f'{v:.3f}' for v in hunger_values)}")
    print(f"Plasticity lr sweep: {', '.join(f'{v:.3f}' for v in plasticity_lr_values)}")
    print(f"Plasticity decay sweep: {', '.join(f'{v:.3f}' for v in plasticity_decay_values)}")
    if summary:
        best = summary[0]
        print(
            "Best mode by objective: "
            f"{best['mode']} @ noise={float(best['neural_noise_std']):.3f}, rewiring={float(best['rewiring_prob']):.3f}, "
            f"hunger={float(best['hunger_drive']):.3f}, lr={float(best['plasticity_lr']):.3f}, decay={float(best['plasticity_decay']):.3f} "
            f"(objective_mean={float(best['objective_mean']):.4f}, "
            f"food_mean={float(best['food_eaten_mean']):.4f}, "
            f"energy_mean={float(best['energy_proxy_mean']):.4f})"
        )
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
