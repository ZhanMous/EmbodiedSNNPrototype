from __future__ import annotations

import argparse
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_git_module(module_name: str, git_spec: str) -> types.ModuleType:
    source = subprocess.check_output(["git", "show", git_spec], cwd=ROOT).decode("utf-8")
    module = types.ModuleType(module_name)
    module.__file__ = f"git:{git_spec}"
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _ensure_legacy_package() -> None:
    package_name = "embodied_snn_prototype"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(SRC / package_name)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package


def _load_legacy_config(path: str | Path, legacy_config: Any) -> tuple[Any, list[int], list[str], list[float], list[float], list[float], list[float], list[float], Path]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sim_raw = data.get("sim", {})
    study_raw = data.get("study", {})

    if "food_position" in sim_raw:
        sim_raw["food_position"] = tuple(sim_raw["food_position"])

    sim_cfg = legacy_config.SimConfig(**sim_raw)
    seeds = [int(s) for s in study_raw.get("seeds", [7, 11, 19, 23])]
    modes = [str(m) for m in study_raw.get("modes", ["structured", "structured_rewired", "random_sparse", "no_recurrence", "structured_plastic"])]
    noise_values = [float(v) for v in study_raw.get("neural_noise_std_sweep", [sim_cfg.neural_noise_std])]
    rewiring_values = [float(v) for v in study_raw.get("rewiring_prob_sweep", [sim_cfg.rewiring_prob])]
    hunger_values = [float(v) for v in study_raw.get("hunger_drive_sweep", [sim_cfg.hunger_drive])]
    plasticity_lr_values = [float(v) for v in study_raw.get("plasticity_lr_sweep", [sim_cfg.plasticity_lr])]
    plasticity_decay_values = [float(v) for v in study_raw.get("plasticity_decay_sweep", [sim_cfg.plasticity_decay])]
    output_dir = ROOT / str(study_raw.get("output_dir", "outputs/benchmark_legacy"))
    return (
        sim_cfg,
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
    _ensure_legacy_package()
    legacy_config = _load_git_module("embodied_snn_prototype.config", "29cd1ac:src/embodied_snn_prototype/config.py")
    legacy_network = _load_git_module("embodied_snn_prototype.network", "29cd1ac:src/embodied_snn_prototype/network.py")
    legacy_simulate = _load_git_module("embodied_snn_prototype.simulate", "29cd1ac:src/embodied_snn_prototype/simulate.py")
    legacy_analysis = _load_git_module("embodied_snn_prototype.analysis", "29cd1ac:src/embodied_snn_prototype/analysis.py")

    if not hasattr(legacy_network.StructuredLIFBrain, "apply_reward_modulated_plasticity") and hasattr(
        legacy_network.StructuredLIFBrain, "_apply_reward_modulated_plasticity"
    ):
        legacy_network.StructuredLIFBrain.apply_reward_modulated_plasticity = (  # type: ignore[attr-defined]
            legacy_network.StructuredLIFBrain._apply_reward_modulated_plasticity
        )

    parser = argparse.ArgumentParser(description="Run legacy benchmark using archived 29cd1ac behavior")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "benchmark.yaml"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "benchmark_legacy"))
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
        _,
    ) = _load_legacy_config(args.config, legacy_config)
    out_dir = Path(args.output_dir)

    records, summary = legacy_analysis.run_connectivity_study(
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

    print("Legacy benchmark complete")
    print(f"Runs: {len(records)}")
    print(f"Output dir: {out_dir}")
    if summary:
        best = summary[0]
        print(
            "Best mode by objective: "
            f"{best['mode']} @ noise={float(best['neural_noise_std']):.3f}, rewiring={float(best['rewiring_prob']):.3f}, "
            f"hunger={float(best['hunger_drive']):.3f}, lr={float(best['plasticity_lr']):.3f}, decay={float(best['plasticity_decay']):.3f} "
            f"(objective_mean={float(best['objective_mean']):.4f}, food_mean={float(best['food_eaten_mean']):.4f}, energy_mean={float(best['energy_proxy_mean']):.4f})"
        )


if __name__ == "__main__":
    main()
