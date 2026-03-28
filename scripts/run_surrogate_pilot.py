from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_snn_prototype.config import SimConfig
from embodied_snn_prototype.edl import (
    EDLBenchmarkConfig,
    EvolutionConfig,
    _run_method_crn,
    _run_method_ea,
    _run_method_plastic_only,
    _run_method_surrogate,
)


def main() -> None:
    base_cfg = SimConfig()
    evo_cfg = EvolutionConfig()
    bench_cfg = EDLBenchmarkConfig(
        train_episodes=4,
        eval_episodes=6,
        train_seed_stride=10,
        eval_seed_offset=1000,
        surrogate_steps=120,
        surrogate_lr=0.01,
    )
    seeds = [3, 7, 11]

    results = {
        "baseline": [],
        "surrogate_backprop": [],
        "ea_readout": [],
        "crn_development": [],
    }

    for seed in seeds:
        local_evo = EvolutionConfig(**evo_cfg.__dict__)
        local_evo.seed = int(seed * 101 + evo_cfg.seed)
        m0, _ = _run_method_plastic_only(base_cfg, seed, bench_cfg)
        m1, _ = _run_method_surrogate(base_cfg, seed, bench_cfg)
        m2, _ = _run_method_ea(base_cfg, seed, local_evo, bench_cfg)
        m3, _ = _run_method_crn(base_cfg, seed, local_evo, bench_cfg)
        results["baseline"].append(m0["objective"])
        results["surrogate_backprop"].append(m1["objective"])
        results["ea_readout"].append(m2["objective"])
        results["crn_development"].append(m3["objective"])

    summary = {}
    for method, values in results.items():
        summary[method] = {
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values,
        }

    out_dir = ROOT / "outputs" / "surrogate_compare_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Pilot comparison", "", "| method | n | objective_mean | objective_std |", "| --- | ---: | ---: | ---: |"]
    for method, row in summary.items():
        lines.append(f"| {method} | {row['n']} | {row['mean']:.6f} | {row['std']:.6f} |")
    lines.append("")
    for method, row in summary.items():
        values = ", ".join(f"{value:.6f}" for value in row["values"])
        lines.append(f"- {method}: {values}")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print((out_dir / "summary.md").as_posix())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
