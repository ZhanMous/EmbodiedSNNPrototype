from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_config(path: str | Path) -> tuple[Any, Any, Any, list[int], Path]:
    from embodied_snn_prototype.config import SimConfig
    from embodied_snn_prototype.edl import EvolutionConfig, EDLBenchmarkConfig

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sim_raw = data.get("sim", {})
    evo_raw = data.get("evolution", {})
    bench_raw = data.get("benchmark", {})

    if "food_position" in sim_raw:
        sim_raw["food_position"] = tuple(sim_raw["food_position"])

    sim_cfg = SimConfig(**sim_raw)
    evo_cfg = EvolutionConfig(**evo_raw)
    bench_cfg = EDLBenchmarkConfig(**bench_raw)
    seeds = [int(s) for s in data.get("seeds", [3, 7, 11, 19])]
    output_dir = ROOT / str(data.get("output_dir", "outputs/edl_benchmark"))
    return sim_cfg, evo_cfg, bench_cfg, seeds, output_dir


def main() -> None:
    from embodied_snn_prototype.edl import run_edl_benchmark

    parser = argparse.ArgumentParser(description="Run EDL benchmark: CRN vs EA vs Evo-Learning")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "edl_benchmark.yaml"),
        help="YAML file containing sim + evolution + benchmark fields",
    )
    args = parser.parse_args()

    sim_cfg, evo_cfg, bench_cfg, seeds, output_dir = _load_config(args.config)
    runs, summary = run_edl_benchmark(
        base_config=sim_cfg,
        seeds=seeds,
        output_dir=output_dir,
        evo_cfg=evo_cfg,
        bench_cfg=bench_cfg,
    )

    print("EDL benchmark complete")
    print(f"Runs: {len(runs)}")
    print(f"Seeds: {', '.join(str(s) for s in seeds)}")
    print(f"Output dir: {output_dir}")
    print("Ranking:")
    for i, row in enumerate(summary, 1):
        print(
            f"{i}. {row['method']} objective={float(row['objective_mean']):.4f} "
            f"food={float(row['food_eaten_mean']):.4f} energy={float(row['energy_proxy_mean']):.4f}"
        )


if __name__ == "__main__":
    main()
