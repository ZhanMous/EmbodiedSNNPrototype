from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_snn_prototype.config import SimConfig
from embodied_snn_prototype.simulate import run_closed_loop, save_summary_figure, summarize_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the embodied LIF prototype demo")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "quick.yaml"))
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs" / "trajectory.png"))
    args = parser.parse_args()

    config = SimConfig.from_yaml(args.config)
    result = run_closed_loop(config)
    metrics = summarize_episode(result, config)
    save_summary_figure(result, config, args.output)

    print("Embodied SNN demo complete")
    print(f"Food eaten: {result.food_eaten:.3f}")
    print(f"Final dust: {result.final_dust:.3f}")
    print(f"Final position: ({result.trajectory[-1, 0]:.3f}, {result.trajectory[-1, 1]:.3f})")
    print(f"Near-food ratio: {metrics.near_food_ratio:.3f}")
    print(f"Energy proxy: {metrics.energy_proxy:.3f}")
    print(f"Objective: {metrics.objective:.3f}")
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()