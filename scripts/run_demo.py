from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_snn_prototype.config import SimConfig
from embodied_snn_prototype.simulate import run_closed_loop, save_summary_figure, save_raster_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the embodied LIF prototype demo")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "quick.yaml"))
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs" / "trajectory.png"))
    parser.add_argument("--raster", type=str, default=str(ROOT / "outputs" / "raster.png"))
    args = parser.parse_args()

    config = SimConfig.from_yaml(args.config)
    result = run_closed_loop(config)
    save_summary_figure(result, config, args.output)
    save_raster_figure(result, config, args.raster)

    print("Embodied SNN demo complete")
    print(f"Food eaten: {result.food_eaten:.3f}")
    print(f"Final dust: {result.final_dust:.3f}")
    print(f"Final position: ({result.trajectory[-1, 0]:.3f}, {result.trajectory[-1, 1]:.3f})")
    print(f"Saved figure: {args.output}")
    print(f"Saved raster: {args.raster}")


if __name__ == "__main__":
    main()