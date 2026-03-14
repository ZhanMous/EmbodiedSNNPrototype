from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SimConfig:
    seed: int = 7
    dt_ms: float = 1.0
    brain_body_sync_ms: float = 15.0
    steps: int = 220
    arena_radius: float = 2.0
    food_position: tuple[float, float] = (1.25, 0.35)
    sensor_offset: float = 0.16
    sensor_angle_deg: float = 32.0
    gradient_sigma: float = 0.75
    dust_gain: float = 0.003
    groom_clean_rate: float = 0.07
    max_linear_speed: float = 0.035
    max_turn_rate: float = 0.14
    hunger_drive: float = 0.55

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if "food_position" in raw:
            raw["food_position"] = tuple(raw["food_position"])
        return cls(**raw)