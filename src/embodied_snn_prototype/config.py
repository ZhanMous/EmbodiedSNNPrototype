from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path

import yaml


@dataclass
class SimConfig:
    seed: int = 7
    dt_ms: float = 1.0
    brain_body_sync_ms: float = 15.0
    tau_mem: float = 20.0
    tau_syn: float = 10.0
    connectivity_mode: str = "structured"
    neural_noise_std: float = 0.0
    rewiring_prob: float = 0.0
    plasticity_lr: float = 0.01
    plasticity_decay: float = 0.01
    readout_trace_decay: float = 0.90
    readout_weight_max: float = 0.08
    reward_food_scale: float = 12.0
    reward_near_food_scale: float = 0.08
    reward_taste_scale: float = 0.10
    reward_turn_alignment_scale: float = 0.06
    reward_dust_scale: float = 0.04
    reward_energy_scale: float = 0.008
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
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in allowed}
        return cls(**filtered)