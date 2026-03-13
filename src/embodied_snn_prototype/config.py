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
    hunger_drive: float = 0.45
    connectivity_mode: str = "structured"
    rewiring_prob: float = 0.1
    neural_noise_std: float = 0.0
    forward_softclip_center: float = 0.68
    forward_softclip_slope: float = 3.2
    turn_softclip_gain: float = 1.3
    plasticity_lr: float = 0.015
    plasticity_decay: float = 0.01
    plasticity_reward_scale: float = 0.5
    plasticity_clip: float = 1.8
    spike_energy_cost: float = 0.002
    linear_energy_cost: float = 0.05
    turn_energy_cost: float = 0.03
    groom_energy_cost: float = 0.025

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if "food_position" in raw:
            raw["food_position"] = tuple(raw["food_position"])
        return cls(**raw)