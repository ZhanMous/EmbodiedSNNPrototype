from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np

from .config import SimConfig


@dataclass
class BodyState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    dust: float = 0.0
    food_eaten: float = 0.0


class EmbodiedArena:
    def __init__(self, config: SimConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.state = BodyState()

    def reset(self) -> BodyState:
        self.state = BodyState(x=-0.9, y=-0.2, heading=0.15)
        return self.state

    def _sensor_position(self, angle_offset: float) -> tuple[float, float]:
        theta = self.state.heading + angle_offset
        return (
            self.state.x + self.config.sensor_offset * cos(theta),
            self.state.y + self.config.sensor_offset * sin(theta),
        )

    def _food_signal_at(self, x_pos: float, y_pos: float) -> float:
        food_x, food_y = self.config.food_position
        dist_sq = (x_pos - food_x) ** 2 + (y_pos - food_y) ** 2
        signal = np.exp(-dist_sq / (2.0 * self.config.gradient_sigma**2))
        return float(signal)

    def _near_food(self) -> bool:
        food_x, food_y = self.config.food_position
        dist = np.hypot(self.state.x - food_x, self.state.y - food_y)
        return dist < 0.22

    def sense(self) -> np.ndarray:
        sensor_angle = self.config.sensor_angle_deg / 180.0 * pi
        left_pos = self._sensor_position(sensor_angle)
        right_pos = self._sensor_position(-sensor_angle)
        left_taste = self._food_signal_at(*left_pos)
        right_taste = self._food_signal_at(*right_pos)
        near_food = 1.0 if self._near_food() else 0.0
        return np.array([left_taste, right_taste, self.state.dust, near_food], dtype=float)

    def step(self, linear_drive: float, turn_drive: float, groom_drive: float, eat_drive: float) -> BodyState:
        linear = np.clip(linear_drive, 0.0, 1.0) * self.config.max_linear_speed
        turn = np.clip(turn_drive, -1.0, 1.0) * self.config.max_turn_rate
        groom = np.clip(groom_drive, 0.0, 1.0)
        eat = np.clip(eat_drive, 0.0, 1.0)

        if groom > 0.45:
            linear *= 0.15
            self.state.dust = max(0.0, self.state.dust - groom * self.config.groom_clean_rate)

        self.state.heading += turn
        self.state.x += linear * cos(self.state.heading)
        self.state.y += linear * sin(self.state.heading)

        radius = np.hypot(self.state.x, self.state.y)
        if radius > self.config.arena_radius:
            scale = self.config.arena_radius / max(radius, 1e-6)
            self.state.x *= scale
            self.state.y *= scale
            self.state.heading += pi

        self.state.dust = min(1.5, self.state.dust + self.config.dust_gain)

        if self._near_food() and eat > 0.35:
            self.state.food_eaten += 0.02 * eat
            self.state.dust = min(1.5, self.state.dust + 0.001)

        return self.state