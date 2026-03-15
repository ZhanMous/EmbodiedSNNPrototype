from __future__ import annotations

import numpy as np

from .config import SimConfig


class StructuredLIFBrain:
    def __init__(self, config: SimConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.num_neurons = 9
        self.v_rest = 0.0
        self.v_reset = 0.0
        self.v_th = 1.0
        self.tau_mem = config.tau_mem
        self.tau_syn = config.tau_syn
        self.rate_decay = 0.92

        self.v = np.zeros(self.num_neurons, dtype=float)
        self.syn = np.zeros(self.num_neurons, dtype=float)
        self.spikes = np.zeros(self.num_neurons, dtype=float)
        self.rate_trace = np.zeros(self.num_neurons, dtype=float)
        self.spike_log: list[np.ndarray] = []

        self.w_in = np.array(
            [
                [1.8, 0.0, 0.0, 0.0],
                [0.0, 1.8, 0.0, 0.0],
                [0.0, 0.0, 2.1, 0.0],
                [0.0, 0.0, 0.0, 1.9],
                [0.8, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0],
                [0.4, 0.4, 0.0, 0.8],
                [0.0, 0.0, 0.0, 1.2],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        self.w_rec = np.zeros((self.num_neurons, self.num_neurons), dtype=float)
        self._wire_brain()

        self.motor_scale = np.ones(5, dtype=float)
        if self.config.connectivity_mode == "no_recurrence":
            self.w_rec[:, :] = 0.0
        elif self.config.connectivity_mode in {"random_sparse", "structured_plastic"}:
            self._apply_rewiring()

    def _wire_brain(self) -> None:
        left_sensor = 0
        right_sensor = 1
        dust_sensor = 2
        near_food = 3
        turn_left = 4
        turn_right = 5
        forward = 6
        eat = 7
        groom = 8

        self.w_rec[turn_left, left_sensor] = 1.1
        self.w_rec[turn_right, right_sensor] = 1.1
        self.w_rec[forward, left_sensor] = 0.5
        self.w_rec[forward, right_sensor] = 0.5
        self.w_rec[eat, left_sensor] = 0.3
        self.w_rec[eat, right_sensor] = 0.3
        self.w_rec[groom, dust_sensor] = 1.5
        self.w_rec[eat, near_food] = 1.1

        self.w_rec[turn_left, turn_right] = -0.9
        self.w_rec[turn_right, turn_left] = -0.9

        self.w_rec[forward, groom] = -1.4
        self.w_rec[eat, groom] = -0.8
        self.w_rec[forward, eat] = -0.5
        self.w_rec[turn_left, groom] = -0.3
        self.w_rec[turn_right, groom] = -0.3

        self.w_rec[turn_left, turn_left] = 0.18
        self.w_rec[turn_right, turn_right] = 0.18
        self.w_rec[forward, forward] = 0.18
        self.w_rec[eat, eat] = 0.2
        self.w_rec[groom, groom] = 0.24

    def _apply_rewiring(self) -> None:
        prob = float(np.clip(self.config.rewiring_prob, 0.0, 1.0))
        if prob <= 0.0:
            return
        mask = self.rng.random(self.w_rec.shape) < prob
        random_weights = self.rng.normal(loc=0.0, scale=0.55, size=self.w_rec.shape)
        self.w_rec = np.where(mask, random_weights, self.w_rec)

    def step(self, sensory: np.ndarray) -> np.ndarray:
        input_drive = self.w_in @ sensory
        input_drive[6] += self.config.hunger_drive
        if self.config.neural_noise_std > 0.0:
            input_drive += self.rng.normal(0.0, self.config.neural_noise_std, size=self.num_neurons)

        syn_decay = np.exp(-self.config.dt_ms / self.tau_syn)
        mem_scale = self.config.dt_ms / self.tau_mem

        self.syn = syn_decay * self.syn + self.w_rec @ self.spikes + input_drive
        self.v += mem_scale * (-(self.v - self.v_rest) + self.syn)

        spikes = (self.v >= self.v_th).astype(float)
        self.v = np.where(spikes > 0.0, self.v_reset, self.v)
        self.spikes = spikes
        self.spike_log.append(spikes.copy())
        self.rate_trace = self.rate_decay * self.rate_trace + (1.0 - self.rate_decay) * spikes * (1000.0 / self.config.dt_ms)

        if self.config.connectivity_mode == "structured_plastic":
            # Keep adaptation bounded and simple: strengthen eat/forward near food, relax with decay.
            reward = float(sensory[3])
            target = np.array([0.0, 0.0, reward, reward, 0.0], dtype=float)
            self.motor_scale += self.config.plasticity_lr * target
            self.motor_scale *= (1.0 - self.config.plasticity_decay)
            self.motor_scale = np.clip(self.motor_scale, 0.6, 1.4)
        return spikes

    def decode_action(self) -> dict[str, float]:
        turn_left = self.motor_scale[0] * self.rate_trace[4] / 30.0
        turn_right = self.motor_scale[1] * self.rate_trace[5] / 30.0
        forward = self.motor_scale[2] * self.rate_trace[6] / 35.0
        eat = self.motor_scale[3] * self.rate_trace[7] / 30.0
        groom = self.motor_scale[4] * self.rate_trace[8] / 25.0

        return {
            "turn": float(np.clip(turn_left - turn_right, -1.0, 1.0)),
            "forward": float(np.clip(forward, 0.0, 1.0)),
            "eat": float(np.clip(eat, 0.0, 1.0)),
            "groom": float(np.clip(groom, 0.0, 1.0)),
        }