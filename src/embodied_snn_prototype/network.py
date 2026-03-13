from __future__ import annotations

import numpy as np

from .config import SimConfig


class StructuredLIFBrain:
    def __init__(self, config: SimConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed + 101)
        self.num_neurons = 9
        self.v_rest = 0.0
        self.v_reset = 0.0
        self.v_th = 1.0
        self.tau_mem = 20.0
        self.tau_syn = 10.0
        self.rate_decay = 0.92

        self.v = np.zeros(self.num_neurons, dtype=float)
        self.syn = np.zeros(self.num_neurons, dtype=float)
        self.spikes = np.zeros(self.num_neurons, dtype=float)
        self.rate_trace = np.zeros(self.num_neurons, dtype=float)

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
        self.base_w_rec = np.zeros((self.num_neurons, self.num_neurons), dtype=float)
        self.plasticity_enabled = False
        self._wire_brain(mode=config.connectivity_mode)

    def _wire_brain(self, mode: str = "structured") -> None:
        self.plasticity_enabled = False

        if mode == "no_recurrence":
            self.w_rec.fill(0.0)
            self.base_w_rec = self.w_rec.copy()
            return

        if mode == "random_sparse":
            self.w_rec = self._build_random_sparse_connectivity()
            self.base_w_rec = self.w_rec.copy()
            return

        if mode == "structured_rewired":
            self._build_structured_connectivity()
            self._apply_rewiring(self.config.rewiring_prob)
            self.base_w_rec = self.w_rec.copy()
            return

        if mode == "structured_plastic":
            self._build_structured_connectivity()
            self.base_w_rec = self.w_rec.copy()
            self.plasticity_enabled = True
            return

        self._build_structured_connectivity()
        self.base_w_rec = self.w_rec.copy()

    def _build_structured_connectivity(self) -> None:
        self.w_rec.fill(0.0)
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

    def _build_random_sparse_connectivity(self) -> np.ndarray:
        density = 0.22
        mask = self.rng.random((self.num_neurons, self.num_neurons)) < density
        weights = self.rng.uniform(-1.2, 1.2, size=(self.num_neurons, self.num_neurons))
        matrix = np.where(mask, weights, 0.0)
        np.fill_diagonal(matrix, np.clip(np.diag(matrix), -0.2, 0.25))
        return matrix

    def _apply_rewiring(self, rewiring_prob: float) -> None:
        if rewiring_prob <= 0.0:
            return

        mask = self.rng.random((self.num_neurons, self.num_neurons)) < rewiring_prob
        random_weights = self.rng.uniform(-1.0, 1.0, size=(self.num_neurons, self.num_neurons))
        self.w_rec = np.where(mask, random_weights, self.w_rec)
        np.fill_diagonal(self.w_rec, np.clip(np.diag(self.w_rec), -0.2, 0.3))

    def step(self, sensory: np.ndarray) -> np.ndarray:
        input_drive = self.w_in @ sensory
        input_drive[6] += self.config.hunger_drive

        if self.config.neural_noise_std > 0.0:
            input_drive += self.rng.normal(0.0, self.config.neural_noise_std, size=self.num_neurons)

        syn_decay = np.exp(-self.config.dt_ms / self.tau_syn)
        mem_scale = self.config.dt_ms / self.tau_mem

        self.syn = syn_decay * self.syn + self.w_rec @ self.spikes + input_drive
        self.v += mem_scale * (-(self.v - self.v_rest) + self.syn)

        prev_spikes = self.spikes.copy()
        spikes = (self.v >= self.v_th).astype(float)
        self.v = np.where(spikes > 0.0, self.v_reset, self.v)
        self.spikes = spikes
        self.rate_trace = self.rate_decay * self.rate_trace + (1.0 - self.rate_decay) * spikes * (1000.0 / self.config.dt_ms)

        if self.plasticity_enabled:
            self._apply_reward_modulated_plasticity(prev_spikes=prev_spikes, post_spikes=spikes, sensory=sensory)

        return spikes

    def _apply_reward_modulated_plasticity(
        self,
        prev_spikes: np.ndarray,
        post_spikes: np.ndarray,
        sensory: np.ndarray,
    ) -> None:
        reward_signal = float(sensory[3] - self.config.plasticity_reward_scale * sensory[2])
        reward_signal = float(np.clip(reward_signal, -1.0, 1.0))

        hebbian = np.outer(post_spikes, prev_spikes)
        delta = self.config.plasticity_lr * reward_signal * hebbian
        self.w_rec += delta

        if self.config.plasticity_decay > 0.0:
            self.w_rec += self.config.plasticity_decay * (self.base_w_rec - self.w_rec)

        self.w_rec = np.clip(self.w_rec, -self.config.plasticity_clip, self.config.plasticity_clip)
        np.fill_diagonal(self.w_rec, np.clip(np.diag(self.w_rec), -0.2, 0.3))

    def decode_action(self) -> dict[str, float]:
        turn_left = self.rate_trace[4] / 30.0
        turn_right = self.rate_trace[5] / 30.0
        forward = self.rate_trace[6] / 35.0
        eat = self.rate_trace[7] / 30.0
        groom = self.rate_trace[8] / 25.0

        forward_soft = 1.0 / (1.0 + np.exp(-self.config.forward_softclip_slope * (forward - self.config.forward_softclip_center)))
        turn_raw = turn_left - turn_right
        turn_soft = np.tanh(self.config.turn_softclip_gain * turn_raw)

        return {
            "turn": float(np.clip(turn_soft, -1.0, 1.0)),
            "forward": float(np.clip(forward_soft, 0.0, 1.0)),
            "eat": float(np.clip(eat, 0.0, 1.0)),
            "groom": float(np.clip(groom, 0.0, 1.0)),
        }