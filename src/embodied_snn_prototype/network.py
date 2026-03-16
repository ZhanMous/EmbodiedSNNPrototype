from __future__ import annotations

import numpy as np

from .config import SimConfig


class StructuredLIFBrain:
    def __init__(self, config: SimConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.num_neurons = 9
        self.motor_neuron_start = 4
        self.motor_neuron_count = 5
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
        self.readout_trace = np.zeros(self.motor_neuron_count, dtype=float)
        self.post_trace = np.zeros(self.motor_neuron_count, dtype=float)
        self.base_readout_weights = np.zeros((self.motor_neuron_count, self.motor_neuron_count), dtype=float)
        self.readout_delta = np.zeros((self.motor_neuron_count, self.motor_neuron_count), dtype=float)
        self.readout_weights = np.zeros((self.motor_neuron_count, self.motor_neuron_count), dtype=float)
        self.readout_plastic_mask = np.zeros((self.motor_neuron_count, self.motor_neuron_count), dtype=float)
        self.readout_history: list[np.ndarray] = []
        self.reward_log: list[float] = []

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
        self._init_readout_weights()
        if self.config.connectivity_mode == "no_recurrence":
            self.w_rec[:, :] = 0.0
        elif self.config.connectivity_mode in {"random_sparse", "structured_plastic"}:
            self._apply_rewiring()
        elif self.config.connectivity_mode == "plastic_readout":
            pass

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

    def _init_readout_weights(self) -> None:
        base_diag = np.array([1.0 / 30.0, 1.0 / 30.0, 1.0 / 35.0, 1.0 / 30.0, 1.0 / 25.0], dtype=float)
        self.base_readout_weights[:, :] = 0.0
        np.fill_diagonal(self.base_readout_weights, base_diag)
        self.readout_delta[:, :] = 0.0
        self.readout_plastic_mask[:, :] = 0.0
        self.readout_plastic_mask[2, 2] = 1.0
        self.readout_plastic_mask[3, 3] = 1.0
        self.readout_plastic_mask[4, 4] = 1.0
        self.readout_weights = self.base_readout_weights + self.readout_delta
        self.readout_history = [self.readout_weights.copy()]

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
        motor_spikes = spikes[self.motor_neuron_start : self.motor_neuron_start + self.motor_neuron_count]
        self.readout_trace = self.config.readout_trace_decay * self.readout_trace + motor_spikes
        return spikes

    def apply_reward_modulated_plasticity(self, sensory: np.ndarray, action: dict[str, float], delta_food: float, delta_dust: float) -> float:
        if self.config.connectivity_mode not in {"structured_plastic", "plastic_readout"}:
            self.reward_log.append(0.0)
            self.readout_history.append(self.readout_weights.copy())
            return 0.0

        taste_left = float(sensory[0])
        taste_right = float(sensory[1])
        taste_mean = 0.5 * (taste_left + taste_right)
        taste_delta = taste_left - taste_right
        turn_alignment = action["turn"] * np.sign(taste_delta) * abs(taste_delta)

        reward = (
            self.config.reward_food_scale * delta_food
            + self.config.reward_near_food_scale * float(sensory[3])
            + self.config.reward_taste_scale * taste_mean
            + self.config.reward_turn_alignment_scale * turn_alignment
            - self.config.reward_dust_scale * max(delta_dust, 0.0)
            - self.config.reward_energy_scale
            * (action["forward"] + 0.5 * abs(action["turn"]) + 0.35 * action["groom"] + 0.2 * action["eat"])
        )
        learning_signal = max(reward, 0.0)

        post_activity = np.array(
            [
                max(action["turn"], 0.0),
                max(-action["turn"], 0.0),
                action["forward"],
                action["eat"],
                action["groom"],
            ],
            dtype=float,
        )
        self.post_trace = self.config.readout_trace_decay * self.post_trace + post_activity
        eligibility = np.outer(self.post_trace, self.readout_trace)

        self.readout_delta *= (1.0 - self.config.plasticity_decay)
        self.readout_delta += self.config.plasticity_lr * learning_signal * eligibility * self.readout_plastic_mask
        self.readout_delta = np.clip(self.readout_delta, 0.0, self.config.readout_weight_max)
        self.readout_weights = np.clip(self.base_readout_weights + self.readout_delta, 0.0, self.config.readout_weight_max)

        self.reward_log.append(float(reward))
        self.readout_history.append(self.readout_weights.copy())
        return float(reward)

    def decode_action(self) -> dict[str, float]:
        motor_rates = self.rate_trace[self.motor_neuron_start : self.motor_neuron_start + self.motor_neuron_count]
        readout = self.readout_weights @ motor_rates
        turn_left = readout[0]
        turn_right = readout[1]
        forward = readout[2]
        eat = readout[3]
        groom = readout[4]

        return {
            "turn": float(np.clip(turn_left - turn_right, -1.0, 1.0)),
            "forward": float(np.clip(forward, 0.0, 1.0)),
            "eat": float(np.clip(eat, 0.0, 1.0)),
            "groom": float(np.clip(groom, 0.0, 1.0)),
        }