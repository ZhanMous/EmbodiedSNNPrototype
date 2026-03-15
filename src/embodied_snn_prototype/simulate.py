from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SimConfig
from .environment import EmbodiedArena
from .network import StructuredLIFBrain


@dataclass
class SimulationResult:
    trajectory: np.ndarray
    sensory_history: np.ndarray
    action_history: np.ndarray
    rate_history: np.ndarray
    spike_history: np.ndarray  # shape (brain_steps, num_neurons), brain_steps = steps * sync_steps
    food_eaten: float
    final_dust: float


@dataclass
class EpisodeMetrics:
    food_eaten: float
    final_dust: float
    near_food_ratio: float
    mean_forward: float
    mean_abs_turn: float
    mean_groom: float
    spike_rate_mean: float
    energy_proxy: float
    objective: float


def run_closed_loop(config: SimConfig) -> SimulationResult:
    arena = EmbodiedArena(config)
    brain = StructuredLIFBrain(config)
    arena.reset()

    sync_steps = max(1, int(round(config.brain_body_sync_ms / config.dt_ms)))
    trajectory = []
    sensory_history = []
    action_history = []
    rate_history = []

    for _ in range(config.steps):
        sensory = arena.sense()
        for _ in range(sync_steps):
            brain.step(sensory)

        action = brain.decode_action()
        state = arena.step(
            linear_drive=action["forward"],
            turn_drive=action["turn"],
            groom_drive=action["groom"],
            eat_drive=action["eat"],
        )

        trajectory.append([state.x, state.y, state.heading, state.dust, state.food_eaten])
        sensory_history.append(sensory.copy())
        action_history.append([action["forward"], action["turn"], action["groom"], action["eat"]])
        rate_history.append(brain.rate_trace.copy())

    return SimulationResult(
        trajectory=np.asarray(trajectory, dtype=float),
        sensory_history=np.asarray(sensory_history, dtype=float),
        action_history=np.asarray(action_history, dtype=float),
        rate_history=np.asarray(rate_history, dtype=float),
        spike_history=np.asarray(brain.spike_log, dtype=float),
        food_eaten=float(arena.state.food_eaten),
        final_dust=float(arena.state.dust),
    )


def summarize_episode(result: SimulationResult, config: SimConfig) -> EpisodeMetrics:
    actions = result.action_history
    sensory = result.sensory_history
    rates = result.rate_history

    near_food_ratio = float(np.mean(sensory[:, 3])) if sensory.size else 0.0
    mean_forward = float(np.mean(actions[:, 0])) if actions.size else 0.0
    mean_abs_turn = float(np.mean(np.abs(actions[:, 1]))) if actions.size else 0.0
    mean_groom = float(np.mean(actions[:, 2])) if actions.size else 0.0
    spike_rate_mean = float(np.mean(rates)) if rates.size else 0.0

    # Small energy proxy encourages useful behavior with parsimonious activity.
    action_cost = 0.30 * mean_forward + 0.25 * mean_abs_turn + 0.20 * mean_groom
    spike_cost = 0.0012 * spike_rate_mean
    energy_proxy = float(action_cost + spike_cost)

    objective = float(
        result.food_eaten
        + 0.35 * near_food_ratio
        - 0.25 * result.final_dust
        - energy_proxy
    )

    return EpisodeMetrics(
        food_eaten=float(result.food_eaten),
        final_dust=float(result.final_dust),
        near_food_ratio=near_food_ratio,
        mean_forward=mean_forward,
        mean_abs_turn=mean_abs_turn,
        mean_groom=mean_groom,
        spike_rate_mean=spike_rate_mean,
        energy_proxy=energy_proxy,
        objective=objective,
    )


def save_summary_figure(result: SimulationResult, config: SimConfig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = result.trajectory
    sensory = result.sensory_history
    actions = result.action_history

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(trajectory[:, 0], trajectory[:, 1], linewidth=2)
    axes[0].scatter([config.food_position[0]], [config.food_position[1]], c="gold", s=120, marker="*", label="food")
    axes[0].set_title("Trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].axis("equal")

    axes[1].plot(sensory[:, 0], label="left_taste")
    axes[1].plot(sensory[:, 1], label="right_taste")
    axes[1].plot(sensory[:, 2], label="dust")
    axes[1].plot(sensory[:, 3], label="near_food")
    axes[1].set_title("Sensory inputs")
    axes[1].legend()

    axes[2].plot(actions[:, 0], label="forward")
    axes[2].plot(actions[:, 1], label="turn")
    axes[2].plot(actions[:, 2], label="groom")
    axes[2].plot(actions[:, 3], label="eat")
    axes[2].set_title("Decoded actions")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


NEURON_NAMES = [
    "left_sensor", "right_sensor", "dust_sensor", "near_food",
    "turn_left", "turn_right", "forward", "eat", "groom",
]


def save_raster_figure(result: SimulationResult, config: SimConfig, output_path: str | Path) -> None:
    """Two-panel raster figure: spike raster (top) + motor firing rates (bottom)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spikes = result.spike_history  # (T_brain, 9)
    T_brain = spikes.shape[0]
    t_brain = np.arange(T_brain) * config.dt_ms  # ms

    fig, (ax_raster, ax_rate) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    # --- Raster plot ---
    for nidx in range(spikes.shape[1]):
        spike_times = t_brain[spikes[:, nidx] > 0]
        ax_raster.scatter(
            spike_times,
            np.full_like(spike_times, nidx),
            s=6, color="black", linewidths=0,
        )
    ax_raster.set_yticks(range(len(NEURON_NAMES)))
    ax_raster.set_yticklabels(NEURON_NAMES, fontsize=8)
    ax_raster.set_ylabel("Neuron")
    ax_raster.set_xlim(0, T_brain * config.dt_ms)
    ax_raster.set_title("Spike raster")
    ax_raster.invert_yaxis()

    # --- Motor firing rates (arena time resolution) ---
    rate = result.rate_history  # (steps, 9)
    T_arena = rate.shape[0]
    t_arena = np.arange(T_arena) * config.brain_body_sync_ms  # approx arena time in ms
    motor_names = ["turn_left", "turn_right", "forward", "eat", "groom"]
    motor_indices = [4, 5, 6, 7, 8]
    for name, idx in zip(motor_names, motor_indices):
        ax_rate.plot(t_arena, rate[:, idx], label=name)
    ax_rate.set_xlabel("Time (ms)")
    ax_rate.set_ylabel("Firing rate (Hz)")
    ax_rate.set_title("Motor neuron firing rates")
    ax_rate.legend(fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)