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
    spike_rate_std: float
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
        food_eaten=float(arena.state.food_eaten),
        final_dust=float(arena.state.dust),
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


def summarize_episode(result: SimulationResult, config: SimConfig) -> EpisodeMetrics:
    actions = result.action_history
    sensory = result.sensory_history
    rates = result.rate_history

    near_food_ratio = float(np.mean(sensory[:, 3]))
    mean_forward = float(np.mean(actions[:, 0]))
    mean_abs_turn = float(np.mean(np.abs(actions[:, 1])))
    mean_groom = float(np.mean(actions[:, 2]))
    spike_rate_mean = float(np.mean(rates))
    spike_rate_std = float(np.std(rates))

    energy_proxy = (
        config.spike_energy_cost * spike_rate_mean
        + config.linear_energy_cost * mean_forward
        + config.turn_energy_cost * mean_abs_turn
        + config.groom_energy_cost * mean_groom
    )

    objective = (
        result.food_eaten
        + 0.35 * near_food_ratio
        - 0.25 * result.final_dust
        - energy_proxy
    )

    return EpisodeMetrics(
        food_eaten=result.food_eaten,
        final_dust=result.final_dust,
        near_food_ratio=near_food_ratio,
        mean_forward=mean_forward,
        mean_abs_turn=mean_abs_turn,
        mean_groom=mean_groom,
        spike_rate_mean=spike_rate_mean,
        spike_rate_std=spike_rate_std,
        energy_proxy=float(energy_proxy),
        objective=float(objective),
    )