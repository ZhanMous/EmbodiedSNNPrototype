from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Callable

import numpy as np

from .config import SimConfig
from .environment import EmbodiedArena
from .network import StructuredLIFBrain


@dataclass
class EvolutionConfig:
    population_size: int = 20
    generations: int = 15
    mutation_rate: float = 0.18
    mutation_scale: float = 0.08
    crossover_rate: float = 0.75
    elite_count: int = 2
    tournament_size: int = 3
    seed: int = 42


@dataclass
class EDLBenchmarkConfig:
    train_episodes: int = 4
    eval_episodes: int = 6
    train_seed_stride: int = 10
    eval_seed_offset: int = 1000


@dataclass
class MethodRun:
    method: str
    seed: int
    objective: float
    food_eaten: float
    near_food_ratio: float
    final_dust: float
    energy_proxy: float
    mean_reward: float


def _rollout_episode(
    config: SimConfig,
    episode_seed: int,
    readout_weights: np.ndarray | None,
    connectivity_mode: str,
    plasticity_lr: float,
    plasticity_decay: float,
) -> dict[str, float]:
    cfg = SimConfig(**asdict(config))
    cfg.seed = int(episode_seed)
    cfg.connectivity_mode = connectivity_mode
    cfg.plasticity_lr = float(plasticity_lr)
    cfg.plasticity_decay = float(plasticity_decay)

    arena = EmbodiedArena(cfg)
    brain = StructuredLIFBrain(cfg)
    arena.reset()

    if readout_weights is not None:
        clipped = np.clip(readout_weights, 0.0, cfg.readout_weight_max)
        brain.base_readout_weights[:, :] = clipped
        brain.readout_delta[:, :] = 0.0
        brain.readout_weights[:, :] = clipped
        brain.readout_history = [brain.readout_weights.copy()]

    sync_steps = max(1, int(round(cfg.brain_body_sync_ms / cfg.dt_ms)))
    sensory_hist: list[np.ndarray] = []
    action_hist: list[np.ndarray] = []
    rate_hist: list[np.ndarray] = []
    reward_hist: list[float] = []

    for _ in range(cfg.steps):
        sensory = arena.sense()
        for _ in range(sync_steps):
            brain.step(sensory)

        action = brain.decode_action()
        prev_dust = arena.state.dust
        prev_food_eaten = arena.state.food_eaten
        state = arena.step(
            linear_drive=action["forward"],
            turn_drive=action["turn"],
            groom_drive=action["groom"],
            eat_drive=action["eat"],
        )
        reward = brain.apply_reward_modulated_plasticity(
            sensory=sensory,
            action=action,
            delta_food=state.food_eaten - prev_food_eaten,
            delta_dust=state.dust - prev_dust,
        )

        sensory_hist.append(sensory.copy())
        action_hist.append(np.array([action["forward"], action["turn"], action["groom"], action["eat"]], dtype=float))
        rate_hist.append(brain.rate_trace.copy())
        reward_hist.append(float(reward))

    sensory_arr = np.asarray(sensory_hist, dtype=float)
    action_arr = np.asarray(action_hist, dtype=float)
    rate_arr = np.asarray(rate_hist, dtype=float)

    near_food_ratio = float(np.mean(sensory_arr[:, 3])) if sensory_arr.size else 0.0
    mean_forward = float(np.mean(action_arr[:, 0])) if action_arr.size else 0.0
    mean_abs_turn = float(np.mean(np.abs(action_arr[:, 1]))) if action_arr.size else 0.0
    mean_groom = float(np.mean(action_arr[:, 2])) if action_arr.size else 0.0
    spike_rate_mean = float(np.mean(rate_arr)) if rate_arr.size else 0.0

    action_cost = 0.30 * mean_forward + 0.25 * mean_abs_turn + 0.20 * mean_groom
    spike_cost = 0.0012 * spike_rate_mean
    energy_proxy = float(action_cost + spike_cost)

    objective = float(
        arena.state.food_eaten
        + 0.35 * near_food_ratio
        - 0.25 * arena.state.dust
        - energy_proxy
    )

    return {
        "objective": objective,
        "food_eaten": float(arena.state.food_eaten),
        "near_food_ratio": near_food_ratio,
        "final_dust": float(arena.state.dust),
        "energy_proxy": energy_proxy,
        "mean_reward": float(np.mean(reward_hist) if reward_hist else 0.0),
    }


def _evaluate_candidate(
    config: SimConfig,
    episode_seeds: list[int],
    readout_weights: np.ndarray | None,
    connectivity_mode: str,
    plasticity_lr: float,
    plasticity_decay: float,
) -> dict[str, float]:
    runs = [
        _rollout_episode(
            config=config,
            episode_seed=seed,
            readout_weights=readout_weights,
            connectivity_mode=connectivity_mode,
            plasticity_lr=plasticity_lr,
            plasticity_decay=plasticity_decay,
        )
        for seed in episode_seeds
    ]

    return {
        "objective": float(mean(r["objective"] for r in runs)),
        "food_eaten": float(mean(r["food_eaten"] for r in runs)),
        "near_food_ratio": float(mean(r["near_food_ratio"] for r in runs)),
        "final_dust": float(mean(r["final_dust"] for r in runs)),
        "energy_proxy": float(mean(r["energy_proxy"] for r in runs)),
        "mean_reward": float(mean(r["mean_reward"] for r in runs)),
    }


def _tournament_select(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    k: int,
) -> np.ndarray:
    idx = rng.choice(len(population), size=k, replace=False)
    best = idx[np.argmax(fitness[idx])]
    return population[best].copy()


def _evolve(
    genome_size: int,
    bounds: tuple[np.ndarray, np.ndarray],
    eval_fn: Callable[[np.ndarray], float],
    config: EvolutionConfig,
) -> tuple[np.ndarray, float, list[float]]:
    rng = np.random.default_rng(config.seed)
    low, high = bounds
    pop = rng.uniform(low, high, size=(config.population_size, genome_size))
    best_history: list[float] = []

    for _ in range(config.generations):
        fit = np.array([eval_fn(ind) for ind in pop], dtype=float)
        order = np.argsort(-fit)
        pop = pop[order]
        fit = fit[order]
        best_history.append(float(fit[0]))

        next_pop = [pop[i].copy() for i in range(config.elite_count)]
        while len(next_pop) < config.population_size:
            p1 = _tournament_select(pop, fit, rng, config.tournament_size)
            p2 = _tournament_select(pop, fit, rng, config.tournament_size)

            if rng.random() < config.crossover_rate:
                alpha = rng.random(genome_size)
                child = alpha * p1 + (1.0 - alpha) * p2
            else:
                child = p1.copy()

            mut_mask = rng.random(genome_size) < config.mutation_rate
            child[mut_mask] += rng.normal(0.0, config.mutation_scale, size=np.sum(mut_mask))
            child = np.clip(child, low, high)
            next_pop.append(child)

        pop = np.asarray(next_pop, dtype=float)

    final_fit = np.array([eval_fn(ind) for ind in pop], dtype=float)
    best_idx = int(np.argmax(final_fit))
    return pop[best_idx].copy(), float(final_fit[best_idx]), best_history


def _decode_crn_genome(genome: np.ndarray, readout_weight_max: float) -> np.ndarray:
    latent = genome[:8]
    dyn = genome[8:8 + 8 * 8].reshape(8, 8)
    bias = genome[8 + 8 * 8:8 + 8 * 8 + 8]
    proj = genome[8 + 8 * 8 + 8:].reshape(25, 8)

    state = latent.copy()
    for _ in range(4):
        state = np.tanh(dyn @ state + bias)

    raw = proj @ state
    weights = 1.0 / (1.0 + np.exp(-raw))
    return (weights.reshape(5, 5) * readout_weight_max).astype(float)


def _run_method_ea(
    config: SimConfig,
    base_seed: int,
    evo_cfg: EvolutionConfig,
    bench_cfg: EDLBenchmarkConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    train_seeds = [base_seed + i * bench_cfg.train_seed_stride for i in range(bench_cfg.train_episodes)]
    eval_seeds = [base_seed + bench_cfg.eval_seed_offset + i * bench_cfg.train_seed_stride for i in range(bench_cfg.eval_episodes)]

    low = np.full(25, 0.0, dtype=float)
    high = np.full(25, config.readout_weight_max, dtype=float)

    def eval_fn(genome: np.ndarray) -> float:
        w = genome.reshape(5, 5)
        metrics = _evaluate_candidate(config, train_seeds, w, "structured", config.plasticity_lr, config.plasticity_decay)
        return metrics["objective"]

    best_genome, best_fit, history = _evolve(25, (low, high), eval_fn, evo_cfg)
    final_train = _evaluate_candidate(config, train_seeds, best_genome.reshape(5, 5), "structured", config.plasticity_lr, config.plasticity_decay)
    final_eval = _evaluate_candidate(config, eval_seeds, best_genome.reshape(5, 5), "structured", config.plasticity_lr, config.plasticity_decay)
    meta = {
        "train_objective": final_train["objective"],
        "search_best_objective": best_fit,
        "history_last": history[-1] if history else best_fit,
    }
    return final_eval, meta


def _run_method_crn(
    config: SimConfig,
    base_seed: int,
    evo_cfg: EvolutionConfig,
    bench_cfg: EDLBenchmarkConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    train_seeds = [base_seed + i * bench_cfg.train_seed_stride for i in range(bench_cfg.train_episodes)]
    eval_seeds = [base_seed + bench_cfg.eval_seed_offset + i * bench_cfg.train_seed_stride for i in range(bench_cfg.eval_episodes)]

    genome_size = 8 + 8 * 8 + 8 + 25 * 8
    low = np.full(genome_size, -1.5, dtype=float)
    high = np.full(genome_size, 1.5, dtype=float)

    def eval_fn(genome: np.ndarray) -> float:
        w = _decode_crn_genome(genome, config.readout_weight_max)
        metrics = _evaluate_candidate(config, train_seeds, w, "structured", config.plasticity_lr, config.plasticity_decay)
        return metrics["objective"]

    best_genome, best_fit, history = _evolve(genome_size, (low, high), eval_fn, evo_cfg)
    best_weights = _decode_crn_genome(best_genome, config.readout_weight_max)
    final_train = _evaluate_candidate(config, train_seeds, best_weights, "structured", config.plasticity_lr, config.plasticity_decay)
    final_eval = _evaluate_candidate(config, eval_seeds, best_weights, "structured", config.plasticity_lr, config.plasticity_decay)
    meta = {
        "train_objective": final_train["objective"],
        "search_best_objective": best_fit,
        "history_last": history[-1] if history else best_fit,
    }
    return final_eval, meta


def _run_method_evo_learn(
    config: SimConfig,
    base_seed: int,
    evo_cfg: EvolutionConfig,
    bench_cfg: EDLBenchmarkConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    train_seeds = [base_seed + i * bench_cfg.train_seed_stride for i in range(bench_cfg.train_episodes)]
    eval_seeds = [base_seed + bench_cfg.eval_seed_offset + i * bench_cfg.train_seed_stride for i in range(bench_cfg.eval_episodes)]

    genome_size = 27
    low = np.concatenate([np.zeros(25, dtype=float), np.array([0.0005, 0.001], dtype=float)])
    high = np.concatenate([
        np.full(25, config.readout_weight_max, dtype=float),
        np.array([0.03, 0.06], dtype=float),
    ])

    def eval_fn(genome: np.ndarray) -> float:
        w = genome[:25].reshape(5, 5)
        lr = float(genome[25])
        decay = float(genome[26])
        metrics = _evaluate_candidate(config, train_seeds, w, "structured_plastic", lr, decay)
        return metrics["objective"]

    best_genome, best_fit, history = _evolve(genome_size, (low, high), eval_fn, evo_cfg)
    best_w = best_genome[:25].reshape(5, 5)
    best_lr = float(best_genome[25])
    best_decay = float(best_genome[26])
    final_train = _evaluate_candidate(config, train_seeds, best_w, "structured_plastic", best_lr, best_decay)
    final_eval = _evaluate_candidate(config, eval_seeds, best_w, "structured_plastic", best_lr, best_decay)
    meta = {
        "train_objective": final_train["objective"],
        "search_best_objective": best_fit,
        "history_last": history[-1] if history else best_fit,
        "best_lr": best_lr,
        "best_decay": best_decay,
    }
    return final_eval, meta


def _run_method_plastic_only(
    config: SimConfig,
    base_seed: int,
    bench_cfg: EDLBenchmarkConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    eval_seeds = [base_seed + bench_cfg.eval_seed_offset + i * bench_cfg.train_seed_stride for i in range(bench_cfg.eval_episodes)]
    final_eval = _evaluate_candidate(config, eval_seeds, None, "structured_plastic", config.plasticity_lr, config.plasticity_decay)
    return final_eval, {
        "train_objective": final_eval["objective"],
        "search_best_objective": final_eval["objective"],
        "history_last": final_eval["objective"],
    }


def run_edl_benchmark(
    base_config: SimConfig,
    seeds: list[int],
    output_dir: str | Path,
    evo_cfg: EvolutionConfig | None = None,
    bench_cfg: EDLBenchmarkConfig | None = None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | str]]]:
    evo_cfg = evo_cfg or EvolutionConfig()
    bench_cfg = bench_cfg or EDLBenchmarkConfig()

    runs: list[dict[str, float | int | str]] = []
    methods = ["plastic_only", "ea_readout", "crn_development", "evo_learning"]

    for seed in seeds:
        local_evo = EvolutionConfig(**asdict(evo_cfg))
        local_evo.seed = int(seed * 101 + evo_cfg.seed)

        m0, meta0 = _run_method_plastic_only(base_config, int(seed), bench_cfg)
        m1, meta1 = _run_method_ea(base_config, int(seed), local_evo, bench_cfg)
        m2, meta2 = _run_method_crn(base_config, int(seed), local_evo, bench_cfg)
        m3, meta3 = _run_method_evo_learn(base_config, int(seed), local_evo, bench_cfg)

        for method, metrics, meta in [
            (methods[0], m0, meta0),
            (methods[1], m1, meta1),
            (methods[2], m2, meta2),
            (methods[3], m3, meta3),
        ]:
            row: dict[str, float | int | str] = {
                "method": method,
                "seed": int(seed),
                "objective": float(metrics["objective"]),
                "food_eaten": float(metrics["food_eaten"]),
                "near_food_ratio": float(metrics["near_food_ratio"]),
                "final_dust": float(metrics["final_dust"]),
                "energy_proxy": float(metrics["energy_proxy"]),
                "mean_reward": float(metrics["mean_reward"]),
                "train_objective": float(meta["train_objective"]),
                "search_best_objective": float(meta["search_best_objective"]),
            }
            if "best_lr" in meta:
                row["best_lr"] = float(meta["best_lr"])
            if "best_decay" in meta:
                row["best_decay"] = float(meta["best_decay"])
            runs.append(row)

    summary = _aggregate_edl_runs(runs)
    _write_edl_outputs(output_dir, base_config, evo_cfg, bench_cfg, runs, summary)
    return runs, summary


def _aggregate_edl_runs(runs: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    methods = sorted({str(r["method"]) for r in runs})
    summary: list[dict[str, float | str]] = []

    for method in methods:
        rows = [r for r in runs if str(r["method"]) == method]
        objective_vals = [float(r["objective"]) for r in rows]
        food_vals = [float(r["food_eaten"]) for r in rows]
        dust_vals = [float(r["final_dust"]) for r in rows]
        energy_vals = [float(r["energy_proxy"]) for r in rows]

        row: dict[str, float | str] = {
            "method": method,
            "n": float(len(rows)),
            "objective_mean": float(mean(objective_vals)),
            "objective_std": float(stdev(objective_vals) if len(objective_vals) > 1 else 0.0),
            "food_eaten_mean": float(mean(food_vals)),
            "final_dust_mean": float(mean(dust_vals)),
            "energy_proxy_mean": float(mean(energy_vals)),
        }
        summary.append(row)

    summary.sort(key=lambda x: float(x["objective_mean"]), reverse=True)
    return summary


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_edl_outputs(
    output_dir: str | Path,
    base_config: SimConfig,
    evo_cfg: EvolutionConfig,
    bench_cfg: EDLBenchmarkConfig,
    runs: list[dict[str, float | int | str]],
    summary: list[dict[str, float | str]],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(out / "edl_runs.csv", runs)
    _write_csv(out / "edl_summary.csv", summary)

    report = {
        "base_config": asdict(base_config),
        "evolution_config": asdict(evo_cfg),
        "benchmark_config": asdict(bench_cfg),
        "summary": summary,
        "recommendation": _build_recommendation(summary),
    }
    (out / "edl_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# EDL Benchmark Report")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| method | n | objective_mean | objective_std | food_eaten_mean | final_dust_mean | energy_proxy_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary:
        lines.append(
            "| {method} | {n:.0f} | {objective_mean:.4f} | {objective_std:.4f} | {food_eaten_mean:.4f} | {final_dust_mean:.4f} | {energy_proxy_mean:.4f} |".format(
                **{k: float(v) if isinstance(v, (float, int)) else v for k, v in row.items()}
            )
        )

    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.extend([f"- {item}" for item in _build_recommendation(summary)])
    (out / "edl_report.md").write_text("\n".join(lines), encoding="utf-8")


def _build_recommendation(summary: list[dict[str, float | str]]) -> list[str]:
    if not summary:
        return ["No valid run available."]

    best = summary[0]
    method = str(best["method"])
    recommendations = [
        f"Top method in this run: {method} (objective_mean={float(best['objective_mean']):.4f}).",
        "Use objective_mean as primary ranking, and keep food_eaten_mean + energy_proxy_mean as secondary constraints.",
    ]

    has_crn = any(str(r["method"]) == "crn_development" for r in summary)
    has_ea = any(str(r["method"]) == "ea_readout" for r in summary)
    has_evo_learn = any(str(r["method"]) == "evo_learning" for r in summary)

    if has_crn and has_ea:
        recommendations.append("CRN beats plain EA when objective_mean is higher at similar energy_proxy_mean; this suggests developmental encoding helps search regularization.")
    if has_evo_learn:
        recommendations.append("If evo_learning is top-1 or top-2, prioritize evolution for initialization + local plasticity for adaptation in downstream projects.")

    recommendations.append("For cross-project transfer, keep the optimizer unchanged and only replace the rollout objective adapter.")
    return recommendations
