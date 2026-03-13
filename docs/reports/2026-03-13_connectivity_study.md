# 2026-03-13 Connectivity Study

## Setup

- Config: `configs/benchmark.yaml`
- Seeds: 7, 11, 19, 23, 29
- Modes: structured, structured_rewired, random_sparse, no_recurrence, structured_plastic
- Neural noise sweep: 0.00, 0.02
- Rewiring sweep: 0.05, 0.15
- Steps per run: 300

## Artifacts

- `outputs/benchmark/benchmark_runs.csv`
- `outputs/benchmark/benchmark_summary.csv`
- `outputs/benchmark/benchmark_report.md`
- `outputs/benchmark/objective_by_mode.png`
- `outputs/benchmark/food_vs_energy.png`

## Ranking by objective_mean

1. structured_plastic (noise=0.00, rewiring=0.05): 1.2370
2. structured_plastic (noise=0.00, rewiring=0.15): 1.2370
3. structured (noise=0.02, rewiring=0.05): 0.2579
4. structured (noise=0.02, rewiring=0.15): 0.2579
5. structured_plastic (noise=0.02, rewiring=0.05): -0.2139

## Observation

- The added `structured_plastic` baseline dominates at low noise and reaches objective 1.2370.
- Forward drive still stays near 1.0 across most settings, so action saturation is reduced but not yet eliminated.
- `no_recurrence` remains the weakest family; high dust and energy penalties suppress objective.
- Rewiring sweep has limited effect for non-rewired modes (as expected), and mild effect for `structured_rewired`.

## Immediate next actions

1. Add hunger sweep (e.g. 0.28/0.35/0.43) to break the remaining forward saturation.
2. Add plasticity ablation over `plasticity_lr` and `plasticity_decay` for stability envelopes.
3. Split summary plots into per-mode panels to improve readability under grid sweeps.
