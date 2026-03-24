# EDL Transfer Guide

This guide explains how to reuse the new EDL benchmark in other projects.

## Scope

The benchmark compares four optimization strategies for SNN control:

- plastic_only: local reward-modulated plasticity only
- ea_readout: direct evolution on readout weights
- crn_development: compressed genotype decoded by a CRN-style developmental process
- evo_learning: evolution for initialization + local plasticity adaptation

## Key Files

- src/embodied_snn_prototype/edl.py
- scripts/run_edl_benchmark.py
- configs/edl_benchmark.yaml

## Output Files

Each run writes to outputs/edl_benchmark by default:

- edl_runs.csv: seed-level metrics
- edl_summary.csv: mean and std grouped by method
- edl_report.md: concise ranking and recommendation
- edl_report.json: full machine-readable report

## Migration Pattern

To use this in another project, keep the optimizer and replace only rollout/evaluation hooks.

1. Keep the evolution core:
- _evolve
- _tournament_select

2. Replace environment-specific rollout:
- _rollout_episode

3. Keep objective structure but map to your task metrics:
- objective = utility - risk - cost

Recommended mapping:

- utility: task score, reward, or accuracy
- risk: privacy leakage, instability, unsafe behavior
- cost: energy, latency, action effort

4. Keep method definitions:
- _run_method_ea
- _run_method_crn
- _run_method_evo_learn

Only adjust genome decode and rollout details.

## Suggested Next Integrations

- Neuro-Symbiosis: map utility/privacy/energy to BCI decoding, MIA risk, and inference cost.
- PrivEnergyBench: use EDL methods as additional model families in benchmark matrix.
- MedSparseSNN: test CRN initialization for sparse SNN layers under low-MAC constraints.

## Practical Notes

- Start with low generations and small population for debugging.
- Increase seeds first, then increase evolution budget.
- Always evaluate with holdout seeds to avoid overfitting search seeds.
