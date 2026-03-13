# EmbodiedSNNPrototype

EmbodiedSNNPrototype is a research-oriented embodied spiking codebase designed for studying the loop:

environment -> sensory encoding -> LIF network -> motor readout -> environment

It is not a fruit-fly emulator. It is a controlled scaffold that keeps core ideas from recent embodied-fly and embodied intelligence work:

- connect or wire structure matters
- the body closes the loop
- low-dimensional motor interfaces are practical
- spiking dynamics can be studied before large-scale backpropagation
- utility-efficiency tradeoffs can be benchmarked with small, repeatable studies

## What is included

- a compact embodied arena with food gradient and dust accumulation
- a structured LIF brain with hand-designed recurrent connectivity
- alternative connectivity modes for ablation (`structured`, `structured_rewired`, `random_sparse`, `no_recurrence`, `structured_plastic`)
- a closed-loop simulation runner
- a connectivity benchmark pipeline that exports CSV + plots + Markdown report
- a reading list focused on SNN, connectomics, and embodied control
- a prototype roadmap for extending this into a more serious research platform

## Project layout

- `configs/quick.yaml`: default simulation configuration
- `configs/benchmark.yaml`: multi-seed connectivity benchmark configuration
- `docs/reading_list.md`: annotated reading list
- `docs/foundations_and_taste.md`: beginner-to-founder knowledge map and taste framework
- `docs/twelve_month_plan.md`: one-year training plan from fundamentals to architecture judgment
- `docs/ari_taste_standards.md`: technical taste standards for Alive, Realize, and Innocence
- `docs/twelve_week_plan.md`: first 12 weeks broken down into weekly execution
- `docs/ari_architecture_drafts.md`: first-principles architecture drafts for Alive, Realize, and Innocence
- `docs/ari_founder_handbook.tex`: LaTeX source for a mobile-friendly PDF handbook
- `docs/ari_founder_book.tex`: expanded founder-book edition with diagrams and richer tablet-friendly layout
- `docs/reports/`: location for curated run reports
- `docs/prototype_roadmap.md`: staged research roadmap
- `scripts/run_demo.py`: entry point for a short demo run
- `scripts/run_benchmark.py`: one-command benchmark and report generation
- `src/embodied_snn_prototype/`: reusable package code
- `tests/test_smoke.py`: smoke validation
- `tests/test_analysis.py`: study-pipeline validation

## Quick start

```bash
cd /home/yanshi/projects/EmbodiedSNNPrototype
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py --config configs/quick.yaml
python scripts/run_benchmark.py --config configs/benchmark.yaml
pytest -q
```

The demo prints key metrics and writes a trajectory figure to `outputs/trajectory.png`.

The benchmark run writes:

- `outputs/benchmark/benchmark_runs.csv`
- `outputs/benchmark/benchmark_summary.csv`
- `outputs/benchmark/benchmark_report.md`
- `outputs/benchmark/objective_by_mode.png`
- `outputs/benchmark/objective_faceted_by_mode.png`
- `outputs/benchmark/food_vs_energy.png`

The default benchmark now performs a small stability sweep over:

- `neural_noise_std`
- `rewiring_prob`
- `hunger_drive`
- `plasticity_lr`
- `plasticity_decay`

## Core metrics

Each run reports a compact set of metrics for utility, behavior, and efficiency proxy:

- `food_eaten`
- `final_dust`
- `near_food_ratio`
- `mean_forward`
- `mean_abs_turn`
- `mean_groom`
- `spike_rate_mean`
- `energy_proxy`
- `objective`

Default objective definition:

`objective = food + 0.35 * near_food - 0.25 * dust - energy_proxy`

## Why this prototype is useful

This scaffold gives you a place to test several questions that matter for your SNN work:

- how much behavior can be recovered from sparse wiring plus embodiment
- how LIF timescales interact with sensorimotor loop timing
- when hand-designed readout mappings stop being sufficient
- how to replace fixed motor mappings with learned or adaptive interfaces
- whether structured connectivity gives better utility-efficiency balance than random sparse wiring

## Founder note

If your aim is not only to run experiments but to build long-horizon technical judgment, start with `docs/foundations_and_taste.md`. That document reframes the reading list as a system for building taste: what to study first, what to ignore, and how to tell whether a research direction is deep or merely fashionable.

For a more concrete execution path, use `docs/twelve_week_plan.md` alongside `docs/twelve_month_plan.md`. For founder-level system design, read `docs/ari_architecture_drafts.md`. A compiled PDF version of the core material can be generated from `docs/ari_founder_handbook.tex`.

For a more polished reading experience on tablet, use `docs/ari_founder_book.tex` and the corresponding PDF build script.

## Immediate extension ideas

1. Replace the hand-written brain connectivity with a sparse connectome-derived adjacency matrix.
2. Add STDP or reward-modulated plasticity to selected synapses.
3. Swap the 2D arena for MuJoCo and use a low-dimensional leg or wheel body.
4. Replace the fixed action decoder with a learned descending-neuron interface.

## Current maturity target

The repository now supports a MedSparseSNN-style minimum research loop:

1. configurable simulation
2. multi-seed experiment execution
3. structured run-level CSV
4. aggregate statistics
5. auto-generated plots and Markdown report
6. test coverage for smoke and analysis pipeline

Next maturity step is adding plasticity ablation and optional real embodied backend integration.

Current maturity step adds a minimal reward-modulated plasticity baseline (`structured_plastic`) and a stability sweep over noise and rewiring.