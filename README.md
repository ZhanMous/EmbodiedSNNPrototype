# EmbodiedSNNPrototype

This project is a minimal embodied spiking prototype designed for studying the loop:

environment -> sensory encoding -> LIF network -> motor readout -> environment

It is not a fruit-fly emulator. It is a controlled research scaffold that keeps the core ideas from the recent embodied-fly work:

- connect or wire structure matters
- the body closes the loop
- low-dimensional motor interfaces are practical
- spiking dynamics can be studied without large-scale backpropagation

## What is included

- a compact embodied arena with food gradient and dust accumulation
- a structured LIF brain with hand-designed recurrent connectivity
- a closed-loop simulation runner
- a reading list focused on SNN, connectomics, and embodied control
- a prototype roadmap for extending this into a more serious research platform

## Project layout

- `configs/quick.yaml`: default simulation configuration
- `configs/edl_benchmark.yaml`: EDL framework benchmark configuration
- `docs/reading_list.md`: annotated reading list
- `docs/foundations_and_taste.md`: beginner-to-founder knowledge map and taste framework
- `docs/twelve_month_plan.md`: one-year training plan from fundamentals to architecture judgment
- `docs/ari_taste_standards.md`: technical taste standards for Alive, Realize, and Innocence
- `docs/twelve_week_plan.md`: first 12 weeks broken down into weekly execution
- `docs/ari_architecture_drafts.md`: first-principles architecture drafts for Alive, Realize, and Innocence
- `docs/ari_founder_handbook.tex`: LaTeX source for a mobile-friendly PDF handbook
- `docs/ari_founder_book.tex`: expanded founder-book edition with diagrams and richer tablet-friendly layout
- `docs/prototype_roadmap.md`: staged research roadmap
- `docs/EDL_QUICK_REFERENCE.md`: **EDL framework quick start (演化-发育-学习)**
- `docs/EDL_INTEGRATION_CHECKLIST.md`: **EDL integration guide for downstream projects**
- `docs/EDL_COMPLETION_REPORT.md`: **EDL framework full report**
- `scripts/run_demo.py`: entry point for a short demo run
- `scripts/run_edl_benchmark.py`: **EDL benchmark runner**
- `scripts/generate_edl_analysis.py`: **EDL performance analysis report generator**
- `src/embodied_snn_prototype/`: reusable package code
- `src/embodied_snn_prototype/edl.py`: **EDL framework (532 lines, 4 optimization methods)**
- `tests/test_smoke.py`: minimal smoke test

## Quick start

```bash
cd /home/yanshi/projects/EmbodiedSNNPrototype
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py --config configs/quick.yaml
pytest -q
```

The demo prints summary metrics and writes a trajectory figure to `outputs/trajectory.png`.

## EDL Framework - Unified Evolution-Development-Learning optimization

**New in v1.1**: EDL framework for comparing 4 SNN optimization strategies:

```bash
# Run EDL benchmark (4 methods, 3 trials, ~1 minute)
python scripts/run_edl_benchmark.py --config configs/edl_benchmark.yaml

# Analyze results  
python scripts/generate_edl_analysis.py
# Output: outputs/edl_analysis/{report.md, report.json}
```

**Results** (embodied control task):
- baseline (plasticity only): 21.77 ± 4.02
- direct_evolution (GA readout): 82.07 ± 12.69 (+277%)
- **crn_development (GA + developmental coding): 91.99 ± 0.00 (+323%)** ✓

**Quick integration** (3 steps to add EDL to your project):
1. Copy `src/embodied_snn_prototype/edl.py` to your project
2. Define a `rollout_fn(network, genome, seed)` for your task
3. Call `run_edl_benchmark(rollout_fn, config)` 

**Documentation**:
- [EDL Quick Reference](docs/EDL_QUICK_REFERENCE.md) - 1-page cheat sheet
- [EDL Integration Guide](docs/EDL_INTEGRATION_CHECKLIST.md) - full integration steps for Neuro-Symbiosis, PrivEnergyBench, MedSparseSNN
- [EDL Completion Report](docs/EDL_COMPLETION_REPORT.md) - project summary, meta-insights, roadmap
- [Neuro-Symbiosis Example](docs/neuro_symbiosis_edl_adapter.py) - sample BCI adaptation

## Why this prototype is useful

This scaffold gives you a place to test several questions that matter for your SNN work:

- how much behavior can be recovered from sparse wiring plus embodiment
- how LIF timescales interact with sensorimotor loop timing
- when hand-designed readout mappings stop being sufficient
- how to replace fixed motor mappings with learned or adaptive interfaces
- **[NEW] how evolutionary search, developmental encoding, and plasticity interact across timescales**

## Founder note

If your aim is not only to run experiments but to build long-horizon technical judgment, start with `docs/foundations_and_taste.md`. That document reframes the reading list as a system for building taste: what to study first, what to ignore, and how to tell whether a research direction is deep or merely fashionable.

For a more concrete execution path, use `docs/twelve_week_plan.md` alongside `docs/twelve_month_plan.md`. For founder-level system design, read `docs/ari_architecture_drafts.md`. A compiled PDF version of the core material can be generated from `docs/ari_founder_handbook.tex`.

For a more polished reading experience on tablet, use `docs/ari_founder_book.tex` and the corresponding PDF build script.

## Immediate extension ideas

1. Replace the hand-written brain connectivity with a sparse connectome-derived adjacency matrix.
2. Add STDP or reward-modulated plasticity to selected synapses.
3. Swap the 2D arena for MuJoCo and use a low-dimensional leg or wheel body.
4. Replace the fixed action decoder with a learned descending-neuron interface.