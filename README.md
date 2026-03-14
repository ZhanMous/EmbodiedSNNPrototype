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
- `docs/reading_list.md`: annotated reading list
- `docs/foundations_and_taste.md`: beginner-to-founder knowledge map and taste framework
- `docs/twelve_month_plan.md`: one-year training plan from fundamentals to architecture judgment
- `docs/ari_taste_standards.md`: technical taste standards for Alive, Realize, and Innocence
- `docs/twelve_week_plan.md`: first 12 weeks broken down into weekly execution
- `docs/ari_architecture_drafts.md`: first-principles architecture drafts for Alive, Realize, and Innocence
- `docs/ari_founder_handbook.tex`: LaTeX source for a mobile-friendly PDF handbook
- `docs/ari_founder_book.tex`: expanded founder-book edition with diagrams and richer tablet-friendly layout
- `docs/prototype_roadmap.md`: staged research roadmap
- `scripts/run_demo.py`: entry point for a short demo run
- `src/embodied_snn_prototype/`: reusable package code
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

## Why this prototype is useful

This scaffold gives you a place to test several questions that matter for your SNN work:

- how much behavior can be recovered from sparse wiring plus embodiment
- how LIF timescales interact with sensorimotor loop timing
- when hand-designed readout mappings stop being sufficient
- how to replace fixed motor mappings with learned or adaptive interfaces

## Founder note

If your aim is not only to run experiments but to build long-horizon technical judgment, start with `docs/foundations_and_taste.md`. That document reframes the reading list as a system for building taste: what to study first, what to ignore, and how to tell whether a research direction is deep or merely fashionable.

For a more concrete execution path, use `docs/twelve_week_plan.md` alongside `docs/twelve_month_plan.md`. For founder-level system design, read `docs/ari_architecture_drafts.md`. A compiled PDF version of the core material can be generated from `docs/ari_founder_handbook.tex`.

For a more polished reading experience on tablet, use `docs/ari_founder_book.tex` and the corresponding PDF build script.

## Immediate extension ideas

1. Replace the hand-written brain connectivity with a sparse connectome-derived adjacency matrix.
2. Add STDP or reward-modulated plasticity to selected synapses.
3. Swap the 2D arena for MuJoCo and use a low-dimensional leg or wheel body.
4. Replace the fixed action decoder with a learned descending-neuron interface.