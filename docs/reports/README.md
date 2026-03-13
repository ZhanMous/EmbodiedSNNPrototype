# Reports Directory

This folder stores reproducible research summaries generated from benchmark runs.

Recommended workflow:

1. Run benchmark:

```bash
python scripts/run_benchmark.py --config configs/benchmark.yaml
```

2. Copy key outputs from `outputs/benchmark/` into this folder for versioned reporting:

- benchmark_report.md
- benchmark_summary.csv
- objective_by_mode.png
- food_vs_energy.png

3. Add a dated note like `2026-03-13_connectivity_study.md` describing:

- hypothesis
- run config
- ranking and interpretation
- next actions
