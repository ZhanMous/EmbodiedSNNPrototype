# EDL Benchmark Report

## Ranking

| method | n | objective_mean | objective_std | food_eaten_mean | final_dust_mean | energy_proxy_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| plastic_only | 2 | -0.4210 | 0.0000 | 0.3600 | 0.0480 | 0.7977 |
| evo_learning | 2 | -0.4812 | 0.0039 | 0.0000 | 0.0030 | 0.4805 |
| ea_readout | 2 | -0.4839 | 0.0001 | 0.0000 | 0.0030 | 0.4832 |
| crn_development | 2 | -0.5635 | 0.0159 | 0.0000 | 0.0030 | 0.5627 |
| surrogate_backprop | 2 | -0.6911 | 0.1197 | 0.0800 | 0.0360 | 0.7685 |

## Recommendation

- Top method in this run: plastic_only (objective_mean=-0.4210).
- Use objective_mean as primary ranking, and keep food_eaten_mean + energy_proxy_mean as secondary constraints.
- CRN beats plain EA when objective_mean is higher at similar energy_proxy_mean; this suggests developmental encoding helps search regularization.
- If evo_learning is top-1 or top-2, prioritize evolution for initialization + local plasticity for adaptation in downstream projects.
- For cross-project transfer, keep the optimizer unchanged and only replace the rollout objective adapter.