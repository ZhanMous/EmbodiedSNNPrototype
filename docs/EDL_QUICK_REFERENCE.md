# EDL 框架快速参考

## 一句话
**演化-发育-学习** (EDL) 统一框架，对标 SNN 编码搜索的四种优化策略。

## 核心论文要点

```
目标: 在受限计算资源下最优化 SNN 参数
策略: 演化搜索 + 发育编码 (压缩) + 可塑性学习

GA基因型  →  发育展开  →  神经网络表现型  →  评估适应度  →  进化下一代
280D基因  → 4步RNN   → 25D权重/偏置    → reward-energy → 锦标赛选择
```

## 四大方法对标

| # | 方法 | 策略 | 性能 (embodied) | 资源 | 用途 |
|---|------|------|-----------------|------|------|
| 1 | **plastic_only** | 仅可塑性 | 21.77±4.02 | 低 | 对照组/基线 |
| 2 | **ea_readout** | GA优化输出权 | 82.07±12.69 | 中 | 快速适应 |
| 3 | **crn_development** | GA+发育编码 | 91.99±0.00 | 中 | 参数压缩 |
| 4 | **evo_learning** | GA+共学习 | TBD | 中-高 | 长期自适应 |

## 三行部署

```python
# 1. 导入框架
from src.embodied_snn_prototype.edl import run_edl_benchmark

# 2. 定义 rollout 函数 (用你的任务)
def my_rollout(network, genome, seed):
    # ... 评估网络参数
    return {'objective': score, 'metrics': {...}}

# 3. 运行基准
results = run_edl_benchmark(
    rollout_fn=my_rollout,
    config_path='configs/edl_benchmark.yaml'
)
```

## 五步快速集成 (新项目)

### Step 1: 复制核心模块
```bash
cp /home/yanshi/EmbodiedSNNPrototype/src/embodied_snn_prototype/edl.py \
   YourProject/src/your_edl.py
```

### Step 2: 编写网络 rollout
```python
# YourProject/scripts/edl_adapter.py
def rollout_your_task(network, genome, seed):
    """
    Args:
        network: 神经网络对象 (SNN/Transformer/...)
        genome: np.ndarray, shape (280,) - 进化参数
        seed: int - 随机种子
    
    Returns:
        dict with keys: 'objective', other metrics...
    """
    cfg = Config(seed=seed)
    params = decode_genome(genome)      # 270→25 或适配你的维度
    network.set_params(params)
    
    train_loss = train_on_task(network, cfg)
    test_score = evaluate_on_task(network, cfg)
    
    objective = test_score - 0.1*train_loss  # 自定义的目标函数
    
    return {
        'objective': objective,
        'train_loss': train_loss,
        'test_score': test_score,
    }
```

### Step 3: 创建配置
```yaml
# YourProject/configs/edl_your_task.yaml
simulation:
  seed: 42
evolution:
  population_size: 20
  generations: 12
benchmark:
  seeds: [3, 7, 11, 19, 23, 31]
  eval_episodes: 6
output:
  output_dir: outputs/edl_your_task
```

### Step 4: 创建启动脚本
```python
# YourProject/scripts/run_edl_your_task.py
from src.your_edl import run_edl_benchmark
from .edl_adapter import rollout_your_task
import yaml

config = yaml.safe_load(open('configs/edl_your_task.yaml'))
results = run_edl_benchmark(rollout_your_task, config)
print(results['summary'])  # Print summary
```

### Step 5: 运行！
```bash
cd YourProject
python scripts/run_edl_your_task.py
# Output: outputs/edl_your_task/{runs.csv, summary.csv, report.md, report.json}
```

## 文件速查表

| 内容 | 位置 |
|------|------|
| **核心框架** | `EmbodiedSNNPrototype/src/embodied_snn_prototype/edl.py` |
| **CLI 入口** | `EmbodiedSNNPrototype/scripts/run_edl_benchmark.py` |
| **配置模板** | `EmbodiedSNNPrototype/configs/edl_benchmark.yaml` |
| **迁移文档** | `EmbodiedSNNPrototype/docs/edl_transfer_guide.md` |
| **集成清单** | `EmbodiedSNNPrototype/docs/EDL_INTEGRATION_CHECKLIST.md` |
| **完成报告** | `EmbodiedSNNPrototype/docs/EDL_COMPLETION_REPORT.md` |
| **Neuro-Symbiosis 示例** | `EmbodiedSNNPrototype/docs/neuro_symbiosis_edl_adapter.py` |
| **分析报告** | `EmbodiedSNNPrototype/outputs/edl_analysis/edl_analysis_report.md` |

## 性能数据

```
Trial 1: GRN=91.99  Direct-EA=84.79  Baseline=22.46
Trial 2: GRN=91.99  Direct-EA=65.35  Baseline=16.53
Trial 3: Direct-EA=96.07  GRN=91.99  Baseline=26.31

Mean score (平均得分):
  ✅ GRN:         91.99 ± 0.00  (+323% vs baseline)
  ✅ Direct-EA:   82.07 ± 12.69 (+277% vs baseline)
  ❌ Baseline:    21.77 ± 4.02
```

## 常见问题排障

| 问题 | 症状 | 修复 |
|------|------|------|
| **EA 卡顿** | 30s 无进度 | 检查 `_rollout_episode()` 中的 O(T²) 循环；添加 accumulator |
| **方差为 0** | std=0.00 (n=3) | 增加 population_size, generations, 或提高 mutation_rate |
| **内存溢出** | OOM in evolution | 减少 eval_episodes 或增加 train_seed_stride |
| **低可重复性** | 跨 trial 差异大 | 增加 eval_episodes (6→12)，或固定硬件配置 |

## 目标函数设计指南

```python
# Bad: 目标太单一
objective = accuracy

# Better: 多约束权衡
objective = accuracy - 0.1 * energy_cost

# Best: 明确的 Pareto 权衡
objective = accuracy 
      - λ_privacy * membership_inference_risk
      - λ_energy * (mac_ops / budget)
      - λ_sparsity * (1 - sparsity_ratio)
```

## 论文/参考资源

- **CRN 发育**: Bongard & Pfeifer (2003) *Evolving Complete Agents*
- **神经可塑STDP**: Gerstner & Kistler (2002) *Spiking Neuron Models*
- **多目标进化**: Deb (2001) *Multi-Objective Optimization*

## 下一步

- [ ] 运行 EmbodiedSNNPrototype 基准 (`python scripts/run_edl_benchmark.py`)
- [ ] 集成到 Neuro-Symbiosis
- [ ] 集成到 PrivEnergyBench
- [ ] 集成到 MedSparseSNN
- [ ] 论文撰写："演化-发育-学习统一范式"

---

**快速获取帮助**: 参考 `/home/yanshi/EmbodiedSNNPrototype/docs/edl_transfer_guide.md`

