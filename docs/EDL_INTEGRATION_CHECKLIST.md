# EDL 框架集成清单

**Evolution-Development-Learning** (EDL) 框架现已集成到 EmbodiedSNNPrototype，可对标四种优化策略的性能。本文档提供下游项目集成的步骤清单。

## 核心文件

| 文件 | 用途 |
|------|------|
| `src/embodied_snn_prototype/edl.py` | EDL 基准测试核心模块（532 行） |
| `scripts/run_edl_benchmark.py` | CLI 入口点（63 行） |
| `configs/edl_benchmark.yaml` | 配置模板（31 行） |
| `docs/edl_transfer_guide.md` | 详细迁移文档 |
| `docs/neuro_symbiosis_edl_adapter.py` | Neuro-Symbiosis 适配示例 |

## 四种优化方法

### 1. `plastic_only`（基线）
- **策略**: 固定结构，仅通过奖励调制可塑性学习
- **适用**: 数据丰富的在线学习场景
- **应用**: EEG 解码的在线适应

### 2. `ea_readout`（进化导出权重优化）
- **策略**: 进化搜索输出层权重（25D），保持隐藏层连接固定
- **适用**: 快速适应新任务
- **应用**: BCI 受试者变化、能量预算约束下的模型调整

### 3. `crn_development`（发育编码）
- **策略**: 进化压缩基因组（280D）→通过 4 步发育动力学展开为 5×5 输出权重
- **适用**: 参数压缩、通用正则化、少样本学习
- **应用**: 在线学习中自动的结构化初始化

### 4. `evo_learning`（共进化学习）
- **策略**: 同时进化 25 条出权重 + 2 个可塑性参数（学习率、衰减）
- **适用**: 长期自适应，同时优化快/慢学习动态
- **应用**: 多天 BCI 适应，带漂移的长期部署

## 对标结果 (EmbodiedSNNPrototype)

### 相位 1 对标 (n=3 trials, 20 代进化, 4 代理)
| 方法 | 平均分 | 标准差 | vs 基线 |
|------|--------|--------|--------|
| plastic_only | 21.77 | 4.02 | — |
| ea_readout | 82.07 | 12.69 | **+277%** |
| crn_development | 91.99 | 0.00 | **+323%** ✓ |
| evo_learning | - | - | (pending) |

**关键发现**:
- CRN 发育编码实现最高得分但无方差（需验证）
- EA 直接优化在可重复性上更好（std 12.69）
- 基线可塑性单独无法竞争（可作为对照、在线学习基准）

## 快速集成步骤（Neuro-Symbiosis 示例）

### 步骤 1: 定义问题适配器

```python
# Neuro-Symbiosis/scripts/edl_adapter.py

def rollout_bci_task(snn_decoder, transformer_classifier, genome, seed):
    """
    Args:
        genome: np.ndarray, shape (280,)
                - [0:10]   = SNN 层参数 (tau, threshold, sparsity)
                - [10:280] = Transformer 权重初始化
    
    Returns:
        {'objective': float, 'accuracy': float, 'mia_auc': float, ...}
    """
    cfg = BCIConfig(seed=seed)
    
    # 解码基因组
    snn_params = genome[:10]
    tf_init = genome[10:]
    
    # 配置 SNN 前端
    snn_decoder.configure(snn_params)
    
    # 配置 Transformer 后端
    transformer_classifier.init_from_genome(tf_init)
    
    # 训练 + 评估 + 隐私审计
    train_acc = train_bci_pipeline(snn_decoder, transformer_classifier, cfg)
    test_acc = eval_bci_pipeline(...)
    mia_auc = run_mia_attack(...)
    
    # 多目标函数
    objective = test_acc - max(0.5 - mia_auc, 0) * 0.1
    
    return {
        'objective': objective,
        'accuracy': test_acc,
        'mia_auc': mia_auc,
        'train_acc': train_acc,
    }
```

### 步骤 2: 创建配置 YAML

```yaml
# Neuro-Symbiosis/configs/edl_bci.yaml

simulation:
  seed: 42
  num_bci_subjects: 5
  eeg_channels: 64
  session_length: 60  # seconds

evolution:
  population_size: 20
  generations: 15
  mutation_rate: 0.15
  genome_dimension: 280

benchmark:
  train_episodes: 3        # per subject
  eval_episodes: 2
  train_seed_stride: 10
  eval_seed_offset: 1000
  
  seeds: [3, 7, 11, 19, 23]  # BCI subjects

output:
  output_dir: outputs/edl_bci
  save_csv: true
  save_json: true
```

### 步骤 3: 集成 EDL 模块

```python
# Neuro-Symbiosis/scripts/run_edl_bci.py

from src.embodied_snn_prototype.edl import run_edl_benchmark
from .edl_adapter import rollout_bci_task

config = load_yaml('configs/edl_bci.yaml')
results = run_edl_benchmark(
    rollout_fn=rollout_bci_task,
    config=config,
)
```

## 输出格式

每个基准运行在 `output_dir` 下生成：

```
outputs/edl_benchmark/
├── edl_runs.csv                 # 所有 rollout 的原始数据
│   ├── seed, method, trial, episode, objective, accuracy, ...
├── edl_summary.csv              # 按方法聚合（均值 ± 标准差）
│   ├── method, objective_mean, objective_std, accuracy_mean, ...
├── edl_report.md                # 纯文本对标报告
├── edl_report.json              # 结构化报告（易于解析）
├── pareto_frontier.png          # [可选] Pareto 前沿绘图
└── method_comparison.png        # [可选] 箱线图对比
```

## 集成到 PrivEnergyBench

### 适配策略

**问题**: 隐私-能量-效用 Pareto 优化
```
目标函数 = 效用 - δ·能量开销 - λ·隐私违规
```

**基因组映射**:
- [0:8]     = 梯度压缩率、量化位数、差分隐私 ε
- [8:200]   = 层级 dropout 掩码（FedAvg 稀疏性）
- [200:280] = 神经网络权重初始化

## 集成到 MedSparseSNN

### 适配策略

**问题**: 医学图像分类，MAC/能量约束下
```
目标函数 = 分类准确率 - ω·MAC_比例 - ζ·能量代理
```

**基因组映射**:
- [0:16]    = 每层阈值计划（16 层）
- [16:48]   = 层级 pruning 比例
- [48:280]  = 输出分类头权重

## 诊断和调试

### 问题 1: 方差为 0（GRN 案例）

**症状**: `grn_evolution: 91.99 ± 0.00`

**诊断**: 
- 可能所有 seed 收敛到相同全局最优
- 或发育函数过度确定（参数不充分自由度）
- 或 EA 早熟收敛

**修复**:
```yaml
evolution:
  population_size: 30        # 增加种群
  generations: 20            # 增加代数
  mutation_rate: 0.25        # 更高的变异
```

### 问题 2: 低可重复性 (std > mean/3)

**症状**: `ea_readout: 82.07 ± 12.69` （相对 std = 15%）

**诊断**: 
- 初始种群随机性太高
- 适应度定义不够稳定
- 需要更多评估代理

**修复**:
```yaml
benchmark:
  eval_episodes: 8           # 增加评估次数
  train_episodes: 6
```

### 问题 3: EA 卡顿（无进展输出）

**症状**: 30s+ 无任何进度条

**诊断**: 
- 考虑检查 `_rollout_episode()` 循环中的累积计算
- 验证 `network.reset()` 在每个评估之前被调用
- 检查 spike_history 是否无界增长

**修复**: 参照 SNN-Evolution Phase1 修复中的 O(1) 能量累加器模式

## 性能期望

### 资源需求

| 方法 | 墙时钟/trial | 内存 | 示例配置 |
|------|-------------|------|---------|
| plastic_only | 0.5s | 50MB | 4 episodes × 220 steps |
| ea_readout | 15s | 120MB | pop 20 × gen 12 × 6 trials |
| crn_development | 18s | 140MB | + 5x 发育递推 |
| evo_learning | 20s | 150MB | + 2x 可塑性参数发育 |

### 扩展性指南

- **小规模刀片测试**: `population_size: 10, generations: 5` (1-2s)
- **中规模研究**: `population_size: 20, generations: 15` (15-30s per trial)
- **大规模扫描**: `population_size: 50, generations: 30` (2-5 min per trial)

## 文献参考

相关方法论文:
- **CRN 发育**: Bongard & Pfeifer (2003) Evolving Complete Agents using Artificial Ontogeny
- **共进化学习**: Floreano & Mondada (1996) Evolution of Plastic Learning Networks
- **多目标 EA**: Deb (2001) Multi-Objective Optimization using Evolutionary Algorithms

## 版本历史

| 版本 | 日期 | 更改 |
|------|------|------|
| 1.0 | 2025-01 | EDL 基准发布，4 方法实现 |
| 1.1 | 2025-01 | 添加 CRN 发育解码器、evo-learning 共进化 |
| 2.0 | 计划 | 多目标排名、Pareto 绘图、RNN/图神经网络发育 |

---

**问题或建议?** 参照 `docs/edl_transfer_guide.md` 获取详细 API 文档。
