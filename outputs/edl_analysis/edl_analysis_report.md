# EDL Framework 性能分析报告

## 汇总统计

| 方法 | 平均得分 | 标准差 | 置信区间 (95%) | 相对提升 |
|------|---------|--------|----------------|---------|
| baseline             |    21.77 |     4.02 | [  11.77,   31.76] |        — |
| surrogate            |    19.73 |     6.88 | [   2.65,   36.82] |    -9.4% |
| direct_evolution     |    82.07 |    12.69 | [  50.55,  113.59] |  +277.0% |
| grn_evolution        |    91.99 |     0.00 | [  91.99,   91.99] |  +322.6% |

## 逐 Trial 排名

### Trial 1

1. **grn_evolution**: 91.99
2. **direct_evolution**: 84.79
3. **baseline**: 22.46
4. **surrogate**: 10.47

### Trial 2

1. **grn_evolution**: 91.99
2. **direct_evolution**: 65.35
3. **surrogate**: 21.80
4. **baseline**: 16.53

### Trial 3

1. **direct_evolution**: 96.07
2. **grn_evolution**: 91.99
3. **surrogate**: 26.93
4. **baseline**: 26.31

## 主要发现

1. **最佳方法**: grn_evolution
   - 平均得分: 91.99

2. **最可重复**: 
   - 最低标准差: grn_evolution
   - Std: 0.00

3. **性能范围**:
   - 中位数: 82.07

## 推荐应用

- **快速适应**: 使用 `direct_evolution` (直接优化权重)
- **参数压缩**: 使用 `grn_evolution` (280D→25D 发育编码)
- **在线学习**: 基线 + 可塑性 (reward-modulated plasticity)
