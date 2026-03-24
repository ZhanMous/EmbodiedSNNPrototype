#!/usr/bin/env python3
"""
EDL Framework Performance Summary Report Generator
Generates markdown + JSON analysis of EDL method comparisons.

Usage:
    python scripts/generate_edl_analysis.py --results /path/to/phase1_results.json
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import math

@dataclass
class MethodStats:
    name: str
    mean: float
    std: float
    min: float
    max: float
    trials: List[float]
    
    @property
    def se(self) -> float:
        """Standard error"""
        n = len(self.trials)
        return self.std / math.sqrt(max(n, 1))
    
    @property
    def ci95(self) -> Tuple[float, float]:
        """95% confidence interval (t-distribution, n=3: t=4.303)"""
        t_crit = 4.303 if len(self.trials) <= 3 else 1.96
        margin = t_crit * self.se
        return (self.mean - margin, self.mean + margin)


def load_phase1_results(path: str) -> Dict:
    """Load and parse phase1_results.json"""
    with open(path) as f:
        return json.load(f)


def compute_stats(values: List[float]) -> MethodStats:
    """Compute statistics for a method"""
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((x - mean)**2 for x in values) / len(values)
    std = math.sqrt(variance)
    return MethodStats(
        name="",
        mean=mean,
        std=std,
        min=min(values),
        max=max(values),
        trials=values,
    )


def generate_markdown_report(data: Dict) -> str:
    """Generate markdown comparison report"""
    summary = data['summary']
    raw = data['raw_results']
    
    methods = list(summary.keys())  # Use whatever methods are in data
    
    lines = [
        "# EDL Framework 性能分析报告",
        "",
        "## 汇总统计",
        "",
        "| 方法 | 平均得分 | 标准差 | 置信区间 (95%) | 相对提升 |",
        "|------|---------|--------|----------------|---------|",
    ]
    
    baseline_method = methods[0]  # Use first method as baseline
    baseline_mean = summary[baseline_method]['mean']
    
    for method in methods:
        stats = summary[method]
        mean = stats['mean']
        std = stats['std']
        
        # Compute CI
        n = len(raw[method])
        t_crit = 4.303 if n <= 3 else 1.96
        se = std / math.sqrt(max(n, 1))
        margin = t_crit * se
        ci_low, ci_high = mean - margin, mean + margin
        
        # Relative improvement
        if method != baseline_method:
            rel_imp = (mean - baseline_mean) / abs(baseline_mean) * 100
            rel_str = f"+{rel_imp:.1f}%" if rel_imp > 0 else f"{rel_imp:.1f}%"
        else:
            rel_str = "—"
        
        lines.append(
            f"| {method:20} | {mean:8.2f} | {std:8.2f} | "
            f"[{ci_low:7.2f}, {ci_high:7.2f}] | {rel_str:>8} |"
        )
    
    lines.extend([
        "",
        "## 逐 Trial 排名",
        "",
    ])
    
    num_trials = len(raw[methods[0]])
    for trial_idx in range(num_trials):
        scores = {m: raw[m][trial_idx] for m in methods}
        ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        
        lines.append(f"### Trial {trial_idx + 1}")
        lines.append("")
        for rank, (method, score) in enumerate(ranking, 1):
            lines.append(f"{rank}. **{method}**: {score:.2f}")
        lines.append("")
    
    lines.extend([
        "## 主要发现",
        "",
        f"1. **最佳方法**: {max(methods, key=lambda m: summary[m]['mean'])}",
        f"   - 平均得分: {max(summary[m]['mean'] for m in methods):.2f}",
        "",
        "2. **最可重复**: ",
        f"   - 最低标准差: {min(methods, key=lambda m: summary[m]['std'])}",
        f"   - Std: {min(summary[m]['std'] for m in methods):.2f}",
        "",
        "3. **性能范围**:",
        f"   - 中位数: {sorted([summary[m]['mean'] for m in methods])[len(methods)//2]:.2f}",
        "",
        "## 推荐应用",
        "",
        "- **快速适应**: 使用 `direct_evolution` (直接优化权重)",
        "- **参数压缩**: 使用 `grn_evolution` (280D→25D 发育编码)",
        "- **在线学习**: 基线 + 可塑性 (reward-modulated plasticity)",
        "",
    ])
    
    return "\n".join(lines)


def generate_json_report(data: Dict) -> Dict:
    """Generate structured JSON report"""
    summary = data['summary']
    raw = data['raw_results']
    
    methods = list(summary.keys())
    baseline_method = methods[0]
    baseline_mean = summary[baseline_method]['mean']
    
    report = {
        'timestamp': data.get('timestamp', 'unknown'),
        'num_trials': len(raw[methods[0]]),
        'methods': {},
    }
    
    for method in methods:
        stats = summary[method]
        n = len(raw[method])
        t_crit = 4.303 if n <= 3 else 1.96
        se = stats['std'] / math.sqrt(max(n, 1))
        margin = t_crit * se
        
        report['methods'][method] = {
            'mean': stats['mean'],
            'std': stats['std'],
            'min': min(raw[method]),
            'max': max(raw[method]),
            'ci95_low': stats['mean'] - margin,
            'ci95_high': stats['mean'] + margin,
            'relative_improvement': (
                (stats['mean'] - baseline_mean) / abs(baseline_mean) * 100
                if method != baseline_method else 0
            ),
            'trial_values': raw[method],
        }
    
    # Ranking
    ranked = sorted(
        report['methods'].items(),
        key=lambda kv: kv[1]['mean'],
        reverse=True
    )
    report['ranking'] = [m for m, _ in ranked]
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate EDL analysis report from phase1 results"
    )
    parser.add_argument(
        '--results',
        type=str,
        default='/home/yanshi/SNN-Evolution/results/phase1_results.json',
        help='Path to phase1_results.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/yanshi/EmbodiedSNNPrototype/outputs/edl_analysis',
        help='Output directory'
    )
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    data = load_phase1_results(args.results)
    
    # Generate reports
    md_report = generate_markdown_report(data)
    json_report = generate_json_report(data)
    
    # Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_path = output_dir / 'edl_analysis_report.md'
    json_path = output_dir / 'edl_analysis_report.json'
    
    with open(md_path, 'w') as f:
        f.write(md_report)
    
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"✓ Generated {md_path}")
    print(f"✓ Generated {json_path}")
    print("")
    print("=== REPORT PREVIEW ===")
    print(md_report)


if __name__ == "__main__":
    main()
