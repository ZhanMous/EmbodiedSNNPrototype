#!/usr/bin/env python3
"""
GRN Variance Verification Script
Runs EDL benchmark with n=10 seeds to verify GRN development's variance anomaly.

Background: Phase1 results showed GRN std=0.00 at n=3, suggesting:
  - Perfect convergence across random seeds
  - Possible overfitting or genetic drift
  - Or developmental decoder is over-constrained (not enough DoF)

This script runs the same benchmark with n=10 to get better variance estimate.
"""

import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime

def run_grn_variance_verification():
    """Execute GRN variance verification workflow"""
    
    print("=" * 80)
    print("GRN VARIANCE VERIFICATION - Extended Trials (n=10)")
    print("=" * 80)
    print()
    
    # Config paths
    config_path = Path('/home/yanshi/EmbodiedSNNPrototype/configs/edl_benchmark.yaml')
    output_dir = Path('/home/yanshi/EmbodiedSNNPrototype/outputs/grn_variance_study')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Verify config has n=10 seeds
    print("[Step 1] Checking configuration...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    seeds = config['seeds']
    print(f"  Seeds in config: {seeds}")
    print(f"  Number of trials: {len(seeds)}")
    
    if len(seeds) < 10:
        print(f"  ⚠️  WARNING: Expected n=10, got n={len(seeds)}")
        print("  Updating config to include 10 seeds...")
        config['seeds'] = [3, 7, 11, 19, 23, 31, 43, 59, 67, 83]
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"  ✓ Updated to {len(config['seeds'])} seeds")
    
    print()
    
    # Step 2: Run EDL benchmark
    print("[Step 2] Running EDL benchmark with n=10 seeds...")
    print("  This may take 5-10 minutes. Progress will be shown below.")
    print()
    
    try:
        result = subprocess.run(
            ['python', '/home/yanshi/EmbodiedSNNPrototype/scripts/run_edl_benchmark.py',
             '--config', str(config_path)],
            cwd='/home/yanshi/EmbodiedSNNPrototype',
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"  ❌ Benchmark failed:")
            print(result.stderr)
            return False
        
        print("  ✓ Benchmark complete")
    except subprocess.TimeoutExpired:
        print("  ❌ Benchmark timeout (exceeded 10 minutes)")
        return False
    
    print()
    
    # Step 3: Generate analysis
    print("[Step 3] Generating variance analysis...")
    
    try:
        result = subprocess.run(
            ['python', '/home/yanshi/EmbodiedSNNPrototype/scripts/generate_edl_analysis.py'],
            cwd='/home/yanshi/EmbodiedSNNPrototype',
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"  ❌ Analysis failed: {result.stderr}")
            return False
            
        print("  ✓ Analysis complete")
    except subprocess.TimeoutExpired:
        print("  ❌ Analysis timeout")
        return False
    
    print()
    
    # Step 4: Compare n=3 vs n=10 results
    print("[Step 4] Comparing Phase1 (n=3) vs Extended Study (n=10)...")
    
    phase1_results = Path('/home/yanshi/SNN-Evolution/results/phase1_results.json')
    current_analysis = Path('/home/yanshi/EmbodiedSNNPrototype/outputs/edl_analysis/edl_analysis_report.json')
    
    if not phase1_results.exists():
        print(f"  ⚠️  Phase1 results not found: {phase1_results}")
        return False
    
    if not current_analysis.exists():
        print(f"  ⚠️  Current analysis not found: {current_analysis}")
        return False
    
    with open(phase1_results) as f:
        phase1 = json.load(f)
    
    with open(current_analysis) as f:
        current = json.load(f)
    
    print()
    print("  VARIANCE COMPARISON")
    print("  " + "─" * 70)
    print("  Method            │ n=3 (Phase1)      n=10 (Extended)   Δ Std")
    print("  ─" * 70)
    
    methods = ['baseline', 'surrogate', 'direct_evolution', 'grn_evolution']
    
    for method in methods:
        if method in phase1['summary'] and method in current['methods']:
            phase1_std = phase1['summary'][method]['std']
            current_std = current['methods'][method]['std']
            delta = current_std - phase1_std
            sign = "↑" if delta > 0 else "↓" if delta < 0 else "="
            
            print(f"  {method:17} │ {phase1_std:6.3f}           {current_std:6.3f}       {sign} {abs(delta):6.3f}")
    
    print()
    print("  INTERPRETATION")
    print("  " + "─" * 70)
    
    grn_phase1_std = phase1['summary']['grn_evolution']['std']
    grn_current_std = current['methods']['grn_evolution']['std']
    grn_mean = current['methods']['grn_evolution']['mean']
    
    if grn_current_std < 0.01:
        print(f"  ⚠️  GRN still shows low variance (std={grn_current_std:.6f})")
        print(f"      Possible explanations:")
        print(f"      1. Global optimum convergence (all seeds → same solution)")
        print(f"      2. Developmental decoder over-constrained (insufficient DoF)")
        print(f"      3. Genetic drift → premature convergence at mean={grn_mean:.2f}")
    elif grn_current_std > grn_phase1_std * 2:
        print(f"  ✓ GRN variance INCREASED with more seeds")
        print(f"    Phase1 n=3: std={grn_phase1_std:.3f}")
        print(f"    Extended n=10: std={grn_current_std:.3f}")
        print(f"    → Suggests initial low variance was sampling artifact")
    else:
        print(f"  ~ GRN variance remains relatively stable")
        print(f"    Phase1 n=3: std={grn_phase1_std:.3f}")
        print(f"    Extended n=10: std={grn_current_std:.3f}")
    
    print()
    print("  RECOMMENDATIONS")
    print("  " + "─" * 70)
    
    if grn_current_std < 0.01:
        print("  • Increase EA mutation_rate from 0.18 → 0.25-0.30")
        print("  • Increase population_size from 20 → 30-40")
        print("  • Increase generations from 12 → 20-25")
        print("  • Add explicit diversity maintenance (niching, adaptive mutation)")
    else:
        print("  ✓ CRN development shows healthy variance")
        print("  • Results are reproducible across seeds")
        print("  • Can proceed with confidence to downstream projects")
    
    print()
    print("=" * 80)
    print("✓ GRN VARIANCE VERIFICATION COMPLETE")
    print("=" * 80)
    
    # Save summary
    summary_path = output_dir / 'grn_variance_summary.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'phase1_n3': {
            'grn_mean': phase1['summary']['grn_evolution']['mean'],
            'grn_std': phase1['summary']['grn_evolution']['std'],
        },
        'extended_n10': {
            'grn_mean': current['methods']['grn_evolution']['mean'],
            'grn_std': current['methods']['grn_evolution']['std'],
            'all_trials': current['methods']['grn_evolution']['trial_values'],
        },
        'variance_inference': (
            'stable' if grn_current_std < 0.01
            else 'increased' if grn_current_std > grn_phase1_std * 2
            else 'moderate'
        )
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {output_dir}/")
    
    return True

if __name__ == "__main__":
    success = run_grn_variance_verification()
    exit(0 if success else 1)
