#!/usr/bin/env python3
"""
verify_alignment.py - Verify Text Simulation vs SUMO Alignment
===============================================================

Compares results from Roundabout.py (text-based) with SUMO roundabout
simulations to verify parameter alignment and model fidelity.

Usage:
    python verify_alignment.py --text-sim ../Roundabout.py --config config/config.yaml --output results/alignment/
"""

import argparse
import subprocess
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import json

class AlignmentVerifier:
    """
    Verifies alignment between text-based and SUMO simulations.
    """
    
    def __init__(self, config_path: str, output_dir: str):
        """Initialize verifier."""
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Results storage
        self.text_results = {}
        self.sumo_results = {}
        
        print(f"Loaded configuration from {config_path}")
    
    def run_text_simulation(self, scenario_name: str, params: Dict) -> Dict:
        """
        Run text-based simulation (Roundabout.py).
        
        Args:
            scenario_name: Identifier for this scenario
            params: Dictionary with parameters (lanes, diameter, arrival, etc.)
        
        Returns:
            Dictionary with metrics
        """
        print(f"\n{'='*60}")
        print(f"Running text simulation: {scenario_name}")
        print(f"  Params: {params}")
        print(f"{'='*60}")
        
        # Build command
        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent.parent / "Roundabout.py"),
            "--lanes", str(params.get('lanes', 2)),
            "--diameter", str(params.get('diameter', 45)),
            "--seed", str(params.get('seed', 42)),
            "--horizon", str(params.get('horizon', 3600)),
            "--arrival", *[str(x) for x in params.get('arrival', [0.18, 0.12, 0.20, 0.15])],
            "--turning", *[str(x) for x in params.get('turning', [0.25, 0.55, 0.20])],
        ]
        
        # Run simulation
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: Text simulation failed!")
            print(result.stderr)
            return {}
        
        # Parse output
        metrics = self._parse_text_output(result.stdout)
        self.text_results[scenario_name] = metrics
        
        print(f"✓ Text simulation complete:")
        print(f"  Throughput: {metrics.get('throughput', 0):.0f} veh/hr")
        print(f"  Avg delay: {metrics.get('avg_delay', 0):.1f}s")
        
        return metrics
    
    def run_sumo_simulation(self, scenario_name: str, params: Dict) -> Dict:
        """
        Run SUMO simulation.
        
        Args:
            scenario_name: Identifier for this scenario
            params: Dictionary with parameters
        
        Returns:
            Dictionary with metrics
        """
        print(f"\n{'='*60}")
        print(f"Running SUMO simulation: {scenario_name}")
        print(f"  Params: {params}")
        print(f"{'='*60}")
        
        # Generate network
        from generate_network import RoundaboutNetworkGenerator
        
        # Modify config temporarily
        temp_config = self.config.copy()
        temp_config['geometry']['diameter'] = params.get('diameter', 45)
        temp_config['geometry']['lanes'] = params.get('lanes', 2)
        temp_config['demand']['arrivals'] = params.get('arrival', [0.18, 0.12, 0.20, 0.15])
        temp_config['demand']['turning_probabilities'] = params.get('turning', [0.25, 0.55, 0.20])
        temp_config['simulation']['seed'] = params.get('seed', 42)
        temp_config['simulation']['horizon'] = params.get('horizon', 3600)
        
        # Generate network and routes
        output_path = self.output_dir / scenario_name
        output_path.mkdir(exist_ok=True)
        
        gen = RoundaboutNetworkGenerator(temp_config)
        gen.generate(str(output_path))
        
        # Generate routes
        from generate_routes import RoundaboutRouteGenerator
        route_gen = RoundaboutRouteGenerator(temp_config)
        route_gen.generate(str(output_path / "roundabout.rou.xml"))
        
        # Run SUMO
        from run_simulation import SUMOSimulation
        sumocfg = output_path / "roundabout.sumocfg"
        
        sim = SUMOSimulation(str(sumocfg), gui=False)
        agg_metrics = sim.run()
        
        # Convert to comparable format
        metrics = {
            'throughput': agg_metrics.throughput_vph,
            'avg_delay': agg_metrics.mean_delay,
            'p95_delay': agg_metrics.p95_delay,
            'max_queue_N': agg_metrics.max_queue_N,
            'max_queue_E': agg_metrics.max_queue_E,
            'max_queue_S': agg_metrics.max_queue_S,
            'max_queue_W': agg_metrics.max_queue_W,
            'total_arrivals': agg_metrics.total_arrivals,
            'total_exits': agg_metrics.total_exits,
        }
        
        self.sumo_results[scenario_name] = metrics
        
        print(f"✓ SUMO simulation complete:")
        print(f"  Throughput: {metrics.get('throughput', 0):.0f} veh/hr")
        print(f"  Avg delay: {metrics.get('avg_delay', 0):.1f}s")
        
        return metrics
    
    def _parse_text_output(self, output: str) -> Dict:
        """Parse Roundabout.py output to extract metrics."""
        metrics = {}
        
        for line in output.split('\n'):
            if '=== Hourly Summary ===' in line:
                # Next few lines contain summary
                pass
            elif 'throughput=' in line:
                # Extract throughput
                parts = line.split('throughput=')
                if len(parts) > 1:
                    try:
                        metrics['throughput'] = float(parts[1].split()[0])
                    except:
                        pass
            elif 'avg_delay=' in line:
                parts = line.split('avg_delay=')
                if len(parts) > 1:
                    try:
                        metrics['avg_delay'] = float(parts[1].replace('s', '').split()[0])
                    except:
                        pass
            elif 'p95=' in line:
                parts = line.split('p95=')
                if len(parts) > 1:
                    try:
                        metrics['p95_delay'] = float(parts[1].replace('s', '').split()[0])
                    except:
                        pass
            elif 'max_queue_per_arm=' in line:
                parts = line.split('max_queue_per_arm=')
                if len(parts) > 1:
                    try:
                        queue_str = parts[1].strip()
                        # Parse [N, E, S, W]
                        queues = eval(queue_str)
                        metrics['max_queue_N'] = queues[0]
                        metrics['max_queue_E'] = queues[1]
                        metrics['max_queue_S'] = queues[2]
                        metrics['max_queue_W'] = queues[3]
                    except:
                        pass
            elif 'arrivals=' in line and 'exits=' in line:
                # Parse arrivals and exits from summary line
                if 'lanes=' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.startswith('arrivals='):
                            try:
                                metrics['total_arrivals'] = int(part.split('=')[1])
                            except:
                                pass
                        elif part.startswith('exits='):
                            try:
                                metrics['total_exits'] = int(part.split('=')[1])
                            except:
                                pass
        
        return metrics
    
    def compare_scenarios(self, scenarios: List[Dict]):
        """
        Run both text and SUMO for each scenario and compare.
        
        Args:
            scenarios: List of scenario dictionaries with 'name' and 'params' keys
        """
        results = []
        
        for scenario in scenarios:
            name = scenario['name']
            params = scenario['params']
            
            # Run both simulations
            text_metrics = self.run_text_simulation(name, params)
            sumo_metrics = self.run_sumo_simulation(name, params)
            
            # Compute differences
            comparison = {
                'scenario': name,
                'text': text_metrics,
                'sumo': sumo_metrics,
                'differences': {}
            }
            
            for metric in ['throughput', 'avg_delay', 'p95_delay']:
                if metric in text_metrics and metric in sumo_metrics:
                    text_val = text_metrics[metric]
                    sumo_val = sumo_metrics[metric]
                    diff_abs = sumo_val - text_val
                    diff_rel = (diff_abs / text_val * 100) if text_val > 0 else 0
                    comparison['differences'][metric] = {
                        'absolute': diff_abs,
                        'relative_pct': diff_rel
                    }
            
            results.append(comparison)
        
        # Save results
        self._save_comparison_results(results)
        self._visualize_comparison(results)
        
        return results
    
    def _save_comparison_results(self, results: List[Dict]):
        """Save comparison results to JSON and CSV."""
        # JSON
        json_path = self.output_dir / 'alignment_comparison.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved comparison to {json_path}")
        
        # CSV
        rows = []
        for result in results:
            scenario = result['scenario']
            text = result['text']
            sumo = result['sumo']
            diffs = result['differences']
            
            row = {
                'scenario': scenario,
                'text_throughput': text.get('throughput', np.nan),
                'sumo_throughput': sumo.get('throughput', np.nan),
                'throughput_diff_pct': diffs.get('throughput', {}).get('relative_pct', np.nan),
                'text_avg_delay': text.get('avg_delay', np.nan),
                'sumo_avg_delay': sumo.get('avg_delay', np.nan),
                'avg_delay_diff_pct': diffs.get('avg_delay', {}).get('relative_pct', np.nan),
                'text_p95_delay': text.get('p95_delay', np.nan),
                'sumo_p95_delay': sumo.get('p95_delay', np.nan),
                'p95_delay_diff_pct': diffs.get('p95_delay', {}).get('relative_pct', np.nan),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / 'alignment_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved comparison CSV to {csv_path}")
    
    def _visualize_comparison(self, results: List[Dict]):
        """Create visualization comparing text vs SUMO results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        scenarios = [r['scenario'] for r in results]
        
        # Throughput comparison
        ax = axes[0, 0]
        text_throughput = [r['text'].get('throughput', 0) for r in results]
        sumo_throughput = [r['sumo'].get('throughput', 0) for r in results]
        
        x = np.arange(len(scenarios))
        width = 0.35
        ax.bar(x - width/2, text_throughput, width, label='Text sim', alpha=0.8)
        ax.bar(x + width/2, sumo_throughput, width, label='SUMO', alpha=0.8)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Throughput (veh/hr)')
        ax.set_title('Throughput Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Average delay comparison
        ax = axes[0, 1]
        text_delay = [r['text'].get('avg_delay', 0) for r in results]
        sumo_delay = [r['sumo'].get('avg_delay', 0) for r in results]
        
        ax.bar(x - width/2, text_delay, width, label='Text sim', alpha=0.8)
        ax.bar(x + width/2, sumo_delay, width, label='SUMO', alpha=0.8)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Average delay (s)')
        ax.set_title('Average Delay Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # P95 delay comparison
        ax = axes[0, 2]
        text_p95 = [r['text'].get('p95_delay', 0) for r in results]
        sumo_p95 = [r['sumo'].get('p95_delay', 0) for r in results]
        
        ax.bar(x - width/2, text_p95, width, label='Text sim', alpha=0.8)
        ax.bar(x + width/2, sumo_p95, width, label='SUMO', alpha=0.8)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('P95 delay (s)')
        ax.set_title('P95 Delay Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Relative differences - Throughput
        ax = axes[1, 0]
        throughput_diffs = [r['differences'].get('throughput', {}).get('relative_pct', 0) for r in results]
        ax.bar(x, throughput_diffs, color='coral', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=10, color='red', linestyle=':', alpha=0.5, label='±10% threshold')
        ax.axhline(y=-10, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Relative difference (%)')
        ax.set_title('Throughput: (SUMO - Text) / Text × 100%')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Relative differences - Avg delay
        ax = axes[1, 1]
        delay_diffs = [r['differences'].get('avg_delay', {}).get('relative_pct', 0) for r in results]
        ax.bar(x, delay_diffs, color='skyblue', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=15, color='red', linestyle=':', alpha=0.5, label='±15% threshold')
        ax.axhline(y=-15, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Relative difference (%)')
        ax.set_title('Avg Delay: (SUMO - Text) / Text × 100%')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary table
        ax = axes[1, 2]
        ax.axis('off')
        
        # Compute summary statistics
        summary_lines = ["=== Alignment Summary ===\n"]
        
        # Throughput
        tp_diffs = [abs(d) for d in throughput_diffs if d != 0]
        if tp_diffs:
            avg_tp_diff = np.mean(tp_diffs)
            max_tp_diff = max(tp_diffs)
            summary_lines.append(f"Throughput:")
            summary_lines.append(f"  Mean |diff|: {avg_tp_diff:.1f}%")
            summary_lines.append(f"  Max |diff|: {max_tp_diff:.1f}%")
            if max_tp_diff < 10:
                summary_lines.append(f"  Status: ✓ Good (<10%)")
            elif max_tp_diff < 20:
                summary_lines.append(f"  Status: ⚠ Acceptable (<20%)")
            else:
                summary_lines.append(f"  Status: ✗ Poor (>20%)")
        
        summary_lines.append("")
        
        # Delay
        dly_diffs = [abs(d) for d in delay_diffs if d != 0]
        if dly_diffs:
            avg_dly_diff = np.mean(dly_diffs)
            max_dly_diff = max(dly_diffs)
            summary_lines.append(f"Average Delay:")
            summary_lines.append(f"  Mean |diff|: {avg_dly_diff:.1f}%")
            summary_lines.append(f"  Max |diff|: {max_dly_diff:.1f}%")
            if max_dly_diff < 15:
                summary_lines.append(f"  Status: ✓ Good (<15%)")
            elif max_dly_diff < 30:
                summary_lines.append(f"  Status: ⚠ Acceptable (<30%)")
            else:
                summary_lines.append(f"  Status: ✗ Poor (>30%)")
        
        ax.text(0.1, 0.9, '\n'.join(summary_lines),
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plot_path = self.output_dir / 'alignment_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {plot_path}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Verify Text vs SUMO Alignment")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results/alignment',
                       help='Output directory for results')
    parser.add_argument('--scenarios', type=str, default=None,
                       help='Path to JSON file with custom scenarios (optional)')
    
    args = parser.parse_args()
    
    # Initialize verifier
    verifier = AlignmentVerifier(args.config, args.output)
    
    # Define test scenarios
    if args.scenarios:
        with open(args.scenarios, 'r') as f:
            scenarios = json.load(f)
    else:
        # Default scenarios
        scenarios = [
            {
                'name': 'baseline_1lane',
                'params': {
                    'lanes': 1,
                    'diameter': 45,
                    'arrival': [0.10, 0.10, 0.10, 0.10],
                    'turning': [0.25, 0.55, 0.20],
                    'seed': 42,
                    'horizon': 1800  # 30 minutes for faster testing
                }
            },
            {
                'name': 'baseline_2lane',
                'params': {
                    'lanes': 2,
                    'diameter': 45,
                    'arrival': [0.10, 0.10, 0.10, 0.10],
                    'turning': [0.25, 0.55, 0.20],
                    'seed': 42,
                    'horizon': 1800
                }
            },
            {
                'name': 'high_demand_2lane',
                'params': {
                    'lanes': 2,
                    'diameter': 45,
                    'arrival': [0.15, 0.12, 0.15, 0.12],
                    'turning': [0.25, 0.55, 0.20],
                    'seed': 42,
                    'horizon': 1800
                }
            },
            {
                'name': 'small_diameter_2lane',
                'params': {
                    'lanes': 2,
                    'diameter': 30,
                    'arrival': [0.10, 0.10, 0.10, 0.10],
                    'turning': [0.25, 0.55, 0.20],
                    'seed': 42,
                    'horizon': 1800
                }
            },
        ]
    
    print(f"\n{'='*70}")
    print(f"TEXT vs SUMO ALIGNMENT VERIFICATION")
    print(f"  Testing {len(scenarios)} scenarios")
    print(f"{'='*70}\n")
    
    # Run comparison
    results = verifier.compare_scenarios(scenarios)
    
    print(f"\n{'='*70}")
    print(f"✅ VERIFICATION COMPLETE")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
