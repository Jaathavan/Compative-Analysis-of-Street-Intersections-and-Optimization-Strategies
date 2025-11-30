#!/usr/bin/env python3
"""
generate_roundabout_comparisons.py - Text vs SUMO Roundabout Comparison
========================================================================

Generates comprehensive comparison visualizations between text-based and SUMO
roundabout simulations across multiple parameters.

Graphs Generated:
1. Max Queue Length vs Arrival Rate (by lane count)
2. Average Delay vs Arrival Rate (by lane count)
3. Throughput vs Arrival Rate (by lane count)
4. 95th Percentile Delay vs Arrival Rate (by lane count)
5. Average Delay vs Diameter (λ=0.10, by lane count)
6. Max Queue vs Diameter (λ=0.10, by lane count)
7. Throughput vs Diameter (λ=0.10, by lane count)
8. 95th Percentile Delay vs Diameter (λ=0.10, by lane count)

Usage:
    python generate_roundabout_comparisons.py --output-dir results/comparisons
"""

import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import yaml
from typing import Dict, List, Tuple
import re

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


class RoundaboutComparisonGenerator:
    """Generates text vs SUMO roundabout comparisons."""
    
    def __init__(self, output_dir: str):
        """Initialize comparison generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_results = []
        self.sumo_results = []
        
        # Parameter ranges
        self.arrival_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # veh/s/arm
        self.diameters = [20, 30, 40, 50, 60]  # meters
        self.lane_counts = [1, 2, 3]
        self.fixed_arrival = 0.10  # veh/s/arm for diameter sweep
        
    def run_text_simulations(self):
        """Run text-based roundabout simulations."""
        print("="*70)
        print("Running Text-Based Simulations")
        print("="*70)
        
        total = len(self.arrival_rates) * len(self.diameters) * len(self.lane_counts)
        count = 0
        
        for lanes in self.lane_counts:
            for diameter in self.diameters:
                for arrival in self.arrival_rates:
                    count += 1
                    print(f"\n[{count}/{total}] Text: {lanes}-lane, d={diameter}m, λ={arrival:.2f} veh/s/arm")
                    
                    # Run simulation
                    cmd = [
                        'python3', 'Roundabout.py',
                        '--lanes', str(lanes),
                        '--diameter', str(diameter),
                        '--arrival', str(arrival), str(arrival), str(arrival), str(arrival),
                        '--horizon', '1200',  # 20 minutes for stable statistics
                        '--seed', '42'
                    ]
                    
                    try:
                        result = subprocess.run(
                            cmd, 
                            capture_output=True, 
                            text=True, 
                            timeout=120
                        )
                        
                        # Parse output
                        metrics = self._parse_text_output(result.stdout)
                        if metrics:
                            metrics.update({
                                'lanes': lanes,
                                'diameter': diameter,
                                'arrival_rate': arrival,
                                'simulator': 'text'
                            })
                            self.text_results.append(metrics)
                            print(f"  ✓ Throughput: {metrics['throughput']:.1f} veh/hr, "
                                  f"Delay: {metrics['avg_delay']:.1f}s")
                        else:
                            print(f"  ✗ Failed to parse output")
                            
                    except subprocess.TimeoutExpired:
                        print(f"  ✗ Timeout")
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
        
        # Save text results
        if self.text_results:
            df = pd.DataFrame(self.text_results)
            output_path = self.output_dir / 'text_simulation_results.csv'
            df.to_csv(output_path, index=False)
            print(f"\n✓ Text results saved: {output_path}")
            print(f"  Total successful runs: {len(self.text_results)}")
    
    def _parse_text_output(self, output: str) -> Dict:
        """Parse text simulation output."""
        try:
            # Find the hourly summary section
            lines = output.split('\n')
            
            metrics = {}
            for i, line in enumerate(lines):
                if '=== Hourly Summary ===' in line or 'SIMULATION SUMMARY' in line:
                    # Parse the summary lines
                    for j in range(i, min(i+10, len(lines))):
                        l = lines[j]
                        
                        # Extract throughput
                        if 'throughput=' in l.lower():
                            match = re.search(r'throughput[=\s]+(\d+\.?\d*)', l, re.IGNORECASE)
                            if match:
                                metrics['throughput'] = float(match.group(1))
                        
                        # Extract average delay
                        if 'avg_delay=' in l.lower() or 'average delay' in l.lower():
                            match = re.search(r'avg[_\s]?delay[=:\s]+(\d+\.?\d*)', l, re.IGNORECASE)
                            if match:
                                metrics['avg_delay'] = float(match.group(1))
                        
                        # Extract p95 delay
                        if 'p95=' in l.lower() or '95' in l:
                            match = re.search(r'p95[=:\s]+(\d+\.?\d*)', l, re.IGNORECASE)
                            if match:
                                metrics['p95_delay'] = float(match.group(1))
                        
                        # Extract max queue
                        if 'max_queue' in l.lower() or 'max queue' in l.lower():
                            match = re.search(r'max[_\s]?queue[^=]*=?\s*\[([^\]]+)\]', l, re.IGNORECASE)
                            if match:
                                queues = [int(x.strip()) for x in match.group(1).split(',')]
                                metrics['max_queue'] = max(queues)
            
            # Validate we got essential metrics
            if 'throughput' in metrics and 'avg_delay' in metrics:
                return metrics
            else:
                return None
                
        except Exception as e:
            print(f"    Parse error: {e}")
            return None
    
    def run_sumo_simulations(self):
        """Run SUMO roundabout simulations."""
        print("\n" + "="*70)
        print("Running SUMO Simulations")
        print("="*70)
        
        total = len(self.arrival_rates) * len(self.diameters) * len(self.lane_counts)
        count = 0
        
        for lanes in self.lane_counts:
            for diameter in self.diameters:
                for arrival in self.arrival_rates:
                    count += 1
                    print(f"\n[{count}/{total}] SUMO: {lanes}-lane, d={diameter}m, λ={arrival:.2f} veh/s/arm")
                    
                    # Create config directory
                    config_name = f"sumo_{lanes}lane_d{diameter}_arr{arrival:.2f}"
                    config_dir = self.output_dir / 'sumo_configs' / config_name
                    config_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate network
                    try:
                        net_cmd = [
                            'python3', 'roundabout/src/generate_network.py',
                            '--diameter', str(diameter),
                            '--lanes', str(lanes),
                            '--output', str(config_dir / 'roundabout.net.xml')
                        ]
                        subprocess.run(net_cmd, check=True, capture_output=True, timeout=30)
                        
                        # Generate routes (convert veh/s to veh/hr)
                        demand_vehhr = arrival * 3600
                        route_cmd = [
                            'python3', 'roundabout/src/generate_routes.py',
                            '--demand', str(demand_vehhr),
                            '--duration', '1200',
                            '--output', str(config_dir / 'routes.rou.xml')
                        ]
                        subprocess.run(route_cmd, check=True, capture_output=True, timeout=30)
                        
                        # Create SUMO config file
                        sumocfg = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="roundabout.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1200"/>
    </time>
    <output>
        <summary-output value="summary.xml"/>
        <tripinfo-output value="tripinfo.xml"/>
    </output>
</configuration>"""
                        with open(config_dir / 'roundabout.sumocfg', 'w') as f:
                            f.write(sumocfg)
                        
                        # Run SUMO simulation
                        sim_cmd = [
                            'sumo',
                            '-c', str(config_dir / 'roundabout.sumocfg'),
                            '--no-warnings', 'true',
                            '--no-step-log', 'true'
                        ]
                        subprocess.run(sim_cmd, check=True, capture_output=True, timeout=120)
                        
                        # Parse results
                        metrics = self._parse_sumo_output(config_dir / 'tripinfo.xml')
                        if metrics:
                            metrics.update({
                                'lanes': lanes,
                                'diameter': diameter,
                                'arrival_rate': arrival,
                                'simulator': 'sumo'
                            })
                            self.sumo_results.append(metrics)
                            print(f"  ✓ Throughput: {metrics['throughput']:.1f} veh/hr, "
                                  f"Delay: {metrics['avg_delay']:.1f}s")
                        else:
                            print(f"  ✗ Failed to parse SUMO output")
                    
                    except subprocess.TimeoutExpired:
                        print(f"  ✗ Timeout")
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
        
        # Save SUMO results
        if self.sumo_results:
            df = pd.DataFrame(self.sumo_results)
            output_path = self.output_dir / 'sumo_simulation_results.csv'
            df.to_csv(output_path, index=False)
            print(f"\n✓ SUMO results saved: {output_path}")
            print(f"  Total successful runs: {len(self.sumo_results)}")
    
    def _parse_sumo_output(self, tripinfo_path: Path) -> Dict:
        """Parse SUMO tripinfo output."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(tripinfo_path)
            root = tree.getroot()
            
            delays = []
            durations = []
            
            for tripinfo in root.findall('tripinfo'):
                delay = float(tripinfo.get('waitingTime', 0))
                duration = float(tripinfo.get('duration', 0))
                delays.append(delay)
                durations.append(duration)
            
            if len(delays) == 0:
                return None
            
            # Calculate metrics
            metrics = {
                'throughput': len(delays) * 3600 / 1200,  # veh/hr
                'avg_delay': np.mean(delays),
                'p95_delay': np.percentile(delays, 95),
                'max_queue': 0  # SUMO doesn't easily provide this from tripinfo
            }
            
            return metrics
            
        except Exception as e:
            print(f"    SUMO parse error: {e}")
            return None
    
    def generate_comparison_plots(self):
        """Generate all comparison plots."""
        print("\n" + "="*70)
        print("Generating Comparison Plots")
        print("="*70)
        
        # Load results
        text_df = pd.DataFrame(self.text_results)
        sumo_df = pd.DataFrame(self.sumo_results)
        
        if len(text_df) == 0 or len(sumo_df) == 0:
            print("Error: No results to plot!")
            return
        
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1-4: Metrics vs Arrival Rate
        self._plot_vs_arrival_rate(text_df, sumo_df, plots_dir)
        
        # 5-8: Metrics vs Diameter (at fixed arrival rate)
        self._plot_vs_diameter(text_df, sumo_df, plots_dir)
        
        print(f"\n✓ All plots saved to: {plots_dir}")
    
    def _plot_vs_arrival_rate(self, text_df: pd.DataFrame, sumo_df: pd.DataFrame, 
                               plots_dir: Path):
        """Generate plots vs arrival rate."""
        metrics = [
            ('avg_delay', 'Average Delay (s)', '1_delay_vs_arrival.png'),
            ('throughput', 'Throughput (veh/hr)', '2_throughput_vs_arrival.png'),
            ('p95_delay', '95th Percentile Delay (s)', '3_p95_delay_vs_arrival.png'),
            ('max_queue', 'Max Queue Length (vehicles)', '4_max_queue_vs_arrival.png')
        ]
        
        for metric, ylabel, filename in metrics:
            # Check if metric exists
            if metric not in text_df.columns and metric not in sumo_df.columns:
                print(f"  Skipping {metric} (not available)")
                continue
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            colors = {'1': '#1f77b4', '2': '#ff7f0e', '3': '#2ca02c'}
            markers = {'text': 'o', 'sumo': 's'}
            
            for lanes in self.lane_counts:
                # Plot text results
                text_subset = text_df[text_df['lanes'] == lanes].groupby('arrival_rate').agg({
                    metric: 'mean'
                }).reset_index()
                
                if len(text_subset) > 0 and metric in text_subset.columns:
                    ax.plot(text_subset['arrival_rate'], text_subset[metric],
                           marker=markers['text'], linestyle='-', linewidth=2,
                           color=colors[str(lanes)], label=f'{lanes}-lane Text',
                           markersize=8, alpha=0.8)
                
                # Plot SUMO results
                sumo_subset = sumo_df[sumo_df['lanes'] == lanes].groupby('arrival_rate').agg({
                    metric: 'mean'
                }).reset_index()
                
                if len(sumo_subset) > 0 and metric in sumo_subset.columns:
                    ax.plot(sumo_subset['arrival_rate'], sumo_subset[metric],
                           marker=markers['sumo'], linestyle='--', linewidth=2,
                           color=colors[str(lanes)], label=f'{lanes}-lane SUMO',
                           markersize=8, alpha=0.8)
            
            ax.set_xlabel('Arrival Rate (veh/s per arm)', fontsize=13, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
            ax.set_title(f'{ylabel} vs Arrival Rate (Text vs SUMO)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=10, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = plots_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ {filename}")
    
    def _plot_vs_diameter(self, text_df: pd.DataFrame, sumo_df: pd.DataFrame,
                          plots_dir: Path):
        """Generate plots vs diameter at fixed arrival rate."""
        # Filter for fixed arrival rate
        text_fixed = text_df[text_df['arrival_rate'] == self.fixed_arrival]
        sumo_fixed = sumo_df[sumo_df['arrival_rate'] == self.fixed_arrival]
        
        if len(text_fixed) == 0 or len(sumo_fixed) == 0:
            print(f"  Warning: No data at arrival rate {self.fixed_arrival}")
            return
        
        metrics = [
            ('avg_delay', 'Average Delay (s)', '5_delay_vs_diameter.png'),
            ('throughput', 'Throughput (veh/hr)', '6_throughput_vs_diameter.png'),
            ('p95_delay', '95th Percentile Delay (s)', '7_p95_delay_vs_diameter.png'),
            ('max_queue', 'Max Queue Length (vehicles)', '8_max_queue_vs_diameter.png')
        ]
        
        for metric, ylabel, filename in metrics:
            if metric not in text_fixed.columns and metric not in sumo_fixed.columns:
                print(f"  Skipping {metric} (not available)")
                continue
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            colors = {'1': '#1f77b4', '2': '#ff7f0e', '3': '#2ca02c'}
            markers = {'text': 'o', 'sumo': 's'}
            
            for lanes in self.lane_counts:
                # Plot text results
                text_subset = text_fixed[text_fixed['lanes'] == lanes].groupby('diameter').agg({
                    metric: 'mean'
                }).reset_index()
                
                if len(text_subset) > 0 and metric in text_subset.columns:
                    ax.plot(text_subset['diameter'], text_subset[metric],
                           marker=markers['text'], linestyle='-', linewidth=2,
                           color=colors[str(lanes)], label=f'{lanes}-lane Text',
                           markersize=8, alpha=0.8)
                
                # Plot SUMO results
                sumo_subset = sumo_fixed[sumo_fixed['lanes'] == lanes].groupby('diameter').agg({
                    metric: 'mean'
                }).reset_index()
                
                if len(sumo_subset) > 0 and metric in sumo_subset.columns:
                    ax.plot(sumo_subset['diameter'], sumo_subset[metric],
                           marker=markers['sumo'], linestyle='--', linewidth=2,
                           color=colors[str(lanes)], label=f'{lanes}-lane SUMO',
                           markersize=8, alpha=0.8)
            
            ax.set_xlabel('Roundabout Diameter (m)', fontsize=13, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
            ax.set_title(f'{ylabel} vs Diameter at λ={self.fixed_arrival} veh/s/arm (Text vs SUMO)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=10, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = plots_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ {filename}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Generate Text vs SUMO roundabout comparison visualizations"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/text_vs_sumo_comparison',
        help='Output directory for results and plots'
    )
    parser.add_argument(
        '--skip-text',
        action='store_true',
        help='Skip text simulations (use existing results)'
    )
    parser.add_argument(
        '--skip-sumo',
        action='store_true',
        help='Skip SUMO simulations (use existing results)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ROUNDABOUT COMPARISON: TEXT vs SUMO")
    print("="*70)
    
    generator = RoundaboutComparisonGenerator(args.output_dir)
    
    # Run simulations
    if not args.skip_text:
        generator.run_text_simulations()
    else:
        # Load existing results
        csv_path = Path(args.output_dir) / 'text_simulation_results.csv'
        if csv_path.exists():
            generator.text_results = pd.read_csv(csv_path).to_dict('records')
            print(f"✓ Loaded {len(generator.text_results)} text results from {csv_path}")
    
    if not args.skip_sumo:
        generator.run_sumo_simulations()
    else:
        # Load existing results
        csv_path = Path(args.output_dir) / 'sumo_simulation_results.csv'
        if csv_path.exists():
            generator.sumo_results = pd.read_csv(csv_path).to_dict('records')
            print(f"✓ Loaded {len(generator.sumo_results)} SUMO results from {csv_path}")
    
    # Generate plots
    generator.generate_comparison_plots()
    
    print("\n" + "="*70)
    print("COMPARISON GENERATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Plots saved to: {args.output_dir}/plots/")


if __name__ == '__main__':
    main()
