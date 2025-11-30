#!/usr/bin/env python3
"""
Simple comparison plot generator - Uses existing text simulation data
and generates all 8 comparison graphs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
import re

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11

def collect_text_data():
    """Run text simulations and collect data."""
    print("="*70)
    print("Collecting Text Simulation Data")
    print("="*70)
    
    results = []
    
    # Parameter ranges
    lanes_list = [1, 2, 3]
    diameters = [30, 40, 50]
    arrivals = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    total = len(lanes_list) * len(diameters) * len(arrivals)
    count = 0
    
    for lanes in lanes_list:
        for diameter in diameters:
            for arrival in arrivals:
                count += 1
                print(f"\n[{count}/{total}] {lanes}-lane, d={diameter}m, λ={arrival:.2f} veh/s/arm", end=" ")
                
                cmd = [
                    'python3', 'Roundabout.py',
                    '--lanes', str(lanes),
                    '--diameter', str(diameter),
                    '--arrival', str(arrival), str(arrival), str(arrival), str(arrival),
                    '--horizon', '900',
                    '--seed', '42'
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                    output = result.stdout
                    
                    # Parse metrics
                    throughput = re.search(r'throughput[=\s]+(\d+\.?\d*)', output, re.I)
                    avg_delay = re.search(r'avg_delay[=\s]+(\d+\.?\d*)', output, re.I)
                    p95_delay = re.search(r'p95[=\s]+(\d+\.?\d*)', output, re.I)
                    max_queue = re.search(r'max_queue_per_arm=\[([^\]]+)\]', output, re.I)
                    
                    if throughput and avg_delay:
                        max_q = 0
                        if max_queue:
                            queues = [int(x.strip()) for x in max_queue.group(1).split(',')]
                            max_q = max(queues)
                        
                        results.append({
                            'lanes': lanes,
                            'diameter': diameter,
                            'arrival_rate': arrival,
                            'throughput': float(throughput.group(1)),
                            'avg_delay': float(avg_delay.group(1)),
                            'p95_delay': float(p95_delay.group(1)) if p95_delay else 0,
                            'max_queue': max_q
                        })
                        print(f"✓")
                    else:
                        print(f"✗ parse failed")
                        
                except Exception as e:
                    print(f"✗ {e}")
    
    return pd.DataFrame(results)

def generate_plots(df, output_dir):
    """Generate all 8 comparison plots."""
    print("\n" + "="*70)
    print("Generating Comparison Plots")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
    
    # Plots vs Arrival Rate
    metrics_arrival = [
        ('avg_delay', 'Average Delay (s)', '1_delay_vs_arrival.png'),
        ('throughput', 'Throughput (veh/hr)', '2_throughput_vs_arrival.png'),
        ('p95_delay', '95th Percentile Delay (s)', '3_p95_delay_vs_arrival.png'),
        ('max_queue', 'Max Queue Length (vehicles)', '4_max_queue_vs_arrival.png')
    ]
    
    for metric, ylabel, filename in metrics_arrival:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for lanes in [1, 2, 3]:
            subset = df[df['lanes'] == lanes].groupby('arrival_rate').agg({
                metric: 'mean'
            }).reset_index()
            
            if len(subset) > 0:
                ax.plot(subset['arrival_rate'], subset[metric],
                       marker='o', linestyle='-', linewidth=2.5,
                       color=colors[lanes], label=f'{lanes}-lane',
                       markersize=10, alpha=0.85)
        
        ax.set_xlabel('Arrival Rate (veh/s per arm)', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f'{ylabel} vs Arrival Rate', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {filename}")
    
    # Plots vs Diameter (at λ=0.10)
    df_fixed = df[df['arrival_rate'] == 0.10]
    
    metrics_diameter = [
        ('avg_delay', 'Average Delay (s)', '5_delay_vs_diameter.png'),
        ('throughput', 'Throughput (veh/hr)', '6_throughput_vs_diameter.png'),
        ('p95_delay', '95th Percentile Delay (s)', '7_p95_delay_vs_diameter.png'),
        ('max_queue', 'Max Queue Length (vehicles)', '8_max_queue_vs_diameter.png')
    ]
    
    for metric, ylabel, filename in metrics_diameter:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for lanes in [1, 2, 3]:
            subset = df_fixed[df_fixed['lanes'] == lanes].groupby('diameter').agg({
                metric: 'mean'
            }).reset_index()
            
            if len(subset) > 0:
                ax.plot(subset['diameter'], subset[metric],
                       marker='s', linestyle='-', linewidth=2.5,
                       color=colors[lanes], label=f'{lanes}-lane',
                       markersize=10, alpha=0.85)
        
        ax.set_xlabel('Roundabout Diameter (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f'{ylabel} vs Diameter (λ=0.10 veh/s/arm)', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {filename}")
    
    print(f"\n✓ All plots saved to: {output_dir}")

def main():
    output_dir = 'results/roundabout_comparisons'
    
    # Collect data
    df = collect_text_data()
    
    # Save data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / 'simulation_data.csv', index=False)
    print(f"\n✓ Data saved: {output_path / 'simulation_data.csv'}")
    print(f"  Total simulations: {len(df)}")
    
    # Generate plots
    generate_plots(df, output_dir)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nResults: {output_dir}/")
    print(f"  - simulation_data.csv")
    print(f"  - 1_delay_vs_arrival.png")
    print(f"  - 2_throughput_vs_arrival.png")
    print(f"  - 3_p95_delay_vs_arrival.png")
    print(f"  - 4_max_queue_vs_arrival.png")
    print(f"  - 5_delay_vs_diameter.png")
    print(f"  - 6_throughput_vs_diameter.png")
    print(f"  - 7_p95_delay_vs_diameter.png")
    print(f"  - 8_max_queue_vs_diameter.png")

if __name__ == '__main__':
    main()
