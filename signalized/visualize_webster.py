#!/usr/bin/env python3
"""
Visualize Webster's Method Results

Creates plots showing:
1. Cycle length vs demand
2. Delay vs demand  
3. Green time allocation
4. Flow ratio vs demand
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from webster_method import WebsterSignalOptimizer

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def main():
    config_path = 'config/config.yaml'
    optimizer = WebsterSignalOptimizer(config_path=config_path)
    
    # Test range of demand multipliers
    demand_mults = np.linspace(0.3, 1.4, 30)
    
    results = {
        'demand_mult': [],
        'cycle_length': [],
        'avg_delay': [],
        'flow_ratio': [],
        'green_times': [],
        'valid': []
    }
    
    print("Running Webster optimization across demand range...")
    for dm in demand_mults:
        result = optimizer.optimize(demand_multiplier=dm, verbose=False)
        
        results['demand_mult'].append(dm)
        results['flow_ratio'].append(result['total_flow_ratio'])
        
        if result['cycle_length']:
            results['cycle_length'].append(result['cycle_length'])
            results['avg_delay'].append(result['avg_delay'])
            results['green_times'].append(result['green_times'])
            results['valid'].append(True)
        else:
            results['cycle_length'].append(None)
            results['avg_delay'].append(None)
            results['green_times'].append(None)
            results['valid'].append(False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Webster's Method: Signal Timing Optimization", 
                 fontsize=16, fontweight='bold')
    
    # --- Plot 1: Flow Ratio vs Demand ---
    ax1 = axes[0, 0]
    valid_dm = [dm for dm, v in zip(results['demand_mult'], results['valid']) if v]
    invalid_dm = [dm for dm, v in zip(results['demand_mult'], results['valid']) if not v]
    valid_y = [y for y, v in zip(results['flow_ratio'], results['valid']) if v]
    invalid_y = [y for y, v in zip(results['flow_ratio'], results['valid']) if not v]
    
    ax1.plot(valid_dm, valid_y, 'o-', color='#2ecc71', linewidth=2, 
             markersize=6, label='Stable (Y < 1.0)')
    if invalid_dm:
        ax1.plot(invalid_dm, invalid_y, 'o-', color='#e74c3c', linewidth=2,
                 markersize=6, label='Over-saturated (Y ≥ 1.0)')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='Capacity limit (Y = 1.0)')
    ax1.axhline(y=0.85, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                label='Practical limit (Y = 0.85)')
    ax1.set_xlabel('Demand Multiplier', fontweight='bold')
    ax1.set_ylabel('Total Flow Ratio (Y)', fontweight='bold')
    ax1.set_title('Critical Flow Ratio vs Demand')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Cycle Length vs Demand ---
    ax2 = axes[0, 1]
    valid_cycle = [c for c, v in zip(results['cycle_length'], results['valid']) if v and c]
    ax2.plot(valid_dm, valid_cycle, 'o-', color='#3498db', linewidth=2, markersize=6)
    ax2.axhline(y=90, color='green', linestyle=':', alpha=0.5, label='Typical optimal')
    ax2.axhline(y=120, color='orange', linestyle=':', alpha=0.5, label='Long cycle')
    ax2.set_xlabel('Demand Multiplier', fontweight='bold')
    ax2.set_ylabel('Optimal Cycle Length (seconds)', fontweight='bold')
    ax2.set_title('Cycle Length Optimization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Average Delay vs Demand ---
    ax3 = axes[1, 0]
    valid_delay = [d for d, v in zip(results['avg_delay'], results['valid']) if v and d]
    ax3.plot(valid_dm, valid_delay, 'o-', color='#e67e22', linewidth=2, markersize=6)
    ax3.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='60s threshold')
    ax3.axhline(y=120, color='red', linestyle='--', alpha=0.5, label='120s threshold')
    ax3.set_xlabel('Demand Multiplier', fontweight='bold')
    ax3.set_ylabel('Average Delay (seconds/vehicle)', fontweight='bold')
    ax3.set_title('Vehicle Delay vs Demand')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Green Time Allocation ---
    ax4 = axes[1, 1]
    
    # Extract green times for key demand levels
    key_demands = [0.5, 0.75, 1.0, 1.25]
    phases = ['NS\nLeft', 'NS\nThrough', 'EW\nLeft', 'EW\nThrough']
    
    x = np.arange(len(phases))
    width = 0.2
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    for i, dm in enumerate(key_demands):
        # Find closest result
        idx = min(range(len(results['demand_mult'])), 
                  key=lambda j: abs(results['demand_mult'][j] - dm))
        
        if results['valid'][idx] and results['green_times'][idx]:
            green_times = results['green_times'][idx]
            offset = (i - 1.5) * width
            bars = ax4.bar(x + offset, green_times, width, 
                          label=f'{dm}× demand', color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar, val in zip(bars, green_times):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}s', ha='center', va='bottom', fontsize=8)
    
    ax4.set_xlabel('Phase', fontweight='bold')
    ax4.set_ylabel('Green Time (seconds)', fontweight='bold')
    ax4.set_title('Green Time Allocation by Phase')
    ax4.set_xticks(x)
    ax4.set_xticklabels(phases)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = 'quickstart_output/plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'webster_optimization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")
    
    # Don't show interactively
    # plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("WEBSTER OPTIMIZATION SUMMARY")
    print("="*70)
    print(f"\nDemand Range Tested: {min(demand_mults):.2f}× to {max(demand_mults):.2f}×")
    print(f"Valid Solutions: {sum(results['valid'])} / {len(demand_mults)}")
    
    # Find capacity limit
    for i, (dm, y, valid) in enumerate(zip(results['demand_mult'], 
                                            results['flow_ratio'], 
                                            results['valid'])):
        if not valid:
            print(f"\n⚠️  Capacity Limit Reached:")
            print(f"   Maximum stable demand: ~{results['demand_mult'][i-1]:.2f}×")
            print(f"   Flow ratio at limit: Y = {results['flow_ratio'][i-1]:.3f}")
            break
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()
