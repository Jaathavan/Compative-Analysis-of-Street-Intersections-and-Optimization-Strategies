"""
enhanced_visualizations.py - Signalized Intersection Visualization Suite
=========================================================================

Comprehensive visualizations for signalized intersection analysis including:
1. Signal timing optimization (Webster's Method)
2. Lane utilization and performance
3. Phase-specific analysis
4. Comparison with roundabout performance

Usage:
    python enhanced_visualizations.py --data results/raw/sweep_results.csv --output results/plots/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class SignalizedVisualizer:
    """
    Visualization suite for signalized intersection analysis.
    """
    
    def __init__(self, output_dir: str = "plots"):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("Set2")
        
        self.colors = {
            'webster': '#2ECC71',    # Green
            'ppo': '#3498DB',        # Blue
            'actuated': '#E74C3C',   # Red
            'fixed': '#F39C12',      # Orange
        }
    
    def visualize_webster_optimization(self, data: pd.DataFrame, save: bool = True):
        """
        Visualize Webster's Method optimization results.
        
        Expected columns: cycle_length, green_times, flow_ratio, delay_estimate
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Panel 1: Optimal cycle length vs demand
        ax1 = plt.subplot(2, 3, 1)
        if 'demand_multiplier' in data.columns and 'cycle_length' in data.columns:
            ax1.plot(data['demand_multiplier'], data['cycle_length'], 
                    marker='o', linewidth=2, color=self.colors['webster'])
            ax1.set_xlabel('Demand multiplier')
            ax1.set_ylabel('Optimal cycle length (s)')
            ax1.set_title("Webster's Optimal Cycle Length")
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='Min cycle')
            ax1.axhline(y=180, color='gray', linestyle='--', alpha=0.5, label='Max cycle')
            ax1.legend()
        
        # Panel 2: Green time allocation
        ax2 = plt.subplot(2, 3, 2)
        if all(col in data.columns for col in ['green_NS_L', 'green_NS_T', 'green_EW_L', 'green_EW_T']):
            phases = ['NS-L', 'NS-T', 'EW-L', 'EW-T']
            for phase, col in zip(phases, ['green_NS_L', 'green_NS_T', 'green_EW_L', 'green_EW_T']):
                ax2.plot(data['demand_multiplier'], data[col], 
                        marker='s', linewidth=2, label=phase)
            ax2.set_xlabel('Demand multiplier')
            ax2.set_ylabel('Green time (s)')
            ax2.set_title('Green Time Allocation by Phase')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Panel 3: Flow ratio (Y) vs demand
        ax3 = plt.subplot(2, 3, 3)
        if 'flow_ratio_Y' in data.columns:
            ax3.plot(data['demand_multiplier'], data['flow_ratio_Y'], 
                    marker='D', linewidth=2, color='darkred')
            ax3.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, 
                       label='Optimal (Y=0.9)')
            ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                       label='Saturation (Y=1.0)')
            ax3.set_xlabel('Demand multiplier')
            ax3.set_ylabel('Critical flow ratio (Y)')
            ax3.set_title('System Flow Ratio')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1.2])
        
        # Panel 4: Predicted delay vs actual delay
        ax4 = plt.subplot(2, 3, 4)
        if 'delay_predicted' in data.columns and 'delay_actual' in data.columns:
            ax4.scatter(data['delay_predicted'], data['delay_actual'], 
                       alpha=0.6, s=80, c=data['demand_multiplier'], 
                       cmap='viridis')
            
            # Add diagonal line (perfect prediction)
            min_val = min(data['delay_predicted'].min(), data['delay_actual'].min())
            max_val = max(data['delay_predicted'].max(), data['delay_actual'].max())
            ax4.plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect prediction')
            
            ax4.set_xlabel('Predicted delay (s) - Webster')
            ax4.set_ylabel('Actual delay (s) - Simulation')
            ax4.set_title('Webster Delay Prediction Accuracy')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            cbar = plt.colorbar(ax4.collections[0], ax=ax4)
            cbar.set_label('Demand multiplier')
        
        # Panel 5: Throughput vs demand
        ax5 = plt.subplot(2, 3, 5)
        if 'throughput' in data.columns:
            ax5.plot(data['demand_multiplier'], data['throughput'], 
                    marker='o', linewidth=2, color=self.colors['webster'])
            ax5.set_xlabel('Demand multiplier')
            ax5.set_ylabel('Throughput (veh/hr)')
            ax5.set_title('System Throughput')
            ax5.grid(True, alpha=0.3)
        
        # Panel 6: Capacity utilization
        ax6 = plt.subplot(2, 3, 6)
        if 'lanes' in data.columns and 'throughput' in data.columns:
            # Theoretical capacity ≈ 1800 veh/hr/lane × 4 arms × lanes
            theoretical_capacity = 1800 * 4 * data['lanes']
            utilization = (data['throughput'] / theoretical_capacity * 100).clip(upper=100)
            
            ax6.plot(data['demand_multiplier'], utilization, 
                    marker='*', linewidth=2, markersize=10, color='purple')
            ax6.axhline(y=85, color='orange', linestyle='--', alpha=0.7, 
                       label='Target (85%)')
            ax6.set_xlabel('Demand multiplier')
            ax6.set_ylabel('Capacity utilization (%)')
            ax6.set_title('System Capacity Utilization')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'webster_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved Webster analysis to {self.output_dir / 'webster_analysis.png'}")
        plt.show()
    
    def compare_control_strategies(self, data: pd.DataFrame, save: bool = True):
        """
        Compare Webster, PPO, and Actuated control strategies.
        
        Expected columns: strategy, demand_multiplier, throughput, avg_delay, max_queue
        """
        fig = plt.figure(figsize=(16, 8))
        
        strategies = data['strategy'].unique() if 'strategy' in data.columns else ['webster']
        
        # Panel 1: Throughput comparison
        ax1 = plt.subplot(2, 3, 1)
        for strategy in strategies:
            df_strat = data[data['strategy'] == strategy]
            ax1.plot(df_strat['demand_multiplier'], df_strat['throughput'], 
                    marker='o', linewidth=2, label=strategy.title(),
                    color=self.colors.get(strategy, 'gray'))
        ax1.set_xlabel('Demand multiplier')
        ax1.set_ylabel('Throughput (veh/hr)')
        ax1.set_title('Throughput Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Average delay comparison
        ax2 = plt.subplot(2, 3, 2)
        for strategy in strategies:
            df_strat = data[data['strategy'] == strategy]
            ax2.plot(df_strat['demand_multiplier'], df_strat['avg_delay'], 
                    marker='s', linewidth=2, label=strategy.title(),
                    color=self.colors.get(strategy, 'gray'))
        ax2.set_xlabel('Demand multiplier')
        ax2.set_ylabel('Average delay (s)')
        ax2.set_title('Average Delay Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Panel 3: Queue length comparison
        ax3 = plt.subplot(2, 3, 3)
        for strategy in strategies:
            df_strat = data[data['strategy'] == strategy]
            if 'max_queue' in df_strat.columns:
                ax3.plot(df_strat['demand_multiplier'], df_strat['max_queue'], 
                        marker='^', linewidth=2, label=strategy.title(),
                        color=self.colors.get(strategy, 'gray'))
        ax3.set_xlabel('Demand multiplier')
        ax3.set_ylabel('Max queue length (veh)')
        ax3.set_title('Queue Length Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Relative performance (normalized to Webster)
        ax4 = plt.subplot(2, 3, 4)
        if 'webster' in strategies:
            webster_baseline = data[data['strategy'] == 'webster']
            for strategy in strategies:
                if strategy == 'webster':
                    continue
                df_strat = data[data['strategy'] == strategy]
                
                # Merge on demand_multiplier
                merged = pd.merge(df_strat, webster_baseline, 
                                 on='demand_multiplier', suffixes=('', '_baseline'))
                
                relative_throughput = ((merged['throughput'] - merged['throughput_baseline']) / 
                                      merged['throughput_baseline'] * 100)
                
                ax4.plot(merged['demand_multiplier'], relative_throughput, 
                        marker='o', linewidth=2, label=strategy.title(),
                        color=self.colors.get(strategy, 'gray'))
            
            ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax4.set_xlabel('Demand multiplier')
            ax4.set_ylabel('Throughput improvement vs Webster (%)')
            ax4.set_title('Relative Performance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Panel 5: Efficiency score
        ax5 = plt.subplot(2, 3, 5)
        for strategy in strategies:
            df_strat = data[data['strategy'] == strategy]
            # Efficiency = throughput / (avg_delay + 1)
            efficiency = df_strat['throughput'] / (df_strat['avg_delay'] + 1)
            ax5.plot(df_strat['demand_multiplier'], efficiency, 
                    marker='D', linewidth=2, label=strategy.title(),
                    color=self.colors.get(strategy, 'gray'))
        ax5.set_xlabel('Demand multiplier')
        ax5.set_ylabel('Efficiency score')
        ax5.set_title('Overall Efficiency (Throughput/Delay)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Summary table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_lines = ["=== Strategy Comparison Summary ===\n"]
        
        for strategy in strategies:
            df_strat = data[data['strategy'] == strategy]
            summary_lines.append(f"{strategy.upper()}:")
            summary_lines.append(f"  Max throughput: {df_strat['throughput'].max():.0f} veh/hr")
            summary_lines.append(f"  Min avg delay: {df_strat['avg_delay'].min():.1f}s")
            if 'max_queue' in df_strat.columns:
                summary_lines.append(f"  Min max queue: {df_strat['max_queue'].min():.0f} veh")
            summary_lines.append("")
        
        ax6.text(0.1, 0.9, '\n'.join(summary_lines),
                transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'strategy_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved strategy comparison to {self.output_dir / 'strategy_comparison.png'}")
        plt.show()
    
    def visualize_roundabout_vs_signalized(self, roundabout_data: pd.DataFrame, 
                                           signalized_data: pd.DataFrame, save: bool = True):
        """
        Compare roundabout vs signalized intersection performance.
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Panel 1: Throughput comparison
        ax1 = plt.subplot(2, 3, 1)
        for lanes in sorted(roundabout_data['lanes'].unique()):
            rb_df = roundabout_data[roundabout_data['lanes'] == lanes]
            ax1.plot(rb_df['arrival_rate'], rb_df['throughput'], 
                    marker='o', linewidth=2, label=f'Roundabout {lanes}L', 
                    linestyle='--', alpha=0.7)
        
        for lanes in sorted(signalized_data['lanes'].unique()):
            sig_df = signalized_data[signalized_data['lanes'] == lanes]
            ax1.plot(sig_df['arrival_rate'], sig_df['throughput'], 
                    marker='s', linewidth=2, label=f'Signalized {lanes}L',
                    linestyle='-')
        
        ax1.set_xlabel('Arrival rate per arm (veh/s)')
        ax1.set_ylabel('Throughput (veh/hr)')
        ax1.set_title('Throughput: Roundabout vs Signalized')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Average delay comparison
        ax2 = plt.subplot(2, 3, 2)
        for lanes in sorted(roundabout_data['lanes'].unique()):
            rb_df = roundabout_data[roundabout_data['lanes'] == lanes]
            ax2.plot(rb_df['arrival_rate'], rb_df['avg_delay'], 
                    marker='o', linewidth=2, label=f'Roundabout {lanes}L',
                    linestyle='--', alpha=0.7)
        
        for lanes in sorted(signalized_data['lanes'].unique()):
            sig_df = signalized_data[signalized_data['lanes'] == lanes]
            ax2.plot(sig_df['arrival_rate'], sig_df['avg_delay'], 
                    marker='s', linewidth=2, label=f'Signalized {lanes}L',
                    linestyle='-')
        
        ax2.set_xlabel('Arrival rate per arm (veh/s)')
        ax2.set_ylabel('Average delay (s)')
        ax2.set_title('Average Delay: Roundabout vs Signalized')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Panel 3: Queue length comparison
        ax3 = plt.subplot(2, 3, 3)
        for lanes in sorted(roundabout_data['lanes'].unique()):
            rb_df = roundabout_data[roundabout_data['lanes'] == lanes]
            if 'max_queue' in rb_df.columns:
                ax3.plot(rb_df['arrival_rate'], rb_df['max_queue'], 
                        marker='o', linewidth=2, label=f'Roundabout {lanes}L',
                        linestyle='--', alpha=0.7)
        
        for lanes in sorted(signalized_data['lanes'].unique()):
            sig_df = signalized_data[signalized_data['lanes'] == lanes]
            if 'max_queue' in sig_df.columns:
                ax3.plot(sig_df['arrival_rate'], sig_df['max_queue'], 
                        marker='s', linewidth=2, label=f'Signalized {lanes}L',
                        linestyle='-')
        
        ax3.set_xlabel('Arrival rate per arm (veh/s)')
        ax3.set_ylabel('Max queue length (veh)')
        ax3.set_title('Queue Length: Roundabout vs Signalized')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Breaking point analysis
        ax4 = plt.subplot(2, 3, 4)
        
        def find_breaking_point(df, threshold=120):
            """Find arrival rate where avg_delay exceeds threshold."""
            failing = df[df['avg_delay'] > threshold]
            if not failing.empty:
                return failing['arrival_rate'].min()
            return df['arrival_rate'].max()
        
        rb_breaking = []
        sig_breaking = []
        lane_configs = []
        
        for lanes in sorted(set(roundabout_data['lanes'].unique()) | set(signalized_data['lanes'].unique())):
            if lanes in roundabout_data['lanes'].values:
                rb_df = roundabout_data[roundabout_data['lanes'] == lanes]
                rb_bp = find_breaking_point(rb_df)
                rb_breaking.append(rb_bp)
            else:
                rb_breaking.append(0)
            
            if lanes in signalized_data['lanes'].values:
                sig_df = signalized_data[signalized_data['lanes'] == lanes]
                sig_bp = find_breaking_point(sig_df)
                sig_breaking.append(sig_bp)
            else:
                sig_breaking.append(0)
            
            lane_configs.append(f'{lanes}L')
        
        x = np.arange(len(lane_configs))
        width = 0.35
        
        ax4.bar(x - width/2, rb_breaking, width, label='Roundabout', alpha=0.8)
        ax4.bar(x + width/2, sig_breaking, width, label='Signalized', alpha=0.8)
        ax4.set_xlabel('Lane configuration')
        ax4.set_ylabel('Breaking point (veh/s/arm)')
        ax4.set_title('System Capacity Breaking Points')
        ax4.set_xticks(x)
        ax4.set_xticklabels(lane_configs)
        ax4.legend()
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Panel 5: Efficiency comparison
        ax5 = plt.subplot(2, 3, 5)
        
        # Compute efficiency for both
        rb_efficiency = roundabout_data['throughput'] / (roundabout_data['avg_delay'] + 1)
        sig_efficiency = signalized_data['throughput'] / (signalized_data['avg_delay'] + 1)
        
        for lanes in sorted(roundabout_data['lanes'].unique()):
            rb_df = roundabout_data[roundabout_data['lanes'] == lanes]
            rb_eff = rb_df['throughput'] / (rb_df['avg_delay'] + 1)
            ax5.plot(rb_df['arrival_rate'], rb_eff, 
                    marker='o', linewidth=2, label=f'Roundabout {lanes}L',
                    linestyle='--', alpha=0.7)
        
        for lanes in sorted(signalized_data['lanes'].unique()):
            sig_df = signalized_data[signalized_data['lanes'] == lanes]
            sig_eff = sig_df['throughput'] / (sig_df['avg_delay'] + 1)
            ax5.plot(sig_df['arrival_rate'], sig_eff, 
                    marker='s', linewidth=2, label=f'Signalized {lanes}L',
                    linestyle='-')
        
        ax5.set_xlabel('Arrival rate per arm (veh/s)')
        ax5.set_ylabel('Efficiency score')
        ax5.set_title('Efficiency: Roundabout vs Signalized')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Summary and recommendations
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_lines = ["=== Intersection Comparison ===\n"]
        
        # Analyze which is better at different demand levels
        summary_lines.append("Low Demand (λ < 0.10):")
        low_rb = roundabout_data[roundabout_data['arrival_rate'] < 0.10]['avg_delay'].mean()
        low_sig = signalized_data[signalized_data['arrival_rate'] < 0.10]['avg_delay'].mean()
        if low_rb < low_sig:
            summary_lines.append("  ✓ Roundabout better")
            summary_lines.append(f"    ({low_rb:.1f}s vs {low_sig:.1f}s)")
        else:
            summary_lines.append("  ✓ Signalized better")
            summary_lines.append(f"    ({low_sig:.1f}s vs {low_rb:.1f}s)")
        
        summary_lines.append("\nMedium Demand (0.10-0.15):")
        med_rb = roundabout_data[(roundabout_data['arrival_rate'] >= 0.10) & 
                                 (roundabout_data['arrival_rate'] < 0.15)]['avg_delay'].mean()
        med_sig = signalized_data[(signalized_data['arrival_rate'] >= 0.10) & 
                                  (signalized_data['arrival_rate'] < 0.15)]['avg_delay'].mean()
        if med_rb < med_sig:
            summary_lines.append("  ✓ Roundabout better")
            summary_lines.append(f"    ({med_rb:.1f}s vs {med_sig:.1f}s)")
        else:
            summary_lines.append("  ✓ Signalized better")
            summary_lines.append(f"    ({med_sig:.1f}s vs {med_rb:.1f}s)")
        
        summary_lines.append("\nHigh Demand (λ ≥ 0.15):")
        high_rb = roundabout_data[roundabout_data['arrival_rate'] >= 0.15]['avg_delay'].mean()
        high_sig = signalized_data[signalized_data['arrival_rate'] >= 0.15]['avg_delay'].mean()
        if high_rb < high_sig:
            summary_lines.append("  ✓ Roundabout better")
            summary_lines.append(f"    ({high_rb:.1f}s vs {high_sig:.1f}s)")
        else:
            summary_lines.append("  ✓ Signalized better")
            summary_lines.append(f"    ({high_sig:.1f}s vs {high_rb:.1f}s)")
        
        ax6.text(0.05, 0.95, '\n'.join(summary_lines),
                transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'roundabout_vs_signalized.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison to {self.output_dir / 'roundabout_vs_signalized.png'}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Signalized Intersection Visualization Suite")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV file with simulation results')
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--mode', type=str, default='webster',
                       choices=['webster', 'strategy_comparison', 'roundabout_comparison', 'all'],
                       help='Visualization mode')
    parser.add_argument('--roundabout-data', type=str, default=None,
                       help='Path to roundabout data for comparison (optional)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = pd.read_csv(args.data)
    print(f"Loaded {len(data)} records")
    
    # Initialize visualizer
    viz = SignalizedVisualizer(output_dir=args.output)
    
    # Generate requested visualizations
    if args.mode in ['all', 'webster']:
        print("\nGenerating Webster's Method analysis...")
        viz.visualize_webster_optimization(data)
    
    if args.mode in ['all', 'strategy_comparison']:
        print("\nGenerating strategy comparison...")
        viz.compare_control_strategies(data)
    
    if args.mode in ['all', 'roundabout_comparison'] and args.roundabout_data:
        print("\nGenerating roundabout vs signalized comparison...")
        rb_data = pd.read_csv(args.roundabout_data)
        viz.visualize_roundabout_vs_signalized(rb_data, data)
    
    print("\n✅ All visualizations complete!")


if __name__ == "__main__":
    main()
