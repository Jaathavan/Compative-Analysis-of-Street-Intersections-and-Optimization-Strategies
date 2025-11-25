#!/usr/bin/env python3
"""
enhanced_visualizations.py - Advanced Roundabout Performance Visualization Suite
=================================================================================

Creates comprehensive visualizations for roundabout analysis including:
1. Lane choice analysis and impact on performance
2. Diameter vs lanes parameter sweep with breaking points
3. Grid search and Bayesian optimization results
4. Multi-dimensional performance surfaces
5. Failure mode demonstrations

Inspired by visualizations.txt but significantly enhanced.

Usage:
    python enhanced_visualizations.py --results results/raw/sweep_results.csv --output results/plots/
    python enhanced_visualizations.py --mode lane_analysis --data results/raw/multilane_data.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import json


class RoundaboutVisualizer:
    """
    Comprehensive visualization suite for roundabout performance analysis.
    """
    
    def __init__(self, output_dir: str = "plots"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("Set2")
        self.colors = {
            '1_lane': '#E74C3C',    # Red
            '2_lanes': '#F39C12',   # Orange
            '3_lanes': '#3498DB',   # Blue
        }
    
    # ========================================================================
    # 1. LANE CHOICE ANALYSIS
    # ========================================================================
    
    def visualize_lane_choice_impact(self, data: pd.DataFrame, save: bool = True):
        """
        Analyze how lane choice strategy impacts performance metrics.
        
        Expected data columns:
        - lanes: number of circulating lanes
        - arrival_rate: per-arm arrival rate (veh/s)
        - lane_0_entries, lane_1_entries, lane_2_entries: vehicles entering each lane
        - lane_0_delay, lane_1_delay, lane_2_delay: average delay per lane
        - throughput, avg_delay, p95_delay, max_queue
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Panel 1: Lane utilization vs arrival rate
        ax1 = plt.subplot(2, 3, 1)
        for lanes in [2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            # Calculate lane utilization percentages
            total_entries = sum([df_lane[f'lane_{i}_entries'].fillna(0) for i in range(lanes)])
            for i in range(lanes):
                if f'lane_{i}_entries' in df_lane.columns:
                    utilization = (df_lane[f'lane_{i}_entries'] / total_entries * 100).fillna(0)
                    ax1.plot(df_lane['arrival_rate'], utilization, 
                            marker='o', label=f'{lanes} lanes - Lane {i}')
        
        ax1.set_xlabel('Arrival rate per arm (veh/s)')
        ax1.set_ylabel('Lane utilization (%)')
        ax1.set_title('Lane Utilization Distribution')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Delay by lane choice
        ax2 = plt.subplot(2, 3, 2)
        lane_delays = []
        lane_labels = []
        for lanes in [2, 3]:
            df_lane = data[(data['lanes'] == lanes) & (data['arrival_rate'] == 0.10)]
            if df_lane.empty:
                continue
            for i in range(lanes):
                if f'lane_{i}_delay' in df_lane.columns:
                    delay_vals = df_lane[f'lane_{i}_delay'].dropna()
                    if not delay_vals.empty:
                        lane_delays.append(delay_vals.values)
                        lane_labels.append(f'{lanes}L-Lane{i}')
        
        if lane_delays:
            bp = ax2.boxplot(lane_delays, labels=lane_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], plt.cm.Set2.colors):
                patch.set_facecolor(color)
        ax2.set_ylabel('Delay (s)')
        ax2.set_title('Delay Distribution by Lane Choice (λ=0.10)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Lane-specific throughput contribution
        ax3 = plt.subplot(2, 3, 3)
        for lanes in [2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            for i in range(lanes):
                if f'lane_{i}_entries' in df_lane.columns:
                    throughput_contribution = df_lane[f'lane_{i}_entries'] * 3600 / df_lane['horizon']
                    ax3.plot(df_lane['arrival_rate'], throughput_contribution,
                            marker='s', label=f'{lanes} lanes - Lane {i}', alpha=0.7)
        
        ax3.set_xlabel('Arrival rate per arm (veh/s)')
        ax3.set_ylabel('Throughput contribution (veh/hr)')
        ax3.set_title('Per-Lane Throughput Contribution')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Lane choice efficiency score
        ax4 = plt.subplot(2, 3, 4)
        for lanes in [2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            # Efficiency = throughput / (lanes * max_theoretical_throughput)
            # max_theoretical_throughput ≈ 1800 veh/hr per lane
            efficiency = (df_lane['throughput'] / (lanes * 1800) * 100).fillna(0)
            ax4.plot(df_lane['arrival_rate'], efficiency,
                    marker='D', linewidth=2, label=f'{lanes} lanes')
        
        ax4.set_xlabel('Arrival rate per arm (veh/s)')
        ax4.set_ylabel('System efficiency (%)')
        ax4.set_title('Lane Capacity Utilization Efficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Target 70%')
        
        # Panel 5: Queue imbalance across lanes
        ax5 = plt.subplot(2, 3, 5)
        for lanes in [2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            # Calculate coefficient of variation in lane utilization
            lane_entries = [df_lane[f'lane_{i}_entries'].fillna(0) for i in range(lanes)]
            cv = np.std(lane_entries, axis=0) / (np.mean(lane_entries, axis=0) + 1e-9)
            ax5.plot(df_lane['arrival_rate'], cv,
                    marker='*', linewidth=2, label=f'{lanes} lanes')
        
        ax5.set_xlabel('Arrival rate per arm (veh/s)')
        ax5.set_ylabel('Coefficient of variation')
        ax5.set_title('Lane Utilization Balance')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Lane-specific merge denial rates
        ax6 = plt.subplot(2, 3, 6)
        for lanes in [2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            for i in range(lanes):
                if f'lane_{i}_denials' in df_lane.columns and f'lane_{i}_attempts' in df_lane.columns:
                    denial_rate = (df_lane[f'lane_{i}_denials'] / 
                                  df_lane[f'lane_{i}_attempts'].replace(0, 1) * 100).fillna(0)
                    ax6.plot(df_lane['arrival_rate'], denial_rate,
                            marker='v', label=f'{lanes}L-Lane{i}', alpha=0.7)
        
        ax6.set_xlabel('Arrival rate per arm (veh/s)')
        ax6.set_ylabel('Merge denial rate (%)')
        ax6.set_title('Lane-Specific Merge Denial Rates')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'lane_choice_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved lane choice analysis to {self.output_dir / 'lane_choice_analysis.png'}")
        plt.show()
    
    # ========================================================================
    # 2. DIAMETER & LANES PARAMETER IMPACT
    # ========================================================================
    
    def visualize_parameter_sweep(self, data: pd.DataFrame, save: bool = True):
        """
        Visualize impact of diameter and lane count on performance metrics.
        Shows breaking points where system fails.
        
        Expected columns: diameter, lanes, arrival_rate, throughput, avg_delay, 
                         p95_delay, max_queue_N/E/S/W
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Arrival rates to analyze
        arrival_rates = sorted(data['arrival_rate'].unique())
        
        # Identify breaking points (where delay > 120s or throughput plateaus)
        breaking_points = {}
        for lanes in [2, 3]:
            df_lane = data[data['lanes'] == lanes].sort_values('arrival_rate')
            # Find where avg_delay exceeds 120s
            bp = df_lane[df_lane['avg_delay'] > 120]
            if not bp.empty:
                breaking_points[lanes] = bp['arrival_rate'].min()
            else:
                breaking_points[lanes] = df_lane['arrival_rate'].max()
        
        # Helper function to add breaking point lines
        def add_break_lines(ax):
            for lanes, bp_val in breaking_points.items():
                color = self.colors.get(f'{lanes}_lanes', 'gray')
                ax.axvline(bp_val, linestyle='--', linewidth=1.5, alpha=0.7,
                          color=color, label=f'{lanes}-lane breakpoint')
        
        # Panel 1: Throughput vs arrival rate (multi-lane comparison)
        ax1 = plt.subplot(3, 3, 1)
        for lanes in [1, 2, 3]:
            df_lane = data[(data['lanes'] == lanes) & (data['diameter'] == 45)]
            if df_lane.empty:
                continue
            ax1.plot(df_lane['arrival_rate'], df_lane['throughput'],
                    marker='o', linewidth=2, label=f'{lanes} lane(s)',
                    color=self.colors.get(f'{lanes}_lane', 'gray'))
        add_break_lines(ax1)
        ax1.set_xlabel('Arrival rate per arm (veh/s)')
        ax1.set_ylabel('Throughput (veh/hr)')
        ax1.set_title('Throughput vs Arrival Rate (D=45m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Average delay vs arrival rate
        ax2 = plt.subplot(3, 3, 2)
        for lanes in [1, 2, 3]:
            df_lane = data[(data['lanes'] == lanes) & (data['diameter'] == 45)]
            if df_lane.empty:
                continue
            ax2.plot(df_lane['arrival_rate'], df_lane['avg_delay'],
                    marker='s', linewidth=2, label=f'{lanes} lane(s)',
                    color=self.colors.get(f'{lanes}_lane', 'gray'))
        add_break_lines(ax2)
        ax2.set_xlabel('Arrival rate per arm (veh/s)')
        ax2.set_ylabel('Average delay (s)')
        ax2.set_title('Average Delay vs Arrival Rate (D=45m)')
        ax2.axhline(y=120, color='red', linestyle=':', alpha=0.5, label='Failure threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Panel 3: 95th percentile delay
        ax3 = plt.subplot(3, 3, 3)
        for lanes in [1, 2, 3]:
            df_lane = data[(data['lanes'] == lanes) & (data['diameter'] == 45)]
            if df_lane.empty:
                continue
            ax3.plot(df_lane['arrival_rate'], df_lane['p95_delay'],
                    marker='^', linewidth=2, label=f'{lanes} lane(s)',
                    color=self.colors.get(f'{lanes}_lane', 'gray'))
        add_break_lines(ax3)
        ax3.set_xlabel('Arrival rate per arm (veh/s)')
        ax3.set_ylabel('P95 delay (s)')
        ax3.set_title('95th Percentile Delay (D=45m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Panel 4: Throughput vs diameter (fixed arrival rate)
        ax4 = plt.subplot(3, 3, 4)
        arrival_ref = 0.10  # Reference arrival rate
        diameters = sorted(data['diameter'].unique())
        for lanes in [1, 2, 3]:
            df_subset = data[(data['lanes'] == lanes) & (data['arrival_rate'] == arrival_ref)]
            if df_subset.empty:
                continue
            ax4.plot(df_subset['diameter'], df_subset['throughput'],
                    marker='o', linewidth=2, label=f'{lanes} lane(s)',
                    color=self.colors.get(f'{lanes}_lane', 'gray'))
        ax4.set_xlabel('Roundabout diameter (m)')
        ax4.set_ylabel('Throughput (veh/hr)')
        ax4.set_title(f'Throughput vs Diameter (λ={arrival_ref} veh/s/arm)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Delay vs diameter
        ax5 = plt.subplot(3, 3, 5)
        for lanes in [1, 2, 3]:
            df_subset = data[(data['lanes'] == lanes) & (data['arrival_rate'] == arrival_ref)]
            if df_subset.empty:
                continue
            ax5.plot(df_subset['diameter'], df_subset['avg_delay'],
                    marker='s', linewidth=2, label=f'{lanes} lane(s)',
                    color=self.colors.get(f'{lanes}_lane', 'gray'))
        ax5.set_xlabel('Roundabout diameter (m)')
        ax5.set_ylabel('Average delay (s)')
        ax5.set_title(f'Average Delay vs Diameter (λ={arrival_ref})')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Maximum queue vs diameter
        ax6 = plt.subplot(3, 3, 6)
        for lanes in [1, 2, 3]:
            df_subset = data[(data['lanes'] == lanes) & (data['arrival_rate'] == arrival_ref)]
            if df_subset.empty:
                continue
            max_q = df_subset[['max_queue_N', 'max_queue_E', 'max_queue_S', 'max_queue_W']].max(axis=1)
            ax6.plot(df_subset['diameter'], max_q,
                    marker='D', linewidth=2, label=f'{lanes} lane(s)',
                    color=self.colors.get(f'{lanes}_lane', 'gray'))
        ax6.set_xlabel('Roundabout diameter (m)')
        ax6.set_ylabel('Maximum queue length (veh)')
        ax6.set_title(f'Max Queue vs Diameter (λ={arrival_ref})')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Panel 7: Heatmap - Throughput as function of (diameter, lanes)
        ax7 = plt.subplot(3, 3, 7)
        pivot_data = data[data['arrival_rate'] == arrival_ref].pivot_table(
            values='throughput', index='diameter', columns='lanes', aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax7, cbar_kws={'label': 'Throughput (veh/hr)'})
        ax7.set_title(f'Throughput Heatmap (λ={arrival_ref})')
        ax7.set_xlabel('Number of lanes')
        ax7.set_ylabel('Diameter (m)')
        
        # Panel 8: Heatmap - Average delay
        ax8 = plt.subplot(3, 3, 8)
        pivot_delay = data[data['arrival_rate'] == arrival_ref].pivot_table(
            values='avg_delay', index='diameter', columns='lanes', aggfunc='mean'
        )
        sns.heatmap(pivot_delay, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax8, cbar_kws={'label': 'Avg Delay (s)'})
        ax8.set_title(f'Average Delay Heatmap (λ={arrival_ref})')
        ax8.set_xlabel('Number of lanes')
        ax8.set_ylabel('Diameter (m)')
        
        # Panel 9: Breaking point analysis
        ax9 = plt.subplot(3, 3, 9)
        bp_data = []
        for lanes in [1, 2, 3]:
            for diameter in sorted(data['diameter'].unique()):
                df_subset = data[(data['lanes'] == lanes) & (data['diameter'] == diameter)]
                if df_subset.empty:
                    continue
                # Find breaking point
                failing = df_subset[df_subset['avg_delay'] > 120]
                if not failing.empty:
                    bp_arrival = failing['arrival_rate'].min()
                else:
                    bp_arrival = df_subset['arrival_rate'].max()
                bp_data.append({'diameter': diameter, 'lanes': lanes, 'breaking_point': bp_arrival})
        
        if bp_data:
            bp_df = pd.DataFrame(bp_data)
            for lanes in [1, 2, 3]:
                subset = bp_df[bp_df['lanes'] == lanes]
                if not subset.empty:
                    ax9.plot(subset['diameter'], subset['breaking_point'],
                            marker='*', linewidth=2, markersize=12, label=f'{lanes} lane(s)',
                            color=self.colors.get(f'{lanes}_lane', 'gray'))
        
        ax9.set_xlabel('Roundabout diameter (m)')
        ax9.set_ylabel('Breaking point arrival rate (veh/s/arm)')
        ax9.set_title('System Capacity Breaking Points')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'parameter_sweep_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved parameter sweep to {self.output_dir / 'parameter_sweep_analysis.png'}")
        plt.show()
    
    # ========================================================================
    # 3. OPTIMIZATION RESULTS VISUALIZATION
    # ========================================================================
    
    def visualize_optimization_results(self, optimization_data: Dict, save: bool = True):
        """
        Visualize grid search and Bayesian optimization results.
        
        Expected structure:
        {
            'grid_search': pd.DataFrame with columns [diameter, lanes, arrival_rate, objective, ...],
            'bayesian_opt': pd.DataFrame with columns [iteration, diameter, lanes, objective, ...]
        }
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Panel 1: Grid search - 3D surface plot
        if 'grid_search' in optimization_data:
            ax1 = plt.subplot(2, 3, 1, projection='3d')
            grid_df = optimization_data['grid_search']
            
            # Create surface for 2-lane configuration
            df_2lane = grid_df[grid_df['lanes'] == 2]
            if not df_2lane.empty:
                x = df_2lane['diameter'].values
                y = df_2lane['arrival_rate'].values
                z = df_2lane['objective'].values  # Lower is better (e.g., avg_delay)
                
                # Create grid
                xi = np.linspace(x.min(), x.max(), 20)
                yi = np.linspace(y.min(), y.max(), 20)
                xi, yi = np.meshgrid(xi, yi)
                
                # Interpolate
                from scipy.interpolate import griddata
                zi = griddata((x, y), z, (xi, yi), method='cubic')
                
                surf = ax1.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.8)
                ax1.scatter(x, y, z, c='red', marker='o', s=50)
                
                ax1.set_xlabel('Diameter (m)')
                ax1.set_ylabel('Arrival rate (veh/s)')
                ax1.set_zlabel('Objective (avg delay)')
                ax1.set_title('Grid Search Surface (2 lanes)')
                fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        # Panel 2: Bayesian optimization convergence
        if 'bayesian_opt' in optimization_data:
            ax2 = plt.subplot(2, 3, 2)
            bayes_df = optimization_data['bayesian_opt']
            
            if 'iteration' in bayes_df.columns:
                # Plot best objective over iterations
                best_so_far = bayes_df['objective'].cummin()
                ax2.plot(bayes_df['iteration'], bayes_df['objective'], 
                        'o-', alpha=0.5, label='Sampled points')
                ax2.plot(bayes_df['iteration'], best_so_far, 
                        'r-', linewidth=2, label='Best so far')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Objective value')
                ax2.set_title('Bayesian Optimization Convergence')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Panel 3: Optimal parameter distribution
        if 'grid_search' in optimization_data:
            ax3 = plt.subplot(2, 3, 3)
            grid_df = optimization_data['grid_search']
            
            # Find top 10% configurations
            threshold = grid_df['objective'].quantile(0.1)
            top_configs = grid_df[grid_df['objective'] <= threshold]
            
            diameter_counts = top_configs['diameter'].value_counts().sort_index()
            ax3.bar(diameter_counts.index, diameter_counts.values, 
                   color='skyblue', edgecolor='black')
            ax3.set_xlabel('Diameter (m)')
            ax3.set_ylabel('Frequency in top 10%')
            ax3.set_title('Optimal Diameter Distribution')
            ax3.grid(True, axis='y', alpha=0.3)
        
        # Panel 4: Objective vs diameter for different lane counts
        if 'grid_search' in optimization_data:
            ax4 = plt.subplot(2, 3, 4)
            grid_df = optimization_data['grid_search']
            
            # Fix arrival rate at median value
            median_arrival = grid_df['arrival_rate'].median()
            df_fixed = grid_df[grid_df['arrival_rate'] == median_arrival]
            
            for lanes in sorted(df_fixed['lanes'].unique()):
                subset = df_fixed[df_fixed['lanes'] == lanes]
                ax4.plot(subset['diameter'], subset['objective'],
                        marker='o', linewidth=2, label=f'{lanes} lane(s)')
            
            ax4.set_xlabel('Diameter (m)')
            ax4.set_ylabel('Objective (lower is better)')
            ax4.set_title(f'Objective vs Diameter (λ={median_arrival:.2f})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Panel 5: Pareto front (throughput vs delay)
        if 'grid_search' in optimization_data:
            ax5 = plt.subplot(2, 3, 5)
            grid_df = optimization_data['grid_search']
            
            if 'throughput' in grid_df.columns and 'avg_delay' in grid_df.columns:
                for lanes in sorted(grid_df['lanes'].unique()):
                    subset = grid_df[grid_df['lanes'] == lanes]
                    ax5.scatter(subset['avg_delay'], subset['throughput'],
                              alpha=0.6, s=50, label=f'{lanes} lane(s)')
                
                ax5.set_xlabel('Average delay (s)')
                ax5.set_ylabel('Throughput (veh/hr)')
                ax5.set_title('Pareto Front: Throughput vs Delay')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # Panel 6: Optimization summary table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = []
        if 'grid_search' in optimization_data:
            grid_df = optimization_data['grid_search']
            best_idx = grid_df['objective'].idxmin()
            best_config = grid_df.loc[best_idx]
            
            summary_text.append("=== Grid Search Best Configuration ===")
            summary_text.append(f"Diameter: {best_config['diameter']:.1f} m")
            summary_text.append(f"Lanes: {int(best_config['lanes'])}")
            summary_text.append(f"Arrival rate: {best_config['arrival_rate']:.3f} veh/s")
            summary_text.append(f"Objective: {best_config['objective']:.2f}")
            if 'throughput' in best_config:
                summary_text.append(f"Throughput: {best_config['throughput']:.0f} veh/hr")
            if 'avg_delay' in best_config:
                summary_text.append(f"Avg delay: {best_config['avg_delay']:.1f} s")
        
        if 'bayesian_opt' in optimization_data:
            bayes_df = optimization_data['bayesian_opt']
            best_idx = bayes_df['objective'].idxmin()
            best_bayes = bayes_df.loc[best_idx]
            
            summary_text.append("\n=== Bayesian Opt Best Configuration ===")
            summary_text.append(f"Diameter: {best_bayes['diameter']:.1f} m")
            summary_text.append(f"Lanes: {int(best_bayes['lanes'])}")
            summary_text.append(f"Objective: {best_bayes['objective']:.2f}")
            summary_text.append(f"Found at iteration: {int(best_bayes['iteration'])}")
        
        ax6.text(0.1, 0.9, '\n'.join(summary_text), 
                transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'optimization_results.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved optimization results to {self.output_dir / 'optimization_results.png'}")
        plt.show()
    
    # ========================================================================
    # 4. FAILURE MODE DEMONSTRATIONS
    # ========================================================================
    
    def visualize_failure_modes(self, data: pd.DataFrame, save: bool = True):
        """
        Demonstrate different failure modes:
        - Queue divergence
        - Throughput plateau
        - Delay explosion
        """
        fig = plt.figure(figsize=(16, 8))
        
        # Panel 1: Queue evolution over time (time series for failing scenario)
        ax1 = plt.subplot(2, 3, 1)
        # This requires time-series data; use aggregated if not available
        if 'time' in data.columns:
            failing_scenario = data[(data['lanes'] == 1) & (data['arrival_rate'] >= 0.12)]
            if not failing_scenario.empty:
                for arm in ['N', 'E', 'S', 'W']:
                    queue_col = f'queue_{arm}'
                    if queue_col in failing_scenario.columns:
                        ax1.plot(failing_scenario['time'], failing_scenario[queue_col],
                                label=f'Arm {arm}', linewidth=1.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Queue length (veh)')
        ax1.set_title('Queue Divergence (1 lane, λ=0.12)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Throughput saturation curve
        ax2 = plt.subplot(2, 3, 2)
        for lanes in [1, 2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            # Sort by arrival rate
            df_sorted = df_lane.sort_values('arrival_rate')
            ax2.plot(df_sorted['arrival_rate'], df_sorted['throughput'],
                    marker='o', linewidth=2, label=f'{lanes} lane(s)',
                    color=self.colors.get(f'{lanes}_lane', 'gray'))
            
            # Mark saturation point (where derivative drops significantly)
            if len(df_sorted) > 2:
                throughput_diff = np.diff(df_sorted['throughput'])
                arrival_diff = np.diff(df_sorted['arrival_rate'])
                derivative = throughput_diff / (arrival_diff + 1e-9)
                
                # Find where derivative drops below 50% of initial
                if len(derivative) > 1 and derivative[0] > 0:
                    saturation_idx = np.where(derivative < 0.5 * derivative[0])[0]
                    if len(saturation_idx) > 0:
                        sat_idx = saturation_idx[0] + 1
                        ax2.axvline(df_sorted.iloc[sat_idx]['arrival_rate'],
                                   color=self.colors.get(f'{lanes}_lane', 'gray'),
                                   linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('Arrival rate per arm (veh/s)')
        ax2.set_ylabel('Throughput (veh/hr)')
        ax2.set_title('Throughput Saturation Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Delay explosion (log scale)
        ax3 = plt.subplot(2, 3, 3)
        for lanes in [1, 2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            df_sorted = df_lane.sort_values('arrival_rate')
            ax3.semilogy(df_sorted['arrival_rate'], df_sorted['avg_delay'] + 0.1,  # +0.1 to avoid log(0)
                        marker='s', linewidth=2, label=f'{lanes} lane(s)',
                        color=self.colors.get(f'{lanes}_lane', 'gray'))
        
        ax3.axhline(y=120, color='red', linestyle='--', linewidth=2, label='Failure threshold')
        ax3.set_xlabel('Arrival rate per arm (veh/s)')
        ax3.set_ylabel('Average delay (s, log scale)')
        ax3.set_title('Delay Explosion at Failure')
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
        
        # Panel 4: Service time distribution (before vs after failure)
        ax4 = plt.subplot(2, 3, 4)
        if 'service_time' in data.columns:
            stable_data = data[(data['lanes'] == 2) & (data['arrival_rate'] == 0.10)]
            failing_data = data[(data['lanes'] == 2) & (data['arrival_rate'] == 0.15)]
            
            if not stable_data.empty and not failing_data.empty:
                ax4.hist(stable_data['service_time'], bins=30, alpha=0.6, 
                        label='Stable (λ=0.10)', color='green', density=True)
                ax4.hist(failing_data['service_time'], bins=30, alpha=0.6,
                        label='Failing (λ=0.15)', color='red', density=True)
        ax4.set_xlabel('Service time (s)')
        ax4.set_ylabel('Probability density')
        ax4.set_title('Service Time Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Queue length distribution comparison
        ax5 = plt.subplot(2, 3, 5)
        stable_queues = []
        failing_queues = []
        
        for arm in ['N', 'E', 'S', 'W']:
            col = f'max_queue_{arm}'
            if col in data.columns:
                stable = data[(data['lanes'] == 2) & (data['arrival_rate'] == 0.10)][col]
                failing = data[(data['lanes'] == 2) & (data['arrival_rate'] == 0.15)][col]
                stable_queues.extend(stable.dropna().tolist())
                failing_queues.extend(failing.dropna().tolist())
        
        if stable_queues and failing_queues:
            ax5.boxplot([stable_queues, failing_queues], 
                       labels=['Stable\n(λ=0.10)', 'Failing\n(λ=0.15)'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
        ax5.set_ylabel('Max queue length (veh)')
        ax5.set_title('Queue Distribution: Stable vs Failing')
        ax5.grid(True, axis='y', alpha=0.3)
        
        # Panel 6: Failure mode classification
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        failure_summary = []
        failure_summary.append("=== Failure Mode Classification ===\n")
        
        for lanes in [1, 2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            # Detect failure modes
            queue_failure = df_lane['max_queue_N'].max() > 50 if 'max_queue_N' in df_lane.columns else False
            delay_failure = df_lane['avg_delay'].max() > 120 if 'avg_delay' in df_lane.columns else False
            throughput_plateau = False
            
            if len(df_lane) > 2:
                df_sorted = df_lane.sort_values('arrival_rate')
                if len(df_sorted) >= 3:
                    # Check if last 3 throughputs are similar despite increasing arrival
                    last_3_throughput = df_sorted['throughput'].tail(3).values
                    if np.std(last_3_throughput) / (np.mean(last_3_throughput) + 1e-9) < 0.05:
                        throughput_plateau = True
            
            failure_summary.append(f"{lanes} Lane(s):")
            if queue_failure:
                failure_summary.append("  ✗ Queue divergence detected")
            if delay_failure:
                failure_summary.append("  ✗ Delay explosion detected")
            if throughput_plateau:
                failure_summary.append("  ✗ Throughput saturation detected")
            if not (queue_failure or delay_failure or throughput_plateau):
                failure_summary.append("  ✓ No failures detected")
            failure_summary.append("")
        
        ax6.text(0.1, 0.9, '\n'.join(failure_summary),
                transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'failure_modes.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved failure modes to {self.output_dir / 'failure_modes.png'}")
        plt.show()
    
    # ========================================================================
    # 5. COMPREHENSIVE COMPARISON PLOT (All Metrics)
    # ========================================================================
    
    def create_comprehensive_comparison(self, data: pd.DataFrame, save: bool = True):
        """
        Create a single comprehensive figure with all key comparisons.
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Define reference scenarios
        arrival_rates = sorted(data['arrival_rate'].unique())
        
        # Row 1: Performance vs arrival rate
        metrics = [
            ('throughput', 'Throughput (veh/hr)', False),
            ('avg_delay', 'Avg Delay (s)', True),
            ('p95_delay', 'P95 Delay (s)', True),
            ('max_queue_N', 'Max Queue (veh)', False)
        ]
        
        for idx, (metric, ylabel, use_log) in enumerate(metrics):
            ax = plt.subplot(3, 4, idx + 1)
            for lanes in [1, 2, 3]:
                df_lane = data[(data['lanes'] == lanes) & (data['diameter'] == 45)]
                if df_lane.empty or metric not in df_lane.columns:
                    continue
                
                ax.plot(df_lane['arrival_rate'], df_lane[metric],
                       marker='o', linewidth=2, label=f'{lanes} lane(s)',
                       color=self.colors.get(f'{lanes}_lane', 'gray'))
            
            ax.set_xlabel('Arrival rate (veh/s/arm)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} vs Arrival (D=45m)')
            if use_log:
                ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Row 2: Performance vs diameter (fixed arrival)
        ref_arrival = 0.10
        for idx, (metric, ylabel, use_log) in enumerate(metrics):
            ax = plt.subplot(3, 4, idx + 5)
            for lanes in [1, 2, 3]:
                df_subset = data[(data['lanes'] == lanes) & (data['arrival_rate'] == ref_arrival)]
                if df_subset.empty or metric not in df_subset.columns:
                    continue
                
                ax.plot(df_subset['diameter'], df_subset[metric],
                       marker='s', linewidth=2, label=f'{lanes} lane(s)',
                       color=self.colors.get(f'{lanes}_lane', 'gray'))
            
            ax.set_xlabel('Diameter (m)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} vs Diameter (λ={ref_arrival})')
            if use_log:
                ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Row 3: Aggregate analysis
        # Panel 9: Efficiency score
        ax9 = plt.subplot(3, 4, 9)
        for lanes in [1, 2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            # Composite efficiency score: throughput / (avg_delay + 1)
            efficiency = df_lane['throughput'] / (df_lane['avg_delay'] + 1)
            ax9.plot(df_lane['arrival_rate'], efficiency,
                    marker='D', linewidth=2, label=f'{lanes} lane(s)',
                    color=self.colors.get(f'{lanes}_lane', 'gray'))
        
        ax9.set_xlabel('Arrival rate (veh/s/arm)')
        ax9.set_ylabel('Efficiency score')
        ax9.set_title('Overall Efficiency (throughput/delay)')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # Panel 10: Capacity utilization
        ax10 = plt.subplot(3, 4, 10)
        for lanes in [1, 2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            # Theoretical max per lane ≈ 1800 veh/hr
            utilization = (df_lane['throughput'] / (lanes * 1800) * 100).clip(upper=100)
            ax10.plot(df_lane['arrival_rate'], utilization,
                     marker='*', linewidth=2, label=f'{lanes} lane(s)',
                     color=self.colors.get(f'{lanes}_lane', 'gray'))
        
        ax10.set_xlabel('Arrival rate (veh/s/arm)')
        ax10.set_ylabel('Capacity utilization (%)')
        ax10.set_title('System Capacity Utilization')
        ax10.axhline(y=85, color='orange', linestyle='--', alpha=0.5, label='Target 85%')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # Panel 11: Queue balance (variance across arms)
        ax11 = plt.subplot(3, 4, 11)
        for lanes in [1, 2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            queue_cols = ['max_queue_N', 'max_queue_E', 'max_queue_S', 'max_queue_W']
            if all(col in df_lane.columns for col in queue_cols):
                queue_variance = df_lane[queue_cols].var(axis=1)
                ax11.plot(df_lane['arrival_rate'], queue_variance,
                         marker='v', linewidth=2, label=f'{lanes} lane(s)',
                         color=self.colors.get(f'{lanes}_lane', 'gray'))
        
        ax11.set_xlabel('Arrival rate (veh/s/arm)')
        ax11.set_ylabel('Queue variance across arms')
        ax11.set_title('Queue Balance (lower is better)')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        # Panel 12: Summary statistics table
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        summary_lines = ["=== Summary Statistics ===\n"]
        for lanes in [1, 2, 3]:
            df_lane = data[data['lanes'] == lanes]
            if df_lane.empty:
                continue
            
            summary_lines.append(f"{lanes} Lane(s):")
            summary_lines.append(f"  Max throughput: {df_lane['throughput'].max():.0f} veh/hr")
            summary_lines.append(f"  Min avg delay: {df_lane['avg_delay'].min():.1f} s")
            summary_lines.append(f"  Breaking point: {df_lane[df_lane['avg_delay'] > 120]['arrival_rate'].min() if not df_lane[df_lane['avg_delay'] > 120].empty else '>0.20'} veh/s")
            summary_lines.append("")
        
        ax12.text(0.05, 0.95, '\n'.join(summary_lines),
                 transform=ax12.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved comprehensive comparison to {self.output_dir / 'comprehensive_comparison.png'}")
        plt.show()


# ============================================================================
# CLI and Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced Roundabout Visualization Suite")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV file with simulation results')
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'lane_analysis', 'parameter_sweep', 
                               'optimization', 'failure_modes', 'comprehensive'],
                       help='Visualization mode')
    parser.add_argument('--optimization-data', type=str, default=None,
                       help='Path to JSON file with optimization results (for optimization mode)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = pd.read_csv(args.data)
    print(f"Loaded {len(data)} records")
    
    # Initialize visualizer
    viz = RoundaboutVisualizer(output_dir=args.output)
    
    # Generate requested visualizations
    if args.mode in ['all', 'lane_analysis']:
        print("\nGenerating lane choice analysis...")
        viz.visualize_lane_choice_impact(data)
    
    if args.mode in ['all', 'parameter_sweep']:
        print("\nGenerating parameter sweep analysis...")
        viz.visualize_parameter_sweep(data)
    
    if args.mode in ['all', 'optimization'] and args.optimization_data:
        print("\nGenerating optimization results...")
        with open(args.optimization_data, 'r') as f:
            opt_data = json.load(f)
        viz.visualize_optimization_results(opt_data)
    
    if args.mode in ['all', 'failure_modes']:
        print("\nGenerating failure mode analysis...")
        viz.visualize_failure_modes(data)
    
    if args.mode in ['all', 'comprehensive']:
        print("\nGenerating comprehensive comparison...")
        viz.create_comprehensive_comparison(data)
    
    print("\n✅ All visualizations complete!")


if __name__ == "__main__":
    main()
