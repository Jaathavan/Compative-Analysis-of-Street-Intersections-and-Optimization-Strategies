"""
visualize_results.py - Generate Visualizations for Simulation Results
======================================================================

Creates static (Matplotlib/Seaborn) and interactive (Plotly) visualizations
for roundabout simulation results.

Generates:
- Static: Throughput curves, delay plots, queue heatmaps, failure boundaries

Usage:
    python visualize_results.py --input results/summary.csv --output results/plots/
    python visualize_results.py --input results/summary.csv --window-data results/raw/*.csv --output results/plots/ --interactive
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import yaml

# Static plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# Interactive plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")


class ResultsVisualizer:
    """Generate comprehensive visualizations for simulation results."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.viz_config = self.config.get('visualization', {})
        self.style = self.viz_config.get('style', 'seaborn-v0_8-darkgrid')
        self.palette = self.viz_config.get('color_palette', 'Set2')
        self.figsize = tuple(self.viz_config.get('figure_size', [12, 8]))
        self.dpi = self.viz_config.get('dpi', 300)
        
        # Set style
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')
        
        sns.set_palette(self.palette)
    
    def generate_all_static(self, df: pd.DataFrame, output_dir: str, window_files: List[str] = None):
        """Generate all static visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating static visualizations...")
        
        # 1. Throughput vs Demand
        if 'throughput_vph' in df.columns:
            self.plot_throughput_vs_demand(df, output_dir)
        
        # 2. Delay vs Demand
        if 'mean_delay' in df.columns:
            self.plot_delay_vs_demand(df, output_dir)
        
        # 3. Queue Heatmap
        queue_cols = [c for c in df.columns if 'max_queue' in c]
        if queue_cols:
            self.plot_queue_heatmap(df, output_dir)
        
        # 4. Performance Scatter
        if 'throughput_vph' in df.columns and 'mean_delay' in df.columns:
            self.plot_performance_scatter(df, output_dir)
        
        # 5. Failure Boundary
        if 'failure_detected' in df.columns:
            self.plot_failure_boundary(df, output_dir)
        
        # 6. Time series (if window data provided)
        if window_files:
            self.plot_time_series_panel(window_files, output_dir)
        
        print(f"✓ Static plots saved to: {output_dir}")
    
    def generate_all_interactive(self, df: pd.DataFrame, output_dir: str, window_files: List[str] = None):
        """Generate all interactive visualizations."""
        if not PLOTLY_AVAILABLE:
            print("Skipping interactive plots (Plotly not installed)")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating interactive visualizations...")
        
        # 1. 3D Performance Surface
        self.plot_3d_surface(df, output_dir)
        
        # 2. Parameter Explorer Dashboard
        self.plot_parameter_explorer(df, output_dir)
        
        # 3. Time Series Animation (if window data provided)
        if window_files:
            self.plot_time_series_animation(window_files, output_dir)
        
        print(f"✓ Interactive plots saved to: {output_dir}")
    
    # =========================== STATIC PLOTS ===========================
    
    def plot_throughput_vs_demand(self, df: pd.DataFrame, output_dir: str):
        """Plot throughput vs demand with capacity curves."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract parameters if available
        if 'diameter' in df.columns and 'lanes' in df.columns:
            for (diam, lanes), group in df.groupby(['diameter', 'lanes']):
                label = f"D={diam}m, {lanes}L"
                group_sorted = group.sort_values('demand_multiplier' if 'demand_multiplier' in group.columns else 'throughput_vph')
                ax.plot(group_sorted.index, group_sorted['throughput_vph'], 'o-', label=label, linewidth=2)
        else:
            ax.plot(df.index, df['throughput_vph'], 'o-', linewidth=2)
        
        ax.set_xlabel('Scenario / Demand Level', fontsize=12)
        ax.set_ylabel('Throughput (veh/hr)', fontsize=12)
        ax.set_title('Roundabout Throughput vs Demand', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'throughput_vs_demand.png'), dpi=self.dpi)
        plt.close()
        print("  ✓ throughput_vs_demand.png")
    
    def plot_delay_vs_demand(self, df: pd.DataFrame, output_dir: str):
        """Plot delay vs demand with failure threshold."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Scatter plot with color by performance class
        if 'performance_class' in df.columns:
            classes = df['performance_class'].unique()
            colors = sns.color_palette(self.palette, n_colors=len(classes))
            for cls, color in zip(classes, colors):
                mask = df['performance_class'] == cls
                ax.scatter(df.loc[mask, 'throughput_vph'], df.loc[mask, 'mean_delay'], 
                          label=cls, s=100, alpha=0.7, color=color, edgecolors='black')
        else:
            ax.scatter(df['throughput_vph'], df['mean_delay'], s=100, alpha=0.7, edgecolors='black')
        
        # Failure threshold line
        failure_threshold = self.config['failure']['max_acceptable_delay']
        ax.axhline(failure_threshold, color='red', linestyle='--', linewidth=2, label=f'Failure threshold ({failure_threshold}s)')
        
        ax.set_xlabel('Throughput (veh/hr)', fontsize=12)
        ax.set_ylabel('Mean Delay (s)', fontsize=12)
        ax.set_title('Delay vs Throughput', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delay_vs_demand.png'), dpi=self.dpi)
        plt.close()
        print("  ✓ delay_vs_demand.png")
    
    def plot_queue_heatmap(self, df: pd.DataFrame, output_dir: str):
        """Plot heatmap of queue lengths by arm and scenario."""
        queue_cols = [c for c in df.columns if 'max_queue_' in c and 'trend' not in c]
        
        if not queue_cols:
            return
        
        # Extract queue data
        queue_data = df[queue_cols].copy()
        queue_data.columns = [c.replace('max_queue_', '') for c in queue_cols]
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1]*0.6))
        sns.heatmap(queue_data.T, annot=True, fmt='.0f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Max Queue Length (vehicles)'}, ax=ax)
        
        ax.set_xlabel('Scenario Index', fontsize=12)
        ax.set_ylabel('Arm', fontsize=12)
        ax.set_title('Maximum Queue Lengths by Arm', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'queue_heatmap.png'), dpi=self.dpi)
        plt.close()
        print("  ✓ queue_heatmap.png")
    
    def plot_performance_scatter(self, df: pd.DataFrame, output_dir: str):
        """Scatter plot of throughput vs delay with failure markers."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Separate failed and successful scenarios
        if 'failure_detected' in df.columns:
            failed = df[df['failure_detected'] == True]
            success = df[df['failure_detected'] == False]
            
            ax.scatter(success['mean_delay'], success['throughput_vph'], 
                      s=120, alpha=0.6, label='Success', color='green', edgecolors='black')
            ax.scatter(failed['mean_delay'], failed['throughput_vph'], 
                      s=120, alpha=0.6, label='Failure', color='red', marker='x', linewidths=3)
        else:
            ax.scatter(df['mean_delay'], df['throughput_vph'], s=120, alpha=0.6, edgecolors='black')
        
        ax.set_xlabel('Mean Delay (s)', fontsize=12)
        ax.set_ylabel('Throughput (veh/hr)', fontsize=12)
        ax.set_title('Performance Trade-off: Throughput vs Delay', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_scatter.png'), dpi=self.dpi)
        plt.close()
        print("  ✓ performance_scatter.png")
    
    def plot_failure_boundary(self, df: pd.DataFrame, output_dir: str):
        """Plot failure boundary in parameter space."""
        if 'diameter' not in df.columns or 'failure_detected' not in df.columns:
            return
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Identify parameter that varies with demand
        demand_param = 'demand_multiplier' if 'demand_multiplier' in df.columns else 'throughput_vph'
        
        failed = df[df['failure_detected'] == True]
        success = df[df['failure_detected'] == False]
        
        ax.scatter(success['diameter'], success[demand_param], 
                  s=150, alpha=0.6, label='Success', color='green', edgecolors='black')
        ax.scatter(failed['diameter'], failed[demand_param], 
                  s=150, alpha=0.6, label='Failure', color='red', marker='x', linewidths=3)
        
        ax.set_xlabel('Diameter (m)', fontsize=12)
        ax.set_ylabel(demand_param.replace('_', ' ').title(), fontsize=12)
        ax.set_title('Failure Boundary in Parameter Space', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'failure_boundary.png'), dpi=self.dpi)
        plt.close()
        print("  ✓ failure_boundary.png")
    
    def plot_time_series_panel(self, window_files: List[str], output_dir: str, max_scenarios: int = 4):
        """Plot time series panel comparing multiple scenarios."""
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0]*1.5, self.figsize[1]*1.2))
        axes = axes.flatten()
        
        metrics = ['throughput_vph', 'avg_delay', 'max_queue_N', 'avg_speed_ring']
        titles = ['Throughput (veh/hr)', 'Average Delay (s)', 'Max Queue (North)', 'Ring Speed (m/s)']
        
        for window_file in window_files[:max_scenarios]:
            df_window = pd.read_csv(window_file)
            label = Path(window_file).stem
            
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                if metric in df_window.columns:
                    time = df_window['end_time'] / 60  # Convert to minutes
                    axes[i].plot(time, df_window[metric], label=label, linewidth=2, alpha=0.7)
        
        for i, (ax, title) in enumerate(zip(axes, titles)):
            ax.set_xlabel('Time (minutes)', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Time Series Comparison Across Scenarios', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series_panel.png'), dpi=self.dpi)
        plt.close()
        print("  ✓ time_series_panel.png")
    
    # =========================== INTERACTIVE PLOTS ===========================
    
    def plot_3d_surface(self, df: pd.DataFrame, output_dir: str):
        """3D surface plot of performance metrics."""
        if not PLOTLY_AVAILABLE:
            return
        
        if 'diameter' not in df.columns or 'mean_delay' not in df.columns:
            return
        
        # Identify demand parameter
        demand_param = 'demand_multiplier' if 'demand_multiplier' in df.columns else 'throughput_vph'
        
        # Create pivot table
        pivot = df.pivot_table(values='mean_delay', index='diameter', columns=demand_param, aggfunc='mean')
        
        fig = go.Figure(data=[go.Surface(
            x=pivot.columns,
            y=pivot.index,
            z=pivot.values,
            colorscale='Viridis',
            colorbar=dict(title='Mean Delay (s)')
        )])
        
        fig.update_layout(
            title='Performance Surface: Delay vs Diameter and Demand',
            scene=dict(
                xaxis_title=demand_param.replace('_', ' ').title(),
                yaxis_title='Diameter (m)',
                zaxis_title='Mean Delay (s)'
            ),
            width=1000,
            height=800
        )
        
        fig.write_html(os.path.join(output_dir, '3d_performance_surface.html'))
        print("  ✓ 3d_performance_surface.html")
    
    def plot_parameter_explorer(self, df: pd.DataFrame, output_dir: str):
        """Interactive parameter explorer dashboard."""
        if not PLOTLY_AVAILABLE:
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Throughput vs Delay', 'Queue Lengths', 'Performance Distribution', 'Failure Analysis'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'pie'}]]
        )
        
        # Scatter: Throughput vs Delay
        fig.add_trace(
            go.Scatter(x=df['mean_delay'], y=df['throughput_vph'], mode='markers',
                      marker=dict(size=10, color=df.index, colorscale='Viridis'),
                      text=df['scenario_name'], name='Scenarios'),
            row=1, col=1
        )
        
        # Bar: Queue lengths
        queue_cols = [c for c in df.columns if 'max_queue_' in c and 'trend' not in c]
        if queue_cols:
            for col in queue_cols[:4]:
                fig.add_trace(
                    go.Bar(x=df.index, y=df[col], name=col.replace('max_queue_', '')),
                    row=1, col=2
                )
        
        # Histogram: Delay distribution
        fig.add_trace(
            go.Histogram(x=df['mean_delay'], nbinsx=20, name='Delay Distribution'),
            row=2, col=1
        )
        
        # Pie: Failure vs Success
        if 'failure_detected' in df.columns:
            failure_counts = df['failure_detected'].value_counts()
            fig.add_trace(
                go.Pie(labels=['Success', 'Failure'], values=[failure_counts.get(False, 0), failure_counts.get(True, 0)]),
                row=2, col=2
            )
        
        fig.update_layout(height=900, width=1400, title_text="Parameter Explorer Dashboard", showlegend=True)
        fig.write_html(os.path.join(output_dir, 'parameter_explorer.html'))
        print("  ✓ parameter_explorer.html")
    
    def plot_time_series_animation(self, window_files: List[str], output_dir: str):
        """Animated time series of queue evolution."""
        if not PLOTLY_AVAILABLE:
            return
        
        # Load first scenario for animation
        df = pd.read_csv(window_files[0])
        
        # Create animation frames
        fig = go.Figure()
        
        arms = ['N', 'E', 'S', 'W']
        for arm in arms:
            col = f'max_queue_{arm}'
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['end_time']/60,
                    y=df[col],
                    mode='lines+markers',
                    name=f'Arm {arm}'
                ))
        
        fig.update_layout(
            title='Queue Evolution Over Time',
            xaxis_title='Time (minutes)',
            yaxis_title='Queue Length (vehicles)',
            hovermode='x unified',
            width=1200,
            height=600
        )
        
        fig.write_html(os.path.join(output_dir, 'time_series_animation.html'))
        print("  ✓ time_series_animation.html")


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for simulation results')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input summary CSV file')
    parser.add_argument('--window-data', type=str, nargs='*',
                        help='Window data CSV files for time series plots')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for plots')
    parser.add_argument('--interactive', action='store_true',
                        help='Generate interactive plots (requires Plotly)')
    parser.add_argument('--static-only', action='store_true',
                        help='Generate only static plots')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    
    # Initialize visualizer
    viz = ResultsVisualizer(args.config)
    
    # Generate visualizations
    viz.generate_all_static(df, args.output, window_files=args.window_data)
    
    if args.interactive and not args.static_only:
        viz.generate_all_interactive(df, args.output, window_files=args.window_data)
    
    print(f"\n✓ All visualizations complete!")
    print(f"View results in: {args.output}")


if __name__ == '__main__':
    main()
