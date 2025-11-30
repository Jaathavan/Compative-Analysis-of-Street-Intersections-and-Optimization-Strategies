#!/usr/bin/env python3
# filepath: generate_enhanced_roundabout_graphs.py
"""
Generate Enhanced Roundabout Performance Graphs
================================================

Creates 4 comparison plots for 2-lane and 3-lane roundabouts showing:
1. Throughput vs Arrival Rate (with capacity breakpoints)
2. Average Delay vs Arrival Rate (with failure thresholds)
3. 95th Percentile Delay vs Arrival Rate (with reliability thresholds)
4. Maximum Queue Length vs Arrival Rate (with storage limits)

Each graph includes vertical lines marking the breakpoint where system
performance begins to degrade unacceptably.

Usage:
    python generate_enhanced_roundabout_graphs.py
    python generate_enhanced_roundabout_graphs.py --input results/roundabout_comparisons/simulation_data.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


class EnhancedRoundaboutGraphGenerator:
    """
    Generates enhanced performance graphs for 2-lane and 3-lane roundabouts
    with breakpoint analysis.
    """
    
    # Performance thresholds for breakpoint detection
    THRESHOLDS = {
        'avg_delay_acceptable': 15.0,      # seconds (Level of Service D)
        'p95_delay_acceptable': 90.0,      # seconds (worst-case tolerance)
        'max_queue_acceptable': 80,        # vehicles (storage capacity)
        'throughput_efficiency': 0.70      # 70% of demand (below = failure)
    }
    
    # Styling constants
    COLORS = {
        '2-lane': '#FF8C00',  # Dark orange
        '3-lane': '#2E8B57'   # Sea green
    }
    
    BREAKPOINT_STYLE = {
        'color': 'red',
        'linestyle': '--',
        'linewidth': 2,
        'alpha': 0.7
    }
    
    def __init__(self, data_path: str, output_dir: str):
        """Initialize generator."""
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Filter for 2-lane and 3-lane only
        self.df = self.df[self.df['lanes'].isin([2, 3])].copy()
        
        # Calculate demand (veh/hr)
        self.df['demand_veh_hr'] = self.df['arrival_rate'] * 4 * 3600
        
        # Calculate efficiency
        self.df['efficiency'] = self.df['throughput'] / self.df['demand_veh_hr']
        
        print(f"✓ Loaded {len(self.df)} data points")
        print(f"  2-lane: {len(self.df[self.df['lanes']==2])} points")
        print(f"  3-lane: {len(self.df[self.df['lanes']==3])} points")
    
    def detect_breakpoints(self) -> dict:
        """
        Detect breakpoint arrival rates where performance degrades.
        
        Returns:
            Dictionary with breakpoints for 2-lane and 3-lane systems
        """
        breakpoints = {}
        
        for lanes in [2, 3]:
            lane_data = self.df[self.df['lanes'] == lanes].copy()
            lane_data = lane_data.sort_values('arrival_rate')
            
            # Group by arrival rate (average across diameters)
            grouped = lane_data.groupby('arrival_rate').agg({
                'avg_delay': 'mean',
                'p95_delay': 'mean',
                'max_queue': 'mean',
                'efficiency': 'mean',
                'throughput': 'mean',
                'demand_veh_hr': 'mean'
            }).reset_index()
            
            # Detect breakpoints based on different criteria
            breakpoint_candidates = []
            
            # Criterion 1: Average delay exceeds threshold
            delay_breach = grouped[grouped['avg_delay'] > self.THRESHOLDS['avg_delay_acceptable']]
            if not delay_breach.empty:
                breakpoint_candidates.append(delay_breach['arrival_rate'].iloc[0])
            
            # Criterion 2: P95 delay exceeds threshold
            p95_breach = grouped[grouped['p95_delay'] > self.THRESHOLDS['p95_delay_acceptable']]
            if not p95_breach.empty:
                breakpoint_candidates.append(p95_breach['arrival_rate'].iloc[0])
            
            # Criterion 3: Max queue exceeds threshold
            queue_breach = grouped[grouped['max_queue'] > self.THRESHOLDS['max_queue_acceptable']]
            if not queue_breach.empty:
                breakpoint_candidates.append(queue_breach['arrival_rate'].iloc[0])
            
            # Criterion 4: Efficiency drops below threshold
            efficiency_breach = grouped[grouped['efficiency'] < self.THRESHOLDS['throughput_efficiency']]
            if not efficiency_breach.empty:
                breakpoint_candidates.append(efficiency_breach['arrival_rate'].iloc[0])
            
            # Criterion 5: Throughput starts decreasing (hypercongestion)
            if len(grouped) > 1:
                throughput_decrease = grouped[
                    grouped['throughput'].diff() < -100  # 100 veh/hr drop
                ]
                if not throughput_decrease.empty:
                    breakpoint_candidates.append(throughput_decrease['arrival_rate'].iloc[0])
            
            # Use the most conservative (earliest) breakpoint
            if breakpoint_candidates:
                breakpoint = min(breakpoint_candidates)
            else:
                # If no breach detected, system is stable across tested range
                breakpoint = None
            
            breakpoints[lanes] = {
                'arrival_rate': breakpoint,
                'candidates': breakpoint_candidates,
                'data': grouped
            }
            
            print(f"\n{lanes}-lane breakpoint analysis:")
            if breakpoint:
                print(f"  Critical breakpoint: λ = {breakpoint:.3f} veh/s/arm")
                print(f"  ({breakpoint * 4 * 3600:.0f} veh/hr total demand)")
                print(f"  Candidate breakpoints: {[f'{x:.3f}' for x in breakpoint_candidates]}")
            else:
                print(f"  No breakpoint detected (stable across range)")
        
        return breakpoints
    
    def plot_throughput_vs_arrival(self, breakpoints: dict):
        """Generate throughput vs arrival rate comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for lanes in [2, 3]:
            lane_data = self.df[self.df['lanes'] == lanes].copy()
            
            # Group by arrival rate and average across diameters
            grouped = lane_data.groupby('arrival_rate').agg({
                'throughput': ['mean', 'std']
            }).reset_index()
            
            x = grouped['arrival_rate'].values
            y_mean = grouped['throughput']['mean'].values
            y_std = grouped['throughput']['std'].values
            
            # Plot line
            label = f'{lanes}-lane'
            ax.plot(x, y_mean, 'o-', linewidth=2.5, markersize=10, 
                   color=self.COLORS[label], label=label, alpha=0.9)
            
            # Add error bars (standard deviation)
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, 
                           color=self.COLORS[label], alpha=0.2)
            
            # Add breakpoint line
            bp = breakpoints[lanes]['arrival_rate']
            if bp:
                ax.axvline(bp, label=f'{lanes}-lane breakpoint',
                          **self.BREAKPOINT_STYLE)
                
                # Add annotation
                bp_data = breakpoints[lanes]['data']
                bp_throughput = bp_data[bp_data['arrival_rate'] == bp]['throughput'].iloc[0]
                ax.annotate(
                    f'{lanes}L breakpoint\nλ={bp:.2f}\n{bp_throughput:.0f} veh/hr',
                    xy=(bp, bp_throughput),
                    xytext=(bp + 0.02, bp_throughput + 200),
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                   color='red', lw=1.5)
                )
        
        # Add ideal demand line (100% efficiency)
        arrival_range = np.linspace(0.05, 0.25, 100)
        ideal_throughput = arrival_range * 4 * 3600
        ax.plot(arrival_range, ideal_throughput, 'k--', linewidth=1.5, 
               alpha=0.4, label='Ideal (100% efficiency)')
        
        ax.set_xlabel('Arrival Rate (veh/s per arm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Throughput (veh/hr)', fontsize=14, fontweight='bold')
        ax.set_title('Throughput vs Arrival Rate: 2-Lane vs 3-Lane Roundabouts\n' +
                    'with Capacity Breakpoints', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.04, 0.26)
        ax.set_ylim(0, 3500)
        
        plt.tight_layout()
        output_path = self.output_dir / '9_throughput_vs_arrival_2lane_3lane.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_avg_delay_vs_arrival(self, breakpoints: dict):
        """Generate average delay vs arrival rate comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for lanes in [2, 3]:
            lane_data = self.df[self.df['lanes'] == lanes].copy()
            
            # Group by arrival rate
            grouped = lane_data.groupby('arrival_rate').agg({
                'avg_delay': ['mean', 'std']
            }).reset_index()
            
            x = grouped['arrival_rate'].values
            y_mean = grouped['avg_delay']['mean'].values
            y_std = grouped['avg_delay']['std'].values
            
            # Plot line
            label = f'{lanes}-lane'
            ax.plot(x, y_mean, 'o-', linewidth=2.5, markersize=10,
                   color=self.COLORS[label], label=label, alpha=0.9)
            
            # Add error bars
            ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                           color=self.COLORS[label], alpha=0.2)
            
            # Add breakpoint line
            bp = breakpoints[lanes]['arrival_rate']
            if bp:
                ax.axvline(bp, label=f'{lanes}-lane breakpoint',
                          **self.BREAKPOINT_STYLE)
                
                # Add annotation
                bp_data = breakpoints[lanes]['data']
                bp_delay = bp_data[bp_data['arrival_rate'] == bp]['avg_delay'].iloc[0]
                ax.annotate(
                    f'{lanes}L breakpoint\nλ={bp:.2f}\ndelay={bp_delay:.1f}s',
                    xy=(bp, bp_delay),
                    xytext=(bp - 0.03, bp_delay + 5),
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3',
                                   color='red', lw=1.5)
                )
        
        # Add acceptable delay threshold line
        ax.axhline(self.THRESHOLDS['avg_delay_acceptable'], 
                  color='gray', linestyle=':', linewidth=2, alpha=0.6,
                  label=f"Acceptable delay threshold ({self.THRESHOLDS['avg_delay_acceptable']}s)")
        
        ax.set_xlabel('Arrival Rate (veh/s per arm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Delay (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('Average Delay vs Arrival Rate: 2-Lane vs 3-Lane Roundabouts\n' +
                    'with Performance Breakpoints', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.04, 0.26)
        ax.set_ylim(0, 60)
        
        plt.tight_layout()
        output_path = self.output_dir / '10_avg_delay_vs_arrival_2lane_3lane.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_p95_delay_vs_arrival(self, breakpoints: dict):
        """Generate 95th percentile delay vs arrival rate comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for lanes in [2, 3]:
            lane_data = self.df[self.df['lanes'] == lanes].copy()
            
            # Group by arrival rate
            grouped = lane_data.groupby('arrival_rate').agg({
                'p95_delay': ['mean', 'std']
            }).reset_index()
            
            x = grouped['arrival_rate'].values
            y_mean = grouped['p95_delay']['mean'].values
            y_std = grouped['p95_delay']['std'].values
            
            # Plot line
            label = f'{lanes}-lane'
            ax.plot(x, y_mean, 'o-', linewidth=2.5, markersize=10,
                   color=self.COLORS[label], label=label, alpha=0.9)
            
            # Add error bars
            ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                           color=self.COLORS[label], alpha=0.2)
            
            # Add breakpoint line
            bp = breakpoints[lanes]['arrival_rate']
            if bp:
                ax.axvline(bp, label=f'{lanes}-lane breakpoint',
                          **self.BREAKPOINT_STYLE)
                
                # Add annotation
                bp_data = breakpoints[lanes]['data']
                bp_p95 = bp_data[bp_data['arrival_rate'] == bp]['p95_delay'].iloc[0]
                ax.annotate(
                    f'{lanes}L breakpoint\nλ={bp:.2f}\nP95={bp_p95:.1f}s',
                    xy=(bp, bp_p95),
                    xytext=(bp + 0.02, bp_p95 - 20),
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                   color='red', lw=1.5)
                )
        
        # Add acceptable P95 delay threshold
        ax.axhline(self.THRESHOLDS['p95_delay_acceptable'],
                  color='gray', linestyle=':', linewidth=2, alpha=0.6,
                  label=f"Acceptable P95 threshold ({self.THRESHOLDS['p95_delay_acceptable']}s)")
        
        ax.set_xlabel('Arrival Rate (veh/s per arm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('95th Percentile Delay (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('95th Percentile Delay vs Arrival Rate: 2-Lane vs 3-Lane\n' +
                    'with Reliability Breakpoints', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.04, 0.26)
        ax.set_ylim(0, 250)
        
        plt.tight_layout()
        output_path = self.output_dir / '11_p95_delay_vs_arrival_2lane_3lane.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_max_queue_vs_arrival(self, breakpoints: dict):
        """Generate maximum queue length vs arrival rate comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for lanes in [2, 3]:
            lane_data = self.df[self.df['lanes'] == lanes].copy()
            
            # Group by arrival rate
            grouped = lane_data.groupby('arrival_rate').agg({
                'max_queue': ['mean', 'std']
            }).reset_index()
            
            x = grouped['arrival_rate'].values
            y_mean = grouped['max_queue']['mean'].values
            y_std = grouped['max_queue']['std'].values
            
            # Plot line
            label = f'{lanes}-lane'
            ax.plot(x, y_mean, 'o-', linewidth=2.5, markersize=10,
                   color=self.COLORS[label], label=label, alpha=0.9)
            
            # Add error bars
            ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                           color=self.COLORS[label], alpha=0.2)
            
            # Add breakpoint line
            bp = breakpoints[lanes]['arrival_rate']
            if bp:
                ax.axvline(bp, label=f'{lanes}-lane breakpoint',
                          **self.BREAKPOINT_STYLE)
                
                # Add annotation
                bp_data = breakpoints[lanes]['data']
                bp_queue = bp_data[bp_data['arrival_rate'] == bp]['max_queue'].iloc[0]
                queue_length_m = bp_queue * 5  # 5m per vehicle
                ax.annotate(
                    f'{lanes}L breakpoint\nλ={bp:.2f}\nqueue={bp_queue:.0f} veh\n({queue_length_m:.0f}m)',
                    xy=(bp, bp_queue),
                    xytext=(bp - 0.04, bp_queue + 15),
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3',
                                   color='red', lw=1.5)
                )
        
        # Add acceptable queue threshold
        ax.axhline(self.THRESHOLDS['max_queue_acceptable'],
                  color='gray', linestyle=':', linewidth=2, alpha=0.6,
                  label=f"Storage capacity limit ({self.THRESHOLDS['max_queue_acceptable']} veh)")
        
        ax.set_xlabel('Arrival Rate (veh/s per arm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Maximum Queue Length (vehicles)', fontsize=14, fontweight='bold')
        ax.set_title('Maximum Queue Length vs Arrival Rate: 2-Lane vs 3-Lane\n' +
                    'with Storage Breakpoints', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.04, 0.26)
        ax.set_ylim(0, 200)
        
        plt.tight_layout()
        output_path = self.output_dir / '12_max_queue_vs_arrival_2lane_3lane.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def generate_all_graphs(self):
        """Generate all enhanced comparison graphs."""
        print(f"\n{'='*70}")
        print("ENHANCED ROUNDABOUT PERFORMANCE GRAPHS")
        print(f"{'='*70}\n")
        
        # Detect breakpoints
        print("Detecting performance breakpoints...")
        breakpoints = self.detect_breakpoints()
        
        # Generate graphs
        print(f"\nGenerating enhanced comparison graphs...")
        print(f"Output directory: {self.output_dir}\n")
        
        self.plot_throughput_vs_arrival(breakpoints)
        self.plot_avg_delay_vs_arrival(breakpoints)
        self.plot_p95_delay_vs_arrival(breakpoints)
        self.plot_max_queue_vs_arrival(breakpoints)
        
        # Generate summary report
        self.generate_breakpoint_report(breakpoints)
        
        print(f"\n{'='*70}")
        print("✅ ALL GRAPHS GENERATED SUCCESSFULLY")
        print(f"{'='*70}\n")
        print(f"Output files:")
        print(f"  • 9_throughput_vs_arrival_2lane_3lane.png")
        print(f"  • 10_avg_delay_vs_arrival_2lane_3lane.png")
        print(f"  • 11_p95_delay_vs_arrival_2lane_3lane.png")
        print(f"  • 12_max_queue_vs_arrival_2lane_3lane.png")
        print(f"  • breakpoint_analysis_summary.md")
        print(f"\nLocation: {self.output_dir}/")
    
    def generate_breakpoint_report(self, breakpoints: dict):
        """Generate markdown report summarizing breakpoint analysis."""
        report_path = self.output_dir / 'breakpoint_analysis_summary.md'
        
        with open(report_path, 'w') as f:
            f.write("# Roundabout Performance Breakpoint Analysis\n\n")
            f.write("## Summary\n\n")
            f.write("This report identifies critical arrival rates where roundabout ")
            f.write("performance begins to degrade unacceptably.\n\n")
            
            f.write("### Breakpoint Definitions\n\n")
            f.write("A **breakpoint** is the arrival rate where at least one of these ")
            f.write("conditions is violated:\n\n")
            f.write(f"- Average delay > {self.THRESHOLDS['avg_delay_acceptable']}s ")
            f.write("(Level of Service D/E boundary)\n")
            f.write(f"- 95th percentile delay > {self.THRESHOLDS['p95_delay_acceptable']}s ")
            f.write("(user experience threshold)\n")
            f.write(f"- Maximum queue > {self.THRESHOLDS['max_queue_acceptable']} vehicles ")
            f.write("(storage capacity)\n")
            f.write(f"- Throughput efficiency < {self.THRESHOLDS['throughput_efficiency']*100:.0f}% ")
            f.write("(demand satisfaction)\n")
            f.write("- Throughput begins decreasing (hypercongestion)\n\n")
            
            f.write("---\n\n")
            
            for lanes in [2, 3]:
                bp_data = breakpoints[lanes]
                bp = bp_data['arrival_rate']
                
                f.write(f"## {lanes}-Lane Roundabout\n\n")
                
                if bp:
                    f.write(f"### Critical Breakpoint: **λ = {bp:.3f} veh/s per arm**\n\n")
                    f.write(f"**Total demand at breakpoint:** {bp * 4 * 3600:.0f} veh/hr\n\n")
                    
                    # Get performance at breakpoint
                    bp_perf = bp_data['data'][bp_data['data']['arrival_rate'] == bp].iloc[0]
                    
                    f.write("#### Performance at Breakpoint\n\n")
                    f.write(f"- **Throughput:** {bp_perf['throughput']:.0f} veh/hr ")
                    f.write(f"({bp_perf['efficiency']*100:.1f}% efficiency)\n")
                    f.write(f"- **Average delay:** {bp_perf['avg_delay']:.1f}s\n")
                    f.write(f"- **95th percentile delay:** {bp_perf['p95_delay']:.1f}s\n")
                    f.write(f"- **Maximum queue:** {bp_perf['max_queue']:.0f} vehicles ")
                    f.write(f"(~{bp_perf['max_queue']*5:.0f}m)\n\n")
                    
                    f.write("#### Breakpoint Criteria Triggered\n\n")
                    candidates = bp_data['candidates']
                    unique_candidates = sorted(set(candidates))
                    
                    for candidate in unique_candidates:
                        cand_perf = bp_data['data'][bp_data['data']['arrival_rate'] == candidate].iloc[0]
                        violations = []
                        
                        if cand_perf['avg_delay'] > self.THRESHOLDS['avg_delay_acceptable']:
                            violations.append(f"avg delay ({cand_perf['avg_delay']:.1f}s)")
                        if cand_perf['p95_delay'] > self.THRESHOLDS['p95_delay_acceptable']:
                            violations.append(f"P95 delay ({cand_perf['p95_delay']:.1f}s)")
                        if cand_perf['max_queue'] > self.THRESHOLDS['max_queue_acceptable']:
                            violations.append(f"queue ({cand_perf['max_queue']:.0f} veh)")
                        if cand_perf['efficiency'] < self.THRESHOLDS['throughput_efficiency']:
                            violations.append(f"efficiency ({cand_perf['efficiency']*100:.1f}%)")
                        
                        if violations:
                            f.write(f"- **λ = {candidate:.3f}**: {', '.join(violations)}\n")
                    
                    f.write("\n")
                    
                    # Recommended operating range
                    safe_margin = 0.8  # 20% safety margin
                    recommended_max = bp * safe_margin
                    f.write("#### Recommended Operating Range\n\n")
                    f.write(f"**Maximum safe arrival rate:** λ ≤ {recommended_max:.3f} veh/s per arm\n")
                    f.write(f"**Maximum safe total demand:** {recommended_max * 4 * 3600:.0f} veh/hr\n\n")
                    f.write(f"*This includes a 20% safety margin below the critical breakpoint.*\n\n")
                    
                else:
                    f.write("### No Breakpoint Detected\n\n")
                    f.write("The system remains **stable and performant** across all tested ")
                    f.write("arrival rates (λ = 0.05 to 0.25 veh/s per arm).\n\n")
                    
                    # Show best performance
                    best = bp_data['data'].loc[bp_data['data']['throughput'].idxmax()]
                    f.write("#### Best Observed Performance\n\n")
                    f.write(f"- **Arrival rate:** λ = {best['arrival_rate']:.3f} veh/s per arm\n")
                    f.write(f"- **Throughput:** {best['throughput']:.0f} veh/hr\n")
                    f.write(f"- **Average delay:** {best['avg_delay']:.1f}s\n")
                    f.write(f"- **95th percentile delay:** {best['p95_delay']:.1f}s\n")
                    f.write(f"- **Maximum queue:** {best['max_queue']:.0f} vehicles\n\n")
                
                f.write("---\n\n")
            
            # Comparative summary
            f.write("## Comparative Summary\n\n")
            f.write("| Configuration | Breakpoint (λ) | Max Safe Demand | Relative Capacity |\n")
            f.write("|---------------|----------------|-----------------|-------------------|\n")
            
            for lanes in [2, 3]:
                bp = breakpoints[lanes]['arrival_rate']
                if bp:
                    safe_demand = bp * 0.8 * 4 * 3600
                    f.write(f"| {lanes}-lane | {bp:.3f} veh/s/arm | {safe_demand:.0f} veh/hr | ")
                else:
                    safe_demand = 0.25 * 4 * 3600  # Max tested
                    f.write(f"| {lanes}-lane | >0.25 veh/s/arm | >{safe_demand:.0f} veh/hr | ")
                
                # Calculate relative capacity (normalized to 2-lane)
                bp_2lane = breakpoints[2]['arrival_rate']
                if bp_2lane and bp:
                    relative = (bp / bp_2lane) * 100
                    f.write(f"{relative:.0f}% |\n")
                elif lanes == 2:
                    f.write("100% (baseline) |\n")
                else:
                    f.write(">125% |\n")
            
            f.write("\n")
            f.write("### Key Insights\n\n")
            
            bp_2 = breakpoints[2]['arrival_rate']
            bp_3 = breakpoints[3]['arrival_rate']
            
            if bp_2 and bp_3:
                improvement = ((bp_3 - bp_2) / bp_2) * 100
                f.write(f"- **3-lane capacity advantage:** +{improvement:.0f}% over 2-lane\n")
                f.write(f"- **2-lane sustainable demand:** {bp_2 * 0.8 * 4 * 3600:.0f} veh/hr\n")
                f.write(f"- **3-lane sustainable demand:** {bp_3 * 0.8 * 4 * 3600:.0f} veh/hr\n")
            elif not bp_2 and not bp_3:
                f.write("- Both configurations show **excellent stability** across tested range\n")
                f.write("- Systems can likely handle **>3,600 veh/hr** total demand\n")
            elif not bp_2:
                f.write("- **2-lane shows no breakpoint** (stable >3,600 veh/hr)\n")
            elif not bp_3:
                f.write("- **3-lane shows no breakpoint** (stable >3,600 veh/hr)\n")
            
            f.write("\n---\n\n")
            f.write(f"*Generated by Enhanced Roundabout Graph Generator*\n")
        
        print(f"✓ Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate enhanced roundabout comparison graphs with breakpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all graphs with default paths
    python generate_enhanced_roundabout_graphs.py
    
    # Specify custom data file
    python generate_enhanced_roundabout_graphs.py \\
        --input results/roundabout_comparisons/simulation_data.csv \\
        --output results/roundabout_comparisons/
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='results/roundabout_comparisons/simulation_data.csv',
        help='Path to simulation data CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/roundabout_comparisons',
        help='Output directory for graphs'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"✗ ERROR: Input file not found: {args.input}")
        print(f"\nExpected structure:")
        print(f"  lanes,diameter,arrival_rate,throughput,avg_delay,p95_delay,max_queue")
        sys.exit(1)
    
    try:
        # Generate graphs
        generator = EnhancedRoundaboutGraphGenerator(args.input, args.output)
        generator.generate_all_graphs()
        
        print(f"\n📊 SUCCESS! Enhanced graphs generated.")
        print(f"\nNext steps:")
        print(f"  1. Review graphs in: {args.output}/")
        print(f"  2. Check breakpoint_analysis_summary.md for detailed findings")
        print(f"  3. Include graphs in final report")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()