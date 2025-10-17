#!/usr/bin/env python3
"""
optimize.py - Parameter Sweep Orchestration
============================================

Orchestrates parameter sweeps to identify optimal roundabout configurations.
Automates the full pipeline: network generation → route generation → 
simulation → analysis → visualization.

Sweep dimensions (from config.yaml):
- Geometry: diameter, lanes
- Demand: demand_multiplier
- (Optional) Behavioral: gap acceptance, reaction time

Usage:
    # Full sweep from config
    python optimize.py --config config/config.yaml --output results/sweep_results/
    
    # Custom parameter ranges
    python optimize.py --config config/config.yaml --diameters 35 45 55 --demand-levels 0.5 1.0 1.5 --output results/custom_sweep/
"""

import argparse
import os
import sys
import subprocess
import itertools
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import pandas as pd
from datetime import datetime
import json


class ParameterSweepOrchestrator:
    """Orchestrate parameter sweeps across the simulation pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize orchestrator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        self.sweep_config = self.config.get('sweep', {})
        self.root_dir = Path(__file__).parent.parent  # /roundabout/
        
        # Default sweep ranges from config
        self.diameter_range = self.sweep_config.get('diameter_range', [45.0])
        self.lanes_range = self.sweep_config.get('lanes_range', [1])
        self.demand_multipliers = self.sweep_config.get('demand_multipliers', [1.0])
    
    def run_sweep(self, output_dir: str, diameters: List[float] = None, 
                  lanes: List[int] = None, demand_levels: List[float] = None,
                  parallel: bool = False) -> pd.DataFrame:
        """
        Execute parameter sweep.
        
        Args:
            output_dir: Directory for all sweep outputs
            diameters: Diameter values to sweep (override config)
            lanes: Lane counts to sweep (override config)
            demand_levels: Demand multipliers to sweep (override config)
            parallel: Run scenarios in parallel (requires GNU parallel or similar)
            
        Returns:
            DataFrame with all scenario results
        """
        # Use provided values or defaults from config
        diameters = diameters or self.diameter_range
        lanes = lanes or self.lanes_range
        demand_levels = demand_levels or self.demand_multipliers
        
        # Generate all parameter combinations
        param_grid = list(itertools.product(diameters, lanes, demand_levels))
        
        print("=" * 70)
        print("PARAMETER SWEEP ORCHESTRATION")
        print("=" * 70)
        print(f"Sweep dimensions:")
        print(f"  Diameters: {diameters}")
        print(f"  Lanes: {lanes}")
        print(f"  Demand multipliers: {demand_levels}")
        print(f"Total scenarios: {len(param_grid)}")
        print("=" * 70)
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        configs_dir = os.path.join(output_dir, 'sumo_configs')
        results_dir = os.path.join(output_dir, 'raw_results')
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Run each scenario
        scenario_results = []
        for i, (diameter, lane_count, demand_mult) in enumerate(param_grid):
            print(f"\n[{i+1}/{len(param_grid)}] Running scenario:")
            print(f"  diameter={diameter}m, lanes={lane_count}, demand={demand_mult}×")
            
            scenario_name = f"d{int(diameter)}_l{lane_count}_dm{demand_mult:.2f}"
            scenario_config_dir = os.path.join(configs_dir, scenario_name)
            scenario_result_file = os.path.join(results_dir, f"{scenario_name}.csv")
            
            try:
                # Run pipeline for this scenario
                self._run_scenario(
                    diameter=diameter,
                    lanes=lane_count,
                    demand_mult=demand_mult,
                    config_dir=scenario_config_dir,
                    result_file=scenario_result_file
                )
                
                scenario_results.append({
                    'scenario_name': scenario_name,
                    'diameter': diameter,
                    'lanes': lane_count,
                    'demand_multiplier': demand_mult,
                    'result_file': scenario_result_file,
                    'status': 'success'
                })
                
                print(f"  ✓ Scenario complete")
                
            except Exception as e:
                print(f"  ✗ Scenario failed: {e}")
                scenario_results.append({
                    'scenario_name': scenario_name,
                    'diameter': diameter,
                    'lanes': lane_count,
                    'demand_multiplier': demand_mult,
                    'result_file': None,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save sweep metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_scenarios': len(param_grid),
            'successful': sum(1 for r in scenario_results if r['status'] == 'success'),
            'failed': sum(1 for r in scenario_results if r['status'] == 'failed'),
            'parameters': {
                'diameters': diameters,
                'lanes': lanes,
                'demand_multipliers': demand_levels
            }
        }
        
        with open(os.path.join(output_dir, 'sweep_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Analyze all results
        print("\n" + "=" * 70)
        print("ANALYZING SWEEP RESULTS")
        print("=" * 70)
        
        summary_df = self._analyze_sweep_results(results_dir, output_dir)
        
        # Generate visualizations
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        self._generate_sweep_visualizations(summary_df, output_dir)
        
        print("\n" + "=" * 70)
        print("SWEEP COMPLETE")
        print("=" * 70)
        print(f"Total scenarios: {metadata['total_scenarios']}")
        print(f"Successful: {metadata['successful']}")
        print(f"Failed: {metadata['failed']}")
        print(f"\nResults saved to: {output_dir}")
        print("=" * 70)
        
        return summary_df
    
    def _run_scenario(self, diameter: float, lanes: int, demand_mult: float,
                      config_dir: str, result_file: str):
        """Run complete pipeline for a single scenario."""
        
        # 1. Generate network
        net_cmd = [
            'python3', 'src/generate_network.py',
            '--config', self.config_path,
            '--diameter', str(diameter),
            '--lanes', str(lanes),
            '--output', config_dir
        ]
        subprocess.run(net_cmd, check=True, capture_output=True, cwd=self.root_dir)
        
        # 2. Generate routes
        net_file = os.path.join(config_dir, 'roundabout.net.xml')
        route_cmd = [
            'python3', 'src/generate_routes.py',
            '--config', self.config_path,
            '--network', net_file,
            '--demand-multiplier', str(demand_mult),
            '--output', config_dir
        ]
        subprocess.run(route_cmd, check=True, capture_output=True, cwd=self.root_dir)
        
        # 3. Run simulation
        sumocfg = os.path.join(config_dir, 'roundabout.sumocfg')
        sim_cmd = [
            'python3', 'src/run_simulation.py',
            '--sumocfg', sumocfg,
            '--config', self.config_path,
            '--output', result_file
        ]
        subprocess.run(sim_cmd, check=True, capture_output=True, cwd=self.root_dir)
    
    def _analyze_sweep_results(self, results_dir: str, output_dir: str) -> pd.DataFrame:
        """Analyze all sweep results."""
        analyze_cmd = [
            'python3', 'src/analyze_results.py',
            '--sweep', os.path.join(results_dir, '*.csv'),
            '--config', self.config_path,
            '--output', os.path.join(output_dir, 'sweep_summary.csv')
        ]
        
        subprocess.run(analyze_cmd, check=True, cwd=self.root_dir)
        
        # Load and return summary
        summary_file = os.path.join(output_dir, 'sweep_summary.csv')
        return pd.read_csv(summary_file)
    
    def _generate_sweep_visualizations(self, summary_df: pd.DataFrame, output_dir: str):
        """Generate visualizations for sweep results."""
        plots_dir = os.path.join(output_dir, 'plots')
        
        # Collect window data files
        results_dir = os.path.join(output_dir, 'raw_results')
        window_files = list(Path(results_dir).glob('*.csv'))
        window_files = [str(f) for f in window_files if not str(f).endswith('_aggregate.csv')]
        
        # Static plots
        viz_cmd = [
            'python3', 'src/visualize_results.py',
            '--input', os.path.join(output_dir, 'sweep_summary.csv'),
            '--window-data', *window_files[:5],  # Limit to 5 for time series
            '--config', self.config_path,
            '--output', plots_dir
        ]
        subprocess.run(viz_cmd, check=True, cwd=self.root_dir)
        
        # Interactive plots
        viz_cmd_interactive = viz_cmd + ['--interactive']
        try:
            subprocess.run(viz_cmd_interactive, check=True, cwd=self.root_dir)
        except:
            print("  Warning: Interactive plots skipped (Plotly may not be installed)")
    
    def identify_optimal_configuration(self, summary_df: pd.DataFrame) -> Dict:
        """
        Identify optimal configuration based on multiple criteria.
        
        Returns:
            Dictionary with optimal configurations for different objectives
        """
        # Filter out failures
        viable = summary_df[summary_df['failure_detected'] == False]
        
        if viable.empty:
            print("Warning: No viable configurations found (all failed)")
            return {}
        
        optimal = {
            'max_throughput': viable.loc[viable['throughput_vph'].idxmax()].to_dict(),
            'min_delay': viable.loc[viable['mean_delay'].idxmin()].to_dict(),
            'best_balanced': viable.loc[viable['combined_score'].idxmin()].to_dict() if 'combined_score' in viable.columns else None
        }
        
        print("\n" + "=" * 70)
        print("OPTIMAL CONFIGURATIONS")
        print("=" * 70)
        
        for objective, config in optimal.items():
            if config:
                print(f"\n{objective.upper()}:")
                print(f"  Scenario: {config['scenario_name']}")
                print(f"  Diameter: {config['diameter']}m, Lanes: {config['lanes']}")
                print(f"  Throughput: {config['throughput_vph']:.0f} veh/hr")
                print(f"  Mean delay: {config['mean_delay']:.1f}s")
        
        return optimal


def main():
    parser = argparse.ArgumentParser(
        description='Orchestrate parameter sweep for roundabout optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for sweep results')
    
    # Override sweep ranges
    parser.add_argument('--diameters', type=float, nargs='+',
                        help='Diameter values to sweep (overrides config)')
    parser.add_argument('--lanes', type=int, nargs='+', choices=[1, 2],
                        help='Lane counts to sweep (overrides config)')
    parser.add_argument('--demand-levels', type=float, nargs='+',
                        help='Demand multipliers to sweep (overrides config)')
    
    parser.add_argument('--parallel', action='store_true',
                        help='Run scenarios in parallel (experimental)')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ParameterSweepOrchestrator(args.config)
    
    # Run sweep
    summary_df = orchestrator.run_sweep(
        output_dir=args.output,
        diameters=args.diameters,
        lanes=args.lanes,
        demand_levels=args.demand_levels,
        parallel=args.parallel
    )
    
    # Identify optimal configurations
    optimal = orchestrator.identify_optimal_configuration(summary_df)


if __name__ == '__main__':
    main()
