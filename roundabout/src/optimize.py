"""
optimize.py - Parameter Sweep Orchestrator
===========================================

Orchestrates automated parameter sweeps for roundabout optimization.
Runs multiple scenarios with varying geometry and demand parameters,
then identifies optimal configurations.

Usage:
    # Grid search (exhaustive)
    python optimize.py --config config/config.yaml --output results/sweep_results/
    
    # Bayesian optimization (intelligent)
    python optimize.py --config config/config.yaml --output results/bayesian_results/ --method bayesian --n-calls 50
    
    # Re-analyze existing results
    python optimize.py --config config/config.yaml --output results/sweep_results/ --skip-simulation

Key Features:
- Grid search over diameter × lanes × demand multipliers
- Bayesian optimization for efficient parameter exploration
- Automated network/route generation and simulation
- Failure detection and classification
- Multi-objective optimization (throughput, delay, balance)
- Comparative visualizations
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

# Bayesian optimization imports (optional)
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Bayesian optimization unavailable.")
    print("Install with: pip install scikit-optimize")


class ParameterSweepOrchestrator:
    """Manages automated parameter sweep experiments."""
    
    def __init__(self, config_path: str, output_dir: str):
        """
        Initialize sweep orchestrator.
        
        Args:
            config_path: Path to base config.yaml
            output_dir: Root directory for sweep results
        """
        self.config_path = config_path
        self.output_dir = output_dir
        
        # Load base configuration
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Extract sweep parameters from config
        self.sweep_params = self.base_config.get('parameter_sweep', {})
        self.diameters = self.sweep_params.get('diameters', [35, 45, 55])
        self.lane_configs = self.sweep_params.get('lane_configs', [1, 2])
        self.demand_multipliers = self.sweep_params.get('demand_multipliers', [0.5, 0.75, 1.0, 1.25, 1.5])
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.sumo_dir = os.path.join(output_dir, 'sumo_configs')
        self.results_dir = os.path.join(output_dir, 'raw_results')
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.sumo_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Tracking
        self.scenarios: List[Dict] = []
        self.results: List[Dict] = []
    
    def generate_scenarios(self) -> List[Dict]:
        """
        Generate all scenario combinations from sweep parameters.
        
        Returns:
            List of scenario dictionaries with parameter sets
        """
        scenarios = []
        
        for diameter in self.diameters:
            for lanes in self.lane_configs:
                for demand_mult in self.demand_multipliers:
                    scenario = {
                        'name': f'd{diameter}_l{lanes}_dm{demand_mult:.2f}',
                        'diameter': diameter,
                        'lanes': lanes,
                        'demand_multiplier': demand_mult,
                        'config_dir': os.path.join(self.sumo_dir, f'd{diameter}_l{lanes}_dm{demand_mult:.2f}'),
                        'result_file': os.path.join(self.results_dir, f'd{diameter}_l{lanes}_dm{demand_mult:.2f}.csv')
                    }
                    scenarios.append(scenario)
        
        self.scenarios = scenarios
        print(f"Generated {len(scenarios)} scenarios")
        return scenarios
    
    def run_scenario(self, scenario: Dict) -> bool:
        """
        Execute a single scenario: generate network, routes, run simulation.
        
        Args:
            scenario: Scenario parameter dictionary
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*70}")
        print(f"  Diameter: {scenario['diameter']}m")
        print(f"  Lanes: {scenario['lanes']}")
        print(f"  Demand multiplier: {scenario['demand_multiplier']}x")
        
        try:
            # Step 1: Generate network
            print("\n[1/3] Generating network...")
            net_cmd = [
                'python3', 'src/generate_network.py',
                '--config', self.config_path,
                '--diameter', str(scenario['diameter']),
                '--lanes', str(scenario['lanes']),
                '--output', scenario['config_dir']
            ]
            subprocess.run(net_cmd, check=True, capture_output=True, text=True)
            network_file = os.path.join(scenario['config_dir'], 'roundabout.net.xml')
            print(f"  ✓ Network: {network_file}")
            
            # Step 2: Generate routes
            print("[2/3] Generating routes...")
            route_cmd = [
                'python3', 'src/generate_routes.py',
                '--config', self.config_path,
                '--network', network_file,
                '--demand-multiplier', str(scenario['demand_multiplier']),
                '--output', scenario['config_dir']
            ]
            subprocess.run(route_cmd, check=True, capture_output=True, text=True)
            sumocfg_file = os.path.join(scenario['config_dir'], 'roundabout.sumocfg')
            print(f"  ✓ Config: {sumocfg_file}")
            
            # Step 3: Run simulation
            print("[3/3] Running simulation...")
            sim_cmd = [
                'python3', 'src/run_simulation.py',
                '--sumocfg', sumocfg_file,
                '--config', self.config_path,
                '--output', scenario['result_file']
            ]
            subprocess.run(sim_cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ Results: {scenario['result_file']}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED: {e}")
            if e.stderr:
                print(f"  Error output: {e.stderr}")
            return False
    
    def run_all_scenarios(self):
        """Execute all scenarios in the sweep."""
        print(f"\n{'='*70}")
        print(f"STARTING PARAMETER SWEEP")
        print(f"{'='*70}")
        print(f"Total scenarios: {len(self.scenarios)}")
        print(f"Output directory: {self.output_dir}")
        
        successful = 0
        failed = 0
        
        for i, scenario in enumerate(self.scenarios, 1):
            print(f"\n[{i}/{len(self.scenarios)}] Running scenario: {scenario['name']}")
            
            if self.run_scenario(scenario):
                successful += 1
                scenario['status'] = 'success'
            else:
                failed += 1
                scenario['status'] = 'failed'
        
        print(f"\n{'='*70}")
        print(f"SWEEP COMPLETE")
        print(f"{'='*70}")
        print(f"Total scenarios: {len(self.scenarios)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        # Save sweep metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config_file': self.config_path,
            'total_scenarios': len(self.scenarios),
            'successful': successful,
            'failed': failed,
            'scenarios': self.scenarios
        }
        
        metadata_file = os.path.join(self.output_dir, 'sweep_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def analyze_sweep_results(self) -> pd.DataFrame:
        """
        Aggregate and analyze results from all scenarios.
        
        Returns:
            DataFrame with summary statistics per scenario
        """
        print(f"\n{'='*70}")
        print(f"ANALYZING SWEEP RESULTS")
        print(f"{'='*70}")
        
        # Run analysis on each successful scenario
        for scenario in self.scenarios:
            if scenario.get('status') != 'success':
                continue
            
            print(f"\nAnalyzing: {scenario['name']}")
            
            try:
                # Run analyze_results.py
                analyze_cmd = [
                    'python3', 'src/analyze_results.py',
                    '--input', scenario['result_file'],
                    '--output', scenario['result_file'].replace('.csv', '_analysis.csv')
                ]
                subprocess.run(analyze_cmd, check=True, capture_output=True, text=True)
                print(f"  ✓ Analysis complete")
                
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Analysis failed: {e}")
        
        # Aggregate all analysis results
        summary_rows = []
        
        for scenario in self.scenarios:
            if scenario.get('status') != 'success':
                continue
            
            analysis_file = scenario['result_file'].replace('.csv', '_analysis.csv')
            
            if not os.path.exists(analysis_file):
                continue
            
            try:
                df = pd.read_csv(analysis_file)
                
                # Extract key metrics
                row = {
                    'scenario': scenario['name'],
                    'diameter': scenario['diameter'],
                    'lanes': scenario['lanes'],
                    'demand_multiplier': scenario['demand_multiplier'],
                    'mean_throughput_vph': df['mean_throughput_vph'].iloc[0],
                    'mean_delay_s': df['mean_delay_s'].iloc[0],
                    'p95_delay_s': df['p95_delay_s'].iloc[0],
                    'max_queue_overall': df['max_queue_overall'].iloc[0],
                    'total_co2_g': df['total_co2_g'].iloc[0],
                    'failure_detected': df['failure_detected'].iloc[0] if 'failure_detected' in df.columns else False,
                    'failure_reason': df['failure_reason'].iloc[0] if 'failure_reason' in df.columns else 'none'
                }
                
                summary_rows.append(row)
                
            except Exception as e:
                print(f"  Warning: Could not parse {analysis_file}: {e}")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'sweep_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n✓ Analysis summary saved: {summary_file}")
        print(f"  Total analyzed scenarios: {len(summary_rows)}")
        
        return summary_df
    
    def generate_visualizations(self, summary_df: pd.DataFrame):
        """
        Generate comparative visualizations for sweep results.
        
        Args:
            summary_df: Summary DataFrame from analyze_sweep_results()
        """
        print(f"\n{'='*70}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        # Use visualize_results.py for both static and interactive plots
        summary_file = os.path.join(self.output_dir, 'sweep_summary.csv')
        
        try:
            # Generate static plots
            viz_cmd = [
                'python3', 'src/visualize_results.py',
                '--input', summary_file,
                '--output', self.plots_dir,
                '--static'
            ]
            subprocess.run(viz_cmd, check=True)
            print(f"✓ Static plots saved to: {self.plots_dir}")
            
            # Generate interactive plots
            viz_cmd = [
                'python3', 'src/visualize_results.py',
                '--input', summary_file,
                '--output', self.plots_dir,
                '--interactive'
            ]
            subprocess.run(viz_cmd, check=True)
            print(f"✓ Interactive plots saved to: {self.plots_dir}")
            
            print(f"\n✓ All visualizations complete!")
            print(f"View results in: {self.plots_dir}")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Visualization generation failed: {e}")
    
    def identify_optimal_configuration(self, summary_df: pd.DataFrame) -> Dict:
        """
        Identify optimal configurations based on different criteria.
        
        Args:
            summary_df: Summary DataFrame with all scenario results
            
        Returns:
            Dictionary mapping criteria to optimal scenario names
        """
        print(f"\n{'='*70}")
        print(f"OPTIMAL CONFIGURATIONS")
        print(f"{'='*70}")
        
        # Filter out failed scenarios
        valid_df = summary_df[summary_df['failure_detected'] == False].copy()
        
        if len(valid_df) == 0:
            print("⚠ No valid (non-failed) scenarios found!")
            return {}
        
        optimal = {}
        
        # 1. Maximum throughput
        max_throughput_idx = valid_df['mean_throughput_vph'].idxmax()
        max_throughput_row = valid_df.loc[max_throughput_idx]
        optimal['MAX_THROUGHPUT'] = max_throughput_row['scenario']
        
        print(f"\nMAX_THROUGHPUT:")
        print(f"  Scenario: {max_throughput_row['scenario']}")
        print(f"  Diameter: {max_throughput_row['diameter']}m, Lanes: {max_throughput_row['lanes']}")
        print(f"  Demand: {max_throughput_row['demand_multiplier']}x")
        print(f"  Throughput: {max_throughput_row['mean_throughput_vph']:.1f} veh/hr")
        print(f"  Delay: {max_throughput_row['mean_delay_s']:.1f}s")
        
        # 2. Minimum delay
        min_delay_idx = valid_df['mean_delay_s'].idxmin()
        min_delay_row = valid_df.loc[min_delay_idx]
        optimal['MIN_DELAY'] = min_delay_row['scenario']
        
        print(f"\nMIN_DELAY:")
        print(f"  Scenario: {min_delay_row['scenario']}")
        print(f"  Diameter: {min_delay_row['diameter']}m, Lanes: {min_delay_row['lanes']}")
        print(f"  Demand: {min_delay_row['demand_multiplier']}x")
        print(f"  Delay: {min_delay_row['mean_delay_s']:.1f}s")
        print(f"  Throughput: {min_delay_row['mean_throughput_vph']:.1f} veh/hr")
        
        # 3. Best balance (high throughput, low delay)
        # Normalize metrics and compute composite score
        valid_df['throughput_norm'] = (valid_df['mean_throughput_vph'] - valid_df['mean_throughput_vph'].min()) / \
                                       (valid_df['mean_throughput_vph'].max() - valid_df['mean_throughput_vph'].min())
        valid_df['delay_norm'] = (valid_df['mean_delay_s'].max() - valid_df['mean_delay_s']) / \
                                 (valid_df['mean_delay_s'].max() - valid_df['mean_delay_s'].min())
        valid_df['balance_score'] = 0.6 * valid_df['throughput_norm'] + 0.4 * valid_df['delay_norm']
        
        best_balance_idx = valid_df['balance_score'].idxmax()
        best_balance_row = valid_df.loc[best_balance_idx]
        optimal['BEST_BALANCE'] = best_balance_row['scenario']
        
        print(f"\nBEST_BALANCE (60% throughput, 40% delay):")
        print(f"  Scenario: {best_balance_row['scenario']}")
        print(f"  Diameter: {best_balance_row['diameter']}m, Lanes: {best_balance_row['lanes']}")
        print(f"  Demand: {best_balance_row['demand_multiplier']}x")
        print(f"  Balance score: {best_balance_row['balance_score']:.3f}")
        print(f"  Throughput: {best_balance_row['mean_throughput_vph']:.1f} veh/hr")
        print(f"  Delay: {best_balance_row['mean_delay_s']:.1f}s")
        
        # 4. Lowest emissions
        min_co2_idx = valid_df['total_co2_g'].idxmin()
        min_co2_row = valid_df.loc(min_co2_idx)
        optimal['MIN_EMISSIONS'] = min_co2_row['scenario']
        
        print(f"\nMIN_EMISSIONS:")
        print(f"  Scenario: {min_co2_row['scenario']}")
        print(f"  Diameter: {min_co2_row['diameter']}m, Lanes: {min_co2_row['lanes']}")
        print(f"  Demand: {min_co2_row['demand_multiplier']}x")
        print(f"  CO2: {min_co2_row['total_co2_g']:.1f}g")
        
        # Save optimal configurations
        optimal_file = os.path.join(self.output_dir, 'optimal_configurations.json')
        with open(optimal_file, 'w') as f:
            json.dump(optimal, f, indent=2)
        
        print(f"\n✓ Optimal configurations saved: {optimal_file}")
        print("="*70 + "\n")
        
        return optimal
    
    def bayesian_optimize(self, n_calls: int = 50, objective: str = 'balance') -> Dict:
        """
        Use Bayesian optimization to intelligently search parameter space.
        
        Args:
            n_calls: Number of evaluations to perform (default: 50)
            objective: Optimization objective - 'throughput', 'delay', or 'balance'
            
        Returns:
            Dictionary with optimization results and best parameters
        """
        if not BAYESIAN_AVAILABLE:
            print("ERROR: scikit-optimize not installed!")
            print("Install with: pip install scikit-optimize")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print(f"BAYESIAN OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Objective: {objective}")
        print(f"Budget: {n_calls} evaluations")
        print(f"Parameter space:")
        print(f"  - Diameter: [30, 60] meters")
        print(f"  - Lanes: {{1, 2}}")
        print(f"  - Demand multiplier: [0.5, 1.5]")
        print("="*70 + "\n")
        
        # Define search space
        space = [
            Real(30.0, 60.0, name='diameter'),       # Continuous diameter
            Integer(1, 2, name='lanes'),             # Discrete lanes (1 or 2)
            Real(0.5, 1.5, name='demand_multiplier') # Continuous demand
        ]
        
        # Track evaluations
        self.bayesian_results = []
        
        @use_named_args(space)
        def objective_function(diameter, lanes, demand_multiplier):
            """
            Objective function to minimize (negative performance for maximization).
            
            Returns:
                Scalar objective value (lower is better)
            """
            # Round diameter to nearest 5m for practical implementations
            diameter_rounded = round(diameter / 5) * 5
            
            # Create scenario
            scenario_name = f"bayes_d{diameter_rounded:.0f}_l{lanes}_dm{demand_multiplier:.2f}"
            scenario = {
                'name': scenario_name,
                'diameter': diameter_rounded,
                'lanes': lanes,
                'demand_multiplier': demand_multiplier,
                'config_dir': os.path.join(self.sumo_dir, scenario_name),
                'result_file': os.path.join(self.results_dir, f'{scenario_name}.csv')
            }
            
            print(f"\n[Evaluation {len(self.bayesian_results) + 1}/{n_calls}]")
            print(f"  Testing: diameter={diameter_rounded:.0f}m, lanes={lanes}, demand={demand_multiplier:.2f}x")
            
            # Run scenario
            success = self.run_scenario(scenario)
            
            if not success:
                print(f"  ✗ Simulation failed - returning penalty")
                # Return large penalty for failed simulations
                return 1e6
            
            # Analyze results
            try:
                analysis_file = scenario['result_file'].replace('.csv', '_analysis.csv')
                
                # Run analysis
                analyze_cmd = [
                    'python3', 'src/analyze_results.py',
                    '--input', scenario['result_file'],
                    '--output', analysis_file,
                    '--config', self.config_path
                ]
                subprocess.run(analyze_cmd, check=True, capture_output=True, text=True)
                
                # Load results
                df = pd.read_csv(analysis_file)
                
                throughput = df['mean_throughput_vph'].iloc[0]
                delay = df['mean_delay_s'].iloc[0]
                failure = df['failure_detected'].iloc[0] if 'failure_detected' in df.columns else False
                
                # Apply penalty for failures
                if failure:
                    print(f"  ⚠ Failure detected - applying penalty")
                    obj_value = 1e5
                else:
                    # Compute objective based on chosen criterion
                    if objective == 'throughput':
                        obj_value = -throughput  # Minimize negative = maximize throughput
                    elif objective == 'delay':
                        obj_value = delay  # Minimize delay
                    else:  # 'balance'
                        # Composite: minimize delay while maximizing throughput
                        # Normalize and combine (weights: 60% throughput, 40% delay)
                        # Use penalties if outside reasonable ranges
                        throughput_score = max(0, min(1, (throughput - 1500) / 2000))
                        delay_score = max(0, min(1, (60 - delay) / 50))
                        obj_value = -(0.6 * throughput_score + 0.4 * delay_score)
                
                # Store result
                result = {
                    'iteration': len(self.bayesian_results) + 1,
                    'diameter': diameter_rounded,
                    'lanes': lanes,
                    'demand_multiplier': demand_multiplier,
                    'throughput': throughput,
                    'delay': delay,
                    'failure': failure,
                    'objective_value': obj_value
                }
                self.bayesian_results.append(result)
                
                print(f"  → Throughput: {throughput:.1f} veh/hr, Delay: {delay:.1f}s")
                print(f"  → Objective: {obj_value:.4f}")
                
                return obj_value
                
            except Exception as e:
                print(f"  ✗ Analysis failed: {e}")
                return 1e6
        
        # Run Bayesian optimization
        print("\nStarting optimization...\n")
        
        result = gp_minimize(
            objective_function,
            space,
            n_calls=n_calls,
            n_initial_points=10,  # Random exploration first
            acq_func='EI',        # Expected Improvement acquisition
            random_state=42,       # Reproducibility
            verbose=False
        )
        
        # Extract best parameters
        best_diameter = round(result.x[0] / 5) * 5
        best_lanes = result.x[1]
        best_demand = result.x[2]
        best_objective = result.fun
        
        # Find corresponding result
        best_result = min(self.bayesian_results, key=lambda r: r['objective_value'])
        
        print(f"\n{'='*70}")
        print(f"BAYESIAN OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nBest configuration found:")
        print(f"  Diameter: {best_diameter:.0f}m")
        print(f"  Lanes: {best_lanes}")
        print(f"  Demand multiplier: {best_demand:.2f}x")
        print(f"\nPerformance:")
        print(f"  Throughput: {best_result['throughput']:.1f} veh/hr")
        print(f"  Mean delay: {best_result['delay']:.1f}s")
        print(f"  Objective value: {best_objective:.4f}")
        print(f"\nTotal evaluations: {len(self.bayesian_results)}")
        print("="*70 + "\n")
        
        # Save results
        results_df = pd.DataFrame(self.bayesian_results)
        results_file = os.path.join(self.output_dir, 'bayesian_optimization_history.csv')
        results_df.to_csv(results_file, index=False)
        print(f"✓ Optimization history saved: {results_file}")
        
        # Save best config
        best_config = {
            'best_parameters': {
                'diameter': float(best_diameter),
                'lanes': int(best_lanes),
                'demand_multiplier': float(best_demand)
            },
            'performance': {
                'throughput_vph': float(best_result['throughput']),
                'mean_delay_s': float(best_result['delay']),
                'objective_value': float(best_objective)
            },
            'optimization_info': {
                'method': 'Bayesian Optimization (Gaussian Process + EI)',
                'n_evaluations': n_calls,
                'objective': objective,
                'acquisition_function': 'Expected Improvement'
            }
        }
        
        best_file = os.path.join(self.output_dir, 'bayesian_best_config.json')
        with open(best_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"✓ Best configuration saved: {best_file}\n")
        
        return best_config


def main():
    parser = argparse.ArgumentParser(
        description='Run parameter sweep for roundabout optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run grid search (exhaustive - 30 scenarios)
  python optimize.py --config config/config.yaml --output results/sweep_results/
  
  # Run Bayesian optimization (intelligent - 50 evaluations)
  python optimize.py --config config/config.yaml --output results/bayesian_results/ --method bayesian --n-calls 50
  
  # Bayesian with different objective
  python optimize.py --config config/config.yaml --output results/bayesian_throughput/ --method bayesian --objective throughput
  
  # Skip simulation, analyze existing results
  python optimize.py --config config/config.yaml --output results/sweep_results/ --skip-simulation
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base config.yaml file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for sweep results')
    parser.add_argument('--method', type=str, choices=['grid', 'bayesian'], default='grid',
                        help='Optimization method: grid (exhaustive) or bayesian (intelligent)')
    parser.add_argument('--n-calls', type=int, default=50,
                        help='Number of evaluations for Bayesian optimization (default: 50)')
    parser.add_argument('--objective', type=str, choices=['throughput', 'delay', 'balance'], default='balance',
                        help='Optimization objective for Bayesian method')
    parser.add_argument('--skip-simulation', action='store_true',
                        help='Skip simulation, only analyze existing results')
    
    args = parser.parse_args()
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    # Create orchestrator
    orchestrator = ParameterSweepOrchestrator(args.config, args.output)
    
    if args.skip_simulation:
        print(f"\n{'='*70}")
        print(f"SKIPPING SIMULATION - USING EXISTING RESULTS")
        print(f"{'='*70}\n")
        
        # Try to load existing scenarios from metadata
        metadata_file = os.path.join(args.output, 'sweep_metadata.json')
        scenarios_loaded = False
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if 'scenarios' in metadata:
                        orchestrator.scenarios = metadata['scenarios']
                        print(f"✓ Loaded {len(orchestrator.scenarios)} scenarios from metadata")
                        scenarios_loaded = True
                    else:
                        print("⚠ Metadata file found but 'scenarios' key missing")
            except Exception as e:
                print(f"⚠ Could not load metadata: {e}")
        
        # Fallback: detect scenarios from existing result files
        if not scenarios_loaded:
            print("Detecting scenarios from existing result files...")
            
            import glob
            result_files = glob.glob(os.path.join(args.output, 'raw_results', '*.csv'))
            
            # Filter out analysis files
            result_files = [f for f in result_files if not f.endswith('_analysis.csv') and not f.endswith('_aggregate.csv')]
            
            if not result_files:
                print("✗ No result files found in results/sweep_results/raw_results/")
                sys.exit(1)
            
            scenarios = []
            for result_file in result_files:
                # Parse scenario name from filename
                scenario_name = os.path.basename(result_file).replace('.csv', '')
                
                # Parse parameters from name (e.g., "d35_l1_dm0.50")
                try:
                    parts = scenario_name.split('_')
                    diameter = int(parts[0][1:])  # Remove 'd' prefix
                    lanes = int(parts[1][1:])     # Remove 'l' prefix
                    demand_mult = float(parts[2][2:])  # Remove 'dm' prefix
                    
                    scenario = {
                        'name': scenario_name,
                        'diameter': diameter,
                        'lanes': lanes,
                        'demand_multiplier': demand_mult,
                        'config_dir': os.path.join(orchestrator.sumo_dir, scenario_name),
                        'result_file': result_file,
                        'status': 'success'  # Assume success if file exists
                    }
                    scenarios.append(scenario)
                    
                except Exception as e:
                    print(f"  Warning: Could not parse scenario from {scenario_name}: {e}")
            
            orchestrator.scenarios = scenarios
            print(f"✓ Detected {len(scenarios)} scenarios from result files")
    
    elif args.method == 'bayesian':
        # Run Bayesian optimization
        best_config = orchestrator.bayesian_optimize(
            n_calls=args.n_calls,
            objective=args.objective
        )
        
        print("\n" + "="*70)
        print("BAYESIAN OPTIMIZATION COMPLETE!")
        print("="*70)
        print(f"Results directory: {args.output}")
        print(f"Best config: {os.path.join(args.output, 'bayesian_best_config.json')}")
        print(f"History: {os.path.join(args.output, 'bayesian_optimization_history.csv')}")
        print("="*70 + "\n")
        
    else:  # grid search
        # Generate scenarios
        orchestrator.generate_scenarios()
        
        # Run all scenarios
        orchestrator.run_all_scenarios()
        
        # Analyze results
        summary_df = orchestrator.analyze_sweep_results()
        
        # Generate visualizations
        orchestrator.generate_visualizations(summary_df)
        
        # Identify optimal configurations
        optimal = orchestrator.identify_optimal_configuration(summary_df)
        
        print("\n" + "="*70)
        print("GRID SEARCH COMPLETE!")
        print("="*70)
        print(f"Results directory: {args.output}")
        print(f"Summary: {os.path.join(args.output, 'sweep_summary.csv')}")
        print(f"Plots: {os.path.join(args.output, 'plots/')}")
        print(f"Optimal configs: {os.path.join(args.output, 'optimal_configurations.json')}")
        print("="*70 + "\n")


if __name__ == '__main__':
    main()
