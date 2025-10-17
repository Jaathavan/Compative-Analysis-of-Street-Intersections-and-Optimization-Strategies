#!/usr/bin/env python3
"""
analyze_results.py - Post-Process and Analyze Simulation Results
=================================================================

Computes aggregate statistics, identifies failure points, and prepares
data for visualization and comparison.

Usage:
    python analyze_results.py --input results/raw/*.csv --output results/summary.csv
    python analyze_results.py --sweep results/raw/sweep_*.csv --output results/sweep_summary.csv
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import yaml


class ResultsAnalyzer:
    """
    Analyzes simulation results and computes derived metrics.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize analyzer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.failure_config = self.config['failure']
    
    def analyze_single_scenario(self, window_file: str, aggregate_file: str) -> Dict:
        """
        Analyze a single scenario's results.
        
        Args:
            window_file: Path to window metrics CSV
            aggregate_file: Path to aggregate metrics CSV
            
        Returns:
            Dictionary with analysis results
        """
        # Load data
        df_windows = pd.read_csv(window_file)
        df_agg = pd.read_csv(aggregate_file)
        
        # Extract aggregate metrics
        agg = df_agg.iloc[0].to_dict()
        
        # Compute additional derived metrics
        analysis = {
            **agg,
            'scenario_name': Path(window_file).stem,
            'num_windows': len(df_windows),
            'simulation_duration': df_windows['end_time'].max()
        }
        
        # Throughput stability analysis
        throughput_values = df_windows['throughput_vph'].values
        analysis['throughput_mean'] = np.mean(throughput_values)
        analysis['throughput_std'] = np.std(throughput_values)
        analysis['throughput_cv'] = analysis['throughput_std'] / analysis['throughput_mean'] if analysis['throughput_mean'] > 0 else 0
        
        # Delay trends
        delay_values = df_windows['avg_delay'].values
        analysis['delay_trend'] = self._compute_trend(delay_values)
        analysis['delay_max'] = np.max(delay_values)
        
        # Queue growth analysis
        queue_cols = ['max_queue_N', 'max_queue_E', 'max_queue_S', 'max_queue_W']
        for col in queue_cols:
            analysis[f'{col}_max'] = df_windows[col].max()
            analysis[f'{col}_trend'] = self._compute_trend(df_windows[col].values)
        
        # Overall queue metrics
        total_queues = df_windows[queue_cols].sum(axis=1)
        analysis['total_queue_max'] = total_queues.max()
        analysis['total_queue_mean'] = total_queues.mean()
        analysis['queue_growth_rate'] = self._compute_trend(total_queues.values)
        
        # Failure detection
        analysis['failure_detected'] = self._detect_failure(df_windows, analysis)
        analysis['failure_reasons'] = self._identify_failure_reasons(df_windows, analysis)
        
        # Performance classification
        analysis['performance_class'] = self._classify_performance(analysis)
        
        return analysis
    
    def analyze_sweep(self, sweep_pattern: str) -> pd.DataFrame:
        """
        Analyze multiple scenarios from a parameter sweep.
        
        Args:
            sweep_pattern: Glob pattern for sweep result files
            
        Returns:
            DataFrame with all scenario analyses
        """
        window_files = sorted(glob.glob(sweep_pattern))
        
        if not window_files:
            print(f"Warning: No files matched pattern: {sweep_pattern}")
            return pd.DataFrame()
        
        analyses = []
        for window_file in window_files:
            # Find corresponding aggregate file
            agg_file = window_file.replace('.csv', '_aggregate.csv')
            
            if not os.path.exists(agg_file):
                print(f"Warning: Aggregate file not found for {window_file}")
                continue
            
            try:
                analysis = self.analyze_single_scenario(window_file, agg_file)
                analyses.append(analysis)
                print(f"✓ Analyzed: {Path(window_file).name}")
            except Exception as e:
                print(f"✗ Error analyzing {window_file}: {e}")
        
        df = pd.DataFrame(analyses)
        
        # Add comparative metrics
        if len(df) > 1:
            df = self._add_comparative_metrics(df)
        
        return df
    
    def _compute_trend(self, values: np.ndarray) -> float:
        """
        Compute linear trend coefficient.
        
        Positive = increasing, Negative = decreasing, ~0 = stable
        
        Args:
            values: Time series values
            
        Returns:
            Slope of linear regression
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _detect_failure(self, df_windows: pd.DataFrame, analysis: Dict) -> bool:
        """
        Detect if scenario exhibits failure conditions.
        
        Failure defined as:
        - Capacity saturation (throughput plateaus)
        - Queue divergence (unbounded growth)
        - Excessive delay
        
        Args:
            df_windows: Window metrics DataFrame
            analysis: Analysis dictionary
            
        Returns:
            True if failure detected
        """
        # Check queue divergence
        if analysis['queue_growth_rate'] > self.failure_config['queue_growth_rate']:
            return True
        
        if analysis['total_queue_max'] > self.failure_config['queue_divergence_threshold']:
            return True
        
        # Check excessive delays
        if analysis['delay_max'] > self.failure_config['max_acceptable_delay']:
            return True
        
        if analysis['mean_delay'] > self.failure_config['max_acceptable_delay'] / 2:
            return True
        
        # Check for instability in last windows
        last_n = 3
        if len(df_windows) >= last_n:
            last_queues = df_windows[['max_queue_N', 'max_queue_E', 'max_queue_S', 'max_queue_W']].tail(last_n).sum(axis=1)
            if last_queues.mean() > self.failure_config['max_acceptable_queue']:
                return True
        
        return False
    
    def _identify_failure_reasons(self, df_windows: pd.DataFrame, analysis: Dict) -> List[str]:
        """Identify specific failure reasons."""
        reasons = []
        
        if analysis['queue_growth_rate'] > self.failure_config['queue_growth_rate']:
            reasons.append('queue_divergence')
        
        if analysis['total_queue_max'] > self.failure_config['queue_divergence_threshold']:
            reasons.append('excessive_queue')
        
        if analysis['delay_max'] > self.failure_config['max_acceptable_delay']:
            reasons.append('excessive_delay')
        
        if analysis['throughput_cv'] > 0.3:
            reasons.append('unstable_throughput')
        
        return reasons if reasons else ['none']
    
    def _classify_performance(self, analysis: Dict) -> str:
        """
        Classify performance into categories.
        
        Returns:
            'excellent', 'good', 'acceptable', 'poor', or 'failure'
        """
        if analysis['failure_detected']:
            return 'failure'
        
        mean_delay = analysis['mean_delay']
        max_queue = analysis['total_queue_max']
        
        if mean_delay < 10 and max_queue < 5:
            return 'excellent'
        elif mean_delay < 20 and max_queue < 10:
            return 'good'
        elif mean_delay < 40 and max_queue < 20:
            return 'acceptable'
        else:
            return 'poor'
    
    def _add_comparative_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comparative metrics across scenarios."""
        # Normalize metrics to baseline (if identifiable)
        baseline_idx = df['scenario_name'].str.contains('baseline', case=False)
        
        if baseline_idx.any():
            baseline = df[baseline_idx].iloc[0]
            
            df['throughput_vs_baseline'] = df['throughput_vph'] / baseline['throughput_vph']
            df['delay_vs_baseline'] = df['mean_delay'] / baseline['mean_delay']
        
        # Rank scenarios
        df['throughput_rank'] = df['throughput_vph'].rank(ascending=False)
        df['delay_rank'] = df['mean_delay'].rank(ascending=True)
        
        # Combined score (lower is better)
        df['combined_score'] = df['delay_rank'] + df['throughput_rank']
        df['overall_rank'] = df['combined_score'].rank(ascending=True)
        
        return df
    
    def save_summary(self, df: pd.DataFrame, output_file: str):
        """Save analysis summary to CSV."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n✓ Analysis summary saved to: {output_file}")
        
        # Print summary statistics
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Total scenarios analyzed: {len(df)}")
        
        if 'failure_detected' in df.columns:
            n_failures = df['failure_detected'].sum()
            print(f"Failures detected: {n_failures} ({n_failures/len(df)*100:.1f}%)")
        
        if 'performance_class' in df.columns:
            print("\nPerformance distribution:")
            for cls, count in df['performance_class'].value_counts().items():
                print(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")
        
        print("\nTop 5 scenarios by throughput:")
        top5 = df.nlargest(5, 'throughput_vph')[['scenario_name', 'throughput_vph', 'mean_delay', 'performance_class']]
        print(top5.to_string(index=False))
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SUMO simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', type=str, nargs='+',
                        help='Input window CSV file(s) (can use wildcards)')
    parser.add_argument('--sweep', type=str,
                        help='Glob pattern for sweep results (alternative to --input)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--output', type=str, required=True,
                        help='Output summary CSV file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(args.config)
    
    # Determine input mode
    if args.sweep:
        # Analyze sweep pattern
        df = analyzer.analyze_sweep(args.sweep)
    elif args.input:
        # Analyze individual files
        analyses = []
        for window_file in args.input:
            agg_file = window_file.replace('.csv', '_aggregate.csv')
            if os.path.exists(agg_file):
                analysis = analyzer.analyze_single_scenario(window_file, agg_file)
                analyses.append(analysis)
        df = pd.DataFrame(analyses)
    else:
        print("Error: Must specify either --input or --sweep", file=sys.stderr)
        sys.exit(1)
    
    # Save results
    if not df.empty:
        analyzer.save_summary(df, args.output)
    else:
        print("Warning: No data to analyze")


if __name__ == '__main__':
    main()
