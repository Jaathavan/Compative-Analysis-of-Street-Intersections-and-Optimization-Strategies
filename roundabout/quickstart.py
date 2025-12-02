#!/usr/bin/env python3
"""
quickstart.py - Quick Start Demo Script
========================================

Demonstrates the complete roundabout simulation pipeline with a simple example.
This script runs a baseline scenario and generates all outputs.

Usage:
    python quickstart.py
    python quickstart.py --gui  # Launch SUMO-GUI for visualization
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description, cwd=None):
    """Run a command and display status."""
    print(f"\n{'='*70}")
    print(f"Step: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        if e.stderr:
            print(f"Error output:\n{e.stderr}")
        return False


def main():
    """Run quickstart demonstration."""
    
    # Determine if GUI mode requested
    gui_mode = '--gui' in sys.argv
    
    # Get project root
    project_root = Path(__file__).parent.parent
    roundabout_dir = project_root / 'roundabout'
    
    # Change to roundabout directory
    os.chdir(roundabout_dir)
    
    print(f"\nWorking directory: {roundabout_dir}")
    print(f"GUI mode: {'Enabled' if gui_mode else 'Disabled'}")
    
    # Check SUMO installation
    print("\nChecking SUMO installation...")
    if 'SUMO_HOME' not in os.environ:
        print("✗ ERROR: SUMO_HOME environment variable not set")
        print("\nPlease set SUMO_HOME to your SUMO installation directory:")
        print("  export SUMO_HOME=/usr/share/sumo  # Adjust path as needed")
        sys.exit(1)
    else:
        print(f"✓ SUMO_HOME = {os.environ['SUMO_HOME']}")
    
    # Create output directories
    output_base = 'quickstart_output'
    config_dir = f'{output_base}/sumo_configs/baseline'
    results_dir = f'{output_base}/results'
    plots_dir = f'{output_base}/plots'
    
    for d in [config_dir, results_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)
    
    print(f"\nOutput directory: {output_base}/")
    
    # Step 1: Generate network
    if not run_command(
        ['python3', 'src/generate_network.py',
         '--config', 'config/config.yaml',
         '--output', config_dir],
        "Generate roundabout network (.net.xml)"
    ):
        print("\n✗ Pipeline failed at network generation")
        sys.exit(1)
    
    # Step 2: Generate routes
    if not run_command(
        ['python3', 'src/generate_routes.py',
         '--config', 'config/config.yaml',
         '--network', f'{config_dir}/roundabout.net.xml',
         '--output', config_dir],
        "Generate traffic demand (.rou.xml, .sumocfg)"
    ):
        print("\n✗ Pipeline failed at route generation")
        sys.exit(1)
    
    # Step 3: Run simulation
    sim_cmd = [
        'python3', 'src/run_simulation.py',
        '--sumocfg', f'{config_dir}/roundabout.sumocfg',
        '--config', 'config/config.yaml',
        '--output', f'{results_dir}/baseline.csv'
    ]
    
    if gui_mode:
        sim_cmd.append('--gui')
    
    if not run_command(
        sim_cmd,
        f"Run SUMO simulation ({'GUI' if gui_mode else 'headless'})"
    ):
        print("\n✗ Pipeline failed at simulation")
        sys.exit(1)
    
    # Step 4: Analyze results
    if not run_command(
        ['python3', 'src/analyze_results.py',
         '--input', f'{results_dir}/baseline.csv',
         '--config', 'config/config.yaml',
         '--output', f'{results_dir}/baseline_analysis.csv'],
        "Analyze simulation results"
    ):
        print("\n✗ Pipeline failed at analysis")
        sys.exit(1)
    
    # Step 5: Generate visualizations
    if not run_command(
        ['python3', 'src/visualize_results.py',
         '--input', f'{results_dir}/baseline_analysis.csv',
         '--window-data', f'{results_dir}/baseline.csv',
         '--config', 'config/config.yaml',
         '--output', plots_dir,
         '--interactive'],
        "Generate visualizations (static + interactive)"
    ):
        print("\n✗ Pipeline failed at visualization (continuing...)")


if __name__ == '__main__':
    main()
