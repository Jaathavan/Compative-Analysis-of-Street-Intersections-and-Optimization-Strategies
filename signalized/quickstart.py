#!/usr/bin/env python3
"""
Quick Start Script for Phase 2: Signalized Intersection

This script demonstrates the complete workflow:
1. Webster's Method optimization
2. Network generation
3. Route generation
4. Simulation with Webster timing
5. Preparation for PPO training

Usage:
    python quickstart.py
"""

import os
import sys
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from webster_method import WebsterSignalOptimizer
from generate_network import SignalizedNetworkGenerator
from generate_routes import SignalizedRouteGenerator


def print_banner(text: str):
    """Print formatted section banner."""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")


def main():
    """Run quickstart demonstration."""
    print("\n🚦 PHASE 2: SIGNALIZED INTERSECTION QUICK START")
    print("=" * 70)
    
    # Paths
    config_path = os.path.join(os.path.dirname(__file__), 'config/config.yaml')
    output_dir = os.path.join(os.path.dirname(__file__), 'quickstart_output')
    
    # ========================================================================
    # Step 1: Webster's Method Optimization
    # ========================================================================
    print_banner("STEP 1: Webster's Method Optimization")
    
    optimizer = WebsterSignalOptimizer(config_path=config_path)
    
    print("Testing different demand levels:\n")
    demand_results = {}
    
    for dm in [0.5, 0.75, 1.0, 1.25]:
        print(f"\n{'─'*70}")
        print(f"Demand Multiplier: {dm}x")
        print(f"{'─'*70}")
        
        result = optimizer.optimize(demand_multiplier=dm, verbose=False)
        demand_results[dm] = result
        
        if result['cycle_length']:
            print(f"✅ Optimal Cycle: {result['cycle_length']:.1f}s")
            print(f"   Avg Delay: {result['avg_delay']:.2f}s/veh")
            print(f"   Flow Ratio Y: {result['total_flow_ratio']:.3f}")
        else:
            print(f"❌ OVER-SATURATED (Y = {result['total_flow_ratio']:.3f} >= 1.0)")
            break
    
    # Use baseline demand (1.0x) for network generation
    baseline_result = demand_results[1.0]
    
    if not baseline_result['cycle_length']:
        print("\n❌ ERROR: Baseline demand is over-saturated!")
        print("   Please reduce demand in config.yaml")
        return
    
    # ========================================================================
    # Step 2: Network Generation
    # ========================================================================
    print_banner("STEP 2: SUMO Network Generation")
    
    net_output_dir = os.path.join(output_dir, 'sumo_configs/webster')
    
    print("Generating network with Webster-optimized timing...")
    generator = SignalizedNetworkGenerator(config_path, net_output_dir)
    
    try:
        net_file = generator.compile_network(
            use_webster=True,
            webster_result=baseline_result
        )
        print(f"\n✅ Network generated successfully!")
        print(f"   Output: {net_file}")
    except Exception as e:
        print(f"\n❌ Network generation failed: {e}")
        print("   Make sure SUMO is installed and in your PATH")
        print("   Test with: sumo --version")
        return
    
    # ========================================================================
    # Step 3: Route Generation
    # ========================================================================
    print_banner("STEP 3: Traffic Demand Generation")
    
    print("Generating routes with Poisson arrivals...")
    route_gen = SignalizedRouteGenerator(config_path, demand_multiplier=1.0)
    route_file = route_gen.generate(net_output_dir)
    
    # ========================================================================
    # Step 4: Create SUMO Config File
    # ========================================================================
    print_banner("STEP 4: SUMO Configuration")
    
    # Read config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create .sumocfg file
    from xml.etree import ElementTree as ET
    from xml.dom import minidom
    
    root = ET.Element('configuration')
    root.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
    root.set('xsi:noNamespaceSchemaLocation', "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
    
    # Input
    input_elem = ET.SubElement(root, 'input')
    ET.SubElement(input_elem, 'net-file', {'value': 'intersection.net.xml'})
    ET.SubElement(input_elem, 'route-files', {'value': 'intersection.rou.xml'})
    
    # Time
    time_elem = ET.SubElement(root, 'time')
    ET.SubElement(time_elem, 'begin', {'value': '0'})
    ET.SubElement(time_elem, 'end', {'value': str(config['simulation']['horizon'])})
    ET.SubElement(time_elem, 'step-length', {'value': str(config['simulation']['step_length'])})
    
    # Output
    output_elem = ET.SubElement(root, 'output')
    ET.SubElement(output_elem, 'tripinfo-output', {'value': 'tripinfo.xml'})
    ET.SubElement(output_elem, 'summary-output', {'value': 'summary.xml'})
    
    # Processing
    proc_elem = ET.SubElement(root, 'processing')
    ET.SubElement(proc_elem, 'time-to-teleport', {'value': '-1'})
    
    # Write SUMO config
    sumocfg_path = os.path.join(net_output_dir, 'intersection.sumocfg')
    rough_string = ET.tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_string = reparsed.toprettyxml(indent='  ')
    
    with open(sumocfg_path, 'w') as f:
        f.write(pretty_string)
    
    print(f"✅ SUMO config created: {sumocfg_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_banner("QUICKSTART COMPLETE! 🎉")
    
    print("Generated Files:")
    print(f"  📁 {net_output_dir}/")
    print(f"     ├── intersection.net.xml     (Network topology)")
    print(f"     ├── intersection.rou.xml     (Traffic demand)")
    print(f"     ├── intersection.tll.xml     (Signal timing - Webster optimized)")
    print(f"     └── intersection.sumocfg     (SUMO configuration)")
    
    print("\n" + "─"*70)
    print("Webster-Optimized Signal Timing (Baseline Demand):")
    print("─"*70)
    print(f"  Cycle Length: {baseline_result['cycle_length']:.1f}s")
    print(f"  Green Times:")
    phases = ['NS Left', 'NS Through', 'EW Left', 'EW Through']
    for i, phase in enumerate(phases):
        print(f"    Phase {i+1} ({phase:>11}): {baseline_result['green_times'][i]:>5.1f}s")
    print(f"  Expected Avg Delay: {baseline_result['avg_delay']:.2f}s/veh")
    
    print("\n" + "─"*70)
    print("Next Steps:")
    print("─"*70)
    print("\n1. Run SUMO Simulation with Webster Timing:")
    print(f"   cd {net_output_dir}")
    print(f"   sumo -c intersection.sumocfg")
    
    print("\n2. Visualize in SUMO-GUI:")
    print(f"   sumo-gui -c {os.path.join(net_output_dir, 'intersection.sumocfg')}")
    
    print("\n3. Train PPO Agent (coming next):")
    print(f"   python src/train_ppo.py --timesteps 500000")
    
    print("\n4. Compare Strategies:")
    print(f"   python src/compare_strategies.py")
    
    print("\n" + "="*70)
    print("✨ Phase 2 infrastructure ready for simulation and optimization!")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Quickstart cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during quickstart: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
