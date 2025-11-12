"""
Run SUMO Simulation with Different Control Strategies

Supports three control modes:
1. Webster: Fixed-time optimal timing from Webster's Method
2. PPO: Adaptive control using trained RL agent
3. Actuated: Simple vehicle-actuated control

Usage:
    python run_simulation.py --control webster --demand 1.0
    python run_simulation.py --control ppo --model ../models/ppo_signal_best.zip
    python run_simulation.py --control actuated --demand 1.25
"""

import argparse
import os
import sys
import yaml
import pandas as pd
from datetime import datetime

# Import control strategies
from webster_method import WebsterSignalOptimizer


def run_webster_simulation(
    config_path: str,
    sumo_cfg_path: str,
    demand_multiplier: float,
    output_dir: str
):
    """
    Run simulation with Webster-optimized fixed timing.
    
    Args:
        config_path: Path to config.yaml
        sumo_cfg_path: Path to SUMO config
        demand_multiplier: Demand scaling factor
        output_dir: Output directory for results
    """
    print("\n🚦 Running simulation with Webster's Method...")
    
    # Optimize timing
    optimizer = WebsterSignalOptimizer(config_path=config_path)
    result = optimizer.optimize(demand_multiplier=demand_multiplier, verbose=True)
    
    if not result['cycle_length']:
        print("❌ Cannot run simulation - demand exceeds capacity!")
        return None
    
    # Generate network with Webster timing
    from generate_network import SignalizedNetworkGenerator
    
    webster_dir = os.path.join(output_dir, f'webster_dm{demand_multiplier:.2f}')
    generator = SignalizedNetworkGenerator(config_path, webster_dir)
    net_file = generator.compile_network(use_webster=True, webster_result=result)
    
    # Generate routes
    from generate_routes import SignalizedRouteGenerator
    route_gen = SignalizedRouteGenerator(config_path, demand_multiplier=demand_multiplier)
    route_file = route_gen.generate(webster_dir)
    
    # Run SUMO (basic run, not TraCI)
    import subprocess
    
    sumocfg = os.path.join(webster_dir, 'intersection.sumocfg')
    tripinfo = os.path.join(webster_dir, 'tripinfo.xml')
    summary = os.path.join(webster_dir, 'summary.xml')
    
    # Create sumocfg if not exists
    if not os.path.exists(sumocfg):
        _create_sumocfg(config_path, webster_dir)
    
    print(f"\n⏳ Running SUMO simulation...")
    cmd = [
        'sumo',
        '-c', sumocfg,
        '--tripinfo-output', tripinfo,
        '--summary-output', summary,
        '--no-warnings',
        '--no-step-log'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Simulation complete!")
        
        # Parse results
        metrics = _parse_tripinfo(tripinfo)
        metrics['control_strategy'] = 'webster'
        metrics['demand_multiplier'] = demand_multiplier
        
        return metrics
    else:
        print(f"❌ Simulation failed: {result.stderr}")
        return None


def run_ppo_simulation(
    config_path: str,
    sumo_cfg_path: str,
    model_path: str,
    demand_multiplier: float,
    output_dir: str
):
    """
    Run simulation with PPO adaptive control.
    
    Args:
        config_path: Path to config.yaml
        sumo_cfg_path: Path to SUMO config
        model_path: Path to trained PPO model
        demand_multiplier: Demand scaling factor
        output_dir: Output directory
    """
    print("\n🤖 Running simulation with PPO adaptive control...")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train model first with: python train_ppo.py")
        return None
    
    from stable_baselines3 import PPO
    from ppo_environment import SignalControlEnv
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = SignalControlEnv(
        config_path=config_path,
        sumo_cfg_path=sumo_cfg_path,
        use_gui=False,
        demand_multiplier=demand_multiplier,
        episode_length=3600
    )
    
    # Run episode
    print("⏳ Running episode with trained agent...")
    obs, info = env.reset()
    
    total_reward = 0
    episode_stats = []
    
    done = False
    step = 0
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        episode_stats.append(info.copy())
        
        step += 1
        done = terminated or truncated
        
        if step % 10 == 0:
            print(f"   Step {step}: Reward={reward:.2f}, "
                  f"Throughput={info['throughput']:.1f} veh/hr, "
                  f"Delay={info['avg_delay']:.1f}s")
    
    env.close()
    
    print(f"✅ Episode complete! Total reward: {total_reward:.2f}")
    
    # Aggregate statistics
    metrics = {
        'control_strategy': 'ppo',
        'demand_multiplier': demand_multiplier,
        'total_reward': total_reward,
        'avg_throughput': np.mean([s['throughput'] for s in episode_stats]),
        'avg_delay': np.mean([s['avg_delay'] for s in episode_stats]),
        'max_queue': max([s['max_queue'] for s in episode_stats]),
        'vehicles_completed': episode_stats[-1]['episode_stats']['vehicles_completed']
    }
    
    return metrics


def _create_sumocfg(config_path: str, output_dir: str):
    """Create SUMO config file."""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    root = ET.Element('configuration')
    
    # Input
    input_elem = ET.SubElement(root, 'input')
    ET.SubElement(input_elem, 'net-file', {'value': 'intersection.net.xml'})
    ET.SubElement(input_elem, 'route-files', {'value': 'intersection.rou.xml'})
    
    # Time
    time_elem = ET.SubElement(root, 'time')
    ET.SubElement(time_elem, 'begin', {'value': '0'})
    ET.SubElement(time_elem, 'end', {'value': str(config['simulation']['horizon'])})
    
    # Processing
    proc_elem = ET.SubElement(root, 'processing')
    ET.SubElement(proc_elem, 'time-to-teleport', {'value': '-1'})
    
    # Write file
    sumocfg_path = os.path.join(output_dir, 'intersection.sumocfg')
    rough_string = ET.tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_string = reparsed.toprettyxml(indent='  ')
    
    with open(sumocfg_path, 'w') as f:
        f.write(pretty_string)


def _parse_tripinfo(tripinfo_path: str) -> dict:
    """Parse SUMO tripinfo output."""
    import xml.etree.ElementTree as ET
    
    if not os.path.exists(tripinfo_path):
        return {}
    
    tree = ET.parse(tripinfo_path)
    root = tree.getroot()
    
    trips = root.findall('tripinfo')
    
    if not trips:
        return {}
    
    delays = [float(t.get('waitingTime', 0)) for t in trips]
    durations = [float(t.get('duration', 0)) for t in trips]
    
    return {
        'vehicles_completed': len(trips),
        'avg_delay': np.mean(delays),
        'p95_delay': np.percentile(delays, 95),
        'avg_duration': np.mean(durations),
        'throughput': len(trips) / 3600.0 * 3600  # veh/hr
    }


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Run signalized intersection simulation')
    
    parser.add_argument('--control', type=str, choices=['webster', 'ppo', 'actuated'],
                       default='webster', help='Control strategy')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                       help='Path to config.yaml')
    parser.add_argument('--sumo-cfg', type=str,
                       default='../quickstart_output/sumo_configs/webster/intersection.sumocfg',
                       help='Path to SUMO config')
    parser.add_argument('--model', type=str, default='../models/ppo_signal_best.zip',
                       help='Path to PPO model (for --control ppo)')
    parser.add_argument('--demand', type=float, default=1.0,
                       help='Demand multiplier')
    parser.add_argument('--output', type=str, default='../results/raw',
                       help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Run simulation
    if args.control == 'webster':
        metrics = run_webster_simulation(
            args.config, args.sumo_cfg, args.demand, args.output
        )
    elif args.control == 'ppo':
        metrics = run_ppo_simulation(
            args.config, args.sumo_cfg, args.model, args.demand, args.output
        )
    else:
        print("❌ Actuated control not yet implemented")
        return
    
    if metrics:
        # Save results
        df = pd.DataFrame([metrics])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(args.output, 
                                   f'{args.control}_dm{args.demand:.2f}_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        print(f"\n📊 Results saved: {output_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("SIMULATION RESULTS")
        print("="*70)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:>25}: {value:>10.2f}")
            else:
                print(f"{key:>25}: {value:>10}")
        print("="*70 + "\n")


if __name__ == '__main__':
    import numpy as np
    main()
