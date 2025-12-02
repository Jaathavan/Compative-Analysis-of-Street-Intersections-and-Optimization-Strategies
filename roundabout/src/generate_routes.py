"""
generate_routes.py - SUMO Route and Traffic Demand Generator
=============================================================

Creates route definitions (.rou.xml) and traffic flow specifications
matching the demand patterns from Roundabout.py text simulation.

Key features:
- Poisson arrival process per arm (matching text sim's exponential draws)
- Turning movement ratios (L/T/R percentages)
- Vehicle type distribution (passenger/truck/bus)
- Time-dependent demand (optional scaling over simulation period)

Usage:
    python generate_routes.py --config config/config.yaml --network sumo_configs/baseline/roundabout.net.xml --output sumo_configs/baseline
    python generate_routes.py --config config/config.yaml --network sumo_configs/baseline/roundabout.net.xml --demand-multiplier 1.5 --output sumo_configs/high_demand
"""

import argparse
import os
import sys
import random
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Tuple


class RouteGenerator:
    """
    Generates SUMO route files matching text simulation demand patterns.
    
    Traffic is generated using:
    - Flow definitions with vehsPerHour (Poisson process)
    - Route distributions based on turning probabilities
    - Vehicle type mix from config
    """
    
    def __init__(self, config: Dict, demand_multiplier: float = 1.0):
        """
        Initialize generator with configuration.
        
        Args:
            config: Configuration dictionary from config.yaml
            demand_multiplier: Scale factor for demand (1.0 = baseline)
        """
        self.config = config
        self.demand = config['demand']
        self.driver = config['driver']
        self.gap = config['gap_acceptance']
        self.sim = config['simulation']
        self.demand_multiplier = demand_multiplier
        
        # Extract parameters
        self.arrivals = [rate * demand_multiplier for rate in self.demand['arrivals']]
        self.turning = self.demand['turning_probabilities']
        self.vehicle_types = self.demand['vehicle_types']
        
        # Arm and direction mappings
        self.arms = ['N', 'E', 'S', 'W']
        self.arm_indices = {arm: i for i, arm in enumerate(self.arms)}
        
        # Turning movement mappings
        # Right = +1, Through = +2, Left = +3 (modulo 4)
        self.turn_steps = {'R': 1, 'T': 2, 'L': 3}
        
        # Random seed
        random.seed(self.sim['seed'])
    
    def generate(self, output_dir: str) -> str:
        """
        Generate complete route file and save to output directory.
        
        Args:
            output_dir: Directory to save .rou.xml file
            
        Returns:
            Path to generated route file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Build XML tree
        root = ET.Element('routes')
        root.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
        root.set('xsi:noNamespaceSchemaLocation', "http://sumo.dlr.de/xsd/routes_file.xsd")
        
        # Add vehicle type definitions
        vtypes = self._create_vehicle_types()
        for vtype in vtypes:
            vtype_elem = ET.SubElement(root, 'vType')
            for key, val in vtype.items():
                vtype_elem.set(key, str(val))
        
        # Add route definitions
        routes = self._create_routes()
        for route in routes:
            route_elem = ET.SubElement(root, 'route')
            for key, val in route.items():
                route_elem.set(key, str(val))
        
        # Add flow definitions
        flows = self._create_flows()
        for flow in flows:
            flow_elem = ET.SubElement(root, 'flow')
            for key, val in flow.items():
                flow_elem.set(key, str(val))
        
        # Pretty print and save
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        route_file = os.path.join(output_dir, 'roundabout.rou.xml')
        
        with open(route_file, 'w') as f:
            f.write(xml_str)
        
        # Print summary
        total_rate = sum(self.arrivals)
        total_vph = total_rate * 3600
        print(f"✓ Generated routes: {route_file}")
        print(f"  - Per-arm rates: {[f'{r:.3f}' for r in self.arrivals]} veh/s")
        print(f"  - Total inflow: {total_rate:.3f} veh/s = {total_vph:.0f} veh/hr")
        print(f"  - Turning ratios: L={self.turning[0]:.2f}, T={self.turning[1]:.2f}, R={self.turning[2]:.2f}")
        print(f"  - Demand multiplier: {self.demand_multiplier:.2f}x")
        
        return route_file
    
    def _create_vehicle_types(self) -> List[Dict]:
        """
        Create vehicle type definitions matching driver behavior parameters.
        
        Returns:
            List of vType dictionaries
        """
        vtypes = []
        
        # Base parameters from config
        base_params = {
            'accel': self.driver['accel'],
            'decel': self.driver['decel'],
            'emergencyDecel': self.driver['emergency_decel'],
            'sigma': self.driver['sigma'],
            'tau': self.driver['tau'],
            'minGap': self.driver['min_gap'],
            'maxSpeed': self.driver['max_speed'],
            'speedFactor': self.driver['speed_factor'],
            'speedDev': self.driver['speed_dev'],
            'carFollowModel': 'IDM',
            'actionStepLength': self.config['driver'].get('tau', 0.8),  # Reaction delay
            # Junction model parameters for gap acceptance
            'jmTimegapMinor': self.gap['jm_timegap_minor'],
            'jmDriveAfterRedTime': self.gap['jm_drive_after_red'],
            'jmIgnoreKeepClearTime': 5.0,
            'jmSigmaMinor': 0.3,  # Variation in gap acceptance
            # Lane-changing
            'lcStrategic': self.driver['lcStrategic'],
            'lcCooperative': self.driver['lcCooperative'],
            'lcSpeedGain': self.driver['lcSpeedGain']
        }
        
        # Passenger car (85%)
        vtypes.append({
            'id': 'passenger',
            'length': '5.0',
            'width': '1.8',
            'vClass': 'passenger',
            'color': '0.8,0.8,0.8',
            **base_params
        })
        
        # Truck (10%) - longer, slower acceleration
        vtypes.append({
            'id': 'truck',
            'length': '12.0',
            'width': '2.4',
            'vClass': 'truck',
            'color': '0.6,0.4,0.2',
            **{**base_params, 
               'accel': base_params['accel'] * 0.6,  # Slower accel
               'maxSpeed': base_params['maxSpeed'] * 0.8,  # Lower speed
               'speedFactor': 0.9}
        })
        
        # Bus (5%) - longest, moderate speed
        vtypes.append({
            'id': 'bus',
            'length': '15.0',
            'width': '2.5',
            'vClass': 'bus',
            'color': '0.2,0.4,0.8',
            **{**base_params,
               'accel': base_params['accel'] * 0.7,
               'maxSpeed': base_params['maxSpeed'] * 0.85,
               'speedFactor': 0.92}
        })
        
        return vtypes
    
    def _create_routes(self) -> List[Dict]:
        """
        Create route definitions for all origin-destination pairs.
        
        Routes named as: route_{origin}_{destination}
        E.g., route_N_S (North to South = through movement)
        
        Returns:
            List of route dictionaries
        """
        routes = []
        
        for origin in self.arms:
            origin_idx = self.arm_indices[origin]
            
            # For each turning movement
            for turn_type, steps in self.turn_steps.items():
                dest_idx = (origin_idx + steps) % 4
                dest = self.arms[dest_idx]
                
                # Build edge sequence: approach → ring segments → exit
                edges = [f'approach_{origin}']
                
                # Add ring segments until destination
                current_idx = origin_idx
                for _ in range(steps):
                    edges.append(f'ring_{self.arms[current_idx]}')
                    current_idx = (current_idx + 1) % 4
                
                # Add exit
                edges.append(f'exit_{dest}')
                
                routes.append({
                    'id': f'route_{origin}_{dest}',
                    'edges': ' '.join(edges)
                })
        
        return routes
    
    def _create_flows(self) -> List[Dict]:
        """
        Create flow definitions with Poisson arrivals and route distributions.
        
        Each arm gets a flow with:
        - Rate matching arrivals[arm] (converted to vehsPerHour)
        - Route distribution based on turning_probabilities
        - Vehicle type distribution
        
        Returns:
            List of flow dictionaries
        """
        flows = []
        flow_id = 0
        
        for arm in self.arms:
            arm_idx = self.arm_indices[arm]
            rate_veh_per_sec = self.arrivals[arm_idx]
            rate_veh_per_hour = rate_veh_per_sec * 3600
            
            # Create flow for each turning movement at this arm
            for turn_type, steps in [('R', 1), ('T', 2), ('L', 3)]:
                dest_idx = (arm_idx + steps) % 4
                dest = self.arms[dest_idx]
                route_id = f'route_{arm}_{dest}'
                
                # Turning probability
                prob_idx = {'R': 2, 'T': 1, 'L': 0}[turn_type]
                turn_prob = self.turning[prob_idx]
                
                # Flow rate for this movement
                movement_vph = rate_veh_per_hour * turn_prob
                
                if movement_vph < 0.01:  # Skip negligible flows
                    continue
                
                # Distribute vehicle types
                for vtype, fraction in self.vehicle_types.items():
                    type_vph = movement_vph * fraction
                    
                    if type_vph < 0.01:
                        continue
                    
                    flows.append({
                        'id': f'flow_{flow_id}',
                        'type': vtype,
                        'route': route_id,
                        'begin': '0.00',
                        'end': f'{self.sim["horizon"]:.2f}',
                        'vehsPerHour': f'{type_vph:.2f}',
                        'departLane': 'best',
                        'departSpeed': 'max'
                    })
                    flow_id += 1
        
        return flows
    
    def generate_sumocfg(self, network_file: str, route_file: str, output_dir: str) -> str:
        """
        Generate SUMO configuration file (.sumocfg) tying network and routes together.
        
        Args:
            network_file: Path to .net.xml file
            route_file: Path to .rou.xml file
            output_dir: Output directory
            
        Returns:
            Path to generated .sumocfg file
        """
        root = ET.Element('configuration')
        root.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
        root.set('xsi:noNamespaceSchemaLocation', "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
        
        # Input section
        input_elem = ET.SubElement(root, 'input')
        ET.SubElement(input_elem, 'net-file', value=os.path.basename(network_file))
        ET.SubElement(input_elem, 'route-files', value=os.path.basename(route_file))
        
        # Time section
        time_elem = ET.SubElement(root, 'time')
        ET.SubElement(time_elem, 'begin', value='0')
        ET.SubElement(time_elem, 'end', value=str(int(self.sim['horizon'])))
        ET.SubElement(time_elem, 'step-length', value=str(self.sim['step_length']))
        
        # Processing section
        proc_elem = ET.SubElement(root, 'processing')
        ET.SubElement(proc_elem, 'lateral-resolution', value=str(self.sim['lateral_resolution']))
        ET.SubElement(proc_elem, 'collision.action', value='warn')
        ET.SubElement(proc_elem, 'time-to-teleport', value='-1')  # Disable teleporting
        
        # Report section
        report_elem = ET.SubElement(root, 'report')
        ET.SubElement(report_elem, 'verbose', value='true')
        ET.SubElement(report_elem, 'no-step-log', value='false')
        
        # Pretty print and save
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        config_file = os.path.join(output_dir, 'roundabout.sumocfg')
        
        with open(config_file, 'w') as f:
            f.write(xml_str)
        
        print(f"✓ Generated SUMO config: {config_file}")
        
        return config_file


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SUMO route files for roundabout simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate routes from config
  python generate_routes.py --config config/config.yaml --network sumo_configs/baseline/roundabout.net.xml --output sumo_configs/baseline
  
  # Generate with 1.5x demand
  python generate_routes.py --config config/config.yaml --network sumo_configs/baseline/roundabout.net.xml --demand-multiplier 1.5 --output sumo_configs/high_demand
        """
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config.yaml file')
    parser.add_argument('--network', type=str, required=True,
                        help='Path to generated .net.xml file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for route files')
    parser.add_argument('--demand-multiplier', type=float, default=1.0,
                        help='Demand scaling factor (default: 1.0)')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.network):
        print(f"Error: Network file not found: {args.network}", file=sys.stderr)
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Generate routes
    generator = RouteGenerator(config, demand_multiplier=args.demand_multiplier)
    route_file = generator.generate(args.output)
    
    # Generate .sumocfg
    config_file = generator.generate_sumocfg(args.network, route_file, args.output)
    
    print(f"\nRoute generation complete!")
    print(f"Next step: python src/run_simulation.py --sumocfg {config_file} --output results/raw/baseline.csv")


if __name__ == '__main__':
    main()
