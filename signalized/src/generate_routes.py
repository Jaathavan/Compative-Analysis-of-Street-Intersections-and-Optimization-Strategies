"""
Generate SUMO route files for signalized intersection.

Creates route definitions matching demand patterns from roundabout simulation
to ensure fair comparison between intersection types.
"""

import os
import random
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List


class SignalizedRouteGenerator:
    """Generate SUMO routes for signalized intersection with Poisson arrivals."""
    
    def __init__(self, config_path: str, demand_multiplier: float = 1.0):
        """
        Initialize route generator.
        
        Args:
            config_path: Path to config YAML
            demand_multiplier: Scale factor for demand (1.0 = baseline)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.demand_multiplier = demand_multiplier
        self.arrivals = [rate * demand_multiplier for rate in self.config['demand']['arrivals']]
        self.turning_probs = self.config['demand']['turning_probabilities']  # [L, T, R]
        self.vehicle_types = self.config['demand']['vehicle_types']
        
        random.seed(self.config['simulation']['seed'])
    
    def create_vehicle_types(self) -> List[ET.Element]:
        """Create vehicle type definitions."""
        vtypes = []
        
        driver = self.config['driver']
        
        # Passenger car (85%)
        vtype_pass = ET.Element('vType', {
            'id': 'passenger',
            'vClass': 'passenger',
            'accel': str(driver['accel']),
            'decel': str(driver['decel']),
            'sigma': str(driver['sigma']),
            'length': '5.0',
            'width': '1.8',
            'minGap': str(driver['min_gap']),
            'maxSpeed': str(driver['max_speed']),
            'speedFactor': str(driver['speed_factor']),
            'speedDev': str(driver['speed_dev']),
            'tau': str(driver['tau']),
            'carFollowModel': 'IDM',
            'color': '0.8,0.8,0.0'
        })
        vtypes.append(vtype_pass)
        
        # Truck (10%)
        vtype_truck = ET.Element('vType', {
            'id': 'truck',
            'vClass': 'truck',
            'accel': str(driver['accel'] * 0.7),
            'decel': str(driver['decel'] * 0.8),
            'sigma': str(driver['sigma']),
            'length': '12.0',
            'width': '2.5',
            'minGap': str(driver['min_gap'] * 1.5),
            'maxSpeed': str(driver['max_speed'] * 0.85),
            'speedFactor': '0.9',
            'speedDev': '0.05',
            'tau': str(driver['tau'] * 1.2),
            'carFollowModel': 'IDM',
            'color': '0.5,0.5,0.5'
        })
        vtypes.append(vtype_truck)
        
        # Bus (5%)
        vtype_bus = ET.Element('vType', {
            'id': 'bus',
            'vClass': 'bus',
            'accel': str(driver['accel'] * 0.6),
            'decel': str(driver['decel'] * 0.9),
            'sigma': str(driver['sigma']),
            'length': '15.0',
            'width': '2.5',
            'minGap': str(driver['min_gap'] * 1.5),
            'maxSpeed': str(driver['max_speed'] * 0.8),
            'speedFactor': '0.85',
            'speedDev': '0.05',
            'tau': str(driver['tau'] * 1.3),
            'carFollowModel': 'IDM',
            'color': '0.0,0.5,1.0'
        })
        vtypes.append(vtype_bus)
        
        return vtypes
    
    def create_routes(self) -> List[ET.Element]:
        """Create route definitions for all OD pairs."""
        routes = []
        
        # Define all possible routes (4 arms × 3 movements = 12 routes)
        # Format: (route_id, from_arm, to_arm, movement_type)
        route_defs = [
            # North origin
            ('N_R', 'north_in', 'east_out', 'R'),
            ('N_T', 'north_in', 'south_out', 'T'),
            ('N_L', 'north_in', 'west_out', 'L'),
            
            # East origin
            ('E_R', 'east_in', 'south_out', 'R'),
            ('E_T', 'east_in', 'west_out', 'T'),
            ('E_L', 'east_in', 'north_out', 'L'),
            
            # South origin
            ('S_R', 'south_in', 'west_out', 'R'),
            ('S_T', 'south_in', 'north_out', 'T'),
            ('S_L', 'south_in', 'east_out', 'L'),
            
            # West origin
            ('W_R', 'west_in', 'north_out', 'R'),
            ('W_T', 'west_in', 'east_out', 'T'),
            ('W_L', 'west_in', 'south_out', 'L'),
        ]
        
        for route_id, from_edge, to_edge, _ in route_defs:
            route = ET.Element('route', {
                'id': route_id,
                'edges': f"{from_edge} {to_edge}"
            })
            routes.append(route)
        
        return routes
    
    def create_flows(self, simulation_end: float) -> List[ET.Element]:
        """
        Create flow definitions for each arm and movement.
        
        Args:
            simulation_end: Simulation end time (seconds)
        
        Returns:
            List of flow elements
        """
        flows = []
        
        arms = ['north', 'east', 'south', 'west']
        arm_prefixes = ['N', 'E', 'S', 'W']
        
        for i, (arm, prefix) in enumerate(zip(arms, arm_prefixes)):
            arrival_rate = self.arrivals[i]  # veh/s
            arrival_rate_hr = arrival_rate * 3600  # veh/hr
            
            # Split by turning movement
            movements = [
                ('R', self.turning_probs[2]),  # Right
                ('T', self.turning_probs[1]),  # Through
                ('L', self.turning_probs[0]),  # Left
            ]
            
            for movement, prob in movements:
                route_id = f"{prefix}_{movement}"
                flow_rate_hr = arrival_rate_hr * prob
                
                if flow_rate_hr < 1.0:  # Skip very low flows
                    continue
                
                # Create flow for each vehicle type
                vtype_dist = list(self.vehicle_types.items())
                
                for vtype, fraction in vtype_dist:
                    vtype_flow_rate = flow_rate_hr * fraction
                    
                    if vtype_flow_rate < 0.1:  # Skip negligible flows
                        continue
                    
                    flow = ET.Element('flow', {
                        'id': f"flow_{route_id}_{vtype}",
                        'route': route_id,
                        'begin': '0',
                        'end': str(simulation_end),
                        'vehsPerHour': f"{vtype_flow_rate:.2f}",
                        'type': vtype,
                        'departLane': 'best',
                        'departSpeed': 'max'
                    })
                    flows.append(flow)
        
        return flows
    
    def generate(self, output_dir: str) -> str:
        """
        Generate complete route file.
        
        Args:
            output_dir: Directory to save route file
        
        Returns:
            Path to generated .rou.xml file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create root element
        root = ET.Element('routes')
        root.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
        root.set('xsi:noNamespaceSchemaLocation', "http://sumo.dlr.de/xsd/routes_file.xsd")
        
        # Add vehicle types
        for vtype in self.create_vehicle_types():
            root.append(vtype)
        
        # Add routes
        for route in self.create_routes():
            root.append(route)
        
        # Add flows
        simulation_end = self.config['simulation']['horizon']
        for flow in self.create_flows(simulation_end):
            root.append(flow)
        
        # Write to file
        output_path = os.path.join(output_dir, 'intersection.rou.xml')
        self._write_xml(root, output_path)
        
        print(f"✅ Routes generated: {output_path}")
        return output_path
    
    def _write_xml(self, root: ET.Element, output_path: str):
        """Write XML with pretty formatting."""
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_string = reparsed.toprettyxml(indent='  ')
        
        with open(output_path, 'w') as f:
            f.write(pretty_string)


def main():
    """Example usage."""
    config_path = '../config/config.yaml'
    output_dir = '../sumo_configs/test_signal'
    
    generator = SignalizedRouteGenerator(config_path, demand_multiplier=1.0)
    route_file = generator.generate(output_dir)
    
    print(f"\nRoute generation complete!")
    print(f"   Output: {route_file}")


if __name__ == '__main__':
    main()
