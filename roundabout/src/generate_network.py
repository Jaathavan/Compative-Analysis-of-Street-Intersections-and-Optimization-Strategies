"""
generate_network.py - SUMO Roundabout Network Generator
========================================================

Creates a programmatic roundabout network (.net.xml) based on parameters
from config.yaml. The network includes:
- 4 approach roads (N, E, S, W)
- Circular roundabout with configurable diameter and lanes
- Proper connection priorities for right-of-way rules

Usage:
    python generate_network.py --config config/config.yaml --output sumo_configs/scenario_01
    python generate_network.py --diameter 55 --lanes 2 --output sumo_configs/test
"""

import argparse
import math
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess


class RoundaboutNetworkGenerator:
    """
    Generates SUMO network files for roundabout simulations.
    
    Uses SUMO's netconvert tool to build a proper network with correct
    priorities and connections.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize generator with configuration parameters.
        
        Args:
            config: Dictionary from config.yaml with geometry, demand, etc.
        """
        self.config = config
        self.geo = config['geometry']
        self.driver = config['driver']
        
        # Extract key parameters
        self.diameter = self.geo['diameter']
        self.lanes = self.geo['lanes']
        self.lane_width = self.geo['lane_width']
        self.approach_length = self.geo['approach_length']
        self.a_lat_max = self.geo['a_lat_max']
        
        # Compute derived parameters
        self.radius = self.diameter / 2.0
        self.ring_speed_limit = self._compute_ring_speed_limit()
        self.approach_speed = min(self.driver['max_speed'], 16.67)  # ~60 km/h cap
        
        # Positioning
        self.arms = ['N', 'E', 'S', 'W']
        self.angles = [90, 0, 270, 180]  # Degrees for N, E, S, W
        
    def _compute_ring_speed_limit(self) -> float:
        """
        Compute ring speed limit based on lateral acceleration constraint.
        
        From text sim: v_max = sqrt(a_lat_max * R)
        For diameter=45m: v_max = sqrt(1.6 * 22.5) ≈ 6.0 m/s ≈ 21.6 km/h
        
        Returns:
            Speed limit in m/s
        """
        v_max = math.sqrt(self.a_lat_max * self.radius)
        return min(v_max, self.driver['max_speed'])
    
    def generate(self, output_dir: str) -> str:
        """
        Generate complete SUMO network using netconvert.
        
        Args:
            output_dir: Directory path to save .net.xml file
            
        Returns:
            Path to generated .net.xml file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create intermediate files
        nodes_file = os.path.join(output_dir, 'roundabout.nod.xml')
        edges_file = os.path.join(output_dir, 'roundabout.edg.xml')
        types_file = os.path.join(output_dir, 'roundabout.typ.xml')
        net_file = os.path.join(output_dir, 'roundabout.net.xml')
        
        # Generate XML files
        self._write_nodes(nodes_file)
        self._write_edges(edges_file)
        self._write_types(types_file)
        
        # Run netconvert
        netconvert_cmd = [
            'netconvert',
            '--node-files', nodes_file,
            '--edge-files', edges_file,
            '--type-files', types_file,
            '--output-file', net_file,
            '--no-turnarounds',
            '--junctions.corner-detail', '5',
            '--roundabouts.guess', 'true',
            '--default.priority', '10',
            '--lefthand', 'false'
        ]
        
        try:
            result = subprocess.run(netconvert_cmd, capture_output=True, text=True, check=True)
            print(f"✓ Generated network: {net_file}")
            print(f"  - Diameter: {self.diameter}m, Lanes: {self.lanes}")
            print(f"  - Ring speed limit: {self.ring_speed_limit:.2f} m/s ({self.ring_speed_limit*3.6:.1f} km/h)")
            return net_file
        except subprocess.CalledProcessError as e:
            print(f"✗ netconvert failed:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            sys.exit(1)
    
    def _write_nodes(self, filepath: str):
        """Write nodes XML file for netconvert."""
        with open(filepath, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<nodes>\n')
            
            for arm, angle in zip(self.arms, self.angles):
                rad = math.radians(angle)
                
                # Ring entry/exit node
                x = self.radius * math.cos(rad)
                y = self.radius * math.sin(rad)
                f.write(f'    <node id="ring_{arm}" x="{x:.2f}" y="{y:.2f}" type="priority"/>\n')
                
                # Spawn node (traffic source)
                spawn_dist = self.radius + self.approach_length
                spawn_x = spawn_dist * math.cos(rad)
                spawn_y = spawn_dist * math.sin(rad)
                f.write(f'    <node id="spawn_{arm}" x="{spawn_x:.2f}" y="{spawn_y:.2f}" type="priority"/>\n')
                
                # Sink node (traffic destination) - offset perpendicular
                offset_angle = rad - math.radians(15)
                sink_dist = self.radius + 30
                sink_x = sink_dist * math.cos(offset_angle)
                sink_y = sink_dist * math.sin(offset_angle)
                f.write(f'    <node id="sink_{arm}" x="{sink_x:.2f}" y="{sink_y:.2f}" type="priority"/>\n')
            
            f.write('</nodes>\n')
    
    def _write_edges(self, filepath: str):
        """Write edges XML file for netconvert."""
        with open(filepath, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<edges>\n')
            
            # Approach roads
            for arm in self.arms:
                f.write(f'    <edge id="approach_{arm}" from="spawn_{arm}" to="ring_{arm}" ')
                f.write(f'type="approach" numLanes="{self.geo["approach_lanes"]}" ')
                f.write(f'speed="{self.approach_speed:.2f}"/>\n')
            
            # Ring segments with curved shape
            for i, arm in enumerate(self.arms):
                next_arm = self.arms[(i + 1) % 4]
                shape = self._compute_arc_shape(i)
                f.write(f'    <edge id="ring_{arm}" from="ring_{arm}" to="ring_{next_arm}" ')
                f.write(f'type="ring" numLanes="{self.lanes}" ')
                f.write(f'speed="{self.ring_speed_limit:.2f}" ')
                f.write(f'shape="{shape}"/>\n')
            
            # Exit roads
            for arm in self.arms:
                f.write(f'    <edge id="exit_{arm}" from="ring_{arm}" to="sink_{arm}" ')
                f.write(f'type="exit" numLanes="{self.geo["approach_lanes"]}" ')
                f.write(f'speed="{self.approach_speed:.2f}"/>\n')
            
            f.write('</edges>\n')
    
    def _write_types(self, filepath: str):
        """Write edge types XML file for netconvert."""
        with open(filepath, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<types>\n')
            
            # Approach roads (lower priority)
            f.write(f'    <type id="approach" priority="10" numLanes="{self.geo["approach_lanes"]}" ')
            f.write(f'speed="{self.approach_speed:.2f}"/>\n')
            
            # Ring (highest priority)
            f.write(f'    <type id="ring" priority="15" numLanes="{self.lanes}" ')
            f.write(f'speed="{self.ring_speed_limit:.2f}"/>\n')
            
            # Exit roads
            f.write(f'    <type id="exit" priority="10" numLanes="{self.geo["approach_lanes"]}" ')
            f.write(f'speed="{self.approach_speed:.2f}"/>\n')
            
            f.write('</types>\n')
    
    def _compute_arc_shape(self, segment_idx: int) -> str:
        """
        Compute shape string for a ring arc segment.
        
        Args:
            segment_idx: Index of ring segment (0=N→E, 1=E→S, 2=S→W, 3=W→N)
            
        Returns:
            Shape string "x1,y1 x2,y2 x3,y3 ..."
        """
        start_angle = self.angles[segment_idx]
        end_angle = self.angles[(segment_idx + 1) % 4]
        
        # Normalize for interpolation
        if end_angle < start_angle:
            end_angle += 360
        
        # Generate arc points
        num_points = 12
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            angle = start_angle + t * (end_angle - start_angle)
            rad = math.radians(angle)
            x = self.radius * math.cos(rad)
            y = self.radius * math.sin(rad)
            points.append(f'{x:.2f},{y:.2f}')
        
        return ' '.join(points)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SUMO roundabout network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from config file
  python generate_network.py --config config/config.yaml --output sumo_configs/baseline
  
  # Override specific parameters
  python generate_network.py --config config/config.yaml --diameter 55 --lanes 2 --output sumo_configs/large_2lane
        """
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config.yaml file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for generated files')
    
    # Optional parameter overrides
    parser.add_argument('--diameter', type=float,
                        help='Roundabout diameter (m) - overrides config')
    parser.add_argument('--lanes', type=int, choices=[1, 2],
                        help='Number of circulating lanes - overrides config')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Apply overrides
    if args.diameter:
        config['geometry']['diameter'] = args.diameter
        print(f"Override: diameter = {args.diameter}m")
    
    if args.lanes:
        config['geometry']['lanes'] = args.lanes
        print(f"Override: lanes = {args.lanes}")
    
    # Generate network
    generator = RoundaboutNetworkGenerator(config)
    net_file = generator.generate(args.output)
    
    print(f"\nNetwork generation complete!")
    print(f"Next step: python src/generate_routes.py --network {net_file} --config {args.config}")


if __name__ == '__main__':
    main()
