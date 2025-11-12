"""
Generate SUMO network files for 4-way signalized intersection.

Creates:
- .nod.xml: Node definitions
- .edg.xml: Edge definitions
- .typ.xml: Edge type definitions
- .con.xml: Connection definitions (optional)
- .tll.xml: Traffic light logic
- .net.xml: Final compiled network
"""

import os
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Tuple


class SignalizedNetworkGenerator:
    """Generate SUMO network for 4-way signalized intersection."""
    
    def __init__(self, config_path: str, output_dir: str):
        """
        Initialize network generator.
        
        Args:
            config_path: Path to config YAML
            output_dir: Directory for output files
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract geometry parameters
        geom = self.config['geometry']
        self.approach_length = geom['approach_length']
        self.approach_lanes = geom['approach_lanes']
        self.lane_width = geom['lane_width']
        self.approach_speed = geom['approach_speed']
        self.intersection_speed = geom['intersection_speed']
        self.left_turn_lane = geom.get('left_turn_lane', True)
        
    def generate_nodes(self) -> str:
        """
        Generate .nod.xml file defining intersection nodes.
        
        Returns:
            Path to generated .nod.xml file
        """
        root = ET.Element('nodes')
        
        # Central intersection node
        ET.SubElement(root, 'node', {
            'id': 'center',
            'x': '0.0',
            'y': '0.0',
            'type': 'traffic_light',
            'tl': 'center'  # Traffic light ID
        })
        
        # Approach endpoints (N, E, S, W)
        endpoints = {
            'north_end': (0, self.approach_length),
            'east_end': (self.approach_length, 0),
            'south_end': (0, -self.approach_length),
            'west_end': (-self.approach_length, 0)
        }
        
        for node_id, (x, y) in endpoints.items():
            ET.SubElement(root, 'node', {
                'id': node_id,
                'x': str(x),
                'y': str(y),
                'type': 'priority'
            })
        
        # Write file
        output_path = os.path.join(self.output_dir, 'intersection.nod.xml')
        self._write_xml(root, output_path)
        return output_path
    
    def generate_edges(self) -> str:
        """
        Generate .edg.xml file defining road segments.
        
        Returns:
            Path to generated .edg.xml file
        """
        root = ET.Element('edges')
        
        # Edge type with multiple lanes
        speed_str = str(self.approach_speed)
        lanes_str = str(self.approach_lanes)
        
        # Incoming edges (approach → center)
        incoming = [
            ('north_in', 'north_end', 'center'),
            ('east_in', 'east_end', 'center'),
            ('south_in', 'south_end', 'center'),
            ('west_in', 'west_end', 'center')
        ]
        
        for edge_id, from_node, to_node in incoming:
            ET.SubElement(root, 'edge', {
                'id': edge_id,
                'from': from_node,
                'to': to_node,
                'numLanes': lanes_str,
                'speed': speed_str,
                'priority': '3'
            })
        
        # Outgoing edges (center → approach)
        outgoing = [
            ('north_out', 'center', 'north_end'),
            ('east_out', 'center', 'east_end'),
            ('south_out', 'center', 'south_end'),
            ('west_out', 'center', 'west_end')
        ]
        
        for edge_id, from_node, to_node in outgoing:
            ET.SubElement(root, 'edge', {
                'id': edge_id,
                'from': from_node,
                'to': to_node,
                'numLanes': lanes_str,
                'speed': speed_str,
                'priority': '3'
            })
        
        output_path = os.path.join(self.output_dir, 'intersection.edg.xml')
        self._write_xml(root, output_path)
        return output_path
    
    def generate_connections(self) -> str:
        """
        Generate .con.xml file defining allowed movements.
        
        Returns:
            Path to generated .con.xml file
        """
        root = ET.Element('connections')
        
        # Define turning movements
        # Format: (from_edge, to_edge, from_lane, to_lane, direction)
        # Lane 0 = rightmost, lane 1 = leftmost (for 2-lane approach)
        
        connections = []
        
        if self.approach_lanes == 1:
            # Single lane per approach - all movements from lane 0
            connections = [
                # North approach
                ('north_in', 'east_out', 0, 0, 'r'),   # Right turn
                ('north_in', 'south_out', 0, 0, 's'),  # Through
                ('north_in', 'west_out', 0, 0, 'l'),   # Left turn
                
                # East approach
                ('east_in', 'south_out', 0, 0, 'r'),
                ('east_in', 'west_out', 0, 0, 's'),
                ('east_in', 'north_out', 0, 0, 'l'),
                
                # South approach
                ('south_in', 'west_out', 0, 0, 'r'),
                ('south_in', 'north_out', 0, 0, 's'),
                ('south_in', 'east_out', 0, 0, 'l'),
                
                # West approach
                ('west_in', 'north_out', 0, 0, 'r'),
                ('west_in', 'east_out', 0, 0, 's'),
                ('west_in', 'south_out', 0, 0, 'l'),
            ]
        else:  # 2+ lanes
            # Lane 0 (rightmost): Through + Right
            # Lane 1 (leftmost): Left + Through
            connections = [
                # North approach
                ('north_in', 'east_out', 0, 0, 'r'),   # Right from right lane
                ('north_in', 'south_out', 0, 0, 's'),  # Through from right lane
                ('north_in', 'south_out', 1, 1, 's'),  # Through from left lane
                ('north_in', 'west_out', 1, 0, 'l'),   # Left from left lane
                
                # East approach
                ('east_in', 'south_out', 0, 0, 'r'),
                ('east_in', 'west_out', 0, 0, 's'),
                ('east_in', 'west_out', 1, 1, 's'),
                ('east_in', 'north_out', 1, 0, 'l'),
                
                # South approach
                ('south_in', 'west_out', 0, 0, 'r'),
                ('south_in', 'north_out', 0, 0, 's'),
                ('south_in', 'north_out', 1, 1, 's'),
                ('south_in', 'east_out', 1, 0, 'l'),
                
                # West approach
                ('west_in', 'north_out', 0, 0, 'r'),
                ('west_in', 'east_out', 0, 0, 's'),
                ('west_in', 'east_out', 1, 1, 's'),
                ('west_in', 'south_out', 1, 0, 'l'),
            ]
        
        for from_edge, to_edge, from_lane, to_lane, direction in connections:
            ET.SubElement(root, 'connection', {
                'from': from_edge,
                'to': to_edge,
                'fromLane': str(from_lane),
                'toLane': str(to_lane),
                'dir': direction
            })
        
        output_path = os.path.join(self.output_dir, 'intersection.con.xml')
        self._write_xml(root, output_path)
        return output_path
    
    def generate_traffic_light_logic(
        self, 
        cycle_length: float = None,
        green_times: List[float] = None
    ) -> str:
        """
        Generate .tll.xml file defining traffic light logic.
        
        Args:
            cycle_length: Total cycle length (seconds). If None, uses default 4-phase.
            green_times: List of green times [g1, g2, g3, g4] for each phase.
                        If None, uses equal distribution.
        
        Returns:
            Path to generated .tll.xml file
        """
        root = ET.Element('tlLogics')
        
        # Get timing parameters
        yellow = self.config['signal']['webster']['yellow_time']
        all_red = self.config['signal']['webster']['all_red_time']
        
        if cycle_length is None or green_times is None:
            # Default: equal green times with 90s cycle
            cycle_length = 90.0
            clearance_time = yellow + all_red
            total_clearance = 4 * clearance_time
            available_green = cycle_length - total_clearance
            green_times = [available_green / 4] * 4
        
        # Create traffic light logic
        tl_logic = ET.SubElement(root, 'tlLogic', {
            'id': 'center',
            'type': 'static',
            'programID': '0',
            'offset': '0'
        })
        
        # State encoding (12 connections: 3 per approach × 4 approaches)
        # State string: RRRRRRRRRRRR (all red) → customize per phase
        # Each character: r=red, y=yellow, g=green, G=green (priority)
        
        # Phase states (4-phase signal)
        # For 2-lane approach: [Right_lane0, Through_lane0, Through_lane1, Left_lane1]
        # Repeated for [North, East, South, West]
        
        if self.approach_lanes == 1:
            # Single lane: [RTL] × 4 = 12 positions
            phases = [
                # Phase 1: NS Left turns (protected)
                ('rrGrrrGrrrr', green_times[0]),
                ('rryrrryrrry', yellow),
                ('rrrrrrrrrrrr', all_red),
                
                # Phase 2: NS Through + Right
                ('GGrGGGrGGGr', green_times[1]),
                ('yyryyyryyyr', yellow),
                ('rrrrrrrrrrrr', all_red),
                
                # Phase 3: EW Left turns (protected)
                ('rrrGrrrGrrr', green_times[2]),
                ('rrryrrryrry', yellow),
                ('rrrrrrrrrrrr', all_red),
                
                # Phase 4: EW Through + Right
                ('rrrGGGrGGGG', green_times[3]),
                ('rrryyyryyy', yellow),
                ('rrrrrrrrrrrr', all_red),
            ]
        else:
            # 2 lanes: [R, T0, T1, L] × 4 = 16 positions
            # Order: N_R, N_T0, N_T1, N_L, E_R, E_T0, E_T1, E_L, S_R, S_T0, S_T1, S_L, W_R, W_T0, W_T1, W_L
            phases = [
                # Phase 1: NS Left turns (protected)
                ('rrrGrrrrrrrGrrrr', green_times[0]),
                ('rrryrrrrrrryrrrr', yellow),
                ('rrrrrrrrrrrrrrrr', all_red),
                
                # Phase 2: NS Through + Right
                ('GGGrGGGGGGGrGGGG', green_times[1]),
                ('yyyryyyyyyyryyyy', yellow),
                ('rrrrrrrrrrrrrrrr', all_red),
                
                # Phase 3: EW Left turns (protected)
                ('rrrrrrrGrrrrrrrG', green_times[2]),
                ('rrrrrrryrrrrrrry', yellow),
                ('rrrrrrrrrrrrrrrr', all_red),
                
                # Phase 4: EW Through + Right
                ('rrrrGGGrGGGGGGGr', green_times[3]),
                ('rrrryyyryyyyyyyr', yellow),
                ('rrrrrrrrrrrrrrrr', all_red),
            ]
        
        # Add phases to logic
        for state, duration in phases:
            ET.SubElement(tl_logic, 'phase', {
                'duration': str(duration),
                'state': state
            })
        
        output_path = os.path.join(self.output_dir, 'intersection.tll.xml')
        self._write_xml(root, output_path)
        return output_path
    
    def compile_network(self, use_webster: bool = False, webster_result: Dict = None) -> str:
        """
        Compile complete SUMO network using netconvert.
        
        Args:
            use_webster: Whether to use Webster-optimized timing
            webster_result: Webster optimization result (if use_webster=True)
        
        Returns:
            Path to generated .net.xml file
        """
        print("\n🏗️  Generating SUMO network for signalized intersection...")
        
        # Generate component files
        nod_file = self.generate_nodes()
        edg_file = self.generate_edges()
        con_file = self.generate_connections()
        
        # Generate traffic light logic
        if use_webster and webster_result:
            print(f"   Using Webster-optimized timing (C={webster_result['cycle_length']:.1f}s)")
            tll_file = self.generate_traffic_light_logic(
                cycle_length=webster_result['cycle_length'],
                green_times=webster_result['green_times']
            )
        else:
            print("   Using default timing (equal green splits)")
            tll_file = self.generate_traffic_light_logic()
        
        # Compile network with netconvert
        net_file = os.path.join(self.output_dir, 'intersection.net.xml')
        
        cmd = [
            'netconvert',
            '--node-files', nod_file,
            '--edge-files', edg_file,
            '--connection-files', con_file,
            '--tllogic-files', tll_file,
            '--output-file', net_file,
            '--no-turnarounds',
            '--junctions.corner-detail', '5',
            '--junctions.limit-turn-speed', '5.5',
            '--default.junctions.keep-clear', 'true',
            '--default.junctions.radius', '4'
        ]
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ netconvert failed:")
            print(result.stderr)
            raise RuntimeError("Network compilation failed")
        
        print(f"✅ Network compiled: {net_file}")
        return net_file
    
    def _write_xml(self, root: ET.Element, output_path: str):
        """Write XML element tree to file with pretty formatting."""
        # Convert to string
        rough_string = ET.tostring(root, encoding='unicode')
        
        # Pretty print
        reparsed = minidom.parseString(rough_string)
        pretty_string = reparsed.toprettyxml(indent='  ')
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(pretty_string)


def main():
    """Example usage."""
    import sys
    
    config_path = '../config/config.yaml'
    output_dir = '../sumo_configs/test_signal'
    
    generator = SignalizedNetworkGenerator(config_path, output_dir)
    
    # Option 1: Generate with default timing
    print("\n" + "="*70)
    print("OPTION 1: Default Equal Green Split")
    print("="*70)
    net_file = generator.compile_network(use_webster=False)
    
    # Option 2: Generate with Webster-optimized timing
    print("\n" + "="*70)
    print("OPTION 2: Webster-Optimized Timing")
    print("="*70)
    
    from webster_method import WebsterSignalOptimizer
    
    optimizer = WebsterSignalOptimizer(config_path=config_path)
    webster_result = optimizer.optimize(demand_multiplier=1.0, verbose=True)
    
    if webster_result['cycle_length']:
        output_dir2 = '../sumo_configs/webster_signal'
        generator2 = SignalizedNetworkGenerator(config_path, output_dir2)
        net_file2 = generator2.compile_network(
            use_webster=True,
            webster_result=webster_result
        )
        print(f"\n✅ Webster-optimized network: {net_file2}")


if __name__ == '__main__':
    main()
