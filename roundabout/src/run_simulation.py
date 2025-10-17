#!/usr/bin/env python3
"""
run_simulation.py - SUMO Simulation Runner with TraCI Metric Collection
========================================================================

Executes SUMO simulations via TraCI API and collects performance metrics
matching the windowed reporting structure of Roundabout.py.

Features:
- 5-minute windowed metrics (arrivals, exits, delays, queues)
- Hourly aggregate statistics
- SUMO-specific metrics (emissions, fuel, time loss)
- Optional GUI mode for visualization
- Real-time metric collection via TraCI

Usage:
    # Headless mode (production)
    python run_simulation.py --sumocfg sumo_configs/baseline/roundabout.sumocfg --output results/raw/baseline.csv
    
    # GUI mode (visualization/debugging)
    python run_simulation.py --sumocfg sumo_configs/baseline/roundabout.sumocfg --gui --output results/raw/baseline.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
import pandas as pd
import yaml
import traci
import sumolib


@dataclass
class WindowMetrics:
    """Metrics collected during a 5-minute window."""
    window_id: int
    start_time: float
    end_time: float
    
    # Arrivals and exits
    arrivals: int = 0
    exits: int = 0
    
    # Throughput
    throughput_vph: float = 0.0  # Vehicles per hour
    
    # Delays
    avg_delay: float = 0.0  # Mean entry delay (s)
    p95_delay: float = 0.0  # 95th percentile delay (s)
    delays: List[float] = field(default_factory=list)  # Raw delays for percentile calc
    
    # Queue lengths (per arm)
    max_queue_N: int = 0
    max_queue_E: int = 0
    max_queue_S: int = 0
    max_queue_W: int = 0
    
    # Average speeds
    avg_speed_ring: float = 0.0  # Average speed in roundabout (m/s)
    
    # Stops
    avg_stops_per_vehicle: float = 0.0
    
    # SUMO-specific
    total_co2: float = 0.0  # mg
    total_fuel: float = 0.0  # ml
    total_waiting_time: float = 0.0  # s
    total_time_loss: float = 0.0  # s


@dataclass
class AggregateMetrics:
    """Aggregate metrics over full simulation."""
    total_arrivals: int = 0
    total_exits: int = 0
    
    mean_delay: float = 0.0
    p95_delay: float = 0.0
    all_delays: List[float] = field(default_factory=list)
    
    max_queue_N: int = 0
    max_queue_E: int = 0
    max_queue_S: int = 0
    max_queue_W: int = 0
    
    throughput_vph: float = 0.0
    avg_travel_time: float = 0.0
    avg_time_loss: float = 0.0
    
    total_co2: float = 0.0
    total_fuel: float = 0.0


class SUMOSimulation:
    """
    Manages SUMO simulation execution and metric collection via TraCI.
    """
    
    def __init__(self, sumocfg: str, config_yaml: str, gui: bool = False):
        """
        Initialize simulation manager.
        
        Args:
            sumocfg: Path to SUMO configuration file
            config_yaml: Path to config.yaml for parameters
            gui: Launch SUMO-GUI if True, otherwise run headless
        """
        self.sumocfg = sumocfg
        self.gui = gui
        
        # Load configuration
        with open(config_yaml, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sim_config = self.config['simulation']
        self.horizon = self.sim_config['horizon']
        self.report_every = self.sim_config['report_every']
        self.step_length = self.sim_config['step_length']
        
        # Arm definitions
        self.arms = ['N', 'E', 'S', 'W']
        self.approach_edges = {arm: f'approach_{arm}' for arm in self.arms}
        self.ring_edges = [f'ring_{arm}' for arm in self.arms]
        
        # Tracking structures
        self.vehicle_data = {}  # vid -> {enter_time, queue_start_time, stops, ...}
        self.window_metrics: List[WindowMetrics] = []
        self.aggregate = AggregateMetrics()
        
        # Current window tracking
        self.current_window = None
        self.window_id = 0
        
        # TraCI connection
        self.traci_conn = None
    
    def run(self, output_file: str):
        """
        Execute complete simulation and save results.
        
        Args:
            output_file: Path to save CSV results
        """
        print("=" * 70)
        print("SUMO Roundabout Simulation")
        print("=" * 70)
        print(f"Configuration: {self.sumocfg}")
        print(f"Horizon: {self.horizon}s ({self.horizon/60:.0f} min)")
        print(f"Report interval: {self.report_every}s ({self.report_every/60:.0f} min)")
        print(f"GUI mode: {'Yes' if self.gui else 'No'}")
        print("=" * 70)
        
        # Start SUMO
        self._start_sumo()
        
        # Initialize first window
        self._start_window(0)
        
        try:
            # Main simulation loop
            step = 0
            while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.horizon:
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                
                # Collect metrics at this step
                self._collect_step_metrics(current_time)
                
                # Check if window boundary reached
                if current_time >= (self.window_id + 1) * self.report_every:
                    self._end_window(current_time)
                    self._start_window(current_time)
                
                step += 1
                
                # Progress indicator every 300 steps (~60s sim time)
                if step % 300 == 0:
                    print(f"  Progress: {current_time:.0f}s / {self.horizon:.0f}s ({current_time/self.horizon*100:.1f}%)")
            
            # Finalize last window
            if self.current_window:
                self._end_window(self.horizon)
            
            # Compute aggregate statistics
            self._compute_aggregates()
            
        finally:
            traci.close()
        
        # Save results
        self._save_results(output_file)
        
        # Print summary
        self._print_summary()
    
    def _start_sumo(self):
        """Start SUMO with TraCI."""
        sumo_binary = sumolib.checkBinary('sumo-gui' if self.gui else 'sumo')
        
        traci_port = 8813
        sumo_cmd = [
            sumo_binary,
            '-c', self.sumocfg,
            '--step-length', str(self.step_length),
            '--no-warnings', 'true',
            '--duration-log.statistics', 'true',
            '--device.emissions.probability', '1.0',  # Enable emission devices
            '--tripinfo-output.write-unfinished', 'true'
        ]
        
        if self.gui:
            sumo_cmd.extend(['--start', 'true', '--quit-on-end', 'true'])
        
        traci.start(sumo_cmd, port=traci_port)
        print("✓ SUMO started successfully")
    
    def _start_window(self, start_time: float):
        """Initialize a new reporting window."""
        self.current_window = WindowMetrics(
            window_id=self.window_id,
            start_time=start_time,
            end_time=start_time + self.report_every
        )
    
    def _end_window(self, end_time: float):
        """Finalize current window and compute statistics."""
        if not self.current_window:
            return
        
        self.current_window.end_time = end_time
        window_duration = end_time - self.current_window.start_time
        
        # Avoid division by zero
        if window_duration <= 0:
            window_duration = 1.0  # Fallback to 1 second to avoid crash
            
        # Compute throughput (veh/hr)
        self.current_window.throughput_vph = (self.current_window.exits * 3600.0) / window_duration
        
        # Compute delay statistics
        if self.current_window.delays:
            self.current_window.avg_delay = sum(self.current_window.delays) / len(self.current_window.delays)
            sorted_delays = sorted(self.current_window.delays)
            p95_idx = int(len(sorted_delays) * 0.95)
            self.current_window.p95_delay = sorted_delays[p95_idx] if p95_idx < len(sorted_delays) else self.current_window.avg_delay
        
        # Store window
        self.window_metrics.append(self.current_window)
        self.window_id += 1
        
        # Print window summary
        self._print_window()
    
    def _collect_step_metrics(self, current_time: float):
        """
        Collect metrics at current simulation step.
        
        Args:
            current_time: Current simulation time (s)
        """
        # Track new arrivals (vehicles entering simulation)
        departed_ids = traci.simulation.getDepartedIDList()
        for vid in departed_ids:
            self.vehicle_data[vid] = {
                'arrive_time': current_time,
                'queue_start_time': None,
                'enter_ring_time': None,
                'exit_time': None,
                'stops': 0,
                'co2': 0.0,
                'fuel': 0.0,
                'waiting_time': 0.0
            }
            self.current_window.arrivals += 1
            self.aggregate.total_arrivals += 1
        
        # Track vehicles reaching roundabout entry (start of queue)
        for vid in traci.vehicle.getIDList():
            if vid not in self.vehicle_data:
                continue
            
            vdata = self.vehicle_data[vid]
            edge_id = traci.vehicle.getRoadID(vid)
            
            # Check if on approach and queuing
            if edge_id in self.approach_edges.values() and vdata['queue_start_time'] is None:
                speed = traci.vehicle.getSpeed(vid)
                if speed < 0.5:  # Consider stopped if < 0.5 m/s
                    vdata['queue_start_time'] = current_time
            
            # Check if entered ring
            if edge_id in self.ring_edges and vdata['enter_ring_time'] is None:
                vdata['enter_ring_time'] = current_time
                
                # Compute entry delay if queued
                if vdata['queue_start_time']:
                    delay = current_time - vdata['queue_start_time']
                    self.current_window.delays.append(delay)
                    self.aggregate.all_delays.append(delay)
            
            # Accumulate emissions
            vdata['co2'] += traci.vehicle.getCO2Emission(vid) * self.step_length / 1000.0  # mg
            vdata['fuel'] += traci.vehicle.getFuelConsumption(vid) * self.step_length / 1000.0  # ml
            vdata['waiting_time'] += traci.vehicle.getWaitingTime(vid)
        
        # Track exits
        arrived_ids = traci.simulation.getArrivedIDList()
        for vid in arrived_ids:
            if vid in self.vehicle_data:
                self.vehicle_data[vid]['exit_time'] = current_time
                self.current_window.exits += 1
                self.aggregate.total_exits += 1
                
                # Accumulate window emissions
                vdata = self.vehicle_data[vid]
                self.current_window.total_co2 += vdata['co2']
                self.current_window.total_fuel += vdata['fuel']
                self.current_window.total_waiting_time += vdata['waiting_time']
        
        # Track queue lengths per arm
        for arm in self.arms:
            edge = self.approach_edges[arm]
            queue_length = traci.edge.getLastStepHaltingNumber(edge)
            
            # Update window max
            setattr(self.current_window, f'max_queue_{arm}', 
                    max(getattr(self.current_window, f'max_queue_{arm}'), queue_length))
            
            # Update aggregate max
            setattr(self.aggregate, f'max_queue_{arm}',
                    max(getattr(self.aggregate, f'max_queue_{arm}'), queue_length))
        
        # Track ring speeds
        ring_speeds = []
        for vid in traci.vehicle.getIDList():
            edge = traci.vehicle.getRoadID(vid)
            if edge in self.ring_edges:
                ring_speeds.append(traci.vehicle.getSpeed(vid))
        
        if ring_speeds:
            self.current_window.avg_speed_ring = sum(ring_speeds) / len(ring_speeds)
    
    def _compute_aggregates(self):
        """Compute aggregate statistics over full simulation."""
        # Overall delays
        if self.aggregate.all_delays:
            self.aggregate.mean_delay = sum(self.aggregate.all_delays) / len(self.aggregate.all_delays)
            sorted_delays = sorted(self.aggregate.all_delays)
            p95_idx = int(len(sorted_delays) * 0.95)
            self.aggregate.p95_delay = sorted_delays[p95_idx] if p95_idx < len(sorted_delays) else self.aggregate.mean_delay
        
        # Overall throughput
        self.aggregate.throughput_vph = (self.aggregate.total_exits * 3600.0) / self.horizon
        
        # Travel times
        travel_times = []
        for vid, vdata in self.vehicle_data.items():
            if vdata['exit_time']:
                tt = vdata['exit_time'] - vdata['arrive_time']
                travel_times.append(tt)
        
        if travel_times:
            self.aggregate.avg_travel_time = sum(travel_times) / len(travel_times)
        
        # Aggregate emissions
        for vdata in self.vehicle_data.values():
            self.aggregate.total_co2 += vdata['co2']
            self.aggregate.total_fuel += vdata['fuel']
    
    def _print_window(self):
        """Print current window summary (matching Roundabout.py format)."""
        w = self.current_window
        print(f"[{self._format_time(w.end_time)}] "
              f"arrivals={w.arrivals}  "
              f"exits={w.exits}  "
              f"throughput={w.throughput_vph:.0f} veh/hr  "
              f"avg_delay={w.avg_delay:.1f}s  "
              f"p95={w.p95_delay:.1f}s  "
              f"max_q=[{w.max_queue_N}, {w.max_queue_E}, {w.max_queue_S}, {w.max_queue_W}]")
    
    def _print_summary(self):
        """Print aggregate summary (matching Roundabout.py format)."""
        print("\n" + "=" * 70)
        print("HOURLY SUMMARY")
        print("=" * 70)
        print(f"arrivals_total:     {self.aggregate.total_arrivals}")
        print(f"exits_total:        {self.aggregate.total_exits}")
        print(f"throughput:         {self.aggregate.throughput_vph:.0f} veh/hr")
        print(f"mean_delay:         {self.aggregate.mean_delay:.1f} s/veh")
        print(f"p95_delay:          {self.aggregate.p95_delay:.1f} s")
        print(f"max_queue_by_arm:   [{self.aggregate.max_queue_N}, {self.aggregate.max_queue_E}, {self.aggregate.max_queue_S}, {self.aggregate.max_queue_W}]")
        print(f"avg_travel_time:    {self.aggregate.avg_travel_time:.1f} s")
        print(f"\nSUMO-Specific Metrics:")
        print(f"total_CO2:          {self.aggregate.total_co2:.2f} mg")
        print(f"total_fuel:         {self.aggregate.total_fuel:.2f} ml")
        print("=" * 70)
    
    def _save_results(self, output_file: str):
        """
        Save simulation results to CSV files.
        
        Args:
            output_file: Base path for output files
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save window metrics
        window_data = []
        for w in self.window_metrics:
            window_data.append({
                'window_id': w.window_id,
                'start_time': w.start_time,
                'end_time': w.end_time,
                'arrivals': w.arrivals,
                'exits': w.exits,
                'throughput_vph': w.throughput_vph,
                'avg_delay': w.avg_delay,
                'p95_delay': w.p95_delay,
                'max_queue_N': w.max_queue_N,
                'max_queue_E': w.max_queue_E,
                'max_queue_S': w.max_queue_S,
                'max_queue_W': w.max_queue_W,
                'avg_speed_ring': w.avg_speed_ring,
                'total_co2': w.total_co2,
                'total_fuel': w.total_fuel,
                'total_waiting_time': w.total_waiting_time
            })
        
        df_windows = pd.DataFrame(window_data)
        df_windows.to_csv(output_file, index=False)
        print(f"\n✓ Window metrics saved to: {output_file}")
        
        # Save aggregate as single-row CSV
        agg_file = output_file.replace('.csv', '_aggregate.csv')
        df_agg = pd.DataFrame([{
            'total_arrivals': self.aggregate.total_arrivals,
            'total_exits': self.aggregate.total_exits,
            'mean_delay': self.aggregate.mean_delay,
            'p95_delay': self.aggregate.p95_delay,
            'max_queue_N': self.aggregate.max_queue_N,
            'max_queue_E': self.aggregate.max_queue_E,
            'max_queue_S': self.aggregate.max_queue_S,
            'max_queue_W': self.aggregate.max_queue_W,
            'throughput_vph': self.aggregate.throughput_vph,
            'avg_travel_time': self.aggregate.avg_travel_time,
            'total_co2': self.aggregate.total_co2,
            'total_fuel': self.aggregate.total_fuel
        }])
        df_agg.to_csv(agg_file, index=False)
        print(f"✓ Aggregate metrics saved to: {agg_file}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description='Run SUMO roundabout simulation with TraCI metric collection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--sumocfg', type=str, required=True,
                        help='Path to SUMO configuration file (.sumocfg)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config.yaml file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path for results')
    parser.add_argument('--gui', action='store_true',
                        help='Launch SUMO-GUI for visualization')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.sumocfg):
        print(f"Error: SUMO config not found: {args.sumocfg}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    # Run simulation
    sim = SUMOSimulation(args.sumocfg, args.config, gui=args.gui)
    sim.run(args.output)


if __name__ == '__main__':
    main()
