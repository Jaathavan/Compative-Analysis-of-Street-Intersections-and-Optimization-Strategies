#!/usr/bin/env python3
"""
Signalized.py - Text-Based Signalized Intersection Microsimulation
===================================================================

A Python microsimulation of a 4-way signalized intersection with:
- Webster's Method for optimal fixed-time signal control
- IDM car-following model
- Multi-lane support (1-3 lanes per approach)
- Poisson arrival process
- Realistic queuing and spillback

Mirrors the structure of Roundabout.py for easy comparison.

Usage:
    python Signalized.py --lanes 2 --arrival 0.18 0.12 0.20 0.15
    python Signalized.py --diameter 45 --cycle-length 90 --horizon 3600
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Deque, Dict, Optional, Tuple
from collections import deque
import math
import random
import argparse
import statistics as stats

# ============================================================================
# HELPERS
# ============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value between bounds."""
    return lo if x < lo else hi if x > hi else x

def mmss(t: float) -> str:
    """Format time as MM:SS."""
    m = int(t // 60)
    s = int(t - m * 60)
    return f"{m:02d}:{s:02d}"

# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class DriverParams:
    """IDM car-following parameters."""
    a_max: float = 1.5          # Maximum acceleration (m/s²)
    b_comf: float = 2.0         # Comfortable deceleration (m/s²)
    delta: int = 4              # Acceleration exponent
    s0: float = 2.0             # Minimum gap (m)
    T: float = 1.2              # Desired time headway (s)
    tau: float = 0.8            # Reaction delay (s)
    v0: float = 13.9            # Desired speed (m/s) ≈ 50 km/h

@dataclass
class SignalParams:
    """Signal timing parameters (Webster's Method)."""
    saturation_flow: float = 1800.0    # Saturation flow rate (veh/hr/lane)
    lost_time_startup: float = 2.0     # Startup lost time per phase (s)
    lost_time_clearance: float = 1.0   # Clearance (yellow+all-red) time (s)
    yellow_time: float = 3.0           # Yellow light duration (s)
    all_red_time: float = 1.0          # All-red clearance (s)
    min_green: float = 7.0             # Minimum green time (s)
    max_green: float = 60.0            # Maximum green time (s)
    min_cycle: float = 60.0            # Minimum cycle length (s)
    max_cycle: float = 180.0           # Maximum cycle length (s)

@dataclass
class Geometry:
    """Intersection geometry."""
    approach_length: float = 200.0     # Distance from spawning to stop line (m)
    lanes: int = 2                     # Lanes per approach (1-3)
    lane_width: float = 3.5            # Lane width (m)
    intersection_size: float = 20.0    # Distance across intersection (m)

@dataclass
class Demand:
    """Traffic demand parameters."""
    arrival: Tuple[float, float, float, float] = (0.18, 0.12, 0.20, 0.15)  # veh/s per arm [N,E,S,W]
    turning: Tuple[float, float, float] = (0.25, 0.55, 0.20)  # [L, T, R] probabilities

@dataclass
class SimConfig:
    """Complete simulation configuration."""
    seed: int = 42
    horizon: float = 3600.0            # Simulation duration (s)
    report_every: float = 300.0        # Reporting interval (s)
    dt: float = 0.2                    # Time step (s)
    
    driver: DriverParams = field(default_factory=DriverParams)
    signal: SignalParams = field(default_factory=SignalParams)
    geo: Geometry = field(default_factory=Geometry)
    dem: Demand = field(default_factory=Demand)
    
    # Webster's Method auto-optimization
    use_webster: bool = True
    cycle_length: Optional[float] = None  # If None, compute optimal via Webster

# ============================================================================
# VEHICLE
# ============================================================================

@dataclass
class Vehicle:
    """Individual vehicle state."""
    id: int
    arm: int                           # Origin arm (0=N, 1=E, 2=S, 3=W)
    turn: str                          # Movement: 'L', 'T', 'R'
    lane: int                          # Lane index (0=leftmost)
    
    pos: float = 0.0                   # Position along approach (0 = stop line)
    speed: float = 0.0                 # Current speed (m/s)
    
    t_arrival: float = 0.0             # Time of arrival to system
    t_departure: Optional[float] = None  # Time of departure (None if still in system)
    
    in_queue: bool = True              # Whether currently queued
    stops: int = 0                     # Number of stops
    
    crossed: bool = False              # Whether crossed stop line

# ============================================================================
# WEBSTER'S METHOD
# ============================================================================

class WebsterOptimizer:
    """
    Optimal signal timing via Webster's Method (1958).
    
    Reference: Webster, F. V. (1958). "Traffic Signal Settings."
    Road Research Technical Paper No. 39.
    """
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.signal = config.signal
        self.demand = config.dem
        
        # 4-phase operation: NS-Left, NS-Through, EW-Left, EW-Through
        # Phases: [NS-L, NS-T, EW-L, EW-T]
        self.num_phases = 4
    
    def compute_flow_ratios(self) -> List[float]:
        """
        Compute critical flow ratio (y_i) for each phase.
        
        y_i = (arrival rate) / (saturation flow rate)
        """
        arrivals = list(self.demand.arrival)  # [N, E, S, W] in veh/s
        turning_probs = list(self.demand.turning)  # [L, T, R]
        
        # Convert to veh/hr
        lambda_N, lambda_E, lambda_S, lambda_W = [arr * 3600 for arr in arrivals]
        p_L, p_T, p_R = turning_probs
        
        # Phase flows (assuming lanes accommodate movements)
        # Phase 0: NS-Left (North+South left turns)
        q_NS_L = (lambda_N + lambda_S) * p_L
        
        # Phase 1: NS-Through (North+South through+right)
        q_NS_T = (lambda_N + lambda_S) * (p_T + p_R)
        
        # Phase 2: EW-Left
        q_EW_L = (lambda_E + lambda_W) * p_L
        
        # Phase 3: EW-Through
        q_EW_T = (lambda_E + lambda_W) * (p_T + p_R)
        
        # Saturation flow per phase (depends on lane count)
        lanes = self.config.geo.lanes
        s = self.signal.saturation_flow
        
        # Assume left turns use 1 lane, through+right use remaining lanes
        s_left = s * min(1, lanes)
        s_through = s * lanes
        
        y = [
            q_NS_L / s_left,
            q_NS_T / s_through,
            q_EW_L / s_left,
            q_EW_T / s_through
        ]
        
        return y
    
    def compute_optimal_cycle_length(self) -> float:
        """
        Webster's optimal cycle length formula:
        
        C_opt = (1.5 * L + 5) / (1 - Y)
        
        where:
        - L = total lost time per cycle = num_phases * (startup + clearance)
        - Y = sum of critical flow ratios
        """
        y_values = self.compute_flow_ratios()
        Y = sum(y_values)
        
        L = self.num_phases * (self.signal.lost_time_startup + self.signal.lost_time_clearance)
        
        if Y >= 1.0:
            print(f"⚠ WARNING: Flow ratio Y={Y:.3f} >= 1.0 (oversaturated!)")
            return self.signal.max_cycle
        
        C_opt = (1.5 * L + 5) / (1 - Y)
        C_opt = clamp(C_opt, self.signal.min_cycle, self.signal.max_cycle)
        
        print(f"Webster's Method: Y={Y:.3f}, L={L:.1f}s, C_opt={C_opt:.1f}s")
        return C_opt
    
    def allocate_green_times(self, cycle_length: float) -> List[float]:
        """
        Allocate effective green times proportional to flow ratios.
        
        g_i = (y_i / Y) * (C - L)
        """
        y_values = self.compute_flow_ratios()
        Y = sum(y_values)
        L = self.num_phases * (self.signal.lost_time_startup + self.signal.lost_time_clearance)
        
        effective_green_time = cycle_length - L
        
        green_times = []
        for y_i in y_values:
            g_i = (y_i / Y) * effective_green_time
            g_i = clamp(g_i, self.signal.min_green, self.signal.max_green)
            green_times.append(g_i)
        
        print(f"Green times: NS-L={green_times[0]:.1f}s, NS-T={green_times[1]:.1f}s, "
              f"EW-L={green_times[2]:.1f}s, EW-T={green_times[3]:.1f}s")
        
        return green_times

# ============================================================================
# SIGNALIZED INTERSECTION SIMULATION
# ============================================================================

class SignalizedIntersectionSim:
    """
    Microsimulation of a 4-way signalized intersection with Webster's Method.
    """
    
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        
        # Geometry
        self.N_arms = 4
        self.stop_line_pos = 0.0  # Stop line at position 0
        self.spawn_pos = -cfg.geo.approach_length
        
        # Signal timing
        if cfg.use_webster:
            webster = WebsterOptimizer(cfg)
            self.cycle_length = cfg.cycle_length or webster.compute_optimal_cycle_length()
            self.green_times = webster.allocate_green_times(self.cycle_length)
        else:
            # Manual timing
            self.cycle_length = cfg.cycle_length or 90.0
            # Equal split (simplified)
            base_green = (self.cycle_length - 4 * (cfg.signal.yellow_time + cfg.signal.all_red_time)) / 4
            self.green_times = [base_green] * 4
        
        # Phase structure: [NS-L, NS-T, EW-L, EW-T]
        self.yellow_time = cfg.signal.yellow_time
        self.all_red_time = cfg.signal.all_red_time
        
        # Build phase schedule
        self._build_phase_schedule()
        
        # State
        self.t = 0.0
        self.current_phase = 0
        self.phase_start_time = 0.0
        
        # Vehicles
        self._veh_seq = 0
        self.lanes: List[List[List[Vehicle]]] = [
            [[] for _ in range(cfg.geo.lanes)] for _ in range(self.N_arms)
        ]  # lanes[arm][lane] = list of vehicles
        
        # Metrics
        self.total_arrivals = 0
        self.total_departures = 0
        self.delays: List[float] = []
        self.max_queue_lengths = [[0] * cfg.geo.lanes for _ in range(self.N_arms)]
        
        # Window metrics
        self.win_arrivals = 0
        self.win_departures = 0
        self.win_delays: List[float] = []
        self.win_queue_max = [[0] * cfg.geo.lanes for _ in range(self.N_arms)]
        
        # Reaction delay history (for DDE-like IDM)
        self.history: Deque[Dict[Tuple[int, int, int], Tuple[float, float]]] = deque()
        self.max_hist_len = max(1, int(round(cfg.driver.tau / cfg.dt)))
    
    def _build_phase_schedule(self):
        """Build phase schedule with green/yellow/all-red intervals."""
        self.phases = []
        phase_names = ['NS-Left', 'NS-Through', 'EW-Left', 'EW-Through']
        
        for i, (green, name) in enumerate(zip(self.green_times, phase_names)):
            self.phases.append({
                'id': i,
                'name': name,
                'green': green,
                'yellow': self.yellow_time,
                'all_red': self.all_red_time,
                'total': green + self.yellow_time + self.all_red_time
            })
        
        print(f"\nPhase Schedule (Cycle={self.cycle_length:.1f}s):")
        for phase in self.phases:
            print(f"  Phase {phase['id']} ({phase['name']}): "
                  f"Green={phase['green']:.1f}s, Yellow={phase['yellow']:.1f}s, "
                  f"AllRed={phase['all_red']:.1f}s, Total={phase['total']:.1f}s")
    
    def _can_proceed(self, arm: int, turn: str) -> bool:
        """Check if a vehicle from (arm, turn) can proceed in current phase."""
        phase = self.phases[self.current_phase]
        elapsed = self.t - self.phase_start_time
        
        # Only proceed during green
        if elapsed > phase['green']:
            return False
        
        # Map movements to phases
        # Phase 0 (NS-L): North/South left turns
        # Phase 1 (NS-T): North/South through/right
        # Phase 2 (EW-L): East/West left turns
        # Phase 3 (EW-T): East/West through/right
        
        if self.current_phase == 0:
            return (arm in [0, 2]) and (turn == 'L')
        elif self.current_phase == 1:
            return (arm in [0, 2]) and (turn in ['T', 'R'])
        elif self.current_phase == 2:
            return (arm in [1, 3]) and (turn == 'L')
        elif self.current_phase == 3:
            return (arm in [1, 3]) and (turn in ['T', 'R'])
        
        return False
    
    def _update_signal(self):
        """Advance signal phase if needed."""
        phase = self.phases[self.current_phase]
        elapsed = self.t - self.phase_start_time
        
        if elapsed >= phase['total']:
            # Move to next phase
            self.current_phase = (self.current_phase + 1) % len(self.phases)
            self.phase_start_time = self.t
    
    # ========================================================================
    # ARRIVALS
    # ========================================================================
    
    def _draw_turn(self) -> str:
        """Sample turning movement."""
        p_L, p_T, p_R = self.cfg.dem.turning
        u = random.random()
        if u < p_L:
            return 'L'
        elif u < p_L + p_T:
            return 'T'
        else:
            return 'R'
    
    def _select_lane(self, arm: int, turn: str) -> int:
        """Select lane based on movement and availability."""
        lanes = self.cfg.geo.lanes
        
        if lanes == 1:
            return 0
        elif lanes == 2:
            # Lane 0: T+R, Lane 1: L+T
            if turn == 'L':
                return 1
            elif turn == 'R':
                return 0
            else:  # Through
                # Choose less congested lane
                len0 = len(self.lanes[arm][0])
                len1 = len(self.lanes[arm][1])
                return 0 if len0 <= len1 else 1
        else:  # 3 lanes
            # Lane 0: R+T, Lane 1: T, Lane 2: L+T
            if turn == 'L':
                return 2
            elif turn == 'R':
                return 0
            else:  # Through
                # Choose among lanes 0, 1, 2 (prefer lane 1)
                lens = [len(self.lanes[arm][i]) for i in range(3)]
                return lens.index(min(lens))
    
    def _spawn_arrivals(self):
        """Generate new arrivals via Poisson process."""
        dt = self.cfg.dt
        for arm in range(self.N_arms):
            lambda_arm = self.cfg.dem.arrival[arm]
            if random.random() < lambda_arm * dt:
                turn = self._draw_turn()
                lane = self._select_lane(arm, turn)
                
                v = Vehicle(
                    id=self._veh_seq,
                    arm=arm,
                    turn=turn,
                    lane=lane,
                    pos=self.spawn_pos,
                    speed=self.cfg.driver.v0 * 0.5,  # Start at half desired speed
                    t_arrival=self.t
                )
                self._veh_seq += 1
                
                self.lanes[arm][lane].append(v)
                self.total_arrivals += 1
                self.win_arrivals += 1
    
    # ========================================================================
    # HISTORY (for reaction delay)
    # ========================================================================
    
    def _push_history(self):
        """Save current state for DDE."""
        snap: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
        for arm in range(self.N_arms):
            for lane_idx in range(self.cfg.geo.lanes):
                for v in self.lanes[arm][lane_idx]:
                    snap[(arm, lane_idx, v.id)] = (v.pos, v.speed)
        
        self.history.append(snap)
        if len(self.history) > self.max_hist_len:
            self.history.popleft()
    
    def _snapshot(self, steps_back: int) -> Dict[Tuple[int, int, int], Tuple[float, float]]:
        """Get historical snapshot."""
        if steps_back <= 0 or steps_back > len(self.history):
            return {}
        return list(self.history)[-steps_back]
    
    # ========================================================================
    # CAR-FOLLOWING (IDM)
    # ========================================================================
    
    def _advance_vehicles(self):
        """Move all vehicles using IDM car-following."""
        dt = self.cfg.dt
        drv = self.cfg.driver
        
        steps_back = int(round(drv.tau / dt))
        snap = self._snapshot(steps_back) if steps_back > 0 else {}
        
        for arm in range(self.N_arms):
            for lane_idx in range(self.cfg.geo.lanes):
                lane_vehicles = self.lanes[arm][lane_idx]
                if not lane_vehicles:
                    continue
                
                # Sort by position (increasing)
                lane_vehicles.sort(key=lambda v: v.pos)
                
                new_states: List[Tuple[Vehicle, float, float]] = []
                
                for i, v in enumerate(lane_vehicles):
                    # Delayed state
                    key = (arm, lane_idx, v.id)
                    pos_d, v_d = snap.get(key, (v.pos, v.speed))
                    
                    # Find leader
                    if i < len(lane_vehicles) - 1:
                        leader = lane_vehicles[i + 1]
                        leader_key = (arm, lane_idx, leader.id)
                        leader_pos_d, leader_v_d = snap.get(leader_key, (leader.pos, leader.speed))
                        gap = leader_pos_d - pos_d - 5.0  # Assume 5m vehicle length
                        v_lead = leader_v_d
                    else:
                        # No leader; check if approaching stop line
                        if v.pos < self.stop_line_pos:
                            # Check signal
                            if self._can_proceed(arm, v.turn):
                                # Green: no constraint
                                gap = 1e9
                                v_lead = drv.v0
                            else:
                                # Red: treat stop line as obstacle
                                gap = max(0.1, self.stop_line_pos - pos_d)
                                v_lead = 0.0
                        else:
                            # Past stop line: free flow
                            gap = 1e9
                            v_lead = drv.v0
                    
                    # IDM acceleration
                    v0 = drv.v0
                    s0 = drv.s0
                    T = drv.T
                    a_max = drv.a_max
                    b = drv.b_comf
                    delta = drv.delta
                    
                    dv = v_d - v_lead
                    s_star = s0 + max(0, v_d * T + (v_d * dv) / (2 * math.sqrt(a_max * b)))
                    
                    acc = a_max * (
                        1.0 - (v_d / max(0.1, v0)) ** delta
                        - (s_star / max(1e-3, gap)) ** 2
                    )
                    
                    # Update
                    v_new = clamp(v.speed + acc * dt, 0.0, v0)
                    pos_new = v.pos + v_new * dt
                    
                    # Track stops
                    if v.speed > 0.5 and v_new < 0.5:
                        v.stops += 1
                    
                    new_states.append((v, pos_new, v_new))
                
                # Apply new states
                for v, pos_new, v_new in new_states:
                    v.pos = pos_new
                    v.speed = v_new
                    
                    # Update queue status
                    if v.pos < self.stop_line_pos:
                        v.in_queue = True
                    else:
                        v.in_queue = False
                        v.crossed = True
                
                # Track queue lengths
                queue_len = sum(1 for v in lane_vehicles if v.in_queue)
                self.win_queue_max[arm][lane_idx] = max(self.win_queue_max[arm][lane_idx], queue_len)
                self.max_queue_lengths[arm][lane_idx] = max(self.max_queue_lengths[arm][lane_idx], queue_len)
    
    # ========================================================================
    # DEPARTURES
    # ========================================================================
    
    def _process_departures(self):
        """Remove vehicles that have exited the system."""
        departure_threshold = self.cfg.geo.intersection_size
        
        for arm in range(self.N_arms):
            for lane_idx in range(self.cfg.geo.lanes):
                survivors = []
                for v in self.lanes[arm][lane_idx]:
                    if v.crossed and v.pos > departure_threshold:
                        # Vehicle has exited
                        v.t_departure = self.t
                        delay = v.t_departure - v.t_arrival
                        self.delays.append(delay)
                        self.win_delays.append(delay)
                        self.total_departures += 1
                        self.win_departures += 1
                    else:
                        survivors.append(v)
                
                self.lanes[arm][lane_idx] = survivors
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def _report_window(self, end_time: float):
        """Report windowed metrics."""
        if self.win_delays:
            avg_delay = sum(self.win_delays) / len(self.win_delays)
            p95 = stats.quantiles(self.win_delays, n=20)[18] if len(self.win_delays) >= 20 else avg_delay
        else:
            avg_delay = 0.0
            p95 = 0.0
        
        throughput = (self.win_departures * 3600.0) / max(1e-9, self.cfg.report_every)
        
        # Max queue per arm (across all lanes)
        max_q_per_arm = [max(self.win_queue_max[arm]) for arm in range(self.N_arms)]
        max_q_str = f"[{', '.join(str(q) for q in max_q_per_arm)}]"
        
        print(f"[{mmss(end_time)}] arrivals={self.win_arrivals} exits={self.win_departures} "
              f"throughput={throughput:.0f} veh/hr  avg_delay={avg_delay:.1f}s  p95={p95:.1f}s  "
              f"max_q={max_q_str}")
        
        # Reset window
        self.win_arrivals = 0
        self.win_departures = 0
        self.win_delays = []
        self.win_queue_max = [[0] * self.cfg.geo.lanes for _ in range(self.N_arms)]
    
    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    
    def run(self):
        """Execute simulation."""
        next_report = self.cfg.report_every
        
        print(f"\n{'='*70}")
        print(f"Starting Signalized Intersection Simulation")
        print(f"  Horizon: {self.cfg.horizon}s ({self.cfg.horizon/60:.1f} min)")
        print(f"  Lanes per approach: {self.cfg.geo.lanes}")
        print(f"  Cycle length: {self.cycle_length:.1f}s")
        print(f"  Arrival rates: {[f'{a:.3f}' for a in self.cfg.dem.arrival]} veh/s per arm")
        print(f"{'='*70}\n")
        
        while self.t < self.cfg.horizon + 1e-9:
            self._spawn_arrivals()
            self._push_history()
            self._update_signal()
            self._advance_vehicles()
            self._process_departures()
            
            self.t += self.cfg.dt
            
            if self.t + 1e-9 >= next_report:
                self._report_window(next_report)
                next_report += self.cfg.report_every
        
        # Final summary
        self._final_summary()
    
    def _final_summary(self):
        """Print final aggregate statistics."""
        throughput_hr = (self.total_departures * 3600.0) / max(1e-9, self.cfg.horizon)
        
        if self.delays:
            avg_delay = sum(self.delays) / len(self.delays)
            p95_delay = stats.quantiles(self.delays, n=20)[18] if len(self.delays) >= 20 else avg_delay
        else:
            avg_delay = 0.0
            p95_delay = 0.0
        
        max_q_per_arm = [max(self.max_queue_lengths[arm]) for arm in range(self.N_arms)]
        
        print(f"\n{'='*70}")
        print(f"=== SIMULATION SUMMARY ===")
        print(f"  Total arrivals: {self.total_arrivals}")
        print(f"  Total departures: {self.total_departures}")
        print(f"  Throughput: {throughput_hr:.0f} veh/hr")
        print(f"  Average delay: {avg_delay:.1f}s")
        print(f"  P95 delay: {p95_delay:.1f}s")
        print(f"  Max queue per arm [N,E,S,W]: {max_q_per_arm}")
        print(f"{'='*70}\n")

# ============================================================================
# CLI
# ============================================================================

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Signalized Intersection Microsimulation")
    
    # Simulation parameters
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--horizon', type=float, default=3600.0, help='Simulation duration (s)')
    p.add_argument('--report-every', type=float, default=300.0, help='Reporting interval (s)')
    p.add_argument('--dt', type=float, default=0.2, help='Time step (s)')
    
    # Geometry
    p.add_argument('--lanes', type=int, default=2, choices=[1, 2, 3],
                  help='Lanes per approach')
    p.add_argument('--approach-length', type=float, default=200.0,
                  help='Approach length (m)')
    
    # Demand
    p.add_argument('--arrival', type=float, nargs=4, default=[0.18, 0.12, 0.20, 0.15],
                  help='Arrival rates per arm [N,E,S,W] (veh/s)')
    p.add_argument('--turning', type=float, nargs=3, default=[0.25, 0.55, 0.20],
                  help='Turning probabilities [L,T,R]')
    
    # Signal timing
    p.add_argument('--use-webster', action='store_true', default=True,
                  help='Use Webster\'s Method for optimal timing')
    p.add_argument('--cycle-length', type=float, default=None,
                  help='Manual cycle length (s); overrides Webster if set')
    
    # Driver behavior
    p.add_argument('--accel', type=float, default=1.5, help='Max acceleration (m/s²)')
    p.add_argument('--decel', type=float, default=2.0, help='Comfortable deceleration (m/s²)')
    p.add_argument('--tau', type=float, default=1.2, help='Desired time headway (s)')
    
    return p

def from_args(args: argparse.Namespace) -> SimConfig:
    """Build configuration from CLI arguments."""
    geo = Geometry(
        approach_length=args.approach_length,
        lanes=args.lanes
    )
    
    dem = Demand(
        arrival=tuple(args.arrival),
        turning=tuple(args.turning)
    )
    
    driver = DriverParams(
        a_max=args.accel,
        b_comf=args.decel,
        T=args.tau
    )
    
    cfg = SimConfig(
        seed=args.seed,
        horizon=args.horizon,
        report_every=args.report_every,
        dt=args.dt,
        driver=driver,
        geo=geo,
        dem=dem,
        use_webster=args.use_webster,
        cycle_length=args.cycle_length
    )
    
    return cfg

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    cfg = from_args(args)
    
    sim = SignalizedIntersectionSim(cfg)
    sim.run()
