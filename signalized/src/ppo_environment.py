"""
PPO Gymnasium Environment for Adaptive Signal Control

Implements a Gymnasium environment for training RL agents to optimize
traffic signal timing. The agent observes traffic conditions (queues,
waiting times, throughput) and adjusts green times for each phase.

State Space:
- Queue length per approach (4 values)
- Average waiting time per approach (4 values)
- Recent throughput per approach (4 values)
- Current phase elapsed time (1 value)
- Time of day (2 values: sin, cos encoding)
Total: 17 dimensions

Action Space:
- Adjust green time for each phase: {-1, 0, +1} → {-5s, 0s, +5s}
- 3^4 = 81 discrete actions (or 4-dimensional MultiDiscrete)

Reward Function:
- Maximize throughput
- Minimize delay
- Minimize queue length
- Minimize stops
- Promote fairness across approaches
"""

import os
import sys
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml
import traci
import sumolib
from typing import Dict, List, Tuple, Optional


class SignalControlEnv(gym.Env):
    """
    Gymnasium environment for traffic signal control using PPO.
    
    The agent controls a 4-phase traffic signal by adjusting green times
    based on real-time traffic conditions from SUMO simulation.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        config_path: str,
        sumo_cfg_path: str,
        use_gui: bool = False,
        demand_multiplier: float = 1.0,
        episode_length: int = 3600,
        action_step: float = 5.0
    ):
        """
        Initialize the environment.
        
        Args:
            config_path: Path to config.yaml
            sumo_cfg_path: Path to SUMO .sumocfg file
            use_gui: Whether to use SUMO-GUI (for debugging)
            demand_multiplier: Scale factor for traffic demand
            episode_length: Episode duration in seconds
            action_step: Green time adjustment step (seconds)
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sumo_cfg_path = sumo_cfg_path
        self.use_gui = use_gui
        self.demand_multiplier = demand_multiplier
        self.episode_length = episode_length
        self.action_step = action_step
        
        # PPO parameters
        ppo_cfg = self.config['signal']['ppo']
        self.min_green = ppo_cfg['min_green_time']
        self.max_green = ppo_cfg['max_green_time']
        self.reward_weights = ppo_cfg['reward_weights']
        
        # Traffic light parameters
        self.tl_id = 'center'
        self.num_phases = 4
        self.phase_names = ['NS_left', 'NS_through', 'EW_left', 'EW_through']
        
        # Approach edges (for measurement)
        self.approach_edges = ['north_in', 'east_in', 'south_in', 'west_in']
        
        # State tracking
        self.current_step = 0
        self.current_phase = 0
        self.phase_elapsed = 0.0
        self.episode_stats = {
            'total_throughput': 0,
            'total_delay': 0,
            'total_stops': 0,
            'vehicles_completed': 0
        }
        
        # History for throughput calculation
        self.throughput_window = 300  # 5-minute window
        self.arrival_times = {edge: [] for edge in self.approach_edges}
        self.departure_times = []
        
        # Define observation space
        # [queue_lengths (4), waiting_times (4), throughput (4), 
        #  phase_elapsed (1), time_sin (1), time_cos (1)] = 17 dimensions
        self.observation_space = spaces.Box(
            low=np.array([0]*4 + [0]*4 + [0]*4 + [0] + [-1, -1]),
            high=np.array([50]*4 + [300]*4 + [100]*4 + [self.max_green] + [1, 1]),
            dtype=np.float32
        )
        
        # Define action space
        # 4 phases × 3 actions (-1, 0, +1) = 81 total actions
        # We'll use MultiDiscrete: [3, 3, 3, 3]
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])
        
        # SUMO connection
        self.sumo = None
        self.sumo_port = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Close existing SUMO connection
        if self.sumo is not None:
            try:
                traci.close()
            except:
                pass
        
        # Start SUMO
        self._start_sumo()
        
        # Reset state
        self.current_step = 0
        self.current_phase = 0
        self.phase_elapsed = 0.0
        self.episode_stats = {
            'total_throughput': 0,
            'total_delay': 0,
            'total_stops': 0,
            'vehicles_completed': 0
        }
        self.arrival_times = {edge: [] for edge in self.approach_edges}
        self.departure_times = []
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step.
        
        Args:
            action: Action array [a1, a2, a3, a4] where each ai ∈ {0, 1, 2}
                   0 = decrease green by 5s
                   1 = keep green same
                   2 = increase green by 5s
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to green time adjustments
        adjustments = (action - 1) * self.action_step  # [-5, 0, +5]
        
        # Apply action (adjust signal timing)
        self._apply_action(adjustments)
        
        # Run SUMO for one phase duration
        phase_duration = self._get_current_green_time()
        steps_to_run = int(phase_duration / self.config['simulation']['step_length'])
        
        for _ in range(steps_to_run):
            traci.simulationStep()
            self.current_step += 1
            self.phase_elapsed += self.config['simulation']['step_length']
            
            # Track vehicles
            self._track_vehicles()
        
        # Move to next phase
        self.current_phase = (self.current_phase + 1) % self.num_phases
        self.phase_elapsed = 0.0
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Info
        info = {
            'throughput': self._get_recent_throughput(),
            'avg_delay': self._get_average_delay(),
            'max_queue': self._get_max_queue(),
            'episode_stats': self.episode_stats.copy()
        }
        
        return obs, reward, terminated, truncated, info
    
    def _start_sumo(self):
        """Start SUMO simulation."""
        import random
        
        # Generate random port to avoid conflicts
        self.sumo_port = random.randint(10000, 60000)
        
        sumo_cmd = [
            'sumo-gui' if self.use_gui else 'sumo',
            '-c', self.sumo_cfg_path,
            '--step-length', str(self.config['simulation']['step_length']),
            '--no-warnings',
            '--no-step-log',
            '--time-to-teleport', '-1',
            '--remote-port', str(self.sumo_port)
        ]
        
        traci.start(sumo_cmd)
        self.sumo = traci
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation of traffic state.
        
        Returns:
            Observation vector (17 dimensions)
        """
        # Queue lengths per approach
        queue_lengths = []
        for edge in self.approach_edges:
            try:
                vehicles = traci.edge.getLastStepVehicleIDs(edge)
                stopped = sum(1 for vid in vehicles if traci.vehicle.getSpeed(vid) < 0.1)
                queue_lengths.append(stopped)
            except:
                queue_lengths.append(0)
        
        # Average waiting times per approach
        waiting_times = []
        for edge in self.approach_edges:
            try:
                vehicles = traci.edge.getLastStepVehicleIDs(edge)
                if vehicles:
                    avg_wait = np.mean([traci.vehicle.getWaitingTime(vid) for vid in vehicles])
                else:
                    avg_wait = 0.0
                waiting_times.append(avg_wait)
            except:
                waiting_times.append(0.0)
        
        # Recent throughput per approach (vehicles/hour)
        throughput = []
        for edge in self.approach_edges:
            recent_arrivals = [t for t in self.arrival_times[edge] 
                             if self.current_step - t < self.throughput_window]
            rate = len(recent_arrivals) / (self.throughput_window / 3600.0) if recent_arrivals else 0.0
            throughput.append(rate)
        
        # Current phase elapsed time
        phase_elapsed = [self.phase_elapsed]
        
        # Time of day (cyclical encoding)
        time_fraction = (self.current_step % self.episode_length) / self.episode_length
        time_sin = [math.sin(2 * math.pi * time_fraction)]
        time_cos = [math.cos(2 * math.pi * time_fraction)]
        
        # Concatenate all features
        obs = np.array(
            queue_lengths + waiting_times + throughput + 
            phase_elapsed + time_sin + time_cos,
            dtype=np.float32
        )
        
        return obs
    
    def _apply_action(self, adjustments: np.ndarray):
        """
        Apply action by adjusting green times.
        
        Args:
            adjustments: Green time adjustments for each phase (seconds)
        """
        # Get current traffic light program
        try:
            program_logic = traci.trafficlight.getAllProgramLogics(self.tl_id)[0]
            
            # Modify green times
            new_phases = []
            phase_idx = 0
            
            for phase in program_logic.phases:
                # Identify if this is a green phase (not yellow/red transition)
                if 'G' in phase.state or 'g' in phase.state:
                    # Apply adjustment
                    adjustment = adjustments[phase_idx % self.num_phases]
                    new_duration = max(self.min_green, 
                                     min(self.max_green, phase.duration + adjustment))
                    new_phase = traci.trafficlight.Phase(
                        new_duration, phase.state, phase.minDur, phase.maxDur
                    )
                    new_phases.append(new_phase)
                    phase_idx += 1
                else:
                    # Keep yellow/red phases unchanged
                    new_phases.append(phase)
            
            # Create new program
            new_logic = traci.trafficlight.Logic(
                program_logic.programID,
                program_logic.type,
                program_logic.currentPhaseIndex,
                new_phases
            )
            
            # Set new program
            traci.trafficlight.setProgramLogic(self.tl_id, new_logic)
            
        except Exception as e:
            # If adjustment fails, continue with current timing
            pass
    
    def _get_current_green_time(self) -> float:
        """Get current phase green time."""
        try:
            program = traci.trafficlight.getAllProgramLogics(self.tl_id)[0]
            phase_idx = traci.trafficlight.getPhase(self.tl_id)
            return program.phases[phase_idx].duration
        except:
            return 30.0  # Default
    
    def _track_vehicles(self):
        """Track vehicle arrivals and departures."""
        # Track arrivals
        for edge in self.approach_edges:
            try:
                vehicles = traci.edge.getLastStepVehicleIDs(edge)
                for vid in vehicles:
                    if vid not in [item for sublist in self.arrival_times.values() for item in sublist]:
                        self.arrival_times[edge].append(self.current_step)
            except:
                pass
        
        # Track departures
        try:
            arrived = traci.simulation.getArrivedIDList()
            self.departure_times.extend([self.current_step] * len(arrived))
            self.episode_stats['vehicles_completed'] += len(arrived)
        except:
            pass
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on current traffic conditions.
        
        Reward components:
        - Throughput (maximize)
        - Delay (minimize)
        - Queue length (minimize)
        - Stops (minimize)
        - Fairness (promote balanced service)
        """
        weights = self.reward_weights
        
        # Throughput (vehicles/hour)
        throughput = self._get_recent_throughput()
        throughput_reward = weights['throughput'] * throughput / 100.0
        
        # Average delay (seconds)
        avg_delay = self._get_average_delay()
        delay_penalty = weights['delay'] * avg_delay / 100.0
        
        # Maximum queue (vehicles)
        max_queue = self._get_max_queue()
        queue_penalty = weights['queue'] * max_queue / 50.0
        
        # Total stops
        stops = self._get_total_stops()
        stops_penalty = weights['stops'] * stops / 100.0
        
        # Fairness (reward balanced service)
        fairness = self._get_fairness_bonus()
        fairness_reward = weights['fairness'] * fairness
        
        # Total reward
        reward = (throughput_reward + delay_penalty + 
                 queue_penalty + stops_penalty + fairness_reward)
        
        return reward
    
    def _get_recent_throughput(self) -> float:
        """Get recent throughput (vehicles/hour)."""
        recent_departures = [t for t in self.departure_times 
                            if self.current_step - t < self.throughput_window]
        return len(recent_departures) / (self.throughput_window / 3600.0)
    
    def _get_average_delay(self) -> float:
        """Get average waiting time across all vehicles."""
        try:
            all_vehicles = traci.vehicle.getIDList()
            if not all_vehicles:
                return 0.0
            
            waiting_times = [traci.vehicle.getWaitingTime(vid) for vid in all_vehicles]
            return np.mean(waiting_times)
        except:
            return 0.0
    
    def _get_max_queue(self) -> float:
        """Get maximum queue length across all approaches."""
        queues = []
        for edge in self.approach_edges:
            try:
                vehicles = traci.edge.getLastStepVehicleIDs(edge)
                stopped = sum(1 for vid in vehicles if traci.vehicle.getSpeed(vid) < 0.1)
                queues.append(stopped)
            except:
                queues.append(0)
        
        return max(queues) if queues else 0
    
    def _get_total_stops(self) -> int:
        """Get total number of stopped vehicles."""
        try:
            all_vehicles = traci.vehicle.getIDList()
            return sum(1 for vid in all_vehicles if traci.vehicle.getSpeed(vid) < 0.1)
        except:
            return 0
    
    def _get_fairness_bonus(self) -> float:
        """
        Calculate fairness bonus (reward balanced service).
        
        Fairness measured as inverse of coefficient of variation in queue lengths.
        """
        queues = []
        for edge in self.approach_edges:
            try:
                vehicles = traci.edge.getLastStepVehicleIDs(edge)
                stopped = sum(1 for vid in vehicles if traci.vehicle.getSpeed(vid) < 0.1)
                queues.append(stopped)
            except:
                queues.append(0)
        
        if not queues or sum(queues) == 0:
            return 1.0  # Perfect fairness if no queues
        
        mean_queue = np.mean(queues)
        std_queue = np.std(queues)
        
        if mean_queue == 0:
            return 1.0
        
        cv = std_queue / mean_queue  # Coefficient of variation
        fairness = 1.0 / (1.0 + cv)  # Fairness ∈ [0, 1]
        
        return fairness
    
    def close(self):
        """Close the environment and SUMO connection."""
        if self.sumo is not None:
            try:
                traci.close()
            except:
                pass
        self.sumo = None
    
    def render(self):
        """Render environment (SUMO-GUI handles visualization)."""
        pass


def test_environment():
    """Test the environment."""
    import os
    
    config_path = '../config/config.yaml'
    sumo_cfg_path = '../quickstart_output/sumo_configs/webster/intersection.sumocfg'
    
    if not os.path.exists(sumo_cfg_path):
        print("❌ SUMO config not found. Run quickstart.py first!")
        return
    
    print("Testing PPO environment...")
    
    env = SignalControlEnv(
        config_path=config_path,
        sumo_cfg_path=sumo_cfg_path,
        use_gui=False,
        episode_length=300  # 5 minutes for testing
    )
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful. Observation shape: {obs.shape}")
    print(f"   Observation: {obs}")
    
    # Test steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n   Step {i+1}:")
        print(f"   Action: {action}")
        print(f"   Reward: {reward:.2f}")
        print(f"   Throughput: {info['throughput']:.1f} veh/hr")
        print(f"   Avg Delay: {info['avg_delay']:.1f} s")
        print(f"   Max Queue: {info['max_queue']:.0f} veh")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n✅ Environment test complete!")


if __name__ == '__main__':
    test_environment()
