"""
Webster's Method for Optimal Signal Timing

Implements F.V. Webster's analytical method for determining optimal cycle length
and green time allocation to minimize average vehicle delay at signalized intersections.

References:
- Webster, F. V. (1958). Traffic signal settings. Road Research Technical Paper No. 39.
- Transportation Research Board (2010). Highway Capacity Manual 2010.
"""

import math
from typing import Dict, List, Tuple
import yaml


class WebsterSignalOptimizer:
    """
    Computes optimal signal timing using Webster's Method.
    
    Webster's Method minimizes average vehicle delay through:
    1. Optimal cycle length based on demand and lost time
    2. Green time allocation proportional to critical flow ratios
    """
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        Initialize Webster optimizer.
        
        Args:
            config_path: Path to YAML config file
            config_dict: Config dictionary (alternative to file)
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")
        
        self.saturation_flow = self.config['signal']['webster']['saturation_flow']
        self.startup_lost_time = self.config['signal']['webster']['startup_lost_time']
        self.clearance_lost_time = self.config['signal']['webster']['clearance_lost_time']
        self.yellow_time = self.config['signal']['webster']['yellow_time']
        self.all_red_time = self.config['signal']['webster']['all_red_time']
        self.min_cycle = self.config['signal']['webster']['min_cycle']
        self.max_cycle = self.config['signal']['webster']['max_cycle']
        
        # Demand
        self.arrivals = self.config['demand']['arrivals']  # veh/s per arm
        self.turning_probs = self.config['demand']['turning_probabilities']  # [L, T, R]
        
    def compute_flow_rates(self, demand_multiplier: float = 1.0) -> Dict[str, Dict[str, float]]:
        """
        Compute flow rates by movement (veh/hour).
        
        Args:
            demand_multiplier: Scale factor for demand
        
        Returns:
            Dictionary of flow rates by arm and movement:
            {
                'north': {'left': X, 'through': Y, 'right': Z},
                'east': {...},
                'south': {...},
                'west': {...}
            }
        """
        arms = ['north', 'east', 'south', 'west']
        flows = {}
        
        for i, arm in enumerate(arms):
            arrival_rate = self.arrivals[i] * demand_multiplier  # veh/s
            arrival_rate_hr = arrival_rate * 3600  # veh/hour
            
            flows[arm] = {
                'left': arrival_rate_hr * self.turning_probs[0],
                'through': arrival_rate_hr * self.turning_probs[1],
                'right': arrival_rate_hr * self.turning_probs[2],
                'total': arrival_rate_hr
            }
        
        return flows
    
    def identify_critical_movements(self, flows: Dict) -> Dict[str, str]:
        """
        Identify critical movements for each phase.
        
        For a 4-phase signal (NS-Left, NS-Through, EW-Left, EW-Through),
        critical movement is the one with highest flow ratio in each phase.
        
        Args:
            flows: Flow rates dictionary
        
        Returns:
            Dictionary mapping phase to critical movement:
            {
                'phase1': 'north_left',
                'phase2': 'north_through',
                'phase3': 'east_left',
                'phase4': 'east_through'
            }
        """
        # Phase 1: North-South Left turns (protected)
        ns_left = max(flows['north']['left'], flows['south']['left'])
        
        # Phase 2: North-South Through + Right (right turn on red allowed, so through dominates)
        ns_through = max(
            flows['north']['through'] + flows['north']['right'],
            flows['south']['through'] + flows['south']['right']
        )
        
        # Phase 3: East-West Left turns (protected)
        ew_left = max(flows['east']['left'], flows['west']['left'])
        
        # Phase 4: East-West Through + Right
        ew_through = max(
            flows['east']['through'] + flows['east']['right'],
            flows['west']['through'] + flows['west']['right']
        )
        
        return {
            'phase1': ('NS_left', ns_left),
            'phase2': ('NS_through', ns_through),
            'phase3': ('EW_left', ew_left),
            'phase4': ('EW_through', ew_through)
        }
    
    def compute_flow_ratios(self, critical_movements: Dict) -> Tuple[List[float], float]:
        """
        Compute flow ratios y_i = q_i / s_i for each phase.
        
        Args:
            critical_movements: Critical movements and their flows
        
        Returns:
            Tuple of (flow_ratios, total_flow_ratio):
            - flow_ratios: List of y_i for each phase
            - total_flow_ratio: Y = sum(y_i)
        """
        flow_ratios = []
        
        for phase, (movement, flow) in critical_movements.items():
            # Flow ratio y_i = q_i / s_i
            # q_i in veh/hour, s_i = saturation_flow (veh/hour/lane)
            # Assume 1 lane per movement for simplicity
            y_i = flow / self.saturation_flow
            flow_ratios.append(y_i)
        
        total_flow_ratio = sum(flow_ratios)
        
        return flow_ratios, total_flow_ratio
    
    def compute_optimal_cycle_length(self, total_flow_ratio: float) -> float:
        """
        Compute optimal cycle length using Webster's formula.
        
        Webster's Formula:
            C_opt = (1.5 * L + 5) / (1 - Y)
        
        Where:
            L = total lost time per cycle (sum of lost times for all phases)
            Y = sum of critical flow ratios
        
        Args:
            total_flow_ratio: Y = sum(y_i)
        
        Returns:
            Optimal cycle length (seconds)
        
        Raises:
            ValueError: If Y >= 1 (intersection is over-saturated)
        """
        if total_flow_ratio >= 1.0:
            raise ValueError(
                f"Intersection is over-saturated! Y = {total_flow_ratio:.3f} >= 1.0\n"
                f"Reduce demand or increase capacity (more lanes, higher saturation flow)."
            )
        
        # Total lost time per cycle (seconds)
        # L = sum of (startup + clearance) for each phase
        num_phases = 4
        L = num_phases * (self.startup_lost_time + self.clearance_lost_time)
        
        # Webster's formula
        C_opt = (1.5 * L + 5) / (1 - total_flow_ratio)
        
        # Constrain to practical range
        C_opt = max(self.min_cycle, min(C_opt, self.max_cycle))
        
        return C_opt
    
    def allocate_green_times(
        self, 
        cycle_length: float, 
        flow_ratios: List[float],
        total_flow_ratio: float
    ) -> List[float]:
        """
        Allocate green time to each phase proportional to flow ratios.
        
        Formula:
            g_i = (y_i / Y) * (C - L)
        
        Where:
            g_i = green time for phase i
            y_i = flow ratio for phase i
            Y = total flow ratio
            C = cycle length
            L = total lost time
        
        Args:
            cycle_length: Cycle length (seconds)
            flow_ratios: Flow ratios [y1, y2, y3, y4]
            total_flow_ratio: Y = sum(y_i)
        
        Returns:
            List of green times [g1, g2, g3, g4] (seconds)
        """
        num_phases = 4
        L = num_phases * (self.startup_lost_time + self.clearance_lost_time)
        
        # Effective green time available
        effective_green = cycle_length - L
        
        # Allocate proportionally
        green_times = []
        for y_i in flow_ratios:
            g_i = (y_i / total_flow_ratio) * effective_green
            green_times.append(g_i)
        
        return green_times
    
    def compute_delays(
        self,
        flows: Dict,
        cycle_length: float,
        green_times: List[float],
        critical_movements: Dict
    ) -> Dict[str, float]:
        """
        Compute average delay per vehicle using Webster's delay formula.
        
        Webster's Delay Formula:
            d_i = C*(1 - λ_i)^2 / (2*(1 - y_i)) + x_i^2 / (2*q_i*(1 - x_i))
        
        Where:
            λ_i = g_i / C (effective green ratio)
            y_i = q_i / s_i (flow ratio)
            x_i = y_i / λ_i (degree of saturation)
            d_i = average delay (seconds/vehicle)
        
        Args:
            flows: Flow rates dictionary
            cycle_length: Cycle length (seconds)
            green_times: Green times for each phase (seconds)
            critical_movements: Critical movements dictionary
        
        Returns:
            Dictionary of average delays by phase (seconds/vehicle)
        """
        delays = {}
        
        for i, (phase, (movement, flow)) in enumerate(critical_movements.items()):
            g_i = green_times[i]
            lambda_i = g_i / cycle_length  # Effective green ratio
            
            # Flow ratio
            y_i = flow / self.saturation_flow
            
            # Degree of saturation
            x_i = y_i / lambda_i if lambda_i > 0 else 999
            
            # Check stability
            if x_i >= 1.0:
                delays[phase] = 999.0  # Unstable (queue grows indefinitely)
                continue
            
            # Flow rate (veh/second)
            q_i = flow / 3600.0
            
            # Webster's delay formula
            # First term: uniform delay (deterministic arrivals)
            uniform_delay = (cycle_length * (1 - lambda_i)**2) / (2 * (1 - y_i))
            
            # Second term: overflow delay (random arrivals)
            overflow_delay = (x_i**2) / (2 * q_i * (1 - x_i))
            
            d_i = uniform_delay + overflow_delay
            delays[phase] = d_i
        
        return delays
    
    def optimize(self, demand_multiplier: float = 1.0, verbose: bool = True) -> Dict:
        """
        Perform complete Webster's Method optimization.
        
        Args:
            demand_multiplier: Scale factor for demand
            verbose: Print detailed results
        
        Returns:
            Dictionary containing:
            {
                'cycle_length': optimal cycle length (s),
                'green_times': [g1, g2, g3, g4] (s),
                'flow_ratios': [y1, y2, y3, y4],
                'total_flow_ratio': Y,
                'delays': {phase: delay} (s/veh),
                'avg_delay': weighted average delay (s/veh),
                'flows': flow rates dictionary
            }
        """
        # Step 1: Compute flow rates
        flows = self.compute_flow_rates(demand_multiplier)
        
        # Step 2: Identify critical movements
        critical_movements = self.identify_critical_movements(flows)
        
        # Step 3: Compute flow ratios
        flow_ratios, total_flow_ratio = self.compute_flow_ratios(critical_movements)
        
        # Step 4: Compute optimal cycle length
        try:
            cycle_length = self.compute_optimal_cycle_length(total_flow_ratio)
        except ValueError as e:
            if verbose:
                print(f"ERROR: {e}")
            return {
                'cycle_length': None,
                'green_times': None,
                'flow_ratios': flow_ratios,
                'total_flow_ratio': total_flow_ratio,
                'delays': None,
                'avg_delay': None,
                'flows': flows,
                'error': str(e)
            }
        
        # Step 5: Allocate green times
        green_times = self.allocate_green_times(cycle_length, flow_ratios, total_flow_ratio)
        
        # Step 6: Compute delays
        delays = self.compute_delays(flows, cycle_length, green_times, critical_movements)
        
        # Compute weighted average delay
        total_flow = sum(crit[1] for crit in critical_movements.values())
        avg_delay = sum(
            delays[phase] * critical_movements[phase][1] / total_flow
            for phase in delays.keys()
        )
        
        result = {
            'cycle_length': cycle_length,
            'green_times': green_times,
            'flow_ratios': flow_ratios,
            'total_flow_ratio': total_flow_ratio,
            'delays': delays,
            'avg_delay': avg_delay,
            'flows': flows,
            'critical_movements': critical_movements
        }
        
        if verbose:
            self.print_results(result)
        
        return result
    
    def print_results(self, result: Dict):
        """Print formatted optimization results."""
        print("\n" + "="*70)
        print("WEBSTER'S METHOD - OPTIMAL SIGNAL TIMING")
        print("="*70)
        
        print(f"\n{'DEMAND ANALYSIS':^70}")
        print("-"*70)
        flows = result['flows']
        for arm in ['north', 'east', 'south', 'west']:
            print(f"{arm.upper():>10}: L={flows[arm]['left']:>6.0f} "
                  f"T={flows[arm]['through']:>6.0f} "
                  f"R={flows[arm]['right']:>6.0f} "
                  f"Total={flows[arm]['total']:>6.0f} veh/hr")
        
        print(f"\n{'CRITICAL MOVEMENTS':^70}")
        print("-"*70)
        for phase, (movement, flow) in result['critical_movements'].items():
            y_i = result['flow_ratios'][int(phase[-1])-1]
            print(f"{phase:>10}: {movement:<15} Flow={flow:>6.0f} veh/hr  y={y_i:.3f}")
        
        print(f"\nTotal Flow Ratio Y = {result['total_flow_ratio']:.3f}")
        
        if result['cycle_length'] is None:
            print("\n⚠️  INTERSECTION OVER-SATURATED - No valid solution")
            return
        
        print(f"\n{'OPTIMAL TIMING':^70}")
        print("-"*70)
        print(f"Cycle Length: {result['cycle_length']:.1f} seconds")
        print(f"\nGreen Times:")
        phases = ['Phase 1 (NS Left)', 'Phase 2 (NS Through)', 
                  'Phase 3 (EW Left)', 'Phase 4 (EW Through)']
        for i, phase_name in enumerate(phases):
            g = result['green_times'][i]
            print(f"  {phase_name:>20}: {g:>5.1f} s")
        
        print(f"\n{'PERFORMANCE METRICS':^70}")
        print("-"*70)
        print(f"Average Delay: {result['avg_delay']:.2f} seconds/vehicle")
        print(f"\nDelay by Phase:")
        for phase, delay in result['delays'].items():
            print(f"  {phase:>10}: {delay:>6.2f} s/veh")
        
        print("\n" + "="*70 + "\n")


def main():
    """Example usage and validation."""
    import sys
    import os
    
    # Load config
    config_path = os.path.join(
        os.path.dirname(__file__), 
        '../config/config.yaml'
    )
    
    optimizer = WebsterSignalOptimizer(config_path=config_path)
    
    # Test different demand levels
    print("\n🚦 WEBSTER'S METHOD SIGNAL OPTIMIZATION")
    print("Testing different demand levels...\n")
    
    for dm in [0.5, 0.75, 1.0, 1.25, 1.5]:
        print(f"\n{'='*70}")
        print(f"DEMAND MULTIPLIER: {dm}")
        print(f"{'='*70}")
        
        result = optimizer.optimize(demand_multiplier=dm, verbose=True)
        
        if result['cycle_length'] is None:
            print(f"\n❌ FAILURE at demand multiplier {dm}")
            break


if __name__ == '__main__':
    main()
