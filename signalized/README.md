# Phase 2: Signalized Intersection with Webster's Method & PPO

## Overview

This phase implements a 4-way signalized intersection in SUMO with two control strategies:
1. **Webster's Method**: Analytical optimization for fixed-time signal control
2. **PPO (Proximal Policy Optimization)**: Reinforcement learning for adaptive signal control

## Project Structure

```
signalized/
├── config/
│   └── config.yaml              # Main configuration
├── src/
│   ├── webster_method.py        # Webster's analytical optimization ✅
│   ├── generate_network.py      # SUMO network generator ✅
│   ├── generate_routes.py       # Traffic demand generator ✅
│   ├── run_simulation.py        # SUMO simulation runner ✅
│   ├── ppo_environment.py       # Gymnasium environment for RL ✅
│   └── train_ppo.py             # PPO training script ✅
├── models/                      # Trained PPO models
├── results/                     # Simulation outputs
└── sumo_configs/                # Generated SUMO files
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure SUMO is installed and in your PATH:
```bash
sumo --version
```

### 2. Test Webster's Method

```bash
cd src
python webster_method.py
```

This will calculate optimal signal timing for different demand levels using Webster's analytical method.

### 3. Generate SUMO Network

```bash
python generate_network.py
```

This creates:
- Intersection topology (.nod.xml, .edg.xml)
- Traffic light logic (.tll.xml)
- Compiled network (.net.xml)

### 4. Generate Traffic Demand

```bash
python generate_routes.py
```

Creates route files with Poisson arrivals matching roundabout simulation demand.

### 5. Run Webster-Optimized Simulation

```bash
python run_simulation.py --control webster --demand-multiplier 1.0
```

### 6. Train PPO Agent

```bash
python train_ppo.py --timesteps 500000
```

### 7. Test PPO Agent

```bash
python run_simulation.py --control ppo --model models/ppo_signal_best.zip --demand-multiplier 1.0
```

### 8. Compare All Strategies

```bash
python compare_strategies.py --strategies webster ppo actuated --output results/comparison.csv
```

## Webster's Method

Webster's Method (1958) provides analytical formulas for optimal signal timing:

### Optimal Cycle Length
```
C_opt = (1.5L + 5) / (1 - Y)
```

Where:
- `L`: Total lost time per cycle (startup + clearance)
- `Y`: Sum of critical flow ratios

### Green Time Allocation
```
g_i = (y_i / Y) × (C - L)
```

Where:
- `g_i`: Green time for phase i
- `y_i`: Flow ratio for phase i (demand/capacity)

### Delay Formula
```
d_i = C(1 - λ_i)² / (2(1 - y_i)) + x_i² / (2q_i(1 - x_i))
```

Components:
- **Uniform delay**: Deterministic component
- **Overflow delay**: Stochastic component from random arrivals

## PPO Optimization

### State Space
- Queue length per approach (4 values)
- Average waiting time per approach (4 values)
- Recent throughput per approach (4 values)
- Current phase elapsed time (1 value)
- Time of day (cyclical encoding: sin, cos)

### Action Space
- Adjust green time for each phase: {-5s, 0s, +5s}
- Subject to constraints: 10s ≤ green ≤ 90s

### Reward Function
```python
reward = (
    1.0 * throughput 
    - 0.5 * avg_delay
    - 0.3 * max_queue
    - 0.1 * total_stops
    + 0.2 * fairness_bonus
)
```

### Training
- Total timesteps: 500,000
- Algorithm: PPO (Proximal Policy Optimization)
- Framework: Stable-Baselines3
- Evaluation: Every 10,000 steps with best model saving

## Performance Comparison

| Metric | Webster (Fixed) | PPO (Adaptive) | Improvement |
|--------|----------------|----------------|-------------|
| Throughput (veh/hr) | TBD | TBD | TBD |
| Avg Delay (s) | TBD | TBD | TBD |
| Max Queue (veh) | TBD | TBD | TBD |
| CO₂ Emissions (mg) | TBD | TBD | TBD |

## Configuration

Key parameters in `config/config.yaml`:

### Signal Timing (Webster)
```yaml
signal:
  webster:
    saturation_flow: 1800.0    # veh/hr/lane
    startup_lost_time: 2.0     # seconds
    clearance_lost_time: 2.0   # seconds
    yellow_time: 3.0           # seconds
    all_red_time: 2.0          # seconds
```

### PPO Training
```yaml
signal:
  ppo:
    min_green_time: 10.0       # seconds
    max_green_time: 90.0       # seconds
    learning_rate: 0.0003
    total_timesteps: 500000
```

### Demand (Same as Roundabout)
```yaml
demand:
  arrivals: [0.18, 0.12, 0.20, 0.15]  # veh/s per arm [N,E,S,W]
  turning_probabilities: [0.25, 0.55, 0.20]  # [L, T, R]
```

## Comparison with Roundabout

This simulation uses **identical demand patterns** as the roundabout simulation from Phase 1, enabling direct comparison:

- Same arrival rates per arm
- Same turning movement distributions
- Same driver behavior parameters (IDM)
- Same vehicle type mix
- Same performance metrics

## References

1. Webster, F. V. (1958). *Traffic signal settings*. Road Research Technical Paper No. 39.
2. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
3. Transportation Research Board (2010). *Highway Capacity Manual 2010*.

## Next Steps

1. ✅ Implement Webster's Method
2. ✅ Generate SUMO network
3. ✅ Generate traffic demand
4. 🔄 Implement simulation runner
5. 🔄 Implement PPO environment
6. 🔄 Train PPO agent
7. 🔄 Compare strategies
8. 🔄 Compare with roundabout results

## Contact

For questions or issues, please refer to the main project repository.
