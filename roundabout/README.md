# Phase 1: Roundabout Simulation & Optimization

## 🎯 Overview

This directory contains the complete SUMO-based roundabout simulation pipeline that replicates and extends the functionality of the text-based `Roundabout.py` microsimulation.

---

## 📁 Directory Structure

```
/roundabout/
├── config/
│   ├── config.yaml              # Central configuration (all parameters)
│   └── templates/               # XML templates (if needed)
├── src/
│   ├── generate_network.py      # Creates .net.xml files
│   ├── generate_routes.py       # Creates .rou.xml and .sumocfg files
│   ├── run_simulation.py        # Runs SUMO via TraCI, collects metrics
│   ├── analyze_results.py       # Post-processes results
│   ├── visualize_results.py     # Generates plots
│   ├── compare_with_text_sim.py # Compares SUMO vs text simulation
│   └── optimize.py              # Parameter sweep orchestration
├── results/
│   ├── raw/                     # Per-scenario CSV files
│   ├── plots/                   # Generated visualizations
│   └── summary.csv              # Aggregated sweep results
├── sumo_configs/                # Generated SUMO files per scenario
├── PARAMETER_MAPPING.md         # Parameter equivalence documentation
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Prerequisites

1. **SUMO Installation**: Install SUMO and set `SUMO_HOME` environment variable
   ```bash
   export SUMO_HOME="/usr/share/sumo"  # Adjust path as needed
   ```

2. **Python Dependencies**:
   ```bash
   pip install pyyaml pandas numpy matplotlib seaborn plotly
   ```

### Basic Workflow

#### 1. Generate Network
Create a roundabout network from parameters:

```bash
cd roundabout
python src/generate_network.py \
  --config config/config.yaml \
  --output sumo_configs/baseline
```

**Output**: `sumo_configs/baseline/roundabout.net.xml`

#### 2. Generate Routes
Create traffic demand matching text simulation:

```bash
python src/generate_routes.py \
  --config config/config.yaml \
  --network sumo_configs/baseline/roundabout.net.xml \
  --output sumo_configs/baseline
```

**Output**: 
- `sumo_configs/baseline/roundabout.rou.xml`
- `sumo_configs/baseline/roundabout.sumocfg`

#### 3. Run Simulation
Execute simulation and collect metrics:

```bash
# Headless mode (for batch processing)
python src/run_simulation.py \
  --sumocfg sumo_configs/baseline/roundabout.sumocfg \
  --config config/config.yaml \
  --output results/raw/baseline.csv

# GUI mode (for visualization/debugging)
python src/run_simulation.py \
  --sumocfg sumo_configs/baseline/roundabout.sumocfg \
  --config config/config.yaml \
  --gui \
  --output results/raw/baseline.csv
```

**Output**:
- `results/raw/baseline.csv` (window metrics)
- `results/raw/baseline_aggregate.csv` (hourly summary)

#### 4. Analyze Results
Compute derived metrics and detect failures:

```bash
python src/analyze_results.py \
  --input results/raw/baseline.csv \
  --config config/config.yaml \
  --output results/summary.csv
```

#### 5. Visualize
Generate plots (static and interactive):

```bash
python src/visualize_results.py \
  --input results/summary.csv \
  --config config/config.yaml \
  --output results/plots/
```

#### 6. Compare with Text Simulation
Side-by-side comparison:

```bash
python src/compare_with_text_sim.py \
  --sumo-results results/raw/baseline.csv \
  --text-sim-params diameter=45,lanes=1,demand=1.0 \
  --output results/comparison.csv
```

---

## ⚙️ Configuration

All parameters are defined in `config/config.yaml`. Key sections:

### Geometry
```yaml
geometry:
  diameter: 45.0        # Roundabout diameter (m)
  lanes: 1              # Circulating lanes (1 or 2)
  approach_length: 200.0
```

### Demand
```yaml
demand:
  arrivals: [0.18, 0.12, 0.20, 0.15]  # Per-arm rates (veh/s)
  turning_probabilities: [0.25, 0.55, 0.20]  # L/T/R
```

### Driver Behavior
```yaml
driver:
  accel: 1.5            # Max acceleration (m/s²)
  decel: 2.0            # Comfortable deceleration
  tau: 1.2              # Desired time headway (s)
  max_speed: 12.0       # Desired free-flow speed (m/s)
```

See `config/config.yaml` for complete parameter list and documentation.

---

## 🔬 Parameter Sweeps

### Manual Sweep
Run multiple scenarios with different parameters:

```bash
# Vary diameter
for diam in 35 45 55; do
  python src/generate_network.py --config config/config.yaml --diameter $diam --output sumo_configs/diam_${diam}
  python src/generate_routes.py --config config/config.yaml --network sumo_configs/diam_${diam}/roundabout.net.xml --output sumo_configs/diam_${diam}
  python src/run_simulation.py --sumocfg sumo_configs/diam_${diam}/roundabout.sumocfg --config config/config.yaml --output results/raw/diam_${diam}.csv
done

# Analyze all results
python src/analyze_results.py --sweep "results/raw/diam_*.csv" --output results/diameter_sweep.csv
```

### Automated Sweep (via optimize.py)
```bash
python src/optimize.py \
  --config config/config.yaml \
  --sweep-params diameter,demand_multiplier \
  --output results/sweep_results/
```

This orchestrates the full pipeline for all parameter combinations defined in `config.yaml`.

---

## 📊 Metrics Collected

### Window Metrics (5-minute intervals)
- `arrivals`, `exits`: Vehicle counts
- `throughput_vph`: Vehicles/hour
- `avg_delay`, `p95_delay`: Entry delays (seconds)
- `max_queue_{N,E,S,W}`: Per-arm max queue lengths
- `avg_speed_ring`: Average speed in roundabout (m/s)
- `total_co2`, `total_fuel`: Emissions and fuel (SUMO-specific)

### Aggregate Metrics (hourly)
- `total_arrivals`, `total_exits`
- `mean_delay`, `p95_delay`
- `throughput_vph`
- `max_queue_{N,E,S,W}`: All-time maxima
- `avg_travel_time`: Mean trip duration

---

## 🎨 Visualization Examples

The `visualize_results.py` script generates:

### Static Plots (Matplotlib/Seaborn)
- **Throughput vs Demand**: Line plot showing capacity curves
- **Delay vs Demand**: Scatter plot with failure threshold
- **Queue Heatmap**: Per-arm queue lengths across scenarios
- **Speed Distribution**: Histogram of ring speeds
- **Failure Boundary**: Parameter space with failure regions

### Interactive Plots (Plotly)
- **3D Performance Surface**: Diameter × Demand × Delay
- **Time Series Animation**: Queue evolution over simulation
- **Parameter Explorer**: Interactive dashboard for scenario comparison

---

## 🔍 Comparison with Text Simulation

See `PARAMETER_MAPPING.md` for detailed parameter equivalences.

### Expected Discrepancies

| Metric | Expected Δ | Reason |
|--------|-----------|--------|
| Mean delay | ±10-15% | Stochastic gap acceptance vs fixed |
| p95 delay | ±15-20% | Tail behavior sensitivity |
| Throughput | ±5% | Similar capacity if speed limits match |

### Validation Approach
1. **Baseline replication**: Run identical parameters in both simulators
2. **Sensitivity analysis**: Vary one parameter at a time
3. **Multi-lane comparison**: Document divergence due to lane-changing models

---

## 🧪 Example Scenarios

### Baseline (Text Simulation Match)
```bash
python src/generate_network.py --config config/config.yaml --output sumo_configs/baseline
python src/generate_routes.py --config config/config.yaml --network sumo_configs/baseline/roundabout.net.xml --output sumo_configs/baseline
python src/run_simulation.py --sumocfg sumo_configs/baseline/roundabout.sumocfg --config config/config.yaml --output results/raw/baseline.csv
```

**Expected results** (baseline: d=45m, 1 lane, demand=1.0×):
- Throughput: ~1900 veh/hr
- Mean delay: ~12-15s
- Max queue: ~8-10 vehicles

### High Demand (1.5× baseline)
```bash
python src/generate_routes.py --config config/config.yaml --network sumo_configs/baseline/roundabout.net.xml --demand-multiplier 1.5 --output sumo_configs/high_demand
python src/run_simulation.py --sumocfg sumo_configs/high_demand/roundabout.sumocfg --config config/config.yaml --output results/raw/high_demand.csv
```

**Expected**: Potential failure (queue divergence, excessive delays)

### Large Roundabout (55m diameter)
```bash
python src/generate_network.py --config config/config.yaml --diameter 55 --output sumo_configs/large
python src/generate_routes.py --config config/config.yaml --network sumo_configs/large/roundabout.net.xml --output sumo_configs/large
python src/run_simulation.py --sumocfg sumo_configs/large/roundabout.sumocfg --config config/config.yaml --output results/raw/large.csv
```

**Expected**: Higher capacity (lower delays) due to higher speed limit

---

## 🐛 Troubleshooting

### SUMO not found
```
Error: Please declare environment variable 'SUMO_HOME'
```
**Solution**: Set environment variable pointing to SUMO installation:
```bash
export SUMO_HOME="/usr/share/sumo"
```

### TraCI connection errors
```
Error: Could not connect to TraCI server
```
**Solution**: Ensure SUMO binary is in PATH and no other instance is running on port 8813

### Empty results
```
Warning: No vehicles exited during simulation
```
**Solution**: Check route definitions (approach → ring → exit edges must be connected)

### Parameter mismatch warnings
**Solution**: Review `PARAMETER_MAPPING.md` for SUMO ↔ text sim equivalences

---

## 📚 References

- **SUMO Documentation**: https://sumo.dlr.de/docs/
- **TraCI API**: https://sumo.dlr.de/docs/TraCI.html
- **IDM Model**: Treiber & Kesting, *Traffic Flow Dynamics* (2013)
- **Gap Acceptance**: Brilon et al., *Useful estimation procedures for critical gaps* (1997)

---

## 🎯 Next Steps

1. **Run baseline scenarios** to validate against text simulation
2. **Execute parameter sweeps** to identify optimal configurations
3. **Analyze failure points** (capacity limits, geometric constraints)
4. **Generate visualizations** for report and presentations
5. **Document findings** in comparison tables and plots

For Phase 2 (signalized intersections), the structure can be replicated in `/signalized/` with intersection-specific parameters.

---

*Last updated: Phase 1 Implementation*
