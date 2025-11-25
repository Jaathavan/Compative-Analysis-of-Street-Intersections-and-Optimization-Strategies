# Enhanced Roundabout & Signalized Intersection Analysis

## Overview

This document describes the enhanced analysis tools and text-based simulations created for comprehensive comparison of roundabout and signalized intersection performance.

## New Components

### 1. Text-Based Signalized Intersection Simulation (`Signalized.py`)

A microsimulation matching the structure of `Roundabout.py` for direct comparison.

**Features:**
- Webster's Method for optimal fixed-time signal control
- IDM car-following model with reaction delay
- Multi-lane support (1-3 lanes per approach)
- Poisson arrival process
- 4-phase signal operation (NS-Left, NS-Through, EW-Left, EW-Through)
- Realistic queuing and discharge

**Usage:**
```bash
# Basic run with default parameters
python Signalized.py

# Custom configuration
python Signalized.py --lanes 2 --arrival 0.18 0.12 0.20 0.15 --cycle-length 90

# High demand scenario
python Signalized.py --lanes 3 --arrival 0.20 0.15 0.20 0.15 --horizon 3600

# Manual timing (disable Webster's)
python Signalized.py --cycle-length 120 --no-use-webster
```

**Webster's Method Implementation:**
- Computes optimal cycle length: `C_opt = (1.5L + 5) / (1 - Y)`
- Allocates green times proportional to flow ratios
- Accounts for startup lost time and clearance intervals
- Validates against oversaturation (Y < 1.0)

**Example Output:**
```
Webster's Method: Y=0.700, L=12.0s, C_opt=96.7s
Green times: NS-L=11.5s, NS-T=34.6s, EW-L=8.6s, EW-T=25.9s

[05:00] arrivals=284 exits=271 throughput=1955 veh/hr  avg_delay=12.3s  p95=35.2s  max_q=[8, 6, 10, 7]
[10:00] arrivals=291 exits=289 throughput=2080 veh/hr  avg_delay=14.1s  p95=38.7s  max_q=[9, 7, 11, 8]

=== SIMULATION SUMMARY ===
  Total arrivals: 2340
  Total departures: 2298
  Throughput: 2298 veh/hr
  Average delay: 13.5s
  P95 delay: 37.2s
  Max queue per arm [N,E,S,W]: [12, 9, 15, 11]
```

### 2. Enhanced Visualization Suite (`roundabout/src/enhanced_visualizations.py`)

Comprehensive analysis tools inspired by `visualizations.txt` but significantly expanded.

**Visualization Modules:**

#### 2.1 Lane Choice Analysis
- Lane utilization distribution vs arrival rate
- Delay by lane choice (boxplots)
- Per-lane throughput contributions
- Lane efficiency scores
- Queue balance metrics
- Lane-specific merge denial rates

**Command:**
```bash
python enhanced_visualizations.py --data results/multilane_sweep.csv --mode lane_analysis
```

#### 2.2 Parameter Sweep Analysis
- Throughput/delay/queue vs arrival rate (multi-lane comparison)
- Breaking point identification and visualization
- Throughput/delay/queue vs diameter (fixed arrival)
- Heatmaps (diameter × lanes for throughput and delay)
- Breaking point curves

**Command:**
```bash
python enhanced_visualizations.py --data results/parameter_sweep.csv --mode parameter_sweep
```

#### 2.3 Optimization Results Visualization
- 3D surface plots for grid search
- Bayesian optimization convergence curves
- Optimal parameter distributions
- Pareto fronts (throughput vs delay)
- Configuration comparison tables

**Command:**
```bash
python enhanced_visualizations.py \
  --data results/grid_search.csv \
  --mode optimization \
  --optimization-data results/bayesian_opt.json
```

#### 2.4 Failure Mode Demonstrations
- Queue divergence time series
- Throughput saturation curves
- Delay explosion (log scale)
- Service time distributions (stable vs failing)
- Queue length distributions
- Failure mode classification

**Command:**
```bash
python enhanced_visualizations.py --data results/failure_scenarios.csv --mode failure_modes
```

#### 2.5 Comprehensive Comparison
- All key metrics in a single 12-panel figure
- Performance vs arrival rate (4 metrics)
- Performance vs diameter (4 metrics)
- Aggregate analysis (efficiency, capacity utilization, queue balance)
- Summary statistics table

**Command:**
```bash
python enhanced_visualizations.py --data results/full_sweep.csv --mode comprehensive
```

**Generate All Visualizations:**
```bash
python enhanced_visualizations.py --data results/complete_data.csv --mode all --output plots/
```

### 3. Alignment Verification Tool (`roundabout/src/verify_alignment.py`)

Validates consistency between text-based (`Roundabout.py`) and SUMO simulations.

**Features:**
- Automated execution of both simulators
- Side-by-side metric comparison
- Relative difference computation
- Visual comparison plots
- Alignment quality assessment

**Usage:**
```bash
# Run default validation scenarios
python verify_alignment.py --config config/config.yaml --output results/alignment/

# Custom scenarios from JSON
python verify_alignment.py --scenarios custom_scenarios.json --output results/alignment/
```

**Scenario Format (JSON):**
```json
[
  {
    "name": "baseline_2lane",
    "params": {
      "lanes": 2,
      "diameter": 45,
      "arrival": [0.10, 0.10, 0.10, 0.10],
      "turning": [0.25, 0.55, 0.20],
      "seed": 42,
      "horizon": 1800
    }
  }
]
```

**Output:**
- `alignment_comparison.json` - Detailed results
- `alignment_comparison.csv` - Tabular format
- `alignment_comparison.png` - 6-panel visualization

**Validation Criteria:**
- **Throughput**: Difference < 10% (good), < 20% (acceptable)
- **Average Delay**: Difference < 15% (good), < 30% (acceptable)
- **P95 Delay**: Difference < 20% (good), < 40% (acceptable)

### 4. Failure Video Generator (`roundabout/src/generate_failure_videos.py`)

Creates slowed-down demonstration videos of failure scenarios.

**Features:**
- Automatic SUMO-GUI launch with optimal settings
- Slowed playback for visibility
- Critical arrival rate scenarios (1, 2, 3 lanes)
- Queue buildup and congestion visualization

**Usage:**
```bash
# Generate all failure videos
python generate_failure_videos.py --config config/config.yaml --output videos/

# Custom speed and duration
python generate_failure_videos.py --speed 0.1 --duration 600 --output videos/
```

**Default Scenarios:**
1. **1-lane, λ=0.12 veh/s**: Queue divergence (oversaturated)
2. **2-lane, λ=0.15 veh/s**: Near saturation
3. **3-lane, λ=0.18 veh/s**: High but stable (contrast)

**Video Recording Options:**

**Option A: Screen Recording (Recommended)**
1. Script launches SUMO-GUI with slowed playback
2. Manually record using:
   - OBS Studio (free, cross-platform)
   - macOS: QuickTime Player
   - Windows: Game Bar (Win+G)
   - Linux: SimpleScreenRecorder

**Option B: SUMO Built-in (if available)**
```bash
sumo-gui -c scenario/roundabout.sumocfg \
  --gui-settings-file gui_settings.xml \
  --start --quit-on-end \
  --video-encoding png \
  --video-output video.mp4
```

**Option C: Screenshot Compilation**
```bash
# After capturing screenshots
ffmpeg -framerate 30 -i screenshot_%04d.png \
       -c:v libx264 -pix_fmt yuv420p output.mp4
```

**Output:**
- Network/route files for each scenario
- `gui_settings.xml` - Optimal visualization settings
- `README_VIDEOS.md` - Detailed instructions

## Workflow Examples

### Complete Roundabout Analysis

```bash
# 1. Run parameter sweep (text simulation)
for lanes in 1 2 3; do
  for arrival in 0.05 0.07 0.10 0.12 0.15; do
    python ../Roundabout.py --lanes $lanes \
      --arrival $arrival $arrival $arrival $arrival \
      --horizon 3600 >> results/raw/sweep_text_${lanes}lane.log
  done
done

# 2. Run SUMO parameter sweep
python src/optimize.py --mode grid_search --output results/raw/sweep_sumo.csv

# 3. Verify alignment
python src/verify_alignment.py --output results/alignment/

# 4. Generate visualizations
python src/enhanced_visualizations.py \
  --data results/raw/sweep_sumo.csv \
  --mode all \
  --output results/plots/

# 5. Create failure videos
python src/generate_failure_videos.py \
  --speed 0.1 \
  --duration 600 \
  --output videos/
```

### Signalized vs Roundabout Comparison

```bash
# 1. Run roundabout (baseline)
python Roundabout.py --lanes 2 --diameter 45 \
  --arrival 0.18 0.12 0.20 0.15 \
  --horizon 3600 > results/roundabout_baseline.log

# 2. Run signalized (baseline)
python Signalized.py --lanes 2 \
  --arrival 0.18 0.12 0.20 0.15 \
  --horizon 3600 > results/signalized_baseline.log

# 3. Compare SUMO versions
cd roundabout
python src/run_simulation.py \
  --sumocfg sumo_configs/baseline/roundabout.sumocfg \
  --output results/sumo_roundabout.csv

cd ../signalized
python src/run_simulation.py \
  --config config/config.yaml \
  --strategy webster \
  --output results/sumo_signalized.csv

# 4. Visualize comparison
python compare_intersections.py \
  --roundabout results/sumo_roundabout.csv \
  --signalized results/sumo_signalized.csv \
  --output results/comparison_plots/
```

### Optimization Study

```bash
# 1. Grid search (roundabout)
cd roundabout
python src/optimize.py --mode grid_search \
  --diameter-range 30 35 40 45 50 55 \
  --lanes-range 1 2 3 \
  --demand-multipliers 0.5 0.75 1.0 1.25 1.5 \
  --output results/grid_search.csv

# 2. Bayesian optimization
python src/optimize.py --mode bayesian \
  --n-iterations 100 \
  --output results/bayesian_opt.json

# 3. Visualize optimization results
python src/enhanced_visualizations.py \
  --data results/grid_search.csv \
  --mode optimization \
  --optimization-data results/bayesian_opt.json \
  --output results/plots/

# 4. Verify optimal configuration
python ../Roundabout.py --lanes 2 --diameter 42.5 \
  --arrival 0.15 0.15 0.15 0.15 \
  --horizon 3600
```

## Data Format Specifications

### Simulation Results CSV

Required columns for `enhanced_visualizations.py`:

```csv
scenario,lanes,diameter,arrival_rate,throughput,avg_delay,p95_delay,max_queue_N,max_queue_E,max_queue_S,max_queue_W
baseline_1L,1,45,0.10,721,23.8,115.6,7,6,12,12
baseline_2L,2,45,0.10,1414,26.0,131.2,14,27,19,14
...
```

**Optional columns for lane analysis:**
- `lane_0_entries`, `lane_1_entries`, `lane_2_entries`
- `lane_0_delay`, `lane_1_delay`, `lane_2_delay`
- `lane_0_denials`, `lane_0_attempts`

### Optimization Results JSON

```json
{
  "grid_search": [
    {
      "diameter": 45,
      "lanes": 2,
      "arrival_rate": 0.10,
      "objective": 26.0,
      "throughput": 1414,
      "avg_delay": 26.0,
      "p95_delay": 131.2
    }
  ],
  "bayesian_opt": [
    {
      "iteration": 1,
      "diameter": 42.3,
      "lanes": 2,
      "objective": 18.5
    }
  ]
}
```

## Parameter Mapping

### Text Simulation ↔ SUMO

| Concept | Roundabout.py | SUMO Parameter | Notes |
|---------|---------------|----------------|-------|
| Max accel | `a_max=1.5` | `accel="1.5"` | m/s² |
| Comfort decel | `b_comf=2.0` | `decel="2.0"` | m/s² |
| Time headway | `T=1.2` | `tau="1.2"` | seconds |
| Min gap | `s0=2.0` | `minGap="2.0"` | meters |
| Reaction delay | `tau=0.8` (DDE) | `actionStepLength="0.8"` | seconds |
| Desired speed | `v0_ring=12.0` | `maxSpeed="12.0"` | m/s |
| Critical gap | `crit_gap_mean=3.0` | `jmTimegapMinor="3.0"` | seconds |
| Follow-up | `followup_mean=2.0` | `jmDriveAfterRed="2.0"` | seconds |

### Signalized.py ↔ SUMO Signalized

| Concept | Signalized.py | SUMO Parameter | Notes |
|---------|---------------|----------------|-------|
| Cycle length | `cycle_length` | `<phase>` sum | seconds |
| Green times | `green_times[]` | `duration` in `<phase>` | seconds |
| Yellow time | `yellow_time=3.0` | `duration` (yellow phase) | seconds |
| All-red | `all_red_time=1.0` | `duration` (all-red phase) | seconds |
| Saturation flow | `saturation_flow=1800` | Implicit in SUMO | veh/hr/lane |

## Performance Benchmarks

### Typical Execution Times (3600s simulation)

| Tool | Configuration | Runtime | Output |
|------|---------------|---------|--------|
| Roundabout.py | 2 lanes, λ=0.10 | ~30s | Terminal log |
| Signalized.py | 2 lanes, λ=0.10 | ~25s | Terminal log |
| SUMO roundabout | 2 lanes, headless | ~15s | CSV metrics |
| SUMO signalized | 2 lanes, headless | ~12s | CSV metrics |
| Parameter sweep (9 configs) | Grid search | ~5 min | CSV database |
| Bayesian optimization (100 iter) | With SUMO evals | ~2 hours | JSON results |
| Visualization suite | All modes | ~30s | PNG plots |

## Troubleshooting

### Issue: SUMO simulations run too fast

**Symptom:** SUMO-GUI playback is too fast to observe behavior

**Solution:**
```bash
# Use generate_failure_videos.py with slowed playback
python generate_failure_videos.py --speed 0.1  # 10x slower

# Or manually adjust GUI delay
sumo-gui -c config.sumocfg --delay 100  # 100ms per step
```

### Issue: Text vs SUMO alignment poor

**Symptom:** Metrics differ by >20%

**Diagnosis:**
```bash
# Run verification tool
python verify_alignment.py --output results/alignment/

# Check alignment report
cat results/alignment/alignment_comparison.csv
```

**Common causes:**
1. **Parameter mismatch**: Verify config.yaml matches Roundabout.py defaults
2. **Random seed**: Ensure both use same seed
3. **Warmup period**: SUMO may need warmup time excluded
4. **Network topology**: Check diameter, lane count consistency

**Fix:**
```yaml
# In config.yaml
driver:
  accel: 1.5      # Must match Roundabout.py a_max
  decel: 2.0      # Must match b_comf
  tau: 1.2        # Must match T
  
geometry:
  diameter: 45.0  # Must match --diameter
  lanes: 2        # Must match --lanes
```

### Issue: Visualizations fail with missing columns

**Symptom:** `KeyError` or empty plots

**Solution:**
Ensure CSV has required columns. Generate template:
```python
import pandas as pd

# Minimum required columns
df = pd.DataFrame({
    'scenario': ['test'],
    'lanes': [2],
    'diameter': [45],
    'arrival_rate': [0.10],
    'throughput': [1400],
    'avg_delay': [25],
    'p95_delay': [130],
    'max_queue_N': [10],
    'max_queue_E': [15],
    'max_queue_S': [12],
    'max_queue_W': [14],
})

df.to_csv('template.csv', index=False)
```

### Issue: Webster's Method produces oversaturation warning

**Symptom:** `Y >= 1.0` warning in Signalized.py

**Explanation:** Total demand exceeds capacity

**Solution:**
```bash
# Reduce arrival rates
python Signalized.py --arrival 0.15 0.12 0.15 0.12  # Instead of 0.20

# Or increase lanes
python Signalized.py --lanes 3 --arrival 0.18 0.15 0.18 0.15
```

## References

1. **Webster's Method**: Webster, F. V. (1958). "Traffic Signal Settings." Road Research Technical Paper No. 39.

2. **IDM**: Treiber, M., Hennecke, A., & Helbing, D. (2000). "Congested traffic states in empirical observations and microscopic simulations."

3. **Gap Acceptance**: Troutbeck, R. J. (1989). "Evaluating the Performance of a Roundabout."

4. **SUMO Documentation**: https://sumo.dlr.de/docs/

## Future Enhancements

Potential additions for future work:

1. **Adaptive Signal Control**: SCATS/SCOOT implementations
2. **Connected Vehicle Integration**: V2I communication
3. **Real-time Optimization**: Online learning for signal timing
4. **Multi-objective Analysis**: Pareto optimization for emissions, delay, throughput
5. **Stochastic Weather Effects**: Rain/snow impact on capacity
6. **Incident Simulation**: Lane blockage scenarios
7. **Mixed Autonomy**: CAV penetration rate studies

---

**Author**: Enhanced Analysis Suite  
**Date**: 2024  
**Version**: 1.0  
**License**: MIT
