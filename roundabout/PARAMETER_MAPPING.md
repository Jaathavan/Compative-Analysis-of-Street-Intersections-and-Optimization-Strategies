# Parameter Mapping: Text Simulation ↔ SUMO

This document maps parameters between the **text-based DDE simulation** (`Roundabout.py`) and the **SUMO implementation** to ensure comparable results.

---

## 🎯 Overview

The text simulation uses:
- **IDM (Intelligent Driver Model)** with **reaction delay τ** (DDE approach via history buffer)
- **Gap acceptance** with per-vehicle critical gap (lognormal) and follow-up headway (Gaussian)
- **Geometric speed limit** based on lateral acceleration

SUMO provides:
- Built-in **IDM car-following** (without explicit DDE delay)
- **Junction models** for gap acceptance and merging
- **Action step length** to approximate reaction delays

---

## 📊 Parameter Mapping Table

| **Concept** | **Text Sim (Roundabout.py)** | **SUMO Equivalent** | **Notes** |
|-------------|------------------------------|---------------------|-----------|
| **Geometry** | | | |
| Roundabout diameter | `diameter = 45.0` m | Network geometry (node positions) | Inscribed circle diameter |
| Circulating lanes | `lanes = 1` or `2` | `<edge numLanes="...">` | Number of ring lanes |
| Lane width | Implicit (3.5m) | `width="3.5"` | Standard lane width |
| Approach length | Implicit (~200m) | Edge length in `.net.xml` | Distance from spawn to entry |
| **Speed Limits** | | | |
| Ring speed cap | `v_max = sqrt(a_lat_max * R)` | `<edge speed="...">` | Derived from lateral accel limit |
| | `a_lat_max = 1.6 m/s²` | Computed: ~8.49 m/s for d=45m | Physics-based constraint |
| Desired speed | `v0_ring = 12.0 m/s` | `desiredMaxSpeed` in vType | Driver preference (may exceed limit) |
| **Car-Following (IDM)** | | | |
| Max acceleration | `a_max = 1.5 m/s²` | `accel="1.5"` | IDM parameter |
| Comfortable decel | `b_comf = 2.0 m/s²` | `decel="2.0"` | IDM parameter |
| Emergency decel | Implicit (2×b_comf) | `emergencyDecel="4.5"` | Hard braking limit |
| Accel exponent | `delta = 4` | Fixed in SUMO IDM | Shape of accel curve |
| Min gap | `s0 = 2.0 m` | `minGap="2.0"` | Standstill spacing |
| Time headway | `T = 1.2 s` | `tau="1.2"` | Desired following distance |
| **Reaction Delay** | | | |
| Reaction time | `tau = 0.8 s` | `actionStepLength="0.8"` | **Key difference** |
| | Used in DDE (history buffer) | SUMO delays action updates | Approximate equivalence |
| **Gap Acceptance** | | | |
| Critical gap (1st veh) | Lognormal: μ=3.0s, σ=0.6s | `jmTimegapMinor="3.0"` | Entry merging threshold |
| Follow-up headway | Gaussian: μ=2.0s, σ=0.3s | `jmDriveAfterRedTime="2.0"` | Platoon entry rate |
| | Per-vehicle draws | SUMO uses fixed + impatience | Stochastic vs deterministic |
| **Driver Variability** | | | |
| Speed preference | Fixed `v0 = 12.0 m/s` | `speedFactor ~ N(1.0, 0.1)` | SUMO adds speed variation |
| Imperfection | Implicit in draws | `sigma="0.5"` | Sub-optimal driving |
| **Demand** | | | |
| Arrival process | Poisson per arm (veh/s) | Flow definitions in `.rou.xml` | Exponential inter-arrival |
| Arrival rates | `[0.18, 0.12, 0.20, 0.15]` | `vehsPerHour` in flows | Scaled to veh/hr |
| Turning ratios | `[0.25, 0.55, 0.20]` (L/T/R) | Route probabilities | Distribution over exits |
| **Metrics** | | | |
| Entry delay | `t_enter_ring - t_queue_start` | `waitingTime` or custom | Time from queue join to merge |
| Queue length | `len(queues[i])` | Edge occupancy / stopped count | Vehicles waiting at entry |
| Throughput | `exits * 3600 / window_time` | Detector counts | Vehicles/hour through system |

---

## 🔄 Key Differences & Approximations

### 1. **Reaction Delay (τ)**
- **Text sim**: Implements true DDE by storing position/speed history and using state from `t - τ` for IDM calculations
- **SUMO**: Uses `actionStepLength` to delay control decisions, which approximates reaction time but isn't a true DDE
- **Impact**: SUMO may show slightly different oscillation damping and platoon behavior

**Mitigation**: Set `actionStepLength = tau` (0.8s) and validate against text sim under identical conditions

---

### 2. **Gap Acceptance Stochasticity**
- **Text sim**: Each vehicle draws unique critical gap (lognormal) and follow-up (Gaussian)
- **SUMO**: Uses fixed `jmTimegapMinor` + dynamic `impatience` parameter that grows with waiting time
- **Impact**: SUMO gap acceptance less variable initially, but becomes more aggressive over time

**Mitigation**: 
- Tune `jmTimegapMinor` to match mean critical gap
- Use `jmIgnoreKeepClearTime` to prevent blocking
- Acknowledge that exact platoon behavior will differ

---

### 3. **Speed Distribution**
- **Text sim**: All vehicles desire same `v0_ring = 12.0 m/s`
- **SUMO**: `speedFactor` adds heterogeneity (e.g., `N(1.0, 0.1)` → 10% variation)
- **Impact**: SUMO traffic more realistic but less deterministic

**Mitigation**: Can set `speedDev="0.0"` for deterministic comparison, or embrace added realism

---

### 4. **Lane Choice Logic**
- **Text sim**: Simple rule for 2-lane case (left turns → inner lane)
- **SUMO**: Complex lane-changing model (`LC2013` or `SL2015`)
- **Impact**: Multi-lane scenarios will differ significantly

**Mitigation**: Start with single-lane validation, document multi-lane as "SUMO-enhanced"

---

### 5. **Lateral Acceleration Limit**
- **Text sim**: Explicitly caps ring speed via `v_max = sqrt(a_lat_max * R)`
- **SUMO**: Set as edge `speed` attribute, but vehicles may try to exceed it
- **Impact**: Need to ensure SUMO respects geometric speed limit

**Mitigation**: Set `maxSpeed` in vType to computed limit, use `speedFactor < 1.0` if needed

---

## 🧪 Validation Strategy

### Phase 1: Baseline Replication
Run identical scenarios in both simulators:
- **Geometry**: 45m diameter, 1 lane
- **Demand**: [0.18, 0.12, 0.20, 0.15] veh/s, [0.25, 0.55, 0.20] turning
- **Duration**: 3600s (1 hour)
- **Seed**: 42 (for reproducibility)

**Compare**:
- Total throughput (exits)
- Mean and p95 entry delay
- Max queue lengths per arm
- Time-series of queue evolution

**Success criteria**: 
- Throughput within ±5%
- Mean delay within ±10%
- Qualitative queue behavior similar (no divergence in SUMO where text sim stable)

---

### Phase 2: Sensitivity Analysis
Vary one parameter at a time:
1. **Diameter**: 35m, 45m, 55m → Test speed cap impact
2. **Demand**: 0.5×, 1.0×, 1.5× base rates → Test capacity limits
3. **Gap acceptance**: Vary `jmTimegapMinor` 2.5–3.5s → Test entry behavior

**Goal**: Ensure both models show same trends (e.g., larger diameter → higher capacity)

---

### Phase 3: Multi-Lane Comparison
- Compare 1-lane vs 2-lane for same demand
- Document where SUMO's lane-changing causes divergence
- Use SUMO results as "enhanced realism" rather than strict validation

---

## 📐 SUMO Configuration Template

```xml
<!-- Vehicle Type with IDM + Junction Model -->
<vType id="passenger" 
       accel="1.5" 
       decel="2.0" 
       emergencyDecel="4.5"
       sigma="0.5" 
       length="5.0" 
       minGap="2.0" 
       maxSpeed="12.0"
       speedFactor="1.0" 
       speedDev="0.1"
       tau="1.2"
       carFollowModel="IDM"
       actionStepLength="0.8"
       jmTimegapMinor="3.0"
       jmDriveAfterRedTime="2.0"
       jmIgnoreKeepClearTime="5.0"
       lcStrategic="1.0"
       lcCooperative="1.0"/>
```

---

## 📊 Expected Discrepancies

| **Metric** | **Expected Difference** | **Reason** |
|------------|-------------------------|------------|
| Mean delay | ±10-15% | Stochastic gap acceptance vs fixed |
| p95 delay | ±15-20% | Tail behavior sensitive to model details |
| Throughput | ±5% | Capacity similar if speed limits match |
| Queue dynamics | Qualitative similarity | SUMO may show smoother platoons (no DDE) |
| Oscillations | SUMO less pronounced | Reaction delay approximation |

---

## 🎯 Deliverable Comparisons

### 1. Side-by-Side Tables
```
| Metric              | Text Sim | SUMO  | Δ (%)  |
|---------------------|----------|-------|--------|
| Throughput (veh/hr) | 1845     | 1912  | +3.6%  |
| Mean delay (s)      | 12.3     | 13.7  | +11.4% |
| p95 delay (s)       | 28.1     | 32.5  | +15.7% |
| Max queue (veh)     | 8        | 9     | +12.5% |
```

### 2. Overlay Plots
- Queue evolution over time (both models on same axes)
- Throughput vs demand curves
- Delay distributions (histograms)

### 3. Failure Point Analysis
- Identify demand levels where both models show instability
- Compare critical capacity estimates
- Document divergence conditions

---

## 🔧 Implementation Notes

### File: `compare_with_text_sim.py`
This script should:
1. Run `Roundabout.py` with specified parameters → capture output
2. Run SUMO simulation with equivalent parameters → collect metrics
3. Parse both outputs into comparable DataFrames
4. Generate comparison tables and plots
5. Flag discrepancies exceeding thresholds

### Usage Example
```bash
python src/compare_with_text_sim.py \
  --diameter 45 \
  --lanes 1 \
  --demand 0.18 0.12 0.20 0.15 \
  --output results/comparison_baseline.csv
```

---

## 📚 References

1. **IDM Model**: Treiber, M., & Kesting, A. (2013). *Traffic Flow Dynamics*
2. **SUMO Documentation**: https://sumo.dlr.de/docs/
3. **Gap Acceptance**: Brilon, W., et al. (1997). *Useful estimation procedures for critical gaps*
4. **DDE in Traffic**: Orosz, G., et al. (2009). *Traffic jams: dynamics and control*

---

*Last updated: [Auto-generated during Phase 1 implementation]*
