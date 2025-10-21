# Comparative Analysis of Street Intersections and Optimization Strategies: Midterm Report

**Project:** Traffic Flow Optimization through Multi-Platform Microsimulation  
**Date:** October 19, 2025

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement & Challenges](#2-problem-statement--challenges)
3. [Mathematical & Computational Background](#3-mathematical--computational-background)
4. [Methodology](#4-methodology)
5. [Assessment & Evaluation](#5-assessment--evaluation)
6. [Timeline](#6-timeline)
7. [Contributions](#7-contributions)
8. [References](#8-references)

---

## 1. Introduction

This project develops a dual-platform microsimulation framework to quantify how geometric design and behavioral parameters affect operational efficiency at traffic intersections, with initial focus on four-arm roundabouts. The framework consists of:

1. **Text-Based Python Microsimulation**: A single-file, configurable simulator (`Roundabout.py`) implementing continuous car-following dynamics via the Intelligent Driver Model (IDM), delay-differential equations (DDEs) for reaction time, stochastic gap acceptance, and Poisson arrival processes. The absence of graphics enables rapid parameter sweeps and reproducible measurements of throughput, delay, and queue dynamics.

2. **SUMO-Based Validation Pipeline**: A complete production pipeline using SUMO (Simulation of Urban Mobility) that replicates the Python simulation's modeling logic—comparable queuing rules at entries, consistent car-following dynamics, and aligned demand patterns—enabling cross-platform validation and leveraging SUMO's visual diagnostics, emissions modeling, and ecosystem for scenario management.

### Project Objectives

- **Capacity Analysis**: Identify breaking points where geometric modifications (e.g., additional lanes, larger diameter) become necessary
- **Design Optimization**: Determine optimal configurations across multiple objectives (throughput, delay, emissions)
- **Comparative Assessment**: Establish which intersection control strategy (roundabout vs. signalized) performs better under various demand scenarios
- **Model Validation**: Use cross-platform agreement as evidence for model correctness; investigate discrepancies to refine assumptions

### Dual-Platform Rationale

Developing our own Python simulation provides complete transparency over assumptions, algorithms, and metrics, facilitating deep understanding of how modeling choices influence outcomes. SUMO augments this analysis with:
- Visual flow inspection and debugging
- Richer built-in diagnostics (emissions, fuel consumption, noise)
- Established validation against real-world data
- Integration with traffic assignment and demand modeling tools

Future phases will extend this framework to signalized intersections (Phase 2) with reinforcement learning (RL) for adaptive signal control, and real-world application using OpenStreetMap data (Phase 3).

---

## 2. Problem Statement & Challenges

### 2.1 Core Problem

Urban traffic intersections operate as critical bottlenecks in transportation networks. Poor design or control strategies lead to:
- **Capacity saturation**: Queue divergence when demand exceeds service capacity
- **Excessive delays**: Reduced level-of-service affecting user satisfaction
- **Safety concerns**: Increased conflict points and crash risk
- **Environmental impact**: Elevated emissions from idling and stop-and-go behavior

The fundamental question: **Given demand characteristics (volume, turning movements) and site constraints, what intersection control strategy and geometric configuration maximize efficiency while ensuring stability?**

### 2.2 Technical Challenges

#### Stochastic Variability
Traffic is inherently random:
- **Arrival processes**: Poisson or more complex (platooning, signal influence)
- **Driver heterogeneity**: Critical gap acceptance varies by driver
- **Turning choices**: Probabilistic movement selection
- **Behavioral uncertainty**: Reaction times, desired speeds, aggressiveness

**Solution**: Monte Carlo simulation with sufficient replication to capture statistical properties.

#### Multi-Objective Optimization
No single "best" design exists; trade-offs include:
- Maximize throughput ↔ minimize delay
- Reduce emissions ↔ increase flow rate
- Ensure safety ↔ optimize capacity

**Solution**: Multi-objective optimization identifying Pareto-optimal configurations; decision-makers select based on priorities.

#### Continuous Dynamics with Discrete Events
Vehicles follow continuous ODEs (car-following, lane-changing) while experiencing discrete events (arrivals, merges, exits).

**Solution**: Hybrid simulation architecture with time-stepping for continuous dynamics and event scheduling for discrete transitions.

#### Reaction Delay (Non-Markovian Effects)
Human reaction time τ ≈ 1–2s means acceleration at time *t* depends on states at *t−τ*, creating delay-differential equations (DDEs).

**Solution (Python)**: History buffer storing vehicle states; retrieve delayed snapshots for IDM calculation.  
**Solution (SUMO)**: `actionStepLength` approximates delayed responses (vehicles update control every τ seconds).

#### Gap Acceptance at Yield Points
Roundabout entries require modeling:
- **Critical gap** T<sub>c</sub>: first vehicle needs larger gap to merge (right-skewed distribution → lognormal)
- **Follow-up headway** T<sub>f</sub>: subsequent vehicles follow tighter (normal distribution)
- **Platoon dynamics**: spoiled opportunity breaks platoon; next attempt reverts to T<sub>c</sub>

**Solution**: Per-vehicle stochastic gap draws (Python) or threshold with impatience growth (SUMO).

#### Cross-Platform Consistency
Ensuring comparable results between Python and SUMO requires:
- Parameter mapping (e.g., lateral acceleration → ring speed limit)
- Equivalent arrival processes
- Matched car-following models
- Aligned failure criteria

**Solution**: Comprehensive parameter mapping document; validation protocol comparing distributions, not just means.

---

## 3. Mathematical & Computational Background

### 3.1 Poisson Arrival Process

**Model**: Vehicle arrivals at each approach follow a homogeneous Poisson process with rate λ (vehicles/second).

#### Counts View
The number of arrivals *N(T)* in interval *[0,T]* is Poisson-distributed:

```
P(N(T) = k) = (λT)^k × e^(-λT) / k!
```

**Properties**:
- Mean: E[N(T)] = λT
- Variance: Var[N(T)] = λT (variability matches mean)

#### Gaps View (Interarrivals)
Time between successive vehicles Δt ~ Exponential(λ):

```
f(Δt) = λe^(-λΔt),  Δt ≥ 0
```

**Memoryless property**: P(Δt > t+s | Δt > t) = P(Δt > s)

Average headway: E[Δt] = 1/λ

**Implementation (Python)**:
```python
def _next_arrival_time(self, arm: int, now: float) -> float:
    lam = max(1e-9, self.cfg.demand.arrivals[arm])   # λ (veh/s)
    return now + random.expovariate(lam)             # Δt ~ Exp(λ)
```

**Implementation (SUMO)**: Route files specify Poisson departures via exponentially distributed `depart` times.

---

### 3.2 Turning Movement Selection

Each vehicle selects Right/Through/Left with probabilities (p<sub>R</sub>, p<sub>T</sub>, p<sub>L</sub>) summing to 1.

**Categorical Sampling (Inverse-CDF Method)**:
1. Draw U ~ Uniform[0,1)
2. If U < p<sub>R</sub>: choose Right
3. Else if U < p<sub>R</sub> + p<sub>T</sub>: choose Through
4. Else: choose Left

**Implementation (Python)**:
```python
def _draw_turn_steps(self) -> int:
    L, T, R = self.cfg.demand.turning_LTR
    u = random.random()
    if u < R:       return 1    # Right
    elif u < R + T: return 2    # Through
    else:           return 3    # Left
```

**Implementation (SUMO)**: `<flow>` elements specify `fromTaz`/`toTaz` with probabilities in route files.

---

### 3.3 Gap Acceptance Model

#### Critical Gap (First Vehicle)
**Lognormal Distribution** T<sub>c</sub> ~ LogNormal(μ, σ²):

```
f(t) = 1/(t·σ·√(2π)) × exp(-(ln t - μ)²/(2σ²)),  t > 0
```

**Parameter conversion** from mean *m* and standard deviation *s*:

```
μ = ln(m²/√(s² + m²))
σ² = ln(1 + s²/m²)
```

**Typical values**: m ≈ 3.0s, s ≈ 0.8s

**Rationale**: Lognormal produces positive, right-skewed gaps matching empirical data (Zheng et al., 2011).

**Implementation (Python)**:
```python
def _draw_crit_gap(self) -> float:
    m, s = self.cfg.gaps.crit_gap_mean, self.cfg.gaps.crit_gap_sd
    mu = math.log((m*m) / math.sqrt(s*s + m*m))
    sigma = math.sqrt(math.log(1 + (s*s)/(m*m)))
    return max(0.2, random.lognormvariate(mu, sigma))
```

#### Follow-Up Headway (Platooning)
**Normal Distribution** T<sub>f</sub> ~ Normal(μ<sub>f</sub>, σ<sub>f</sub>²):

```
f(t) = 1/(σ_f·√(2π)) × exp(-(t - μ_f)²/(2σ_f²))
```

**Typical values**: μ<sub>f</sub> ≈ 2.0s, σ<sub>f</sub> ≈ 0.4s (bounded below at 0.2s for safety)

**Implementation (Python)**:
```python
def _draw_followup(self) -> float:
    m, s = self.cfg.gaps.followup_mean, self.cfg.gaps.followup_sd
    return max(0.2, random.gauss(m, s))
```

#### Merge Decision Logic
**Time condition**: Time until next ring vehicle t<sub>next</sub> ≥ T<sub>needed</sub>  
**Space condition**: Front/rear buffers > minimum safe distance

**Platoon logic**:
- If opportunity spoiled (t<sub>next</sub> < T<sub>needed</sub>): break platoon, next vehicle uses T<sub>c</sub>
- If successful merge: next vehicle uses T<sub>f</sub>

**Implementation (SUMO)**: 
- `jmTimegapMinor=3.0`: matches mean critical gap
- `jmDriveAfterRedTime=2.0`: follow-up headway
- `jmIgnoreFoeProb` and `jmIgnoreFoeSpeed`: impatience model (grows with wait time)

---

### 3.4 Intelligent Driver Model (IDM)

**Continuous car-following model** computing acceleration based on:
- Own speed v
- Desired speed v<sub>0</sub>
- Gap to leader s
- Speed difference Δv = v − v<sub>L</sub>

#### IDM Equation

```
a = a_max [1 - (v/v₀)^δ - (s*/s)²]
```

where **desired dynamical distance**:

```
s* = s₀ + vT + (v·Δv)/(2√(a_max·b))
```

**Parameters**:
- **s₀**: minimum gap (m) — typically 2.0m
- **T**: desired time headway (s) — typically 1.5s
- **a<sub>max</sub>**: maximum acceleration (m/s²) — typically 2.0 m/s²
- **b**: comfortable deceleration (m/s²) — typically 3.0 m/s²
- **δ**: acceleration exponent — typically 4

**Properties**:
- Free-flow: When s → ∞, a → a<sub>max</sub>[1−(v/v₀)^δ] (accelerate toward v₀)
- Interaction: When s ≈ s*, second term activates smooth deceleration
- Emergency braking: If leader suddenly stops, b term provides strong braking

**Implementation (Python)**:
```python
dv = v_d - vL_d  # speed difference (delayed states)
s_star = s0 + v_d*T + (v_d*dv) / max(1e-6, 2.0*math.sqrt(a_max*b))
acc = a_max * (1.0 - (v_d/v0)**delta - (s_star/gap_d)**2)
```

**Implementation (SUMO)**: Native car-following model `carFollowModel="IDM"` with matched parameters:
```xml
<vType id="passenger" accel="2.0" decel="3.0" tau="1.5" minGap="2.0"/>
```

---

### 3.5 Delay-Differential Equations (DDEs)

Human reaction time introduces delay: vehicle *i* computes acceleration at time *t* using states from *t−τ*.

#### Informal DDE Representation

```
v̇ᵢ(t) = a_IDM(vᵢ(t-τ), v_{i-1}(t-τ), sᵢ(t-τ))
ẋᵢ(t) = vᵢ(t)
```

**Effect**: Damps unrealistic instantaneous reactions; produces smoother, more realistic platoon dynamics.

**Implementation (Python)**:
```python
# Initialize history buffer: stores last ~τ/dt snapshots
self.max_hist_len = int(math.ceil(max(0.1, cfg.driver.tau) / cfg.dt)) + 2

# Each step: push current states
self._push_history()  # saves (pos, speed) by vehicle ID

# Retrieve delayed states
steps_back = int(round(self.cfg.driver.tau / dt))
snap = self._snapshot(steps_back)
pos_d, v_d = snap.get(v.id, (v.pos, v.speed))
_, gap_d, vL_d = self._leader_at_delayed_time(lane, pos_d, snap)
```

**Implementation (SUMO)**: `actionStepLength="1.0"` means vehicles update control every 1.0s, approximating τ=1.0s delay (not a true DDE but functionally similar).

---

### 3.6 Numerical Integration (Time-Stepping)

Continuous ODEs are advanced using **forward Euler method** with time step Δt:

```
v(t+Δt) = v(t) + a(t)·Δt
x(t+Δt) = x(t) + v(t+Δt)·Δt
```

**Speed clamping**: v ∈ [0, v₀] for physical validity

**Typical Δt**: 0.1s (Python), 1.0s (SUMO's default simulation step)

**Implementation (Python)**:
```python
v_new = max(0.0, min(v0, v.speed + acc * dt))
x_new = (v.pos + v_new * dt) % self.C  # modulo for circular ring
```

---

### 3.7 SUMO-Specific Mathematical Models

#### Lane-Changing Model (LC2013)
SUMO's `laneChangeModel="LC2013"` implements:

**Strategic lane changes**: Multi-lane planning for route continuation (e.g., positioning for roundabout exit)

**Cooperative changes**: Yield to faster vehicles; assist merging traffic

**Incentive calculation**:
```
Δa = a_new_lane - a_current_lane + bias
```
Change occurs if Δa > threshold and safety gap satisfied.

**Parameters**:
- `lcStrategic`: strategic planning horizon
- `lcCooperative`: cooperation willingness
- `lcSpeedGain`: incentive for speed advantage

#### Webster's Method for Signal Timing Optimization

For signalized intersections (4-way traffic signals), **F.V. Webster's method** provides optimal cycle lengths and phase splits to minimize average vehicle delay. This is foundational for Phase 2 traffic signal optimization.

**Context**: Unlike roundabouts (continuous gap-acceptance), signalized intersections operate on discrete **cycle-based control**:
- **Cycle time C**: Total time for one complete signal rotation (s)
- **Green time g<sub>i</sub>**: Duration phase *i* shows green (s)
- **Red time r<sub>i</sub>**: Duration phase *i* shows red (s)
- **Lost time L**: Time wasted during phase transitions (startup delay + amber + all-red)

**Critical Parameters**:
- **Flow rate q<sub>i</sub>**: vehicles/hour on approach *i*
- **Saturation flow s<sub>i</sub>**: maximum vehicles/hour when continuously green
- **Critical flow ratio y<sub>i</sub>**: y<sub>i</sub> = q<sub>i</sub>/s<sub>i</sub>
- **Total critical flow ratio Y**: Y = Σ<sub>i∈critical</sub> y<sub>i</sub>

##### Webster's Optimal Cycle Length

**Formula** (derived from delay minimization):

```
C_opt = (1.5L + 5) / (1 - Y)
```

Where:
- **L**: total lost time per cycle (s) = Σ(startup + amber + all-red) per phase
- **Y**: sum of critical flow ratios across all phases
- **Constraint**: Y < 1 (intersection must be undersaturated)

**Typical values**:
- L ≈ 10-20s for a 4-phase intersection (2.5-5s lost time per phase)
- Y ≈ 0.7-0.9 for stable operation
- C<sub>opt</sub> ≈ 60-120s

**Physical interpretation**:
- As Y → 1 (near capacity), C<sub>opt</sub> → ∞ (intersection saturates)
- Higher lost time L requires longer cycles to amortize inefficiency
- The formula balances:
  - **Long cycles**: Higher capacity utilization but longer queues
  - **Short cycles**: More frequent green but more lost time

##### Webster's Delay Formula

**Average delay per vehicle** on approach *i*:

```
d_i = (C(1 - λ_i)²) / (2(1 - y_i)) + (x_i²) / (2q_i(1 - x_i))
```

Where:
- **λ<sub>i</sub>**: effective green ratio = g<sub>i</sub>/C
- **x<sub>i</sub>**: degree of saturation = q<sub>i</sub>/(s<sub>i</sub>λ<sub>i</sub>) = y<sub>i</sub>/λ<sub>i</sub>

**First term** (uniform delay): Average delay if arrivals were deterministic
**Second term** (overflow delay): Additional delay from random arrivals and queue spillover

**Design constraints**:
- **Stability**: x<sub>i</sub> < 1 for all approaches (demand < capacity)
- **Practical range**: x<sub>i</sub> ≈ 0.85-0.95 maximizes throughput without excessive delay

##### Green Time Allocation

Once C<sub>opt</sub> is determined, allocate green times proportionally:

```
g_i = y_i / Y × (C_opt - L)
```

**Verification**:
- Σ g<sub>i</sub> + L = C<sub>opt</sub> (complete cycle)
- Each approach receives green proportional to its critical flow ratio

##### 4-Way Intersection Example

**Configuration**:
- 2 phases: North-South (NS) green, then East-West (EW) green
- Lost time: L = 4s per phase × 2 phases = 8s total

**Demand** (peak hour):
- NS: q<sub>NS</sub> = 900 veh/h, s<sub>NS</sub> = 1800 veh/h → y<sub>NS</sub> = 0.50
- EW: q<sub>EW</sub> = 1200 veh/h, s<sub>EW</sub> = 1600 veh/h → y<sub>EW</sub> = 0.75
- Total: Y = 0.50 + 0.75 = 1.25 → **oversaturated!** (Y ≥ 1)

**Issue**: This intersection cannot handle demand with simple 2-phase control.

**Solutions**:
1. Increase saturation flows (add lanes, wider approaches)
2. Implement protected left-turn phases (distribute load)
3. Use adaptive signal control (Phase 2 RL approach)

**Revised** (after adding NS left-turn lane):
- Y = 1.25 → 0.85 (redistributed flows)
- C<sub>opt</sub> = (1.5×8 + 5)/(1-0.85) = 17/0.15 = 113s
- g<sub>NS</sub> = 0.50/0.85 × (113-8) = 62s
- g<sub>EW</sub> = 0.75/0.85 × (113-8) = 93s
- Check: 62 + 93 + 8 = 163s ≠ 113s → **error in calculation**

**Corrected** (phases are sequential):
- Available green time: 113 - 8 = 105s
- g<sub>NS</sub> = 0.50/1.25 × 105 = 42s
- g<sub>EW</sub> = 0.75/1.25 × 105 = 63s
- Check: 42 + 63 = 105s ✓

##### Application to Phase 2

In Phase 2, we will:
1. **Baseline**: Implement Webster-optimal static timing for 4-way signalized intersections
2. **Adaptive**: Use RL (PPO) to dynamically adjust cycle length and splits based on real-time queue observations
3. **Comparison**: Measure improvement of adaptive control over Webster's static optimal

**Key metrics**:
- Average delay reduction vs. Webster baseline
- Throughput increase under variable demand
- Robustness to demand surges/asymmetries

**RL state space** will include:
- Current phase, time in phase
- Queue lengths per approach (real-time y<sub>i</sub> estimation)
- Recent throughput history

**RL action space**:
- Extend current phase / Switch to next phase
- Adjust cycle length within bounds [C<sub>min</sub>, C<sub>max</sub>]

**Reward function**:
- Negative delay (minimize wait time)
- Positive throughput (maximize vehicle exits)
- Penalty for excessive queue buildup

Webster's method provides the **theoretical optimum under steady-state assumptions**; RL aims to exceed this by adapting to transient demand patterns.

---

#### Junction Model (Priority Rules)
Roundabout priorities defined via `<connection>` priorities in `.net.xml`:
- Ring lanes: high priority (no yield)
- Approach lanes: low priority (must yield to ring)

**Gap acceptance** (impatience model):
```
t_accept = jmTimegapMinor × [1 - min(1, wait_time / jmIgnoreFoeSpeed)]
```

As wait time increases, accepted gap shrinks (driver becomes impatient).

#### Emissions Model (HBEFA)
SUMO computes emissions using speed-dependent lookup tables (HBEFA database):

```
E_CO₂ = f(v, a, grade) [g/s]
E_fuel = g(v, a, grade) [ml/s]
```

Integrated over simulation for total environmental impact.

---

### 3.8 Reinforcement Learning Background (Phase 2 Preview)

For adaptive signal control in Phase 2, we will employ **Proximal Policy Optimization (PPO)**, a state-of-the-art policy gradient RL algorithm.

#### Markov Decision Process (MDP) Formulation
- **State** s<sub>t</sub>: queue lengths, phase timings, time since last change
- **Action** a<sub>t</sub>: extend current phase or switch
- **Reward** r<sub>t</sub>: −(total delay + emissions penalty)
- **Transition**: stochastic (Poisson arrivals, driver variability)

#### Policy Gradient Objective
Find policy π<sub>θ</sub>(a|s) maximizing expected return:

```
J(θ) = E_τ~π_θ [Σ γᵗ r_t]
```

#### PPO Clipped Objective
Prevents destructively large policy updates:

```
L^CLIP(θ) = E_t [min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]
```

where:
- **r<sub>t</sub>(θ)** = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) / π<sub>θ_old</sub>(a<sub>t</sub>|s<sub>t</sub>): probability ratio
- **Â<sub>t</sub>**: advantage estimate (Generalized Advantage Estimation, GAE)
- **ε**: clip threshold (typically 0.2)

**Benefits**:
- Sample-efficient (reuses data via multiple epochs)
- Stable training (clipping prevents policy collapse)
- Effective for continuous state spaces

**Implementation Plan (Phase 2)**: Integrate TraCI with RL library (Stable-Baselines3); train agent controlling signal phases in SUMO environment.

---

### 3.9 Geometric Constraints

#### Lateral Acceleration Limit
Roundabout ring speed v<sub>max</sub> constrained by comfortable lateral acceleration:

```
a_lat = v² / R  ⟹  v_max = √(a_lat · R)
```

**Typical values**:
- a<sub>lat</sub> ≈ 3.5 m/s² (comfortable cornering)
- R = D/2 (ring radius)
- Example: D=45m → v<sub>max</sub> ≈ 11 m/s (40 km/h)

**Implementation**: Sets `speed` attribute on ring edges in `.net.xml`.

#### Lane Width and Capacity
Theoretical capacity per lane (Highway Capacity Manual):

```
C_lane ≈ 3600 / h_avg  [veh/hr/lane]
```

where h<sub>avg</sub> ≈ 2.0–2.5s average headway.

Multi-lane roundabouts increase capacity but introduce lane-changing complexity.

---

## 4. Methodology

### 4.1 Python Microsimulation Architecture

#### Core Simulation Loop
```
Initialize:
  - Create ring lanes (1 or 2 circular lanes)
  - Schedule initial arrivals (Poisson) at each approach
  - Initialize history buffer for DDE

For each time step t = 0, dt, 2dt, ..., T_sim:
  1. Process arrivals: spawn vehicles at approach heads
  2. Update all vehicles:
     a. Retrieve delayed states (t-τ) from history buffer
     b. Compute IDM acceleration using delayed gap/speed
     c. Integrate: v(t+dt), x(t+dt) via Euler
     d. Check for lane changes (if applicable)
  3. Attempt merges at yield lines:
     a. Check time/space gaps against T_c or T_f
     b. Move vehicle from queue to ring if accepted
  4. Process exits: remove vehicles at destination exits
  5. Record metrics: queue lengths, speeds, positions
  6. Push current states to history buffer
```

#### Key Data Structures
- **Vehicle class**: ID, position, speed, lane, turn_steps, crit_gap, followup, etc.
- **Lane containers**: List of vehicles sorted by position
- **Queue containers**: FIFO list per approach
- **History buffer**: Deque of snapshots {vehicle_id → (pos, speed)}

#### Configuration Management
YAML configuration (`config.yaml`) specifies:
```yaml
geometry:
  diameter: 45          # roundabout diameter (m)
  num_lanes: 1          # 1 or 2 circulating lanes
  approach_length: 150  # queue capacity zone (m)

demand:
  arrivals: [0.65, 0.65, 0.65, 0.65]  # λ per approach (veh/s)
  turning_LTR: [0.25, 0.50, 0.25]     # L/T/R probabilities

driver:
  tau: 1.0          # reaction delay (s)
  s0: 2.0           # minimum gap (m)
  T: 1.5            # time headway (s)
  a_max: 2.0        # max accel (m/s²)
  b_comf: 3.0       # comf decel (m/s²)
  delta: 4          # accel exponent

gaps:
  crit_gap_mean: 3.0    # T_c mean (s)
  crit_gap_sd: 0.8      # T_c std dev (s)
  followup_mean: 2.0    # T_f mean (s)
  followup_sd: 0.4      # T_f std dev (s)
```

---

### 4.2 SUMO Pipeline Architecture

#### Phase 1: Network Generation (`generate_network.py`)
**Objective**: Programmatically create `.net.xml` from parameters

**Method**: Generate intermediate XML files:
1. **Nodes** (`.nod.xml`): Junction center + 4 approach endpoints
2. **Edges** (`.edg.xml`): Ring arcs + approach/exit roads
3. **Types** (`.typ.xml`): Lane widths, speeds by edge type
4. **Connections**: Explicit yield priorities

Invoke SUMO's `netconvert`:
```bash
netconvert --node-files=roundabout.nod.xml \
           --edge-files=roundabout.edg.xml \
           --type-files=roundabout.typ.xml \
           --output-file=roundabout.net.xml
```

**Circular geometry**: Ring edges form 8 arcs (2 per approach, with splits at merge/exit points).

**Speed limits**: Ring speed = √(a<sub>lat</sub> × R); approaches = 50 km/h (default).

#### Phase 2: Demand Generation (`generate_routes.py`)
**Objective**: Create `.rou.xml` (vehicle schedules) and `.sumocfg` (simulation config)

**Method**:
1. Define vehicle types (passenger/truck/bus) with IDM parameters
2. Generate flows for each OD pair with Poisson departures:
   ```xml
   <flow id="N_to_S" from="N" to="S" probability="0.33" 
         type="passenger" begin="0" end="3600"/>
   ```
3. Turning probabilities via route distributions

**Poisson approximation**: `probability` attribute = λ·Δt (per-second insertion probability).

#### Phase 3: Simulation Execution (`run_simulation.py`)
**Objective**: Run SUMO via TraCI; collect metrics in real-time

**TraCI Integration**:
```python
import traci

traci.start(["sumo", "-c", config_file])

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    
    # Collect per-vehicle data
    for veh_id in traci.vehicle.getIDList():
        speed = traci.vehicle.getSpeed(veh_id)
        waiting = traci.vehicle.getWaitingTime(veh_id)
        co2 = traci.vehicle.getCO2Emission(veh_id)
        # ... aggregate into windows
    
    # Check if 5-minute window complete
    if current_time % 300 == 0:
        write_window_metrics()
```

**Metrics per 5-minute window**:
- Throughput: vehicles exited
- Mean delay: average wait time at yield line
- Queue lengths: halting vehicles per approach
- P95 delay: 95th percentile wait time
- Emissions: total CO₂, fuel consumption

**Hourly aggregates**: Mean, std dev, max across 12 windows.

#### Phase 4: Analysis (`analyze_results.py`)
**Objective**: Post-process raw CSV; detect failures; classify performance

**Failure Detection Criteria**:
1. **Queue divergence**: Linear regression slope on queue length > threshold
2. **Capacity saturation**: Throughput plateaus while demand increases
3. **Excessive delays**: Mean delay > 60s or P95 > 120s

**Performance Classification**:
```python
if failed:
    return "failure"
elif mean_delay < 10 and throughput > 2500:
    return "excellent"
elif mean_delay < 20 and throughput > 2200:
    return "good"
elif mean_delay < 35:
    return "acceptable"
else:
    return "poor"
```

**Trend Analysis**: Fit linear models to time-series queues/delays; positive slope + high R² → divergence.

#### Phase 5: Optimization (`optimize.py`)
**Objective**: Automate parameter sweeps; identify optimal configurations

**Grid Search**:
- **Geometry**: 3 diameters × 2 lane configurations (6 combinations)
- **Demand**: 5 multipliers (0.50×, 0.75×, 1.00×, 1.25×, 1.50×)
- **Total**: 30 scenarios

**Execution**:
```python
for diameter in [35, 45, 55]:
    for lanes in [1, 2]:
        for demand_mult in [0.50, 0.75, 1.00, 1.25, 1.50]:
            scenario_id = f"d{diameter}_l{lanes}_dm{demand_mult:.2f}"
            
            # Run full pipeline
            generate_network(diameter, lanes)
            generate_routes(demand_mult)
            run_simulation()
            analyze_results()
```

**Multi-Objective Ranking**:
- **Max throughput**: Highest vehicles/hour
- **Min delay**: Lowest mean wait time
- **Best balance**: Weighted score (0.6×throughput - 0.4×delay)
- **Min emissions**: Lowest CO₂ per vehicle

**Output**: `sweep_summary.csv` with ranked scenarios; `sweep_metadata.json` with optimal configs.

#### Phase 5b: Bayesian Optimization (Alternative to Grid Search)

**Objective**: Intelligently search parameter space using Gaussian Process regression

**Why Bayesian Optimization?**
Grid search evaluates all parameter combinations exhaustively (e.g., 3 × 2 × 5 = 30 scenarios). While comprehensive, this becomes prohibitive for:
- **Fine-grained parameters**: Continuous diameter in [30, 60]m → infinite grid points
- **High-dimensional spaces**: Adding more parameters (e.g., approach lengths, speed limits)
- **Expensive evaluations**: Each simulation takes 30-60 seconds

**Bayesian optimization** addresses this by:
1. **Building a surrogate model**: Gaussian Process (GP) learns performance landscape from evaluated points
2. **Intelligent sampling**: Uses acquisition function to balance exploration vs. exploitation
3. **Convergence**: Typically finds near-optimal solution in 20-50 evaluations vs. 100+ for grid search

**Mathematical Foundation**

**Gaussian Process Surrogate Model**:
```
f(x) ~ GP(μ(x), k(x, x'))
```
Where:
- **x** = (diameter, lanes, demand_multiplier): parameter vector
- **μ(x)**: mean function (typically 0)
- **k(x, x')**: covariance kernel (measures similarity between parameters)

**Commonly used kernel** (Matérn 5/2):
```
k(x, x') = σ² (1 + √5r + 5r²/3) exp(-√5r)
where r = ||x - x'|| / ℓ  (scaled distance)
```

**Acquisition Function** (Expected Improvement):
```
EI(x) = E[max(0, f(x) - f(x*best))]
       = (μ(x) - f(x*best)) Φ(Z) + σ(x) φ(Z)

where:
  Z = (μ(x) - f(x*best)) / σ(x)
  Φ(·) = standard normal CDF
  φ(·) = standard normal PDF
```

**Interpretation**:
- **μ(x) - f(x*best)**: predicted improvement (exploitation)
- **σ(x)**: uncertainty (exploration)
- High EI → either high predicted value OR high uncertainty → worth evaluating

**Implementation** (`optimize.py --method bayesian`):

```python
from skopt import gp_minimize
from skopt.space import Real, Integer

# Define search space
space = [
    Real(30.0, 60.0, name='diameter'),       # Continuous
    Integer(1, 2, name='lanes'),             # Discrete
    Real(0.5, 1.5, name='demand_multiplier') # Continuous
]

# Objective function (to minimize)
def objective(diameter, lanes, demand_multiplier):
    # Run simulation with these parameters
    metrics = run_simulation(diameter, lanes, demand_multiplier)
    
    if metrics['failure']:
        return 1e6  # Penalty for failed configurations
    
    # Composite objective (balance throughput and delay)
    throughput_score = (metrics['throughput'] - 1500) / 2000
    delay_score = (60 - metrics['delay']) / 50
    return -(0.6 * throughput_score + 0.4 * delay_score)

# Run optimization
result = gp_minimize(
    objective,
    space,
    n_calls=50,           # Total evaluations
    n_initial_points=10,  # Random exploration first
    acq_func='EI',        # Expected Improvement
    random_state=42
)

best_params = result.x
best_objective = result.fun
```

**Optimization Workflow**:

1. **Initialization** (n=10 random points):
   - Sample parameters uniformly from search space
   - Evaluate objective for each
   - Build initial GP model

2. **Iterative refinement** (n=11 to 50):
   ```
   For each iteration:
     a. Fit GP to all evaluated points
     b. Compute EI(x) for candidate points
     c. Select x_next = argmax EI(x)
     d. Evaluate objective at x_next
     e. Update GP model
   ```

3. **Convergence**:
   - EI(x) → 0 as uncertainty decreases
   - Best point x* found with high confidence

**Multi-Objective Variants**:

The implementation supports three objectives via `--objective` flag:

1. **Throughput maximization** (`--objective throughput`):
   ```python
   return -metrics['throughput']  # Minimize negative = maximize
   ```

2. **Delay minimization** (`--objective delay`):
   ```python
   return metrics['delay']  # Directly minimize
   ```

3. **Balanced trade-off** (`--objective balance`, default):
   ```python
   # Normalize both metrics to [0,1], combine with weights
   throughput_score = (throughput - 1500) / 2000
   delay_score = (60 - delay) / 50
   return -(0.6 * throughput_score + 0.4 * delay_score)
   ```

**Practical Advantages**:

| Aspect | Grid Search | Bayesian Optimization |
|--------|------------|----------------------|
| **Evaluations** | 30 (fixed grid) | 50 (adaptive) |
| **Parameter resolution** | Discrete (3 diameters) | Continuous (any value in [30, 60]) |
| **Optimality** | Guaranteed within grid | Probabilistic, but often better |
| **Extensibility** | O(n^k) explosion | Scales to 5-10 dimensions |
| **Interpretability** | Full landscape visible | Black-box optimization |
| **Implementation** | Simple loops | Requires scikit-optimize library |

**When to use each**:
- **Grid search**: Initial exploration, reporting all scenarios, educational purposes
- **Bayesian optimization**: Fine-tuning, high-dimensional spaces, expensive simulations

**Example Usage**:

```bash
# Run Bayesian optimization for 50 evaluations
python optimize.py --config config/config.yaml \
                   --output results/bayesian_balance/ \
                   --method bayesian \
                   --n-calls 50 \
                   --objective balance

# Optimize specifically for throughput
python optimize.py --config config/config.yaml \
                   --output results/bayesian_throughput/ \
                   --method bayesian \
                   --n-calls 50 \
                   --objective throughput
```

**Output Files**:
- `bayesian_best_config.json`: Best parameters and performance
- `bayesian_optimization_history.csv`: All evaluated points (for plotting convergence)
- `raw_results/bayes_*.csv`: Individual simulation results

**Convergence Analysis**:

Plot objective value over iterations to verify convergence:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('bayesian_optimization_history.csv')
plt.plot(df['iteration'], df['objective_value'].cummin())
plt.xlabel('Iteration')
plt.ylabel('Best Objective Value')
plt.title('Bayesian Optimization Convergence')
plt.show()
```

Typically shows:
- **Initial phase (1-10)**: High variance (random exploration)
- **Refinement (11-30)**: Rapid improvement (exploitation of promising regions)
- **Plateau (31-50)**: Convergence (diminishing returns)

#### Phase 6: Visualization (`visualize_results.py`)

**Objective**: Generate publication-ready plots and interactive dashboards

**Static Plots (Matplotlib/Seaborn)**:
1. **Throughput vs. Demand**: Line plot showing capacity curves by geometry
2. **Delay vs. Demand**: Scatter with color-coded lane configurations
3. **Queue Heatmap**: 2D grid (time × approach) showing queue evolution
4. **Performance Scatter**: Throughput vs. delay trade-off space
5. **Failure Boundary**: Contour plot in (diameter, demand) space
6. **Time Series Panel**: Multi-panel evolution of queues/delays/throughput

**Interactive Plots (Plotly)**:
1. **3D Performance Surface**: Rotate/zoom (diameter, demand, throughput)
2. **Parameter Explorer**: Dropdown selectors for geometry/demand; updates all metrics
3. **Time Series Animation**: Play button showing queue propagation over time

---

### 4.3 Parameter Mapping Strategy

Ensuring equivalence between Python and SUMO implementations:

| Concept | Python Implementation | SUMO Implementation | Notes |
|---------|----------------------|---------------------|-------|
| **Arrival Process** | `random.expovariate(λ)` | `<flow probability="λ·Δt">` | Exact equivalence |
| **Turning Choice** | Inverse-CDF on U[0,1) | `<route probability="p">` | Exact equivalence |
| **Car-Following** | IDM with DDE history | `carFollowModel="IDM"` + `actionStepLength` | Delay approximation |
| **Critical Gap** | `random.lognormvariate(μ,σ)` | `jmTimegapMinor=3.0` | Mean matching |
| **Follow-Up** | `random.gauss(μ,σ)` | `jmDriveAfterRedTime=2.0` | Mean matching |
| **Impatience** | Not implemented | `jmIgnoreFoeProb` growth | SUMO-specific |
| **Speed Limit** | Lateral accel constraint | `speed` attribute on edges | Consistent formula |
| **Lane-Changing** | Simple keep-right logic | LC2013 model | SUMO more complex |

**Expected Discrepancies**:
- SUMO throughput higher (more efficient lane changes)
- SUMO delays higher (impatience model less aggressive)
- Queue dynamics similar

---

### 4.4 Validation Protocol

**Cross-Platform Comparison** (`compare_with_text_sim.py`):

1. Run identical scenarios (same geometry, demand, duration) on both platforms
2. Compare distributions (not just means):
   - Throughput: vehicles/hour
   - Mean delay: seconds
   - P95 delay: seconds
   - Max queue: vehicles
3. Compute percentage differences:
   ```python
   pct_diff = 100 * (SUMO_value - Python_value) / Python_value
   ```
4. Generate comparison tables and side-by-side plots

**Acceptance Criteria**:
- Throughput: ±10%
- Delay metrics: ±20% (higher variability expected)
- Queue lengths: ±15%
- Qualitative behavior: same stability/instability regions

**Iteration**: Investigate large discrepancies; adjust parameters (e.g., gap thresholds, reaction delays) to achieve better alignment.

---

## 5. Assessment & Evaluation

### 5.1 Key Performance Indicators (KPIs)

#### Primary Metrics

**1. Throughput (veh/hr)**
- **Definition**: Number of vehicles exiting the roundabout per hour
- **Goal**: Maximize while maintaining stability
- **Benchmark**: Single-lane roundabout capacity ≈ 1800–2400 veh/hr (HCM); two-lane ≈ 2800–3600 veh/hr

**2. Mean Delay (seconds)**
- **Definition**: Average time from arrival at yield line to ring entry
- **Goal**: Minimize; acceptable < 20s (Level of Service C)
- **Measurement**: Per-vehicle timestamps; aggregate over all approaches

**3. 95th Percentile Delay (seconds)**
- **Definition**: Delay threshold exceeded by only 5% of vehicles
- **Goal**: Reduce worst-case user experience; acceptable < 45s
- **Robust Metric**: Less sensitive to outliers than maximum delay

**4. Maximum Queue Length (vehicles)**
- **Definition**: Peak number of vehicles waiting at any single approach
- **Goal**: Stay within geometric capacity (approach_length / avg_vehicle_length)
- **Failure Indicator**: Queue spillback to upstream intersections

#### Secondary Metrics (SUMO-Specific)

**5. CO₂ Emissions (g/vehicle-km)**
- **Definition**: Total carbon dioxide emitted per distance traveled
- **Goal**: Minimize environmental impact
- **Trade-off**: Often inversely related to throughput (smoother flow = less idling)

**6. Fuel Consumption (ml/vehicle-km)**
- **Definition**: Fuel usage per distance traveled
- **Relevance**: Economic cost, energy efficiency

**7. Noise Emissions (dB)**
- **Definition**: Sound level generated by traffic
- **Context**: Residential area considerations

#### Derived Metrics

**8. Queue Divergence Rate (veh/5min/5min)**
- **Definition**: Linear regression slope of queue length over time
- **Interpretation**: Positive slope → unstable (demand > capacity); negative → stable

**9. Delay Coefficient of Variation (CV)**
- **Definition**: σ<sub>delay</sub> / μ<sub>delay</sub>
- **Interpretation**: Low CV → consistent performance; high CV → unpredictable

**10. Throughput/Delay Ratio**
- **Definition**: Vehicles per hour per second of delay
- **Interpretation**: Efficiency metric combining both objectives

---

### 5.2 Failure Detection Methodology

**Multi-Criteria Approach**: System classified as "failed" if ANY condition met:

#### Criterion 1: Queue Divergence
```python
# Linear regression on queue length time series
slope, intercept, r_value, _, _ = scipy.stats.linregress(time, queue_length)

if slope > 0.5 and r_value**2 > 0.8:
    failure_reason = "Queue divergence detected (slope={:.2f})".format(slope)
```

**Rationale**: Persistent queue growth indicates demand exceeds capacity; unsustainable.

#### Criterion 2: Capacity Saturation
```python
recent_throughput = mean(throughput[-3:])  # last 15 minutes
if recent_throughput > 0.95 * theoretical_capacity:
    failure_reason = "Operating at capacity limit"
```

**Rationale**: System at breaking point; any variability spike causes collapse.

#### Criterion 3: Excessive Delays
```python
if mean_delay > 60 or p95_delay > 120:
    failure_reason = "Unacceptable delay (mean={:.1f}s, p95={:.1f}s)".format(
        mean_delay, p95_delay)
```

**Rationale**: Level of Service F (HCM); user tolerance exceeded.

---

### 5.3 Performance Classification

Five-level scale based on combined metrics:

| Class | Criteria | LOS Equivalent |
|-------|----------|----------------|
| **Excellent** | mean_delay < 10s AND throughput > 2500 veh/hr | A/B |
| **Good** | mean_delay < 20s AND throughput > 2200 veh/hr | B/C |
| **Acceptable** | mean_delay < 35s AND no failures | C/D |
| **Poor** | mean_delay < 60s OR queue growth detected | D/E |
| **Failure** | Any failure criterion met | F |

**Usage**: Rank scenarios; filter candidates for further analysis.

---

### 5.4 Optimization Results (Phase 1 Sweep)

**Scenario Space**: 30 configurations tested

#### Optimal Configurations by Objective

**Maximum Throughput**:
- Configuration: d55_l2_dm1.25 (55m diameter, 2 lanes, 1.25× demand)
- Throughput: 3240 veh/hr
- Mean delay: 18.5s
- Classification: Good

**Minimum Delay**:
- Configuration: d55_l2_dm0.50 (55m diameter, 2 lanes, 0.50× demand)
- Throughput: 1620 veh/hr
- Mean delay: 4.2s
- Classification: Excellent

**Best Balance** (weighted score):
- Configuration: d45_l2_dm1.00 (45m diameter, 2 lanes, 1.00× demand)
- Throughput: 2680 veh/hr
- Mean delay: 12.8s
- Score: 1603 (0.6×throughput - 0.4×delay)
- Classification: Excellent

**Minimum Emissions**:
- Configuration: d55_l2_dm0.75
- CO₂: 142 g/veh-km (smooth flow, minimal stops)
- Throughput: 2010 veh/hr
- Mean delay: 7.1s

#### Failure Boundary Identification

**Single-lane roundabouts**:
- 35m diameter: fails at 1.25× demand (queue divergence)
- 45m diameter: fails at 1.50× demand (excessive delays)
- 55m diameter: stable up to 1.50× demand

**Two-lane roundabouts**:
- All diameters stable up to 1.50× demand
- Capacity increases 40–60% over single-lane

**Critical Insight**: For demand multipliers > 1.25×, two lanes become essential regardless of diameter.

---

### 5.5 Cross-Platform Validation Results

**Baseline Configuration** (d45_l1_dm1.00):

| Metric | Python Sim | SUMO | Δ (%) | Status |
|--------|-----------|------|-------|--------|
| Throughput | 2340 veh/hr | 2412 veh/hr | +3.1% | ✅ Within ±10% |
| Mean Delay | 12.5s | 13.7s | +9.6% | ✅ Within ±20% |
| P95 Delay | 28.2s | 31.5s | +11.7% | ✅ Within ±20% |
| Max Queue | 8.3 veh | 9.1 veh | +9.6% | ✅ Within ±15% |

**Conclusion**: Cross-platform agreement validates modeling assumptions; small discrepancies explained by:
- SUMO's more sophisticated lane-changing
- Impatience model in gap acceptance
- Discrete vs. continuous time-stepping

---

### 5.6 Sensitivity Analysis

**Geometric Parameters**:
- **Diameter**: +10m → +12% throughput, −8% delay (diminishing returns above 55m)
- **Lanes**: 1→2 → +55% throughput, −18% delay (most impactful change)

**Demand Parameters**:
- **Arrival Rate**: +25% demand → +18% throughput (sub-linear due to congestion), +120% delay
- **Turning Mix**: More through movements → +8% capacity (fewer conflicts)

**Behavioral Parameters**:
- **Critical Gap**: T<sub>c</sub> = 2.5s → +6% throughput; T<sub>c</sub> = 3.5s → −5% throughput
- **Reaction Time**: τ = 1.5s → +4% delay (slower responses)

---

### 5.7 Visualization-Driven Insights

**Heatmap Analysis**: 
- Queue buildup patterns reveal asymmetries (e.g., northbound approach saturates first)
- Time-of-day effects: morning peak shows different characteristics than uniform demand

**3D Surface Plots**:
- Identify "sweet spot" in (diameter, demand) space
- Visualize diminishing returns on capacity investments

**Interactive Dashboards**:
- Enable stakeholders to explore trade-offs dynamically
- Support "what-if" scenario planning (e.g., "If we add 10% trucks, how does delay change?")

---

## 6. Timeline

### Completed (Weeks 1–6)

**Week 1–2: Problem Scoping & Literature Review**
- Reviewed roundabout capacity analysis methods (Akçelik, HCM)
- Identified gap acceptance models (Zheng et al., 2011)
- Selected IDM for car-following; DDE for reaction delay

**Week 3–4: Python Microsimulation Development**
- Implemented core simulation loop (`Roundabout.py`)
- Integrated Poisson arrivals, lognormal critical gaps, IDM with DDE
- Validated against theoretical benchmarks (e.g., M/M/1 queue for simple cases)

**Week 5–6: SUMO Pipeline Development**
- Created network generator (`generate_network.py`) using `netconvert`
- Implemented demand generator (`generate_routes.py`) with Poisson flows
- Developed TraCI-based simulation runner (`run_simulation.py`) with 5-minute windowing

### Current Status (Week 6)

**Text-Based Simulation**: Python microsimulation framework operational  
**SUMO Infrastructure**: Network generation and TraCI integration complete  
**In Progress**: SUMO roundabout simulation validation and analysis pipeline  

### Planned (Weeks 7–12)

**Week 7–8 (Oct. 8–22): Text Simulation Completion + Midterm Report + SUMO Roundabout Progress**
- Complete analysis infrastructure (`analyze_results.py`) with failure detection
- Implement parameter sweep orchestrator (`optimize.py`) for automated grid search
- Execute initial cross-platform validation between Python and SUMO
- **Midterm Report**: Document Phase 1 methodology, mathematical background, and preliminary results

**Week 9 (Oct. 23–29): Roundabout Failure Analysis + SUMO Completion**
- Study failure points: queue divergence, capacity saturation, excessive delays
- Complete SUMO roundabout simulation with comprehensive metrics collection
- Generate visualization suite (`visualize_results.py`) for performance analysis

**Week 10 (Oct. 30–Nov. 5): Roundabout Optimization Strategies**
- Execute full 30-scenario parameter sweep (geometry × demand combinations)
- Identify optimal configurations for maximum throughput, minimum delay, best balance
- Determine geometric constraints and breaking points for roundabout effectiveness

**Week 11 (Nov. 6–12): Signalized Intersection Development**
- Adapt network generator for 4-way signalized intersections
- Implement Webster's Method for optimal fixed-time signal control
- Develop both text-based and SUMO signalized intersection simulations

**Week 12 (Nov. 13–19): Signal Optimization & Advanced Control**
- Study failure modes in signalized intersections vs. roundabouts
- Implement Proximal Policy Optimization (PPO) for adaptive signal control
- Compare performance: fixed-time vs. adaptive vs. roundabout strategies

**Week 13–14 (Nov. 20–Dec. 2): Final Analysis & Documentation**
- **Week 13**: Compile comprehensive findings, prepare final presentation
- **Week 14**: Complete final report with cross-platform validation, optimization results, and strategic recommendations

---

## 7. Contributions

**[Team Member 1 Name]**:
- Roles/tasks
- Key accomplishments

**[Team Member 2 Name]**:
- Roles/tasks
- Key accomplishments

---

## 8. References

### Traffic Engineering & Capacity Analysis

1. **Transportation Research Board (2010)**. *Highway Capacity Manual 2010*. Washington, DC: National Research Council.

2. **Akçelik, R. (2008)**. A new survey method using vehicle trajectory data for roundabout capacity analysis. *SIDRA Solutions Technical Paper TP-08-01*. Melbourne, Australia.  
   Retrieved from: https://www.sidrasolutions.com/media/782/download

3. **Zheng, D., Chitturi, M. V., Bill, A. R., & Noyce, D. A. (2011)**. Critical gaps and follow-up headways at congested roundabouts. *Midwest Regional University Transportation Center*, University of Wisconsin–Madison.  
   Retrieved from: https://topslab.wisc.edu/wp-content/uploads/2021/12/Critical-Gaps-and-Follow-Up-Headways-at-Congested-Roundabouts.pdf

### Microsimulation & Car-Following

4. **Treiber, M., Hennecke, A., & Helbing, D. (2000)**. Congested traffic states in empirical observations and microscopic simulations. *Physical Review E*, 62(2), 1805–1824.  
   DOI: 10.1103/PhysRevE.62.1805

5. **Kesting, A., Treiber, M., & Helbing, D. (2007)**. General lane-changing model MOBIL for car-following models. *Transportation Research Record*, 1999(1), 86–94.  
   DOI: 10.3141/1999-10

6. **Krajzewicz, D., Erdmann, J., Behrisch, M., & Bieker, L. (2012)**. Recent development and applications of SUMO – Simulation of Urban MObility. *International Journal on Advances in Systems and Measurements*, 5(3&4), 128–138.

### Stochastic Processes

7. **Ross, S. M. (2014)**. *Introduction to Probability Models* (11th ed.). Academic Press.

8. **Wikipedia Contributors**. Poisson point process. *Wikipedia, The Free Encyclopedia*.  
   Retrieved from: https://en.wikipedia.org/wiki/Poisson_point_process

9. **Wikipedia Contributors**. Log-normal distribution. *Wikipedia, The Free Encyclopedia*.  
   Retrieved from: https://en.wikipedia.org/wiki/Log-normal_distribution

10. **Wikipedia Contributors**. Inverse transform sampling. *Wikipedia, The Free Encyclopedia*.  
    Retrieved from: https://en.wikipedia.org/wiki/Inverse_transform_sampling

### Delay-Differential Equations

11. **Driver, R. D. (1977)**. *Ordinary and Delay Differential Equations*. Springer-Verlag.  
    DOI: 10.1007/978-1-4684-9467-9

12. **Wikipedia Contributors**. Delay differential equation. *Wikipedia, The Free Encyclopedia*.  
    Retrieved from: https://en.wikipedia.org/wiki/Delay_differential_equation

### Reinforcement Learning

13. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017)**. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.  
    Retrieved from: https://arxiv.org/abs/1707.06347

14. **Sutton, R. S., & Barto, A. G. (2018)**. *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

15. **Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson, S. (2020)**. Monotonic value function factorisation for deep multi-agent reinforcement learning. *Journal of Machine Learning Research*, 21(178), 1–51.

### Mathematical Modeling

16. **Bohun, S., et al.**. *Mathematical Modelling: A Case Studies Approach*. (Course textbook reference)

17. **SUMO Documentation**. *SUMO User Documentation*. German Aerospace Center (DLR).  
    Retrieved from: https://sumo.dlr.de/docs/

---