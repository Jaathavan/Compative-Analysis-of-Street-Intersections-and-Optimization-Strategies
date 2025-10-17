# Comparative Analysis of Street Intersections and Optimization Strategies

## 🧭 Project Overview

**Title:** Comparative Analysis of Street Intersections and Optimization Strategies

**Goal:**
Simulate, analyze, and optimize both **roundabouts** and **4-way signalized intersections** to determine their efficiency under various traffic and geometric conditions.

The project has **three phases**:
1. **Roundabout Simulation & Optimization (Phase 1)** ✅ In Progress
2. **Signalized Intersection Optimization (Phase 2)** 🔜 Upcoming
3. **Real-World Intersection Application (Phase 3, optional)** 🔜 Future

---

## 📂 Project Structure

```
/roundabout/               # Phase 1: SUMO-based roundabout simulation
  ├── config/
  │   ├── config.yaml      # Central parameter definitions
  │   └── templates/       # SUMO XML templates
  ├── src/
  │   ├── generate_network.py      # Creates .net.xml from parameters
  │   ├── generate_routes.py       # Creates .rou.xml with demand patterns
  │   ├── run_simulation.py        # Executes SUMO via TraCI, collects metrics
  │   ├── analyze_results.py       # Computes aggregated statistics
  │   ├── visualize_results.py     # Generates all plots (static + interactive)
  │   ├── compare_with_text_sim.py # Side-by-side comparison with Roundabout.py
  │   └── optimize.py              # Orchestrates parameter sweeps
  ├── results/             # Auto-generated outputs
  │   ├── raw/             # Per-scenario CSVs
  │   ├── plots/           # Visualization outputs
  │   └── summary.csv      # Aggregated results table
  ├── sumo_configs/        # Generated SUMO files per scenario
  └── README.md            # Phase 1 usage instructions

/Roundabout.py             # Original text-based DDE simulation
/README.md                 # This file
```

---

## 🚀 Quick Start

See `/roundabout/README.md` for detailed Phase 1 usage instructions.

---

## 📊 Current Status

**Phase 1 (In Progress):**
- ✅ Parameter mapping documented (SUMO ↔ text simulation)
- ✅ Network generation pipeline
- ✅ Route generation with demand patterns
- ✅ TraCI simulation runner with windowed metrics
- ⏳ Analysis and comparison scripts
- ⏳ Visualization suite (static + interactive)
- ⏳ Parameter sweep optimization

---

## 📚 Documentation

- [Phase 1 Details](roundabout/README.md)
- [Parameter Mapping](roundabout/PARAMETER_MAPPING.md)
- [Text Simulation](Roundabout.py)
