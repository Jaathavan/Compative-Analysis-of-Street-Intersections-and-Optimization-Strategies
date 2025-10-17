# Installation & Setup Guide

## 🔧 Prerequisites

### 1. SUMO Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

**macOS (via Homebrew):**
```bash
brew install sumo
```

**From source or other methods:**
See official documentation: https://sumo.dlr.de/docs/Installing/index.html

### 2. Set SUMO_HOME Environment Variable

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# For Ubuntu/Debian (adjust path if needed)
export SUMO_HOME="/usr/share/sumo"

# For macOS Homebrew
export SUMO_HOME="/opt/homebrew/opt/sumo/share/sumo"

# Add SUMO tools to Python path
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

Reload your shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

Verify installation:
```bash
echo $SUMO_HOME
sumo --version
```

Expected output:
```
Eclipse SUMO sumo Version 1.x.x
```

### 3. Python Environment

**Python 3.8+ required**

Check version:
```bash
python3 --version
```

---

## 📦 Python Dependencies Installation

### Option 1: Using pip (Recommended)

```bash
cd roundabout
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
cd roundabout
conda create -n roundabout python=3.10
conda activate roundabout
pip install -r requirements.txt
```

### Verify Installation

```bash
python3 -c "import yaml, pandas, numpy, matplotlib, seaborn; print('✓ All core dependencies installed')"
python3 -c "import plotly; print('✓ Plotly installed (optional)')" || echo "⚠ Plotly not installed (interactive plots will be skipped)"
python3 -c "import traci; print('✓ TraCI (SUMO Python API) available')" || echo "✗ TraCI not found - check SUMO_HOME"
```

---

## 🚀 Quick Start

### Automated Demo

Run the quickstart script to test the complete pipeline:

```bash
cd roundabout
python3 quickstart.py
```

This will:
1. Generate a baseline roundabout network
2. Create traffic demand patterns
3. Run a 1-hour simulation
4. Analyze results
5. Generate visualizations

**With GUI visualization:**
```bash
python3 quickstart.py --gui
```

---

## 🧪 Manual Pipeline Walkthrough

### Step 1: Generate Network

Create a roundabout network:

```bash
python3 src/generate_network.py \
  --config config/config.yaml \
  --output sumo_configs/baseline
```

**With parameter overrides:**
```bash
python3 src/generate_network.py \
  --config config/config.yaml \
  --diameter 55 \
  --lanes 2 \
  --output sumo_configs/large_2lane
```

### Step 2: Generate Routes

Create traffic demand:

```bash
python3 src/generate_routes.py \
  --config config/config.yaml \
  --network sumo_configs/baseline/roundabout.net.xml \
  --output sumo_configs/baseline
```

**With demand scaling:**
```bash
python3 src/generate_routes.py \
  --config config/config.yaml \
  --network sumo_configs/baseline/roundabout.net.xml \
  --demand-multiplier 1.5 \
  --output sumo_configs/high_demand
```

### Step 3: Run Simulation

Execute simulation:

```bash
# Headless (for batch processing)
python3 src/run_simulation.py \
  --sumocfg sumo_configs/baseline/roundabout.sumocfg \
  --config config/config.yaml \
  --output results/raw/baseline.csv

# With GUI (for visualization)
python3 src/run_simulation.py \
  --sumocfg sumo_configs/baseline/roundabout.sumocfg \
  --config config/config.yaml \
  --gui \
  --output results/raw/baseline.csv
```

### Step 4: Analyze Results

Compute statistics and detect failures:

```bash
python3 src/analyze_results.py \
  --input results/raw/baseline.csv \
  --config config/config.yaml \
  --output results/summary.csv
```

**Analyze multiple scenarios (sweep):**
```bash
python3 src/analyze_results.py \
  --sweep "results/raw/*.csv" \
  --config config/config.yaml \
  --output results/sweep_summary.csv
```

### Step 5: Visualize

Generate plots:

```bash
# Static plots only
python3 src/visualize_results.py \
  --input results/summary.csv \
  --config config/config.yaml \
  --output results/plots/

# With interactive plots
python3 src/visualize_results.py \
  --input results/summary.csv \
  --window-data results/raw/baseline.csv \
  --config config/config.yaml \
  --interactive \
  --output results/plots/
```

### Step 6: Compare with Text Simulation

Direct comparison:

```bash
python3 src/compare_with_text_sim.py \
  --diameter 45 \
  --lanes 1 \
  --demand 1.0 \
  --output results/comparison.csv
```

---

## 🔬 Running Parameter Sweeps

### Full Automated Sweep

```bash
python3 src/optimize.py \
  --config config/config.yaml \
  --output results/sweep_results/
```

This uses sweep ranges from `config.yaml`:
- Diameters: [35, 45, 55] m
- Lanes: [1, 2]
- Demand multipliers: [0.5, 0.75, 1.0, 1.25, 1.5]

**Total scenarios:** 3 × 2 × 5 = 30

### Custom Sweep Ranges

```bash
python3 src/optimize.py \
  --config config/config.yaml \
  --diameters 40 50 60 \
  --lanes 1 \
  --demand-levels 0.8 1.0 1.2 \
  --output results/custom_sweep/
```

---

## 📊 Understanding Outputs

### Directory Structure After Quickstart

```
roundabout/
├── quickstart_output/
│   ├── sumo_configs/baseline/
│   │   ├── roundabout.net.xml      # Network geometry
│   │   ├── roundabout.rou.xml      # Traffic demand
│   │   └── roundabout.sumocfg      # SUMO configuration
│   ├── results/
│   │   ├── baseline.csv            # Window metrics (5-min)
│   │   ├── baseline_aggregate.csv  # Hourly summary
│   │   └── baseline_analysis.csv   # Analyzed results
│   └── plots/
│       ├── throughput_vs_demand.png
│       ├── delay_vs_demand.png
│       ├── queue_heatmap.png
│       ├── 3d_performance_surface.html  # Interactive
│       └── parameter_explorer.html      # Interactive
```

### Key Result Files

**Window metrics (`baseline.csv`):**
- Per 5-minute window statistics
- Columns: arrivals, exits, throughput_vph, avg_delay, p95_delay, max_queue_*, avg_speed_ring, CO2, fuel

**Aggregate metrics (`baseline_aggregate.csv`):**
- Hourly summary statistics
- Total arrivals/exits, mean delay, max queues, throughput

**Analysis (`baseline_analysis.csv`):**
- Derived metrics: trends, failure detection, performance classification
- Rankings and comparisons

---

## 🐛 Troubleshooting

### "SUMO_HOME not set"

**Solution:**
```bash
export SUMO_HOME="/usr/share/sumo"  # Adjust path
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

### "No module named 'traci'"

**Solution:**
```bash
# Ensure SUMO tools are in PYTHONPATH
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

# Verify
python3 -c "import traci; print(traci.__file__)"
```

### "TraCI server not reachable"

**Causes:**
- Another SUMO instance running on port 8813
- SUMO binary not in PATH

**Solution:**
```bash
# Kill existing SUMO processes
pkill -9 sumo

# Ensure SUMO binary is accessible
which sumo
```

### Empty simulation results (no vehicles)

**Causes:**
- Network edges not properly connected
- Route definitions invalid

**Solution:**
1. Check `.net.xml` visually in SUMO-GUI:
   ```bash
   sumo-gui -n sumo_configs/baseline/roundabout.net.xml
   ```

2. Validate route file:
   ```bash
   sumo -c sumo_configs/baseline/roundabout.sumocfg --duration-log.statistics
   ```

### Plotly import errors (interactive plots)

**Solution:**
```bash
pip install plotly
```

Or skip interactive plots:
```bash
python3 src/visualize_results.py --static-only ...
```

---

## 🎯 Next Steps

1. **Run baseline validation:**
   ```bash
   python3 quickstart.py
   ```

2. **Compare with text simulation:**
   ```bash
   python3 src/compare_with_text_sim.py --output results/comparison.csv
   ```

3. **Execute parameter sweep:**
   ```bash
   python3 src/optimize.py --output results/sweep_results/
   ```

4. **Customize parameters:**
   Edit `config/config.yaml` and re-run pipeline

5. **Generate report:**
   Use generated CSV files and plots for analysis document

---

## 📚 Documentation

- **Usage:** `roundabout/README.md`
- **Parameter mapping:** `roundabout/PARAMETER_MAPPING.md`
- **Configuration:** `roundabout/config/config.yaml`
- **SUMO docs:** https://sumo.dlr.de/docs/

---

## 🆘 Getting Help

If you encounter issues:

1. Check SUMO installation:
   ```bash
   sumo --version
   echo $SUMO_HOME
   ```

2. Verify Python dependencies:
   ```bash
   pip list | grep -E "pandas|numpy|matplotlib|yaml"
   ```

3. Review error logs in simulation output

4. Check SUMO documentation: https://sumo.dlr.de/docs/

---

*Installation guide for Phase 1: Roundabout Simulation*
