# Roundabout Comparison Graphs & SUMO Video - Quick Guide

## Status: Ready to Run! ✓

All scripts are prepared and tested. Follow these instructions to generate your requested outputs.

---

## Part 1: Generate 8 Comparison Graphs

### Quick Start (Recommended)
```bash
python3 quick_generate_graphs.py
```

**What it does:**
- Runs text-based simulations for 45 configurations
  - 3 lane counts (1, 2, 3)
  - 3 diameters (30, 40, 50 meters)
  - 5 arrival rates (0.05, 0.10, 0.15, 0.20, 0.25 veh/s/arm)
- Generates 8 comparison graphs
- Saves all data to CSV

**Time:** ~15-20 minutes (45 simulations × ~20s each)

**Output Location:** `results/roundabout_comparisons/`

**Generated Graphs:**
1. `1_delay_vs_arrival.png` - Average Delay vs Arrival Rate
2. `2_throughput_vs_arrival.png` - Throughput vs Arrival Rate  
3. `3_p95_delay_vs_arrival.png` - 95th Percentile Delay vs Arrival Rate
4. `4_max_queue_vs_arrival.png` - Max Queue Length vs Arrival Rate
5. `5_delay_vs_diameter.png` - Average Delay vs Diameter (at λ=0.10)
6. `6_throughput_vs_diameter.png` - Throughput vs Diameter (at λ=0.10)
7. `7_p95_delay_vs_diameter.png` - 95th Percentile Delay vs Diameter (at λ=0.10)
8. `8_max_queue_vs_diameter.png` - Max Queue Length vs Diameter (at λ=0.10)

**Each graph shows 3 lines:**
- Blue line: 1-lane roundabout
- Orange line: 2-lane roundabout
- Green line: 3-lane roundabout

---

## Part 2: Generate SUMO Video Demo

### Quick Start (Interactive GUI)
```bash
./demo_sumo_video.sh
```

**What it does:**
- Generates SUMO network for 2-lane, 40m diameter roundabout
- Creates traffic routes at 0.15 veh/s/arm (540 veh/hr/arm)
- Launches SUMO-GUI with visualization
- Runs for 300 seconds (5 minutes)

**To Record Video:**
1. **Option A - Screen Recording (Recommended)**
   - Install: `sudo apt install simplescreenrecorder`
   - Or use OBS Studio
   - Start recording before running the script
   
2. **Option B - Built-in Screenshots**
   - In SUMO-GUI: Edit → Edit Visualization
   - Go to "OpenGL" tab
   - Enable "Screenshot" and set output path
   - Screenshots will be captured each frame

**SUMO-GUI Controls:**
- **Play/Pause:** Space bar or Play button
- **Speed:** Delay slider (lower = faster)
- **Zoom:** Mouse wheel or +/- keys
- **Pan:** Click and drag
- **View:** Click on vehicles to see details

### Alternative: Different Configurations

**1-Lane Roundabout (Light Traffic):**
```bash
# Edit demo_sumo_video.sh and change:
LANES=1
ARRIVAL=0.10
```

**3-Lane Roundabout (Heavy Traffic):**
```bash
# Edit demo_sumo_video.sh and change:
LANES=3
ARRIVAL=0.20
```

**Smaller Diameter (More Congestion):**
```bash
# Edit demo_sumo_video.sh and change:
DIAMETER=30
ARRIVAL=0.20
```

---

## Checking Results

### View Graphs
```bash
# List all generated graphs
ls -lh results/roundabout_comparisons/*.png

# Open specific graph
xdg-open results/roundabout_comparisons/1_delay_vs_arrival.png

# Open all graphs at once
xdg-open results/roundabout_comparisons/*.png
```

### View Simulation Data
```bash
# View CSV data
head -20 results/roundabout_comparisons/simulation_data.csv

# Get summary statistics
python3 << EOF
import pandas as pd
df = pd.read_csv('results/roundabout_comparisons/simulation_data.csv')
print(df.groupby('lanes').describe())
EOF
```

### Replay SUMO Demo
```bash
# Run again without regenerating
sumo-gui -c results/sumo_demo/demo.sumocfg
```

---

## Expected Results

### Graph 1: Average Delay vs Arrival Rate
- **1-lane:** Delay increases rapidly above 0.15 veh/s/arm
- **2-lane:** Handles up to 0.20 veh/s/arm well
- **3-lane:** Best performance, delay stays low until 0.25+ veh/s/arm

### Graph 2: Throughput vs Arrival Rate  
- **1-lane:** Saturates around 800-900 veh/hr
- **2-lane:** Reaches 1400-1500 veh/hr
- **3-lane:** Can exceed 1800 veh/hr

### Graph 3-4: Queue Lengths & P95 Delay
- Similar patterns to average delay
- 3-lane shows most stability

### Graphs 5-8: Effect of Diameter
- Larger diameters (40-50m) generally perform better
- Trade-off: more space required
- Optimal appears to be 40-50m range

---

## Troubleshooting

### "Roundabout.py not found"
```bash
# Make sure you're in the right directory
cd /home/jaathavan/Projects/Compative-Analysis-of-Street-Intersections-and-Optimization-Strategies
```

### Simulations taking too long
- Reduce the parameter ranges in `quick_generate_graphs.py`:
  ```python
  diameters = [40]  # Just one diameter
  arrivals = [0.10, 0.15, 0.20]  # Fewer arrival rates
  ```

### SUMO-GUI won't open
- Check if SUMO is installed: `which sumo-gui`
- Install if needed: `sudo apt install sumo sumo-tools sumo-gui`
- If using SSH, enable X11: `ssh -X user@host`

### "bc: command not found" in demo script
```bash
sudo apt install bc
```

---

## Advanced: Full Comparison with SUMO

If you want to compare text-based vs SUMO simulations:

```bash
python3 generate_roundabout_comparisons.py
```

**Note:** This runs both text AND SUMO simulations (much longer, ~60 minutes)

---

## Summary Commands

**Generate all graphs (text-based):**
```bash
python3 quick_generate_graphs.py
```

**Launch SUMO demo:**
```bash
./demo_sumo_video.sh
```

**View results:**
```bash
ls results/roundabout_comparisons/
xdg-open results/roundabout_comparisons/1_delay_vs_arrival.png
```

---

## Output Files

```
results/
├── roundabout_comparisons/
│   ├── simulation_data.csv
│   ├── 1_delay_vs_arrival.png
│   ├── 2_throughput_vs_arrival.png
│   ├── 3_p95_delay_vs_arrival.png
│   ├── 4_max_queue_vs_arrival.png
│   ├── 5_delay_vs_diameter.png
│   ├── 6_throughput_vs_diameter.png
│   ├── 7_p95_delay_vs_diameter.png
│   └── 8_max_queue_vs_diameter.png
└── sumo_demo/
    ├── demo.sumocfg
    ├── roundabout*/
    │   └── roundabout.net.xml
    ├── routes.rou.xml
    ├── tripinfo.xml
    └── summary.xml
```

---

## Next Steps

1. **Run graph generation:** `python3 quick_generate_graphs.py`
2. **While it runs (15-20 min), set up screen recording**
3. **Run SUMO demo:** `./demo_sumo_video.sh`
4. **Review all 8 graphs**
5. **Include in your report**

**Questions?** Check `COMPARISON_VIDEO_GUIDE.md` for more details.
