# Roundabout Comparison & Video Generation Guide

## Overview
This guide shows how to generate comparison graphs between text-based and SUMO simulations, and create SUMO visualization videos.

---

## 1. Generate Comparison Graphs

### Full Analysis (All Parameters)
```bash
python3 generate_roundabout_comparisons.py --output-dir results/text_vs_sumo_comparison
```

**What it does:**
- Runs text simulations for 90 configurations (3 lanes × 5 diameters × 6 arrival rates)
- Runs corresponding SUMO simulations
- Generates 8 comparison graphs

**Output:**
- `results/text_vs_sumo_comparison/text_simulation_results.csv`
- `results/text_vs_sumo_comparison/sumo_simulation_results.csv`
- `results/text_vs_sumo_comparison/plots/*.png` (8 graphs)

**Time:** ~45-60 minutes (180 total simulations)

### Skip Already-Run Simulations
```bash
# Skip text simulations (use existing results)
python3 generate_roundabout_comparisons.py --output-dir results/text_vs_sumo_comparison --skip-text

# Skip SUMO simulations (use existing results)
python3 generate_roundabout_comparisons.py --output-dir results/text_vs_sumo_comparison --skip-sumo

# Only generate plots from existing results
python3 generate_roundabout_comparisons.py --output-dir results/text_vs_sumo_comparison --skip-text --skip-sumo
```

---

## 2. Generated Graphs

### Metrics vs Arrival Rate (3 lines each: 1, 2, 3 lanes)
1. **Average Delay vs Arrival Rate** - `1_delay_vs_arrival.png`
2. **Throughput vs Arrival Rate** - `2_throughput_vs_arrival.png`
3. **95th Percentile Delay vs Arrival Rate** - `3_p95_delay_vs_arrival.png`
4. **Max Queue Length vs Arrival Rate** - `4_max_queue_vs_arrival.png`

### Metrics vs Diameter (at λ=0.10 veh/s/arm)
5. **Average Delay vs Diameter** - `5_delay_vs_diameter.png`
6. **Throughput vs Diameter** - `6_throughput_vs_diameter.png`
7. **95th Percentile Delay vs Diameter** - `7_p95_delay_vs_diameter.png`
8. **Max Queue Length vs Diameter** - `8_max_queue_vs_diameter.png`

**Graph Features:**
- Solid lines = Text simulation
- Dashed lines = SUMO simulation
- Circle markers = Text
- Square markers = SUMO
- Color coding: Blue=1-lane, Orange=2-lane, Green=3-lane

---

## 3. Generate SUMO Videos

### Interactive GUI Mode (Manual Recording)
```bash
python3 generate_sumo_video.py --lanes 2 --diameter 40 --arrival 0.15 --duration 300 --mode gui
```

**Instructions:**
1. SUMO-GUI opens automatically
2. Use screen recording software (e.g., OBS Studio, SimpleScreenRecorder)
3. Click "Start" to run simulation
4. Simulation auto-stops after 300s

### Screenshot Mode (Automated Captures)
```bash
python3 generate_sumo_video.py --lanes 2 --diameter 40 --arrival 0.15 --duration 300 --mode screenshots
```

**Output:** Screenshots every 60 seconds at `results/sumo_videos/video_*/screenshots/`

### Video Mode (Automated Video Creation)
```bash
python3 generate_sumo_video.py --lanes 2 --diameter 40 --arrival 0.15 --duration 300 --mode video
```

**Requires:** ffmpeg (`sudo apt install ffmpeg`)

**Output:** `results/sumo_videos/video_*.mp4`

---

## 4. Example Scenarios

### Low Demand (Smooth Flow)
```bash
python3 generate_sumo_video.py --lanes 1 --diameter 40 --arrival 0.10 --duration 300 --mode gui
```

### Moderate Demand (Some Queuing)
```bash
python3 generate_sumo_video.py --lanes 2 --diameter 40 --arrival 0.20 --duration 300 --mode gui
```

### High Demand (Heavy Congestion)
```bash
python3 generate_sumo_video.py --lanes 2 --diameter 30 --arrival 0.30 --duration 300 --mode gui
```

### Multi-Lane Comparison
```bash
# 1-lane
python3 generate_sumo_video.py --lanes 1 --diameter 40 --arrival 0.15 --duration 180 --mode video

# 2-lane
python3 generate_sumo_video.py --lanes 2 --diameter 40 --arrival 0.15 --duration 180 --mode video

# 3-lane
python3 generate_sumo_video.py --lanes 3 --diameter 40 --arrival 0.15 --duration 180 --mode video
```

---

## 5. Parameter Reference

### Arrival Rates (veh/s per arm)
- `0.05` = 180 veh/hr = Very light
- `0.10` = 360 veh/hr = Light
- `0.15` = 540 veh/hr = Moderate
- `0.20` = 720 veh/hr = Moderate-heavy
- `0.25` = 900 veh/hr = Heavy
- `0.30` = 1080 veh/hr = Very heavy (likely congestion)

### Diameters
- `20m` = Very small (minimum practical size)
- `30m` = Small
- `40m` = Medium (typical)
- `50m` = Large
- `60m` = Very large

### Lane Counts
- `1` = Single-lane roundabout (simplest)
- `2` = Two-lane roundabout (common urban)
- `3` = Three-lane roundabout (high-capacity)

---

## 6. Quick Start Commands

### Generate All Comparison Graphs
```bash
# Full run (~60 minutes)
python3 generate_roundabout_comparisons.py

# Check progress
tail -f comparison_generation.log
```

### Create Example Video
```bash
# Interactive (best for demonstrations)
python3 generate_sumo_video.py --lanes 2 --diameter 40 --arrival 0.15 --mode gui

# Or automated video file
python3 generate_sumo_video.py --lanes 2 --diameter 40 --arrival 0.15 --mode video
```

---

## 7. Viewing Results

### Comparison Graphs
```bash
# View all plots
ls results/text_vs_sumo_comparison/plots/

# Open specific graph
xdg-open results/text_vs_sumo_comparison/plots/1_delay_vs_arrival.png
```

### Simulation Data
```bash
# View text simulation results
head -20 results/text_vs_sumo_comparison/text_simulation_results.csv

# View SUMO simulation results
head -20 results/text_vs_sumo_comparison/sumo_simulation_results.csv

# Summary statistics
python3 -c "
import pandas as pd
text = pd.read_csv('results/text_vs_sumo_comparison/text_simulation_results.csv')
sumo = pd.read_csv('results/text_vs_sumo_comparison/sumo_simulation_results.csv')
print('Text Simulation Stats:')
print(text.describe())
print('\nSUMO Simulation Stats:')
print(sumo.describe())
"
```

### Videos
```bash
# List generated videos
ls results/sumo_videos/*.mp4

# Play video
vlc results/sumo_videos/video_2lane_d40_arr0.15.mp4

# Or use default player
xdg-open results/sumo_videos/video_2lane_d40_arr0.15.mp4
```

---

## 8. Troubleshooting

### "No module named 'sumolib'"
```bash
pip install sumolib traci
```

### "sumo: command not found"
```bash
# Ubuntu/Debian
sudo apt install sumo sumo-tools sumo-gui

# Or download from: https://eclipse.dev/sumo/
```

### "ffmpeg: command not found"
```bash
sudo apt install ffmpeg
```

### Simulations timing out
- Reduce `--horizon` or `--duration`
- Reduce number of parameter combinations
- Check if system is overloaded

### SUMO-GUI won't open
- Check X11 forwarding if using SSH: `ssh -X user@host`
- Try headless mode: use `--mode video` instead of `--mode gui`

---

## 9. Advanced Usage

### Custom Parameter Ranges
Edit `generate_roundabout_comparisons.py`:
```python
# Line ~38-42
self.arrival_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
self.diameters = [20, 30, 40, 50, 60]
self.lane_counts = [1, 2, 3]
self.fixed_arrival = 0.10  # For diameter sweep
```

### Higher Resolution Videos
Edit `generate_sumo_video.py`:
```python
# Line ~180 (in generate_screenshots)
'--window-size', '1920,1080',  # Change to 2560,1440 or 3840,2160
```

### Longer Simulations
```bash
python3 generate_roundabout_comparisons.py --output-dir results/long_run
# Then edit the script to change horizon from 1200 to 3600
```

---

## 10. Integration with LaTeX Report

### Include Graphs in Report
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{results/text_vs_sumo_comparison/plots/1_delay_vs_arrival.png}
    \caption{Average Delay vs Arrival Rate: Text vs SUMO Comparison}
    \label{fig:delay_arrival}
\end{figure}
```

### Batch Include All Graphs
```bash
cd results/text_vs_sumo_comparison/plots/
for f in *.png; do
    echo "\\begin{figure}[H]"
    echo "    \\centering"
    echo "    \\includegraphics[width=0.95\\textwidth]{results/text_vs_sumo_comparison/plots/$f}"
    echo "    \\caption{${f%.png}}"
    echo "    \\label{fig:${f%.png}}"
    echo "\\end{figure}"
    echo ""
done
```

---

## Summary

**To generate all comparison graphs:**
```bash
python3 generate_roundabout_comparisons.py
```

**To create a demo video:**
```bash
python3 generate_sumo_video.py --lanes 2 --diameter 40 --arrival 0.15 --mode gui
```

**Results locations:**
- Graphs: `results/text_vs_sumo_comparison/plots/`
- Data: `results/text_vs_sumo_comparison/*.csv`
- Videos: `results/sumo_videos/`

**Time estimates:**
- Full comparison: ~60 minutes
- Single video: ~5 minutes
- Plot generation only: <1 minute
