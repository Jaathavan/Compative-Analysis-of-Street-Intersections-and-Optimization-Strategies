# 🚀 Quick Reference Card - Traffic Intersection Analysis

## 📋 Command Cheat Sheet

### Text Simulations

```bash
# ROUNDABOUT - Basic run
python Roundabout.py --lanes 2 --diameter 45 \
  --arrival 0.18 0.12 0.20 0.15 --horizon 3600

# ROUNDABOUT - Quick test (5 min)
python Roundabout.py --lanes 2 --horizon 300 \
  --arrival 0.10 0.10 0.10 0.10

# SIGNALIZED - With Webster optimization
python Signalized.py --lanes 2 \
  --arrival 0.18 0.12 0.20 0.15 --horizon 3600

# SIGNALIZED - Manual timing
python Signalized.py --lanes 2 --cycle-length 90 \
  --arrival 0.15 0.15 0.15 0.15
```

### Visualizations

```bash
# ROUNDABOUT - All visualizations
cd roundabout && python src/enhanced_visualizations.py \
  --data results/data.csv --mode all --output plots/

# ROUNDABOUT - Specific mode
cd roundabout && python src/enhanced_visualizations.py \
  --data results/data.csv --mode lane_analysis

# SIGNALIZED - Webster analysis
cd signalized && python src/enhanced_visualizations.py \
  --data results/data.csv --mode webster

# SIGNALIZED - Strategy comparison
cd signalized && python src/enhanced_visualizations.py \
  --data results/data.csv --mode strategy_comparison

# COMPARISON - Roundabout vs Signalized
cd signalized && python src/enhanced_visualizations.py \
  --data results/sig_data.csv \
  --mode roundabout_comparison \
  --roundabout-data ../roundabout/results/rb_data.csv
```

### Validation & Testing

```bash
# Verify text vs SUMO alignment
cd roundabout && python src/verify_alignment.py \
  --config config/config.yaml --output results/alignment/

# Generate failure demonstration videos
cd roundabout && python src/generate_failure_videos.py \
  --speed 0.1 --duration 600 --output videos/

# Run all tests
./test_all_components.sh
```

## 🎯 Key Parameters

### Roundabout
| Parameter | Flag | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| Lanes | `--lanes` | 2 | 1-3 | Circulating lanes |
| Diameter | `--diameter` | 45 | 20-90 | Diameter (m) |
| Arrival | `--arrival` | [0.18,0.12,0.20,0.15] | 0.05-0.25 | veh/s per arm [N,E,S,W] |
| Horizon | `--horizon` | 3600 | 300-7200 | Simulation time (s) |
| Turning | `--turning` | [0.25,0.55,0.20] | - | [Left, Through, Right] |

### Signalized
| Parameter | Flag | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| Lanes | `--lanes` | 2 | 1-3 | Lanes per approach |
| Arrival | `--arrival` | [0.18,0.12,0.20,0.15] | 0.05-0.25 | veh/s per arm |
| Cycle | `--cycle-length` | Auto | 60-180 | Cycle length (s) |
| Webster | `--use-webster` | True | - | Use Webster's Method |
| Horizon | `--horizon` | 3600 | 300-7200 | Simulation time (s) |

## 📊 Expected Metrics

### Roundabout (2 lanes, λ=0.10)
- **Throughput**: 1,200-1,400 veh/hr
- **Avg Delay**: 3-10s
- **P95 Delay**: 10-50s
- **Max Queue**: 2-5 veh per arm

### Signalized (2 lanes, λ=0.10)
- **Throughput**: 1,400-1,600 veh/hr
- **Avg Delay**: 30-50s
- **P95 Delay**: 60-90s
- **Max Queue**: 4-8 veh per arm
- **Cycle Length**: 60-80s

## 🎨 Visualization Modes

### Roundabout
- `lane_analysis` - Lane choice and utilization (6 panels)
- `parameter_sweep` - Diameter/lanes impact (9 panels)
- `optimization` - Grid search/Bayesian (6 panels)
- `failure_modes` - Queue divergence/saturation (6 panels)
- `comprehensive` - All metrics (12 panels)
- `all` - Generate everything

### Signalized
- `webster` - Webster's Method analysis (6 panels)
- `strategy_comparison` - Webster/PPO/Actuated (6 panels)
- `roundabout_comparison` - vs Roundabout (6 panels)
- `all` - Generate everything

## 🔍 Quick Diagnosis

### High Delays?
```bash
# Check if oversaturated
# Rule of thumb: λ_total < 0.85 × capacity
# 1-lane roundabout: λ < 0.11
# 2-lane roundabout: λ < 0.13
# 2-lane signalized: λ < 0.18

# Try adding lanes or reducing demand
python Roundabout.py --lanes 3 \
  --arrival 0.15 0.15 0.15 0.15
```

### Poor Lane Utilization?
```bash
# Generate lane analysis
cd roundabout && python src/enhanced_visualizations.py \
  --data results/data.csv --mode lane_analysis

# Check lane_X_entries in output
```

### Webster's Y ≥ 1.0?
```bash
# System oversaturated - reduce demand or add lanes
python Signalized.py --lanes 3 \
  --arrival 0.15 0.12 0.15 0.12
```

## 📁 Output Files

### Text Simulations
- **Terminal output**: Real-time metrics
- **No files**: Pipe to file if needed: `> output.log`

### Visualizations
- `lane_choice_analysis.png`
- `parameter_sweep_analysis.png`
- `optimization_results.png`
- `failure_modes.png`
- `comprehensive_comparison.png`
- `webster_analysis.png`
- `strategy_comparison.png`
- `roundabout_vs_signalized.png`

### Validation
- `alignment_comparison.json`
- `alignment_comparison.csv`
- `alignment_comparison.png`

## 🐛 Common Issues

### "No module named X"
```bash
cd roundabout && pip install -r requirements.txt
cd signalized && pip install -r requirements.txt
```

### "sumo-gui not found"
```bash
# Add SUMO to PATH
export PATH=$PATH:/usr/share/sumo/bin
export SUMO_HOME=/usr/share/sumo
```

### Visualization "KeyError"
```bash
# Check required CSV columns:
# - scenario, lanes, arrival_rate, throughput
# - avg_delay, p95_delay, max_queue_N/E/S/W

# Use test data generator in test_all_components.sh
```

## 💡 Pro Tips

1. **Fast testing**: Use `--horizon 300` (5 min) instead of 3600
2. **Parallel runs**: Run multiple seeds for statistical significance
3. **Data collection**: Pipe output to CSV for analysis
4. **Video recording**: Use OBS Studio for SUMO-GUI capture
5. **Comparison**: Always use same random seed for fair comparison

## 📖 Documentation Links

- **Main guide**: `ENHANCED_ANALYSIS_README.md`
- **Project summary**: `PROJECT_COMPLETE_SUMMARY.md`
- **Roundabout details**: `roundabout/README.md`
- **Signalized details**: `signalized/README.md`
- **Parameter mapping**: `roundabout/PARAMETER_MAPPING.md`
- **Optimization**: `roundabout/BAYESIAN_OPTIMIZATION.md`

## 🎓 Citation

If using this code in research, please cite:
```
Comparative Analysis of Street Intersections and Optimization Strategies
[Your Name], 2024
GitHub: [repository URL]
```

## 📞 Support

For issues or questions:
1. Check documentation files
2. Review `test_all_components.sh` for examples
3. Check log files in `/tmp/test_output_*.log`
4. Verify SUMO installation: `sumo --version`

---

**Version**: 1.0  
**Last Updated**: November 2024  
**Status**: Production Ready ✅
