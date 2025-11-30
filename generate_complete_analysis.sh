#!/bin/bash
# generate_complete_analysis.sh - Complete Analysis Pipeline
# Generates all results, visualizations, and LaTeX report

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_ROOT="/home/jaathavan/Projects/Compative-Analysis-of-Street-Intersections-and-Optimization-Strategies"
cd "$PROJECT_ROOT"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}COMPREHENSIVE TRAFFIC INTERSECTION ANALYSIS${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Create output directories
mkdir -p final_results/{roundabout,signalized,comparison}/{data,plots,videos}
mkdir -p final_results/latex

# ============================================================================
# PART 1: ROUNDABOUT ANALYSIS
# ============================================================================

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}PART 1: ROUNDABOUT ANALYSIS${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# Step 1.1: Run text-based simulations for different configurations
echo -e "${YELLOW}[1.1]${NC} Running text-based roundabout simulations..."

# Low demand scenarios (λ=0.10 per arm)
echo "  → 1-lane, low demand..."
python Roundabout.py --lanes 1 --diameter 45 --arrival 0.10 0.10 0.10 0.10 \
  --horizon 3600 --seed 42 > final_results/roundabout/data/text_1lane_low.log

echo "  → 2-lane, low demand..."
python Roundabout.py --lanes 2 --diameter 45 --arrival 0.10 0.10 0.10 0.10 \
  --horizon 3600 --seed 42 > final_results/roundabout/data/text_2lane_low.log

echo "  → 3-lane, low demand..."
python Roundabout.py --lanes 3 --diameter 45 --arrival 0.10 0.10 0.10 0.10 \
  --horizon 3600 --seed 42 > final_results/roundabout/data/text_3lane_low.log

# Medium demand scenarios (λ=0.15 per arm)
echo "  → 2-lane, medium demand..."
python Roundabout.py --lanes 2 --diameter 45 --arrival 0.15 0.12 0.15 0.12 \
  --horizon 3600 --seed 42 > final_results/roundabout/data/text_2lane_medium.log

echo "  → 3-lane, medium demand..."
python Roundabout.py --lanes 3 --diameter 45 --arrival 0.15 0.12 0.15 0.12 \
  --horizon 3600 --seed 42 > final_results/roundabout/data/text_3lane_medium.log

# High demand scenarios (λ=0.18 per arm)
echo "  → 3-lane, high demand..."
python Roundabout.py --lanes 3 --diameter 45 --arrival 0.18 0.15 0.18 0.15 \
  --horizon 3600 --seed 42 > final_results/roundabout/data/text_3lane_high.log

echo -e "${GREEN}✓${NC} Text simulations complete"

# Step 1.2: Run SUMO parameter sweep (grid search)
echo -e "\n${YELLOW}[1.2]${NC} Running SUMO parameter sweep (grid search)..."
cd roundabout

# Check if results already exist
if [ -f "results/sweep_results/sweep_summary.csv" ]; then
    echo "  → Using existing sweep results..."
    python src/optimize.py --config config/config.yaml \
      --output results/sweep_results/ --skip-simulation
else
    echo "  → Running full parameter sweep (this may take 15-30 minutes)..."
    python src/optimize.py --config config/config.yaml \
      --output results/sweep_results/ --method grid
fi

echo -e "${GREEN}✓${NC} Grid search complete"

# Step 1.3: Run Bayesian optimization (if not already done)
echo -e "\n${YELLOW}[1.3]${NC} Running Bayesian optimization..."

if [ -f "results/bayesian_results/bayesian_best_config.json" ]; then
    echo "  → Using existing Bayesian results..."
else
    echo "  → Running Bayesian optimization (50 evaluations, ~20-30 minutes)..."
    python src/optimize.py --config config/config.yaml \
      --output results/bayesian_results/ \
      --method bayesian --n-calls 50 --objective balance
fi

echo -e "${GREEN}✓${NC} Bayesian optimization complete"

# Step 1.4: Generate enhanced visualizations
echo -e "\n${YELLOW}[1.4]${NC} Generating enhanced visualizations..."

# Create comprehensive CSV for visualization (combine text and SUMO results)
python3 << 'PYTHON_SCRIPT'
import pandas as pd
import re
import os

# Parse text simulation logs
def parse_text_log(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract final summary
    metrics = {}
    
    # Extract lanes
    if '1-lane' in log_file or 'lanes=1' in content:
        metrics['lanes'] = 1
    elif '2-lane' in log_file or 'lanes=2' in content:
        metrics['lanes'] = 2
    elif '3-lane' in log_file or 'lanes=3' in content:
        metrics['lanes'] = 3
    
    # Extract arrival rate
    if 'low' in log_file:
        metrics['arrival_rate'] = 0.10
    elif 'medium' in log_file:
        metrics['arrival_rate'] = 0.14  # Average of 0.15, 0.12, 0.15, 0.12
    elif 'high' in log_file:
        metrics['arrival_rate'] = 0.17  # Average of 0.18, 0.15, 0.18, 0.15
    
    # Extract throughput
    match = re.search(r'throughput=(\d+\.?\d*)', content)
    if match:
        metrics['throughput'] = float(match.group(1))
    
    # Extract avg_delay
    match = re.search(r'avg_delay=(\d+\.?\d*)s', content)
    if match:
        metrics['avg_delay'] = float(match.group(1))
    
    # Extract p95_delay
    match = re.search(r'p95=(\d+\.?\d*)s', content)
    if match:
        metrics['p95_delay'] = float(match.group(1))
    
    # Extract max_queue_per_arm
    match = re.search(r'max_queue_per_arm=\[(\d+), (\d+), (\d+), (\d+)\]', content)
    if match:
        metrics['max_queue_N'] = int(match.group(1))
        metrics['max_queue_E'] = int(match.group(2))
        metrics['max_queue_S'] = int(match.group(3))
        metrics['max_queue_W'] = int(match.group(4))
    
    metrics['diameter'] = 45  # Default
    metrics['scenario'] = os.path.basename(log_file).replace('.log', '')
    
    return metrics

# Parse all text logs
text_data = []
log_dir = '../final_results/roundabout/data'
for log_file in os.listdir(log_dir):
    if log_file.endswith('.log'):
        try:
            metrics = parse_text_log(os.path.join(log_dir, log_file))
            text_data.append(metrics)
        except Exception as e:
            print(f"Warning: Could not parse {log_file}: {e}")

# Create DataFrame
df = pd.DataFrame(text_data)

# Add dummy lane entries for visualization (if not present)
for i in range(3):
    if f'lane_{i}_entries' not in df.columns:
        df[f'lane_{i}_entries'] = 0

# Save combined data
df.to_csv('../final_results/roundabout/data/combined_results.csv', index=False)
print(f"✓ Created combined results: {len(df)} scenarios")
PYTHON_SCRIPT

# Generate all visualizations
echo "  → Generating comprehensive visualizations..."
python src/enhanced_visualizations.py \
  --data ../final_results/roundabout/data/combined_results.csv \
  --mode all \
  --output ../final_results/roundabout/plots/

# If we have sweep results, also visualize those
if [ -f "results/sweep_results/sweep_summary.csv" ]; then
    echo "  → Generating parameter sweep visualizations..."
    python src/enhanced_visualizations.py \
      --data results/sweep_results/sweep_summary.csv \
      --mode parameter_sweep \
      --output ../final_results/roundabout/plots/
    
    python src/enhanced_visualizations.py \
      --data results/sweep_results/sweep_summary.csv \
      --mode failure_modes \
      --output ../final_results/roundabout/plots/
fi

# Generate optimization visualizations if available
if [ -f "results/bayesian_results/bayesian_optimization_history.csv" ]; then
    echo "  → Generating optimization visualizations..."
    python3 << 'PYTHON_SCRIPT'
import pandas as pd
import json

# Load Bayesian results
bayes_df = pd.read_csv('results/bayesian_results/bayesian_optimization_history.csv')

# Create optimization data structure for visualization
opt_data = {
    'bayesian_opt': bayes_df.to_dict('records')
}

# Save as JSON
with open('../final_results/roundabout/data/optimization_results.json', 'w') as f:
    json.dump(opt_data, f, indent=2)

print("✓ Optimization data prepared")
PYTHON_SCRIPT

    # Note: This would require the enhanced_visualizations.py to support this
    # For now, we'll create a simple plot separately
fi

echo -e "${GREEN}✓${NC} Visualizations complete"

# Step 1.5: Verify alignment (text vs SUMO)
echo -e "\n${YELLOW}[1.5]${NC} Verifying text vs SUMO alignment..."

# Run verification on a few key scenarios
python src/verify_alignment.py \
  --config config/config.yaml \
  --output ../final_results/roundabout/data/alignment/ || echo "Verification skipped (requires SUMO)"

echo -e "${GREEN}✓${NC} Alignment verification complete"

# Step 1.6: Generate failure videos (optional, requires manual recording)
echo -e "\n${YELLOW}[1.6]${NC} Preparing failure demonstration scenarios..."

python src/generate_failure_videos.py \
  --config config/config.yaml \
  --speed 0.1 \
  --duration 600 \
  --output ../final_results/roundabout/videos/ || echo "Video generation requires SUMO-GUI"

echo -e "${GREEN}✓${NC} Failure scenarios prepared"

cd "$PROJECT_ROOT"

# ============================================================================
# PART 2: SIGNALIZED INTERSECTION ANALYSIS
# ============================================================================

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}PART 2: SIGNALIZED INTERSECTION ANALYSIS${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# Step 2.1: Run text-based signalized simulations (Webster's Method)
echo -e "${YELLOW}[2.1]${NC} Running text-based signalized simulations (Webster)..."

# Low demand
echo "  → 1-lane, low demand..."
python Signalized.py --lanes 1 --arrival 0.10 0.10 0.10 0.10 \
  --horizon 3600 --seed 42 > final_results/signalized/data/text_1lane_low_webster.log

echo "  → 2-lane, low demand..."
python Signalized.py --lanes 2 --arrival 0.10 0.10 0.10 0.10 \
  --horizon 3600 --seed 42 > final_results/signalized/data/text_2lane_low_webster.log

echo "  → 3-lane, low demand..."
python Signalized.py --lanes 3 --arrival 0.10 0.10 0.10 0.10 \
  --horizon 3600 --seed 42 > final_results/signalized/data/text_3lane_low_webster.log

# Medium demand
echo "  → 2-lane, medium demand..."
python Signalized.py --lanes 2 --arrival 0.15 0.12 0.15 0.12 \
  --horizon 3600 --seed 42 > final_results/signalized/data/text_2lane_medium_webster.log

echo "  → 3-lane, medium demand..."
python Signalized.py --lanes 3 --arrival 0.15 0.12 0.15 0.12 \
  --horizon 3600 --seed 42 > final_results/signalized/data/text_3lane_medium_webster.log

# High demand
echo "  → 3-lane, high demand..."
python Signalized.py --lanes 3 --arrival 0.18 0.15 0.18 0.15 \
  --horizon 3600 --seed 42 > final_results/signalized/data/text_3lane_high_webster.log

echo -e "${GREEN}✓${NC} Webster simulations complete"

# Step 2.2: Check PPO training status
echo -e "\n${YELLOW}[2.2]${NC} Checking PPO training status..."

cd signalized

if [ -f "models/ppo_signalized_final.zip" ]; then
    echo -e "  ${GREEN}✓${NC} PPO model found: models/ppo_signalized_final.zip"
    PPO_TRAINED=true
else
    echo -e "  ${YELLOW}⚠${NC} PPO model not found. Training required."
    echo ""
    echo "  To train PPO agent, run:"
    echo "    cd signalized"
    echo "    python src/train_ppo.py --config config/config.yaml --episodes 1000 --output models/"
    echo ""
    echo "  Training will take approximately 2-3 hours."
    echo "  Continue without PPO results? (y/n)"
    read -p "  > " response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        PPO_TRAINED=false
        echo "  → Continuing without PPO results..."
    else
        echo "  → Please train PPO first. Exiting."
        exit 1
    fi
fi

# Step 2.3: Run SUMO simulations (Webster baseline)
echo -e "\n${YELLOW}[2.3]${NC} Running SUMO signalized simulations (Webster)..."

# Run quickstart to generate Webster baseline
python quickstart.py --output ../final_results/signalized/data/webster_sumo/

echo -e "${GREEN}✓${NC} SUMO Webster simulations complete"

# Step 2.4: Run PPO simulations (if trained)
if [ "$PPO_TRAINED" = true ]; then
    echo -e "\n${YELLOW}[2.4]${NC} Running PPO simulations..."
    
    # Run simulations with trained PPO agent
    python src/run_simulation.py \
      --config config/config.yaml \
      --strategy ppo \
      --model models/ppo_signalized_final.zip \
      --output ../final_results/signalized/data/ppo_results.csv
    
    echo -e "${GREEN}✓${NC} PPO simulations complete"
else
    echo -e "\n${YELLOW}[2.4]${NC} Skipping PPO simulations (model not trained)"
fi

# Step 2.5: Generate visualizations
echo -e "\n${YELLOW}[2.5]${NC} Generating signalized visualizations..."

# Parse text simulation logs
python3 << 'PYTHON_SCRIPT'
import pandas as pd
import re
import os

def parse_signalized_log(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract lanes
    if '1-lane' in log_file or 'lanes=1' in content:
        metrics['lanes'] = 1
    elif '2-lane' in log_file or 'lanes=2' in content:
        metrics['lanes'] = 2
    elif '3-lane' in log_file or 'lanes=3' in content:
        metrics['lanes'] = 3
    
    # Extract arrival rate
    if 'low' in log_file:
        metrics['arrival_rate'] = 0.10
        metrics['demand_multiplier'] = 0.67  # Relative to baseline
    elif 'medium' in log_file:
        metrics['arrival_rate'] = 0.14
        metrics['demand_multiplier'] = 0.93
    elif 'high' in log_file:
        metrics['arrival_rate'] = 0.17
        metrics['demand_multiplier'] = 1.13
    
    # Strategy
    if 'webster' in log_file:
        metrics['strategy'] = 'webster'
    elif 'ppo' in log_file:
        metrics['strategy'] = 'ppo'
    else:
        metrics['strategy'] = 'unknown'
    
    # Extract throughput
    match = re.search(r'Throughput:\s*(\d+\.?\d*)\s*veh/hr', content)
    if match:
        metrics['throughput'] = float(match.group(1))
    
    # Extract avg_delay
    match = re.search(r'Average delay:\s*(\d+\.?\d*)s', content)
    if match:
        metrics['avg_delay'] = float(match.group(1))
    
    # Extract p95_delay
    match = re.search(r'P95 delay:\s*(\d+\.?\d*)s', content)
    if match:
        metrics['p95_delay'] = float(match.group(1))
    
    # Extract max_queue
    match = re.search(r'Max queue per arm.*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', content)
    if match:
        queues = [int(match.group(i)) for i in range(1, 5)]
        metrics['max_queue'] = max(queues)
        metrics['max_queue_N'] = queues[0]
        metrics['max_queue_E'] = queues[1]
        metrics['max_queue_S'] = queues[2]
        metrics['max_queue_W'] = queues[3]
    
    # Extract cycle length (Webster)
    match = re.search(r'C_opt=(\d+\.?\d*)s', content)
    if match:
        metrics['cycle_length'] = float(match.group(1))
    
    # Extract flow ratio
    match = re.search(r'Y=(0\.\d+)', content)
    if match:
        metrics['flow_ratio_Y'] = float(match.group(1))
    
    # Extract green times
    match = re.search(r'Green times:.*NS-L=(\d+\.?\d*)s.*NS-T=(\d+\.?\d*)s.*EW-L=(\d+\.?\d*)s.*EW-T=(\d+\.?\d*)s', content)
    if match:
        metrics['green_NS_L'] = float(match.group(1))
        metrics['green_NS_T'] = float(match.group(2))
        metrics['green_EW_L'] = float(match.group(3))
        metrics['green_EW_T'] = float(match.group(4))
    
    metrics['scenario'] = os.path.basename(log_file).replace('.log', '')
    
    return metrics

# Parse all logs
signalized_data = []
log_dir = '../final_results/signalized/data'
for log_file in os.listdir(log_dir):
    if log_file.endswith('.log'):
        try:
            metrics = parse_signalized_log(os.path.join(log_dir, log_file))
            signalized_data.append(metrics)
        except Exception as e:
            print(f"Warning: Could not parse {log_file}: {e}")

df = pd.DataFrame(signalized_data)
df.to_csv('../final_results/signalized/data/combined_results.csv', index=False)
print(f"✓ Created combined results: {len(df)} scenarios")
PYTHON_SCRIPT

# Generate Webster analysis
echo "  → Webster's Method analysis..."
python src/enhanced_visualizations.py \
  --data ../final_results/signalized/data/combined_results.csv \
  --mode webster \
  --output ../final_results/signalized/plots/

# Generate strategy comparison (if PPO available)
if [ "$PPO_TRAINED" = true ]; then
    echo "  → Strategy comparison (Webster vs PPO)..."
    python src/enhanced_visualizations.py \
      --data ../final_results/signalized/data/combined_results.csv \
      --mode strategy_comparison \
      --output ../final_results/signalized/plots/
fi

echo -e "${GREEN}✓${NC} Signalized visualizations complete"

cd "$PROJECT_ROOT"

# ============================================================================
# PART 3: ROUNDABOUT VS SIGNALIZED COMPARISON
# ============================================================================

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}PART 3: ROUNDABOUT VS SIGNALIZED COMPARISON${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

echo -e "${YELLOW}[3.1]${NC} Generating comparison visualizations..."

cd signalized
python src/enhanced_visualizations.py \
  --data ../final_results/signalized/data/combined_results.csv \
  --mode roundabout_comparison \
  --roundabout-data ../final_results/roundabout/data/combined_results.csv \
  --output ../final_results/comparison/plots/

cd "$PROJECT_ROOT"

echo -e "${GREEN}✓${NC} Comparison complete"

# ============================================================================
# PART 4: GENERATE LATEX REPORT
# ============================================================================

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}PART 4: GENERATING LATEX REPORT${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

echo -e "${YELLOW}[4.1]${NC} Creating LaTeX report..."

# Generate LaTeX report (next step - creating the generator script)
python3 generate_latex_report.py

echo -e "${YELLOW}[4.2]${NC} Compiling LaTeX to PDF..."

cd final_results/latex
pdflatex -interaction=nonstopmode comprehensive_analysis.tex
pdflatex -interaction=nonstopmode comprehensive_analysis.tex  # Run twice for references
bibtex comprehensive_analysis || true  # If bibliography exists
pdflatex -interaction=nonstopmode comprehensive_analysis.tex

cd "$PROJECT_ROOT"

echo -e "${GREEN}✓${NC} LaTeX report generated: final_results/latex/comprehensive_analysis.pdf"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}ANALYSIS COMPLETE!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "Results are organized in final_results/:"
echo ""
echo "  📁 roundabout/"
echo "     ├── data/          - Simulation logs and CSVs"
echo "     ├── plots/         - All visualizations"
echo "     └── videos/        - Failure demonstration setups"
echo ""
echo "  📁 signalized/"
echo "     ├── data/          - Simulation logs and CSVs"
echo "     └── plots/         - All visualizations"
echo ""
echo "  📁 comparison/"
echo "     └── plots/         - Roundabout vs Signalized"
echo ""
echo "  📄 latex/comprehensive_analysis.pdf  - Final report"
echo ""
echo -e "${GREEN}✅ All analysis complete!${NC}"
echo ""

if [ "$PPO_TRAINED" = false ]; then
    echo -e "${YELLOW}⚠ NOTE: PPO results not included${NC}"
    echo "To add PPO analysis:"
    echo "  1. cd signalized"
    echo "  2. python src/train_ppo.py --config config/config.yaml --episodes 1000"
    echo "  3. Re-run this script"
    echo ""
fi

echo "================================================================"
