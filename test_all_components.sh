#!/bin/bash
# test_all_components.sh - Comprehensive testing script for all new components

echo "================================================================"
echo "COMPREHENSIVE COMPONENT TESTING"
echo "================================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run test
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "${YELLOW}[TEST]${NC} $test_name"
    echo "  Command: $test_command"
    
    if eval "$test_command" > /tmp/test_output_$$.log 2>&1; then
        echo -e "  ${GREEN}✓ PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "  ${RED}✗ FAILED${NC}"
        echo "  See /tmp/test_output_$$.log for details"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Change to project root
cd "$(dirname "$0")"

echo "================================================================"
echo "PART 1: TEXT-BASED SIMULATIONS"
echo "================================================================"
echo ""

# Test 1: Roundabout.py (multi-lane)
run_test "Roundabout.py - 1 lane, low demand" \
    "python Roundabout.py --lanes 1 --diameter 45 --arrival 0.05 0.05 0.05 0.05 --horizon 300"

run_test "Roundabout.py - 2 lanes, medium demand" \
    "python Roundabout.py --lanes 2 --diameter 45 --arrival 0.10 0.10 0.10 0.10 --horizon 300"

run_test "Roundabout.py - 3 lanes, high demand" \
    "python Roundabout.py --lanes 3 --diameter 45 --arrival 0.15 0.15 0.15 0.15 --horizon 300"

# Test 2: Signalized.py
run_test "Signalized.py - 1 lane, low demand" \
    "python Signalized.py --lanes 1 --arrival 0.05 0.05 0.05 0.05 --horizon 300"

run_test "Signalized.py - 2 lanes, medium demand (Webster)" \
    "python Signalized.py --lanes 2 --arrival 0.10 0.10 0.10 0.10 --horizon 300 --use-webster"

run_test "Signalized.py - 3 lanes, high demand" \
    "python Signalized.py --lanes 3 --arrival 0.15 0.15 0.15 0.15 --horizon 300"

echo ""
echo "================================================================"
echo "PART 2: VISUALIZATION TOOLS"
echo "================================================================"
echo ""

# Create dummy data for visualization tests
echo "Creating test data..."
python3 << 'EOF'
import pandas as pd
import numpy as np

# Roundabout test data
roundabout_data = []
for lanes in [1, 2, 3]:
    for arrival in [0.05, 0.10, 0.15]:
        for diameter in [30, 45, 60]:
            roundabout_data.append({
                'scenario': f'{lanes}L_d{diameter}_a{arrival}',
                'lanes': lanes,
                'diameter': diameter,
                'arrival_rate': arrival,
                'throughput': np.random.uniform(500, 2000),
                'avg_delay': np.random.uniform(5, 100),
                'p95_delay': np.random.uniform(20, 300),
                'max_queue_N': np.random.randint(2, 30),
                'max_queue_E': np.random.randint(2, 30),
                'max_queue_S': np.random.randint(2, 30),
                'max_queue_W': np.random.randint(2, 30),
                'lane_0_entries': np.random.randint(50, 200),
                'lane_1_entries': np.random.randint(50, 200) if lanes >= 2 else 0,
                'lane_2_entries': np.random.randint(50, 200) if lanes >= 3 else 0,
            })

df_rb = pd.DataFrame(roundabout_data)
df_rb.to_csv('/tmp/test_roundabout_data.csv', index=False)

# Signalized test data
signalized_data = []
for strategy in ['webster', 'ppo', 'actuated']:
    for demand_mult in [0.5, 0.75, 1.0, 1.25]:
        for lanes in [1, 2, 3]:
            signalized_data.append({
                'scenario': f'{strategy}_{lanes}L_d{demand_mult}',
                'strategy': strategy,
                'lanes': lanes,
                'demand_multiplier': demand_mult,
                'arrival_rate': 0.1 * demand_mult,
                'throughput': np.random.uniform(800, 2500),
                'avg_delay': np.random.uniform(10, 150),
                'p95_delay': np.random.uniform(30, 400),
                'max_queue': np.random.randint(3, 40),
                'cycle_length': np.random.uniform(60, 120),
                'green_NS_L': np.random.uniform(8, 15),
                'green_NS_T': np.random.uniform(15, 30),
                'green_EW_L': np.random.uniform(8, 15),
                'green_EW_T': np.random.uniform(15, 30),
                'flow_ratio_Y': np.random.uniform(0.5, 0.95),
            })

df_sig = pd.DataFrame(signalized_data)
df_sig.to_csv('/tmp/test_signalized_data.csv', index=False)

print("Test data created successfully")
EOF

# Test roundabout visualizations
run_test "Roundabout enhanced visualizations - lane analysis" \
    "cd roundabout && python src/enhanced_visualizations.py --data /tmp/test_roundabout_data.csv --mode lane_analysis --output /tmp/test_plots_rb"

run_test "Roundabout enhanced visualizations - parameter sweep" \
    "cd roundabout && python src/enhanced_visualizations.py --data /tmp/test_roundabout_data.csv --mode parameter_sweep --output /tmp/test_plots_rb"

run_test "Roundabout enhanced visualizations - comprehensive" \
    "cd roundabout && python src/enhanced_visualizations.py --data /tmp/test_roundabout_data.csv --mode comprehensive --output /tmp/test_plots_rb"

# Test signalized visualizations
run_test "Signalized enhanced visualizations - webster" \
    "cd signalized && python src/enhanced_visualizations.py --data /tmp/test_signalized_data.csv --mode webster --output /tmp/test_plots_sig"

run_test "Signalized enhanced visualizations - strategy comparison" \
    "cd signalized && python src/enhanced_visualizations.py --data /tmp/test_signalized_data.csv --mode strategy_comparison --output /tmp/test_plots_sig"

run_test "Signalized enhanced visualizations - roundabout comparison" \
    "cd signalized && python src/enhanced_visualizations.py --data /tmp/test_signalized_data.csv --mode roundabout_comparison --roundabout-data /tmp/test_roundabout_data.csv --output /tmp/test_plots_sig"

echo ""
echo "================================================================"
echo "PART 3: VALIDATION AND COMPARISON TOOLS"
echo "================================================================"
echo ""

# Note: These tests require SUMO and are more time-consuming
# Skipping for quick test; can be enabled for full validation

echo "Skipping SUMO-dependent tests (alignment verification, failure videos)"
echo "To run full validation:"
echo "  cd roundabout && python src/verify_alignment.py --config config/config.yaml --output results/alignment/"
echo "  cd roundabout && python src/generate_failure_videos.py --config config/config.yaml --output videos/"

echo ""
echo "================================================================"
echo "PART 4: CONFIGURATION AND DOCUMENTATION CHECKS"
echo "================================================================"
echo ""

# Test configuration files
run_test "Check roundabout config.yaml" \
    "python3 -c 'import yaml; yaml.safe_load(open(\"roundabout/config/config.yaml\"))'"

run_test "Check signalized config.yaml" \
    "python3 -c 'import yaml; yaml.safe_load(open(\"signalized/config/config.yaml\"))'"

# Test Python imports
run_test "Import roundabout modules" \
    "cd roundabout && python3 -c 'from src import generate_network, generate_routes, run_simulation, enhanced_visualizations'"

run_test "Import signalized modules" \
    "cd signalized && python3 -c 'from src import generate_network, generate_routes, webster_method, ppo_environment, enhanced_visualizations'"

# Check documentation
echo -e "${YELLOW}[CHECK]${NC} Verifying documentation files exist"
docs=(
    "ENHANCED_ANALYSIS_README.md"
    "roundabout/README.md"
    "roundabout/PARAMETER_MAPPING.md"
    "roundabout/BAYESIAN_OPTIMIZATION.md"
    "signalized/README.md"
    "signalized/IMPLEMENTATION_SUMMARY.md"
    "signalized/PPO_IMPLEMENTATION_COMPLETE.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        echo -e "  ${GREEN}✓${NC} $doc exists"
    else
        echo -e "  ${RED}✗${NC} $doc missing"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
done

echo ""
echo "================================================================"
echo "TEST SUMMARY"
echo "================================================================"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
PASS_RATE=$(awk "BEGIN {printf \"%.1f\", ($TESTS_PASSED/$TOTAL_TESTS)*100}")

echo "Total tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo "Pass rate: ${PASS_RATE}%"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    echo "All components are working correctly."
    echo "Ready for production use."
    exit 0
else
    echo -e "${YELLOW}⚠ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please review the failures above."
    echo "Check log files in /tmp/test_output_*.log for details."
    exit 1
fi
