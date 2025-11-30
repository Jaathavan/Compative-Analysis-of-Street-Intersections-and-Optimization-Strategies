#!/bin/bash
#
# run_complete_analysis.sh - Streamlined Analysis Pipeline
# =========================================================
#
# Executes complete intersection optimization analysis:
# 1. Text-based simulations (Roundabout + Signalized)
# 2. Creates visualizations
# 3. Generates LaTeX report
#
# Usage: bash run_complete_analysis.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESULTS_DIR="results"
VIZ_DIR="$RESULTS_DIR/visualizations"
REPORT_FILE="final_report.tex"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Intersection Optimization - Complete Analysis Pipeline${NC}"
echo -e "${BLUE}============================================================${NC}"

# Create results directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$VIZ_DIR"

#=============================================================================
# PART 1: ROUNDABOUT TEXT SIMULATIONS
#=============================================================================

echo -e "\n${GREEN}[1/4] Running Roundabout Text Simulations...${NC}"

# Create CSV header
echo "diameter,lanes,arrival_rate,throughput,avg_delay,p95_delay,max_queue,status" > "$RESULTS_DIR/roundabout_text_results.csv"

# Test configurations
# Format: diameter lanes arrival_rate
configs=(
    "30 1 0.15"
    "40 1 0.15"
    "50 1 0.15"
    "40 2 0.25"
    "50 2 0.25"
    "50 3 0.35"
)

for config in "${configs[@]}"; do
    read -r diam lanes arr <<< "$config"
    echo -e "  ${YELLOW}→${NC} Testing: ${diam}m, ${lanes} lane(s), ${arr} veh/s"
    
    # Run simulation
    output=$(python3 Roundabout.py --diameter $diam --lanes $lanes \
             --arrival $arr $arr $arr $arr --horizon 600 2>&1 | tail -5)
    
    # Parse results
    throughput=$(echo "$output" | grep "throughput=" | sed 's/.*throughput=\([0-9]*\).*/\1/')
    avg_delay=$(echo "$output" | grep "avg_delay=" | sed 's/.*avg_delay=\([0-9.]*\)s.*/\1/')
    p95_delay=$(echo "$output" | grep "p95=" | sed 's/.*p95=\([0-9.]*\)s.*/\1/')
    max_q=$(echo "$output" | grep "max_queue_per_arm=" | sed 's/.*max_queue_per_arm=\[\([0-9, ]*\)\].*/\1/')
    
    # Determine status (failure if delay > 300s or queue > 100)
    status="success"
    if [ -n "$avg_delay" ] && [ $(echo "$avg_delay > 300" | awk '{print ($1 > $2)}') -eq 1 ]; then
        status="failure"
    fi
    
    # Append to CSV
    echo "$diam,$lanes,$arr,$throughput,$avg_delay,$p95_delay,\"$max_q\",$status" >> "$RESULTS_DIR/roundabout_text_results.csv"
done

echo -e "  ${GREEN}✓${NC} Roundabout simulations complete: $RESULTS_DIR/roundabout_text_results.csv"

#=============================================================================
# PART 2: SIGNALIZED TEXT SIMULATIONS
#=============================================================================

echo -e "\n${GREEN}[2/4] Running Signalized Text Simulations (Webster's Method)...${NC}"

# Create CSV header
echo "lanes,arrival_rate,cycle_length,throughput,avg_delay,p95_delay,max_queue,status" > "$RESULTS_DIR/signalized_text_results.csv"

# Test configurations
sig_configs=(
    "1 0.20"
    "1 0.25"
    "1 0.30"
    "2 0.35"
    "2 0.40"
    "3 0.50"
)

for config in "${sig_configs[@]}"; do
    read -r lanes arr <<< "$config"
    echo -e "  ${YELLOW}→${NC} Testing: ${lanes} lane(s), ${arr} veh/s (Webster)"
    
    # Run simulation with Webster's Method
    output=$(python3 Signalized.py --lanes $lanes --arrival $arr $arr $arr $arr \
             --use-webster --horizon 600 2>&1)
    
    # Parse results
    cycle=$(echo "$output" | grep "Cycle length:" | sed 's/.*Cycle length: \([0-9.]*\)s.*/\1/')
    throughput=$(echo "$output" | grep "Throughput:" | sed 's/.*Throughput: \([0-9]*\).*/\1/')
    avg_delay=$(echo "$output" | grep "Average delay:" | sed 's/.*Average delay: \([0-9.]*\)s.*/\1/')
    p95_delay=$(echo "$output" | grep "P95 delay:" | sed 's/.*P95 delay: \([0-9.]*\)s.*/\1/')
    max_q=$(echo "$output" | grep "Max queue per arm" | sed 's/.*: \[\([0-9, ]*\)\].*/\1/')
    
    # Determine status
    status="success"
    if [ -z "$throughput" ] || [ -z "$avg_delay" ]; then
        status="failure"
        throughput=${throughput:-0}
        avg_delay=${avg_delay:-999}
        p95_delay=${p95_delay:-999}
    elif [ $(echo "$avg_delay > 300" | awk '{print ($1 > $2)}') -eq 1 ]; then
        status="failure"
    fi
    
    # Append to CSV
    echo "$lanes,$arr,$cycle,$throughput,$avg_delay,$p95_delay,\"$max_q\",$status" >> "$RESULTS_DIR/signalized_text_results.csv"
done

echo -e "  ${GREEN}✓${NC} Signalized simulations complete: $RESULTS_DIR/signalized_text_results.csv"

#=============================================================================
# PART 3: GENERATE VISUALIZATIONS
#=============================================================================

echo -e "\n${GREEN}[3/4] Generating Visualizations...${NC}"

# Create basic comparison plots using matplotlib
python3 << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
rb_df = pd.read_csv('results/roundabout_text_results.csv')
sig_df = pd.read_csv('results/signalized_text_results.csv')

# Create visualization directory
Path('results/visualizations').mkdir(parents=True, exist_ok=True)

# 1. Delay Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Roundabout
for lanes in rb_df['lanes'].unique():
    data = rb_df[rb_df['lanes'] == lanes]
    ax1.plot(data['arrival_rate'] * 3600, data['avg_delay'], 
             marker='o', label=f'{lanes} lane(s)', linewidth=2)

ax1.set_xlabel('Arrival Rate (veh/hr per approach)', fontsize=12)
ax1.set_ylabel('Average Delay (seconds)', fontsize=12)
ax1.set_title('Roundabout Performance', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=120, color='r', linestyle='--', alpha=0.5, label='Failure threshold')

# Signalized
for lanes in sig_df['lanes'].unique():
    data = sig_df[sig_df['lanes'] == lanes]
    ax2.plot(data['arrival_rate'] * 3600, data['avg_delay'], 
             marker='s', label=f'{lanes} lane(s)', linewidth=2)

ax2.set_xlabel('Arrival Rate (veh/hr per approach)', fontsize=12)
ax2.set_ylabel('Average Delay (seconds)', fontsize=12)
ax2.set_title('Signalized (Webster) Performance', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=120, color='r', linestyle='--', alpha=0.5, label='Failure threshold')

plt.tight_layout()
plt.savefig('results/visualizations/delay_comparison.png', dpi=300, bbox_inches='tight')
print('  ✓ Created: delay_comparison.png')

# 2. Throughput Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Roundabout
for lanes in rb_df['lanes'].unique():
    data = rb_df[rb_df['lanes'] == lanes]
    ax1.plot(data['arrival_rate'] * 3600, data['throughput'], 
             marker='o', label=f'{lanes} lane(s)', linewidth=2)

ax1.set_xlabel('Arrival Rate (veh/hr per approach)', fontsize=12)
ax1.set_ylabel('Throughput (veh/hr)', fontsize=12)
ax1.set_title('Roundabout Throughput', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Signalized
for lanes in sig_df['lanes'].unique():
    data = sig_df[sig_df['lanes'] == lanes]
    ax2.plot(data['arrival_rate'] * 3600, data['throughput'], 
             marker='s', label=f'{lanes} lane(s)', linewidth=2)

ax2.set_xlabel('Arrival Rate (veh/hr per approach)', fontsize=12)
ax2.set_ylabel('Throughput (veh/hr)', fontsize=12)
ax2.set_title('Signalized (Webster) Throughput', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/visualizations/throughput_comparison.png', dpi=300, bbox_inches='tight')
print('  ✓ Created: throughput_comparison.png')

# 3. Direct Comparison (1-lane)
fig, ax = plt.subplots(figsize=(10, 6))

rb_1lane = rb_df[rb_df['lanes'] == 1]
sig_1lane = sig_df[sig_df['lanes'] == 1]

ax.plot(rb_1lane['arrival_rate'] * 3600, rb_1lane['avg_delay'], 
        marker='o', label='Roundabout (1 lane)', linewidth=2, color='blue')
ax.plot(sig_1lane['arrival_rate'] * 3600, sig_1lane['avg_delay'], 
        marker='s', label='Signalized (1 lane, Webster)', linewidth=2, color='red')

ax.set_xlabel('Arrival Rate (veh/hr per approach)', fontsize=12)
ax.set_ylabel('Average Delay (seconds)', fontsize=12)
ax.set_title('Roundabout vs Signalized: Single-Lane Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=120, color='gray', linestyle='--', alpha=0.5, label='Failure threshold')

plt.tight_layout()
plt.savefig('results/visualizations/roundabout_vs_signalized.png', dpi=300, bbox_inches='tight')
print('  ✓ Created: roundabout_vs_signalized.png')

# 4. Webster Cycle Length Analysis
fig, ax = plt.subplots(figsize=(10, 6))

for lanes in sig_df['lanes'].unique():
    data = sig_df[sig_df['lanes'] == lanes]
    ax.plot(data['arrival_rate'] * 3600, data['cycle_length'], 
            marker='o', label=f'{lanes} lane(s)', linewidth=2)

ax.set_xlabel('Arrival Rate (veh/hr per approach)', fontsize=12)
ax.set_ylabel('Optimal Cycle Length (seconds)', fontsize=12)
ax.set_title("Webster's Method: Optimal Cycle Length", fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=180, color='r', linestyle='--', alpha=0.5, label='Maximum cycle')

plt.tight_layout()
plt.savefig('results/visualizations/webster_analysis.png', dpi=300, bbox_inches='tight')
print('  ✓ Created: webster_analysis.png')

print('\n  All visualizations saved to: results/visualizations/')
EOF

echo -e "  ${GREEN}✓${NC} Visualizations complete"

#=============================================================================
# PART 4: GENERATE LATEX REPORT
#=============================================================================

echo -e "\n${GREEN}[4/4] Generating LaTeX Report...${NC}"

python3 generate_latex_report.py --results-dir "$RESULTS_DIR" --output "$REPORT_FILE"

echo -e "  ${GREEN}✓${NC} LaTeX report generated: $REPORT_FILE"

#=============================================================================
# FINAL SUMMARY
#=============================================================================

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✓ Analysis Complete!${NC}"
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${YELLOW}Results Summary:${NC}"
echo -e "  📊 Roundabout results: $RESULTS_DIR/roundabout_text_results.csv"
echo -e "  📊 Signalized results: $RESULTS_DIR/signalized_text_results.csv"
echo -e "  📈 Visualizations: $VIZ_DIR/"
echo -e "  📄 LaTeX report: $REPORT_FILE"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. Review results in: ${BLUE}$RESULTS_DIR/${NC}"
echo -e "  2. View visualizations: ${BLUE}$VIZ_DIR/${NC}"
echo -e "  3. Compile PDF:"
echo -e "     ${BLUE}pdflatex $REPORT_FILE${NC}"
echo -e "     ${BLUE}pdflatex $REPORT_FILE${NC}  ${YELLOW}# Run twice for TOC${NC}"
echo -e "  4. View PDF: ${BLUE}${REPORT_FILE%.tex}.pdf${NC}"

echo -e "\n${GREEN}Done! 🎉${NC}"
