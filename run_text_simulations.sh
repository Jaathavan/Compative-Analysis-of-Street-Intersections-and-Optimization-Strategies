#!/bin/bash
# Quick comparison generation script
# Generates 8 comparison graphs for roundabouts

cd /home/jaathavan/Projects/Compative-Analysis-of-Street-Intersections-and-Optimization-Strategies

OUTPUT_DIR="results/text_vs_sumo_comparison"
mkdir -p "$OUTPUT_DIR/plots"

echo "======================================================================"
echo "ROUNDABOUT TEXT-BASED SIMULATION DATA COLLECTION"
echo "======================================================================"

# Run smaller set for testing (can expand later)
LANES=(1 2 3)
DIAMETERS=(30 40 50)
ARRIVALS=(0.10 0.15 0.20)

# Create CSV header
echo "lanes,diameter,arrival_rate,throughput,avg_delay,p95_delay,max_queue,simulator" > "$OUTPUT_DIR/text_simulation_results.csv"

COUNT=0
TOTAL=$((${#LANES[@]} * ${#DIAMETERS[@]} * ${#ARRIVALS[@]}))

for lanes in "${LANES[@]}"; do
    for diameter in "${DIAMETERS[@]}"; do
        for arrival in "${ARRIVALS[@]}"; do
            COUNT=$((COUNT + 1))
            echo ""
            echo "[$COUNT/$TOTAL] Running text: ${lanes}-lane, d=${diameter}m, λ=${arrival} veh/s/arm"
            
            # Run Roundabout.py
            output=$(python3 Roundabout.py \
                --lanes $lanes \
                --diameter $diameter \
                --arrival $arrival $arrival $arrival $arrival \
                --horizon 900 \
                --seed 42 2>&1)
            
            # Parse output
            throughput=$(echo "$output" | grep -oP 'throughput[=\s]+\K[\d.]+' | tail -1)
            avg_delay=$(echo "$output" | grep -oP 'avg_delay[=\s]+\K[\d.]+' | tail -1)
            p95_delay=$(echo "$output" | grep -oP 'p95[=\s]+\K[\d.]+' | tail -1)
            max_queue=$(echo "$output" | grep -oP 'max_queue_per_arm=\[\K[^\]]+' | tr ',' '\n' | sort -n | tail -1)
            
            if [[ -n "$throughput" && -n "$avg_delay" ]]; then
                echo "$lanes,$diameter,$arrival,$throughput,$avg_delay,$p95_delay,$max_queue,text" >> "$OUTPUT_DIR/text_simulation_results.csv"
                echo "  ✓ Throughput: $throughput veh/hr, Delay: $avg_delay s"
            else
                echo "  ✗ Failed to parse"
            fi
        done
    done
done

echo ""
echo "✓ Text simulations complete!"
echo ""
echo "Now run: python3 generate_comparison_plots.py"
