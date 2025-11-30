#!/bin/bash
# Quick SUMO video demo generator

cd /home/jaathavan/Projects/Compative-Analysis-of-Street-Intersections-and-Optimization-Strategies

LANES=2
DIAMETER=40
ARRIVAL=0.15
OUTPUT_DIR="results/sumo_demo"

echo "======================================================================"
echo "SUMO Roundabout Video Demo Generator"
echo "======================================================================"
echo "Configuration:"
echo "  Lanes: $LANES"
echo "  Diameter: ${DIAMETER}m"
echo "  Arrival Rate: $ARRIVAL veh/s/arm (540 veh/hr/arm)"
echo ""

mkdir -p "$OUTPUT_DIR"

# Generate network
echo "Step 1: Generating network..."
python3 roundabout/src/generate_network.py \
    --config roundabout/config/config.yaml \
    --diameter $DIAMETER \
    --lanes $LANES \
    --output "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "✗ Network generation failed"
    exit 1
fi

# Find the generated network file
NET_FILE=$(find "$OUTPUT_DIR" -name "roundabout.net.xml" | head -1)
if [ -z "$NET_FILE" ]; then
    echo "✗ Network file not found"
    exit 1
fi
echo "✓ Network: $NET_FILE"

# Generate routes
echo ""
echo "Step 2: Generating routes..."
DEMAND_VEHHR=$(echo "$ARRIVAL * 3600" | bc)
python3 roundabout/src/generate_routes.py \
    --config roundabout/config/config.yaml \
    --demand $DEMAND_VEHHR \
    --duration 300 \
    --output "$OUTPUT_DIR/routes.rou.xml"

if [ $? -ne 0 ]; then
    echo "✗ Route generation failed"
    exit 1
fi
echo "✓ Routes: $OUTPUT_DIR/routes.rou.xml"

# Create SUMO config
echo ""
echo "Step 3: Creating SUMO configuration..."
cat > "$OUTPUT_DIR/demo.sumocfg" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="$(basename $(dirname $NET_FILE))/roundabout.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="300"/>
        <step-length value="0.1"/>
    </time>
    <output>
        <summary-output value="summary.xml"/>
        <tripinfo-output value="tripinfo.xml"/>
    </output>
    <gui_only>
        <start value="true"/>
        <delay value="100"/>
    </gui_only>
</configuration>
EOF

echo "✓ Config: $OUTPUT_DIR/demo.sumocfg"

# Launch SUMO-GUI
echo ""
echo "======================================================================"
echo "Launching SUMO-GUI..."
echo "======================================================================"
echo ""
echo "Instructions:"
echo "  1. The simulation will start automatically"
echo "  2. Use these controls:"
echo "     - Play/Pause: Click the play button or press Space"
echo "     - Speed: Use the delay slider (lower = faster)"
echo "     - Zoom: Mouse wheel or +/- keys"
echo "  3. To record:"
echo "     - Use screen recording software (OBS, SimpleScreenRecorder, etc.)"
echo "     - Or: Edit > Edit Visualization > OpenGL > Enable Screenshots"
echo "  4. Simulation runs for 300 seconds (5 minutes)"
echo ""
echo "Starting in 3 seconds..."
sleep 3

cd "$OUTPUT_DIR"
sumo-gui -c demo.sumocfg --start --delay 100 --quit-on-end

echo ""
echo "======================================================================"
echo "Demo Complete!"
echo "======================================================================"
echo ""
echo "Configuration saved in: $OUTPUT_DIR"
echo ""
echo "To run again: sumo-gui -c $OUTPUT_DIR/demo.sumocfg"
echo "To record video: Use OBS Studio or SimpleScreenRecorder"
