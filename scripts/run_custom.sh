#!/bin/bash

# --- Function to check if CARLA is running ---
carla_is_running() {
    pgrep -f "CarlaUE4-Linux-Shipping" > /dev/null
}

# --- Start simulator in the background ---
cd /home/driving_sim/CARLA_0.9.15/ || exit 1
./CarlaUE4.sh -prefernvidia -quality-level=Medium &
carla_pid=$!  # Store the PID

# --- Wait for the simulator to initialize (more robust than sleep) ---
echo "Waiting for CARLA simulator to start..."

sleep 10

echo "CARLA simulator started (hopefully...)"

# --- Change Map to Town15 ---
echo "Changing map to Town15..."
cd /home/driving_sim/CARLA_0.9.15/PythonAPI/util/ || exit 1
./config.py --map Town15

if [ $? -ne 0 ]; then
    echo "Error: Failed to change the map."
    exit 1
fi

# --- Start the steeringwheel script ---
echo "Starting steeringwheel_custom.py..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1
python src/steeringwheel_custom.py