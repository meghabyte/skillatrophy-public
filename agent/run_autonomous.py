#!/usr/bin/env python
"""
run_autonomous.py

Standalone script to run the NMPC parking agent in fully autonomous mode
(no human input). This demonstrates the agent that forms the basis of the
shared autonomy system used in the user study.

The agent uses a two-phase approach:
  1. Approach: Drive forward from spawn to a staging position near the parking spot.
  2. Park: Execute an NMPC-guided parallel parking maneuver into the target spot.

Requirements:
  - CARLA 0.9.15 running with Town15 loaded.

Usage:
  python run_autonomous.py
"""

import os
import sys
import math

# -- Path setup --
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

import carla
from parking_agent import run_two_phase_parking


def main():
    # Ego vehicle spawn location
    ego_spawn_transform = carla.Transform(
        carla.Location(x=-352.0, y=144.1, z=160.0),
        carla.Rotation(yaw=180)
    )

    # Staging pose for approach phase [x, y, theta_radians]
    staging_pose = [-370.0, 144.7, math.radians(180)]

    # Final goal pose [x, y, theta_radians]
    goal_pose = [-363.3, 141.2, math.radians(180)]

    # Run the two-phase autonomous parking system
    success = run_two_phase_parking(
        ego_spawn_transform=ego_spawn_transform,
        staging_pose=staging_pose,
        goal_pose=goal_pose,
        timeout_s=40.0
    )

    if success:
        print("=== AUTONOMOUS PARKING COMPLETED SUCCESSFULLY ===")
    else:
        print("=== AUTONOMOUS PARKING FAILED ===")


if __name__ == "__main__":
    main()
