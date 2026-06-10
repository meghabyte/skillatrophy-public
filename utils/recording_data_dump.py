#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import csv
import re
from collections import defaultdict

try:
    # Attempt to find the CARLA egg file and add it to the path
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse

def find_hero_id(all_data):
    """
    Analyzes vehicle animation data to dynamically find the ID of the most
    active vehicle, which is assumed to be the hero vehicle.
    """
    print("Analyzing data to identify hero vehicle...")
    activity_scores = defaultdict(int)
    vehicle_anim_pattern = re.compile(
        r"  Id: (\d+) Steering: (.+?) Throttle: (.+?) Brake: (.+?) Handbrake: (\d+) Gear: (\d+)"
    )

    for line in all_data.splitlines():
        match = vehicle_anim_pattern.match(line)
        if match:
            vehicle_id = int(match.group(1))
            steering = float(match.group(2))
            throttle = float(match.group(3))
            brake = float(match.group(4))

            if steering != 0.0 or throttle > 0.0 or (brake > 0.0 and brake < 1.0):
                activity_scores[vehicle_id] += 1
    
    if not activity_scores:
        raise RuntimeError("Could not find any vehicle animation data to determine the hero vehicle.")

    hero_id = max(activity_scores, key=activity_scores.get)
    return hero_id


def process_hero_data_and_write_csv(all_data, base_filepath, hero_id):
    """
    Parses position data (in cm) for the hero vehicle, converts it to meters,
    calculates instantaneous velocity, and writes all data to a single CSV file.
    """
    print(f"Filtering and processing data for Hero ID: {hero_id}...")
    
    hero_data_by_frame = defaultdict(dict)
    
    # --- STEP 1: Parse position and time data, converting units immediately ---
    current_frame = None
    frame_pattern = re.compile(r"Frame (\d+) at ([\d.]+) seconds")
    positions_pattern = re.compile(r"  Id: (\d+) Location: \((.+?), (.+?), (.+?)\) Rotation: \((.+?), (.+?), (.+?)\)")

    for line in all_data.splitlines():
        frame_match = frame_pattern.match(line)
        if frame_match:
            current_frame = int(frame_match.group(1))
            current_time = float(frame_match.group(2))
            hero_data_by_frame[current_frame]['time'] = current_time
            continue

        pos_match = positions_pattern.match(line)
        if pos_match and int(pos_match.group(1)) == hero_id:
            groups = pos_match.groups()
            
            # --- CRITICAL FIX: Convert location from CM to M ---
            loc_x_meters = float(groups[1]) / 100.0
            loc_y_meters = float(groups[2]) / 100.0
            loc_z_meters = float(groups[3]) / 100.0
            
            hero_data_by_frame[current_frame]['position'] = [loc_x_meters, loc_y_meters, loc_z_meters]
            hero_data_by_frame[current_frame]['rotation'] = [float(r) for r in groups[4:7]]
            continue
    
    # --- STEP 2: Calculate instantaneous velocity in m/s ---
    print("Calculating instantaneous velocity...")
    sorted_frames = sorted(hero_data_by_frame.keys())
    
    last_frame_data = None
    
    for frame in sorted_frames:
        current_frame_data = hero_data_by_frame[frame]
        
        if last_frame_data is None or 'position' not in current_frame_data or 'position' not in last_frame_data:
            current_frame_data['velocity'] = [0.0, 0.0, 0.0]
            last_frame_data = current_frame_data
            continue

        delta_time = current_frame_data['time'] - last_frame_data['time']
        
        if delta_time > 0:
            current_pos = current_frame_data['position']
            last_pos = last_frame_data['position']
            
            vel_x = (current_pos[0] - last_pos[0]) / delta_time
            vel_y = (current_pos[1] - last_pos[1]) / delta_time
            vel_z = (current_pos[2] - last_pos[2]) / delta_time
            current_frame_data['velocity'] = [vel_x, vel_y, vel_z]
        else:
            current_frame_data['velocity'] = last_frame_data.get('velocity', [0.0, 0.0, 0.0])

        last_frame_data = current_frame_data

    # --- STEP 3: Write the final data to the CSV file ---
    output_filename = f"{base_filepath}_hero_data.csv"
    print(f"Writing combined hero data to: {output_filename}")
    with open(output_filename, 'w', newline='') as f:
        header = [
            'Frame', 'Time', 'Id', 'Location_X (m)', 'Location_Y (m)', 'Location_Z (m)',
            'Rotation_Pitch', 'Heading (Yaw)', 'Rotation_Roll',
            'Velocity_X (m/s)', 'Velocity_Y (m/s)', 'Velocity_Z (m/s)'
        ]
        writer = csv.writer(f)
        writer.writerow(header)

        for frame in sorted_frames:
            data = hero_data_by_frame.get(frame)
            if not data: continue
            
            pos_data = data.get('position', [None]*3)
            rot_data = data.get('rotation', [None]*3)
            vel_data = data.get('velocity', [None]*3)
            time = data.get('time', 0.0)
            
            writer.writerow([frame, time, hero_id] + pos_data + rot_data + vel_data)


def main():
    argparser = argparse.ArgumentParser(
        description="Dumps position and calculated velocity data for the hero vehicle from a CARLA recording.")
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-f', '--recorder-filename', metavar='F', required=True,
        help='The .log file to read from the CARLA data directory (e.g., "recording.log")')
    argparser.add_argument(
        '-o', '--output-file', metavar='O', default=None,
        help='Optional: Base name for the output CSV file. Defaults to the recorder filename.')
    # --- NEW FEATURE: Manual Hero ID override ---
    argparser.add_argument(
        '--hero-id', type=int, default=None,
        help='Manually specify the actor ID of the hero vehicle. Overrides automatic detection.')

    args = argparser.parse_args()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        full_path = args.recorder_filename
        if not os.path.exists(full_path):
             search_path = os.path.join("CarlaUE4/Saved/records", args.recorder_filename)
             if os.path.exists(search_path):
                 full_path = search_path
             else:
                raise FileNotFoundError(f"Recording file not found at '{full_path}' or common search paths.")

        print(f"Reading all data from recording file: {full_path}")
        all_data = client.show_recorder_file_info(full_path, True)

        base_output_name = os.path.splitext(os.path.basename(args.recorder_filename))[0] if args.output_file is None else os.path.splitext(args.output_file)[0]
        output_dir = 'data_csv'
        os.makedirs(output_dir, exist_ok=True)
        base_filepath = os.path.join(output_dir, base_output_name)

        hero_id_to_filter = None
        if args.hero_id is not None:
            hero_id_to_filter = args.hero_id
            print(f"Using manually specified Hero Vehicle ID: {hero_id_to_filter}")
        else:
            hero_id_to_filter = find_hero_id(all_data)
            print(f"Successfully identified Hero Vehicle ID: {hero_id_to_filter}")
        
        process_hero_data_and_write_csv(all_data, base_filepath, hero_id_to_filter)
        print(f"Successfully saved hero vehicle data to the '{output_dir}' directory.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print('\nDone.')

if __name__ == '__main__':
    main()