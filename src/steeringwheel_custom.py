#!/usr/bin/env python

"""
Welcome to CARLA two-phase manual control with steering wheel Logitech G29.

DYNAMIC PHASE SWITCHING:
- Phase 1: Approach staging position (automatic goal switch when reached)
- Phase 2: Park from staging to final spot

To drive start by pressing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.
"""

from __future__ import print_function

import glob
import os
import sys
import json

# -- Path setup: resolve project root (PythonAPI/examples/) for cross-folder imports --
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)                                   # src/
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'agent'))         # agent/
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'analysis'))      # analysis/

from scenarios import *
from parking_agent import NMPCController, MAX_OBSTACLES

# -- find carla module ---------------------------------------------------------
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import cv2
import csv

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from calculate_error import calc_total_error, calc_score, add_collision, get_collision, reset_all_collisions, increment_collision_frames, get_total_collision_frames
from parking_agent import *
import time

# ==============================================================================
# -- Global Trial Information & JSON Logger ------------------------------------
# ==============================================================================

# These global variables will be populated in main() from command-line arguments.
# They are accessible by any class or function in this script.
TRIAL_INFO = {
    'user': 'N/A',
    'shared_autonomy': False,
    'state': 'N/A',
    'trial': 'N/A',
    'log_to_json': False,  # This will be True unless --ignore is used
    'recording_name': 'N/A'
}

JSON_LOG_FILE = os.path.join(_PROJECT_ROOT, 'data', 'results.json')

def log_trial_to_json(data):
    """
    Appends a trial's data to the JSON log file.
    """
    try:
        with open(JSON_LOG_FILE, 'r') as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log_data = []

    log_data.append(data)

    with open(JSON_LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=4)

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, actor_filter, starting_pos, starting_phase=0):
        """
        Args:
            starting_position: scenario name
            starting_phase: variant within scenario (0,1,2)
        """
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter

        # Use approach positions (farther starting points)
        start_positions_x = [
            -358.5,  # state 0
            -362.0,  # state 1  
            -365.5,  # state 2
            -369.0,  # state 3
            -372.5,  # state 4
            -365.5,  # state 5 (same x as state 2 but further out)
            -369.0,  # state 6 (same x as state 3 but further out)
            -372.5,  # state 7 (same x as state 4 but further out)
            -280,    # state 8 (the far right state)
            -335,    # state 9 (for the new studies - 1/9/2026)  
        ]

        # Initial heading and vertical positions (Added 1/9/2026)
        USUAL_Y = 144.1 if (TRIAL_INFO['state'] <= 4 or TRIAL_INFO['state'] == 9) else 145.5
        USUAL_HEADING_DEG = 180.0

        # Modified starting positions (Added 1/9/2026)
        safe_state_idx = TRIAL_INFO['state'] if TRIAL_INFO['state'] < len(start_positions_x) else 0
        self.x_start = start_positions_x[safe_state_idx]

        y_offset = TRIAL_INFO.get('y_offset', 0.0)
        heading_offset = TRIAL_INFO.get('heading_offset', 0.0)

        self.y_start = USUAL_Y + y_offset
        self.heading = USUAL_HEADING_DEG + heading_offset
        self.z_start = 160
        
        # forcefully reset the world first
        self.world.tick()
        actors = list(self.world.get_actors().filter('vehicle.*'))
        print(f'Attempting to delete {len(actors)} actors in custom_actor_spawn')
        for actor in actors:
            try:
                actor.destroy()
            except Exception as e:
                print(f"Failed to destroy actor {actor}: {e}")

        self.restart()
        self.world.on_tick(hud.on_world_tick)

        # Find Trigger Friction Blueprint
        friction_bp = self.world.get_blueprint_library().find('static.trigger.friction')
        extent = carla.Location(5000.0, 500.0, 500.0)
        friction_bp.set_attribute('friction', str(0.8))
        friction_bp.set_attribute('extent_x', str(extent.x))
        friction_bp.set_attribute('extent_y', str(extent.y))
        friction_bp.set_attribute('extent_z', str(extent.z))

        # Spawn Trigger Friction
        transform = carla.Transform()
        transform.location = carla.Location(-250, 142.5, 160.0)
        self.world.spawn_actor(friction_bp, transform)

    def restart(self):
        reset_all_collisions()
        
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        blueprint = self.world.get_blueprint_library().filter('model3')[0]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            self.destroy()
            spawn_point = carla.Transform(carla.Location(x=self.x_start, y=self.y_start, z=self.z_start), carla.Rotation(pitch=0, yaw=self.heading, roll=0))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point = carla.Transform(carla.Location(x=self.x_start, y=self.y_start, z=self.z_start), carla.Rotation(pitch=0, yaw=self.heading, roll=0))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # Apply brakes to make sure car doesn't start rolling back
        self.player.apply_control(carla.VehicleControl(brake=0.2))

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor
            ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.camera_manager and self.camera_manager.sensor is not None:
           self.camera_manager.sensor.stop()
           self.camera_manager.sensor.destroy()
        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# -- DualControl with Dynamic Goal Switching ----------------------------------
# ==============================================================================

class DualControl(object):
    def __init__(self, world, start_in_autopilot, shared_autonomy, controller=None, 
                 staging_goal=(-370.0, 145.1, math.radians(180)), 
                 final_goal=(-363.3, 141.2, math.radians(180)),
                 csv_filename=None,
                 no_spawn=False
                ):
        self._autopilot_enabled = start_in_autopilot
        self.shared_autonomy = shared_autonomy
        self.nmpc_controller = controller
        self.no_spawn = no_spawn
        
        # Dynamic goal switching variables
        self.current_phase = 1  # Start with Phase 1 (approach)
        self.staging_goal = staging_goal
        self.final_goal = final_goal

        self.phase1_complete = False
        self.phase1_time = None
        self.mission_complete = False  # NEW: Prevent repeated success messages
        self.timeout_triggered = False # NEW: Track if timeout occurred
        self.previous_steer = 0.0
        self.simulation_start_time = time.time()
        
        self._csv_log_data = []
        self._csv_filename = csv_filename
        self._csv_header = [
            'frame', 'timestamp', 'agent_active', 'x', 'y', 
            'velocity_x', 'velocity_y', 'velocity_z',
            'heading_deg', 'yaw_deg', 'pitch_deg', 'roll_deg', 
            'user_throttle', 'user_brake', 'user_steer', 
            'agent_throttle', 'agent_brake', 'agent_steer',
            'in_collision',
            'current_phase', 
            'applied_throttle', 
            'applied_brake', 
            'applied_steer'
        ]
        
        print(f'Found controller {self.nmpc_controller}')
        print(f'PHASE 1 GOAL: Staging pose {self.staging_goal}')
        print(f'PHASE 2 GOAL: Final parking {self.final_goal}')

        self.agent_query_frequency = 8
        self.frame_counter = 0
        self.last_agent_control = carla.VehicleControl()

        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")
        
        if joystick_count > 0:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            self._parser = ConfigParser()
            self._parser.read(os.path.join(_PROJECT_ROOT, 'config', 'racingwheel_config.ini'))
            self._steer_idx = int(self._parser.get('G29 Racing Wheel', 'steering_wheel'))
            self._throttle_idx = int(self._parser.get('G29 Racing Wheel', 'throttle'))
            self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
            self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
            self._handbrake_idx = int(self._parser.get('G29 Racing Wheel', 'handbrake'))
        else:
            self._joystick = None

    def update_controller_goal(self, new_goal):
        """Dynamically update the NMPC controller's goal"""
        if self.nmpc_controller:
            goal_x, goal_y, goal_theta = new_goal
            self.nmpc_controller.goal_pose = [goal_x, goal_y, goal_theta]
            print(f"GOAL UPDATED: ({goal_x:.1f}, {goal_y:.1f}, {math.degrees(goal_theta):.0f}°)")

    def check_goal_reached(self, world_player, current_goal, collision_sensor, hud):
        """Check if current goal is reached and handle phase switching"""
        current_pos = world_player.get_location()
        current_transform = world_player.get_transform()
        current_yaw = math.radians(current_transform.rotation.yaw)
        
        goal_x, goal_y, goal_theta = current_goal

        # ==================================================================
        # --- VISUALIZATION (Draw Staging Goal on Ground) ---
        # Only draw the guide box during Phase 1 if SA is ON.
        # ==================================================================
        if self.current_phase == 1 and not self.phase1_complete and not self.no_spawn and self.shared_autonomy:
            # 1. Get Z-level
            try:
                # Look up ground level at goal x,y. Start search from z=150
                goal_waypoint = world_player.get_world().get_map().get_waypoint(carla.Location(goal_x, goal_y, 150.0))
                ground_z = goal_waypoint.transform.location.z + 0.05 # Lift 5cm just above ground
            except:
                ground_z = current_pos.z

            # 2. Define color and thickness (Lighter and Thinner)
            # Cyan color with very low alpha (35/255) for faint glow
            box_color = carla.Color(0, 255, 255, 35)
            # Very thin lines
            box_thickness = 0.02
            
            # 3. Draw the box
            # Size is (Length, Width, Height) / 2 (Extent)
            box_extent = carla.Vector3D(2.5, 1.25, 0.05) 
            box_loc = carla.Location(goal_x, goal_y, ground_z)
            box_rot = carla.Rotation(yaw=math.degrees(goal_theta))
            
            world_player.get_world().debug.draw_box(
                carla.BoundingBox(box_loc, box_extent),
                box_rot,
                thickness=box_thickness,
                color=box_color,
                life_time=0.15 # Persist slightly longer than a frame to prevent flickering
            )
        # ==================================================================


        pos_distance = math.hypot(current_pos.x - goal_x, current_pos.y - goal_y)
        angle_error = abs((current_yaw - goal_theta + math.pi) % (2 * math.pi) - math.pi)

        # Check for active collision
        current_frame = world_player.get_world().get_snapshot().frame
        in_collision = collision_sensor.is_in_active_collision(current_frame)

        v = world_player.get_velocity()
        current_speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # Thresholds
        speed_threshold_kmh = 1.0  # Must be almost stopped
        pos_threshold = 1.0        # 1.0m tolerance for staging
        angle_threshold = math.radians(10) # 10 degree tolerance
        
        final_pos_threshold = 0.5         # Tighter: 50cm
        final_angle_threshold = math.radians(8) # Tighter: 8 degrees

        # ==================================================================
        # --- GLOBAL FINAL SUCCESS AND TIMEOUT CHECK ---
        # ==================================================================
        
        # 1. Time Check
        TIMEOUT_SECONDS = 180.0 # 3 Minutes
        current_time = time.time()
        elapsed_time = current_time - self.simulation_start_time
        is_timeout = elapsed_time > TIMEOUT_SECONDS

        # 2. Success Check
        final_goal_x, final_goal_y, final_goal_theta = self.final_goal
        final_dist = math.hypot(current_pos.x - final_goal_x, current_pos.y - final_goal_y)
        final_angle_err = abs((current_yaw - final_goal_theta + math.pi) % (2 * math.pi) - math.pi)

        final_goal_reached = (final_dist < final_pos_threshold and 
                              final_angle_err < final_angle_threshold and 
                              not in_collision and 
                              current_speed_kmh < speed_threshold_kmh)

        # Trigger termination if either goal is reached OR time is out
        if (final_goal_reached or is_timeout) and not self.mission_complete:
            self.mission_complete = True
            self.timeout_triggered = is_timeout
            
            # Use appropriate Phase 1 time if we skipped it or didn't record it
            if self.phase1_time is None:
                self.phase1_time = elapsed_time

            total_time = elapsed_time

            print("\n" + "="*60)
            if is_timeout:
                hud.show_announcement("TIMEOUT - TRIAL FAILED", (255, 0, 0), 10.0)
                print("TIMEOUT REACHED (180s) - TRIAL ENDED")
            else:
                hud.show_announcement("TRIAL RUN COMPLETE", (50, 255, 50), 10.0)
                print("PHASE 2 (PARKING) COMPLETE!")
                print("TWO-PHASE PARKING SUCCESSFUL!")

            print(f"Phase 1 time: {self.phase1_time:.2f}s")
            print(f"Phase 2 time: {total_time - self.phase1_time:.2f}s") 
            print(f"Total time: {total_time:.2f}s")
            print(f"Final position error: {final_dist:.3f}m, angle error: {math.degrees(final_angle_err):.1f}°")
            print(f"Total unique collisions: {get_collision()}") 
            print(f"Total collision frames (severity): {get_total_collision_frames()}")
            print(f"Total collisions: {get_collision()}")
            
            # Calculate final score
            score = calc_score(final_dist, math.degrees(final_angle_err), total_time)
            print(f"Final score: {score:.2f}")
            print("="*60)

            if self._csv_filename:
                try:
                    # Create data directory if it doesn't exist
                    os.makedirs(os.path.dirname(self._csv_filename), exist_ok=True)
                    
                    with open(self._csv_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(self._csv_header)
                        writer.writerows(self._csv_log_data)
                    print(f"Successfully saved data to {self._csv_filename}")
                except Exception as e:
                    print(f"Error saving CSV file: {e}")

            # logging
            if TRIAL_INFO['log_to_json']:
                final_data = {
                    'user': TRIAL_INFO['user'],
                    'state': TRIAL_INFO['state'],
                    'trial': TRIAL_INFO['trial'],
                    'shared_autonomy': TRIAL_INFO['shared_autonomy'],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'outcome': 'timeout' if is_timeout else 'success',
                    'success': not is_timeout,
                    'timed_out': is_timeout,
                    'phase1_time_s': round(self.phase1_time, 2),
                    'phase2_time_s': round(total_time - self.phase1_time, 2),
                    'total_time_s': round(total_time, 2),
                    'final_pos_error_m': round(final_dist, 3),
                    'final_angle_error_deg': round(math.degrees(final_angle_err), 1),
                    'total_collisions': get_collision(),
                    'total_collision_frames': get_total_collision_frames(),
                    'final_score': round(score, 2)
                }
                log_trial_to_json(final_data)
                print("Trial data successfully logged to trial_log.json")
            return True


        # ==================================================================
        # --- PHASE 1 LOGIC (Staging) ---
        # ==================================================================
        if self.current_phase == 1 and not self.phase1_complete:
            
            # Use Distance/Angle/Speed criteria for Phase 1 success
            is_staging_success = (pos_distance < pos_threshold and 
                                  angle_error < angle_threshold and 
                                  not in_collision and 
                                  current_speed_kmh < speed_threshold_kmh)
            
            if is_staging_success:
                self.phase1_complete = True
                
                # Only show prompt if SA is on (user sees box)
                if self.shared_autonomy:
                    hud.show_announcement("READY TO PARK", (50, 255, 50), 3.0)
                
                self.phase1_time = time.time() - self.simulation_start_time
                self.current_phase = 2
                
                print("\n" + "="*60)
                print("PHASE 1 (APPROACH) COMPLETE!")
                print(f"Goal Reached: ({goal_x:.1f}, {goal_y:.1f})")
                print(f"Time: {self.phase1_time:.2f} seconds")
                print("SWITCHING TO PHASE 2 (PARKING)...")

                # --- NEW LOGIC: DISABLE SHARED AUTONOMY ---
                if self.shared_autonomy:
                    print(">>> DISABLING SHARED AUTONOMY FOR PHASE 2 (MANUAL CONTROL ONLY) <<<")
                    self.shared_autonomy = False

                    hud.sa_active = False
                # ------------------------------------------

                print("="*60)
                
                # Update controller goal to final parking spot
                self.update_controller_goal(self.final_goal)
                return False  # Continue to phase 2
            
        return False

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN and self._joystick:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                    world.camera_manager.toggle_camera()
                elif event.button == 23:
                    world.camera_manager.next_sensor()
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        if not self._autopilot_enabled:
            if self._csv_filename:
                t = world.player.get_transform()
                v = world.player.get_velocity()
                
                # Agent control is self.last_agent_control
                # User control is self._control
                agent_throttle = self.last_agent_control.throttle if self.shared_autonomy else 0.0
                agent_brake = self.last_agent_control.brake if self.shared_autonomy else 0.0
                agent_steer = self.last_agent_control.steer if self.shared_autonomy else 0.0

                # Get the current frame for the collision check
                current_frame_num = world.world.get_snapshot().frame
                # This method returns True if in a collision, False otherwise.
                is_colliding_flag = world.collision_sensor.is_in_active_collision(current_frame_num)

                # 1. DETERMINE FINAL APPLIED VALUES
                # We look at what was actually sent to the car in the logic above
                final_applied_control = self._control # Default to user control
                
                if self.mission_complete:
                    # If mission is done, we applied full brake
                    final_applied_control = carla.VehicleControl(brake=1.0, hand_brake=True)
                elif self.shared_autonomy and self.nmpc_controller:
                    # If SA was active, we used the calculated mixed_control
                    # (We can access mixed_control here because it was defined in the block above)
                    try:
                        final_applied_control = mixed_control
                    except UnboundLocalError:
                        # Fallback if mixed_control wasn't calculated this specific frame
                        final_applied_control = self._control
                
                row_data = [
                    world.world.get_snapshot().frame,
                    time.time(),
                    self.shared_autonomy,
                    t.location.x,
                    t.location.y,
                    v.x,
                    v.y,
                    v.z,
                    t.rotation.yaw,
                    t.rotation.yaw,
                    t.rotation.pitch,
                    t.rotation.roll,
                    self._control.throttle,
                    self._control.brake,
                    self._control.steer,
                    agent_throttle,
                    agent_brake,
                    agent_steer,
                    is_colliding_flag,
                    self.current_phase,
                    final_applied_control.throttle,
                    final_applied_control.brake,
                    final_applied_control.steer
                ]
                self._csv_log_data.append(row_data)

            # Check collision state every frame for the severity metric.
            current_frame = world.world.get_snapshot().frame
            if world.collision_sensor.is_in_active_collision(current_frame):
                increment_collision_frames()

            if isinstance(self._control, carla.VehicleControl):
                # Prioritize joystick for primary vehicle control if it exists
                if self._joystick:
                    self._parse_vehicle_wheel()
                    # We still want to parse keyboard for the handbrake
                    keys = pygame.key.get_pressed()
                    self._control.hand_brake = keys[K_SPACE]
                else:
                    # Fallback to full keyboard control if no joystick
                    self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                
                # This logic is independent of the input method
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())

            # Dynamic goal switching shared autonomy
            # This block now runs for both manual and shared autonomy modes.
            current_goal = self.staging_goal if self.current_phase == 1 else self.final_goal
            # The check_goal_reached method handles printing success messages and updating the phase.
            self.check_goal_reached(world.player, current_goal, world.collision_sensor, world.hud)

            # Apply vehicle controls based on the mission status
            if self.mission_complete:
                # If parking is finished, apply the brakes and stop.
                world.player.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))

            elif self.shared_autonomy and self.nmpc_controller and world.player:
                # If shared autonomy is ON and mission is not complete, run the agent.
                self.frame_counter += 1
                
                if self.frame_counter % self.agent_query_frequency == 0:
                    start_debug_time = time.time()
                    try:
                        self.last_agent_control = self.nmpc_controller.get_current_carla_control_object(world.player, world.world)
                        solve_time = time.time() - start_debug_time
                    except Exception as e:
                        print(f'Agent control error: {e}')
                        self.last_agent_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.3)
                
                agent_control = self.last_agent_control
                user_control = self._control

                # <---- START OF REPLACEMENT ---->
                # =============================================================================================================
                # SHARED AUTONOMY CONTORL MIXING 
                # =============================================================================================================
                mixed_control = carla.VehicleControl()
                alpha = 0.25 # Changed to 0.25 (1/9/26)
                
                if user_control.throttle or user_control.brake:
                    if user_control.brake > 0.1:
                        mixed_control.throttle = 0.0
                        mixed_control.brake = user_control.brake
                        mixed_control.reverse = False 
                        mixed_steer = (alpha * agent_control.steer) + ((1 - alpha) * user_control.steer)
                        mixed_control.steer = np.clip(mixed_steer, -1.0, 1.0)
                    else:
                        mixed_steer = (alpha * agent_control.steer) + ((1 - alpha) * user_control.steer)
                        mixed_control.steer = np.clip(mixed_steer, -1.0, 1.0)
                        user_effective_throttle = user_control.throttle if not user_control.reverse else -user_control.throttle
                        agent_effective_throttle = agent_control.throttle if not agent_control.reverse else -agent_control.throttle
                        blended_effective_throttle = (alpha * agent_effective_throttle) + ((1 - alpha) * user_effective_throttle)
                        if blended_effective_throttle > 0:
                            mixed_control.throttle = np.clip(blended_effective_throttle, 0.0, 1.0)
                            mixed_control.brake = 0.0
                            mixed_control.reverse = False
                        elif blended_effective_throttle < 0:
                            mixed_control.throttle = np.clip(abs(blended_effective_throttle), 0.0, 1.0)
                            mixed_control.brake = 0.0
                            mixed_control.reverse = True
                        else: 
                            mixed_control.throttle = 0.0
                            mixed_control.brake = 0.0
                            mixed_control.reverse = False
                else:
                    mixed_control = self._control
                mixed_control.manual_gear_shift = self._control.manual_gear_shift
                mixed_control.gear = self._control.gear
                mixed_control.hand_brake = self._control.hand_brake

                if self.frame_counter % 20 == 0:
                    goal_distance = math.hypot(world.player.get_location().x - current_goal[0], world.player.get_location().y - current_goal[1])
                    print(f'  Agent: T{agent_control.throttle:.3f} B{agent_control.brake:.3f} S{agent_control.steer:.3f} R{agent_control.reverse}')
                    print(f'  User:  T{user_control.throttle:.3f} B{user_control.brake:.3f} S{user_control.steer:.3f} R{user_control.reverse}')
                    print(f'  Mixed: T{mixed_control.throttle:.3f} B{mixed_control.brake:.3f} S{mixed_control.steer:.3f} R{mixed_control.reverse}')
                # <---- END OF REPLACEMENT ---->
                
                # Apply the mixed control
                world.player.apply_control(mixed_control)
            else:
                # If shared autonomy is OFF and mission is not complete, apply direct user control.
                world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        K1 = 1.0
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd
        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)

        self._damage_alpha = 0  # Current transparency (0 = invisible)
        self.sa_active = TRIAL_INFO['shared_autonomy']
        self._damage_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Draw a thick, semi-transparent red border
        # (255, 0, 0) is Red. The last number (50) is the base transparency intensity.
        border_thickness = 40 
        pygame.draw.rect(self._damage_surface, (255, 0, 0, 50), (0, 0, width, height), border_thickness)

        self._notifications = FadingText(font, (width, 40), (0, height - 40))

        # 1. Create a huge font (Size 50)
        self._font_huge = pygame.font.Font(mono, 50) 
        # 2. Create a new FadingText surface in the center of the screen
        # Position is (0, middle_of_height)
        self._announcement = FadingText(self._font_huge, (width, 100), (0, height // 2 - 50))

        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        self._announcement.tick(world, clock)

        if self._damage_alpha > 0:
            # Decrease alpha. Higher number = faster fade.
            self._damage_alpha -= 8 
            if self._damage_alpha < 0:
                self._damage_alpha = 0

        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Shared Autonomy: %s' % ('ON' if self.sa_active else 'OFF'),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        display_steer = -c.steer if c.reverse else c.steer
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def show_announcement(self, text, color=(0, 255, 0), seconds=3.0):
        """Displays a big message in the center of the screen"""
        self._announcement.set_text(text, color=color, seconds=seconds)

    def trigger_damage_effect(self):
        self._damage_alpha = 200  # Set to high visibility (max 255)

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18

        if self._damage_alpha > 0:
            # Apply the current fade level
            self._damage_surface.set_alpha(self._damage_alpha)
            display.blit(self._damage_surface, (0, 0))

        self._notifications.render(display)
        self._announcement.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        # CHANGE 1: Add pygame.SRCALPHA for transparency support
        self.surface = pygame.Surface(self.dim, pygame.SRCALPHA)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        
        # CHANGE 2: Re-create surface with SRCALPHA (transparency)
        self.surface = pygame.Surface(self.dim, pygame.SRCALPHA)
        self.seconds_left = seconds
        
        # Fill with fully transparent color (R, G, B, Alpha=0)
        self.surface.fill((0, 0, 0, 0))
        
        # CHANGE 3: Calculate center position
        text_rect = text_texture.get_rect(center=(self.dim[0] // 2, self.dim[1] // 2))
        
        # Blit at the calculated center
        self.surface.blit(text_texture, text_rect)

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================

class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.last_collision_frame = -1
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def is_in_active_collision(self, current_frame):
        """Checks if a collision occurred in the last 2 frames."""
        return current_frame - self.last_collision_frame < 2

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        self.hud.trigger_damage_effect()

        self.last_collision_frame = event.frame

        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        add_collision()

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-0.12,y=-0.4, z=1.2), carla.Rotation()),
            carla.Transform(carla.Location(x=-0.4, y=0.0, z=1.32), carla.Rotation(pitch=0, yaw=180, roll=0))
        ]

        self.transform_index = 0
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

# ==============================================================================
# -- Simplified Game Loop -----------------------------------------------------
# ==============================================================================

def game_loop(starting_position, starting_phase, dist, target_location, args):
    pygame.init()
    pygame.font.init()
    world = None
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        filepath = args.filepath
        
        if TRIAL_INFO['log_to_json']:
            filepath = TRIAL_INFO['recording_name']
            print('[WARNING] json logging has been enabled, any recording name passed in with --filepath parameter will be overwritten')

        if not filepath.lower().endswith(".log"):
            filepath += ".log"
        client.start_recorder(f'/home/driving_sim/CARLA_0.9.15/data/{filepath}')
        
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        screen = pygame.display.set_mode((args.window_width, args.window_height), flags)
        render_surf = pygame.Surface((args.width, args.height))

        # Initialize NMPC controller with staging goal first
        nmpc_controller = None
        final_goal = (-363.3, 141.3, math.radians(183))
        staging_goal = (-370.7, 144, math.radians(180))

        if args.shared_autonomy:
            controller_params = {
                "L": 2.7,
                "dt_plan": 0.15,
                "N": 10,
                "Q_diag": [0.08, 0.08, 0.08],
                "R_diag": [0.01, 0.01],
                "Qf_diag": [2.0, 4.0, 10.0],
                "obstacle_weight": 80,
                "obstacle_sigma": 0.4,
                "reverse_penalty": 0.1,
                "ego_length": 5.0,
                "ego_width": 1.5,
                "steer_rate_weight": 1.0,
                "dist_slow_thresh": 1.0,
                "slow_weight_v": 0.3,
                "slow_weight_delta": 0.0,
                "max_obstacles": MAX_OBSTACLES,
                "goal_pose_init": staging_goal,
                "goal_threshold_pos_init": 1.5,
                "goal_threshold_angle_init": math.radians(20),
                "max_converge_samples_init": 20,
                "converge_tolerance_init": 0.1
            }
            nmpc_controller = NMPCController(**controller_params)

        csv_filepath = None
        if args.csv_overwrite:
            csv_filepath = args.csv_overwrite
        else:
            # Generate filename from trial info
            user = TRIAL_INFO['user']
            state = TRIAL_INFO['state']
            trial = TRIAL_INFO['trial']
            sa = TRIAL_INFO['shared_autonomy']
            csv_filepath = os.path.join(_PROJECT_ROOT, 'data', f"{user}_state{state}_trial{trial}_sa{sa}.csv")

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter, starting_position, starting_phase)
        controller = DualControl(
            world, 
            args.autopilot, 
            args.shared_autonomy, 
            nmpc_controller, 
            staging_goal=staging_goal,
            final_goal=final_goal,
            csv_filename=csv_filepath,
            no_spawn=args.no_spawn  # <--- PASS THE ARGUMENT HERE
        )

        clock = pygame.time.Clock()
        settings = world.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.world.apply_settings(settings)

        if not args.no_spawn:
            try:
                custom_actor_spawn(client=client, position=starting_position, dist_dev=dist)
            except NameError:
                print("custom_actor_spawn function not found - skipping custom actor spawning")
        else:
            print("NPC spawning disabled via --no-spawn flag.")

        while True:
            world.world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                return

            render_surf.fill((0,0,0))
            world.tick(clock)
            world.render(render_surf)

            scaled = pygame.transform.scale(render_surf, (args.window_width, args.window_height))
            screen.blit(scaled, (0,0))
            pygame.display.flip()

    finally:
        client.stop_recorder()
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            world.world.apply_settings(settings)
            world.destroy()
        pygame.quit()
        cv2.destroyAllWindows()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def validate_args(TRIAL_INFO):
    assert isinstance(TRIAL_INFO['state'], int), "Argument --state must be an integer."
    assert -1 < TRIAL_INFO['state'] < 10, f"State must be between 0 and 9, but got {TRIAL_INFO['state']}."
    assert isinstance(TRIAL_INFO['log_to_json'], bool), "log_to_json flag must be a boolean."

def main():
    argparser = argparse.ArgumentParser(description='CARLA Dynamic Two-Phase Parking')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int)
    argparser.add_argument('-a', '--autopilot', action='store_true')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720')
    argparser.add_argument('--window-size', metavar='WIDTHxHEIGHT', default='3072x1728')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*')
    argparser.add_argument('--filepath', default='parking_play.log')
    argparser.add_argument('--phase', default=0, type=int)
    argparser.add_argument('--distance', default=4, type=int)
    argparser.add_argument('-s', '--shared-autonomy', action='store_true')
    argparser.add_argument('--csv_overwrite', default=None, type=str, help='Manually specify the CSV output filename, overwriting the default naming scheme.')

    argparser.add_argument('--no-spawn', action='store_true', help='Disable spawning of NPC vehicles')

    # Added flags for new user studies with randomized heading and vertical positioning (Added 1/9/2026)
    argparser.add_argument('--y-offset', default=0.0, type=float, help='Offset to add to starting Y (meters)')
    argparser.add_argument('--heading-offset', default=0.0, type=float, help='Offset to add to starting Heading (degrees)')

    # json trial arguments
    argparser.add_argument('--state', default=4, type=int)
    argparser.add_argument('--trial', default='N/A')
    argparser.add_argument('--user', default='N/A')
    argparser.add_argument('--ignore', action='store_true')
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.window_width, args.window_height = [int(x) for x in args.window_size.split('x')]

    # initialize args
    global TRIAL_INFO
    TRIAL_INFO['user'] = args.user
    TRIAL_INFO['state'] = args.state
    TRIAL_INFO['trial'] = args.trial
    TRIAL_INFO['shared_autonomy'] = args.shared_autonomy
    TRIAL_INFO['log_to_json'] = not args.ignore  # Will be True if --ignore is NOT passed
    TRIAL_INFO['recording_name'] = f"{args.user}_state{args.state}_trial{args.trial}_sa{args.shared_autonomy}"

    # Added new trial info for heading and y-axis randomization (Added 1/9/2026)
    TRIAL_INFO['y_offset'] = args.y_offset
    TRIAL_INFO['heading_offset'] = args.heading_offset

    validate_args(TRIAL_INFO) # ensure that args are valid

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    
    starting_positions = ["Parallel Parking", "Angled Parking (Incline)", "Angled Parking (Flat)", "Standard Parking", "Experimental"]
    
    try:
        final_target_location = (-363.3, 141.2, math.radians(183)) # NOTE: THIS SHOULD MATCH THE VARIABLE `final_goal` IN THE GAME LOOP
        start_phase = min(max(args.phase, 0), 2)
        start_distance = min(max(args.distance, 0), 15)
        
        print('DYNAMIC TWO-PHASE PARKING SYSTEM')
        print(f'Starting position: {starting_positions[4]}')
        print(f'Phase 1: Approach to staging (-370.0, 145.1, 180°)')
        print(f'Phase 2: Park at final spot (-363.3, 141.2, 180°)')
        print('Goals will switch automatically when Phase 1 completes')
        
        game_loop(starting_positions[4], start_phase, start_distance, final_target_location, args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()