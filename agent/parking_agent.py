import os
import sys

# -- Path setup: resolve project root for cross-folder imports --
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

import carla
import casadi as ca
import numpy as np
import math
import time
from concurrent.futures import ThreadPoolExecutor

from scenarios import custom_actor_spawn

# === CasADi Clean-up === 

# To clean up temp file
import tempfile
import atexit
import shutil

temp_dir = tempfile.mkdtemp(prefix="casadi_temp_")

ca.GlobalOptions.setCasadiPath(temp_dir)

def cleanup_casadi_files():
    try:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up CasADi temporary files from {temp_dir}")
    except:
        pass

atexit.register(cleanup_casadi_files)

 
# === Global Constants (Used by multiple classes or for overall simulation) ===
MAX_OBSTACLES = 4
DT_APPLY = 0.05   # how often we actually send a control to CARLA

# Helper function (can be static method or standalone)
def normalize_angle(angle_rad):
    """Normalize angle to be within [-pi, pi]."""
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi

def angle_diff_python(a, b):
    """Calculates the signed difference between two angles for standard Python floats."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def linear_angle_diff(a, b):
    """Fast linear approximation for small angle differences."""
    diff = a - b
    # Wrap to [-pi, pi] range quickly
    if diff > math.pi:
        diff -= 2 * math.pi
    elif diff < -math.pi:
        diff += 2 * math.pi
    return diff

class ApproachController:
    """
    General approach controller that maintains vehicle orientation.
    If goal is behind, just reverse. If goal is ahead, just go forward.
    No unnecessary turning to change orientation during approach.
    """
    def __init__(self, 
                 target_speed=0.8,
                 kp_speed=0.4,
                 d_switch=2.5,
                 gamma_switch=math.radians(15),
                 stop_distance=2.0,
                 slow_distance=4.0):
        self.TARGET_SPEED = target_speed
        self.KP_SPEED = kp_speed
        self.D_SWITCH = d_switch
        self.GAMMA_SWITCH = gamma_switch
        self.STOP_DISTANCE = stop_distance
        self.SLOW_DISTANCE = slow_distance
        
        # State tracking
        self.close_count = 0
        self.last_distance = None
        self.wrong_direction_count = 0
        
        # Steering smoothing
        self.last_steer_cmd = 0.0
        self.steer_history = []
        self.max_steer_history = 5

    def calculate_movement_direction(self, current_state, target_point):
        """
        Determine if we should go forward or reverse based on where the goal is.
        Key insight: Don't change vehicle orientation, just pick forward/reverse.
        """
        dx = target_point[0] - current_state[0]
        dy = target_point[1] - current_state[1]
        distance = math.hypot(dx, dy)
        
        # Vector to goal
        vec_to_goal = np.array([dx, dy])
        
        # Vehicle's forward direction (current orientation)
        vec_forward = np.array([math.cos(current_state[2]), math.sin(current_state[2])])
        
        # Check if goal is more in front or behind
        projection = np.dot(vec_to_goal, vec_forward)
        
        # Determine direction
        should_reverse = projection < 0
        
        return should_reverse, distance

    def calculate_steering_for_direction(self, current_state, target_point, is_reversing):
        """
        Calculate steering to reach target while maintaining current orientation approach.
        Fixed logic for reverse steering with smoother control.
        """
        dx = target_point[0] - current_state[0]
        dy = target_point[1] - current_state[1]
        distance = math.hypot(dx, dy)
        
        # Angle from vehicle to target
        angle_to_target = math.atan2(dy, dx)
        
        # Reference direction based on movement mode
        if is_reversing:
            # When reversing, we want the REAR to point toward target
            # Rear direction is current heading + 180°
            reference_heading = current_state[2] + math.pi
        else:
            # When going forward, we want the FRONT to point toward target  
            reference_heading = current_state[2]
        
        # Calculate heading error
        heading_error = angle_diff_python(angle_to_target, reference_heading)
        
        # IMPORTANT: For reverse, steering input is INVERTED
        # If we need to turn the rear left, we steer right, and vice versa
        if is_reversing:
            heading_error = -heading_error
        
        # Progressive steering gain based on distance
        # Reduced base gain for smoother control
        base_gain = 0.35  # Reduced from 0.6
        
        # Multi-zone distance factor for very smooth transitions
        if distance > 5.0:
            distance_factor = 1.0
        elif distance > 3.0:
            distance_factor = 0.8 + 0.2 * (distance - 3.0) / 2.0
        elif distance > 1.5:
            distance_factor = 0.5 + 0.3 * (distance - 1.5) / 1.5
        else:
            # Very gentle when close
            distance_factor = 0.2 + 0.3 * (distance / 1.5)
        
        # Apply gain
        steering_gain = base_gain * distance_factor
        
        # Calculate raw steering command
        steer_cmd = steering_gain * heading_error
        
        # Dead zone to prevent micro-corrections
        if abs(heading_error) < math.radians(2) and distance < 2.0:
            steer_cmd *= 0.3  # Reduce steering for small errors when close
        
        # Limit steering based on distance (gentler limits when close)
        if distance < 1.5:
            max_steer = 0.4
        elif distance < 3.0:
            max_steer = 0.6
        else:
            max_steer = 0.8
            
        steer_cmd = np.clip(steer_cmd, -max_steer, max_steer)
        
        # Apply steering rate limiting for smoothness
        max_steer_change = 0.1  # Maximum change per timestep
        steer_change = steer_cmd - self.last_steer_cmd
        if abs(steer_change) > max_steer_change:
            steer_cmd = self.last_steer_cmd + np.sign(steer_change) * max_steer_change
        
        # Store for next iteration
        self.last_steer_cmd = steer_cmd
        
        # Add to history for averaging
        self.steer_history.append(steer_cmd)
        if len(self.steer_history) > self.max_steer_history:
            self.steer_history.pop(0)
        
        # Return averaged steering for extra smoothness
        if len(self.steer_history) > 0:
            return sum(self.steer_history) / len(self.steer_history)
        else:
            return steer_cmd

    def check_obstacles_in_path(self, vehicle_pos, vehicle_heading, is_reverse, actors, vehicle_id):
        """Check for obstacles in movement path"""
        movement_heading = vehicle_heading + (math.pi if is_reverse else 0)
        
        for actor in actors:
            if actor.id == vehicle_id:
                continue
                
            obs_pos = actor.get_location()
            distance = obs_pos.distance(vehicle_pos)
            
            if distance < self.SLOW_DISTANCE:
                # Check if obstacle is in movement path
                angle_to_obs = math.atan2(obs_pos.y - vehicle_pos.y, obs_pos.x - vehicle_pos.x)
                angle_diff = angle_diff_python(angle_to_obs, movement_heading)
                
                # If obstacle is in a 45-degree cone in movement direction
                if abs(angle_diff) < math.radians(45):
                    # Check distance along movement direction
                    distance_along_path = distance * math.cos(angle_diff)
                    lateral_distance = abs(distance * math.sin(angle_diff))
                    
                    # If obstacle is ahead in path and not too far to the side
                    if distance_along_path > 0 and lateral_distance < 2.5:
                        if distance < self.STOP_DISTANCE:
                            return "STOP", distance
                        else:
                            return "SLOW", distance
        
        return "CLEAR", float('inf')

    def calculate_speed(self, base_speed, distance_to_goal, obstacle_status, obstacle_distance, steer_cmd):
        """Calculate target speed based on various factors"""
        target_speed = base_speed
        
        # Slow down based on obstacle proximity
        if obstacle_status == "SLOW":
            obstacle_factor = max(obstacle_distance / self.SLOW_DISTANCE, 0.3)
            target_speed *= obstacle_factor
        
        # Progressive speed reduction when approaching goal
        if distance_to_goal < 0.5:
            if distance_to_goal < 1.0:
                goal_factor = 0.2
            elif distance_to_goal < 2.0:
                goal_factor = 0.3
            elif distance_to_goal < 3.0:
                goal_factor = 0.5
            else:
                goal_factor = 0.7
            target_speed *= goal_factor
        
        # Further reduce speed when steering hard
        steer_factor = 1.0 - min(abs(steer_cmd), 0.8)
        target_speed *= (0.5 + 0.5 * steer_factor)
        
        return target_speed

    def approach_target(self, carla_env, staging_pose, start_time, timeout_s, 
                       collision_flag, curb_hit_flag):
        """
        Main approach control loop with smooth transitions
        """
        print(">>> PHASE 1: SMOOTH ORIENTATION-PRESERVING APPROACH")
        
        # Reset steering history
        self.last_steer_cmd = 0.0
        self.steer_history = []
        
        # Track if we're in orientation correction mode
        orientation_mode = False
        orientation_stable_count = 0
        
        while True:
            x0 = carla_env.get_ego_state()
            current_pos = carla_env.vehicle.get_location()
            
            # Determine movement direction (forward/reverse) based on goal position
            should_reverse, distance_to_goal = self.calculate_movement_direction(x0, staging_pose[:2])
            
            # Safety check: are we getting closer to the goal?
            if self.last_distance is not None:
                distance_change = distance_to_goal - self.last_distance
                if distance_change > 0.1:  # Moving away from goal significantly
                    self.wrong_direction_count += 1
                    print(f"WARNING: Moving away from goal! Distance change: +{distance_change:.2f}m")
                else:
                    self.wrong_direction_count = 0
            
            # If we've been moving away for too long, force direction change
            if self.wrong_direction_count > 10:
                print("FORCED DIRECTION CHANGE: Reversing movement direction")
                should_reverse = not should_reverse
                self.wrong_direction_count = 0
            
            self.last_distance = distance_to_goal
            
            # Check for obstacles in the intended path
            all_actors = carla_env.world.get_actors().filter("vehicle.*")
            obstacle_status, obstacle_distance = self.check_obstacles_in_path(
                current_pos, x0[2], should_reverse, all_actors, carla_env.vehicle.id
            )
            
            # Handle obstacle situations
            if obstacle_status == "STOP":
                print(f"STOPPING: Obstacle at {obstacle_distance:.2f}m in path")
                carla_env.vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, reverse=False
                ))
                
            # Check orientation
            desired_yaw = staging_pose[2]
            current_yaw = x0[2]
            yaw_error = abs(angle_diff_python(current_yaw, desired_yaw))
            
            # Smooth transition to orientation mode
            if distance_to_goal < 1.5 and yaw_error > math.radians(20) and not orientation_mode:
                orientation_mode = True
                orientation_stable_count = 0
                print("Entering orientation correction mode")
            
            # Handle orientation correction with smooth transitions
            if orientation_mode:
                yaw_correction = angle_diff_python(desired_yaw, current_yaw)
                
                # Progressive orientation correction
                if abs(yaw_correction) > math.radians(15):
                    turn_speed = 0.25
                    turn_gain = 2.0
                elif abs(yaw_correction) > math.radians(8):
                    turn_speed = 0.15
                    turn_gain = 1.5
                else:
                    turn_speed = 0.1
                    turn_gain = 1.0
                
                turn_steer = np.clip(yaw_correction * turn_gain, -0.6, 0.6)
                
                carla_env.vehicle.apply_control(carla.VehicleControl(
                    throttle=turn_speed, steer=turn_steer, brake=0.0, reverse=False
                ))
                
                # Check if orientation is good enough
                if yaw_error < math.radians(10):
                    orientation_stable_count += 1
                    if orientation_stable_count > 10:
                        orientation_mode = False
                        print("Orientation correction complete")
                else:
                    orientation_stable_count = 0
                
                continue
            
            # Normal approach mode with smooth steering
            steer_cmd = self.calculate_steering_for_direction(x0, staging_pose[:2], should_reverse)
            
            # Calculate target speed with steering consideration
            target_speed = self.calculate_speed(
                self.TARGET_SPEED, distance_to_goal, obstacle_status, obstacle_distance, steer_cmd
            )
            
            # Apply speed control with smooth transitions
            v_current = carla_env.vehicle.get_velocity().length()
            speed_error = target_speed - v_current
            
            if abs(speed_error) < 0.05:
                throttle_cmd = 0.0
                brake_cmd = 0.0
            elif speed_error > 0:
                # Gentler acceleration
                throttle_cmd = np.clip(self.KP_SPEED * speed_error * 0.6, 0.0, 0.2)
                brake_cmd = 0.0
            else:
                throttle_cmd = 0.0
                # Gentler braking
                brake_cmd = np.clip(self.KP_SPEED * abs(speed_error) * 0.5, 0.0, 0.3)

            # Apply vehicle control
            carla_env.vehicle.apply_control(carla.VehicleControl(
                throttle=throttle_cmd,
                steer=np.clip(steer_cmd / math.radians(30), -1.0, 1.0),
                brake=brake_cmd,
                reverse=bool(should_reverse)
            ))

            # Track proximity to goal
            if distance_to_goal < 0.8:
                self.close_count += 1
            else:
                self.close_count = 0

            # Check transition conditions with stable criteria
            handoff_x_line = -370  # This is the X-coordinate that triggers the switch to Phase 2
            orientation_ok = yaw_error < self.GAMMA_SWITCH

            # The vehicle's x-position starts high (e.g., -375) and decreases.
            # We trigger the handoff once it crosses the line.
            print(x0[0])
            if x0[0] >= handoff_x_line - 0.5 and x0[0] <= handoff_x_line + 0.5 and orientation_ok:
                print(f">>> Handoff line at x={handoff_x_line} crossed! Vehicle x={x0[0]:.2f}")
                print(">>> Switching to NMPC parking.")
                # Smooth stop
                carla_env.vehicle.apply_control(carla.VehicleControl(brake=0.5))
                time.sleep(0.3)
                return True

            # Update display
            direction_str = "REVERSE" if should_reverse else "FORWARD"
            hud_text = f"APPROACH | {direction_str} | dist: {distance_to_goal:.2f}m"
            
            if self.wrong_direction_count > 0:
                hud_text += f" | WRONG DIR: {self.wrong_direction_count}"
            
            if obstacle_status != "CLEAR":
                hud_text += f" | OBS: {obstacle_status}"
            
            display_lines = [
                hud_text,
                f"Pos: ({x0[0]:.1f}, {x0[1]:.1f}) -> ({staging_pose[0]:.1f}, {staging_pose[1]:.1f})",
                f"Steer: {steer_cmd:.2f} | Speed: {v_current:.2f}->{target_speed:.2f}"
            ]
            
            if distance_to_goal < 2.0:
                current_heading_deg = math.degrees(x0[2])
                target_heading_deg = math.degrees(staging_pose[2])
                display_lines.append(f"Heading: {current_heading_deg:.1f}° -> {target_heading_deg:.1f}°")
            
            time.sleep(DT_APPLY)

class NMPCController:
    def __init__(self,
             L=2.7,
             dt_plan=0.15,  # OPTIMIZATION: Slightly reduced from 0.2 for better dynamics
             N=10,          # OPTIMIZATION: Increased from 8 to 10 for better planning
             Q_diag=[0.08, 0.08, 0.08],    # OPTIMIZATION: Slightly increased for stability
             R_diag=[0.01, 0.01],          # OPTIMIZATION: Slightly increased for smoothness
             Qf_diag=[2.0, 4.0, 10.0],     # OPTIMIZATION: Reduced terminal weights slightly
             obstacle_weight=80,            # OPTIMIZATION: Further reduced
             obstacle_sigma=0.4,            # OPTIMIZATION: Increased for smoother gradients
             reverse_penalty=0.1,           # OPTIMIZATION: Further reduced
             ego_length=5.0,
             ego_width=1.5,
             steer_rate_weight=1.0,         # OPTIMIZATION: Further reduced
             dist_slow_thresh=1.0,
             slow_weight_v=0.3,             # OPTIMIZATION: Further reduced
             slow_weight_delta=0.0,
             max_obstacles=MAX_OBSTACLES,
             goal_pose_init=None,
             goal_threshold_pos_init=0.5,   # OPTIMIZATION: Further relaxed
             goal_threshold_angle_init=math.radians(8),  # OPTIMIZATION: Further relaxed
             max_converge_samples_init=12,  # OPTIMIZATION: Reduced
             converge_tolerance_init=0.2,   # OPTIMIZATION: Further relaxed
             # Distance-based speed limiting parameters
             max_speed_far=1.0,             # OPTIMIZATION: Reduced from 1.2
             max_speed_near=0.5,            # OPTIMIZATION: Reduced from 0.6
             max_speed_very_close=0.25,     # OPTIMIZATION: Reduced from 0.3
             distance_near=2.5,
             distance_very_close=1.0):

        self.L = L
        self.DT_PLAN = dt_plan
        self.N = N
        self.MAX_OBSTACLES = max_obstacles

        self.Q = ca.diag(Q_diag)
        self.R = ca.diag(R_diag)
        self.Qf = ca.diag(Qf_diag)
        self.OBSTACLE_WEIGHT = obstacle_weight
        self.OBSTACLE_SIGMA = obstacle_sigma
        self.REVERSE_PENALTY = reverse_penalty
        self.EGO_LENGTH = ego_length
        self.EGO_WIDTH = ego_width
        self.STEER_RATE_WEIGHT = steer_rate_weight
        self.DIST_SLOW_THRESH = dist_slow_thresh
        self.SLOW_WEIGHT_V = slow_weight_v
        self.SLOW_WEIGHT_DELTA = slow_weight_delta

        # Distance-based speed limiting
        self.MAX_SPEED_FAR = max_speed_far
        self.MAX_SPEED_NEAR = max_speed_near
        self.MAX_SPEED_VERY_CLOSE = max_speed_very_close
        self.DISTANCE_NEAR = distance_near
        self.DISTANCE_VERY_CLOSE = distance_very_close

        if goal_pose_init is None:
            raise ValueError("goal_pose_init must be provided to NMPCController.")
        self.goal_pose = list(goal_pose_init)
        self.goal_threshold_pos = goal_threshold_pos_init
        self.goal_threshold_angle = goal_threshold_angle_init
        self.max_converge_samples = max_converge_samples_init
        self.converge_tolerance = converge_tolerance_init
        self.converge_data = []

        # OPTIMIZATION: Initialize thread pool for parallel computation
        self.executor = ThreadPoolExecutor(max_workers=2)

        # === NEW: Inference time tracking ===
        self.inference_times = []
        self.max_inference_history = 100  # Keep last 100 inference times
        self.total_inference_time = 0.0
        self.inference_count = 0

        self._build_model()
        self._build_nlp()

        self.latest_optimal_v_command = 0.0
        self.latest_optimal_delta_command = 0.0

        # Initialize warm-start solutions
        self.u_prev_solution = np.tile([[0.0], [0.0]], (1, self.N))
        self.x_prev_solution = np.zeros((3, self.N + 1))

    def add_inference_time(self, solve_time):
        """Add a new inference time measurement and update statistics"""
        self.inference_times.append(solve_time)
        self.total_inference_time += solve_time
        self.inference_count += 1
        
        # Keep only the most recent measurements to prevent memory growth
        if len(self.inference_times) > self.max_inference_history:
            oldest_time = self.inference_times.pop(0)
            self.total_inference_time -= oldest_time
            self.inference_count -= 1

    def get_average_inference_time(self):
        """Get the current average inference time"""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count

    def get_recent_average_inference_time(self, num_recent=10):
        """Get average of the most recent N inference times"""
        if len(self.inference_times) == 0:
            return 0.0
        recent_times = self.inference_times[-min(num_recent, len(self.inference_times)):]
        return sum(recent_times) / len(recent_times)

    def get_inference_statistics(self):
        """Get comprehensive inference time statistics"""
        if len(self.inference_times) == 0:
            return {
                'count': 0,
                'average': 0.0,
                'recent_average': 0.0,
                'min': 0.0,
                'max': 0.0,
                'current': 0.0
            }
        
        return {
            'count': len(self.inference_times),
            'average': self.get_average_inference_time(),
            'recent_average': self.get_recent_average_inference_time(),
            'min': min(self.inference_times),
            'max': max(self.inference_times),
            'current': self.inference_times[-1] if self.inference_times else 0.0
        }

    def reset_inference_statistics(self):
        """Reset all inference time statistics"""
        self.inference_times = []
        self.total_inference_time = 0.0
        self.inference_count = 0

    @staticmethod
    def _linear_angle_diff_casadi(a, b):
        """OPTIMIZATION: Fast linear angle difference for CasADi (simplified)"""
        diff = a - b
        # Simple wrapping without trigonometric functions for speed
        return ca.if_else(diff > math.pi, diff - 2*math.pi, 
                         ca.if_else(diff < -math.pi, diff + 2*math.pi, diff))

    def get_speed_limit_for_distance(self, pos_dist):
        """Calculate maximum allowed speed based on distance to goal"""
        if pos_dist > self.DISTANCE_NEAR:
            return self.MAX_SPEED_FAR
        elif pos_dist > self.DISTANCE_VERY_CLOSE:
            # Linear interpolation between near and very close
            progress = (pos_dist - self.DISTANCE_VERY_CLOSE) / (self.DISTANCE_NEAR - self.DISTANCE_VERY_CLOSE)
            return self.MAX_SPEED_VERY_CLOSE + progress * (self.MAX_SPEED_NEAR - self.MAX_SPEED_VERY_CLOSE)
        else:
            return self.MAX_SPEED_VERY_CLOSE

    def _build_model(self):
        # === CasADi model ===
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)

        v = ca.SX.sym('v_control')
        delta = ca.SX.sym('delta_control')
        controls = ca.vertcat(v, delta)

        rhs = ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            v * ca.tan(delta) / self.L
        )

        # OPTIMIZATION: Ultra-aggressive compilation flags
        opts = {
            'jit': True, 
            'compiler': 'shell', 
            'jit_options': {
                'compiler': 'gcc', 
                'flags': ['-Ofast', '-ffast-math', '-march=native', '-funroll-loops', '-fomit-frame-pointer']
            }
        }
        self.f = ca.Function('f', [states, controls], [rhs], opts)

    def _build_nlp(self):
        # === NMPC Variables ===
        self.X_sym = ca.SX.sym('X', 3, self.N + 1)
        self.U_sym = ca.SX.sym('U', 2, self.N)
        self.P_sym = ca.SX.sym('P', 3 + 3 + 3 * self.MAX_OBSTACLES)

        cost = 0
        g = []

        # Initial state constraint
        g.append(self.X_sym[:, 0] - self.P_sym[0:3])

        # Extract goal state from P_sym
        goal_state_ref = self.P_sym[3:6]

        # Extract obstacle data from P_sym
        obs_params_flat = self.P_sym[6:]
        current_obs_sym = ca.reshape(obs_params_flat, 3, self.MAX_OBSTACLES)

        for k in range(self.N):
            st = self.X_sym[:, k]
            con = self.U_sym[:, k]
            ref = goal_state_ref

            e_xy = st[0:2] - ref[0:2]
            # OPTIMIZATION: Use linear angle difference instead of atan2
            e_theta = self._linear_angle_diff_casadi(st[2], ref[2])

            cost += (
                ca.mtimes([e_xy.T, self.Q[0:2, 0:2], e_xy])
                + self.Q[2, 2] * e_theta**2
                + ca.mtimes([con.T, self.R, con])
            )

            if k > 0:
                delta_steer = self.U_sym[1, k] - self.U_sym[1, k-1]
                cost += self.STEER_RATE_WEIGHT * delta_steer**2

            # OPTIMIZATION: Simplified distance calculation for speed
            dx_goal = st[0] - ref[0]
            dy_goal = st[1] - ref[1]
            dist_goal_sq = dx_goal**2 + dy_goal**2  # Avoid sqrt for speed
            slow_scale = ca.fmax(0, 1 - dist_goal_sq / (self.DIST_SLOW_THRESH**2))

            reverse_cost = self.REVERSE_PENALTY * ca.fmax(-con[0], 0)**2
            cost += reverse_cost

            cost += self.SLOW_WEIGHT_V * slow_scale * con[0]**2
            cost += self.SLOW_WEIGHT_DELTA * slow_scale * con[1]**2

            # OPTIMIZATION: Simplified distance-based obstacle avoidance
            for i in range(self.MAX_OBSTACLES):
                dx = st[0] - current_obs_sym[0, i]
                dy = st[1] - current_obs_sym[1, i]
                
                # OPTIMIZATION: Simple distance-based penalty, no complex geometry
                dist_sq = dx**2 + dy**2
                # Use a simpler exponential penalty based on distance squared
                cost += self.OBSTACLE_WEIGHT * ca.exp(-dist_sq / (2 * self.OBSTACLE_SIGMA**2))

            st_next = self.X_sym[:, k + 1]
            st_next_pred = st + self.DT_PLAN * self.f(st, con)
            g.append(st_next - st_next_pred)

        # Terminal cost with higher weights
        e_xy_T = self.X_sym[0:2, -1] - goal_state_ref[0:2]
        e_theta_T = self._linear_angle_diff_casadi(self.X_sym[2, -1], goal_state_ref[2])

        cost += ca.mtimes([e_xy_T.T, self.Qf[0:2, 0:2], e_xy_T]) \
              + self.Qf[2, 2] * e_theta_T**2

        opt_vars = ca.vertcat(ca.reshape(self.U_sym, -1, 1), ca.reshape(self.X_sym, -1, 1))

        nlp = {'f': cost, 'x': opt_vars, 'g': ca.vertcat(*g), 'p': self.P_sym}

        # OPTIMIZATION: More robust IPOPT settings for reliable convergence
        solver_opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-2,                          # Relaxed tolerance
            'ipopt.max_iter': 100,                      # Increased significantly from 35
            'ipopt.acceptable_tol': 5e-2,               # Very relaxed fallback tolerance
            'ipopt.acceptable_iter': 10,                # Accept after 10 iterations at fallback
            'ipopt.max_cpu_time': 0.15,                 # Increased to 150ms
            'ipopt.limited_memory_max_history': 10,     # More memory for convergence
            'ipopt.linear_solver': 'mumps',
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.constr_viol_tol': 1e-2,              # Relaxed constraint violation
            'ipopt.dual_inf_tol': 1e2,                  # Relaxed dual infeasibility
            'ipopt.compl_inf_tol': 1e-2,                # Relaxed complementarity
            'ipopt.mu_init': 1e-1,                      # Larger initial barrier parameter
            'ipopt.alpha_for_y': 'primal-and-full'     # More robust step calculation
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opts)
       
        # OPTIMIZATION: Relaxed constraints for speed
        self.lbg = [0.0] * (3 * (self.N + 1))
        self.ubg = [0.0] * (3 * (self.N + 1))

        self.lbx = []
        self.ubx = []
        for _ in range(self.N):
            # OPTIMIZATION: More conservative bounds for better convergence
            self.lbx += [-1.2, -math.radians(25)]       # Tighter velocity and steering bounds
            self.ubx += [1.2, math.radians(25)]
        for _ in range(self.N + 1):
            self.lbx += [-1e10, -1e10, -math.pi * 2]
            self.ubx += [1e10, 1e10, math.pi * 2]

    def _vectorized_warm_start(self, current_state):
        """OPTIMIZATION: Improved vectorized warm-start computation with smoother initialization"""
        # If we have a previous solution, shift it
        if hasattr(self, 'u_prev_solution') and self.u_prev_solution is not None:
            u_guess = np.roll(self.u_prev_solution, -1, axis=1)
            # Smooth the last control instead of just copying
            u_guess[0, -1] = u_guess[0, -2] * 0.8  # Gradually reduce velocity
            u_guess[1, -1] = u_guess[1, -2] * 0.5  # Gradually reduce steering
        else:
            # Initialize with very conservative defaults
            u_guess = np.zeros((2, self.N))
            # Very small forward velocity and zero steering
            u_guess[0, :] = 0.05  # Very small forward velocity
            u_guess[1, :] = 0.0   # Zero steering
        
        # Ensure control bounds are respected in initial guess
        u_guess[0, :] = np.clip(u_guess[0, :], -1.2, 1.2)
        u_guess[1, :] = np.clip(u_guess[1, :], -math.radians(25), math.radians(25))
        
        # Vectorized state prediction with current state
        x_guess = np.zeros((3, self.N + 1))
        x_guess[:, 0] = current_state  # Use actual current state
        
        # Batch predict all states at once (simplified dynamics)
        for k in range(self.N):
            v, delta = u_guess[0, k], u_guess[1, k]
            theta = x_guess[2, k]
            
            # More stable integration with smaller steps
            dt_substep = self.DT_PLAN / 2.0
            for _ in range(2):  # Two substeps for stability
                x_guess[0, k+1] = x_guess[0, k] + dt_substep * v * math.cos(theta)
                x_guess[1, k+1] = x_guess[1, k] + dt_substep * v * math.sin(theta)
                x_guess[2, k+1] = normalize_angle(x_guess[2, k] + dt_substep * v * math.tan(delta) / self.L)
                # Update for next substep
                if _ == 0:
                    x_guess[0, k] = x_guess[0, k+1]
                    x_guess[1, k] = x_guess[1, k+1] 
                    x_guess[2, k] = x_guess[2, k+1]
        
        return np.concatenate([u_guess.flatten(), x_guess.flatten()])

    def check_goal_and_convergence(self, current_ego_state):
        """
        Checks if the ego vehicle has reached the goal and assesses convergence/stagnation.
        """
        pos_dist = math.hypot(current_ego_state[0] - self.goal_pose[0],
                              current_ego_state[1] - self.goal_pose[1])
        angle_error = abs(normalize_angle(current_ego_state[2] - self.goal_pose[2]))

        self.converge_data.append(pos_dist)
        if len(self.converge_data) > self.max_converge_samples:
            self.converge_data.pop(0)

        is_stagnated = (len(self.converge_data) == self.max_converge_samples and
                        (max(self.converge_data) - min(self.converge_data)) < self.converge_tolerance)

        goal_pos_ok = pos_dist < self.goal_threshold_pos
        goal_angle_ok = angle_error < self.goal_threshold_angle
        goal_met = goal_pos_ok and goal_angle_ok

        return goal_met, pos_dist, angle_error, is_stagnated

    def solve(self, p_values, initial_guess_flat):
        """OPTIMIZATION: Faster solve with improved fallback strategies"""
        try:
            solution = self.solver(x0=initial_guess_flat,
                                   lbx=self.lbx, ubx=self.ubx,
                                   lbg=self.lbg, ubg=self.ubg,
                                   p=p_values)
            
            u_opt_flat = solution['x'][:2 * self.N]
            x_opt_flat = solution['x'][2 * self.N:]
            
            u_optimal = ca.reshape(u_opt_flat, 2, self.N).full()
            x_optimal = ca.reshape(x_opt_flat, 3, self.N + 1).full()
            
            success = self.solver.stats()['success']
            status = self.solver.stats()['return_status']
            
            if success or status == 'Solved_To_Acceptable_Level':
                self.latest_optimal_v_command = float(u_optimal[0, 0])
                self.latest_optimal_delta_command = float(u_optimal[1, 0])
                # Update warm-start solutions
                self.u_prev_solution = u_optimal
                self.x_prev_solution = x_optimal
                return u_optimal, x_optimal, True, status
            else:
                # Use previous solution if available
                if hasattr(self, 'u_prev_solution') and self.u_prev_solution is not None:
                    self.latest_optimal_v_command = float(self.u_prev_solution[0, 0])
                    self.latest_optimal_delta_command = float(self.u_prev_solution[1, 0])
                    return self.u_prev_solution, self.x_prev_solution, True, "Using_Previous"
                else:
                    # Fallback to safe controls
                    self.latest_optimal_v_command = 0
                    self.latest_optimal_delta_command = 0
                    return u_optimal, x_optimal, False, status
        
        except Exception as e:
            # Emergency fallback
            if hasattr(self, 'u_prev_solution') and self.u_prev_solution is not None:
                self.latest_optimal_v_command = float(self.u_prev_solution[0, 0])
                self.latest_optimal_delta_command = float(self.u_prev_solution[1, 0])
                return self.u_prev_solution, self.x_prev_solution, True, "Exception_Fallback"
            else:
                # Complete fallback
                u_fallback = np.zeros((2, self.N))
                x_fallback = np.zeros((3, self.N + 1))
                self.latest_optimal_v_command = 0
                self.latest_optimal_delta_command = 0
                return u_fallback, x_fallback, False, "Exception"

    def park_with_nmpc(self, carla_env, start_time, timeout_s, collision_flag, curb_hit_flag):
        """
        NMPC parking control loop with distance-based speed limiting and optimizations
        Enhanced with comprehensive inference time tracking
        """
        print(">>> PHASE 2: PARKING WITH NMPC...")

        # Reset inference statistics for this parking session
        self.reset_inference_statistics()

        while True:
            loop_start_time = time.time()
            current_ego_state = carla_env.get_ego_state()
            
            # Calculate current distance to goal
            pos_dist = math.hypot(current_ego_state[0] - self.goal_pose[0], 
                                 current_ego_state[1] - self.goal_pose[1])
            
            # === NMPC SOLVER WITH OPTIMIZATIONS ===
            data_prep_start = time.time()
            flat_obs_data = carla_env.get_obstacle_data(current_ego_state[0], current_ego_state[1])
            p_values_nmpc = np.array(current_ego_state + self.goal_pose + flat_obs_data)

            # OPTIMIZATION: Use vectorized warm-start with current state
            initial_solver_guess_flat = self._vectorized_warm_start(current_ego_state)
            data_prep_time = time.time() - data_prep_start
            
            # Solve NMPC with detailed timing
            solve_start_time = time.time()
            u_opt, x_opt, success, status = self.solve(p_values_nmpc, initial_solver_guess_flat)
            solve_time = time.time() - solve_start_time
            
            # === NEW: Track inference time ===
            self.add_inference_time(solve_time)
            
            # Get inference statistics
            stats = self.get_inference_statistics()
            
            # Enhanced inference time display
            print(f'Current inference time: {solve_time:.5f}s | '
                  f'Average: {stats["average"]:.5f}s | '
                  f'Recent avg (10): {stats["recent_average"]:.5f}s ')
            
            # Control application timing
            control_start = time.time()
            if success:
                # Apply distance-based speed limiting
                raw_v_cmd = float(u_opt[0, 0])
                max_allowed_speed = self.get_speed_limit_for_distance(pos_dist)
                
                # Limit the velocity command based on distance to goal
                v_cmd = np.clip(raw_v_cmd, -max_allowed_speed, max_allowed_speed)
                
                # Additional safety: if very close and moving fast, reduce further
                if pos_dist < 0.5:
                    v_cmd *= 0.5
                    
                delta_cmd = float(np.clip(u_opt[1, 0], -math.radians(30), math.radians(30)))
                
                # Apply enhanced vehicle control
                self._apply_enhanced_vehicle_control(carla_env, v_cmd, delta_cmd, pos_dist)
                
            else:
                print(f"NMPC solver failed: {status}")
                # Apply gentle brake if solver fails completely
                carla_env.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.3))
            
            control_time = time.time() - control_start
            total_loop_time = time.time() - loop_start_time

            # Check goal and convergence
            goal_met, pos_dist, angle_error, is_stagnated = self.check_goal_and_convergence(current_ego_state)
            
            if (pos_dist < self.goal_threshold_pos and angle_error < self.goal_threshold_angle) or is_stagnated:
                if is_stagnated and not goal_met:
                    print(f"Vehicle stagnated. Pos dist: {pos_dist:.3f}, Angle error: {math.degrees(angle_error):.2f}°")
                else:
                    print(f"Goal met: d={pos_dist:.2f} m, yaw_err={math.degrees(angle_error):.1f}°")
                
                # === NEW: Print final inference statistics ===
                final_stats = self.get_inference_statistics()
                print("\n=== FINAL INFERENCE TIME STATISTICS ===")
                print(f"Total solver calls: {final_stats['count']}")
                print(f"Average inference time: {final_stats['average']:.5f}s")
                print(f"Recent average (last 10): {final_stats['recent_average']:.5f}s")
                print(f"Minimum inference time: {final_stats['min']:.5f}s")
                print(f"Maximum inference time: {final_stats['max']:.5f}s")
                print(f"Final inference time: {final_stats['current']:.5f}s")
                print("=========================================\n")
                
                return goal_met or is_stagnated
            
            if pos_dist > 20.0:
                print("Vehicle diverged too far. Aborting.")
                return False

            # === Enhanced display with inference statistics ===
            display_lines = [
                f"OPTIMIZED PARK | dist: {pos_dist:.2f}m | time: {solve_time:.5f}s",
                f"Avg: {stats['average']:.5f}s | Recent: {stats['recent_average']:.5f}s | Count: {stats['count']}",
                f"Goal: ({self.goal_pose[0]:.1f}, {self.goal_pose[1]:.1f}, {math.degrees(self.goal_pose[2]):.1f}°)",
                f"Current: ({current_ego_state[0]:.1f}, {current_ego_state[1]:.1f}, {math.degrees(current_ego_state[2]):.1f}°)",
                f"Error: {pos_dist:.2f}m, {math.degrees(angle_error):.1f}°"
            ]
            
            if success and pos_dist < self.DISTANCE_NEAR:
                display_lines.append(f"CLOSE MODE | Status: {status}")
            
            # Add performance summary line
            if stats['count'] > 5:  # Only show after we have some data
                perf_line = f"Min: {stats['min']:.5f}s | Max: {stats['max']:.5f}s"
                display_lines.append(perf_line)
            
            time.sleep(DT_APPLY)

    def _apply_enhanced_vehicle_control(self, carla_env, v_cmd, delta_cmd, pos_dist):
        """
        Enhanced vehicle control with improved throttle/brake mapping
        """
        v_current = carla_env.vehicle.get_velocity().length()
        KP_VELOCITY = 0.3
        SPEED_DEAD_BAND = 0.05
        MAX_BRAKE_FOR_CORRECTION = 0.5
        
        ctrl = carla.VehicleControl(steer=delta_cmd / math.radians(30))
        speed_error = abs(v_cmd) - v_current

        if abs(v_cmd) < 1e-3:
            ctrl.throttle, ctrl.brake, ctrl.reverse = 0.0, 1.0, False
        else:
            ctrl.reverse = bool(v_cmd < 0)
            if abs(speed_error) < SPEED_DEAD_BAND:
                ctrl.throttle, ctrl.brake = 0.0, 0.0
            elif speed_error > 0:
                # Reduce throttle gain when close to goal for smoother control
                throttle_gain = KP_VELOCITY
                if pos_dist < 1.0:
                    throttle_gain *= 0.5
                ctrl.throttle = np.clip(throttle_gain * speed_error, 0.0, 0.1)
                ctrl.brake = 0.0
            else:
                ctrl.throttle = 0.0
                ctrl.brake = np.clip(KP_VELOCITY * abs(speed_error), 0.0, MAX_BRAKE_FOR_CORRECTION)
        
        carla_env.vehicle.apply_control(ctrl)

    def get_current_carla_control_object(self, vehicle, world):
        """
        Generate a CARLA VehicleControl object for the current timestep.
        This method adapts the NMPC controller for shared autonomy use.
        """
        try:
            # Get current vehicle state
            current_ego_state = self._get_ego_state_from_vehicle(vehicle)
            
            # Get obstacle data
            flat_obs_data = self._get_obstacle_data_from_world(world, current_ego_state[0], current_ego_state[1])
            
            # Prepare parameters for NMPC solver
            p_values_nmpc = np.array(current_ego_state + self.goal_pose + flat_obs_data)
            
            # OPTIMIZATION: Use vectorized warm-start with current state
            initial_solver_guess_flat = self._vectorized_warm_start(current_ego_state)
            
            # Solve NMPC
            u_opt, x_opt, success, status = self.solve(p_values_nmpc, initial_solver_guess_flat)
            
            if success:
                # Update warm-start solutions
                self.u_prev_solution = u_opt
                self.x_prev_solution = x_opt
                
                # Get control commands
                v_cmd = float(u_opt[0, 0])
                delta_cmd = float(u_opt[1, 0])
                
                # Apply distance-based speed limiting
                pos_dist = math.hypot(current_ego_state[0] - self.goal_pose[0], 
                                    current_ego_state[1] - self.goal_pose[1])
                max_allowed_speed = self.get_speed_limit_for_distance(pos_dist)
                v_cmd = np.clip(v_cmd, -max_allowed_speed, max_allowed_speed)
                
                # Convert to CARLA control
                return self._convert_to_carla_control(v_cmd, delta_cmd, vehicle)
            else:
                # If solver fails, return safe control (brake)
                return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5, reverse=False)
                
        except Exception as e:
            print(f"Error in get_current_carla_control_object: {e}")
            # Return safe control on any error
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5, reverse=False)

    def _get_ego_state_from_vehicle(self, vehicle):
        """Helper method to get ego state from CARLA vehicle"""
        if not vehicle:
            return [0, 0, 0]
        t = vehicle.get_transform()
        return [t.location.x, t.location.y, normalize_angle(math.radians(t.rotation.yaw))]

    def _get_obstacle_data_from_world(self, world, ego_x, ego_y):
        """Helper method to get obstacle data from CARLA world"""
        obs_data = []
        
        # Find the ego vehicle ID
        ego_id = None
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name", "") == "hero":
                ego_id = actor.id
                break
        
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id == ego_id:
                continue
            tx = actor.get_transform()
            obs_data.append([
                tx.location.x,
                tx.location.y,
                normalize_angle(math.radians(tx.rotation.yaw))
            ])
        
        # Sort by distance to ego
        obs_data.sort(key=lambda o: math.hypot(o[0] - ego_x, o[1] - ego_y))
        obs_data = obs_data[:self.MAX_OBSTACLES]
        
        # Pad if fewer than MAX_OBSTACLES
        while len(obs_data) < self.MAX_OBSTACLES:
            obs_data.append([1e5, 1e5, 0.0])
        
        return np.array(obs_data).flatten(order='F').tolist()

    def _convert_to_carla_control(self, v_cmd, delta_cmd, vehicle):
        """Convert NMPC commands to CARLA VehicleControl using REAL vehicle speed."""
        
        # --- THE FIX: Get the vehicle's actual current velocity ---
        v = vehicle.get_velocity()
        v_current = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # Constants
        KP_VELOCITY = 0.5  # Increased for better response
        SPEED_DEAD_BAND = 0.05
        MAX_BRAKE_FOR_CORRECTION = 0.5
        
        ctrl = carla.VehicleControl(steer=np.clip(delta_cmd / math.radians(30), -1.0, 1.0))
        
        # The error between desired absolute speed and current absolute speed
        speed_error = abs(v_cmd) - v_current
        
        if abs(v_cmd) < 1e-3 and v_current < SPEED_DEAD_BAND:
            # Command is to stop, and we are stopped -> Full brake
            ctrl.throttle, ctrl.brake, ctrl.reverse = 0.0, 1.0, False
        else:
            ctrl.reverse = bool(v_cmd < 0)
            
            if speed_error > SPEED_DEAD_BAND: # We need to accelerate
                ctrl.throttle = np.clip(KP_VELOCITY * speed_error, 0.0, 0.6) # Increased max throttle
                ctrl.brake = 0.0
            elif speed_error < -SPEED_DEAD_BAND: # We need to decelerate
                ctrl.throttle = 0.0
                ctrl.brake = np.clip(KP_VELOCITY * abs(speed_error), 0.0, MAX_BRAKE_FOR_CORRECTION)
            else: # We are at the correct speed
                ctrl.throttle = 0.0
                ctrl.brake = 0.0
                
        return ctrl

    def __del__(self):
        """OPTIMIZATION: Clean up thread pool"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
            
class CarlaEnvironment:
    def __init__(self, host="localhost", port=2000, timeout=10.0):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.actor_list = []
        
        # Collision and lane invasion sensors
        self.collision_sensor = None
        self.lane_sensor = None
        self.collision_flag = {'hit': False}
        self.curb_hit_flag = {'hit': False}
    
    def reset_for_new_trial(self, ego_spawn_transform, ego_bp_name="vehicle.audi.tt"):
        """
        Efficiently resets the environment for a new trial.
        Destroys only the old ego vehicle and its sensors, then spawns a new one.
        """
        # 1. Clean up the previous ego vehicle and its sensors
        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        if self.lane_sensor and self.lane_sensor.is_alive:
            self.lane_sensor.destroy()
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()

        # 2. Reset state flags
        self.collision_flag['hit'] = False
        self.curb_hit_flag['hit'] = False
        self.actor_list = [] # Clear the actor list

        # 3. Spawn the new ego vehicle
        vehicle_bp = self.blueprint_library.find(ego_bp_name)
        vehicle_bp.set_attribute("role_name", "hero")
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, ego_spawn_transform)
        if self.vehicle is None:
            ego_spawn_transform.location.z += 0.5
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, ego_spawn_transform)
            if self.vehicle is None:
                raise RuntimeError(f"Failed to spawn ego vehicle for trial: {ego_spawn_transform}")
        
        self.vehicle.set_autopilot(False)
        self.actor_list.append(self.vehicle)
        
        # 4. Setup sensors for the new vehicle
        self._setup_sensors()
        print(f"Ego vehicle for new trial {self.vehicle.id} spawned.")

    def setup_simulation(self, ego_spawn_transform, ego_bp_name="vehicle.audi.tt", parking_scenario_name="Parallel Parking"):
        # Clear existing non-hero vehicles
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name", "") != "hero":
                actor.destroy()
        
        # Clean up any existing hero vehicle
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name", "") == "hero":
                print("Destroying existing hero vehicle.")
                actor.destroy()
        
        # Clean up existing sensors
        for actor in self.world.get_actors().filter("sensor.*"):
            actor.destroy()
        
        time.sleep(0.5)

        # Spawn Parking NPCs
        custom_actor_spawn(self.client, parking_scenario_name, dist_dev=0)
        time.sleep(1)

        # Spawn Ego Vehicle
        vehicle_bp = self.blueprint_library.find(ego_bp_name)
        vehicle_bp.set_attribute("role_name", "hero")
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, ego_spawn_transform)
        if self.vehicle is None:
            ego_spawn_transform.location.z += 0.5 
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, ego_spawn_transform)
            if self.vehicle is None:
                raise RuntimeError(f"Failed to spawn ego vehicle: {ego_spawn_transform}")
        
        self.vehicle.set_autopilot(False)
        self.actor_list.append(self.vehicle)
        
        # Setup sensors
        self._setup_sensors()
        
        print(f"Ego vehicle {self.vehicle.id} spawned.")

    def _setup_sensors(self):
        """Setup collision and lane invasion sensors"""
        # Collision sensor
        def on_collision(event): 
            self.collision_flag['hit'] = True
        
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(on_collision)
        
        # Lane invasion sensor (for curb detection)
        def on_curb_hit(event):
            if any(m.type == carla.LaneMarkingType.Curb for m in event.crossed_lane_markings):
                self.curb_hit_flag['hit'] = True
        
        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(on_curb_hit)

    def setup_static_birds_eye_view(self, goal_pose):
        """
        Setup a static bird's eye view camera positioned to the left of the final parking position
        
        Args:
            goal_pose: [x, y, theta] of the final parking position
        """
        # Position camera to the left of the goal position
        camera_x = goal_pose[0] - 15.0  # 15 meters to the left
        camera_y = goal_pose[1]         # Same Y as goal
        camera_z = goal_pose[2] if len(goal_pose) > 2 and isinstance(goal_pose[2], (int, float)) and goal_pose[2] > 100 else 180.0  # Use Z if provided, else default
        camera_z += 25.0  # 25 meters above ground
        
        # Create camera transform looking down
        camera_location = carla.Location(x=camera_x, y=camera_y, z=camera_z)
        camera_rotation = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)  # Looking straight down
        camera_transform = carla.Transform(camera_location, camera_rotation)
        
        # Set spectator camera
        spectator = self.world.get_spectator()
        spectator.set_transform(camera_transform)
        
        print(f"Static bird's eye view camera set at ({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f})")

    def get_ego_state(self):
        if not self.vehicle:
            return [0,0,0]
        t = self.vehicle.get_transform()
        return [t.location.x, t.location.y, normalize_angle(math.radians(t.rotation.yaw))]

    def get_obstacle_data(self, current_ego_pos_x, current_ego_pos_y):
        obs_data = []
        if not self.vehicle:
            return obs_data 

        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.vehicle.id:
                continue
            tx = actor.get_transform()
            obs_data.append([
                tx.location.x,
                tx.location.y,
                normalize_angle(math.radians(tx.rotation.yaw))
            ])
        
        # Sort by distance to ego
        obs_data.sort(key=lambda o: math.hypot(o[0] - current_ego_pos_x, o[1] - current_ego_pos_y))
        obs_data = obs_data[:MAX_OBSTACLES]
        
        # Pad if fewer than MAX_OBSTACLES
        while len(obs_data) < MAX_OBSTACLES:
            obs_data.append([1e5, 1e5, 0.0])
        
        return np.array(obs_data).flatten(order='F').tolist()

    def freeze_vehicle(self):
        if self.vehicle:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))

    def cleanup(self):
        print("Destroying spawned actors...")
        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        if self.lane_sensor and self.lane_sensor.is_alive:
            self.lane_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None

def run_two_phase_parking(
    ego_spawn_transform,
    staging_pose,
    goal_pose,
    timeout_s=40.0,
    controller_params=None,
    approach_params=None,
    wait_on_completion=True,
    existing_env=None
):
    """
    Main orchestrator function that runs the optimized two-phase parking system
    """
    if controller_params is None:
        # OPTIMIZATION: Updated default parameters for better convergence
        controller_params = {
            "L": 2.7,
            "dt_plan": 0.15,                            # Reduced from 0.2
            "N": 10,                                    # Increased from 8
            "Q_diag": [0.08, 0.08, 0.08],               # Slightly increased
            "R_diag": [0.01, 0.01],                     # Slightly increased
            "Qf_diag": [2.0, 4.0, 10.0],                # Reduced terminal weights
            "obstacle_weight": 80,                      # Further reduced
            "obstacle_sigma": 0.4,                      # Increased for smoother gradients
            "reverse_penalty": 0.1,                     # Further reduced
            "ego_length": 5.0,
            "ego_width": 1.5,
            "steer_rate_weight": 1.0,                   # Further reduced
            "dist_slow_thresh": 1.0,
            "slow_weight_v": 0.3,                       # Further reduced
            "slow_weight_delta": 0.0,
            "max_obstacles": MAX_OBSTACLES,
            "goal_threshold_pos_init": 0.5,             # Further relaxed
            "goal_threshold_angle_init": math.radians(8), # Further relaxed
            "max_converge_samples_init": 12,            # Reduced
            "converge_tolerance_init": 0.2              # Further relaxed
        }
    
    if approach_params is None:
        approach_params = {
            "target_speed": 0.8,
            "kp_speed": 0.4,
            "d_switch": 2.5,
            "gamma_switch": math.radians(15),
            "stop_distance": 2.0,
            "slow_distance": 4.0
        }

    # Handle environment setup
    if existing_env is None:
        # If no environment is provided, create and set up a new one.
        # This is for running the script as a standalone file.
        carla_env = CarlaEnvironment()
        carla_env.setup_simulation(ego_spawn_transform)
        is_managed_externally = False
    else:
        # If an environment is provided, use it and just reset the ego vehicle.
        # This is for the automated test loop.
        carla_env = existing_env
        carla_env.reset_for_new_trial(ego_spawn_transform)
        is_managed_externally = True

    # Initialize components
    approach_controller = ApproachController(**approach_params)
    nmpc_controller = NMPCController(**controller_params, goal_pose_init=goal_pose)

    try:
        start_time = time.time()

        # Phase 1: Approach to staging pose
        print("=== STARTING PHASE 1: APPROACH ===")
        approach_success = approach_controller.approach_target(
            carla_env, staging_pose, start_time, timeout_s,
            carla_env.collision_flag, carla_env.curb_hit_flag,
        )

        if not approach_success:
            print("Overall trial failed during approach phase.")
            return False

        # Phase 2: OPTIMIZED NMPC parking
        print("=== STARTING PHASE 2: OPTIMIZED NMPC PARKING ===")
        park_success = nmpc_controller.park_with_nmpc(
            carla_env, start_time, timeout_s,
            carla_env.collision_flag, carla_env.curb_hit_flag,
        )

        if park_success:
            print("TRIAL SUCCEEDED: Vehicle parked successfully with optimizations.")
            return True
        else:
            print("Overall trial failed during parking phase.")
            return False

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("Entering finalization stage...")
        if carla_env and carla_env.vehicle:
            carla_env.freeze_vehicle()

