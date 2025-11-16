#!/usr/bin/env python3
# pico_server.py

import asyncio # we use this in line 138.
import math
from tf_transformations import euler_from_quaternion

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# --- Imports ---
from waypoint_navigation.action import NavToWaypoint
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry

# These were just suggested to import I dont know why?
from error_msg.msg import Error
from controller_msg.msg import PIDTune
from collections import deque


class WayPointServer(Node):
    def __init__(self):
        super().__init__('waypoint_server')

        self.pid_or_lqr_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()
        self.odometry_callback_group = ReentrantCallbackGroup()

        # timing / stabilization book keeping
        self.time_inside_sphere = 0
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None # float seconds.
        self.duration = 0

        self.yaw = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        
        self.xyz = [0.0, 0.0, 0.0, 0.0]     # This line is of no use.
        self.dtime = 0

        # Declaring a cmd of message type swift_msgs and initializing values
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1400

        self.drone_position = [0.0, 0.0, 0.0]
        self.desired_state = [0.0, 0.0, 0.0]
        self.curr_state = [0.0, 0.0, 0.0]
        self.timer = None  # Will be initialized after action server
        


        self.sample_time = 0.03333 # (30 Hz)    put the appropriate value according to your controller 

        # --- PID / filter related defaults (merged from PID_new_Dipankar.py)
        self.m = 0.152
        self.g = 9.81

        # Filter buffers
        self.altitude_window = deque(maxlen=3)
        self.avg_window = deque(maxlen=2)

        # PID gains (index: 0->pitch/x, 1->roll/y, 2->throttle/z)  
        self.Kp = [300.0 * 0.06, 300.0 * 0.06, 725 * 0.06]      # Reduced pitch/roll Kp, throttle unchanged
        self.Ki = [25.0 * 0.0008, 25.0 * 0.0008, 2320 * 0.0008] # Lower pitch/roll Ki, throttle unchanged
        self.Kd = [20.0 * 0.3, 20.0 * 0.3, 388.5 * 0.3]         # Reduced pitch/roll Kd, throttle unchanged

        # PID state
        self.prev_error = [0.0, 0.0, 0.0]
        self.sum_error = [0.0, 0.0, 0.0]
        self.pos_error = Error()

        # command & integrator limits
        self.max_values = [2000, 2000, 2000]
        self.min_values = [1000, 1000, 1000]
        self.error_sum_limits = [500, 500, 150]  # Reduced throttle integral limit from 300 to 150
        self.throttle_trim = 1458

        # Butterworth filter design
        self.cutoff_freq = 3.5
        self.filter_order = 2
        self.b_biquad = [0.0, 0.0, 0.0]
        self.a_biquad = [1.0, 0.0, 0.0]
        self.z_x1 = self.z_x2 = self.z_y1 = self.z_y2 = 0.0

        # derivative filter
        self.d_filter_alpha = 0.75
        self.prev_d_throttle = 0.0

        # anti-windup
        self.enable_integrator_when_saturated = False

        # design filters using current sample_time
        try:
            self._design_butterworth(self.cutoff_freq, self.sample_time, self.filter_order)
        except Exception:
            pass


        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/position_error', 10)
        # Subscribers
        self.whycon_sub = self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, '/rotors/odometry', self.odometry_callback, 10, callback_group=self.odometry_callback_group)

        # Action Server
        self._action_server = ActionServer(
            self,
            NavToWaypoint,
            'waypoint_navigation',
            execute_callback=self.execute_callback,
            callback_group=self.action_callback_group,
        )

        # Initialize PID timer immediately
        self.timer = self.create_timer(self.sample_time, self.pid_or_lqr_timer_callback, callback_group=self.pid_or_lqr_callback_group)

        self.get_logger().info('Waypoint Server initialized.')

        ## pid controller
        self.target_pub = self.create_publisher(PointStamped, '/target_point', 10)

        # PID tune topic subscribers (allow runtime tuning)
        self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1)

        # ---------------------------------------------------------------------- #

        ## This was later added 
        self.arm()
        self.get_logger().info('PID Timer already created in constructor - controller ready')
    # PID or LQR controller timer callback
    def pid_or_lqr_timer_callback(self):
        # Force PID enabled if we're in waypoint mode (prevents landing between waypoints)
        if getattr(self, 'waypoint_mode_active', False):
            self.pid_enabled = True
            
        if not getattr(self, 'pid_enabled', False):
            # PID disabled: do nothing and return
            if not hasattr(self, 'pid_disabled_logged'):
                self.get_logger().warn('PID DISABLED - drone will land! Waiting for goal...')
                self.pid_disabled_logged = True
            return
        
        # Reset the disabled flag when PID becomes enabled
        if hasattr(self, 'pid_disabled_logged'):
            delattr(self, 'pid_disabled_logged')

        # PID loop adapted from PID_new_Dipankar.py
        try:
            median_z = self._median_filter(self.curr_state[2])
            butter_z = self._butterworth_filter(median_z)
            filtered_z = self._moving_average(butter_z)
        except Exception:
            filtered_z = self.curr_state[2]

        # compute errors (current - desired, as it was working before)
        error_x = self.curr_state[0] - self.desired_state[0]
        error_y = self.curr_state[1] - self.desired_state[1]
        error_z = filtered_z - self.desired_state[2]
        current_errors = [error_x, error_y, error_z]

        # Pitch PID
        p_term_pitch = self.Kp[0] * error_x
        d_term_pitch = self.Kd[0] * (error_x - self.prev_error[0]) / max(self.sample_time, 1e-6)
        self.sum_error[0] = max(min(self.sum_error[0] + error_x * self.sample_time, self.error_sum_limits[0]), -self.error_sum_limits[0])
        i_term_pitch = self.Ki[0] * self.sum_error[0]

        # Roll PID
        p_term_roll = self.Kp[1] * error_y
        d_term_roll = self.Kd[1] * (error_y - self.prev_error[1]) / max(self.sample_time, 1e-6)
        self.sum_error[1] = max(min(self.sum_error[1] + error_y * self.sample_time, self.error_sum_limits[1]), -self.error_sum_limits[1])
        i_term_roll = self.Ki[1] * self.sum_error[1]

        # Throttle PID
        p_term_throttle = self.Kp[2] * error_z
        raw_d_throttle = self.Kd[2] * (error_z - self.prev_error[2]) / max(self.sample_time, 1e-6)
        d_filtered = self.d_filter_alpha * self.prev_d_throttle + (1.0 - self.d_filter_alpha) * raw_d_throttle
        self.prev_d_throttle = d_filtered

        tentative_out_no_i = p_term_throttle + d_filtered
        current_cmd_no_i = int(self.throttle_trim + tentative_out_no_i)

        if self.enable_integrator_when_saturated or (self.min_values[2] < current_cmd_no_i < self.max_values[2]):
            self.sum_error[2] += error_z * self.sample_time
        else:
            test_i = self.sum_error[2] + error_z * self.sample_time
            test_cmd = int(self.throttle_trim + tentative_out_no_i + self.Ki[2] * test_i)
            if self.min_values[2] < test_cmd < self.max_values[2]:
                self.sum_error[2] = test_i

        self.sum_error[2] = max(min(self.sum_error[2], self.error_sum_limits[2]), -self.error_sum_limits[2])
        i_term_throttle = self.Ki[2] * self.sum_error[2]

        out_pitch = p_term_pitch + i_term_pitch + d_term_pitch
        out_roll = p_term_roll + i_term_roll + d_term_roll
        out_throttle = p_term_throttle + i_term_throttle + d_filtered

        # Apply to command message and saturate (Gazebo simulation - pitch inverted)
        self.cmd.rc_pitch = max(min(1500 - int(out_pitch), self.max_values[1]), self.min_values[1])
        self.cmd.rc_roll = max(min(1500 - int(out_roll), self.max_values[0]), self.min_values[0])
        self.cmd.rc_throttle = max(min(int(self.throttle_trim + out_throttle), self.max_values[2]), self.min_values[2])

        self.prev_error = current_errors

        # Debug: Print errors and commands every 30 loops (~0.5 seconds)
        if not hasattr(self, 'debug_counter'):
            self.debug_counter = 0
        self.debug_counter += 1
        distance = ((error_x**2 + error_y**2 + error_z**2) ** 0.5)
        if self.debug_counter % 30 == 0:
            self.get_logger().info(f'Current: [{self.curr_state[0]:.2f}, {self.curr_state[1]:.2f}, {self.curr_state[2]:.2f}]')
            self.get_logger().info(f'Desired: [{self.desired_state[0]:.2f}, {self.desired_state[1]:.2f}, {self.desired_state[2]:.2f}]')
            self.get_logger().info(f'Errors: [{error_x:.2f}, {error_y:.2f}, {error_z:.2f}]')
            self.get_logger().info(f'RC Commands: pitch={self.cmd.rc_pitch}, roll={self.cmd.rc_roll}, throttle={self.cmd.rc_throttle}')
            self.get_logger().info(f'Distance: {distance:.2f} meters')

        # Publish
        self.command_pub.publish(self.cmd)
        self.pos_error.pitch_error = float(error_x)
        self.pos_error.roll_error = float(error_y)
        self.pos_error.throttle_error = float(error_z)
        self.pos_error_pub.publish(self.pos_error)

    # ---------------------------- Filters & PID setters ----------------------------
    def _median_filter(self, new_value):
        self.altitude_window.append(new_value)
        if len(self.altitude_window) < 3:
            return new_value
        sorted_vals = sorted(self.altitude_window)
        return sorted_vals[1]

    def _design_butterworth(self, fc, sample_time, order=2):
        fs = 1.0 / sample_time
        if fc <= 0 or sample_time <= 0:
            self.b_biquad = [1.0, 0.0, 0.0]
            self.a_biquad = [1.0, 0.0, 0.0]
            return

        Q = 1.0 / math.sqrt(2.0)
        w0 = 2.0 * math.pi * fc / fs
        w0 = max(min(w0, math.pi - 1e-6), 1e-6)
        alpha = math.sin(w0) / (2.0 * Q)
        cos_w0 = math.cos(w0)

        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        self.b_biquad = [b0 / a0, b1 / a0, b2 / a0]
        self.a_biquad = [1.0, a1 / a0, a2 / a0]
        self.z_x1 = self.z_x2 = self.z_y1 = self.z_y2 = 0.0

    def _butterworth_filter(self, x):
        b0, b1, b2 = self.b_biquad
        a0, a1, a2 = self.a_biquad
        y = b0 * x + b1 * self.z_x1 + b2 * self.z_x2 - a1 * self.z_y1 - a2 * self.z_y2
        self.z_x2, self.z_x1 = self.z_x1, x
        self.z_y2, self.z_y1 = self.z_y1, y
        return y

    def _moving_average(self, new_value):
        self.avg_window.append(new_value)
        return sum(self.avg_window) / len(self.avg_window)

    def altitude_set_pid(self, alt: PIDTune):
        try:
            self.Kp[2] = alt.kp * 0.06
            self.Ki[2] = alt.ki * 0.0008
            self.Kd[2] = alt.kd * 0.3
        except Exception:
            pass

    def pitch_set_pid(self, pitch: PIDTune):
        try:
            self.Kp[0] = pitch.kp * 0.06
            self.Ki[0] = pitch.ki * 0.0008
            self.Kd[0] = pitch.kd * 0.3
        except Exception:
            pass

    def roll_set_pid(self, roll: PIDTune):
        try:
            self.Kp[1] = roll.kp * 0.06
            self.Ki[1] = roll.ki * 0.0008
            self.Kd[1] = roll.kd * 0.3
        except Exception:
            pass


    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)


    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

        # start disarmed / safe
        self.cmd.rc_throttle = 1000   # lowest safe throttle
        self.cmd.rc_aux4 = 1000       # aux for disarm
        self.command_pub.publish(self.cmd)

        # PID gating flag (only disable on initial arm, not between waypoints)
        if not hasattr(self, 'initial_arm_done'):
            self.pid_enabled = False
            self.initial_arm_done = True
            self.get_logger().info('Initial arm: PID disabled')
        else:
            self.get_logger().info('Re-arm: PID state preserved')


    # ------------------------------
    # CALLBACKS
    # ------------------------------

    def whycon_callback(self, msg: PoseArray):
        if not msg.poses:
            return
        pose = msg.poses[0].position
        self.curr_state = [pose.x, pose.y, pose.z]
        self.drone_position = self.curr_state.copy()
        
        # Initialize desired_state to current position on first position reading
        if not hasattr(self, '_position_initialized'):
            self.desired_state = self.curr_state.copy()
            self._position_initialized = True
            self.get_logger().info(f'Initialized desired_state to current position: [{self.desired_state[0]:.2f}, {self.desired_state[1]:.2f}, {self.desired_state[2]:.2f}]')

    def odometry_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = math.degrees(yaw)
        self.roll = math.degrees(roll)
        self.pitch = math.degrees(pitch)
        
        # Safety check: warn if tilt is too extreme
        if abs(self.roll) > 30 or abs(self.pitch) > 30:
            self.get_logger().warn(f'Excessive tilt! Roll={self.roll:.1f}°, Pitch={self.pitch:.1f}°')

    async def execute_callback(self, goal_handle):
        """
        Called when a new goal arrives. Waits until drone is inside radius for >= 3s,
        publishing feedback while doing so. Returns NavToWaypoint.Result with hov_time.
        """

        self.get_logger().info('Executing goal...')
        # goal = goal_handle.goal  # geometry_msgs/Pose - the goal pose
        goal = getattr(goal_handle, "goal", None)

        if goal is None:
            goal = getattr(goal_handle, "request", None)
        if goal is None:
            self.get_logger().error("Goal handle has neither .goal nor .request!")
            goal_handle.abort()
            return NavToWaypoint.Result()
    
        # Store old waypoint for debugging
        old_waypoint = self.desired_state.copy()
        
        self.desired_state = [
            goal.target.point.x,
            goal.target.point.y,
            goal.target.point.z,
        ]
        
        # Calculate waypoint change distance
        dx = self.desired_state[0] - old_waypoint[0]
        dy = self.desired_state[1] - old_waypoint[1]
        dz = self.desired_state[2] - old_waypoint[2]
        distance = (dx**2 + dy**2 + dz**2)**0.5
        
        self.get_logger().info(f'Waypoint change: {distance:.2f}m from [{old_waypoint[0]:.2f}, {old_waypoint[1]:.2f}, {old_waypoint[2]:.2f}]')
        
        # Smart integral reset: Only reset pitch/roll, preserve throttle for altitude
        # Also clamp throttle integral if it's excessive to prevent windup issues
        self.sum_error[0] = 0.0  # Reset pitch integral
        self.sum_error[1] = 0.0  # Reset roll integral  
        
        # Clamp throttle integral if excessive (but don't reset to 0)
        if abs(self.sum_error[2]) > 100:  # If throttle integral is unreasonably large
            self.sum_error[2] = 100 if self.sum_error[2] > 0 else -100
            self.get_logger().warn(f'Throttle integral clamped to prevent windup')
        
        self.get_logger().info(f'Pitch/Roll integrals reset, throttle integral: {self.sum_error[2]:.2f}')
        
        # Debug: Print the new waypoint
        self.get_logger().info(f'NEW WAYPOINT SET: [{goal.target.point.x:.2f}, {goal.target.point.y:.2f}, {goal.target.point.z:.2f}]')

        feedback_msg = NavToWaypoint.Feedback()
        result = NavToWaypoint.Result()

        # Publish the target point for PID controller
        target_msg = PointStamped()

        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = 'map'
        target_msg.point.x = goal.target.point.x
        target_msg.point.y = goal.target.point.y
        target_msg.point.z = goal.target.point.z

        # before publishing target or starting to stabilize
        # Optionally arm here (if you require arming): self.arm()
        
        # Initialize desired_state to current position if this is first waypoint
        if not getattr(self, 'pid_enabled', False):
            self.get_logger().info(f'First waypoint - initializing desired_state to current position: [{self.curr_state[0]:.2f}, {self.curr_state[1]:.2f}, {self.curr_state[2]:.2f}]')
        
        self.pid_enabled = True
        self.waypoint_mode_active = True  # Flag to keep PID always enabled once waypoints start
        self.get_logger().info(f'PID ENABLED - Controller should now track waypoint')
        self.target_pub.publish(target_msg)

        # reset stabalization bookkeeping
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.time_inside_sphere = 0

        # small radius (whycon units) — adjust if needed
        radius = 0.90  # 90 cm radius sphere (loose tolerance)

        # main loop: send feedback frequently, check stabilization condition
        while True:
            feedback_msg.current_position.point.x = self.curr_state[0]
            feedback_msg.current_position.point.y = self.curr_state[1]
            feedback_msg.current_position.point.z = self.curr_state[2]

            # set header stamp properly (Time msg)
            # feedback_msg.current_position.header.stamp.sec = int(self.max_time_inside_sphere)

            feedback_msg.current_position.header.stamp = self.get_clock().now().to_msg()
            feedback_msg.current_position.header.frame_id = 'map'

            # send feedback
            goal_handle.publish_feedback(feedback_msg)

            # check if drone is within sphere around desired waypoint
            drone_in_sphere = self.is_drone_in_sphere(self.drone_position, goal, radius) # radius can be adjusted

            # Debug: Log sphere status every few seconds
            if not hasattr(self, 'sphere_debug_counter'):
                self.sphere_debug_counter = 0
            self.sphere_debug_counter += 1
            if self.sphere_debug_counter % 50 == 0:  # Every 5 seconds
                distance = ((self.drone_position[0] - goal.target.point.x)**2 + 
                           (self.drone_position[1] - goal.target.point.y)**2 + 
                           (self.drone_position[2] - goal.target.point.z)**2)**0.5
                self.get_logger().info(f'Waypoint Status: Distance={distance:.2f}m, InSphere={drone_in_sphere}, Radius={radius:.2f}m, Time={self.max_time_inside_sphere:.1f}s')

            now_sec = self.get_clock().now().seconds_nanoseconds()[0]

            if drone_in_sphere:
                # start timer when first entering sphere
                if self.point_in_sphere_start_time is None:
                    self.point_in_sphere_start_time = now_sec
                # compute how long we've been inside sphere                    
                self.time_inside_sphere = now_sec - self.point_in_sphere_start_time
                # update max_time (we only grow while continuously inside)
                self.max_time_inside_sphere = self.time_inside_sphere
            else:
                # left sphere: reset count and start-time
                self.point_in_sphere_start_time = None
                self.time_inside_sphere = 0.0
                self.max_time_inside_sphere = 0.0

            # success condition: inside sphere continuously for >= 2 seconds
            if self.max_time_inside_sphere >= 2:
                break

            # non-blocking wait for a short interval
            # await asyncio.sleep(0.1)                        # Instead of that we could have used rclpy.sleep but asyncio is non-blocking.
            # And we did use it.
            
            rclpy.spin_once(self, timeout_sec=0.1)

        # goal succeeded -> fill result and finish  
        # Keep PID enabled for next waypoint (don't disable it)
        goal_handle.succeed()
        result.hov_time = float(self.max_time_inside_sphere)
        self.get_logger().info(f'Waypoint reached. Stabilization time: {result.hov_time:.3f} s')
        
        # Brief pause to let system stabilize before accepting next waypoint
        import time
        time.sleep(0.5)  # 500ms stabilization pause
        
        self.get_logger().info('Ready for next waypoint (PID remains enabled)')
        return result

    def is_drone_in_sphere(self, drone_pos, goal_msg, radius):
        """
        drone_pos: [x,y,z]
        goal_request: the action request object (NavToWaypoint.Goal request),
                      i.e. goal_request.waypoint.position.x/y/z
        radius: float (same units as poses)
        """

        try:
            gx = goal_msg.target.point.x
            gy = goal_msg.target.point.y
            gz = goal_msg.target.point.z
        except Exception:
            # If goal_msg is not what we expect, log and return False (safe fallback)
            self.get_logger().warn('is_drone_in_sphere called with invalid goal_msg')
            return False


        # goal = goal_msg
        # gx = goal.target.point.x
        # gy = goal.target.point.y
        # gz = goal.target.point.z

        dx = drone_pos[0] - gx
        dy = drone_pos[1] - gy
        dz = drone_pos[2] - gz
        dist2 = dx * dx + dy * dy + dz * dz
        dist = dist2 ** 0.5     ## under root

        return (
            dist
        ) <= radius
# -------------------------------
### MAIN Function
# -------------------------------
def main(args=None):
    rclpy.init(args=args)
    waypoint_server = WayPointServer()
    executor = MultiThreadedExecutor()
    executor.add_node(waypoint_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        waypoint_server.get_logger().info('Shutting down...')
    finally:
        waypoint_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()