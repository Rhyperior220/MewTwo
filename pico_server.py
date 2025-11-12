#!/usr/bin/env python3

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
        


        self.sample_time = 0.01666 #put the appropriate value according to your controller

        # --- PID / filter related defaults (merged from PID_new_Dipankar.py)
        self.m = 0.152
        self.g = 9.81

        # Filter buffers
        self.altitude_window = deque(maxlen=3)
        self.avg_window = deque(maxlen=5)

        # PID gains (index: 0->pitch/x, 1->roll/y, 2->throttle/z)
        self.Kp = [0.0, 0.0, 725 * 0.06]
        self.Ki = [0.0, 0.0, 2320 * 0.0008]
        self.Kd = [0.0, 0.0, 388.5 * 0.3]

        # PID state
        self.prev_error = [0.0, 0.0, 0.0]
        self.sum_error = [0.0, 0.0, 0.0]
        self.pos_error = Error()

        # command & integrator limits
        self.max_values = [2000, 2000, 2000]
        self.min_values = [1000, 1000, 1000]
        self.error_sum_limits = [500, 500, 300]
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
        #define the function to be run inside the timer callback. This function will implement the PID or LQR algorithm
        self.timer = None
    # PID or LQR controller timer callback
    def pid_or_lqr_timer_callback(self):
        if not getattr(self, 'pid_enabled', False):
            # PID disabled: do nothing and return
            return

        # PID loop adapted from PID_new_Dipankar.py
        try:
            median_z = self._median_filter(self.curr_state[2])
            butter_z = self._butterworth_filter(median_z)
            filtered_z = self._moving_average(butter_z)
        except Exception:
            filtered_z = self.curr_state[2]

        # compute errors
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

        # Apply to command message and saturate
        self.cmd.rc_pitch = max(min(1500 + int(out_pitch), self.max_values[1]), self.min_values[1])
        self.cmd.rc_roll = max(min(1500 - int(out_roll), self.max_values[0]), self.min_values[0])
        self.cmd.rc_throttle = max(min(int(self.throttle_trim + out_throttle), self.max_values[2]), self.min_values[2])

        self.prev_error = current_errors

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

        # PID gating flag
        self.pid_enabled = False


    # ------------------------------
    # CALLBACKS
    # ------------------------------

    def whycon_callback(self, msg: PoseArray):
        if not msg.poses:
            return
        pose = msg.poses[0].position
        self.curr_state = [pose.x, pose.y, pose.z]
        self.drone_position = self.curr_state.copy()

    def odometry_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = math.degrees(yaw)

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
    
        self.desired_state = [
            goal.target.point.x,
            goal.target.point.y,
            goal.target.point.z,
        ]

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
        self.pid_enabled = True
        self.target_pub.publish(target_msg)

        # reset stabalization bookkeeping
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.time_inside_sphere = 0

        # small radius (whycon units) — adjust if needed
        radius = 0.08  # 8 cm radius sphere

        # main loop: send feedback frequently, check stabilization condition

        if self.timer is None:
            self.timer = self.create_timer(self.sample_time, self.pid_or_lqr_timer_callback, callback_group=self.pid_or_lqr_callback_group)
            # optionally call self.arm() here once you're ready

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

            # success condition: inside sphere continuously for >= 3 seconds
            if self.max_time_inside_sphere >= 3:
                break

            # non-blocking wait for a short interval
            # await asyncio.sleep(0.1)                        # Instead of that we could have used rclpy.sleep but asyncio is non-blocking.
            # And we did use it.
            
            rclpy.spin_once(self, timeout_sec=0.1)

        # goal succeeded -> fill result and finish
        goal_handle.succeed()
        result.hov_time = float(self.max_time_inside_sphere)
        self.get_logger().info(f'Waypoint reached. Stabilization time: {result.hov_time:.3f} s')
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

        return (
            dist2
        ) <= radius**2
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
