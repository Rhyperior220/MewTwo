#!/usr/bin/env python3

'''
Modified pico_controller with:
 - 3-sample Median filter + 2nd-order Butterworth low-pass filter + 5-sample Moving Average on altitude
 - Filtered derivative on throttle
 - Improved anti-windup and integrator clamping
 - Throttle trim variable
Keep Gazebo sign convention (no negation of throttle PID output).
'''

from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error
import rclpy
from rclpy.node import Node
import math
from collections import deque


class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')

        self.m = 0.152
        self.g = 9.81
        self.sample_time = 0.05  # seconds (20 Hz)

        # Drone states
        self.current_state = [0.0, 0.0, 0.0]
        self.desired_state = [-7.0, 0.0, 29.22]
        

        # Median filter buffer
        self.altitude_window = deque(maxlen=3)

        # Moving average buffer (for final smoothing)
        self.avg_window = deque(maxlen=5)

        # Initialize SwiftMsgs
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 150

        # PID gains (adjust these for tuning)
        self.Kp = [0.0, 0.0, 700 * 0.06]
        self.Ki = [0.0, 0.0, 2000 * 0.0008]
        self.Kd = [0.0, 0.0, 390 * 0.3]

        # PID variables
        self.prev_error = [0.0, 0.0, 0.0]
        self.sum_error = [0.0, 0.0, 0.0]
        self.pos_error = Error()

        # Command & integrator limits
        self.max_values = [2000, 2000, 2000]
        self.min_values = [1000, 1000, 1000]
        self.error_sum_limits = [500, 500, 300]
        self.throttle_trim = 1458

        # Filter design parameters
        self.cutoff_freq = 3.5
        self.filter_order = 2
        self.b_biquad = [0.0, 0.0, 0.0]
        self.a_biquad = [1.0, 0.0, 0.0]
        self.z_x1 = self.z_x2 = self.z_y1 = self.z_y2 = 0.0

        # Derivative filter
        self.d_filter_alpha = 0.75
        self.prev_d_throttle = 0.0

        # Anti-windup config
        self.enable_integrator_when_saturated = False

        # Design Butterworth filter
        self._design_butterworth(self.cutoff_freq, self.sample_time, self.filter_order)

        # ROS publishers/subscribers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, "/pitch_pid", self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, "/roll_pid", self.roll_set_pid, 1)

        self.arm()
        self.pid_timer = self.create_timer(self.sample_time, self.pid)

    # ---------------------------- Filters ----------------------------

    # 3-sample median filter
    def _median_filter(self, new_value):
        self.altitude_window.append(new_value)
        if len(self.altitude_window) < 3:
            return new_value
        sorted_vals = sorted(self.altitude_window)
        return sorted_vals[1]

    # Butterworth filter design
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

    # Apply Butterworth filter
    def _butterworth_filter(self, x):
        b0, b1, b2 = self.b_biquad
        a0, a1, a2 = self.a_biquad
        y = b0 * x + b1 * self.z_x1 + b2 * self.z_x2 - a1 * self.z_y1 - a2 * self.z_y2
        self.z_x2, self.z_x1 = self.z_x1, x
        self.z_y2, self.z_y1 = self.z_y1, y
        return y

    # Moving Average filter
    def _moving_average(self, new_value):
        self.avg_window.append(new_value)
        return sum(self.avg_window) / len(self.avg_window)

    # ---------------------------- Control ----------------------------

    def disarm(self):
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)
        self.get_logger().info('Disarmed')

    def arm(self):
        self.sum_error = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]
        self.prev_d_throttle = 0.0
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 100
        self.cmd.rc_aux4 = 1500
        self.command_pub.publish(self.cmd)
        self.get_logger().info('Armed')
        rclpy.spin_once(self, timeout_sec=1.0)

    def whycon_callback(self, msg):
        if not msg.poses:
            return
        self.current_state[0] = msg.poses[0].position.x
        self.current_state[1] = msg.poses[0].position.y
        self.current_state[2] = msg.poses[0].position.z

    def altitude_set_pid(self, alt):
        self.Kp[2] = alt.kp * 0.06
        self.Ki[2] = alt.ki * 0.0008
        self.Kd[2] = alt.kd * 0.3

    def pitch_set_pid(self, pitch):
        self.Kp[0] = pitch.kp * 0.06
        self.Ki[0] = pitch.ki * 0.0008
        self.Kd[0] = pitch.kd * 0.3

    def roll_set_pid(self, roll):
        self.Kp[1] = roll.kp * 0.06
        self.Ki[1] = roll.ki * 0.0008
        self.Kd[1] = roll.kd * 0.3

    # ---------------------------- PID Loop ----------------------------

    def pid(self):
        # Filter chain: Median → Butterworth → Moving Average
        median_z = self._median_filter(self.current_state[2])
        butter_z = self._butterworth_filter(median_z)
        filtered_z = self._moving_average(butter_z)

        # Errors
        error_x = self.current_state[0] - self.desired_state[0]
        error_y = self.current_state[1] - self.desired_state[1]
        error_z = filtered_z - self.desired_state[2]
        current_errors = [error_x, error_y, error_z]

        # Pitch PID
        p_term_pitch = self.Kp[0] * error_x
        d_term_pitch = self.Kd[0] * (error_x - self.prev_error[0]) / self.sample_time
        self.sum_error[0] = max(min(self.sum_error[0] + error_x * self.sample_time, self.error_sum_limits[0]), -self.error_sum_limits[0])
        i_term_pitch = self.Ki[0] * self.sum_error[0]

        # Roll PID
        p_term_roll = self.Kp[1] * error_y
        d_term_roll = self.Kd[1] * (error_y - self.prev_error[1]) / self.sample_time
        self.sum_error[1] = max(min(self.sum_error[1] + error_y * self.sample_time, self.error_sum_limits[1]), -self.error_sum_limits[1])
        i_term_roll = self.Ki[1] * self.sum_error[1]

        # Throttle PID
        p_term_throttle = self.Kp[2] * error_z
        raw_d_throttle = self.Kd[2] * (error_z - self.prev_error[2]) / self.sample_time
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

        self.cmd.rc_pitch = max(min(1500 + int(out_pitch), self.max_values[1]), self.min_values[1])
        self.cmd.rc_roll = max(min(1500 - int(out_roll), self.max_values[0]), self.min_values[0])
        self.cmd.rc_throttle = max(min(int(self.throttle_trim + out_throttle), self.max_values[2]), self.min_values[2])

        self.prev_error = current_errors

        # Publish commands and errors
        self.command_pub.publish(self.cmd)
        self.pos_error.pitch_error = float(error_x)
        self.pos_error.roll_error = float(error_y)
        self.pos_error.throttle_error = float(error_z)
        self.pos_error_pub.publish(self.pos_error)


def main(args=None):
    rclpy.init(args=args)
    swift_pico = Swift_Pico()
    try:
        rclpy.spin(swift_pico)
    except KeyboardInterrupt:
        swift_pico.get_logger().info('KeyboardInterrupt, disarming and shutting down.\n')
        swift_pico.disarm()
    finally:
        swift_pico.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# ive applied median and average filter along with butterworth