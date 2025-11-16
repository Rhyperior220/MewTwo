#!/usr/bin/env python3
# pico_client.py

"""
Waypoint Navigation Client Node

This node:
1. Requests all waypoints from the waypoint_navigation service
2. Sends each waypoint sequentially to the action server (pico_server)
3. Handles continuous execution - immediately sends next waypoint after success
4. Provides feedback logging and result handling
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints


class WaypointClient(Node):
    """
    ROS2 Action Client that orchestrates waypoint navigation.
    
    Flow:
    1. Request waypoints from service
    2. Send each waypoint as action goal
    3. Wait for completion
    4. Immediately send next waypoint (no stopping)
    """
    
    def __init__(self):
        super().__init__('waypoint_client')
        
        # Waypoint storage
        self.goals = []
        self.goal_index = 0
        self.total_waypoints = 0
        
        # Create action client for waypoint navigation
        self.action_client = ActionClient(
            self,
            NavToWaypoint,
            'waypoint_navigation'
        )
        
        # Create service client for waypoint retrieval
        self.service_client = self.create_client(GetWaypoints, 'waypoints')
        
        # State tracking
        self.current_goal_handle = None
        self.is_navigating = False
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Waypoint Client Node Initialized')
        self.get_logger().info('Waiting for waypoint service and action server...')
        self.get_logger().info('=' * 60)
        
        # Wait for service and action server
        self._wait_for_services()
        
        # Request waypoints and start navigation
        self.request_waypoints()
    
    def _wait_for_services(self):
        """Wait for both service and action server to be available."""
        # Wait for waypoint service
        self.get_logger().info('Waiting for waypoint service...')
        while not self.service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waypoint service not available, waiting...')
        
        # Wait for action server
        self.get_logger().info('Waiting for action server...')
        while not self.action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Action server not available, waiting...')
        
        self.get_logger().info('✓ All services available!')
    
    def request_waypoints(self):
        """Request waypoints from service and start navigation."""
        self.get_logger().info('Requesting waypoints from service...')
        
        request = GetWaypoints.Request()
        future = self.service_client.call_async(request)
        
        # Wait for service response
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is None:
            self.get_logger().error('Failed to get waypoints from service!')
            return
        
        response = future.result()
        
        if not response.waypoints:
            self.get_logger().warn('No waypoints received from service!')
            return
        
        # Store waypoints
        self.goals = [
            [point.x, point.y, point.z]
            for point in response.waypoints
        ]
        self.total_waypoints = len(self.goals)
        
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Received {self.total_waypoints} waypoints:')
        for i, wp in enumerate(self.goals, 1):
            self.get_logger().info(f'  {i}. [{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}]')
        self.get_logger().info('=' * 60)
        
        # Start navigation with first waypoint
        if self.goals:
            self.get_logger().info('Starting navigation sequence...')
            self.send_goal(self.goals[0])
        else:
            self.get_logger().error('No waypoints to navigate!')
    
    def send_goal(self, waypoint):
        """
        Send a waypoint goal to the action server.
        
        Args:
            waypoint: [x, y, z] list of coordinates
        """
        if self.is_navigating:
            self.get_logger().warn('Already navigating! Ignoring new goal.')
            return
        
        self.is_navigating = True
        
        # Create goal message
        goal_msg = NavToWaypoint.Goal()
        goal_msg.target.point.x = float(waypoint[0])
        goal_msg.target.point.y = float(waypoint[1])
        goal_msg.target.point.z = float(waypoint[2])
        goal_msg.target.header.frame_id = 'map'
        goal_msg.target.header.stamp = self.get_clock().now().to_msg()
        
        self.get_logger().info('-' * 60)
        self.get_logger().info(
            f'Sending waypoint {self.goal_index + 1}/{self.total_waypoints}: '
            f'[{waypoint[0]:.2f}, {waypoint[1]:.2f}, {waypoint[2]:.2f}]'
        )
        
        # Send goal asynchronously
        self.send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self.send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """
        Handle response after goal is sent to action server.
        
        Args:
            future: Future containing goal handle
        """
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server!')
            self.is_navigating = False
            return
        
        self.get_logger().info('✓ Goal accepted by action server')
        self.current_goal_handle = goal_handle
        
        # Request result asynchronously
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        """
        Handle result when waypoint is reached.
        
        Args:
            future: Future containing result
        """
        result = future.result()
        
        if result.status == 4:  # SUCCEEDED
            hov_time = result.result.hov_time
            self.get_logger().info('=' * 60)
            self.get_logger().info(
                f'✅ Waypoint {self.goal_index + 1}/{self.total_waypoints} REACHED!'
            )
            self.get_logger().info(f'   Stabilization time: {hov_time:.2f} seconds')
            self.get_logger().info('=' * 60)
            
            self.goal_index += 1
            self.is_navigating = False
            
            # Check if more waypoints remain
            if self.goal_index < len(self.goals):
                # Immediately send next waypoint (continuous navigation)
                self.get_logger().info(
                    f'Continuing to next waypoint ({self.goal_index + 1}/{self.total_waypoints})...'
                )
                # Small delay to ensure previous goal is fully processed
                # Using a timer that cancels itself after first execution (ROS2 Humble compatible)
                def send_next():
                    self._send_next_waypoint()
                    # Cancel the timer after first execution
                    if hasattr(self, '_next_waypoint_timer'):
                        self._next_waypoint_timer.cancel()
                        delattr(self, '_next_waypoint_timer')
                
                self._next_waypoint_timer = self.create_timer(0.1, send_next)
            else:
                self.get_logger().info('=' * 60)
                self.get_logger().info('🎉 ALL WAYPOINTS REACHED SUCCESSFULLY! 🎉')
                self.get_logger().info('=' * 60)
        else:
            status_names = {
                1: 'UNKNOWN',
                2: 'ACCEPTED',
                3: 'EXECUTING',
                4: 'SUCCEEDED',
                5: 'CANCELED',
                6: 'ABORTED'
            }
            status = status_names.get(result.status, f'UNKNOWN({result.status})')
            self.get_logger().error(f'Waypoint failed with status: {status}')
            self.is_navigating = False
    
    def _send_next_waypoint(self):
        """Helper to send next waypoint (called from timer)."""
        if self.goal_index < len(self.goals):
            self.send_goal(self.goals[self.goal_index])
    
    def feedback_callback(self, feedback_msg):
        """
        Handle feedback during waypoint navigation.
        
        Args:
            feedback_msg: Action feedback message
        """
        feedback = feedback_msg.feedback
        pos = feedback.current_position.point
        
        # Log feedback periodically (every 10th message to avoid spam)
        if not hasattr(self, 'feedback_counter'):
            self.feedback_counter = 0
        
        self.feedback_counter += 1
        if self.feedback_counter % 10 == 0:
            # Safety check: ensure goal_index is valid
            if 0 <= self.goal_index < len(self.goals):
                distance = (
                    (pos.x - self.goals[self.goal_index][0])**2 +
                    (pos.y - self.goals[self.goal_index][1])**2 +
                    (pos.z - self.goals[self.goal_index][2])**2
                ) ** 0.5
                
                self.get_logger().info(
                    f'Navigating to waypoint {self.goal_index + 1}: '
                    f'Current: [{pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}], '
                    f'Distance: {distance:.2f}m'
                )


def main(args=None):
    """Main entry point for waypoint client node."""
    rclpy.init(args=args)
    
    client = WaypointClient()
    
    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.get_logger().info('Keyboard interrupt received. Shutting down...')
        # Cancel current goal if navigating
        if client.is_navigating and client.current_goal_handle:
            client.get_logger().info('Canceling current goal...')
            client.current_goal_handle.cancel_goal_async()
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
