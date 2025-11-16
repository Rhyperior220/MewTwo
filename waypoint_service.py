#!/usr/bin/env python3
# waypoint_service.py

"""
Waypoint Navigation Service Node

Provides a ROS2 service that stores and returns waypoint coordinates.
This node does NOT execute motion - it only holds and shares the list of waypoints.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from waypoint_navigation.srv import GetWaypoints


class WaypointNavigation(Node):
    """
    ROS2 Service Node that provides waypoint storage and retrieval.
    
    Service: 'waypoints' (GetWaypoints)
    - Request: Empty (no fields required)
    - Response: geometry_msgs/Point[] waypoints
    """
    
    def __init__(self):
        super().__init__('waypoint_navigation')
        
        # Create service server
        self.srv = self.create_service(
            GetWaypoints,
            'waypoints',
            self.waypoint_callback
        )
        
        # Define waypoint list (x, y, z coordinates in meters)
        # These are the target positions for the drone to navigate to
        self.waypoints = [
            [-7.00, 0.00, 29.22],
            [-7.64, 3.06, 29.22],
            [-8.22, 6.02, 29.22],
            [-9.11, 9.27, 29.27],
            [-5.98, 8.81, 29.27],
            [-3.26, 8.41, 29.88],
            [0.87, 8.18, 29.05],
            [3.93, 7.35, 29.05]
        ]
        
        # Optional: Limit waypoints for testing
        # self.waypoints = self.waypoints[:5]
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Waypoint Navigation Service Node Started')
        self.get_logger().info(f'Service: "waypoints" (GetWaypoints)')
        self.get_logger().info(f'Stored {len(self.waypoints)} waypoints')
        self.get_logger().info('=' * 60)
        
        # Log all waypoints for debugging
        for i, wp in enumerate(self.waypoints, 1):
            self.get_logger().info(f'  Waypoint {i}: [{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}]')
    
    def waypoint_callback(self, request, response):
        """
        Service callback that returns all stored waypoints.
        
        Args:
            request: GetWaypoints.Request (empty)
            response: GetWaypoints.Response (contains waypoints array)
            
        Returns:
            GetWaypoints.Response with populated waypoints array
        """
        if not self.waypoints:
            self.get_logger().warn('Waypoint list is empty! Returning empty response.')
            response.waypoints = []
            return response
        
        # Convert waypoint list to geometry_msgs/Point array
        response.waypoints = [
            Point(x=float(wp[0]), y=float(wp[1]), z=float(wp[2]))
            for wp in self.waypoints
        ]
        
        self.get_logger().info(
            f'Service request received. Returning {len(response.waypoints)} waypoints.'
        )
        
        return response


def main(args=None):
    """Main entry point for waypoint navigation service node."""
    rclpy.init(args=args)
    
    node = WaypointNavigation()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received. Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
