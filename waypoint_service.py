#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point

# Import the generated service type (package name must match)
# from your_package_name.srv import file_name (which has actual .srv extension)
from waypoint_navigation.srv import GetWaypoints

class WayPoints(Node):

    def __init__(self):
        super().__init__('waypoints_service')
        # created_object_name = function(your .srv file, service_name, callback_function)
        # service_name - what clients will use to call this service         # callback_function - function to process the request and send response
        self.srv = self.create_service(GetWaypoints, 'waypoints', self.waypoint_callback)
        self.get_logger().info('Waypoint Service ready. Call "waypoints" to receive the list.')
        
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
        # If you want only first 5 for the task, slice here:
        # self.waypoints = self.waypoints[:5]

    
    def waypoint_callback(self, request, response):

        if not self.waypoints:
            self.get_logger().warn("Waypoint list is empty.")
            return response  # return empty response immediately
        
        response.waypoints = [Point(x=wp[0], y=wp[1], z=wp[2]) for wp in self.waypoints]
        for i in range(len(self.waypoints)):
            response.waypoints[i].x = self.waypoints[i][0]
            response.waypoints[i].y = self.waypoints[i][1]
            response.waypoints[i].z = self.waypoints[i][2]

        self.get_logger().info("Incoming request for Waypoints")
        return response

def main(args=None):
    rclpy.init(args=args)
    node_waypoints = WayPoints()

    try:
        rclpy.spin(node_waypoints)
    except KeyboardInterrupt:
        node_waypoints.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        node_waypoints.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
        

# logs by Kale

# [1]   Added values in waypoints
# [2]   Made the main function better by understanding it
# [3]   Added return response in else block
# [4]   Added 'if' in waypoint_callback to check for empty waypoints list