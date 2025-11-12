#!/usr/bin/env python3

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node


#import the action and service
from waypoint_navigation.action import NavToWaypoint # this is the action
from waypoint_navigation.srv import GetWaypoints # this is the service


class WayPointClient(Node):

    def __init__(self):
        super().__init__('waypoint_client')
        self.goals = []
        self.goal_index = 0
        #create an action client for the action 'NavToWaypoint'. Refer to Writing an action server and client (Python) in ROS 2 tutorials
        #action name should be 'waypoint_navigation'.
        self.action_client = ActionClient(self, NavToWaypoint, 'waypoint_navigation')
        
        #create a client for the service 'GetWaypoints'. Refer to Writing a simple service and client (Python) in ROS 2 tutorials
        #service name should be 'waypoints'
        self.srv = self.create_client(GetWaypoints, 'waypoints')
        while not self.srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        #create a request object for GetWaypoints service.
        self.req = GetWaypoints.Request()
        self._feedback_count = 0

    ### service client functions

    def send_request(self):
        """Send request to the waypoint service"""
        #  complete send_request method, which will send the request and return a future
        self.get_logger().info('Requesting waypoints from service...')
        return self.srv.call_async(self.req)
    
    def receive_goals(self):
        """Receive waypoints and start sending goals one by one"""
        future = self.send_request()
        #write a statement to execute the service until the future is complete
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None:
            self.get_logger().warn('Waypoint service returned no response')
            return

        self.get_logger().info('Waypoints received by the action client')

        for point in response.waypoints:
            waypoints = [point.x, point.y, point.z]
            self.goals.append(waypoints)
            self.get_logger().info(f'Waypoints: {waypoints}')

        if self.goals:
            self.send_goal(self.goals[0])
        else:
            self.get_logger().warn('No waypoints received to navigate.')    
    
    ### action client functions

    def send_goal(self, waypoint):

        #create a NavToWaypoint goal object.
        goal_msg = NavToWaypoint.Goal()

        goal_msg.target.point.x = waypoint[0]
        goal_msg.target.point.y = waypoint[1]
        goal_msg.target.point.z = waypoint[2]

        #create a method waits for the action server to be available.
        self.action_client.wait_for_server()

        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)    
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):

        #complete the goal_response_callback. Refer to Writing an action server and client (Python) in ROS 2 tutorials
        """Handle server response after goal is sent"""

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Called when the result from the server is ready"""

        result = future.result().result
        self.get_logger().info(f'✅ Waypoint {self.goal_index + 1} reached!')
        self.get_logger().info(f'🕒 Stabilization time: {result.hov_time:.2f} seconds')

        self.goal_index += 1

        if self.goal_index < len(self.goals):
            # schedule next goal after 1 second without blocking the executor
            def send_next():
                self.send_goal(self.goals[self.goal_index])
                timer.cancel()  # stop the one-shot timer
            timer = self.create_timer(1.0, send_next)
        else:
            self.get_logger().info('All waypoints have been reached successfully ✅')      

    def feedback_callback(self, feedback_msg):

        """Receive feedback during navigation"""
        self._feedback_count += 1
        if self._feedback_count % 10 != 0:  # only log every 10 feedbacks
            return
        feedback = feedback_msg.feedback
        x = feedback.current_position.point.x
        y = feedback.current_position.point.y
        z = feedback.current_position.point.z
        t = feedback.current_position.header.stamp.sec
        self.get_logger().info(f'Current position: {x:.3f}, {y:.3f}, {z:.3f}  (stamp sec={t})')


### main function
def main(args=None):
    rclpy.init(args=args)

    waypoint_client = WayPointClient()
    waypoint_client.receive_goals()

    try:
        rclpy.spin(waypoint_client)
    except KeyboardInterrupt:
        waypoint_client.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoint_client.destroy_node()
        rclpy.shutdown()
    
    # rclpy.shutdown()


if __name__ == '__main__':
    main()


