#!/usr/bin/env python3
"""
Controller node for ROS 2 humanoid robotics
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        self.subscription = self.create_subscription(
            String,
            'agent_commands',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Controller received: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    controller_node = ControllerNode()
    rclpy.spin(controller_node)
    controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()