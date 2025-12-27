#!/usr/bin/env python3
"""
Simple subscriber node for joint commands in ROS 2
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class JointCommandSubscriber(Node):
    def __init__(self):
        super().__init__('joint_command_subscriber')
        self.subscription = self.create_subscription(
            String,
            'joint_commands',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    joint_command_subscriber = JointCommandSubscriber()
    rclpy.spin(joint_command_subscriber)
    joint_command_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()