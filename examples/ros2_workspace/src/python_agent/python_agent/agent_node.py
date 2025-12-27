#!/usr/bin/env python3
"""
Simple AI agent node for ROS 2 humanoid robotics
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        self.publisher_ = self.create_publisher(String, 'agent_commands', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Agent command: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Agent publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    agent_node = AgentNode()
    rclpy.spin(agent_node)
    agent_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()