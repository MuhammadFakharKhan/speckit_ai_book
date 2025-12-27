#!/usr/bin/env python3

"""
Basic Physics Simulation Example for Gazebo

This script demonstrates basic physics simulation concepts with a humanoid robot in Gazebo.
It includes examples of joint control, physics interactions, and basic movement patterns.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import time
import math


class BasicPhysicsExample(Node):
    """
    Example node demonstrating basic physics simulation with humanoid robot.
    """

    def __init__(self):
        super().__init__('basic_physics_example')

        # Publisher for joint trajectory commands
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Store joint states
        self.joint_states = JointState()

        # Timer for demonstration
        self.timer = self.create_timer(0.1, self.demo_callback)
        self.demo_step = 0

        self.get_logger().info('Basic Physics Example node initialized')

    def joint_state_callback(self, msg):
        """Callback to receive joint states."""
        self.joint_states = msg

    def demo_callback(self):
        """Demo callback to run physics examples."""
        if self.demo_step == 0:
            self.get_logger().info('Starting basic physics simulation demo...')
            self.demo_step = 1
        elif self.demo_step == 1:
            self.move_to_initial_position()
            self.demo_step = 2
        elif self.demo_step == 2:
            self.stand_up_demo()
            self.demo_step = 3
        elif self.demo_step == 3:
            self.wave_arms_demo()
            self.demo_step = 4
        elif self.demo_step == 4:
            self.simple_walk_demo()
            self.demo_step = 5
        elif self.demo_step == 5:
            self.get_logger().info('Demo completed. Robot in final position.')
            self.timer.cancel()

    def move_to_initial_position(self):
        """Move robot to initial standing position."""
        joint_names = [
            'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
            'left_hip', 'left_knee', 'right_hip', 'right_knee'
        ]

        positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.publish_joint_trajectory(joint_names, positions, duration=2.0)
        self.get_logger().info('Moving to initial position...')

    def stand_up_demo(self):
        """Demonstrate standing up motion."""
        joint_names = ['left_hip', 'left_knee', 'right_hip', 'right_knee']

        # Stand up: slightly bend knees to make it look more natural
        positions = [0.1, -0.2, 0.1, -0.2]

        self.publish_joint_trajectory(joint_names, positions, duration=1.5)
        self.get_logger().info('Standing up demo...')

    def wave_arms_demo(self):
        """Demonstrate waving arms."""
        # Wave left arm
        joint_names = ['left_shoulder', 'left_elbow']
        positions = [0.5, -0.5]  # Raise left arm
        self.publish_joint_trajectory(joint_names, positions, duration=1.0)
        time.sleep(1.2)

        positions = [0.0, 0.0]  # Lower left arm
        self.publish_joint_trajectory(joint_names, positions, duration=1.0)
        time.sleep(0.5)

        # Wave right arm
        joint_names = ['right_shoulder', 'right_elbow']
        positions = [-0.5, -0.5]  # Raise right arm
        self.publish_joint_trajectory(joint_names, positions, duration=1.0)
        time.sleep(1.2)

        positions = [0.0, 0.0]  # Lower right arm
        self.publish_joint_trajectory(joint_names, positions, duration=1.0)

        self.get_logger().info('Waving arms demo...')

    def simple_walk_demo(self):
        """Demonstrate simple walking motion."""
        # Alternate hip and knee movements to simulate walking
        left_leg_joints = ['left_hip', 'left_knee']
        right_leg_joints = ['right_hip', 'right_knee']

        # Step 1: Lift left leg
        self.publish_joint_trajectory(left_leg_joints, [0.3, -0.6], duration=0.8)
        time.sleep(1.0)

        # Step 2: Lower left leg and lift right leg
        self.publish_joint_trajectory(left_leg_joints, [0.0, -0.2], duration=0.8)
        self.publish_joint_trajectory(right_leg_joints, [0.3, -0.6], duration=0.8)
        time.sleep(1.0)

        # Step 3: Lower right leg
        self.publish_joint_trajectory(right_leg_joints, [0.0, -0.2], duration=0.8)

        self.get_logger().info('Simple walking demo...')

    def publish_joint_trajectory(self, joint_names, positions, duration=1.0):
        """
        Publish joint trajectory message to control robot.

        Args:
            joint_names (list): List of joint names to control
            positions (list): List of target positions
            duration (float): Time to reach target position
        """
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions

        # Set the time to reach the position
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        trajectory_msg.points.append(point)
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.header.frame_id = 'base_link'

        self.joint_trajectory_pub.publish(trajectory_msg)


def main(args=None):
    """Main function to run the basic physics example."""
    rclpy.init(args=args)

    basic_physics_example = BasicPhysicsExample()

    try:
        rclpy.spin(basic_physics_example)
    except KeyboardInterrupt:
        basic_physics_example.get_logger().info('Shutting down Basic Physics Example')
    finally:
        basic_physics_example.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()