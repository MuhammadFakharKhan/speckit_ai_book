#!/usr/bin/env python3

"""
Joint Control Interface for Robot in Gazebo Simulation

This module provides an interface to control robot joints in Gazebo simulation
using ROS 2 topics and services.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time
import math


class JointController(Node):
    """
    A ROS 2 node for controlling robot joints in Gazebo simulation.
    """

    def __init__(self):
        super().__init__('joint_controller')

        # Publisher for joint commands
        self.joint_cmd_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_group_position_controller/commands',
            10
        )

        # Publisher for trajectory commands
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Store current joint states
        self.current_joint_states = JointState()

        # Timer for periodic updates
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('Joint Controller node initialized')

    def joint_state_callback(self, msg):
        """Callback function to receive joint state updates."""
        self.current_joint_states = msg
        self.get_logger().debug(f'Received joint states: {msg.name}')

    def timer_callback(self):
        """Timer callback for periodic operations."""
        pass

    def move_joints_to_position(self, joint_names, positions, duration=1.0):
        """
        Move specified joints to target positions.

        Args:
            joint_names (list): List of joint names to control
            positions (list): List of target positions (in radians)
            duration (float): Time to reach target position (in seconds)
        """
        if len(joint_names) != len(positions):
            self.get_logger().error('Joint names and positions must have the same length')
            return False

        # Create trajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        trajectory_msg.points.append(point)

        # Publish trajectory
        self.trajectory_publisher.publish(trajectory_msg)
        self.get_logger().info(f'Moving joints {joint_names} to positions {positions}')

        return True

    def move_arm_to_position(self, left_shoulder_pos=0.0, left_elbow_pos=0.0,
                            right_shoulder_pos=0.0, right_elbow_pos=0.0):
        """
        Move robot arms to specific positions.

        Args:
            left_shoulder_pos (float): Left shoulder joint position (radians)
            left_elbow_pos (float): Left elbow joint position (radians)
            right_shoulder_pos (float): Right shoulder joint position (radians)
            right_elbow_pos (float): Right elbow joint position (radians)
        """
        joint_names = ['left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow']
        positions = [left_shoulder_pos, left_elbow_pos, right_shoulder_pos, right_elbow_pos]

        return self.move_joints_to_position(joint_names, positions)

    def move_leg_to_position(self, left_hip_pos=0.0, left_knee_pos=0.0,
                            right_hip_pos=0.0, right_knee_pos=0.0):
        """
        Move robot legs to specific positions.

        Args:
            left_hip_pos (float): Left hip joint position (radians)
            left_knee_pos (float): Left knee joint position (radians)
            right_hip_pos (float): Right hip joint position (radians)
            right_knee_pos (float): Right knee joint position (radians)
        """
        joint_names = ['left_hip', 'left_knee', 'right_hip', 'right_knee']
        positions = [left_hip_pos, left_knee_pos, right_hip_pos, right_knee_pos]

        return self.move_joints_to_position(joint_names, positions)

    def wave_arm(self, arm='left', cycles=3):
        """
        Make the robot wave its arm.

        Args:
            arm (str): 'left' or 'right' arm to wave
            cycles (int): Number of waving cycles
        """
        if arm == 'left':
            joint_names = ['left_shoulder', 'left_elbow']
        else:
            joint_names = ['right_shoulder', 'right_elbow']

        self.get_logger().info(f'Making {arm} arm wave for {cycles} cycles')

        for i in range(cycles):
            # Wave up
            positions_up = [0.5, -0.5] if arm == 'left' else [-0.5, -0.5]
            self.move_joints_to_position(joint_names, positions_up, 1.0)
            time.sleep(1.0)

            # Wave down
            positions_down = [0.0, 0.0]
            self.move_joints_to_position(joint_names, positions_down, 1.0)
            time.sleep(1.0)

    def get_current_joint_positions(self):
        """Get current joint positions."""
        return dict(zip(self.current_joint_states.name, self.current_joint_states.position))


def main(args=None):
    """Main function to run the joint controller node."""
    rclpy.init(args=args)

    joint_controller = JointController()

    # Example: Move arms to neutral position
    joint_controller.move_arm_to_position(0.0, 0.0, 0.0, 0.0)

    # Example: Wave left arm
    # joint_controller.wave_arm('left', 2)

    try:
        rclpy.spin(joint_controller)
    except KeyboardInterrupt:
        joint_controller.get_logger().info('Shutting down Joint Controller')
    finally:
        joint_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()