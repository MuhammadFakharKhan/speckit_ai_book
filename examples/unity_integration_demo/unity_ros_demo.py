#!/usr/bin/env python3

"""
Unity-ROS Integration Demo

This script demonstrates how to set up a simple ROS node that can communicate
with Unity through rosbridge. It publishes joint states and subscribes to
control commands.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math
import time


class UnityROSDemo(Node):
    """
    A ROS 2 node that demonstrates Unity-ROS integration.
    """

    def __init__(self):
        super().__init__('unity_ros_demo')

        # Create publishers
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/joint_trajectory', 10)

        # Create subscribers
        self.command_sub = self.create_subscription(
            String,
            '/unity_commands',
            self.command_callback,
            10
        )

        # Timer for publishing joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)  # 10 Hz

        # Robot joint names (matching the Unity model)
        self.joint_names = [
            'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
            'left_hip', 'left_knee', 'right_hip', 'right_knee'
        ]

        # Current joint positions (initialize to zero)
        self.joint_positions = [0.0] * len(self.joint_names)

        # Demo state
        self.demo_state = 0
        self.demo_timer = self.create_timer(5.0, self.demo_step)  # Change demo every 5 seconds

        self.get_logger().info('Unity-ROS Demo node initialized')

    def publish_joint_states(self):
        """Publish joint states for Unity visualization."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = [0.0] * len(self.joint_names)  # Zero velocity for now
        msg.effort = [0.0] * len(self.joint_names)    # Zero effort for now

        self.joint_state_pub.publish(msg)

    def command_callback(self, msg):
        """Handle commands received from Unity."""
        command = msg.data.lower()
        self.get_logger().info(f'Received command from Unity: {command}')

        if command == 'wave_left_arm':
            self.wave_arm('left')
        elif command == 'wave_right_arm':
            self.wave_arm('right')
        elif command == 'stand_up':
            self.stand_up()
        elif command == 'sit_down':
            self.sit_down()
        elif command.startswith('move_joint:'):
            # Parse command like "move_joint:left_shoulder:0.5"
            try:
                parts = command.split(':')
                if len(parts) == 3:
                    joint_name = parts[1]
                    position = float(parts[2])
                    self.move_single_joint(joint_name, position)
            except ValueError:
                self.get_logger().error(f'Invalid command format: {command}')

    def wave_arm(self, arm='left'):
        """Make the specified arm wave."""
        if arm == 'left':
            joint_idx = self.joint_names.index('left_shoulder')
            self.joint_positions[joint_idx] = 0.5  # Raise arm
            self.get_logger().info('Left arm raised')
        elif arm == 'right':
            joint_idx = self.joint_names.index('right_shoulder')
            self.joint_positions[joint_idx] = -0.5  # Raise arm
            self.get_logger().info('Right arm raised')

        # Reset after a moment
        self.create_timer(1.0, self.reset_arm_position)

    def reset_arm_position(self):
        """Reset arm positions."""
        left_idx = self.joint_names.index('left_shoulder')
        right_idx = self.joint_names.index('right_shoulder')
        self.joint_positions[left_idx] = 0.0
        self.joint_positions[right_idx] = 0.0
        self.get_logger().info('Arms reset to neutral position')

    def stand_up(self):
        """Move to standing position."""
        hip_joints = ['left_hip', 'right_hip']
        knee_joints = ['left_knee', 'right_knee']

        for joint in hip_joints:
            if joint in self.joint_names:
                idx = self.joint_names.index(joint)
                self.joint_positions[idx] = 0.1  # Slightly bent

        for joint in knee_joints:
            if joint in self.joint_names:
                idx = self.joint_names.index(joint)
                self.joint_positions[idx] = -0.2  # Slightly bent

        self.get_logger().info('Robot standing up')

    def sit_down(self):
        """Move to sitting position."""
        hip_joints = ['left_hip', 'right_hip']
        knee_joints = ['left_knee', 'right_knee']

        for joint in hip_joints:
            if joint in self.joint_names:
                idx = self.joint_names.index(joint)
                self.joint_positions[idx] = 0.8  # Bent more

        for joint in knee_joints:
            if joint in self.joint_names:
                idx = self.joint_names.index(joint)
                self.joint_positions[idx] = -1.2  # Bent more

        self.get_logger().info('Robot sitting down')

    def move_single_joint(self, joint_name, position):
        """Move a single joint to the specified position."""
        if joint_name in self.joint_names:
            idx = self.joint_names.index(joint_name)
            self.joint_positions[idx] = position
            self.get_logger().info(f'Moved {joint_name} to {position} radians')
        else:
            self.get_logger().error(f'Joint {joint_name} not found')

    def demo_step(self):
        """Execute demo steps."""
        if self.demo_state == 0:
            self.get_logger().info('Demo: Moving to neutral position')
            self.joint_positions = [0.0] * len(self.joint_names)
        elif self.demo_state == 1:
            self.get_logger().info('Demo: Waving left arm')
            self.wave_arm('left')
        elif self.demo_state == 2:
            self.get_logger().info('Demo: Waving right arm')
            self.wave_arm('right')
        elif self.demo_state == 3:
            self.get_logger().info('Demo: Standing up')
            self.stand_up()
        elif self.demo_state == 4:
            self.get_logger().info('Demo: Sitting down')
            self.sit_down()

        self.demo_state = (self.demo_state + 1) % 5

    def send_trajectory_command(self, joint_names, positions, duration=2.0):
        """Send a trajectory command to move joints."""
        msg = JointTrajectory()
        msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        msg.points.append(point)
        msg.header.stamp = self.get_clock().now().to_msg()

        self.trajectory_pub.publish(msg)
        self.get_logger().info(f'Sent trajectory command for joints: {joint_names}')


def main(args=None):
    """Main function to run the Unity-ROS integration demo."""
    rclpy.init(args=args)

    unity_ros_demo = UnityROSDemo()

    try:
        rclpy.spin(unity_ros_demo)
    except KeyboardInterrupt:
        unity_ros_demo.get_logger().info('Shutting down Unity-ROS Demo')
    finally:
        unity_ros_demo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()