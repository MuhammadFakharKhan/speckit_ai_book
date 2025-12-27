#!/usr/bin/env python3

"""
Sensor Demo for Gazebo Simulation

This script demonstrates how to use and process sensor data from the simulated humanoid robot.
It includes examples of subscribing to sensor topics, processing the data, and visualizing it.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from std_msgs.msg import String
import numpy as np
import math

try:
    from cv_bridge import CvBridge
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("OpenCV not available, image processing will be limited")


class SensorDemo(Node):
    """
    A ROS 2 node demonstrating sensor data processing from Gazebo simulation.
    """

    def __init__(self):
        super().__init__('sensor_demo')

        # Create subscribers for sensor data
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.lidar_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publisher for processed sensor data
        self.sensor_status_pub = self.create_publisher(String, '/sensor_demo/status', 10)

        # Timer for periodic sensor analysis
        self.timer = self.create_timer(2.0, self.sensor_analysis)

        # Sensor data storage
        self.latest_camera_data = None
        self.latest_lidar_data = None
        self.latest_imu_data = None

        # Sensor status tracking
        self.sensor_status = {
            'camera': {'active': False, 'last_update': 0},
            'lidar': {'active': False, 'last_update': 0},
            'imu': {'active': False, 'last_update': 0}
        }

        self.get_logger().info('Sensor Demo node initialized')

    def camera_callback(self, msg):
        """Process camera image data."""
        self.latest_camera_data = msg
        self.sensor_status['camera']['active'] = True
        self.sensor_status['camera']['last_update'] = self.get_clock().now().nanoseconds

        self.get_logger().debug(f'Received camera image: {msg.width}x{msg.height}, encoding: {msg.encoding}')

    def lidar_callback(self, msg):
        """Process LIDAR scan data."""
        self.latest_lidar_data = msg
        self.sensor_status['lidar']['active'] = True
        self.sensor_status['lidar']['last_update'] = self.get_clock().now().nanoseconds

        # Analyze the LIDAR data
        if len(msg.ranges) > 0:
            valid_ranges = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max]
            if valid_ranges:
                avg_distance = sum(valid_ranges) / len(valid_ranges)
                min_distance = min(valid_ranges) if valid_ranges else float('inf')

                self.get_logger().debug(
                    f'LIDAR: {len(valid_ranges)}/{len(msg.ranges)} valid ranges, '
                    f'avg distance: {avg_distance:.2f}m, min distance: {min_distance:.2f}m'
                )

    def imu_callback(self, msg):
        """Process IMU data."""
        self.latest_imu_data = msg
        self.sensor_status['imu']['active'] = True
        self.sensor_status['imu']['last_update'] = self.get_clock().now().nanoseconds

        # Calculate orientation from quaternion
        quat = msg.orientation
        norm = (quat.x**2 + quat.y**2 + quat.z**2 + quat.w**2)**0.5

        # Calculate acceleration magnitude
        accel_mag = (msg.linear_acceleration.x**2 +
                    msg.linear_acceleration.y**2 +
                    msg.linear_acceleration.z**2)**0.5

        self.get_logger().debug(
            f'IMU: orientation norm={norm:.3f}, acceleration={accel_mag:.3f}m/s²'
        )

    def sensor_analysis(self):
        """Periodically analyze sensor data and report status."""
        status_msg = String()
        status_text = "Sensor Analysis Report:\n"

        # Check camera status
        if self.sensor_status['camera']['active']:
            if self.latest_camera_data:
                status_text += f"✓ Camera: Active, Last update: {self.sensor_status['camera']['last_update']}\n"
            else:
                status_text += "○ Camera: Active but no data received\n"
        else:
            status_text += "✗ Camera: Inactive\n"

        # Check LIDAR status
        if self.sensor_status['lidar']['active']:
            if self.latest_lidar_data:
                status_text += f"✓ LIDAR: Active, {len(self.latest_lidar_data.ranges)} ranges, "
                valid_ranges = [r for r in self.latest_lidar_data.ranges
                              if self.latest_lidar_data.range_min <= r <= self.latest_lidar_data.range_max]
                status_text += f"{len(valid_ranges)} valid\n"
            else:
                status_text += "○ LIDAR: Active but no data received\n"
        else:
            status_text += "✗ LIDAR: Inactive\n"

        # Check IMU status
        if self.sensor_status['imu']['active']:
            if self.latest_imu_data:
                status_text += f"✓ IMU: Active, Orientation norm: {(
                    self.latest_imu_data.orientation.x**2 +
                    self.latest_imu_data.orientation.y**2 +
                    self.latest_imu_data.orientation.z**2 +
                    self.latest_imu_data.orientation.w**2)**0.5:.3f}\n"
            else:
                status_text += "○ IMU: Active but no data received\n"
        else:
            status_text += "✗ IMU: Inactive\n"

        # Object detection analysis
        if self.latest_lidar_data:
            objects_detected = self.detect_objects()
            status_text += f"Object Detection: {objects_detected} objects detected\n"

        # Stability analysis from IMU
        if self.latest_imu_data:
            stability = self.estimate_stability()
            status_text += f"Robot Stability: {stability}\n"

        status_msg.data = status_text
        self.sensor_status_pub.publish(status_msg)
        self.get_logger().info(status_text)

    def detect_objects(self):
        """Simple object detection using LIDAR data."""
        if not self.latest_lidar_data:
            return 0

        # Simple clustering approach to detect objects
        ranges = self.latest_lidar_data.ranges
        min_range = self.latest_lidar_data.range_min
        max_range = self.latest_lidar_data.range_max

        # Filter valid ranges
        valid_ranges = [(i, r) for i, r in enumerate(ranges) if min_range <= r <= max_range]

        if not valid_ranges:
            return 0

        # Simple object clustering based on distance jumps
        object_count = 0
        if len(valid_ranges) > 1:
            # Calculate differences between consecutive readings
            distances = [r for i, r in valid_ranges]
            for i in range(1, len(distances)):
                if abs(distances[i] - distances[i-1]) > 0.5:  # Threshold for object boundary
                    object_count += 1

        return min(object_count, 10)  # Cap at 10 objects

    def estimate_stability(self):
        """Estimate robot stability based on IMU data."""
        if not self.latest_imu_data:
            return "Unknown"

        # Calculate total acceleration (excluding gravity)
        ax = self.latest_imu_data.linear_acceleration.x
        ay = self.latest_imu_data.linear_acceleration.y
        az = self.latest_imu_data.linear_acceleration.z

        # Remove gravity component (assuming z is up)
        net_accel = (ax**2 + ay**2 + (az + 9.81)**2)**0.5

        if net_accel < 0.5:
            return "Stable"
        elif net_accel < 2.0:
            return "Moderate movement"
        else:
            return "Unstable"

    def run_sensor_demo(self, duration=60):
        """
        Run the sensor demo for a specified duration.

        Args:
            duration (int): Duration to run the demo in seconds
        """
        self.get_logger().info(f'Running sensor demo for {duration} seconds...')

        # Run for the specified duration
        start_time = self.get_clock().now().nanoseconds / 1e9
        end_time = start_time + duration

        while (self.get_clock().now().nanoseconds / 1e9) < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Sensor demo completed')


def main(args=None):
    """Main function to run the sensor demo."""
    rclpy.init(args=args)

    sensor_demo = SensorDemo()

    try:
        # Run the sensor demo for 60 seconds
        sensor_demo.run_sensor_demo(duration=60)
    except KeyboardInterrupt:
        sensor_demo.get_logger().info('Sensor demo interrupted')
    finally:
        sensor_demo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()