#!/usr/bin/env python3

"""
Sensor Validation Tests for Gazebo Simulation

This module provides tests to validate that sensor data published by the simulation
is of appropriate quality and format for use in ROS 2 applications.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan, Imu
import numpy as np
import time
from collections import deque


class SensorValidator(Node):
    """
    A ROS 2 node for validating sensor data quality and format.
    """

    def __init__(self):
        super().__init__('sensor_validator')

        # Subscribers for sensor data
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
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

        # Data storage for validation
        self.camera_data_queue = deque(maxlen=10)
        self.lidar_data_queue = deque(maxlen=10)
        self.imu_data_queue = deque(maxlen=10)

        # Validation statistics
        self.validation_results = {
            'camera': {'received': 0, 'valid': 0, 'invalid': 0},
            'lidar': {'received': 0, 'valid': 0, 'invalid': 0},
            'imu': {'received': 0, 'valid': 0, 'invalid': 0}
        }

        # Timer for periodic validation reports
        self.timer = self.create_timer(5.0, self.report_validation_status)

        self.get_logger().info('Sensor Validator node initialized')

    def camera_callback(self, msg):
        """Validate camera image data."""
        self.validation_results['camera']['received'] += 1

        # Validate message structure
        is_valid = True
        errors = []

        # Check header
        if not msg.header.stamp.sec and not msg.header.stamp.nanosec:
            errors.append("Invalid timestamp")
            is_valid = False

        # Check image dimensions
        if msg.height <= 0 or msg.width <= 0:
            errors.append(f"Invalid dimensions: {msg.height}x{msg.width}")
            is_valid = False

        # Check encoding
        if not msg.encoding:
            errors.append("Empty encoding")
            is_valid = False

        # Check data size
        expected_size = msg.height * msg.width * (3 if 'rgb' in msg.encoding or 'bgr' in msg.encoding else 1)
        if len(msg.data) != expected_size:
            errors.append(f"Data size mismatch: expected {expected_size}, got {len(msg.data)}")
            is_valid = False

        # Store validation result
        self.camera_data_queue.append({
            'timestamp': msg.header.stamp,
            'valid': is_valid,
            'errors': errors,
            'width': msg.width,
            'height': msg.height,
            'encoding': msg.encoding
        })

        if is_valid:
            self.validation_results['camera']['valid'] += 1
        else:
            self.validation_results['camera']['invalid'] += 1
            self.get_logger().warn(f'Camera data validation failed: {errors}')

    def camera_info_callback(self, msg):
        """Validate camera info data."""
        # Basic validation for camera info - mainly check that it's received properly
        if msg.width <= 0 or msg.height <= 0:
            self.get_logger().warn('Invalid camera info dimensions')

    def lidar_callback(self, msg):
        """Validate LIDAR scan data."""
        self.validation_results['lidar']['received'] += 1

        is_valid = True
        errors = []

        # Check header
        if not msg.header.stamp.sec and not msg.header.stamp.nanosec:
            errors.append("Invalid timestamp")
            is_valid = False

        # Check scan parameters
        if msg.angle_min >= msg.angle_max:
            errors.append(f"Invalid angle range: min {msg.angle_min} >= max {msg.angle_max}")
            is_valid = False

        if msg.range_min >= msg.range_max:
            errors.append(f"Invalid range: min {msg.range_min} >= max {msg.range_max}")
            is_valid = False

        # Check data arrays
        if len(msg.ranges) == 0:
            errors.append("Empty ranges array")
            is_valid = False
        else:
            # Check if ranges are within expected bounds
            invalid_ranges = [r for r in msg.ranges if r < msg.range_min or (r > msg.range_max and r != float('inf'))]
            if invalid_ranges:
                errors.append(f"Out of range values: {len(invalid_ranges)} invalid ranges")
                is_valid = False

        # Store validation result
        self.lidar_data_queue.append({
            'timestamp': msg.header.stamp,
            'valid': is_valid,
            'errors': errors,
            'range_min': msg.range_min,
            'range_max': msg.range_max,
            'num_ranges': len(msg.ranges)
        })

        if is_valid:
            self.validation_results['lidar']['valid'] += 1
        else:
            self.validation_results['lidar']['invalid'] += 1
            self.get_logger().warn(f'LIDAR data validation failed: {errors}')

    def imu_callback(self, msg):
        """Validate IMU data."""
        self.validation_results['imu']['received'] += 1

        is_valid = True
        errors = []

        # Check header
        if not msg.header.stamp.sec and not msg.header.stamp.nanosec:
            errors.append("Invalid timestamp")
            is_valid = False

        # Check orientation quaternion (should be normalized)
        norm = (msg.orientation.x**2 + msg.orientation.y**2 +
                msg.orientation.z**2 + msg.orientation.w**2)**0.5
        if abs(norm - 1.0) > 0.01:  # Allow small floating point errors
            errors.append(f"Quaternion not normalized: {norm}")
            is_valid = False

        # Check for NaN or infinity values
        values_to_check = [
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ]

        for i, val in enumerate(values_to_check):
            if np.isnan(val) or np.isinf(val):
                field_names = [
                    'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w',
                    'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',
                    'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z'
                ]
                errors.append(f"Invalid value ({val}) in {field_names[i]}")
                is_valid = False

        # Store validation result
        self.imu_data_queue.append({
            'timestamp': msg.header.stamp,
            'valid': is_valid,
            'errors': errors,
            'orientation_norm': norm,
            'linear_acceleration': [msg.linear_acceleration.x,
                                  msg.linear_acceleration.y,
                                  msg.linear_acceleration.z]
        })

        if is_valid:
            self.validation_results['imu']['valid'] += 1
        else:
            self.validation_results['imu']['invalid'] += 1
            self.get_logger().warn(f'IMU data validation failed: {errors}')

    def report_validation_status(self):
        """Report validation statistics."""
        self.get_logger().info("=== Sensor Validation Report ===")

        for sensor_type in self.validation_results:
            stats = self.validation_results[sensor_type]
            total = stats['received']
            valid = stats['valid']

            if total > 0:
                success_rate = (valid / total) * 100
                self.get_logger().info(
                    f"{sensor_type.upper()}: {valid}/{total} valid ({success_rate:.1f}%)"
                )
            else:
                self.get_logger().info(f"{sensor_type.upper()}: No data received")

        self.get_logger().info("===============================")

    def run_validation_tests(self, duration=30):
        """
        Run validation tests for a specified duration.

        Args:
            duration (int): Duration to run tests in seconds
        """
        self.get_logger().info(f'Running sensor validation tests for {duration} seconds...')

        start_time = time.time()
        while time.time() - start_time < duration:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.report_validation_status()
        self.print_final_summary()

    def print_final_summary(self):
        """Print final validation summary."""
        self.get_logger().info("=== Final Validation Summary ===")

        all_passed = True
        for sensor_type in self.validation_results:
            stats = self.validation_results[sensor_type]
            total = stats['received']
            valid = stats['valid']

            if total > 0:
                success_rate = (valid / total) * 100
                status = "PASS" if success_rate >= 95.0 else "FAIL"

                if success_rate < 95.0:
                    all_passed = False

                self.get_logger().info(
                    f"{sensor_type.upper()}: {status} - {valid}/{total} valid ({success_rate:.1f}%)"
                )
            else:
                self.get_logger().info(f"{sensor_type.upper()}: FAIL - No data received")
                all_passed = False

        result = "PASS" if all_passed else "FAIL"
        self.get_logger().info(f"Overall validation result: {result}")
        self.get_logger().info("===============================")


def main(args=None):
    """Main function to run the sensor validation tests."""
    rclpy.init(args=args)

    sensor_validator = SensorValidator()

    try:
        # Run validation tests for 30 seconds
        sensor_validator.run_validation_tests(duration=30)
    except KeyboardInterrupt:
        sensor_validator.get_logger().info('Validation tests interrupted')
    finally:
        sensor_validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()