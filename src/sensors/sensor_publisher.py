#!/usr/bin/env python3

"""
Sensor Data Publisher for Gazebo Simulation

This module provides nodes to publish sensor data from Gazebo simulation
to ROS 2 topics for consumption by other nodes and applications.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan, Imu
from std_msgs.msg import Header
import numpy as np
import math

try:
    from cv_bridge import CvBridge
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("OpenCV not available, camera functionality will be limited")


class SensorPublisher(Node):
    """
    A ROS 2 node for publishing sensor data from Gazebo simulation.
    """

    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publishers for different sensor types
        self.camera_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/lidar/scan', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        # Create CvBridge for image conversion
        self.cv_bridge = CvBridge()

        # Timer for publishing sensor data
        self.timer = self.create_timer(0.1, self.publish_sensor_data)  # 10 Hz

        # Sensor data parameters
        self.scan_angle_min = -math.pi
        self.scan_angle_max = math.pi
        self.scan_angle_increment = 2 * math.pi / 360  # 360 samples
        self.scan_range_min = 0.1
        self.scan_range_max = 30.0

        # IMU orientation (initially at rest)
        self.imu_orientation = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w
        self.imu_angular_velocity = [0.0, 0.0, 0.0]
        self.imu_linear_acceleration = [0.0, 0.0, -9.81]  # Gravity

        # Camera parameters
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 1.047  # 60 degrees in radians

        self.get_logger().info('Sensor Publisher node initialized')

    def publish_sensor_data(self):
        """Publish sensor data to ROS topics."""
        self.publish_camera_data()
        self.publish_lidar_data()
        self.publish_imu_data()

    def publish_camera_data(self):
        """Publish camera image and camera info."""
        if CV_AVAILABLE:
            # Create a dummy image for simulation
            image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

            # Add some simple patterns to make the image recognizable
            cv2.circle(image, (self.camera_width//2, self.camera_height//2), 50, (255, 0, 0), -1)
            cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), 2)

            # Convert to ROS Image message
            ros_image = self.cv_bridge.cv2_to_imgmsg(image, encoding='bgr8')
        else:
            # Create a minimal image message without OpenCV
            ros_image = Image()
            ros_image.height = self.camera_height
            ros_image.width = self.camera_width
            ros_image.encoding = 'rgb8'
            ros_image.is_bigendian = False
            ros_image.step = self.camera_width * 3  # 3 bytes per pixel
            ros_image.data = [0] * (self.camera_width * self.camera_height * 3)  # Empty image

        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'camera_link'

        # Publish image
        self.camera_pub.publish(ros_image)

        # Create and publish camera info
        camera_info = self.create_camera_info()
        camera_info.header.stamp = ros_image.header.stamp
        camera_info.header.frame_id = 'camera_link'
        self.camera_info_pub.publish(camera_info)

    def create_camera_info(self):
        """Create camera info message."""
        camera_info = CameraInfo()

        # Image dimensions
        camera_info.height = self.camera_height
        camera_info.width = self.camera_width

        # Distortion parameters (assuming no distortion for simulation)
        camera_info.distortion_model = 'plumb_bob'
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Intrinsic camera matrix
        fx = self.camera_width / (2 * math.tan(self.camera_fov / 2))
        fy = fx  # Assume square pixels
        cx = self.camera_width / 2
        cy = self.camera_height / 2

        camera_info.k = [fx, 0.0, cx,
                         0.0, fy, cy,
                         0.0, 0.0, 1.0]

        # Rectification matrix (identity for monocular camera)
        camera_info.r = [1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0]

        # Projection matrix
        camera_info.p = [fx, 0.0, cx, 0.0,
                         0.0, fy, cy, 0.0,
                         0.0, 0.0, 1.0, 0.0]

        return camera_info

    def publish_lidar_data(self):
        """Publish LIDAR scan data."""
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'lidar_link'

        # Set scan parameters
        scan_msg.angle_min = self.scan_angle_min
        scan_msg.angle_max = self.scan_angle_max
        scan_msg.angle_increment = self.scan_angle_increment
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1  # 10Hz
        scan_msg.range_min = self.scan_range_min
        scan_msg.range_max = self.scan_range_max

        # Generate sample ranges (in a real implementation, these would come from Gazebo)
        num_ranges = int((self.scan_angle_max - self.scan_angle_min) / self.scan_angle_increment) + 1
        # Simulate a circular object at 5m distance
        ranges = []
        for i in range(num_ranges):
            angle = self.scan_angle_min + i * self.scan_angle_increment
            # Create a simple pattern: a circle at 5m distance
            distance = 5.0 + 0.5 * math.sin(4 * angle)  # Add some variation
            ranges.append(distance)

        scan_msg.ranges = ranges
        scan_msg.intensities = [100.0] * len(ranges)  # Constant intensity

        self.lidar_pub.publish(scan_msg)

    def publish_imu_data(self):
        """Publish IMU data."""
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        # Set orientation (as quaternion)
        imu_msg.orientation.x = self.imu_orientation[0]
        imu_msg.orientation.y = self.imu_orientation[1]
        imu_msg.orientation.z = self.imu_orientation[2]
        imu_msg.orientation.w = self.imu_orientation[3]

        # Set angular velocity
        imu_msg.angular_velocity.x = self.imu_angular_velocity[0]
        imu_msg.angular_velocity.y = self.imu_angular_velocity[1]
        imu_msg.angular_velocity.z = self.imu_angular_velocity[2]

        # Set linear acceleration
        imu_msg.linear_acceleration.x = self.imu_linear_acceleration[0]
        imu_msg.linear_acceleration.y = self.imu_linear_acceleration[1]
        imu_msg.linear_acceleration.z = self.imu_linear_acceleration[2]

        # Set covariance matrices (set to 0 for simulation)
        imu_msg.orientation_covariance = [0.0] * 9
        imu_msg.angular_velocity_covariance = [0.0] * 9
        imu_msg.linear_acceleration_covariance = [0.0] * 9

        self.imu_pub.publish(imu_msg)


def main(args=None):
    """Main function to run the sensor publisher node."""
    rclpy.init(args=args)

    sensor_publisher = SensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        sensor_publisher.get_logger().info('Shutting down Sensor Publisher')
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()