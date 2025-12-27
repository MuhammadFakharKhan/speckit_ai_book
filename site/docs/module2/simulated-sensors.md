---
title: "Simulated Sensors"
sidebar_position: 2
description: "Learn how to simulate various sensors in Gazebo and connect them to ROS 2 topics for humanoid robotics applications"
---

# Simulated Sensors in Gazebo

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand different types of sensors used in robotics
- Configure camera, LIDAR, and IMU sensors in Gazebo
- Connect simulated sensors to ROS 2 topics
- Process and validate sensor data from simulation
- Create sensor processing nodes for real-world applications

## Introduction

Sensors are crucial components of any robotic system, providing the robot with information about its environment and internal state. In simulation, we can create realistic sensor models that behave similarly to their real-world counterparts. This chapter will cover how to simulate common sensors in Gazebo and connect them to ROS 2 for processing and analysis.

## Types of Sensors in Robotics

### Camera Sensors

Camera sensors provide visual information about the environment. In Gazebo, camera sensors can be configured with various parameters:

- **Resolution**: Width and height in pixels
- **Field of View (FOV)**: Angular extent of the scene captured
- **Image Format**: Color depth and encoding
- **Update Rate**: How frequently the camera publishes images

### LIDAR Sensors

LIDAR (Light Detection and Ranging) sensors measure distances by illuminating targets with laser light and measuring the reflection. In simulation:

- **Range**: Minimum and maximum detection distances
- **Resolution**: Angular resolution of measurements
- **Scan Pattern**: Horizontal and vertical field of view
- **Update Rate**: Frequency of scans

### IMU Sensors

Inertial Measurement Units (IMUs) measure angular velocity, linear acceleration, and sometimes magnetic field. In Gazebo:

- **Orientation**: 3D orientation of the sensor
- **Angular Velocity**: Rotational velocity in 3D space
- **Linear Acceleration**: Acceleration in 3D space
- **Noise Models**: Realistic noise characteristics

## Configuring Sensors in Gazebo

### Camera Sensor Configuration

Camera sensors in Gazebo are configured using SDF (Simulation Description Format). Here's a basic camera configuration:

```xml
<sensor name="camera" type="camera">
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>300</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### LIDAR Sensor Configuration

LIDAR sensors use ray-based simulation in Gazebo:

```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### IMU Sensor Configuration

IMU sensors provide inertial measurements:

```xml
<sensor name="imu_sensor" type="imu">
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <visualize>false</visualize>
</sensor>
```

## Connecting Sensors to ROS 2

Gazebo can bridge sensor data to ROS 2 topics using the Gazebo ROS packages. The sensor bridge configuration maps Gazebo sensors to ROS 2 topics.

### Sensor Bridge Configuration

```yaml
# ROS 2 to Gazebo Sensor Bridge Configuration
camera_bridge:
  ros__parameters:
    image_topic_name: "/camera/image_raw"
    camera_info_topic_name: "/camera/camera_info"
    camera_frame_id: "camera_link"
    update_rate: 30.0

lidar_bridge:
  ros__parameters:
    scan_topic_name: "/lidar/scan"
    lidar_frame_id: "lidar_link"
    update_rate: 10.0

imu_bridge:
  ros__parameters:
    imu_topic_name: "/imu/data"
    imu_frame_id: "imu_link"
    update_rate: 100.0
```

## Processing Sensor Data

### Subscribing to Sensor Topics

In ROS 2, you can subscribe to sensor data using the appropriate message types:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Create subscribers for different sensor types
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

    def camera_callback(self, msg):
        # Process camera image data
        self.get_logger().info(f'Received image: {msg.width}x{msg.height}')

    def lidar_callback(self, msg):
        # Process LIDAR scan data
        self.get_logger().info(f'Received scan with {len(msg.ranges)} ranges')

    def imu_callback(self, msg):
        # Process IMU data
        self.get_logger().info(f'IMU orientation: {msg.orientation}')
```

## Sensor Validation and Quality Assurance

### Validating Sensor Data

It's important to validate that sensor data is of appropriate quality and format:

- **Timestamp validation**: Ensure messages have valid timestamps
- **Range validation**: Check that sensor readings are within expected ranges
- **Format validation**: Verify message structure and encoding
- **Frequency validation**: Confirm sensors publish at expected rates

### Quality Metrics

Common metrics for sensor validation include:

- **Data completeness**: Percentage of valid sensor readings
- **Data accuracy**: How closely simulated data matches expected values
- **Temporal consistency**: Regularity of sensor updates
- **Spatial consistency**: Accuracy of spatial relationships in sensor data

## Hands-on Example: Sensor Demo

The following example demonstrates how to process and analyze sensor data from the simulated humanoid robot:

```python
# Example of sensor processing in simulation
import rclpy
from sensor_msgs.msg import LaserScan
import numpy as np

class SensorDemo(Node):
    def detect_objects(self, lidar_data):
        """Simple object detection using LIDAR data."""
        # Filter valid ranges
        valid_ranges = [r for r in lidar_data.ranges
                       if lidar_data.range_min <= r <= lidar_data.range_max]

        if not valid_ranges:
            return 0

        # Simple clustering based on distance jumps
        object_count = 0
        for i in range(1, len(valid_ranges)):
            if abs(valid_ranges[i] - valid_ranges[i-1]) > 0.5:
                object_count += 1

        return min(object_count, 10)  # Cap at 10 objects
```

## Sensor Integration Concepts

### Sensor Data Publisher Implementation

The sensor data publisher is responsible for converting Gazebo sensor data into ROS 2 messages. Here's a simplified implementation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publishers for different sensor types
        self.camera_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/lidar/scan', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        # Timer for publishing sensor data
        self.timer = self.create_timer(0.1, self.publish_sensor_data)

    def publish_sensor_data(self):
        """Publish sensor data to ROS topics."""
        # Implementation would convert Gazebo sensor data to ROS messages
        pass
```

### Sensor Validation Implementation

Sensor validation is crucial for ensuring data quality:

```python
def validate_lidar_data(self, msg):
    """Validate LIDAR scan data."""
    errors = []

    # Check if ranges are within expected bounds
    invalid_ranges = [r for r in msg.ranges if r < msg.range_min or (r > msg.range_max and r != float('inf'))]
    if invalid_ranges:
        errors.append(f"Out of range values: {len(invalid_ranges)} invalid ranges")

    return len(errors) == 0, errors
```

### Object Detection with LIDAR

Here's an example of processing LIDAR data for object detection:

```python
def detect_objects(self, lidar_data):
    """Simple object detection using LIDAR data."""
    # Filter valid ranges
    valid_ranges = [r for r in lidar_data.ranges
                   if lidar_data.range_min <= r <= lidar_data.range_max]

    if not valid_ranges:
        return 0

    # Simple clustering based on distance jumps
    object_count = 0
    for i in range(1, len(valid_ranges)):
        if abs(valid_ranges[i] - valid_ranges[i-1]) > 0.5:  # Threshold for object boundary
            object_count += 1

    return min(object_count, 10)  # Cap at 10 objects
```

## Summary

This chapter covered the fundamentals of simulating sensors in Gazebo and connecting them to ROS 2. You learned how to:

1. Configure different types of sensors in Gazebo
2. Connect sensors to ROS 2 topics using bridges
3. Process sensor data in ROS 2 nodes
4. Validate sensor data quality
5. Apply sensor data to practical applications

The next chapter will explore how to integrate Unity for visualization and human-robot interaction, building upon the sensor simulation concepts you've learned here.

## Exercises

1. **Camera Calibration**: Simulate a camera with different focal lengths and observe the effect on the field of view.
2. **LIDAR Range Analysis**: Process LIDAR data to detect objects at different distances and sizes.
3. **IMU Integration**: Use IMU data to estimate robot orientation and movement patterns.
4. **Sensor Fusion**: Combine data from multiple sensors to improve environmental understanding.