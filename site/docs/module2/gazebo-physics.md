---
title: "Gazebo Physics Simulation"
sidebar_position: 1
description: "Learn how to create physics-based simulations in Gazebo for humanoid robots with ROS 2 integration"
---

# Gazebo Physics Simulation for Humanoid Robots

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the fundamentals of physics simulation in Gazebo
- Create world files with proper physics properties for humanoid robots
- Configure physics engines and parameters for realistic simulation
- Launch and run basic physics simulations with humanoid models
- Control robot joints and observe physics interactions

## Introduction to Gazebo Physics

Gazebo is a powerful 3D simulation environment that provides robust physics simulation capabilities. For humanoid robotics, Gazebo enables realistic modeling of physical interactions between robots and their environment. This chapter will cover how to set up and configure physics simulations for humanoid robots.

## Understanding Physics Engines

### Available Physics Engines

Gazebo supports multiple physics engines, each with different characteristics:

- **ODE (Open Dynamics Engine)**: Default engine, good balance of performance and accuracy
- **Bullet**: High-performance engine, good for real-time simulation
- **DART (Dynamic Animation and Robotics Toolkit)**: Advanced engine with constraint-based solver

### Physics Parameters

Key physics parameters that affect simulation behavior:

- **Gravity**: Typically set to Earth's gravity (-9.81 m/s² in Z direction)
- **Time Step**: Simulation time increment (smaller = more accurate but slower)
- **Real Time Factor**: Target simulation speed (1.0 = real-time)
- **Max Contacts**: Maximum contacts between objects

## Creating World Files

### Basic World Structure

A Gazebo world file is an SDF (Simulation Description Format) file that defines the simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add your humanoid robot -->
    <include>
      <uri>file://models/humanoid/basic_humanoid.urdf</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Physics Configuration

The physics section defines how the simulation behaves:

```xml
<physics type="ode">
  <!-- Gravity vector (x, y, z) -->
  <gravity>0 0 -9.8</gravity>

  <!-- Time step for simulation (seconds) -->
  <max_step_size>0.001</max_step_size>

  <!-- Target real-time factor -->
  <real_time_factor>1.0</real_time_factor>

  <!-- Update rate (Hz) -->
  <real_time_update_rate>1000.0</real_time_update_rate>

  <!-- Solver parameters -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Humanoid Robot Model Integration

### URDF Integration

To integrate a humanoid robot model into Gazebo, you need to ensure your URDF is compatible:

```xml
<robot name="basic_humanoid">
  <!-- Links with physical properties -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joints with proper limits -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.3"/>
  </joint>
</robot>
```

### Joint Configuration

Proper joint configuration is crucial for realistic physics simulation:

- **Revolute joints**: For rotational movement with limits
- **Prismatic joints**: For linear movement with limits
- **Fixed joints**: For permanent connections
- **Continuous joints**: For unlimited rotation (use carefully)

## Launching Physics Simulations

### ROS 2 Launch Files

Create launch files to easily start your physics simulations:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os


def generate_launch_description():
    """Launch Gazebo with physics simulation."""

    # World file argument
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='basic_humanoid.sdf',
        description='Choose one of the world files from `/examples/gazebo/worlds`'
    )

    # Get world file path
    world_path = os.path.join(
        get_package_share_directory('examples_gazebo'),
        'worlds',
        LaunchConfiguration('world')
    )

    # Launch Gazebo
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gazebo
    ])
```

### Running the Simulation

To run a physics simulation:

1. Source your ROS 2 environment
2. Launch the simulation: `ros2 launch examples_gazebo basic_physics.launch.py`
3. Monitor the simulation in Gazebo GUI
4. Interact with the robot using ROS 2 commands

## Controlling Robot Joints

### Joint State Publisher

The joint state publisher broadcasts current joint positions:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math


class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')

        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Timer for publishing
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Initialize joint names and positions
        self.joint_names = [
            'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
            'left_hip', 'left_knee', 'right_hip', 'right_knee'
        ]
        self.joint_positions = [0.0] * len(self.joint_names)

    def publish_joint_states(self):
        """Publish joint state messages."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names
        msg.position = self.joint_positions

        self.joint_pub.publish(msg)
```

### Trajectory Control

For more sophisticated control, use trajectory messages:

```python
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def send_trajectory_command(self, joint_names, positions, duration=1.0):
    """Send a trajectory command to move joints."""
    msg = JointTrajectory()
    msg.joint_names = joint_names

    point = JointTrajectoryPoint()
    point.positions = positions
    point.time_from_start.sec = int(duration)
    point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

    msg.points.append(point)

    self.trajectory_pub.publish(msg)
```

## Testing Physics Simulation

### Basic Movement Test

Verify that your physics simulation works by testing basic movements:

1. Launch the simulation
2. Send joint commands to move arms/legs
3. Observe the robot's response in Gazebo
4. Check for realistic physics behavior (gravity, collisions)

### Environmental Interaction Test

Test how the robot interacts with the environment:

1. Position the robot near objects
2. Move limbs to make contact with objects
3. Verify that collisions are handled properly
4. Check that the robot maintains balance appropriately

## Performance Optimization

### Physics Fidelity vs Performance

Balance physics accuracy with simulation speed:

- **High fidelity**: Small time steps, many solver iterations
- **Performance**: Larger time steps, fewer iterations
- **Education**: Medium settings for balance of accuracy and speed

### Real-Time Factor

Monitor the real-time factor to ensure smooth simulation:

- **RTF = 1.0**: Simulation running at real-time speed
- **RTF < 1.0**: Simulation slower than real-time
- **RTF > 1.0**: Simulation faster than real-time (if possible)

## Troubleshooting Common Issues

### Robot Falling Through Ground

- Check collision geometries in URDF
- Verify inertial properties are properly defined
- Ensure mass values are realistic

### Unstable Joint Movements

- Adjust solver parameters (iterations, SOR value)
- Check joint limits and safety controllers
- Verify PID controller gains

### Slow Simulation Performance

- Increase time step size
- Reduce solver iterations
- Simplify collision geometries

## Best Practices

### Model Design

- Keep URDF models as simple as possible while maintaining accuracy
- Use realistic mass and inertia values
- Properly configure joint limits and safety controllers

### Simulation Setup

- Start with basic physics parameters and tune as needed
- Use appropriate world files for different testing scenarios
- Monitor simulation performance metrics

## Summary

This chapter covered the fundamentals of physics simulation in Gazebo for humanoid robots. You learned how to:

1. Configure physics engines and parameters for realistic simulation
2. Create world files that integrate humanoid robot models
3. Launch and run physics simulations with ROS 2
4. Control robot joints and observe physics interactions
5. Optimize simulation performance for different use cases

The next chapter will explore how to simulate various sensors in Gazebo and connect them to ROS 2 topics for perception capabilities.

## Physics Simulation Concepts and Code Examples

### Joint Control Implementation

The joint control interface allows you to control robot joints programmatically. Here's a complete example:

```python
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
```

### Physics Configuration Best Practices

When configuring physics for humanoid robots, consider these best practices:

- **Gravity**: Use standard Earth gravity (9.81 m/s²) unless simulating other environments
- **Time Step**: Start with 0.001s for accuracy, increase if performance is critical
- **Real Time Factor**: 1.0 for real-time simulation, higher for faster-than-real-time
- **Solver Iterations**: Higher values for stability, lower for performance

### Launch File Configuration

Example of a comprehensive launch file for physics simulation:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os


def generate_launch_description():
    """Launch Gazebo with a basic humanoid robot model."""

    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='basic_humanoid.sdf',
        description='Choose one of the world files from `/examples/gazebo/worlds`'
    )

    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up to project root
    world_path = os.path.join(project_root, 'examples', 'gazebo', 'worlds', 'basic_humanoid.sdf')
    urdf_path = os.path.join(project_root, 'examples', 'gazebo', 'models', 'humanoid', 'basic_humanoid.urdf')

    # Launch Gazebo with the world
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path],
        output='screen'
    )

    # Launch robot state publisher to publish URDF
    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_desc
        }]
    )

    # Launch joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_gui': False,
            'rate': 50.0
        }]
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        robot_state_publisher,
        joint_state_publisher
    ])
```

## Exercises

1. **Basic World Creation**: Create a simple world file with a humanoid robot and run the simulation.
2. **Joint Control**: Implement joint controllers to make the robot perform simple movements.
3. **Physics Tuning**: Experiment with different physics parameters to observe their effects.
4. **Environmental Interaction**: Test how the robot interacts with objects in the environment.