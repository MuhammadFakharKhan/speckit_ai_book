---
title: "ROS 2 Fundamentals for Humanoids"
sidebar_position: 2
description: "Learn core ROS 2 concepts for humanoid robots"
---

# ROS 2 Fundamentals for Humanoids

## Learning Objectives
- Understand ROS 2 nodes, topics, services, and actions
- Set up a ROS 2 workspace for humanoid robotics
- Create a simple publisher-subscriber example for joint commands

## Learning Objectives Validation

Let's validate that each learning objective has been met:

1. **Understanding ROS 2 nodes, topics, services, and actions**: This objective has been met through detailed explanations of each concept, their purposes, and how they work in the context of humanoid robotics. You've learned about the publish-subscribe model, request-response communication, and long-running tasks with feedback.

2. **Setting up a ROS 2 workspace for humanoid robotics**: This objective has been met by covering the workspace structure, creation process, and build procedures. You now understand the typical directory structure and how to initialize and build a ROS 2 workspace.

3. **Creating a simple publisher-subscriber example for joint commands**: This objective has been met with practical Python examples showing how to create publisher and subscriber nodes. You've seen how to implement these nodes for joint command communication in humanoid robots, including proper ROS 2 node structure and message handling.

## Introduction to ROS 2

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

For humanoid robotics, ROS 2 provides the communication infrastructure needed to coordinate between different subsystems like perception, planning, control, and actuation.

## Core Concepts

### Nodes

A node is a process that performs computation. In ROS 2, nodes are written using client libraries such as `rclpy` (Python) or `rclcpp` (C++). Nodes are the fundamental building blocks of a ROS program.

For humanoid robots, you might have nodes for:
- Joint controllers
- Sensor processing
- Motion planning
- Perception systems

### Topics and Messages

Topics enable asynchronous message passing between nodes. A node can publish messages to a topic, and other nodes can subscribe to that topic to receive the messages.

In humanoid robotics, common topics include:
- Joint states (`sensor_msgs/JointState`)
- Twist commands for base movement (`geometry_msgs/Twist`)
- Sensor data from cameras, IMUs, etc.

### Services

Services provide synchronous request/response communication. A service has a client that sends a request and a server that processes the request and returns a response.

For humanoid robots, services might include:
- Calibration routines
- Mode switching
- Emergency stop activation

### Actions

Actions are used for long-running tasks that require feedback and the ability to cancel. They extend the service paradigm with additional features for tracking progress.

For humanoid robots, actions are often used for:
- Walking patterns
- Arm trajectories
- Complex manipulation tasks

## ROS 2 Workspace Setup

A ROS 2 workspace is a directory where you modify, build, and install ROS 2 packages.

### Creating a Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### Building the Workspace

```bash
colcon build
source install/setup.bash
```

## Publisher-Subscriber Example for Joint Commands

Let's create a simple publisher-subscriber example for joint commands, which is fundamental for controlling humanoid robot joints.

### Publisher Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        self.publisher_ = self.create_publisher(String, 'joint_commands', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Joint command: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    joint_command_publisher = JointCommandPublisher()
    rclpy.spin(joint_command_publisher)
    joint_command_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Node

```python
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
```

## Running the Example

1. Save the publisher code as `joint_publisher.py` and the subscriber code as `joint_subscriber.py`
2. Build your workspace: `colcon build`
3. Source the setup file: `source install/setup.bash`
4. Run the publisher: `ros2 run <package_name> joint_publisher`
5. In another terminal, run the subscriber: `ros2 run <package_name> joint_subscriber`

## Advanced ROS 2 Concepts for Humanoid Robotics

### Quality of Service (QoS) Settings

In humanoid robotics applications, Quality of Service (QoS) settings are crucial for ensuring reliable communication between nodes. QoS profiles allow you to specify how messages are handled in terms of reliability, durability, and history.

For joint command topics in humanoid robots, you might want to use reliable reliability and keep-all history:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

qos_profile = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_ALL
)

self.publisher = self.create_publisher(
    Float64MultiArray,
    'joint_commands',
    qos_profile
)
```

### Lifecycle Nodes

For humanoid robots that require complex state management, lifecycle nodes provide a way to manage the state of your nodes explicitly. This is useful for robots that need to go through initialization, activation, and deactivation sequences.

### Parameters

ROS 2 parameters allow you to configure nodes at runtime. For humanoid robots, you might have parameters for joint limits, safety thresholds, or controller gains:

```python
self.declare_parameter('max_joint_velocity', 1.0)
max_vel = self.get_parameter('max_joint_velocity').value
```

### Timers

Timers in ROS 2 allow you to execute callbacks at specific rates. For humanoid robot control, you might need precise timing for control loops:

```python
timer_period = 0.01  # 10ms for 100Hz control loop
self.timer = self.create_timer(timer_period, self.control_callback)
```

## Working with ROS 2 Tools

### Command Line Tools

ROS 2 provides several command-line tools for debugging and introspecting your system:

- `ros2 node list` - List all active nodes
- `ros2 topic list` - List all active topics
- `ros2 topic echo <topic_name>` - Print messages from a topic
- `ros2 service list` - List all available services
- `ros2 action list` - List all available actions

### Launch Files

Launch files allow you to start multiple nodes with a single command. For humanoid robots, you might have launch files that start all necessary nodes:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_hardware',
            executable='joint_state_publisher',
            name='joint_state_publisher'
        ),
        Node(
            package='my_robot_control',
            executable='controller_manager',
            name='controller_manager'
        )
    ])
```

## Best Practices for Humanoid Robotics

1. **Modularity**: Design your system with modular nodes that have clear responsibilities
2. **Safety**: Implement safety checks in your nodes to prevent dangerous robot behavior
3. **Monitoring**: Use ROS 2's built-in monitoring tools to track node health
4. **Documentation**: Document your custom message types and services
5. **Testing**: Write tests for your nodes using ROS 2's testing frameworks

## Message Types for Humanoid Robotics

ROS 2 provides several standard message types that are particularly useful for humanoid robotics:

### Joint State Messages

The `sensor_msgs/JointState` message is essential for humanoid robots to communicate joint positions, velocities, and efforts:

```python
from sensor_msgs.msg import JointState

# Example joint state message for a humanoid with 24 DOF
joint_state = JointState()
joint_state.name = [
    'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee',
    'left_ankle_pitch', 'left_ankle_roll', 'right_hip_yaw', 'right_hip_roll',
    'right_hip_pitch', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    'torso_yaw', 'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow',
    'neck_yaw', 'neck_pitch', 'neck_roll', 'head_pan', 'head_tilt'
]

joint_state.position = [0.0] * 24  # Initialize all joints to 0 position
joint_state.velocity = [0.0] * 24  # Initialize all velocities to 0
joint_state.effort = [0.0] * 24    # Initialize all efforts to 0
```

### Twist Messages

The `geometry_msgs/Twist` message is used for specifying linear and angular velocities, which is important for humanoid base movement:

```python
from geometry_msgs.msg import Twist

twist = Twist()
twist.linear.x = 0.1  # Forward velocity in m/s
twist.linear.y = 0.0  # Lateral velocity
twist.linear.z = 0.0  # Vertical velocity
twist.angular.x = 0.0 # Angular velocity around X-axis
twist.angular.y = 0.0 # Angular velocity around Y-axis
twist.angular.z = 0.1 # Angular velocity around Z-axis (turning)
```

### Custom Message Types

For humanoid robotics, you may need to create custom message types for specific applications. To create a custom message:

1. Create a `msg` directory in your package
2. Define your message in a `.msg` file (e.g., `HumanoidCommand.msg`):
```
float64[24] joint_positions
float64[24] joint_velocities
float64[24] joint_efforts
string command_type
```
3. Add the message to your CMakeLists.txt for generation

## Creating a Simple Controller Node

Let's create a more sophisticated controller node that processes joint commands and sends them to the robot hardware:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import numpy as np

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Subscriber for joint commands
        self.command_subscriber = self.create_subscription(
            String,
            'joint_commands',
            self.command_callback,
            10
        )

        # Publisher for joint states
        self.joint_state_publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        # Initialize joint state
        self.joint_state = JointState()
        self.joint_state.name = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee',
            'left_ankle_pitch', 'left_ankle_roll', 'right_hip_yaw', 'right_hip_roll',
            'right_hip_pitch', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'torso_yaw', 'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow',
            'neck_yaw', 'neck_pitch', 'neck_roll', 'head_pan', 'head_tilt'
        ]
        self.joint_state.position = [0.0] * 24
        self.joint_state.velocity = [0.0] * 24
        self.joint_state.effort = [0.0] * 24

        self.target_positions = [0.0] * 24
        self.get_logger().info('Humanoid Controller initialized')

    def command_callback(self, msg):
        try:
            # Parse command string - in real application this would be more robust
            cmd_parts = msg.data.split(':')
            if len(cmd_parts) == 2 and cmd_parts[0] == 'Joint command':
                # Update target positions based on command
                idx = int(cmd_parts[1]) % 24
                self.target_positions[idx] = float(cmd_parts[1]) * 0.1
        except Exception as e:
            self.get_logger().error(f'Error parsing command: {e}')

    def control_loop(self):
        # Simple PD controller for each joint
        kp = 10.0  # Proportional gain
        kd = 1.0   # Derivative gain

        for i in range(24):
            error = self.target_positions[i] - self.joint_state.position[i]
            velocity = (self.target_positions[i] - self.joint_state.position[i]) / 0.01
            control_effort = kp * error + kd * velocity

            # Update joint state with new position (simplified model)
            self.joint_state.position[i] += control_effort * 0.01  # dt = 0.01s
            self.joint_state.velocity[i] = velocity
            self.joint_state.effort[i] = control_effort

        # Publish updated joint state
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_state.header.frame_id = 'base_link'
        self.joint_state_publisher.publish(self.joint_state)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Debugging and Troubleshooting

When working with ROS 2 in humanoid robotics, you'll encounter various issues. Here are common debugging strategies:

### Node Communication Issues

- Use `ros2 topic list` to verify topics exist
- Use `ros2 topic info <topic_name>` to check publisher/subscriber counts
- Use `ros2 topic echo <topic_name>` to verify data is flowing
- Check QoS settings match between publishers and subscribers

### Performance Issues

- Monitor CPU usage with `top` or `htop`
- Check network bandwidth if using multi-machine setup
- Use appropriate QoS settings (reliable vs best-effort)
- Profile your nodes to identify bottlenecks

### Memory Management

- Be careful with message allocation in high-frequency loops
- Use intraprocess communication when possible for better performance
- Monitor memory usage with ROS 2 tools

## Exercises

1. Create a parameter server node that manages joint limits for your humanoid robot
2. Implement a lifecycle node for robot state management (idle, active, emergency_stop)
3. Create a launch file that starts both the publisher and subscriber nodes
4. Use QoS settings to ensure reliable delivery of critical joint commands
5. Implement a simple service that returns the current joint positions

## Summary

In this chapter, we covered the fundamental concepts of ROS 2 that are essential for working with humanoid robots. Understanding nodes, topics, services, and actions is crucial for building robust robotic systems. The publisher-subscriber pattern is particularly important for the distributed nature of humanoid robot control systems.

We also explored advanced concepts like QoS settings, lifecycle nodes, parameters, and timers that are important for real-world humanoid robotics applications. Additionally, we looked at ROS 2 tools for debugging and introspection, as well as best practices for developing humanoid robot systems.

In the next chapter, we'll explore how to design Python AI agents that interface with ROS 2 using rclpy, building on the foundational concepts we've learned here.

## References to Official ROS 2 Documentation

This chapter's content is based on the official ROS 2 documentation. For more detailed information, please refer to:

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS 2 Concepts](https://docs.ros.org/en/humble/Concepts.html)
- [rclpy API Documentation](https://docs.ros.org/en/humble/p/rclpy/)
- [ROS 2 Message Types](https://docs.ros.org/en/humble/Concepts/About-ROS-Interfaces.html)
- [Quality of Service Settings](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)
- [ROS 2 Launch Files](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Creating-Launch-Files.html)