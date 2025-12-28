---
title: Python Agents for ROS 2
sidebar_position: 1
---

# Python Agents for ROS 2

## Learning Objectives

By the end of this chapter, you will be able to:
- Interface Python AI agents with ROS 2 using rclpy
- Design message flows between agents and robot controllers
- Implement Python nodes that bridge AI algorithms with robot systems

## Introduction to rclpy

`rclpy` is the Python client library for ROS 2. It provides a Python API for creating ROS 2 nodes, publishers, subscribers, services, and actions.

### Installing rclpy

`rclpy` is typically installed as part of the ROS 2 Python development packages:

```bash
pip3 install rclpy
# Or through ROS 2 installation
```

## Creating Python AI Agents

### Basic Node Structure

Here's the basic structure of a Python AI agent node:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState

class PythonAIAgent(Node):
    def __init__(self):
        super().__init__('python_ai_agent')

        # Create subscribers to receive data from robot
        self.robot_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.robot_state_callback,
            10
        )

        # Create publishers to send commands to robot
        self.command_publisher = self.create_publisher(
            String,
            'ai_commands',
            10
        )

        # Create timer for AI decision making
        self.ai_timer = self.create_timer(0.1, self.ai_decision_callback)

        self.get_logger().info('Python AI Agent initialized')

    def robot_state_callback(self, msg):
        """Process robot state data and update AI internal state"""
        self.get_logger().info(f'Received joint states: {len(msg.name)} joints')
        # Process the robot state for AI decision making

    def ai_decision_callback(self):
        """Main AI decision making function"""
        # Implement AI logic here
        command_msg = String()
        command_msg.data = 'AI command: move_to_target'
        self.command_publisher.publish(command_msg)
        self.get_logger().info(f'Published: {command_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    ai_agent = PythonAIAgent()

    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        pass
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Message Flow Between Agent and Controller

### Publisher-Subscriber Pattern

The most common pattern for AI agent to robot communication is using topics:

1. Robot controller publishes sensor data to topics
2. AI agent subscribes to sensor data topics
3. AI agent processes data and publishes commands to topics
4. Robot controller subscribes to command topics

### Service-Based Communication

For synchronous communication, services can be used:

```python
from rclpy.qos import QoSProfile
from example_interfaces.srv import Trigger

class AIAgentWithServices(Node):
    def __init__(self):
        super().__init__('ai_agent_with_services')

        # Create client for robot services
        self.robot_service_client = self.create_client(
            Trigger,
            'robot_execute_action'
        )

    def call_robot_service(self):
        """Call robot service to execute an action"""
        if self.robot_service_client.wait_for_service(timeout_sec=1.0):
            request = Trigger.Request()
            future = self.robot_service_client.call_async(request)
            return future
        else:
            self.get_logger().error('Robot service not available')
```

## Advanced AI Agent Patterns

### State Machine Agent

```python
from enum import Enum

class AgentState(Enum):
    IDLE = 1
    PLANNING = 2
    EXECUTING = 3
    ERROR = 4

class StateMachineAIAgent(Node):
    def __init__(self):
        super().__init__('state_machine_ai_agent')
        self.current_state = AgentState.IDLE

        self.command_publisher = self.create_publisher(
            String,
            'robot_commands',
            10
        )

        self.state_timer = self.create_timer(0.05, self.state_machine_callback)

    def state_machine_callback(self):
        """State machine for AI agent"""
        if self.current_state == AgentState.IDLE:
            # Check if we need to plan something
            if self.should_plan():
                self.current_state = AgentState.PLANNING
        elif self.current_state == AgentState.PLANNING:
            # Execute planning algorithm
            plan = self.plan_action()
            if plan:
                self.current_state = AgentState.EXECUTING
        elif self.current_state == AgentState.EXECUTING:
            # Execute the plan
            if self.is_plan_complete():
                self.current_state = AgentState.IDLE
        elif self.current_state == AgentState.ERROR:
            # Handle error state
            self.current_state = AgentState.IDLE
```

### Behavior Tree Agent

```python
class BehaviorNode:
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []

    def tick(self):
        """Execute the behavior and return status"""
        pass

class SequenceNode(BehaviorNode):
    def tick(self):
        for child in self.children:
            status = child.tick()
            if status != 'SUCCESS':
                return status
        return 'SUCCESS'

class AIAgentWithBehaviorTree(Node):
    def __init__(self):
        super().__init__('ai_agent_behavior_tree')

        # Create behavior tree
        self.root = SequenceNode('root')
        self.root.children = [
            self.create_sensor_check_node(),
            self.create_plan_node(),
            self.create_execute_node()
        ]

        self.bt_timer = self.create_timer(0.1, self.behavior_tree_callback)

    def behavior_tree_callback(self):
        """Execute behavior tree"""
        result = self.root.tick()
        self.get_logger().info(f'Behavior tree result: {result}')
```

## Integration with Machine Learning

### TensorFlow/PyTorch Integration

```python
import tensorflow as tf
# or
import torch

class MLIntegratedAIAgent(Node):
    def __init__(self):
        super().__init__('ml_integrated_ai_agent')

        # Load pre-trained model
        self.model = self.load_model()

        self.sensor_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.ml_decision_callback,
            10
        )

    def load_model(self):
        """Load pre-trained ML model"""
        # Load your trained model here
        # model = tf.keras.models.load_model('path/to/model')
        # or
        # model = torch.load('path/to/model.pth')
        pass

    def ml_decision_callback(self, msg):
        """Use ML model for decision making"""
        # Preprocess sensor data
        input_data = self.preprocess_sensor_data(msg)

        # Run inference
        action = self.model.predict(input_data)

        # Publish action
        command_msg = String()
        command_msg.data = f'ML action: {action}'
        self.command_publisher.publish(command_msg)
```

## Best Practices for Python AI Agents

### 1. Error Handling

```python
def safe_robot_command(self, command):
    """Safely send command to robot with validation"""
    try:
        # Validate command before sending
        if self.validate_command(command):
            self.command_publisher.publish(command)
        else:
            self.get_logger().error('Invalid command rejected')
    except Exception as e:
        self.get_logger().error(f'Error sending command: {e}')
```

### 2. Parameter Management

```python
def __init__(self):
    super().__init__('parameterized_ai_agent')

    # Declare parameters
    self.declare_parameter('max_velocity', 1.0)
    self.declare_parameter('safety_threshold', 0.5)

    # Access parameters
    self.max_velocity = self.get_parameter('max_velocity').value
    self.safety_threshold = self.get_parameter('safety_threshold').value
```

### 3. Logging and Monitoring

```python
def log_agent_state(self):
    """Log important agent state information"""
    self.get_logger().info(
        f'Agent state - Position: {self.current_pos}, '
        f'Goal: {self.current_goal}, '
        f'Status: {self.status}'
    )
```

## Testing AI Agents

### Unit Testing

```python
import unittest
from unittest.mock import Mock
import rclpy
from rclpy.clock import Clock
from std_msgs.msg import String

class TestAIAgent(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.ai_agent = PythonAIAgent()

    def tearDown(self):
        self.ai_agent.destroy_node()
        rclpy.shutdown()

    def test_command_publishing(self):
        # Mock publisher
        self.ai_agent.command_publisher = Mock()

        # Call the method that publishes
        self.ai_agent.ai_decision_callback()

        # Verify that publish was called
        self.ai_agent.command_publisher.publish.assert_called_once()
```

## Real-World Example: Navigation AI Agent

```python
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry

class NavigationAIAgent(Node):
    def __init__(self):
        super().__init__('navigation_ai_agent')

        # Publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.odom_subscriber = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        self.goal_subscriber = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_callback,
            10
        )

        # Navigation state
        self.current_pose = None
        self.current_goal = None
        self.navigation_active = False

        # Navigation timer
        self.nav_timer = self.create_timer(0.05, self.navigation_callback)

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg):
        """Set new navigation goal"""
        self.current_goal = msg.pose
        self.navigation_active = True

    def navigation_callback(self):
        """Main navigation logic"""
        if not self.navigation_active or not self.current_pose or not self.current_goal:
            return

        # Simple proportional controller
        cmd_vel = Twist()

        # Calculate distance to goal
        dx = self.current_goal.position.x - self.current_pose.position.x
        dy = self.current_goal.position.y - self.current_pose.position.y
        distance = (dx**2 + dy**2)**0.5

        # Calculate angle to goal
        angle_to_goal = math.atan2(dy, dx)
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        angle_diff = angle_to_goal - current_yaw

        # Set velocity based on distance and angle
        if distance > 0.1:  # Threshold for reaching goal
            cmd_vel.linear.x = min(0.5, distance * 0.5)  # Forward speed
            cmd_vel.angular.z = angle_diff * 1.0  # Turn speed
        else:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.navigation_active = False

        self.cmd_vel_publisher.publish(cmd_vel)

    def get_yaw_from_quaternion(self, q):
        """Extract yaw from quaternion"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
```

## Summary

This chapter covered how to design Python AI agents that interface with ROS 2 using rclpy. We explored different patterns for AI agents, integration with machine learning, and best practices for safety and reliability. In the next chapter, we'll cover URDF design for humanoid robots.