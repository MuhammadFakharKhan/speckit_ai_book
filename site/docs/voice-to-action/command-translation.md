---
title: Command Translation to ROS 2
description: Documentation on translating parsed voice commands to ROS 2 actions and messages in VLA systems
sidebar_position: 4
tags: [vla, command-translation, ros2, robot-control, action-execution]
---

# Command Translation to ROS 2

## Overview

Command translation is the critical component that converts structured intents from the intent parsing stage into executable ROS 2 actions and messages. This component bridges the gap between natural language understanding and robot control, enabling the Vision-Language-Action (VLA) system to execute voice commands through the ROS 2 ecosystem.

## Translation Architecture

### Processing Pipeline

The command translation system follows a structured pipeline:

```
Parsed Intent → Action Mapping → Message Construction → ROS 2 Publication → Execution Monitoring
```

Each stage transforms the command from a high-level intent to specific ROS 2 messages that control the robot's behavior.

### Core Components

1. **Intent Mapper**: Maps parsed intents to ROS 2 action types
2. **Message Constructor**: Builds appropriate ROS 2 messages with parameters
3. **Action Validator**: Validates that actions are feasible given robot capabilities
4. **Message Publisher**: Publishes messages to appropriate ROS 2 topics/services
5. **Execution Monitor**: Tracks action execution and reports status

## Intent-to-Action Mapping

### Navigation Commands

Navigation commands are translated to ROS 2 Navigation actions:

```python
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from builtin_interfaces.msg import Duration

class NavigationCommandTranslator:
    def __init__(self):
        self.nav_client = None  # Navigation action client
        self.known_locations = {}  # Predefined locations

    def translate_move_to_command(self, intent_params):
        """
        Translate MoveTo intent to ROS 2 navigation command
        """
        # Look up destination in known locations
        destination = self.known_locations.get(intent_params.destination)
        if not destination:
            # Try to find destination in environment map
            destination = self.find_location_in_map(intent_params.destination)

        if not destination:
            raise ValueError(f"Unknown destination: {intent_params.destination}")

        # Create ROS 2 navigation goal
        goal = self.create_navigation_goal(destination)
        return goal

    def create_navigation_goal(self, destination):
        """
        Create a ROS 2 navigation goal from destination coordinates
        """
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = destination['x']
        goal.pose.position.y = destination['y']
        goal.pose.position.z = destination['z']
        goal.pose.orientation.w = 1.0  # Default orientation

        return goal

    def translate_move_direction_command(self, intent_params):
        """
        Translate MoveDirection intent to ROS 2 movement command
        """
        # Calculate target position based on current position and direction
        current_pos = self.get_robot_position()
        target_pos = self.calculate_target_position(
            current_pos,
            intent_params.direction,
            intent_params.distance
        )

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = target_pos['x']
        goal.pose.position.y = target_pos['y']
        goal.pose.position.z = target_pos['z']

        return goal
```

### Manipulation Commands

Manipulation commands are translated to ROS 2 manipulation actions:

```python
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import String

class ManipulationCommandTranslator:
    def __init__(self):
        self.manipulation_client = None  # Manipulation action client
        self.object_detector = None      # Object detection service

    def translate_pick_up_command(self, intent_params):
        """
        Translate PickUp intent to ROS 2 manipulation command
        """
        # Find the target object
        target_object = self.find_object_by_type_and_color(
            intent_params.object_type,
            intent_params.object_color
        )

        if not target_object:
            raise ValueError(f"Could not find object: {intent_params.object_type}")

        # Create manipulation goal for picking up
        goal = self.create_pickup_goal(target_object)
        return goal

    def create_pickup_goal(self, object_info):
        """
        Create a ROS 2 manipulation goal for picking up an object
        """
        # This would involve multiple steps: approach, grasp, lift
        goal = {
            'action': 'pickup',
            'object_pose': object_info['pose'],
            'grasp_type': 'top_grasp',  # Default grasp type
            'pre_grasp_distance': 0.1   # Distance before grasp
        }

        return goal

    def translate_place_command(self, intent_params):
        """
        Translate Place intent to ROS 2 manipulation command
        """
        # Determine placement location
        placement_location = self.known_locations.get(intent_params.destination)
        if not placement_location:
            placement_location = self.find_placement_location(intent_params.destination)

        # Create manipulation goal for placing
        goal = {
            'action': 'place',
            'target_pose': placement_location,
            'release_height': 0.1
        }

        return goal
```

### Perception Commands

Perception commands are translated to ROS 2 perception actions:

```python
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

class PerceptionCommandTranslator:
    def __init__(self):
        self.object_detection_client = None
        self.scene_understanding_client = None

    def translate_find_object_command(self, intent_params):
        """
        Translate FindObject intent to ROS 2 perception command
        """
        # Create object detection request
        request = {
            'object_type': intent_params.object_type,
            'object_color': intent_params.object_color,
            'search_area': 'current_view'  # or specific area
        }

        return request

    def translate_describe_command(self, intent_params):
        """
        Translate Describe intent to ROS 2 scene understanding command
        """
        # Create scene description request
        request = {
            'description_type': 'environmental',
            'include_objects': True,
            'include_locations': True
        }

        return request
```

## Message Construction

### Standard ROS 2 Message Types

The system constructs various ROS 2 message types based on the command:

```python
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Bool
from action_msgs.msg import GoalStatus

class MessageConstructor:
    def __init__(self):
        self.robot_capabilities = self.get_robot_capabilities()

    def construct_navigation_message(self, target_pose):
        """
        Construct navigation message for the robot
        """
        if self.robot_capabilities.get('navigation_available', False):
            # Use Navigation2 for complex navigation
            from nav2_msgs.action import NavigateToPose
            goal = NavigateToPose.Goal()
            goal.pose = target_pose
            return goal
        else:
            # Fallback to simple movement commands
            return self.construct_simple_navigation_message(target_pose)

    def construct_simple_navigation_message(self, target_pose):
        """
        Construct simple movement message as fallback
        """
        # Calculate movement command based on target
        movement_cmd = Twist()

        # Calculate linear and angular velocities based on target position
        # This is simplified - real implementation would be more sophisticated
        movement_cmd.linear.x = 0.5  # Default forward speed
        movement_cmd.angular.z = 0.2  # Default turning speed

        return movement_cmd

    def construct_manipulation_message(self, manipulation_goal):
        """
        Construct manipulation message for the robot
        """
        # This would depend on the specific manipulation system
        # For a generic manipulator:
        from control_msgs.msg import FollowJointTrajectoryGoal
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        # Create trajectory for manipulation
        trajectory = JointTrajectory()
        trajectory.joint_names = self.robot_capabilities.get('manipulator_joints', [])

        # Add trajectory points for the manipulation sequence
        point = JointTrajectoryPoint()
        # Set joint positions for grasp, lift, etc.
        trajectory.points.append(point)

        return trajectory

    def construct_perception_message(self, perception_request):
        """
        Construct perception message for the robot
        """
        # Create request for object detection service
        from vision_msgs.srv import DetectObject
        request = DetectObject.Request()
        request.object_type = perception_request['object_type']
        request.object_color = perception_request['object_color']

        return request
```

## Action Validation

### Capability Checking

Before executing commands, the system validates that the robot can perform the requested actions:

```python
class ActionValidator:
    def __init__(self):
        self.robot_capabilities = self.load_robot_capabilities()

    def validate_navigation_action(self, destination):
        """
        Validate that navigation to destination is possible
        """
        # Check if destination is within robot's operational area
        if not self.is_within_operational_area(destination):
            return False, f"Destination {destination} is outside operational area"

        # Check if path to destination is clear
        path_clear = self.check_path_clearance(destination)
        if not path_clear:
            return False, f"Path to {destination} is blocked"

        return True, "Navigation is feasible"

    def validate_manipulation_action(self, object_info):
        """
        Validate that manipulation of object is possible
        """
        # Check if object is within manipulator reach
        if not self.is_within_reach(object_info['pose']):
            return False, "Object is outside manipulator reach"

        # Check if robot has manipulation capabilities
        if not self.robot_capabilities.get('manipulation_available', False):
            return False, "Robot does not have manipulation capabilities"

        # Check if object is graspable
        if not self.is_graspable(object_info):
            return False, f"Object {object_info['type']} is not graspable"

        return True, "Manipulation is feasible"

    def validate_perception_action(self, request):
        """
        Validate that perception request can be fulfilled
        """
        # Check if robot has required sensors
        required_sensors = self.get_required_sensors(request)
        available_sensors = self.robot_capabilities.get('sensors', [])

        for sensor in required_sensors:
            if sensor not in available_sensors:
                return False, f"Missing required sensor: {sensor}"

        return True, "Perception request is feasible"

    def load_robot_capabilities(self):
        """
        Load robot capabilities from configuration
        """
        # This would typically come from robot description or service
        return {
            'navigation_available': True,
            'manipulation_available': True,
            'sensors': ['camera', 'lidar', 'imu'],
            'manipulator_joints': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'max_velocity': 1.0,
            'operational_area': {'x_min': -10, 'x_max': 10, 'y_min': -10, 'y_max': 10}
        }
```

## ROS 2 Integration

### Publisher Implementation

The system publishes commands to appropriate ROS 2 topics:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class CommandPublisher(Node):
    def __init__(self):
        super().__init__('command_publisher')

        # Publishers for different command types
        self.nav_publisher = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.manipulation_publisher = self.create_publisher(
            String,  # This would be a custom manipulation message
            '/manipulation_command',
            10
        )

        self.perception_publisher = self.create_publisher(
            String,  # This would be a custom perception request
            '/perception_request',
            10
        )

    def publish_navigation_command(self, goal_pose):
        """
        Publish navigation command to ROS 2
        """
        self.nav_publisher.publish(goal_pose)
        self.get_logger().info(f'Published navigation goal: {goal_pose}')

    def publish_movement_command(self, twist_cmd):
        """
        Publish movement command to ROS 2
        """
        self.cmd_vel_publisher.publish(twist_cmd)
        self.get_logger().info(f'Published movement command: {twist_cmd}')

    def publish_manipulation_command(self, manipulation_cmd):
        """
        Publish manipulation command to ROS 2
        """
        self.manipulation_publisher.publish(manipulation_cmd)
        self.get_logger().info(f'Published manipulation command: {manipulation_cmd}')

    def publish_perception_request(self, perception_request):
        """
        Publish perception request to ROS 2
        """
        self.perception_publisher.publish(perception_request)
        self.get_logger().info(f'Published perception request: {perception_request}')
```

### Action Client Implementation

For more complex commands, the system uses ROS 2 action clients:

```python
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from control_msgs.action import FollowJointTrajectory

class ActionCommander(Node):
    def __init__(self):
        super().__init__('action_commander')

        # Action clients for complex tasks
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        self.manipulation_client = ActionClient(
            self,
            FollowJointTrajectory,
            'manipulator_controller/follow_joint_trajectory'
        )

    def send_navigation_goal(self, target_pose):
        """
        Send navigation goal using action client
        """
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose

        # Wait for server
        self.nav_client.wait_for_server()

        # Send goal
        future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        )

        future.add_done_callback(self.navigation_done_callback)
        return future

    def navigation_feedback_callback(self, feedback_msg):
        """
        Handle navigation feedback
        """
        self.get_logger().info(
            f'Navigation feedback: {feedback_msg.feedback.distance_remaining}m remaining'
        )

    def navigation_done_callback(self, future):
        """
        Handle navigation completion
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Navigation goal rejected')
            return

        self.get_logger().info('Navigation goal accepted')
```

## Execution Monitoring

### Status Tracking

The system monitors command execution and provides feedback:

```python
from enum import Enum
from dataclasses import dataclass

class ExecutionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExecutionResult:
    status: ExecutionStatus
    success: bool
    message: str
    execution_time: float
    error_details: str = None

class ExecutionMonitor:
    def __init__(self):
        self.active_goals = {}
        self.command_history = []

    def start_monitoring(self, command_id, goal_handle):
        """
        Start monitoring execution of a command
        """
        self.active_goals[command_id] = {
            'goal_handle': goal_handle,
            'start_time': self.get_current_time(),
            'status': ExecutionStatus.PENDING
        }

    def check_execution_status(self, command_id):
        """
        Check the status of a command execution
        """
        if command_id not in self.active_goals:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                success=False,
                message=f"Command {command_id} not found",
                execution_time=0.0
            )

        goal_info = self.active_goals[command_id]
        goal_handle = goal_info['goal_handle']

        # Check if goal is still active
        if not goal_handle.is_active:
            # Goal has finished
            result = goal_handle.result()
            execution_time = self.get_current_time() - goal_info['start_time']

            if result.success:
                status = ExecutionStatus.COMPLETED
                success = True
                message = "Command completed successfully"
            else:
                status = ExecutionStatus.FAILED
                success = False
                message = f"Command failed: {result.message}"

            # Remove from active goals and add to history
            self.command_history.append({
                'command_id': command_id,
                'status': status,
                'execution_time': execution_time,
                'result': result
            })
            del self.active_goals[command_id]

            return ExecutionResult(
                status=status,
                success=success,
                message=message,
                execution_time=execution_time
            )

        # Still executing
        return ExecutionResult(
            status=ExecutionStatus.EXECUTING,
            success=False,
            message="Command still executing",
            execution_time=self.get_current_time() - goal_info['start_time']
        )

    def cancel_command(self, command_id):
        """
        Cancel an executing command
        """
        if command_id in self.active_goals:
            goal_handle = self.active_goals[command_id]['goal_handle']
            goal_handle.cancel_goal()

            self.command_history.append({
                'command_id': command_id,
                'status': ExecutionStatus.CANCELLED,
                'execution_time': self.get_current_time() - self.active_goals[command_id]['start_time'],
                'result': None
            })
            del self.active_goals[command_id]

            return ExecutionResult(
                status=ExecutionStatus.CANCELLED,
                success=False,
                message="Command cancelled",
                execution_time=self.get_current_time() - self.active_goals[command_id]['start_time']
            )
```

## Error Handling and Recovery

### Command Validation Errors

The system handles various validation and execution errors:

```python
class CommandErrorHandler:
    def __init__(self):
        self.recovery_strategies = self.load_recovery_strategies()

    def handle_translation_error(self, error, intent, original_text):
        """
        Handle errors during command translation
        """
        error_type = type(error).__name__

        if error_type == "ValueError":
            # Typically means unknown location or object
            return self.handle_unknown_reference(error, intent, original_text)
        elif error_type == "AttributeError":
            # Missing parameters or attributes
            return self.handle_missing_parameters(error, intent, original_text)
        else:
            # General error
            return self.handle_general_error(error, intent, original_text)

    def handle_unknown_reference(self, error, intent, original_text):
        """
        Handle errors due to unknown locations or objects
        """
        # Suggest alternatives or request clarification
        suggestion = f"I don't know about {str(error)}. Could you specify a known location or describe the object differently?"
        return {
            'success': False,
            'error': str(error),
            'suggestion': suggestion,
            'requires_clarification': True
        }

    def handle_missing_parameters(self, error, intent, original_text):
        """
        Handle errors due to missing command parameters
        """
        # Request missing information
        suggestion = f"I need more information to execute this command. {str(error)}"
        return {
            'success': False,
            'error': str(error),
            'suggestion': suggestion,
            'requires_clarification': True
        }

    def handle_general_error(self, error, intent, original_text):
        """
        Handle general translation errors
        """
        suggestion = f"I couldn't process the command '{original_text}'. Please try rephrasing."
        return {
            'success': False,
            'error': str(error),
            'suggestion': suggestion,
            'requires_clarification': False
        }

    def load_recovery_strategies(self):
        """
        Load predefined recovery strategies for common errors
        """
        return {
            'navigation_blocked': [
                'Try alternative route',
                'Clear path if possible',
                'Use different destination'
            ],
            'object_not_found': [
                'Search in different area',
                'Verify object description',
                'Check if object exists'
            ],
            'manipulation_failed': [
                'Try different grasp approach',
                'Check object stability',
                'Verify manipulator availability'
            ]
        }
```

## Performance Considerations

### Optimization Strategies

1. **Caching**: Cache frequently used location coordinates and object positions
2. **Batching**: Group related commands for efficient execution
3. **Prediction**: Pre-calculate likely next commands based on context
4. **Prioritization**: Prioritize safety-critical commands

### Resource Management

```python
class ResourceManager:
    def __init__(self):
        self.resource_limits = {
            'max_navigation_goals': 5,
            'max_manipulation_attempts': 3,
            'command_queue_size': 10
        }
        self.current_resources = {
            'active_navigation_goals': 0,
            'active_manipulation_goals': 0,
            'queued_commands': 0
        }

    def check_resource_availability(self, command_type):
        """
        Check if sufficient resources are available for command
        """
        if command_type == 'navigation':
            return (self.current_resources['active_navigation_goals'] <
                   self.resource_limits['max_navigation_goals'])
        elif command_type == 'manipulation':
            return (self.current_resources['active_manipulation_goals'] <
                   self.resource_limits['max_manipulation_attempts'])
        else:
            return True

    def acquire_resources(self, command_type):
        """
        Acquire resources for command execution
        """
        if command_type == 'navigation':
            self.current_resources['active_navigation_goals'] += 1
        elif command_type == 'manipulation':
            self.current_resources['active_manipulation_goals'] += 1

    def release_resources(self, command_type):
        """
        Release resources after command completion
        """
        if command_type == 'navigation':
            self.current_resources['active_navigation_goals'] -= 1
        elif command_type == 'manipulation':
            self.current_resources['active_manipulation_goals'] -= 1
```

## Security and Safety

### Safety Validation

```python
class SafetyValidator:
    def __init__(self):
        self.safety_constraints = self.load_safety_constraints()

    def validate_command_safety(self, command, context):
        """
        Validate that command is safe to execute
        """
        # Check navigation safety
        if command.get('type') == 'navigation':
            return self.validate_navigation_safety(command, context)

        # Check manipulation safety
        elif command.get('type') == 'manipulation':
            return self.validate_manipulation_safety(command, context)

        # Check other command types
        else:
            return self.validate_general_safety(command, context)

    def validate_navigation_safety(self, command, context):
        """
        Validate safety of navigation commands
        """
        destination = command.get('destination')

        # Check if destination is in safe area
        if not self.is_safe_area(destination):
            return False, f"Destination {destination} is in unsafe area"

        # Check path for obstacles
        path = self.calculate_path(context['current_position'], destination)
        if self.has_unsafe_obstacles(path):
            return False, "Path contains unsafe obstacles"

        return True, "Navigation is safe"

    def load_safety_constraints(self):
        """
        Load safety constraints from configuration
        """
        return {
            'no_go_zones': [],
            'maximum_speed': 1.0,
            'minimum_distance_to_people': 1.0,
            'safe_operational_area': {'x_min': -10, 'x_max': 10, 'y_min': -10, 'y_max': 10}
        }
```

## Best Practices

### Command Design

1. **Consistency**: Use consistent command structures across different robot types
2. **Flexibility**: Allow for robot-specific adaptations while maintaining compatibility
3. **Feedback**: Provide clear feedback about command status and execution results
4. **Graceful Degradation**: Handle unavailable capabilities gracefully

### Error Handling

1. **Specific Error Messages**: Provide clear, actionable error messages
2. **Recovery Options**: Suggest alternatives when commands fail
3. **Context Awareness**: Consider environmental context in error handling
4. **User Communication**: Clearly communicate robot state and limitations

## Troubleshooting

### Common Issues

1. **Message Type Mismatches**: Ensure message types match ROS 2 interface definitions
2. **Coordinate Frame Issues**: Verify all coordinate frames are properly defined
3. **Action Server Availability**: Check that required action servers are running
4. **Permission Issues**: Ensure proper ROS 2 permissions for command execution

### Diagnostic Tools

```python
def diagnose_command_translation(parsed_intent):
    """
    Diagnose potential issues with command translation
    """
    diagnostics = {
        'intent_type': parsed_intent.intent_type,
        'parameters': parsed_intent.parameters,
        'confidence': parsed_intent.confidence,
        'translation_possible': True,
        'issues': []
    }

    # Check for missing parameters
    if parsed_intent.intent_type in ['move_to', 'fetch'] and not parsed_intent.parameters.destination:
        diagnostics['translation_possible'] = False
        diagnostics['issues'].append('Missing destination parameter')

    # Check for robot capability match
    # Additional checks based on intent type...

    return diagnostics
```

## Future Enhancements

### Advanced Features

- **Predictive Translation**: Pre-translate likely commands based on context
- **Multi-Robot Coordination**: Coordinate commands across multiple robots
- **Learning-Based Adaptation**: Adapt translation based on execution success
- **Natural Language Feedback**: Generate natural language status updates

## Conclusion

Command translation is the essential link between high-level voice commands and low-level robot control in the VLA system. By accurately converting parsed intents into appropriate ROS 2 messages, the system enables intuitive human-robot interaction while maintaining safety and reliability.

For implementation details, refer to the complete [Voice Command Processing](./index.md) overview and continue with the [Voice-to-Action Pipeline](./index.md) documentation.