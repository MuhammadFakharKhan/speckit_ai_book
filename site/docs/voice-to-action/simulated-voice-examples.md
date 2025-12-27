---
title: Simulated Voice Command Examples with ROS 2 Outputs
description: Comprehensive examples of voice commands and their expected ROS 2 outputs in VLA systems
sidebar_position: 8
tags: [vla, voice-examples, ros2-outputs, simulation, testing]
---

# Simulated Voice Command Examples with ROS 2 Outputs

## Overview

This document provides comprehensive examples of voice commands and their expected ROS 2 outputs in the Vision-Language-Action (VLA) system. These simulated examples demonstrate how natural language voice commands are processed through the entire pipeline to generate specific ROS 2 messages for robot control.

## Example Categories

### Navigation Commands

Navigation commands instruct the robot to move to specific locations or in specific directions.

#### Example 1: Simple Navigation
**Voice Command**: "Go to the kitchen"
**Expected Processing Flow**:
```
Audio Input → Speech Recognition → "Go to the kitchen" (confidence: 0.92)
Intent Parsing → Intent: MoveTo, Parameters: {destination: "kitchen"}
Command Translation → Generate navigation goal for kitchen location
```

**Expected ROS 2 Output**:
```python
# Message: geometry_msgs/PoseStamped
{
    "header": {
        "stamp": {"sec": 1234567890, "nanosec": 123456789},
        "frame_id": "map"
    },
    "pose": {
        "position": {"x": 5.0, "y": 3.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
}

# Alternative: nav2_msgs/MoveToPose Action Goal
{
    "pose": {
        "position": {"x": 5.0, "y": 3.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    },
    "behavior_tree_id": "default"
}
```

#### Example 2: Directional Movement
**Voice Command**: "Move forward 2 meters"
**Expected Processing Flow**:
```
Audio Input → Speech Recognition → "Move forward 2 meters" (confidence: 0.88)
Intent Parsing → Intent: MoveDirection, Parameters: {direction: "forward", distance: 2.0}
Command Translation → Generate movement command for 2m forward
```

**Expected ROS 2 Output**:
```python
# Message: geometry_msgs/Twist
{
    "linear": {"x": 0.5, "y": 0.0, "z": 0.0},  # Forward movement
    "angular": {"x": 0.0, "y": 0.0, "z": 0.0}   # No rotation
}

# Duration for 2 meters at 0.5 m/s: 4 seconds
# This would typically be sent as a velocity command for a specific duration
```

#### Example 3: Complex Navigation
**Voice Command**: "Turn left and then go to the living room"
**Expected Processing Flow**:
```
Audio Input → Speech Recognition → "Turn left and then go to the living room" (confidence: 0.85)
Intent Parsing → Intent: MultiStep, Parameters: [
    {action: "turn", direction: "left"},
    {action: "move_to", destination: "living room"}
]
Command Translation → Generate sequence of navigation commands
```

**Expected ROS 2 Output**:
```python
# First: Turn left command
# Message: geometry_msgs/Twist
{
    "linear": {"x": 0.0, "y": 0.0, "z": 0.0},   # No forward movement
    "angular": {"x": 0.0, "y": 0.0, "z": 0.5}    # Left turn (positive z)
}

# After turn completion: Move to living room
# Message: geometry_msgs/PoseStamped
{
    "header": {"stamp": {"sec": 1234567894, "nanosec": 123456789}, "frame_id": "map"},
    "pose": {
        "position": {"x": -2.0, "y": 1.5, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
}
```

### Manipulation Commands

Manipulation commands instruct the robot to interact with objects in the environment.

#### Example 4: Object Grasping
**Voice Command**: "Pick up the red cup"
**Expected Processing Flow**:
```
Audio Input → Speech Recognition → "Pick up the red cup" (confidence: 0.90)
Intent Parsing → Intent: PickUp, Parameters: {object_type: "cup", color: "red"}
Command Translation → Generate manipulation sequence for red cup
```

**Expected ROS 2 Output**:
```python
# Message: control_msgs/FollowJointTrajectory
{
    "trajectory": {
        "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        "points": [
            {
                "positions": [0.1, -0.5, 1.2, 0.0, 0.8, -0.2],
                "velocities": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "time_from_start": {"sec": 2, "nanosec": 0}
            },
            {
                "positions": [0.2, -0.4, 1.1, 0.1, 0.9, -0.1],
                "velocities": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "time_from_start": {"sec": 4, "nanosec": 0}
            }
        ]
    }
}

# Also: perception request to locate the red cup
# Message: vision_msgs/Detection2DArray
{
    "header": {"stamp": {"sec": 1234567895, "nanosec": 123456789}, "frame_id": "camera_frame"},
    "detections": [
        {
            "results": [
                {
                    "id": "cup",
                    "score": 0.85,
                    "hypothesis": {"class_id": "cup", "score": 0.85}
                }
            ],
            "bbox": {"center": {"x": 320, "y": 240}, "size_x": 50, "size_y": 60}
        }
    ]
}
```

#### Example 5: Object Placement
**Voice Command**: "Put the book on the table"
**Expected Processing Flow**:
```
Audio Input → Speech Recognition → "Put the book on the table" (confidence: 0.87)
Intent Parsing → Intent: Place, Parameters: {object: "book", destination: "table"}
Command Translation → Generate placement sequence for book on table
```

**Expected ROS 2 Output**:
```python
# Message: control_msgs/FollowJointTrajectory
{
    "trajectory": {
        "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        "points": [
            {
                "positions": [0.5, 0.2, 0.8, -0.3, 0.4, 0.1],
                "velocities": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "time_from_start": {"sec": 3, "nanosec": 0}
            }
        ]
    }
}

# Message: std_msgs/String (for gripper control)
{
    "data": "release_object"
}
```

### Perception Commands

Perception commands request environmental information from the robot.

#### Example 6: Object Detection
**Voice Command**: "What objects do you see?"
**Expected Processing Flow**:
```
Audio Input → Speech Recognition → "What objects do you see?" (confidence: 0.91)
Intent Parsing → Intent: Describe, Parameters: {request_type: "environmental"}
Command Translation → Generate scene description request
```

**Expected ROS 2 Output**:
```python
# Message: vision_msgs/Detection2DArray
{
    "header": {"stamp": {"sec": 1234567896, "nanosec": 123456789}, "frame_id": "camera_frame"},
    "detections": [
        {
            "results": [
                {
                    "id": "cup",
                    "score": 0.88,
                    "hypothesis": {"class_id": "cup", "score": 0.88}
                }
            ],
            "bbox": {"center": {"x": 100, "y": 150}, "size_x": 40, "size_y": 50}
        },
        {
            "results": [
                {
                    "id": "book",
                    "score": 0.75,
                    "hypothesis": {"class_id": "book", "score": 0.75}
                }
            ],
            "bbox": {"center": {"x": 200, "y": 300}, "size_x": 80, "size_y": 100}
        }
    ]
}

# Message: std_msgs/String (for voice response)
{
    "data": "I see a red cup and a book in front of me."
}
```

#### Example 7: Specific Object Search
**Voice Command**: "Find the blue ball"
**Expected Processing Flow**:
```
Audio Input → Speech Recognition → "Find the blue ball" (confidence: 0.89)
Intent Parsing → Intent: FindObject, Parameters: {object_type: "ball", color: "blue"}
Command Translation → Generate object search sequence
```

**Expected ROS 2 Output**:
```python
# Message: vision_msgs/Detection2DArray
{
    "header": {"stamp": {"sec": 1234567897, "nanosec": 123456789}, "frame_id": "camera_frame"},
    "detections": [
        {
            "results": [
                {
                    "id": "ball",
                    "score": 0.92,
                    "hypothesis": {"class_id": "ball", "score": 0.92}
                }
            ],
            "bbox": {"center": {"x": 400, "y": 200}, "size_x": 60, "size_y": 60}
        }
    ]
}

# Message: geometry_msgs/PoseStamped (if navigation needed to approach object)
{
    "header": {"stamp": {"sec": 1234567898, "nanosec": 123456789}, "frame_id": "map"},
    "pose": {
        "position": {"x": 1.5, "y": 2.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
}
```

### Complex Multi-Step Commands

Complex commands involve multiple actions that must be coordinated.

#### Example 8: Fetch Task
**Voice Command**: "Go to the kitchen and bring me the coffee"
**Expected Processing Flow**:
```
Audio Input → Speech Recognition → "Go to the kitchen and bring me the coffee" (confidence: 0.86)
Intent Parsing → Intent: Fetch, Parameters: {destination: "kitchen", object: "coffee"}
Command Translation → Generate multi-step sequence: navigate → locate → grasp → return
```

**Expected ROS 2 Output Sequence**:
```python
# Step 1: Navigate to kitchen
# Message: geometry_msgs/PoseStamped
{
    "header": {"stamp": {"sec": 1234567899, "nanosec": 123456789}, "frame_id": "map"},
    "pose": {
        "position": {"x": 5.0, "y": 3.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
}

# Step 2: Locate coffee (perception request)
# Message: vision_msgs/Detection2DArray
{
    "header": {"stamp": {"sec": 1234567900, "nanosec": 123456789}, "frame_id": "camera_frame"},
    "detections": [
        {
            "results": [
                {
                    "id": "coffee",
                    "score": 0.84,
                    "hypothesis": {"class_id": "coffee", "score": 0.84}
                }
            ],
            "bbox": {"center": {"x": 250, "y": 180}, "size_x": 45, "size_y": 55}
        }
    ]
}

# Step 3: Grasp coffee
# Message: control_msgs/FollowJointTrajectory
{
    "trajectory": {
        "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        "points": [
            {
                "positions": [0.3, -0.2, 0.9, 0.1, 0.6, 0.0],
                "velocities": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "time_from_start": {"sec": 2, "nanosec": 0}
            }
        ]
    }
}

# Step 4: Return to user
# Message: geometry_msgs/PoseStamped
{
    "header": {"stamp": {"sec": 1234567902, "nanosec": 123456789}, "frame_id": "map"},
    "pose": {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
}
```

#### Example 9: Complex Navigation
**Voice Command**: "Move to the table, turn around, and come back"
**Expected Processing Flow**:
```
Audio Input → Speech Recognition → "Move to the table, turn around, and come back" (confidence: 0.83)
Intent Parsing → Intent: MultiStep, Parameters: [
    {action: "move_to", destination: "table"},
    {action: "turn", degrees: 180},
    {action: "move_to", destination: "original_position"}
]
Command Translation → Generate coordinated navigation sequence
```

**Expected ROS 2 Output Sequence**:
```python
# Step 1: Move to table
# Message: geometry_msgs/PoseStamped
{
    "header": {"stamp": {"sec": 1234567903, "nanosec": 123456789}, "frame_id": "map"},
    "pose": {
        "position": {"x": 1.0, "y": 0.5, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
}

# Step 2: Turn around (180 degrees)
# Message: geometry_msgs/Twist
{
    "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
    "angular": {"x": 0.0, "y": 0.0, "z": 1.0}  # Rotate at 1.0 rad/s
}

# Step 3: Return to starting position
# Message: geometry_msgs/PoseStamped
{
    "header": {"stamp": {"sec": 1234567906, "nanosec": 123456789}, "frame_id": "map"},
    "pose": {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
}
```

## Simulation Environment Setup

### Example Simulation Configuration

To properly test these voice commands in simulation, the following configuration is typically used:

```yaml
# simulation_config.yaml
simulation_environment:
  world: "vla_test_world"
  robot_model: "humanoid_robot"
  sensor_configurations:
    camera:
      resolution: [640, 480]
      fov: 60
      frame_id: "camera_frame"
    lidar:
      range: 10.0
      resolution: 0.01
      frame_id: "lidar_frame"
  navigation_map:
    resolution: 0.05
    origin: [-10, -10, 0]
    size: [20, 20]
  object_locations:
    kitchen: [5.0, 3.0, 0.0]
    living_room: [-2.0, 1.5, 0.0]
    table: [1.0, 0.5, 0.0]
    red_cup: [1.2, 0.6, 0.8]
    book: [0.8, 0.4, 0.85]
    blue_ball: [2.5, 1.2, 0.5]
```

### Example ROS 2 Launch File

```xml
<!-- vla_simulation.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # Navigation stack
        Node(
            package='nav2_bringup',
            executable='nav2_launch.py',
            name='navigation',
            parameters=[
                os.path.join(get_package_share_directory('vla_simulation'), 'config', 'nav2_params.yaml')
            ]
        ),

        # Voice processing node
        Node(
            package='vla_voice_processing',
            executable='voice_processor',
            name='voice_processor',
            parameters=[
                os.path.join(get_package_share_directory('vla_simulation'), 'config', 'voice_params.yaml')
            ]
        ),

        # Perception stack
        Node(
            package='vla_perception',
            executable='object_detector',
            name='object_detector'
        ),

        # Manipulation controller
        Node(
            package='vla_manipulation',
            executable='arm_controller',
            name='arm_controller'
        )
    ])
```

## Testing and Validation

### Unit Test Examples

Here are examples of how to test these voice command simulations:

```python
import unittest
from vla_voice_processing.voice_processor import VoiceProcessor
from vla_command_translator.command_translator import CommandTranslator
import json

class TestVoiceCommandExamples(unittest.TestCase):
    def setUp(self):
        self.voice_processor = VoiceProcessor()
        self.command_translator = CommandTranslator()
        self.test_context = {
            'known_locations': {
                'kitchen': {'x': 5.0, 'y': 3.0, 'z': 0.0},
                'living_room': {'x': -2.0, 'y': 1.5, 'z': 0.0},
                'table': {'x': 1.0, 'y': 0.5, 'z': 0.0}
            },
            'visible_objects': [
                {'type': 'cup', 'color': 'red', 'pose': {'x': 1.2, 'y': 0.6, 'z': 0.8}},
                {'type': 'book', 'color': 'blue', 'pose': {'x': 0.8, 'y': 0.4, 'z': 0.85}}
            ]
        }

    def test_simple_navigation(self):
        """Test 'Go to the kitchen' command"""
        audio_input = self.create_mock_audio("Go to the kitchen")
        stt_result = self.voice_processor.transcribe_audio(audio_input)

        self.assertGreater(stt_result['confidence'], 0.8)
        self.assertIn("kitchen", stt_result['text'].lower())

        intent = self.voice_processor.parse_intent(stt_result['text'])
        self.assertEqual(intent.intent_type.value, 'move_to')
        self.assertEqual(intent.parameters.destination, 'kitchen')

        ros2_commands = self.command_translator.translate_to_ros2(intent, self.test_context)
        self.assertEqual(len(ros2_commands), 1)
        self.assertEqual(ros2_commands[0]['type'], 'geometry_msgs/PoseStamped')
        self.assertEqual(ros2_commands[0]['data']['position']['x'], 5.0)

    def test_manipulation_command(self):
        """Test 'Pick up the red cup' command"""
        audio_input = self.create_mock_audio("Pick up the red cup")
        stt_result = self.voice_processor.transcribe_audio(audio_input)

        self.assertGreater(stt_result['confidence'], 0.8)
        self.assertIn("pick up", stt_result['text'].lower())
        self.assertIn("cup", stt_result['text'].lower())

        intent = self.voice_processor.parse_intent(stt_result['text'])
        self.assertEqual(intent.intent_type.value, 'pick_up')
        self.assertEqual(intent.parameters.object_type, 'cup')
        self.assertEqual(intent.parameters.object_color, 'red')

        ros2_commands = self.command_translator.translate_to_ros2(intent, self.test_context)
        # Should generate perception request and manipulation commands
        self.assertGreater(len(ros2_commands), 1)
        command_types = [cmd['type'] for cmd in ros2_commands]
        self.assertIn('vision_msgs/Detection2DArray', command_types)

    def create_mock_audio(self, text):
        """Create mock audio data for testing"""
        # In real tests, this would generate audio from text
        # For simulation, we can use text directly
        return text.encode('utf-8')

if __name__ == '__main__':
    unittest.main()
```

### Integration Test Examples

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection2DArray
import time

class IntegrationTestNode(Node):
    def __init__(self):
        super().__init__('vla_integration_test')

        # Publishers for sending commands
        self.voice_command_pub = self.create_publisher(String, '/voice_commands', 10)

        # Subscribers for monitoring outputs
        self.nav_goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.nav_goal_callback, 10
        )
        self.perception_sub = self.create_subscription(
            Detection2DArray, '/object_detections', self.perception_callback, 10
        )

        self.nav_goals_received = []
        self.perception_received = []

    def nav_goal_callback(self, msg):
        self.nav_goals_received.append(msg)

    def perception_callback(self, msg):
        self.perception_received.append(msg)

    def test_navigation_command(self):
        """Test end-to-end navigation command"""
        # Clear previous results
        self.nav_goals_received.clear()

        # Send voice command
        cmd_msg = String()
        cmd_msg.data = "Go to the kitchen"
        self.voice_command_pub.publish(cmd_msg)

        # Wait for processing
        time.sleep(2.0)

        # Verify navigation goal was generated
        self.assertEqual(len(self.nav_goals_received), 1)
        goal = self.nav_goals_received[0]
        self.assertAlmostEqual(goal.pose.position.x, 5.0, places=1)
        self.assertAlmostEqual(goal.pose.position.y, 3.0, places=1)

def main():
    rclpy.init()
    test_node = IntegrationTestNode()

    # Run tests
    test_node.test_navigation_command()

    rclpy.spin_once(test_node, timeout_sec=0.1)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Metrics

### Quality Metrics for Voice Command Processing

```python
class VoiceCommandMetrics:
    def __init__(self):
        self.command_success_rate = 0.0
        self.average_confidence = 0.0
        self.average_processing_time = 0.0
        self.command_accuracy = 0.0

    def calculate_metrics(self, test_results):
        """
        Calculate performance metrics from test results
        """
        if not test_results:
            return {}

        successful_commands = sum(1 for result in test_results if result.get('success', False))
        total_commands = len(test_results)

        if total_commands > 0:
            self.command_success_rate = successful_commands / total_commands
            self.average_confidence = sum(
                result.get('confidence', 0.0) for result in test_results
            ) / total_commands
            self.average_processing_time = sum(
                result.get('processing_time', 0.0) for result in test_results
            ) / total_commands

        # Calculate accuracy based on expected vs actual ROS 2 outputs
        correct_commands = sum(1 for result in test_results if result.get('output_correct', False))
        self.command_accuracy = correct_commands / total_commands if total_commands > 0 else 0.0

        return {
            'command_success_rate': self.command_success_rate,
            'average_confidence': self.average_confidence,
            'average_processing_time': self.average_processing_time,
            'command_accuracy': self.command_accuracy
        }

    def expected_metrics(self):
        """
        Expected performance metrics for the VLA system
        """
        return {
            'command_success_rate': 0.90,  # 90% success rate
            'average_confidence': 0.85,    # 85% average confidence
            'average_processing_time': 0.5, # 500ms average processing time
            'command_accuracy': 0.88       # 88% accuracy in ROS 2 output
        }
```

## Error Handling Examples

### Common Error Scenarios

```python
class ErrorHandlingExamples:
    def __init__(self):
        self.error_scenarios = {
            'unknown_location': {
                'command': "Go to the garage",
                'expected_error': "Unknown destination: garage",
                'recovery_action': "Request clarification or suggest alternatives"
            },
            'object_not_found': {
                'command': "Pick up the purple elephant",
                'expected_error': "Could not find object: purple elephant",
                'recovery_action': "Inform user that object is not visible"
            },
            'low_confidence': {
                'command': "Go to the kichen",  # Intentional typo
                'expected_error': "Low confidence recognition",
                'recovery_action': "Request repetition or confirmation"
            },
            'robot_busy': {
                'command': "Move forward",
                'expected_error': "Robot is currently executing another command",
                'recovery_action': "Queue command or inform user of delay"
            }
        }

    def demonstrate_error_handling(self, scenario_name):
        """
        Demonstrate how the system handles specific error scenarios
        """
        scenario = self.error_scenarios.get(scenario_name)
        if not scenario:
            return "Unknown scenario"

        # Simulate the error scenario
        print(f"Simulating scenario: {scenario_name}")
        print(f"Command: {scenario['command']}")
        print(f"Expected error: {scenario['expected_error']}")
        print(f"Recovery action: {scenario['recovery_action']}")

        return {
            'scenario': scenario_name,
            'command': scenario['command'],
            'error_handled': True,
            'recovery_suggested': scenario['recovery_action']
        }
```

## Advanced Examples

### Context-Aware Commands

```python
# Example of context-aware command processing
context_aware_examples = [
    {
        'command': "Do the same thing again",
        'context': {
            'previous_command': "Go to the kitchen",
            'previous_result': "Successfully navigated to kitchen at [5.0, 3.0, 0.0]"
        },
        'expected_behavior': "Navigate to kitchen again",
        'ros2_output': {
            'type': 'geometry_msgs/PoseStamped',
            'data': {
                'position': {'x': 5.0, 'y': 3.0, 'z': 0.0},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
            }
        }
    },
    {
        'command': "What about the other one?",
        'context': {
            'previous_command': "Find the red ball",
            'previous_result': "Found red ball at [2.5, 1.2, 0.5]",
            'environment': "Two red balls visible: [2.5, 1.2, 0.5] and [3.2, 0.8, 0.5]"
        },
        'expected_behavior': "Locate the second red ball",
        'ros2_output': {
            'type': 'vision_msgs/Detection2DArray',
            'data': {
                'detections': [
                    {
                        'bbox': {'center': {'x': 350, 'y': 220}, 'size_x': 60, 'size_y': 60},
                        'results': [{'class_id': 'ball', 'score': 0.89}]
                    }
                ]
            }
        }
    }
]
```

### Multi-Modal Integration

```python
# Example of commands that integrate voice with other modalities
multi_modal_examples = [
    {
        'command': "Go to where I'm pointing",
        'combined_input': {
            'voice': "Go to where I'm pointing",
            'vision': "Person pointing direction vector [0.7, 0.2, 0.0]"
        },
        'expected_behavior': "Navigate in the direction the person is pointing",
        'ros2_output': {
            'type': 'geometry_msgs/PoseStamped',
            'data': {
                'position': {'x': 3.5, 'y': 1.2, 'z': 0.0},  # Calculated from pointing direction
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
            }
        }
    },
    {
        'command': "Pick up what I showed you",
        'combined_input': {
            'voice': "Pick up what I showed you",
            'vision': "Object highlighted/pointed at: blue cup at [1.8, 0.9, 0.8]"
        },
        'expected_behavior': "Grasp the blue cup that was highlighted",
        'ros2_output': [
            {
                'type': 'control_msgs/FollowJointTrajectory',
                'data': {
                    'trajectory': {
                        'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
                        'points': [
                            {
                                'positions': [0.4, -0.1, 0.8, 0.2, 0.7, -0.1],
                                'time_from_start': {'sec': 2, 'nanosec': 0}
                            }
                        ]
                    }
                }
            }
        ]
    }
]
```

## Best Practices for Simulation

### 1. Realistic Audio Simulation
- Use audio samples with realistic background noise
- Simulate different acoustic environments (quiet, noisy, reverberant)
- Include variations in speaker accents and speech patterns

### 2. Comprehensive Testing
- Test with various command formulations for the same action
- Include edge cases and ambiguous commands
- Test error recovery scenarios
- Validate ROS 2 message formats and contents

### 3. Performance Monitoring
- Monitor processing time for each command
- Track success rates under different conditions
- Measure resource usage during processing
- Validate that safety constraints are maintained

### 4. Iterative Improvement
- Use simulation results to improve recognition models
- Refine intent parsing based on common misinterpretations
- Optimize ROS 2 command generation for efficiency
- Update validation parameters based on real-world performance

## Troubleshooting Common Issues

### Issue 1: Low Recognition Accuracy
**Symptoms**: Commands frequently misrecognized or confidence scores low
**Solutions**:
- Check audio preprocessing parameters
- Verify Whisper model configuration
- Adjust confidence thresholds appropriately
- Consider environmental noise factors

### Issue 2: Incorrect ROS 2 Message Generation
**Symptoms**: Generated messages don't match expected robot interfaces
**Solutions**:
- Verify message type definitions match ROS 2 interfaces
- Check coordinate frame conventions
- Validate message field requirements
- Test with actual robot simulation environment

### Issue 3: Context Confusion
**Symptoms**: Commands misinterpreted due to context issues
**Solutions**:
- Improve context tracking mechanisms
- Add disambiguation prompts
- Enhance entity resolution algorithms
- Validate context updates between commands

## Conclusion

These simulated voice command examples provide a comprehensive reference for testing and validating the VLA system's voice processing capabilities. The examples cover various command types, complexity levels, and error scenarios that the system should handle. By implementing and testing these examples in simulation, developers can ensure that the voice-to-ROS 2 translation pipeline functions correctly and reliably.

The examples demonstrate the complete flow from natural language input to specific ROS 2 messages, showing how the system processes different types of commands and generates appropriate robot control commands. This comprehensive set of examples serves as both a testing framework and a reference for expected system behavior.

For implementation details, refer to the complete [Voice Command Processing](./index.md) overview and continue with the [Voice-to-Action Pipeline](./index.md) documentation.