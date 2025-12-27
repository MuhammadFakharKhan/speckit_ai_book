---
title: Complete Example Workflow
sidebar_label: Complete Workflow
description: Complete example workflow demonstrating the full VLA pipeline from voice command to robotic action execution
---

# Complete Example Workflow

## Overview

This document provides a comprehensive example of the complete Vision-Language-Action (VLA) pipeline, demonstrating how voice commands are processed through cognitive planning to execute robotic actions in simulated environments.

## Complete Workflow Example

### Scenario: Fetch Object Task

**Voice Command**: "Please go to the living room and bring me the blue book from the shelf."

#### Step 1: Voice Recognition and Processing

```
Input: "Please go to the living room and bring me the blue book from the shelf."
→ Whisper Processing
→ Intent: Navigation + Object Detection + Manipulation
→ Entities: Location=living room, Object=blue book, Action=fetch
→ Confidence Score: 0.85
```

#### Step 2: Cognitive Planning

The LLM processes the command and generates an action sequence:

```
1. Navigate to living room
   - Identify living room location in map
   - Plan path to destination
   - Execute navigation

2. Locate blue book
   - Identify shelf location
   - Use perception system to detect blue objects
   - Verify book characteristics

3. Grasp blue book
   - Position manipulator appropriately
   - Execute grasping motion
   - Verify successful grasp

4. Return to origin
   - Plan return path
   - Execute navigation with object
   - Deliver object to user
```

#### Step 3: ROS 2 Action Execution

Each action is converted to ROS 2 messages and executed:

```python
# Navigation action
navigation_goal = NavigationGoal()
navigation_goal.target_pose = living_room_location
navigation_client.send_goal(navigation_goal)

# Object detection action
detection_request = DetectionRequest()
detection_request.object_class = "book"
detection_request.color = "blue"
detection_client.send_request(detection_request)

# Manipulation action
manipulation_goal = ManipulationGoal()
manipulation_goal.action = "grasp"
manipulation_goal.object_pose = detected_book_pose
manipulation_client.send_goal(manipulation_goal)
```

#### Step 4: Execution Monitoring

The system monitors execution and handles potential issues:

- **Navigation success**: Proceed to object detection
- **Object not found**: Search alternative locations
- **Grasp failure**: Adjust grasp strategy
- **Path blocked**: Recalculate route

## Complex Multi-Step Example

### Scenario: Inspection Task

**Voice Command**: "Check if the window in the bedroom is closed, and if it's open, please close it and report back."

#### Complete Pipeline Execution:

1. **Voice Processing**
   ```
   Intent: Inspection + Conditional Action + Reporting
   Entities: Location=bedroom, Object=window, Action=close, Condition=is open
   ```

2. **Cognitive Planning**
   - Navigate to bedroom
   - Inspect window state
   - Conditionally execute closure
   - Report results

3. **Execution Sequence**
   ```yaml
   - action: navigate
     target: bedroom
   - action: inspect
     target: window
     property: open_state
   - action: conditional
     condition: window_open == true
     then:
       - action: manipulate
         target: window
         operation: close
   - action: report
     message: "Window in bedroom was {{window_state}} and is now closed"
   ```

## Error Handling and Recovery

### Voice Recognition Errors

- **Low confidence**: Request repetition or clarification
- **Ambiguous command**: Ask for clarification
- **Background noise**: Retry with noise reduction

### Planning Errors

- **Infeasible task**: Break down into simpler steps
- **Missing context**: Request additional information
- **Planning timeout**: Execute partial plan with safety measures

### Execution Errors

- **Navigation failure**: Recalculate path or abort
- **Object detection failure**: Expand search area
- **Manipulation failure**: Try alternative approach

## Validation and Testing

### Simulation Testing

All workflows are validated in simulation before real-world deployment:

```bash
# Run complete workflow in simulation
ros2 launch vla_pipeline complete_demo.launch.py

# Monitor execution
ros2 bag record /vla_pipeline/status /vla_pipeline/debug

# Validate results
python3 validate_workflow.py --scenario fetch_object
```

### Performance Metrics

- **Success rate**: Percentage of tasks completed successfully
- **Execution time**: Total time from command to completion
- **Accuracy**: Correctness of action execution
- **Robustness**: Handling of edge cases and errors

## Integration with Isaac Ecosystem

The complete workflow integrates seamlessly with the Isaac ecosystem:

- **Isaac Sim**: Provides simulation environment for testing
- **Isaac ROS**: Handles perception and sensor processing
- **Nav2**: Manages navigation and path planning for humanoid robots

This complete example demonstrates the end-to-end capability of the VLA pipeline, from natural language understanding to robotic action execution.