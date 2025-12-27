---
title: "Unity Integration"
sidebar_position: 3
description: "Learn how to integrate Unity with Gazebo simulation for visualization and human-robot interaction"
---

# Unity Integration for Robot Visualization

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up Unity with ROS integration using the Unity Robotics Package
- Create Unity visualizations that reflect Gazebo simulation states
- Implement human-robot interaction concepts in Unity
- Connect Unity to ROS bridge for real-time data synchronization
- Design user interfaces for robot control and monitoring

## Introduction

Unity provides a powerful platform for creating immersive visualizations and human-robot interaction interfaces. By connecting Unity to the Gazebo simulation through ROS, we can create real-time visualizations that help developers and researchers understand robot behavior and interact with the simulation in intuitive ways.

## Setting Up Unity for ROS Integration

### Unity Robotics Package

The Unity Robotics Package provides tools and components for connecting Unity to ROS networks. It includes:

- **ROS TCP Connector**: Establishes communication between Unity and ROS
- **Message conversion utilities**: Convert between ROS and Unity data types
- **Sample components**: Pre-built components for common robotics tasks
- **Documentation and examples**: Guides for common use cases

### Installation

To install the Unity Robotics Package:

1. Open Unity and create a new project or open an existing one
2. Go to Window → Package Manager
3. Click the "+" button and select "Add package from git URL..."
4. Enter the Unity Robotics Package git URL
5. The package will be downloaded and integrated into your project

### Basic Setup

After installing the Unity Robotics Package, you need to set up the ROS connection:

```csharp
using Unity.Robotics.ROSTCPConnector;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string rosBridgeIP = "127.0.0.1";
    int rosBridgePort = 9090;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosBridgeIP, rosBridgePort);
    }
}
```

## Creating Unity Visualizations

### Robot Prefab Design

A robot prefab in Unity should match the physical properties of the Gazebo model. The prefab should include:

- **Visual components**: Meshes, materials, and textures that represent the robot
- **Transform hierarchy**: Joints and links that mirror the URDF structure
- **Rigidbody components**: For physics simulation if needed
- **Joint controllers**: Scripts to control joint movements

### Scene Setup

A Unity scene for robot visualization typically includes:

- **Camera setup**: Multiple cameras for different viewpoints
- **Lighting**: Proper lighting to visualize the robot clearly
- **Environment**: Simple environment that matches the Gazebo world
- **UI elements**: Information displays and control interfaces

## Connecting Unity to ROS Bridge

### Establishing Connection

Unity connects to ROS through a WebSocket connection to rosbridge. The connection parameters include:

- **IP Address**: The address of the ROS bridge server
- **Port**: The port on which rosbridge is listening (typically 9090)
- **Message protocols**: JSON or binary message formats

### Subscribing to Robot States

To receive robot state information from ROS, Unity subscribes to topics like `/joint_states`:

```csharp
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using RosMessageTypes.Sensor;

public class JointStateSubscriber : MonoBehaviour
{
    public string topicName = "/joint_states";

    void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<JointStateMsg>(topicName, OnJointStateReceived);
    }

    void OnJointStateReceived(JointStateMsg msg)
    {
        // Update robot joints based on received state
        UpdateRobotJoints(msg);
    }

    void UpdateRobotJoints(JointStateMsg msg)
    {
        // Implementation to update Unity robot joints
    }
}
```

### Publishing Commands

Unity can also publish commands to control the robot:

```csharp
using RosMessageTypes.Std;

void SendCommand()
{
    var command = new StringMsg();
    command.data = "move_forward";

    ros.Send<string>(topicName, command.data);
}
```

## Human-Robot Interaction Concepts

### Control Interfaces

Unity provides an ideal environment for creating intuitive control interfaces:

- **Gamepad controls**: Use gamepad input to control robot movement
- **Touch interfaces**: On mobile devices, touch-based controls
- **Voice commands**: Integration with speech recognition systems
- **Gesture recognition**: Using camera input for gesture-based control

### Visualization Techniques

Effective visualization helps users understand robot state:

- **Joint highlighting**: Highlight active joints during movement
- **Path visualization**: Show planned or executed paths
- **Sensor visualization**: Display sensor data like LIDAR scans
- **Status indicators**: Show battery, connection status, etc.

## Implementing Robot State Synchronization

### Transform Synchronization

To keep Unity transforms synchronized with Gazebo:

1. Subscribe to `/tf` or `/joint_states` topics
2. Parse the transformation data
3. Apply transformations to Unity objects
4. Handle coordinate system differences (ROS uses right-handed, Unity uses left-handed)

### Coordinate System Conversion

ROS and Unity use different coordinate systems:

- **ROS**: X forward, Y left, Z up
- **Unity**: X right, Y up, Z forward

Conversion requires rotation and potential scaling adjustments:

```csharp
Vector3 RosToUnity(Vector3 rosVector)
{
    return new Vector3(rosVector.y, rosVector.z, rosVector.x);
}
```

## Unity-ROS Integration Example

Here's a complete example of a Unity script that connects to ROS and visualizes robot joints:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class RobotVisualization : MonoBehaviour
{
    [SerializeField]
    private List<Transform> jointTransforms;

    private ROSConnection ros;
    private Dictionary<string, int> jointIndexMap;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize("127.0.0.1", 9090);

        ros.Subscribe<JointStateMsg>("/joint_states", OnJointStateReceived);

        CreateJointIndexMap();
    }

    void CreateJointIndexMap()
    {
        jointIndexMap = new Dictionary<string, int>();
        for (int i = 0; i < jointTransforms.Count; i++)
        {
            jointIndexMap[jointTransforms[i].name] = i;
        }
    }

    void OnJointStateReceived(JointStateMsg msg)
    {
        for (int i = 0; i < msg.name.Count; i++)
        {
            string jointName = msg.name[i];
            if (jointIndexMap.ContainsKey(jointName))
            {
                int jointIndex = jointIndexMap[jointName];
                float jointAngle = (float)msg.position[i];

                // Apply rotation to the corresponding joint transform
                jointTransforms[jointIndex].localRotation =
                    Quaternion.Euler(0, 0, jointAngle * Mathf.Rad2Deg);
            }
        }
    }
}
```

## Best Practices

### Performance Optimization

- **LOD systems**: Use Level of Detail to reduce complexity at distance
- **Occlusion culling**: Hide objects not visible to the camera
- **Texture atlasing**: Combine multiple textures to reduce draw calls
- **Object pooling**: Reuse objects instead of creating/destroying frequently

### Network Considerations

- **Message frequency**: Balance update rate with network capacity
- **Data compression**: Send only necessary data
- **Connection reliability**: Handle disconnections gracefully
- **Latency compensation**: Account for network delays in real-time applications

## Troubleshooting Common Issues

### Connection Problems

- Verify ROS bridge is running on the correct IP and port
- Check firewall settings that might block connections
- Ensure ROS and Unity are on the same network
- Verify topic names match between systems

### Synchronization Issues

- Check for coordinate system mismatches
- Verify timing and update frequency alignment
- Validate joint naming conventions match
- Ensure units (radians vs degrees) are consistent

## Summary

This chapter covered the fundamentals of integrating Unity with Gazebo simulation for visualization and human-robot interaction. You learned how to:

1. Set up Unity with the Robotics Package for ROS integration
2. Create robot visualizations that match Gazebo models
3. Connect Unity to ROS bridge for real-time data exchange
4. Implement human-robot interaction concepts
5. Synchronize robot states between Gazebo and Unity

The Unity integration provides a powerful platform for creating intuitive interfaces and immersive visualizations that complement the physics simulation capabilities of Gazebo.

## Unity Visualization Concepts and Setup Instructions

### Setting up Unity for Robotics

1. **Install Unity Hub**: Download and install Unity Hub from the Unity website
2. **Install Unity Editor**: Use Unity Hub to install the latest LTS version of Unity
3. **Create New Project**: Create a 3D project with the name "RobotVisualization"
4. **Install Unity Robotics Package**: Follow the installation steps mentioned earlier

### Project Structure for Robotics

A well-organized Unity robotics project should have the following structure:

```
Assets/
├── Scripts/           # All C# scripts
│   ├── ROS/          # ROS connection scripts
│   ├── Robot/        # Robot control scripts
│   └── UI/           # User interface scripts
├── Prefabs/          # Robot and environment prefabs
├── Materials/        # Material definitions
├── Models/           # 3D models
├── Scenes/           # Unity scenes
├── Plugins/          # Third-party plugins
└── Resources/        # Additional resources
```

### Setting up the ROS Connection

1. **Install rosbridge_suite**: In your ROS workspace, install rosbridge
   ```bash
   sudo apt install ros-humble-rosbridge-suite
   ```

2. **Launch rosbridge**: Start the WebSocket server
   ```bash
   ros2 launch rosbridge_server rosbridge_websocket_launch.xml
   ```

3. **Configure Unity**: Set the correct IP address and port in your Unity script

### Robot Model Setup

To properly visualize a robot in Unity:

1. **Import Robot Model**: Import the robot model with the same joint structure as in Gazebo
2. **Set Up Hierarchy**: Create a hierarchy that matches the URDF structure
3. **Configure Joints**: Set up Unity joints to match the ROS joint names
4. **Add Visualization Components**: Add necessary meshes, colliders, and rigidbodies

### Coordinate System Conversion

When synchronizing between Gazebo and Unity, remember to convert between coordinate systems:

- ROS uses a right-handed coordinate system (X forward, Y left, Z up)
- Unity uses a left-handed coordinate system (X right, Y up, Z forward)

Conversion code:
```csharp
Vector3 RosToUnity(Vector3 rosVector)
{
    return new Vector3(rosVector.y, rosVector.z, rosVector.x);
}

Quaternion RosToUnity(Quaternion rosQuat)
{
    return new Quaternion(rosQuat.y, rosQuat.z, rosQuat.x, -rosQuat.w);
}
```

## Exercises

1. **Basic Connection**: Set up a simple Unity scene that connects to a ROS network and displays a message.
2. **Joint Visualization**: Create a Unity robot model that visualizes joint positions received from a simulated robot.
3. **Interactive Control**: Implement a simple UI that allows sending commands to control a simulated robot.
4. **Sensor Visualization**: Display LIDAR or camera data received from the simulated robot in Unity.