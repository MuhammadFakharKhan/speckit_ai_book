# Module 2 Quickstart Guide: Digital Twin Simulation

Get started quickly with Gazebo-Unity digital twin simulation for humanoid robots. This guide will help you set up, configure, and run your first simulation in under 30 minutes.

## Prerequisites

Before starting, ensure you have:

- **ROS 2 Humble Hawksbill** installed and sourced
- **Gazebo Garden (Ignition)** installed
- **Unity 2022.3 LTS** installed (for visualization)
- **Python 3.8+** with pip
- **Node.js 18+** (for Docusaurus documentation)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd speckit_ai_book
```

### 2. Install Python Dependencies

```bash
cd src
pip install flask requests numpy
```

### 3. Install ROS 2 Packages

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Install Gazebo packages
sudo apt install ros-humble-gazebo-*
sudo apt install ros-humble-ros-gz
sudo apt install ros-humble-rosbridge-suite
```

### 4. Install Unity Robotics Package

1. Open Unity Hub and create a new 3D project
2. Go to Window → Package Manager
3. Click the "+" button → "Add package from git URL..."
4. Enter: `com.unity.robotics.ros-tcp-connector`
5. Install the package

## Quick Setup

### 1. Start the API Server

```bash
cd src/api
python profile_api.py
```

The API server will start on `http://localhost:5001`.

### 2. Launch Gazebo Simulation

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Launch the basic humanoid simulation
gz sim -r examples/gazebo/worlds/basic_humanoid.sdf
```

### 3. Verify API Connection

```bash
# Get available simulation profiles
curl -X GET http://localhost:5001/api/profiles

# Apply education profile
curl -X POST http://localhost:5001/api/profiles/education/apply
```

## Your First Simulation

### 1. Run the Comprehensive Example

```bash
cd examples
python comprehensive_simulation_example.py
```

This will:
- Start the API server
- Apply the education profile
- Run a sample simulation loop
- Demonstrate component integration

### 2. Control Robot Joints

Use the API to control robot joints:

```bash
# Send joint commands via ROS 2 (in another terminal)
source /opt/ros/humble/setup.bash
ros2 topic pub /joint_states sensor_msgs/msg/JointState "{
  name: ['hip_joint', 'knee_joint', 'ankle_joint']
  position: [0.5, -0.3, 0.2]
}"
```

### 3. Check Sensor Data

Monitor sensor data from the simulation:

```bash
# Monitor camera data
ros2 topic echo /camera/image_raw

# Monitor LIDAR data
ros2 topic echo /lidar/scan

# Monitor IMU data
ros2 topic echo /imu/data
```

## Unity Integration

### 1. Set Up ROS Bridge

```bash
# Start ROS bridge in another terminal
source /opt/ros/humble/setup.bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

### 2. Configure Unity Connection

In Unity, create a script with the following connection setup:

```csharp
using Unity.Robotics.ROSTCPConnector;

public class RobotController : MonoBehaviour
{
    void Start()
    {
        var ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize("127.0.0.1", 9090);  // ROS bridge IP and port
    }
}
```

### 3. Subscribe to Joint States

```csharp
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;

public class JointStateSubscriber : MonoBehaviour
{
    void Start()
    {
        ROSConnection.GetOrCreateInstance()
            .Subscribe<JointStateMsg>("/joint_states", OnJointStateReceived);
    }

    void OnJointStateReceived(JointStateMsg msg)
    {
        // Update Unity robot joints based on received state
        UpdateRobotJoints(msg);
    }
}
```

## API Quick Reference

### Simulation Profiles

```bash
# Get all profiles
curl -X GET http://localhost:5001/api/profiles

# Get specific profile
curl -X GET http://localhost:5001/api/profiles/education

# Apply profile
curl -X POST http://localhost:5001/api/profiles/education/apply
```

### Robot Control

```bash
# Get robot joint states
curl -X GET http://localhost:5001/api/robot/humanoid/joints

# Control robot joints
curl -X POST http://localhost:5001/api/robot/humanoid/joints \
  -H "Content-Type: application/json" \
  -d '{"joint_commands": {"hip_joint": 0.5, "knee_joint": -0.3}}'
```

## Running Examples

### Physics Simulation Example

```bash
cd examples/gazebo
python basic_physics_example.py
```

### Sensor Simulation Example

```bash
cd examples/gazebo
python sensor_demo.py
```

### Unity Integration Example

1. Start ROS bridge: `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`
2. Run Unity scene with ROS connection
3. Verify joint state synchronization

## Troubleshooting Quick Fixes

### Gazebo Won't Start

```bash
# Check Gazebo installation
gz --versions

# Try with software rendering
export LIBGL_ALWAYS_SOFTWARE=1
gz sim -r examples/gazebo/worlds/basic_humanoid.sdf
```

### ROS Bridge Connection Issues

```bash
# Check if bridge is running
ros2 node list | grep bridge

# Check topic availability
ros2 topic list
```

### API Server Not Responding

```bash
# Check if server is running
netstat -tuln | grep 5001

# Restart the API server
cd src/api
python profile_api.py
```

## Next Steps

After completing this quickstart:

1. **Explore Chapter 1**: Learn about Gazebo physics simulation in depth
2. **Try Chapter 2**: Experiment with sensor simulation and processing
3. **Advance to Chapter 3**: Implement Unity visualization and interaction
4. **Complete exercises**: Practice with hands-on exercises in `exercises.md`
5. **Build your own simulation**: Apply concepts to create custom scenarios

## Performance Tips

- Start with the "education" profile for balanced performance
- Use smaller time steps for accuracy, larger for performance
- Simplify collision geometry for better performance
- Monitor real-time factor (target: 1.0 for real-time simulation)

## Getting Help

- Check the [troubleshooting guide](./troubleshooting.md) for common issues
- Review the [API reference](./api-reference.md) for detailed endpoints
- Look at example implementations in the `examples/` directory
- Consult the [performance optimization guide](./performance-optimization.md)

You're now ready to explore the full capabilities of digital twin simulation with Gazebo and Unity! 🚀