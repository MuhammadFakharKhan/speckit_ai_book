# Quickstart Guide: Module 2: The Digital Twin (Gazebo & Unity)

## Overview

This guide will help you get started with Module 2, which covers creating digital twins using Gazebo for physics simulation and Unity for visualization, integrated with ROS 2.

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Gazebo Garden (Ignition) installed
- Unity 2022.3 LTS (optional, for visualization)
- Docusaurus prerequisites (Node.js 18+, npm/pnpm)
- Basic knowledge of ROS 2 concepts (covered in Module 1)

## Setup Steps

### 1. Install Dependencies

```bash
# Install ROS 2 Humble (if not already installed)
# Follow official installation guide: https://docs.ros.org/en/humble/Installation.html

# Install Gazebo Garden
sudo apt install ros-humble-gazebo-*

# Install ros-gz bridge for ROS 2 integration
sudo apt install ros-humble-ros-gz
```

### 2. Clone and Setup the Repository

```bash
git clone <repository-url>
cd speckit_ai_book
pnpm install  # or npm install
```

### 3. Build and Run Documentation

```bash
cd site
pnpm start  # Runs the Docusaurus development server
```

### 4. Run Gazebo Simulations

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Launch a sample simulation
ros2 launch examples_gazebo simulation.launch.py
```

## Key Components

### 1. Gazebo Physics Simulation (Chapter 1)

The first chapter focuses on creating physics-based simulations in Gazebo:

- Creating world files (.sdf/.world)
- Adding robot models (URDF)
- Configuring physics properties
- Running simulations with realistic physics

### 2. Simulated Sensors (Chapter 2)

The second chapter covers sensor simulation:

- Camera sensors
- LIDAR sensors
- IMU sensors
- Force/torque sensors
- Connecting sensors to ROS 2 topics

### 3. Unity Integration (Chapter 3)

The third chapter demonstrates Unity integration:

- Visualizing robot states
- Human-robot interaction concepts
- Connecting Unity to ROS 2 via rosbridge

## Running Examples

### Example 1: Basic Humanoid Simulation

```bash
# Launch the basic humanoid simulation
ros2 launch examples_gazebo humanoid_basic.launch.py

# Control the robot using ROS 2 commands
ros2 topic pub /joint_states sensor_msgs/msg/JointState "..."
```

### Example 2: Sensor Simulation

```bash
# Launch simulation with sensors
ros2 launch examples_gazebo sensor_demo.launch.py

# View sensor data
ros2 topic echo /camera/image_raw
ros2 topic echo /lidar/scan
```

## API Endpoints

The simulation environment provides REST APIs for programmatic control:

- `GET /api/simulations` - List available simulations
- `POST /api/simulation/start` - Start a simulation
- `POST /api/simulation/stop` - Stop a simulation
- `GET /api/robot/{robot_id}/sensors` - Get robot sensor data
- `POST /api/robot/{robot_id}/joints` - Control robot joints

## Development Workflow

1. **Create a new branch**: `git checkout -b feature/module2-content`
2. **Add documentation**: Create Docusaurus pages in `docs/module2/`
3. **Add simulation files**: Place Gazebo world/models in `examples/gazebo/`
4. **Test locally**: Run `pnpm start` to preview changes
5. **Validate simulation**: Test Gazebo worlds with `gazebo <world_file>`
6. **Submit PR**: Create a pull request for review

## Troubleshooting

### Common Issues

1. **Gazebo not launching**: Ensure proper X11 forwarding if running in Docker
2. **ROS 2 topics not connecting**: Check that both ROS 2 and Gazebo are using the same RMW
3. **Documentation build errors**: Verify proper frontmatter in Markdown files

### Useful Commands

```bash
# Check ROS 2 network
ros2 topic list

# Check Gazebo status
gz topic -l

# Build documentation
cd site && pnpm build
```

## Next Steps

After completing this quickstart:
1. Proceed to Chapter 1: Gazebo Physics Simulation
2. Try the hands-on examples in each chapter
3. Experiment with different simulation profiles
4. Explore Unity visualization options