# AI-Driven Technical Book with Embedded RAG Chatbot

This repository contains an educational book about ROS 2 (Robot Operating System 2) for humanoid robotics, built with Docusaurus. The book focuses on ROS 2 fundamentals, Python agents with rclpy, URDF for humanoid robots, and digital twin simulation with Gazebo and Unity, with future RAG (Retrieval-Augmented Generation) chatbot integration.

## Structure

- `site/` - Docusaurus static site with documentation content
- `examples/` - ROS 2 example code and URDF models
- `specs/` - Specification files for different modules
- `src/` - Source code for simulation components and APIs
- `prompts/` - Claude Code prompts and outputs
- `history/` - Prompt History Records
- `.specify/` - Spec-Kit Plus configuration

## Modules

### Module 1: The Robotic Nervous System (ROS 2)
Covers ROS 2 fundamentals, Python agents with rclpy, and URDF for humanoid robots.

### Module 2: The Digital Twin (Gazebo & Unity)
Comprehensive coverage of digital twin simulation using:
- **Gazebo Garden** for physics simulation and sensor modeling
- **Unity** for visualization and human-robot interaction
- **ROS 2 Integration** via ros-gz bridge and rosbridge
- **API Control** for programmatic simulation management
- **Simulation Profiles** for different use cases (education, performance, high-fidelity)

## Getting Started

1. Install dependencies:
   ```bash
   pnpm install
   ```

2. Navigate to the site directory and start the development server:
   ```bash
   cd site
   pnpm start
   ```

3. For simulation development, also install ROS 2 dependencies:
   ```bash
   # Source ROS 2
   source /opt/ros/humble/setup.bash

   # Install Gazebo packages
   sudo apt install ros-humble-gazebo-*
   sudo apt install ros-humble-ros-gz
   sudo apt install ros-humble-rosbridge-suite
   ```

## Prerequisites

- Node.js 18+ with pnpm package manager
- Python 3.8+ with ROS 2 Humble Hawksbill
- Gazebo Garden (Ignition) for physics simulation
- Unity 2022.3 LTS for visualization (optional)
- Git for version control

## Module 2 Quick Start

To get started with Module 2 (Digital Twin Simulation):

1. **Start the API server**:
   ```bash
   cd src/api
   python profile_api.py
   ```

2. **Launch Gazebo simulation**:
   ```bash
   gz sim -r examples/gazebo/worlds/basic_humanoid.sdf
   ```

3. **Run the comprehensive example**:
   ```bash
   cd examples
   python comprehensive_simulation_example.py
   ```

4. **Access documentation**:
   - Full documentation: `site/docs/module2/`
   - Quickstart guide: `site/docs/module2/quickstart.md`
   - API reference: `site/docs/module2/api-reference.md`

## Key Features

- **Physics Simulation**: Realistic physics using Gazebo Garden with configurable parameters
- **Sensor Simulation**: Camera, LIDAR, IMU, and other sensor models with ROS 2 integration
- **Unity Visualization**: Real-time visualization and human-robot interaction in Unity
- **API Control**: REST API for programmatic simulation management
- **Simulation Profiles**: Pre-configured profiles for different use cases
- **Quality Assurance**: Comprehensive testing and validation tools
- **Performance Optimization**: Tools for optimizing simulation performance

## License

[License information will be added here]