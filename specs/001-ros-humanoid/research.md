# Research Summary: Module 2: The Digital Twin (Gazebo & Unity)

## Key Decision: Gazebo Classic vs Gazebo (Ignition)

### Decision: Use Gazebo Garden (Ignition) over Gazebo Classic
**Rationale:** Gazebo Garden (Ignition) is the actively maintained version with ongoing development, while Gazebo Classic has been deprecated. It offers better ROS 2 integration, improved performance, and modern rendering capabilities.

**Alternatives considered:**
- Gazebo Classic: Stable but no longer maintained, lacks modern features
- Ignition Fortress: Stable but older than Garden
- Custom simulation: Too complex for educational purposes

## Key Decision: Physics Fidelity vs Performance Tradeoffs

### Decision: Balanced approach with configurable profiles
**Rationale:** For educational purposes, a balance is needed between realistic physics simulation and acceptable performance on development machines. We'll implement different simulation profiles (high-fidelity for research, performance-optimized for learning).

**Alternatives considered:**
- Maximum fidelity: Provides realistic simulation but requires high-end hardware
- Maximum performance: Allows fast simulation but may not reflect real-world physics
- Adaptive fidelity: Dynamic adjustment based on scene complexity

## Key Decision: When to Use Gazebo vs Unity

### Decision: Specialized roles - Gazebo for physics, Unity for visualization
**Rationale:** Gazebo is preferred for physics simulation and ROS 2 integration, while Unity is better for visualization and human-robot interaction design. Gazebo handles realistic sensor simulation and robot dynamics, while Unity excels at user interfaces and immersive visualization.

**Alternatives considered:**
- Exclusively Gazebo: Good for physics but limited visualization capabilities
- Exclusively Unity: Great visualization but less accurate physics
- Combined approach: Leverage strengths of both tools for different aspects

## Research Findings

### Gazebo Integration with ROS 2
- Gazebo Garden provides native ROS 2 support through ros-gz packages
- Can simulate sensors like cameras, LIDAR, IMU, and force/torque sensors
- Supports realistic physics with ODE, Bullet, and DART engines
- Has plugin system for custom sensor models and controllers

### Unity Integration Considerations
- Unity can connect to ROS 2 via rosbridge_suite
- Unity Robotics Package provides components for ROS integration
- Better for visualization and human-robot interaction design
- Less suitable for realistic physics simulation

### Docusaurus Documentation Setup
- Supports interactive code examples
- Can embed simulation videos and images
- Supports versioning for different simulation environments
- Good for educational content with multiple examples

### Technology Stack Validation
- ROS 2 Humble Hawksbill: LTS version with good support
- Gazebo Garden: Latest stable version with active development
- Unity 2022.3 LTS: Long-term support version suitable for this project
- Docusaurus 2.x: Modern static site generator with good plugin ecosystem

## Unknowns Resolved

1. **Gazebo Version**: Using Gazebo Garden (Ignition) instead of Classic
2. **Simulation Focus**: Physics simulation in Gazebo, visualization in Unity
3. **ROS 2 Integration**: Using ros-gz packages for Gazebo, rosbridge for Unity
4. **Performance Strategy**: Multiple fidelity profiles to balance realism and performance
5. **Content Structure**: Chapter 1 (Gazebo physics) → Chapter 2 (Simulated sensors) → Chapter 3 (Unity integration)

## Recommended Architecture
- Gazebo for physics simulation and sensor modeling
- ROS 2 as the communication middleware
- Unity for visualization and human-robot interaction concepts
- Docusaurus for documentation with embedded examples