# Research Document: Module 2 - Digital Twin (Gazebo & Unity)

**Feature**: Module 2: The Digital Twin (Gazebo & Unity)
**Created**: 2025-12-23

## Decision 1: Gazebo Classic vs Gazebo (Ignition/Harmonic)

### Decision: Use Gazebo Harmonic (Ignition)
**Rationale**:
- Gazebo Harmonic is the latest version and is actively maintained
- Better integration with ROS 2 (Rolling and Humble)
- More modern architecture and features
- Official ROS 2 documentation recommends Ignition for new projects
- Better long-term support and community resources

### Alternatives Considered:
- **Gazebo Classic**: Legacy version, no longer actively developed, lacks modern features
- **Gazebo Fortress**: Previous LTS version, but Harmonic is now recommended

## Decision 2: Physics Fidelity vs Performance Tradeoffs

### Decision: Balance for Educational Use
**Rationale**:
- For educational purposes, visual clarity and real-time feedback are more important than maximum physical accuracy
- Default physics parameters (ODE solver) provide good balance for learning
- Can be adjusted for different use cases (fast simulation vs accurate physics)
- Educational focus should be on concepts rather than computational optimization

### Performance Settings:
- Use real-time update rate of 1000 Hz for smooth visualization
- Set max step size to 0.001 for stable physics
- Use ODE as physics engine with default parameters for educational examples
- Enable wireframe mode for debugging and learning

## Decision 3: When to Use Gazebo vs Unity

### Decision: Complementary Roles
**Rationale**:
- **Gazebo**: Physics simulation, sensor simulation, algorithm testing
- **Unity**: High-fidelity visualization, user interaction, immersive experiences
- **ROS 2 Bridge**: Connects the two for complete simulation pipeline

### Use Case Guidelines:
- **Use Gazebo for**: Physics-based simulation, sensor modeling, robot control algorithms
- **Use Unity for**: High-quality visualization, user interfaces, human-robot interaction
- **Use Both**: For complete digital twin that combines accurate physics with high-quality graphics

## Decision 4: ROS 2 Integration Patterns

### Decision: Standard ROS 2 Simulation Patterns
**Rationale**:
- Follow ROS 2 best practices for simulation
- Use standard message types for sensor data
- Implement proper launch files and parameter configurations
- Include RViz visualization for sensor data

### Integration Patterns:
- Use standard sensor message types (sensor_msgs/LaserScan, sensor_msgs/Image, etc.)
- Implement proper TF trees for robot state visualization
- Use launch files for easy simulation setup
- Follow ROS 2 naming conventions for topics and services

## Technical Architecture Summary

### Gazebo Setup:
- Gazebo Harmonic with ROS 2 Humble
- Physics engine: ODE with default parameters
- Sensor plugins: libgazebo_ros_ray_sensor (LiDAR), libgazebo_ros_camera (depth camera), libgazebo_ros_imu (IMU)
- Robot model: URDF with proper joint and link definitions

### ROS 2 Integration:
- Sensor data published to standard ROS 2 topics
- TF tree for robot state visualization
- RViz for sensor data visualization
- Standard ROS 2 message types for compatibility

### Unity Integration (Conceptual):
- ROS 2 bridge for data transmission
- Unity for high-fidelity visualization
- Conceptual pipeline: ROS 2 state → Bridge → Unity visualization
- Focus on visualization and interaction concepts rather than implementation

## Validation Criteria

### Gazebo Simulation:
- Physics simulation runs in real-time
- Robot responds to gravity and collisions appropriately
- Sensors publish data to ROS 2 topics
- Data can be visualized in RViz

### ROS 2 Integration:
- All sensor topics publish valid messages
- TF tree is properly maintained
- Launch files work correctly
- RViz displays sensor data properly

### Educational Value:
- Examples are clear and reproducible
- Concepts are well-explained
- Practical exercises reinforce learning
- Content is accessible to target audience