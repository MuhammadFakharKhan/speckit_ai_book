# Data Model: Module 2: The Digital Twin (Gazebo & Unity)

## Key Entities

### 1. Gazebo Simulation Environment
- **Name**: String (e.g., "humanoid_world_01")
- **Description**: Text description of the simulation environment
- **Physics Engine**: Enum (ode, bullet, dart)
- **Gravity**: Vector3 (x, y, z components)
- **Time Step**: Float (simulation time step in seconds)
- **Real Time Factor**: Float (ratio of simulation time to real time)
- **World File Path**: String (path to .sdf/.world file)
- **Models**: List of Robot/Obstacle models in the environment

### 2. Robot Model (URDF/SDF)
- **Name**: String (robot name)
- **Description**: Text description
- **URDF Path**: String (path to URDF file)
- **Joint States**: List of joint configurations
- **Links**: List of physical links with properties
- **Joints**: List of joint definitions (revolute, prismatic, etc.)
- **Sensors**: List of sensor configurations
- **Initial Pose**: Pose (position and orientation)

### 3. Sensor Configuration
- **Type**: Enum (camera, lidar, imu, force_torque, gps, etc.)
- **Name**: String (sensor identifier)
- **Topic**: String (ROS 2 topic name for sensor data)
- **Update Rate**: Float (frequency in Hz)
- **Range**: Float (sensor range in meters, if applicable)
- **Resolution**: Vector2 (pixel resolution for cameras)
- **Noise Model**: Noise parameters (mean, stddev for Gaussian noise)
- **Parent Link**: String (link to which sensor is attached)

### 4. ROS 2 Integration Data
- **Node Name**: String (name of the ROS 2 node)
- **Topics**: List of topic configurations
- **Services**: List of service configurations
- **Actions**: List of action configurations
- **Message Types**: List of supported message types
- **QoS Profile**: Quality of service settings

### 5. Unity Visualization Scene
- **Scene Name**: String (Unity scene identifier)
- **Camera Configuration**: Camera settings for visualization
- **Lighting**: Light settings and environment maps
- **Robot Prefab**: Unity prefab for robot visualization
- **Interaction Elements**: UI/HUD elements for human-robot interaction
- **Animation Controllers**: Animation setup for robot movements

### 6. Docusaurus Documentation Page
- **Title**: String (page title)
- **Slug**: String (URL-friendly identifier)
- **Sidebar Position**: Integer (position in sidebar navigation)
- **Content**: Markdown content with embedded examples
- **Learning Objectives**: List of educational goals
- **Code Examples**: List of executable code snippets
- **Simulation Assets**: List of related simulation files

### 7. Simulation Profile
- **Name**: String (profile identifier like "high_fidelity", "performance", "education")
- **Physics Settings**: Configuration for physics engine
- **Visual Quality**: Settings for rendering quality
- **Real-time Factor**: Target simulation speed
- **Hardware Requirements**: Minimum system specifications
- **Use Case**: Description of when to use this profile

## Relationships

- A **Gazebo Simulation Environment** contains multiple **Robot Models**
- A **Robot Model** has multiple **Sensor Configurations**
- A **Gazebo Simulation Environment** connects to **ROS 2 Integration Data**
- A **Robot Model** corresponds to a **Unity Visualization Scene**
- **Docusaurus Documentation Pages** reference **Gazebo Simulation Environments** and **Unity Visualization Scenes**
- Multiple **Simulation Profiles** can apply to the same **Gazebo Simulation Environment**

## Validation Rules

1. **Robot Model** must have at least one joint and one link
2. **Sensor Configuration** topic names must follow ROS 2 naming conventions
3. **URDF Path** in Robot Model must point to a valid URDF file
4. **Update Rate** in Sensor Configuration must be positive
5. **Simulation Profile** parameters must be within reasonable bounds for the target hardware
6. **Docusaurus Documentation Page** must have a unique slug within the module
7. **Gazebo Simulation Environment** must have a valid world file path

## State Transitions

### Simulation Environment States
- **Design**: Environment is being created/edited
- **Validated**: Environment passes all validation checks
- **Published**: Environment is ready for use in documentation
- **Deprecated**: Environment is no longer recommended for use

### Documentation States
- **Draft**: Initial content being created
- **Reviewed**: Content reviewed by human expert
- **Published**: Content is live in documentation
- **Archived**: Content is no longer maintained