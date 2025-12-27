# Data Model: Module 2 - Digital Twin (Gazebo & Unity)

**Feature**: Module 2: The Digital Twin (Gazebo & Unity)
**Created**: 2025-12-23

## Entity: Digital Twin

**Description**: A virtual representation of a physical robot system that simulates its behavior, physics, and sensor responses

**Attributes**:
- `id`: Unique identifier for the digital twin instance
- `name`: Human-readable name for the digital twin
- `robot_type`: Type of robot being simulated (e.g., humanoid, wheeled, manipulator)
- `physics_properties`: Configuration for physics simulation (gravity, friction, etc.)
- `sensor_configurations`: List of sensor types and parameters
- `environment`: Virtual environment configuration
- `simulation_state`: Current state of the simulation (running, paused, stopped)

**Relationships**:
- Contains multiple `Sensor` instances
- Associated with one `Environment`
- Connected to `ROS 2 Bridge` for data transmission

## Entity: Physics Simulation

**Description**: The computational modeling of physical forces and interactions in a virtual environment

**Attributes**:
- `gravity`: Gravitational force vector (x, y, z)
- `friction_coefficient`: Coefficient of friction for surfaces
- `solver_type`: Type of physics solver (ODE, Bullet, etc.)
- `step_size`: Time step for physics calculations
- `real_time_factor`: Ratio of simulation time to real time
- `collision_detection`: Method for collision detection

**Relationships**:
- Belongs to a `Digital Twin`
- Applies to multiple `Joint` and `Link` components of a robot

## Entity: Sensor Simulation

**Description**: The emulation of real-world sensors (LiDAR, cameras, IMUs) in a virtual environment

**Attributes**:
- `sensor_type`: Type of sensor (LiDAR, camera, IMU, etc.)
- `topic_name`: ROS 2 topic name for sensor data
- `update_rate`: Frequency of sensor data updates
- `range_min`: Minimum sensing range
- `range_max`: Maximum sensing range
- `resolution`: Spatial or angular resolution
- `noise_model`: Parameters for sensor noise simulation
- `sensor_frame`: Coordinate frame for sensor data

**Relationships**:
- Belongs to a `Digital Twin`
- Publishes to `ROS 2 Topic`
- Associated with a `Robot Link`

## Entity: ROS 2 Bridge

**Description**: The connection mechanism between simulation environments (Gazebo) and visualization engines (Unity)

**Attributes**:
- `connection_type`: Type of connection (TCP, UDP, shared memory)
- `topic_mappings`: Mappings between Gazebo and Unity topics
- `data_format`: Format of transmitted data
- `update_frequency`: Frequency of data transmission
- `connection_status`: Current status of the bridge

**Relationships**:
- Connects `Gazebo Simulation` to `Unity Visualization`
- Manages multiple `ROS 2 Topic` connections

## Entity: Environment

**Description**: The virtual world where the robot simulation takes place

**Attributes**:
- `name`: Name of the environment
- `description`: Brief description of the environment
- `objects`: List of static and dynamic objects in the environment
- `lighting`: Lighting configuration
- `textures`: Visual properties of surfaces
- `physics_properties`: Environment-specific physics parameters

**Relationships**:
- Used by multiple `Digital Twin` instances
- Contains multiple `Object` instances

## Entity: Robot Model

**Description**: The 3D representation of the physical robot with joints, links, and properties

**Attributes**:
- `model_name`: Name of the robot model
- `urdf_path`: Path to the URDF file
- `mesh_files`: Paths to 3D mesh files
- `inertial_properties`: Mass, center of mass, and inertia tensor
- `visual_properties`: Visual appearance parameters
- `collision_properties`: Collision detection parameters

**Relationships**:
- Used by a `Digital Twin`
- Contains multiple `Joint` and `Link` components
- Associated with multiple `Sensor` instances

## Entity: Joint

**Description**: Connection between two links in a robot model that allows relative motion

**Attributes**:
- `joint_name`: Name of the joint
- `joint_type`: Type of joint (revolute, prismatic, fixed, etc.)
- `parent_link`: Parent link in the kinematic chain
- `child_link`: Child link in the kinematic chain
- `axis`: Axis of motion for the joint
- `limits`: Position, velocity, and effort limits
- `dynamics`: Friction and damping parameters

**Relationships**:
- Belongs to a `Robot Model`
- Connected to two `Link` components

## Entity: Link

**Description**: Rigid component of a robot model that connects joints

**Attributes**:
- `link_name`: Name of the link
- `inertial`: Mass properties (mass, inertia tensor)
- `visual`: Visual appearance (mesh, color, material)
- `collision`: Collision geometry for physics simulation
- `pose`: Position and orientation relative to parent

**Relationships**:
- Belongs to a `Robot Model`
- Connected to multiple `Joint` components
- May contain multiple `Sensor` instances

## Entity: ROS 2 Topic

**Description**: Communication channel for publishing and subscribing to sensor data and robot state

**Attributes**:
- `topic_name`: Name of the ROS 2 topic
- `message_type`: Type of ROS 2 message (e.g., sensor_msgs/LaserScan)
- `publishers`: List of nodes publishing to this topic
- `subscribers`: List of nodes subscribing to this topic
- `qos_profile`: Quality of service settings
- `message_frequency`: Expected frequency of messages

**Relationships**:
- Connected through `ROS 2 Bridge`
- Used by multiple `Sensor` instances
- Monitored by `Visualization` tools

## Validation Rules

### Digital Twin Validation
- Must have a valid robot type
- Must have at least one sensor configuration
- Physics properties must be within valid ranges
- Simulation state must be consistent with environment

### Physics Simulation Validation
- Gravity values must be physically reasonable
- Step size must be positive and within stable range
- Real-time factor must be positive
- Solver type must be supported

### Sensor Simulation Validation
- Update rate must be positive
- Range values must be within physical limits
- Topic names must follow ROS 2 conventions
- Sensor frame must be properly defined in TF tree

### ROS 2 Bridge Validation
- Connection type must be supported
- Topic mappings must be valid
- Update frequency must be reasonable
- Connection status must be monitored

## State Transitions

### Digital Twin States
- `created` → `configured`: When physics and sensor parameters are set
- `configured` → `running`: When simulation starts
- `running` → `paused`: When simulation is paused
- `paused` → `running`: When simulation resumes
- `running` → `stopped`: When simulation ends
- `stopped` → `configured`: When simulation is reset

### Sensor States
- `configured` → `publishing`: When sensor starts publishing data
- `publishing` → `error`: When sensor encounters an error
- `error` → `configured`: When sensor is reset