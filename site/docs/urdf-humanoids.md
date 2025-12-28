---
title: URDF for Humanoid Robots
sidebar_position: 1
---

# URDF for Humanoid Robots

## Learning Objectives

By the end of this chapter, you will be able to:
- Design URDF models for humanoid robots
- Understand the structure and components of humanoid robot models
- Visualize and validate URDF models in simulation environments

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML format used to describe robots in ROS. For humanoid robots, URDF defines the physical structure, including links (rigid bodies) and joints (connections between links).

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.5"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Humanoid Robot Structure

Humanoid robots typically have the following components:

- **Torso**: The central body containing the main computer and power systems
- **Head**: Contains sensors like cameras and IMUs
- **Arms**: With shoulders, elbows, and wrists for manipulation
- **Legs**: With hips, knees, and ankles for locomotion
- **Hands/Feet**: For interaction with the environment

### Complete Humanoid URDF Example

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.2"/>
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.2"/>
      <geometry>
        <box size="0.3 0.3 0.4"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.2"/>
      <geometry>
        <box size="0.3 0.3 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.4"/>
  </joint>

  <link name="torso">
    <inertial>
      <mass value="8.0"/>
      <origin xyz="0 0 0.3"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.3"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <box size="0.25 0.25 0.6"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <box size="0.25 0.25 0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="head_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50.0" velocity="2.0"/>
  </joint>

  <link name="left_upper_arm">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.36" upper="0" effort="30.0" velocity="2.0"/>
  </joint>

  <link name="left_lower_arm">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.1"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1"/>
      <geometry>
        <cylinder length="0.2" radius="0.04"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1"/>
      <geometry>
        <cylinder length="0.2" radius="0.04"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Arm (similar structure) -->
  <joint name="right_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.15 0 0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50.0" velocity="2.0"/>
  </joint>

  <link name="right_upper_arm">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_elbow" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.36" upper="0" effort="30.0" velocity="2.0"/>
  </joint>

  <link name="right_lower_arm">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.1"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1"/>
      <geometry>
        <cylinder length="0.2" radius="0.04"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1"/>
      <geometry>
        <cylinder length="0.2" radius="0.04"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="0.1 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="100.0" velocity="1.0"/>
  </joint>

  <link name="left_thigh">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 -0.2"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2"/>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2"/>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.36" effort="100.0" velocity="1.0"/>
  </joint>

  <link name="left_shin">
    <inertial>
      <mass value="4.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.06" ixy="0.0" ixz="0.0" iyy="0.06" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_ankle" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="50.0" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.025"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.025"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Leg (similar structure) -->
  <joint name="right_hip_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="-0.1 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="100.0" velocity="1.0"/>
  </joint>

  <link name="right_thigh">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 -0.2"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2"/>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2"/>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_knee" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.36" effort="100.0" velocity="1.0"/>
  </joint>

  <link name="right_shin">
    <inertial>
      <mass value="4.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.06" ixy="0.0" ixz="0.0" iyy="0.06" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_ankle" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="50.0" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.025"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.025"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
  </link>
</robot>
```

## URDF Properties for Humanoid Robots

### Joint Types

1. **Revolute**: Rotational joint with limited range
2. **Continuous**: Rotational joint without limits
3. **Prismatic**: Linear sliding joint
4. **Fixed**: No movement (welded connection)
5. **Floating**: 6 DOF (rarely used in humanoid models)

### Inertial Properties

For humanoid robots, accurate inertial properties are crucial for realistic simulation:

```xml
<inertial>
  <mass value="5.0"/>  <!-- Mass in kg -->
  <origin xyz="0 0 -0.2"/>  <!-- Center of mass -->
  <inertia ixx="0.1" ixy="0.0" ixz="0.0"
           iyy="0.1" iyz="0.0" izz="0.02"/>  <!-- Inertia matrix -->
</inertial>
```

### Visual and Collision Properties

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <mesh filename="package://my_robot/meshes/link1.dae"/>
    <!-- or -->
    <box size="0.1 0.1 0.1"/>
    <cylinder length="0.1" radius="0.05"/>
    <sphere radius="0.05"/>
  </geometry>
  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>
</visual>

<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.1 0.1 0.1"/>
  </geometry>
</collision>
```

## Creating Materials and Colors

For humanoid robots, you might want to define materials for different parts:

```xml
<material name="red">
  <color rgba="1 0 0 1"/>
</material>

<material name="green">
  <color rgba="0 1 0 1"/>
</material>

<material name="blue">
  <color rgba="0 0 1 1"/>
</material>

<material name="white">
  <color rgba="1 1 1 1"/>
</material>

<material name="black">
  <color rgba="0 0 0 1"/>
</material>

<material name="light_grey">
  <color rgba="0.8 0.8 0.8 1"/>
</material>

<material name="dark_grey">
  <color rgba="0.3 0.3 0.3 1"/>
</material>
```

## URDF Tools and Validation

### Checking URDF Validity

You can validate your URDF using the check_urdf command:

```bash
# First install the tool if needed
sudo apt-get install ros-humble-urdf-tutorial

# Check your URDF file
check_urdf /path/to/your/robot.urdf
```

### Visualizing URDF

To visualize your URDF in RViz:

```bash
# Launch robot state publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat robot.urdf)
```

## Gazebo-Specific Tags

If you plan to use your humanoid URDF in Gazebo, you might need Gazebo-specific tags:

```xml
<gazebo reference="left_foot">
  <mu1>0.8</mu1>
  <mu2>0.8</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
  <material>Gazebo/Blue</material>
</gazebo>

<gazebo>
  <plugin name="robot_state_publisher" filename="libgazebo_ros_robot_state_publisher.so">
    <robot_param>robot_description</robot_param>
  </plugin>
</gazebo>
```

## Best Practices for Humanoid URDF

### 1. Naming Conventions

Use consistent naming for joints and links:

```
left_hip_yaw
left_hip_roll
left_hip_pitch
left_knee
left_ankle_pitch
left_ankle_roll
```

### 2. Proper Mass Distribution

Distribute mass realistically across the robot:

- Torso: 40-50% of total mass
- Legs: 30-35% of total mass (each)
- Arms: 10-15% of total mass (each)
- Head: 2-5% of total mass

### 3. Joint Limits

Set realistic joint limits based on human anatomy or design constraints:

```xml
<!-- Hip joints -->
<limit lower="-0.5236" upper="0.5236" effort="100.0" velocity="1.0"/>  <!-- ±30° -->
<!-- Knee joints -->
<limit lower="0" upper="2.3562" effort="100.0" velocity="1.0"/>      <!-- 0 to 135° -->
<!-- Ankle joints -->
<limit lower="-0.5236" upper="0.5236" effort="50.0" velocity="1.0"/>  <!-- ±30° -->
```

### 4. Safety Margins

Include safety margins in joint limits to prevent damage:

```xml
<!-- Human ankle can go about ±35°, but limit to ±30° for safety -->
<limit lower="-0.5236" upper="0.5236" effort="50.0" velocity="1.0"/>  <!-- ±30° -->
```

## Simulation Considerations

### Realistic Inertias

For stable simulation, use realistic inertia values. For simple shapes:

- Solid cylinder: I = 1/12 * m * (3*r² + h²) around diameter, I = 1/2 * m * r² around axis
- Solid sphere: I = 2/5 * m * r²
- Box: I = 1/12 * m * (h² + d²) around longest axis

### Collision Models

Use simplified collision models for better performance:

```xml
<!-- Use simple shapes for collision, detailed meshes for visual -->
<collision>
  <geometry>
    <cylinder length="0.3" radius="0.05"/>
  </geometry>
</collision>

<visual>
  <geometry>
    <mesh filename="package://my_robot/meshes/complex_arm.dae"/>
  </geometry>
</visual>
```

## Troubleshooting Common Issues

### 1. Robot Falls Through Ground

- Check that base_link has proper collision geometry
- Verify that all links have valid inertial properties
- Ensure joints are properly connected

### 2. Unstable Simulation

- Reduce joint efforts or add damping
- Check inertia values (too high or too low)
- Use fixed joints instead of very stiff revolute joints where appropriate

### 3. Joint Limit Violations

- Add safety margins to joint limits
- Implement software joint limit checking in controllers
- Use soft limits in addition to hard limits

## Advanced URDF Features

### Transmission Elements

For connecting actuators to joints:

```xml
<transmission name="left_hip_yaw_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_hip_yaw">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_hip_yaw_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### ROS Control Integration

For integration with ros2_control:

```xml
<ros2_control name="GazeboSystem" type="system">
  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>
  <joint name="left_hip_yaw">
    <command_interface name="position">
      <param name="min">-0.5236</param>
      <param name="max">0.5236</param>
    </command_interface>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
  </joint>
</ros2_control>
```

## Summary

This chapter covered the fundamentals of creating URDF models for humanoid robots. We explored the structure of humanoid robots, best practices for defining links and joints, and considerations for simulation. The URDF format provides a flexible way to describe the physical structure of humanoid robots for use in ROS and simulation environments.

In the next chapter, we'll explore how to integrate these URDF models with ROS 2 and Gazebo for simulation and control.