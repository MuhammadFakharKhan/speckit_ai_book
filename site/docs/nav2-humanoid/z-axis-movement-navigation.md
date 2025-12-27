---
title: Z-Axis Movement Navigation Examples
description: Navigation examples and validation steps for humanoid robots handling Z-axis movement, stairs, and elevation changes using Nav2
sidebar_position: 7
tags: [z-axis, elevation, stairs, navigation, humanoid, nav2]
---

# Z-Axis Movement Navigation Examples

## Introduction

Z-axis movement navigation is a critical capability for humanoid robots, allowing them to navigate stairs, ramps, and elevation changes. Unlike wheeled robots, humanoid robots can handle discrete elevation changes and three-dimensional navigation. This document covers navigation examples and validation steps for Z-axis movement in humanoid robotics using Nav2.

## Z-Axis Navigation Fundamentals

### Three-Dimensional Navigation Concepts

Humanoid robots operate in 3D space with six degrees of freedom:

- **X-axis**: Forward/backward movement
- **Y-axis**: Left/right movement
- **Z-axis**: Up/down movement (elevation)
- **Roll**: Rotation around X-axis (tilting forward/backward)
- **Pitch**: Rotation around Y-axis (tilting left/right)
- **Yaw**: Rotation around Z-axis (turning)

For navigation, the Z-axis component introduces unique challenges:

- **Stair navigation**: Discrete elevation changes
- **Ramp navigation**: Continuous elevation changes
- **Step climbing**: Individual step navigation
- **Balance management**: Maintaining stability during elevation changes

### Z-Axis Navigation Challenges

```python
# Z-axis navigation challenges
class ZAxisNavigationChallenges:
    def __init__(self):
        self.challenges = {
            "stair_detection": {
                "description": "Identifying and classifying stairs",
                "complexity": "High",
                "solution_approach": "Computer vision and depth sensing"
            },
            "step_planning": {
                "description": "Planning individual steps for elevation changes",
                "complexity": "High",
                "solution_approach": "Step-by-step path planning"
            },
            "balance_management": {
                "description": "Maintaining balance during elevation changes",
                "complexity": "Critical",
                "solution_approach": "Balance control algorithms"
            },
            "terrain_classification": {
                "description": "Identifying different elevation terrains",
                "complexity": "Medium",
                "solution_approach": "Sensor fusion and classification"
            }
        }

    def get_challenge_complexity(self, challenge_name):
        """
        Get complexity rating for a specific challenge
        """
        return self.challenges.get(challenge_name, {}).get("complexity", "Unknown")

    def get_solution_approach(self, challenge_name):
        """
        Get solution approach for a specific challenge
        """
        return self.challenges.get(challenge_name, {}).get("solution_approach", "No solution")
```

## Stair Navigation Implementation

### Stair Detection and Classification

```python
# Stair detection for navigation
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point

class StairDetector:
    def __init__(self):
        self.min_step_height = 0.1  # meters
        self.max_step_height = 0.2  # meters
        self.step_depth = 0.3       # meters
        self.stair_angle_threshold = 10.0  # degrees

    def detect_stairs_from_pointcloud(self, pointcloud_msg):
        """
        Detect stairs from point cloud data
        """
        # Convert PointCloud2 to numpy array
        points = self.pointcloud_to_array(pointcloud_msg)

        # Segment ground plane to identify elevation changes
        ground_points, elevated_points = self.segment_ground_plane(points)

        # Identify potential stair regions
        stair_regions = self.identify_stair_regions(elevated_points, ground_points)

        # Validate stair regions
        validated_stairs = []
        for region in stair_regions:
            if self.validate_stair_region(region):
                validated_stairs.append(region)

        return validated_stairs

    def segment_ground_plane(self, points):
        """
        Segment ground plane from point cloud using RANSAC
        """
        from sklearn.linear_model import RANSACRegressor

        # Prepare data for RANSAC
        X = points[:, [0, 1]]  # x, y coordinates
        y = points[:, 2]       # z coordinates (height)

        # Apply RANSAC to find ground plane
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(X, y)

        # Identify inliers (ground points) and outliers (elevated points)
        inlier_mask = ransac.inlier_mask_

        ground_points = points[inlier_mask]
        elevated_points = points[~inlier_mask]

        return ground_points, elevated_points

    def identify_stair_regions(self, elevated_points, ground_points):
        """
        Identify potential stair regions in elevated points
        """
        stair_regions = []

        # Group elevated points by elevation levels
        elevation_levels = self.group_by_elevation(elevated_points)

        # Check for regular elevation changes (indicating stairs)
        for i in range(len(elevation_levels) - 1):
            current_level = elevation_levels[i]
            next_level = elevation_levels[i + 1]

            height_diff = next_level['elevation'] - current_level['elevation']

            # Check if height difference matches stair step height
            if self.min_step_height <= height_diff <= self.max_step_height:
                # Check if the area has stair-like characteristics
                if self.is_stair_like_area(current_level, next_level):
                    stair_regions.append({
                        'bottom_level': current_level,
                        'top_level': next_level,
                        'height_diff': height_diff,
                        'type': 'stairs'
                    })

        return stair_regions

    def validate_stair_region(self, stair_region):
        """
        Validate if detected region is actually stairs
        """
        # Check dimensions and regularity
        if (stair_region['height_diff'] >= self.min_step_height and
            stair_region['height_diff'] <= self.max_step_height):

            # Additional validation checks
            return True

        return False

    def group_by_elevation(self, points):
        """
        Group points by elevation levels
        """
        # Sort points by elevation
        sorted_points = points[points[:, 2].argsort()]

        # Group points with similar elevations
        elevation_groups = []
        current_group = []
        current_elevation = sorted_points[0][2]

        for point in sorted_points:
            if abs(point[2] - current_elevation) < 0.05:  # 5cm tolerance
                current_group.append(point)
            else:
                if current_group:
                    elevation_groups.append({
                        'points': np.array(current_group),
                        'elevation': current_elevation,
                        'center': np.mean(current_group, axis=0)
                    })
                current_group = [point]
                current_elevation = point[2]

        if current_group:
            elevation_groups.append({
                'points': np.array(current_group),
                'elevation': current_elevation,
                'center': np.mean(current_group, axis=0)
            })

        return elevation_groups

    def is_stair_like_area(self, level1, level2):
        """
        Check if area has stair-like characteristics
        """
        # Check for regular spacing and dimensions
        # This is a simplified check - real implementation would be more complex
        return True
```

### Stair Navigation Planner

```python
# Stair navigation planning
class StairNavigationPlanner:
    def __init__(self):
        self.step_height = 0.15  # meters
        self.step_depth = 0.30   # meters
        self.step_width = 0.80   # meters
        self.foot_clearance = 0.05  # meters

    def plan_stair_navigation(self, start_pose, end_pose, stair_info):
        """
        Plan navigation through stairs
        """
        navigation_steps = []

        # Calculate stair climb path
        stair_path = self.calculate_stair_path(start_pose, end_pose, stair_info)

        # Generate footstep plan
        footsteps = self.generate_footsteps_for_stairs(stair_path, stair_info)

        # Add approach and departure steps
        approach_steps = self.generate_approach_steps(start_pose, stair_path[0])
        departure_steps = self.generate_departure_steps(stair_path[-1], end_pose)

        navigation_steps.extend(approach_steps)
        navigation_steps.extend(footsteps)
        navigation_steps.extend(departure_steps)

        return navigation_steps

    def calculate_stair_path(self, start_pose, end_pose, stair_info):
        """
        Calculate path through stairs
        """
        path = []

        # Calculate number of steps needed
        height_diff = end_pose.position.z - start_pose.position.z
        num_steps = int(abs(height_diff) / self.step_height)

        # Generate intermediate poses for each step
        for i in range(1, num_steps + 1):
            step_pose = self.interpolate_stair_pose(
                start_pose, end_pose, i, num_steps
            )
            path.append(step_pose)

        return path

    def generate_footsteps_for_stairs(self, stair_path, stair_info):
        """
        Generate footstep plan for stair navigation
        """
        footsteps = []
        support_foot = "left"  # Start with left foot as support

        for i, step_pose in enumerate(stair_path):
            # Calculate foot placement for this step
            if support_foot == "left":
                # Right foot moves up
                right_foot_pose = self.calculate_right_foot_stair_pose(
                    step_pose, support_foot
                )
                footsteps.append(("right", right_foot_pose))
            else:
                # Left foot moves up
                left_foot_pose = self.calculate_left_foot_stair_pose(
                    step_pose, support_foot
                )
                footsteps.append(("left", left_foot_pose))

            # Alternate support foot
            support_foot = "right" if support_foot == "left" else "left"

        return footsteps

    def interpolate_stair_pose(self, start_pose, end_pose, step_num, total_steps):
        """
        Interpolate pose for a specific stair step
        """
        from geometry_msgs.msg import Pose
        interpolated_pose = Pose()

        # Linear interpolation for X, Y
        ratio = step_num / total_steps
        interpolated_pose.position.x = start_pose.position.x + \
            (end_pose.position.x - start_pose.position.x) * ratio
        interpolated_pose.position.y = start_pose.position.y + \
            (end_pose.position.y - start_pose.position.y) * ratio

        # Step-wise interpolation for Z (stair steps)
        step_height = self.step_height
        interpolated_pose.position.z = start_pose.position.z + step_height * step_num

        # Maintain orientation
        interpolated_pose.orientation = start_pose.orientation

        return interpolated_pose

    def calculate_right_foot_stair_pose(self, target_pose, support_foot):
        """
        Calculate right foot pose for stair climbing
        """
        from geometry_msgs.msg import Pose
        foot_pose = Pose()

        # Position foot for stair climbing
        foot_pose.position.x = target_pose.position.x
        foot_pose.position.y = target_pose.position.y - 0.1  # Slightly inward
        foot_pose.position.z = target_pose.position.z + self.foot_clearance

        foot_pose.orientation = target_pose.orientation

        return foot_pose

    def generate_approach_steps(self, start_pose, stair_start_pose):
        """
        Generate approach steps before stairs
        """
        approach_steps = []

        # Calculate approach path
        approach_path = self.calculate_approach_path(start_pose, stair_start_pose)

        # Convert to footsteps
        for pose in approach_path:
            approach_steps.append(("approach", pose))

        return approach_steps

    def generate_departure_steps(self, stair_end_pose, end_pose):
        """
        Generate departure steps after stairs
        """
        departure_steps = []

        # Calculate departure path
        departure_path = self.calculate_departure_path(stair_end_pose, end_pose)

        # Convert to footsteps
        for pose in departure_path:
            departure_steps.append(("departure", pose))

        return departure_steps
```

## Ramp Navigation Implementation

### Ramp Detection and Navigation

```python
# Ramp navigation
class RampNavigation:
    def __init__(self):
        self.max_ramp_angle = 15.0  # degrees
        self.ramp_step_size = 0.2   # meters

    def detect_ramps(self, elevation_data):
        """
        Detect ramps in elevation data
        """
        ramps = []

        # Calculate elevation gradients
        gradients = self.calculate_elevation_gradients(elevation_data)

        # Identify regions with consistent moderate slopes
        for i, gradient in enumerate(gradients):
            if self.is_ramp_gradient(gradient):
                ramp = self.extract_ramp_info(elevation_data, i)
                ramps.append(ramp)

        return ramps

    def calculate_elevation_gradients(self, elevation_data):
        """
        Calculate elevation gradients from elevation data
        """
        gradients = []

        for i in range(1, len(elevation_data)):
            dx = elevation_data[i].x - elevation_data[i-1].x
            dz = elevation_data[i].z - elevation_data[i-1].z
            gradient = np.arctan2(abs(dz), abs(dx)) * 180 / np.pi  # Convert to degrees
            gradients.append(gradient)

        return gradients

    def is_ramp_gradient(self, gradient):
        """
        Check if gradient indicates a ramp
        """
        return 2.0 <= gradient <= self.max_ramp_angle  # Between 2 and 15 degrees

    def plan_ramp_navigation(self, start_pose, end_pose, ramp_info):
        """
        Plan navigation through ramp
        """
        # Break ramp into manageable segments
        ramp_segments = self.divide_ramp_into_segments(start_pose, end_pose, ramp_info)

        # Plan footsteps for each segment
        footsteps = []
        for segment in ramp_segments:
            segment_footsteps = self.plan_segment_footsteps(segment)
            footsteps.extend(segment_footsteps)

        return footsteps

    def divide_ramp_into_segments(self, start_pose, end_pose, ramp_info):
        """
        Divide ramp into smaller navigable segments
        """
        segments = []

        # Calculate total distance along ramp
        total_distance = self.calculate_ramp_distance(start_pose, end_pose)

        # Divide into segments based on step size
        num_segments = int(total_distance / self.ramp_step_size)

        for i in range(num_segments):
            segment_start = self.interpolate_pose(
                start_pose, end_pose, i / num_segments
            )
            segment_end = self.interpolate_pose(
                start_pose, end_pose, (i + 1) / num_segments
            )

            segments.append({
                'start': segment_start,
                'end': segment_end,
                'segment_number': i
            })

        return segments

    def plan_segment_footsteps(self, segment):
        """
        Plan footsteps for a ramp segment
        """
        footsteps = []

        # Calculate required steps for this segment
        dx = segment['end'].position.x - segment['start'].position.x
        dy = segment['end'].position.y - segment['start'].position.y
        dz = segment['end'].position.z - segment['start'].position.z

        distance_2d = np.sqrt(dx**2 + dy**2)

        # Number of steps needed
        num_steps = max(1, int(distance_2d / 0.3))  # 30cm per step

        for i in range(num_steps):
            step_ratio = i / num_steps
            step_pose = self.interpolate_pose_with_elevation(
                segment['start'], segment['end'], step_ratio
            )

            footsteps.append(step_pose)

        return footsteps

    def interpolate_pose_with_elevation(self, start_pose, end_pose, ratio):
        """
        Interpolate pose considering elevation changes
        """
        from geometry_msgs.msg import Pose
        interpolated_pose = Pose()

        # Interpolate X, Y, Z positions
        interpolated_pose.position.x = start_pose.position.x + \
            (end_pose.position.x - start_pose.position.x) * ratio
        interpolated_pose.position.y = start_pose.position.y + \
            (end_pose.position.y - start_pose.position.y) * ratio
        interpolated_pose.position.z = start_pose.position.z + \
            (end_pose.position.z - start_pose.position.z) * ratio

        # Interpolate orientation
        interpolated_pose.orientation = self.interpolate_orientation(
            start_pose.orientation, end_pose.orientation, ratio
        )

        return interpolated_pose
```

## Integration with Nav2

### 3D Costmap Layer

```python
# 3D costmap for elevation-aware navigation
from nav2_costmap_2d.layers import CostmapLayer
from nav2_costmap_2d import Costmap2D
import numpy as np

class ZAxisCostmapLayer(CostmapLayer):
    def __init__(self, name, costmap, nh):
        super().__init__()
        self.layer_name = name
        self.costmap = costmap
        self.nh = nh

        # Z-axis specific parameters
        self.max_step_height = self.nh.get_parameter('max_step_height', 0.2)
        self.min_traversable_slope = self.nh.get_parameter('min_traversable_slope', 5.0)
        self.max_traversable_slope = self.nh.get_parameter('max_traversable_slope', 15.0)

    def updateBounds(self, robot_x, robot_y, robot_yaw, min_x, min_y, max_x, max_y):
        """
        Update bounds considering Z-axis constraints
        """
        # Call parent method
        self.updateBoundsFromArea(min_x, min_y, max_x, max_y)

        # Apply Z-axis constraints
        self.apply_z_axis_constraints(robot_x, robot_y, min_x, min_y, max_x, max_y)

    def apply_z_axis_constraints(self, robot_x, robot_y, min_x, min_y, max_x, max_y):
        """
        Apply Z-axis constraints to costmap
        """
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                world_x, world_y = self.costmap.getOriginX() + x, self.costmap.getOriginY() + y

                # Get elevation data at this location
                elevation = self.get_elevation_at(world_x, world_y)

                # Check if elevation change is within robot capabilities
                robot_elevation = self.get_robot_elevation()
                elevation_diff = abs(elevation - robot_elevation)

                if elevation_diff > self.max_step_height:
                    # Mark as lethal obstacle if elevation change too large
                    self.setCost(x, y, 254)  # LETHAL_OBSTACLE

    def updateCosts(self, master_grid, min_i, min_j, max_i, max_j):
        """
        Update costs considering Z-axis factors
        """
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                cost = master_grid.getCost(i, j)

                if cost >= 0 and cost < 253:  # Not unknown or lethal
                    # Apply Z-axis specific costs
                    z_cost = self.calculate_z_axis_cost(i, j)
                    final_cost = self.combine_costs(cost, z_cost)
                    master_grid.setCost(i, j, final_cost)

    def calculate_z_axis_cost(self, i, j):
        """
        Calculate Z-axis specific cost for a cell
        """
        world_x = self.costmap.getOriginX() + i
        world_y = self.costmap.getOriginY() + j

        # Get elevation and slope data
        elevation = self.get_elevation_at(world_x, world_y)
        slope = self.get_slope_at(world_x, world_y)

        # Calculate cost based on slope traversability
        if slope < self.min_traversable_slope:
            return 0  # Easy to traverse
        elif slope > self.max_traversable_slope:
            return 254  # Too steep, lethal obstacle
        else:
            # Cost increases with slope
            normalized_slope = (slope - self.min_traversable_slope) / \
                              (self.max_traversable_slope - self.min_traversable_slope)
            return int(normalized_slope * 200)  # Scale to cost range

    def get_elevation_at(self, x, y):
        """
        Get elevation at specific coordinates
        """
        # Implementation would interface with elevation map
        return 0.0  # Placeholder

    def get_slope_at(self, x, y):
        """
        Get slope at specific coordinates
        """
        # Implementation would calculate local slope
        return 0.0  # Placeholder

    def get_robot_elevation(self):
        """
        Get current robot elevation
        """
        # Implementation would get from robot state
        return 0.0  # Placeholder
```

### Z-Axis Navigation Action Server

```python
# Z-axis navigation action server
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool

class ZAxisNavigationServer(Node):
    def __init__(self):
        super().__init__('z_axis_navigation_server')

        # Action server
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'z_axis_navigate_to_pose',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers
        self.navigation_status_pub = self.create_publisher(
            Bool, '/z_axis_navigation_status', 10
        )

        # Navigation components
        self.stair_planner = StairNavigationPlanner()
        self.ramp_navigator = RampNavigation()
        self.balance_monitor = self.create_subscription(
            Bool, '/balance_status', self.balance_callback, 10
        )

        self.current_balance = True
        self.navigation_in_progress = False

    def goal_callback(self, goal_request):
        """
        Handle navigation goal request
        """
        # Check if goal has Z-axis component
        z_difference = abs(goal_request.pose.pose.position.z - self.get_robot_z_position())

        if z_difference > 0.1:  # If significant Z difference
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """
        Handle navigation cancel request
        """
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """
        Execute navigation goal
        """
        self.navigation_in_progress = True

        try:
            # Get goal pose
            goal_pose = goal_handle.request.pose

            # Analyze Z-axis requirements
            z_analysis = self.analyze_z_axis_requirements(goal_pose)

            # Plan navigation based on Z-axis analysis
            if z_analysis['terrain_type'] == 'stairs':
                navigation_plan = self.stair_planner.plan_stair_navigation(
                    self.get_current_pose(), goal_pose, z_analysis['terrain_info']
                )
            elif z_analysis['terrain_type'] == 'ramp':
                navigation_plan = self.ramp_navigator.plan_ramp_navigation(
                    self.get_current_pose(), goal_pose, z_analysis['terrain_info']
                )
            else:
                # Standard 2D navigation
                navigation_plan = self.plan_2d_navigation(
                    self.get_current_pose(), goal_pose
                )

            # Execute navigation plan
            success = self.execute_navigation_plan(navigation_plan)

            # Create result
            result = NavigateToPose.Result()
            result.result_code = NavigateToPose.Result.SUCCEEDED if success else NavigateToPose.Result.FAILED

            if success:
                goal_handle.succeed()
            else:
                goal_handle.abort()

            return result

        except Exception as e:
            self.get_logger().error(f'Navigation execution failed: {e}')
            goal_handle.abort()

            result = NavigateToPose.Result()
            result.result_code = NavigateToPose.Result.FAILED
            return result
        finally:
            self.navigation_in_progress = False

    def analyze_z_axis_requirements(self, goal_pose):
        """
        Analyze Z-axis requirements for navigation
        """
        current_z = self.get_robot_z_position()
        goal_z = goal_pose.pose.position.z
        z_difference = abs(goal_z - current_z)

        # Detect stairs, ramps, or other elevation changes
        terrain_type = self.detect_terrain_type(current_z, goal_z)

        return {
            'current_z': current_z,
            'goal_z': goal_z,
            'z_difference': z_difference,
            'terrain_type': terrain_type,
            'terrain_info': self.get_terrain_info(current_z, goal_z)
        }

    def detect_terrain_type(self, start_z, end_z):
        """
        Detect terrain type based on elevation change
        """
        z_diff = abs(end_z - start_z)

        if z_diff < 0.1:
            return 'flat'
        elif 0.1 <= z_diff <= 0.5:
            # Check for discrete steps (stairs) vs continuous slope (ramp)
            return self.classify_elevation_change(start_z, end_z)
        else:
            return 'elevation_change'

    def classify_elevation_change(self, start_z, end_z):
        """
        Classify elevation change as stairs or ramp
        """
        # Implementation would analyze intermediate elevation data
        # For simplicity, assume stairs for discrete changes
        return 'stairs'

    def execute_navigation_plan(self, plan):
        """
        Execute navigation plan
        """
        for step in plan:
            if not self.current_balance:
                self.get_logger().error('Balance compromised, stopping navigation')
                return False

            if not self.navigation_in_progress:
                return False

            # Execute step
            success = self.execute_navigation_step(step)
            if not success:
                return False

            # Small delay between steps for humanoid robots
            time.sleep(0.1)

        return True

    def balance_callback(self, msg):
        """
        Update balance status
        """
        self.current_balance = msg.data

    def get_robot_z_position(self):
        """
        Get current robot Z position
        """
        # Implementation would get from robot state
        return 0.0

    def get_current_pose(self):
        """
        Get current robot pose
        """
        # Implementation would get from localization system
        from geometry_msgs.msg import PoseStamped
        pose = PoseStamped()
        return pose
```

## Validation and Testing

### Z-Axis Navigation Validation

```python
# Z-axis navigation validation
import unittest
import numpy as np

class TestZAxisNavigation(unittest.TestCase):
    def setUp(self):
        self.stair_detector = StairDetector()
        self.stair_planner = StairNavigationPlanner()
        self.ramp_navigator = RampNavigation()

    def test_stair_detection(self):
        """
        Test stair detection functionality
        """
        # Create mock point cloud data with stairs
        stair_points = self.create_stair_pointcloud()

        # Detect stairs
        detected_stairs = self.stair_detector.detect_stairs_from_pointcloud(stair_points)

        # Verify detection
        self.assertGreater(len(detected_stairs), 0, "Stairs should be detected")

        # Verify stair properties
        for stair in detected_stairs:
            self.assertGreaterEqual(stair['height_diff'], 0.1, "Stair height should be reasonable")
            self.assertLessEqual(stair['height_diff'], 0.2, "Stair height should be reasonable")

    def test_stair_navigation_planning(self):
        """
        Test stair navigation planning
        """
        from geometry_msgs.msg import Pose

        # Create start and end poses with Z difference
        start_pose = Pose()
        start_pose.position.x = 0.0
        start_pose.position.y = 0.0
        start_pose.position.z = 0.0

        end_pose = Pose()
        end_pose.position.x = 2.0
        end_pose.position.y = 0.0
        end_pose.position.z = 0.3  # 30cm elevation change

        # Create stair info
        stair_info = {
            'type': 'stairs',
            'height_diff': 0.3,
            'num_steps': 2
        }

        # Plan navigation
        navigation_plan = self.stair_planner.plan_stair_navigation(
            start_pose, end_pose, stair_info
        )

        # Verify plan
        self.assertGreater(len(navigation_plan), 0, "Navigation plan should be created")

        # Check that plan includes Z-axis changes
        z_positions = [step[1].position.z for step in navigation_plan if len(step) > 1 and hasattr(step[1], 'position')]
        self.assertTrue(any(z > 0.0 for z in z_positions), "Plan should include Z-axis movement")

    def test_ramp_navigation(self):
        """
        Test ramp navigation
        """
        from geometry_msgs.msg import Pose

        # Create start and end poses for ramp
        start_pose = Pose()
        start_pose.position.x = 0.0
        start_pose.position.y = 0.0
        start_pose.position.z = 0.0

        end_pose = Pose()
        end_pose.position.x = 3.0
        end_pose.position.y = 0.0
        end_pose.position.z = 0.5  # Gentle slope

        # Create ramp info
        ramp_info = {
            'type': 'ramp',
            'slope': 10.0  # degrees
        }

        # Plan ramp navigation
        footsteps = self.ramp_navigator.plan_ramp_navigation(start_pose, end_pose, ramp_info)

        # Verify plan
        self.assertGreater(len(footsteps), 0, "Ramp navigation plan should be created")

        # Check gradual Z changes
        if len(footsteps) > 1:
            z_diffs = [
                footsteps[i+1].position.z - footsteps[i].position.z
                for i in range(len(footsteps)-1)
            ]
            avg_z_diff = np.mean([abs(diff) for diff in z_diffs])
            self.assertLess(avg_z_diff, 0.1, "Z changes should be gradual for ramps")

    def create_stair_pointcloud(self):
        """
        Create mock point cloud data representing stairs
        """
        # Create points representing two steps
        points = []

        # First step (0.15m high, 0.3m deep)
        for x in np.arange(0, 0.3, 0.05):
            for y in np.arange(-0.4, 0.4, 0.05):
                points.append([x, y, 0.15])

        # Second step (0.30m high, 0.3m deep)
        for x in np.arange(0.3, 0.6, 0.05):
            for y in np.arange(-0.4, 0.4, 0.05):
                points.append([x, y, 0.30])

        # Convert to PointCloud2 format (simplified)
        return np.array(points)

if __name__ == '__main__':
    unittest.main()
```

## Configuration Examples

### Z-Axis Navigation Configuration

```yaml
# Z-axis navigation configuration
z_axis_navigation:
  ros__parameters:
    # Stair navigation parameters
    stair:
      max_step_height: 0.2  # meters
      min_step_height: 0.1  # meters
      step_depth: 0.3       # meters
      step_width: 0.8       # meters
      foot_clearance: 0.05  # meters

    # Ramp navigation parameters
    ramp:
      max_slope_angle: 15.0  # degrees
      min_slope_angle: 2.0   # degrees
      step_size: 0.2         # meters

    # General Z-axis parameters
    z_axis:
      enable_3d_navigation: true
      elevation_tolerance: 0.05  # meters
      z_threshold_for_3d: 0.1    # minimum Z difference to trigger 3D nav

    # Costmap parameters
    costmap:
      max_step_height: 0.2
      min_traversable_slope: 5.0
      max_traversable_slope: 15.0

# Integration with Nav2 costmap
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: false
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      # Z-axis costmap layer
      plugins: [
        "z_axis_layer",
        "static_layer",
        "obstacle_layer",
        "inflation_layer"
      ]
      z_axis_layer:
        plugin: "z_axis_costmap_layer.ZAxisCostmapLayer"
        enabled: true
        max_step_height: 0.2
        min_traversable_slope: 5.0
        max_traversable_slope: 15.0
```

## Best Practices

### Z-Axis Navigation Best Practices

1. **Progressive Testing**: Start with simple elevation changes and increase complexity
2. **Safety First**: Always prioritize balance and safety over navigation speed
3. **Sensor Fusion**: Use multiple sensors for reliable elevation detection
4. **Step Validation**: Validate each step before execution
5. **Recovery Planning**: Plan recovery behaviors for failed elevation navigation

### Implementation Tips

1. **Modular Design**: Separate stair, ramp, and general Z-axis navigation logic
2. **Parameter Tuning**: Carefully tune parameters for your specific robot
3. **Simulation Testing**: Extensively test in simulation before real robot deployment
4. **Continuous Monitoring**: Monitor balance and stability throughout navigation
5. **Fallback Strategies**: Implement fallback navigation for challenging terrain

## Troubleshooting

### Common Z-Axis Navigation Issues

```yaml
# Troubleshooting guide for Z-axis navigation
troubleshooting:
  common_issues:
    - issue: "Stair detection failures"
      symptoms:
        - "Robot doesn't recognize stairs"
        - "Attempts to navigate stairs as flat ground"
      causes:
        - "Insufficient sensor resolution"
        - "Poor lighting conditions"
        - "Sensor noise"
      solutions:
        - "Improve sensor quality and resolution"
        - "Use multiple sensors for redundancy"
        - "Apply sensor fusion techniques"

    - issue: "Balance loss during elevation changes"
      symptoms:
        - "Robot falls during stair climbing"
        - "Frequent balance recovery"
      causes:
        - "Step timing too aggressive"
        - "Balance controller not tuned"
        - "Gait parameters incorrect"
      solutions:
        - "Slow down step execution timing"
        - "Tune balance control parameters"
        - "Improve gait planning"

    - issue: "Inefficient Z-axis navigation"
      symptoms:
        - "Long navigation times for elevation changes"
        - "Excessive computation during planning"
      causes:
        - "Complex planning algorithms"
        - "Inefficient path optimization"
      solutions:
        - "Simplify planning where possible"
        - "Use pre-computed navigation maps"
        - "Optimize algorithms for real-time performance"
```

This comprehensive guide provides examples and validation steps for implementing Z-axis movement navigation in humanoid robots using Nav2, covering stairs, ramps, and general elevation changes.