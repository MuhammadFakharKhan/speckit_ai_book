---
title: Behavior Tree Configurations for Humanoid Navigation
description: Behavior tree configurations adapted for humanoid robot navigation using Nav2, including custom nodes for bipedal locomotion
sidebar_position: 5
tags: [behavior-tree, navigation, nav2, humanoid, custom-nodes]
---

# Behavior Tree Configurations for Humanoid Navigation

## Introduction

Behavior trees in Nav2 provide a flexible framework for controlling robot navigation behavior. For humanoid robots, behavior trees require special configurations to account for bipedal locomotion, balance requirements, and unique navigation challenges. This document covers how to configure behavior trees specifically for humanoid robot navigation.

## Behavior Tree Fundamentals for Humanoid Robots

### Core Concepts

Behavior trees in Nav2 for humanoid robots must account for:

- **Balance requirements**: Navigation decisions must consider robot balance
- **Step-by-step execution**: Movement occurs in discrete steps
- **Gait patterns**: Different walking patterns for different situations
- **Recovery behaviors**: Balance recovery and obstacle avoidance
- **3D navigation**: Movement in three dimensions including elevation

### Basic Behavior Tree Structure

```xml
<!-- Basic humanoid navigation behavior tree -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <Sequence>
            <GoalUpdated/>
            <ComputePathToPose goal="{goal}" path="{path}"/>
            <Fallback name="ExecutePathFallback">
                <RecoveryNode number_of_retries="2">
                    <Sequence>
                        <SmoothPath path="{path}" output="{smoothed_path}"/>
                        <PlanFootsteps path="{smoothed_path}" footsteps="{footsteps}"/>
                        <FollowFootsteps footsteps="{footsteps}"/>
                    </Sequence>
                    <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
                </RecoveryNode>
                <ReactiveFallback>
                    <CheckGoalReaching>
                        <GoalReached goal="{goal}" tolerance="0.5"/>
                    </CheckGoalReaching>
                    <RecoveryNode number_of_retries="2">
                        <Spin spin_dist="1.57"/>
                        <Wait wait_duration="5"/>
                    </RecoveryNode>
                </ReactiveFallback>
            </Fallback>
        </Sequence>
    </BehaviorTree>
</root>
```

## Custom Behavior Tree Nodes for Humanoid Robots

### Step Execution Node

```cpp
// Custom behavior tree node for step execution
#include "behaviortree_cpp_v3/action_node.h"
#include "geometry_msgs/msg/pose.hpp"
#include "std_msgs/msg/bool.hpp"
#include "rclcpp/rclcpp.hpp"

class StepExecutionNode : public BT::StatefulActionNode
{
public:
    StepExecutionNode(const std::string& name, const BT::NodeConfiguration& config)
        : BT::StatefulActionNode(name, config)
    {
        node_ = rclcpp::Node::make_shared("step_execution_node");
        step_publisher_ = node_->create_publisher<geometry_msgs::msg::Pose>("/step_command", 10);
        balance_subscriber_ = node_->create_subscription<std_msgs::msg::Bool>(
            "/balance_status", 10,
            std::bind(&StepExecutionNode::balanceCallback, this, std::placeholders::_1)
        );
    }

    static BT::PortsList providedPorts()
    {
        return {
            BT::InputPort<geometry_msgs::msg::Pose>("step_pose", "The pose for the next step")
        };
    }

private:
    // Method invoked at the beginning of the tick
    BT::NodeStatus onStart() override
    {
        if (!getInput<geometry_msgs::msg::Pose>("step_pose", step_pose_)) {
            throw BT::RuntimeError("Missing required input [step_pose]");
        }

        // Check if robot is in balance state
        if (!balance_ok_) {
            return BT::NodeStatus::FAILURE;
        }

        // Execute the step
        step_publisher_->publish(step_pose_);
        step_execution_start_time_ = node_->get_clock()->now();

        return BT::NodeStatus::RUNNING;
    }

    // Method invoked at every tick during the RUNNING state
    BT::NodeStatus onRunning() override
    {
        // Check if step execution has completed
        if (step_execution_completed_) {
            step_execution_completed_ = false;
            return BT::NodeStatus::SUCCESS;
        }

        // Check for timeout
        auto elapsed = node_->get_clock()->now() - step_execution_start_time_;
        if (elapsed.seconds() > 5.0) { // 5 second timeout
            return BT::NodeStatus::FAILURE;
        }

        return BT::NodeStatus::RUNNING;
    }

    // Method invoked upon successful completion or during halt()
    void onHalted() override
    {
        RCLCPP_INFO(node_->get_logger(), "Step execution halted");
    }

    void balanceCallback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        balance_ok_ = msg->data;
    }

    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr step_publisher_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr balance_subscriber_;

    geometry_msgs::msg::Pose step_pose_;
    rclcpp::Time step_execution_start_time_;
    bool step_execution_completed_ = false;
    bool balance_ok_ = true;
};
```

### Balance Recovery Node

```cpp
// Custom behavior tree node for balance recovery
class BalanceRecoveryNode : public BT::StatefulActionNode
{
public:
    BalanceRecoveryNode(const std::string& name, const BT::NodeConfiguration& config)
        : BT::StatefulActionNode(name, config)
    {
        node_ = rclcpp::Node::make_shared("balance_recovery_node");
        recovery_publisher_ = node_->create_publisher<std_msgs::msg::Bool>("/balance_recovery_command", 10);
        balance_subscriber_ = node_->create_subscription<std_msgs::msg::Bool>(
            "/balance_status", 10,
            std::bind(&BalanceRecoveryNode::balanceCallback, this, std::placeholders::_1)
        );
    }

private:
    BT::NodeStatus onStart() override
    {
        RCLCPP_INFO(node_->get_logger(), "Initiating balance recovery");

        // Publish recovery command
        auto recovery_msg = std_msgs::msg::Bool();
        recovery_msg.data = true;
        recovery_publisher_->publish(recovery_msg);

        recovery_start_time_ = node_->get_clock()->now();
        recovery_in_progress_ = true;

        return BT::NodeStatus::RUNNING;
    }

    BT::NodeStatus onRunning() override
    {
        // Check if balance has been recovered
        if (balance_ok_) {
            recovery_in_progress_ = false;
            return BT::NodeStatus::SUCCESS;
        }

        // Check for timeout
        auto elapsed = node_->get_clock()->now() - recovery_start_time_;
        if (elapsed.seconds() > 10.0) { // 10 second timeout
            recovery_in_progress_ = false;
            return BT::NodeStatus::FAILURE;
        }

        return BT::NodeStatus::RUNNING;
    }

    void onHalted() override
    {
        RCLCPP_INFO(node_->get_logger(), "Balance recovery halted");
        recovery_in_progress_ = false;
    }

    void balanceCallback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (recovery_in_progress_) {
            balance_ok_ = msg->data;
        }
    }

    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr recovery_publisher_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr balance_subscriber_;

    rclcpp::Time recovery_start_time_;
    bool recovery_in_progress_ = false;
    bool balance_ok_ = true;
};
```

## Humanoid-Specific Behavior Tree Configurations

### Basic Humanoid Navigation Tree

```xml
<!-- Complete humanoid navigation behavior tree -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <ReactiveSequence name="MainSequence">
            <GoalUpdated/>
            <Fallback name="GlobalPlannerFallback">
                <Sequence name="GlobalPlanAndNavigate">
                    <ComputePathToPose goal="{goal}" path="{path}"/>
                    <Fallback name="LocalPlannerFallback">
                        <RecoveryNode number_of_retries="3" name="NavigateWithRecovery">
                            <Sequence name="NavigateSequence">
                                <SmoothPath path="{path}" output="{smoothed_path}"/>
                                <PlanFootsteps path="{smoothed_path}" footsteps="{footsteps}"/>
                                <Fallback name="StepExecutionFallback">
                                    <FollowFootsteps footsteps="{footsteps}"/>
                                    <ReactiveSequence name="BalanceRecoverySequence">
                                        <CheckBalance status="{balance_status}"/>
                                        <BalanceRecoveryNode/>
                                        <Wait wait_duration="2.0"/>
                                    </ReactiveSequence>
                                </Fallback>
                            </Sequence>
                            <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
                        </RecoveryNode>
                        <ReactiveSequence name="GoalReachingFallback">
                            <CheckGoalReaching goal="{goal}" tolerance="0.5">
                                <GoalReached goal="{goal}" tolerance="0.5"/>
                            </CheckGoalReaching>
                            <RecoveryNode number_of_retries="2" name="GoalRecovery">
                                <Spin spin_dist="1.57"/>
                                <Wait wait_duration="3.0"/>
                            </RecoveryNode>
                        </ReactiveSequence>
                    </Fallback>
                </Sequence>
                <ReactiveSequence name="GlobalPlanFailure">
                    <GoalReached goal="{goal}" tolerance="1.0"/>
                    <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
                </ReactiveSequence>
            </Fallback>
        </ReactiveSequence>
    </BehaviorTree>

    <!-- Custom nodes definition -->
    <TreeNodeModel>
        <Action ID="BalanceRecoveryNode">
            <input_port name="recovery_type" type="std::string" default="step_in_place"/>
        </Action>
        <Condition ID="CheckBalance">
            <output_port name="status" type="bool"/>
        </Condition>
        <Action ID="FollowFootsteps">
            <input_port name="footsteps" type="nav2_msgs::msg::Path"/>
        </Action>
        <Action ID="PlanFootsteps">
            <input_port name="path" type="nav2_msgs::msg::Path"/>
            <output_port name="footsteps" type="nav2_msgs::msg::Path"/>
        </Action>
    </TreeNodeModel>
</root>
```

### Complex Navigation Scenarios

```xml
<!-- Advanced humanoid navigation tree for complex scenarios -->
<root main_tree_to_execute="AdvancedMainTree">
    <BehaviorTree ID="AdvancedMainTree">
        <ReactiveSequence name="AdvancedNavigation">
            <GoalUpdated/>
            <Fallback name="PlanningFallback">
                <!-- Primary navigation approach -->
                <Sequence name="PrimaryNavigation">
                    <ComputePathToPose goal="{goal}" path="{path}"/>
                    <Fallback name="ExecutionFallback">
                        <RecoveryNode number_of_retries="5" name="PrimaryExecution">
                            <Sequence name="PrimaryExecutionSequence">
                                <SmoothPath path="{path}" output="{smoothed_path}"/>
                                <PlanFootsteps path="{smoothed_path}" footsteps="{footsteps}"/>
                                <NavigateWithSteps footsteps="{footsteps}"/>
                            </Sequence>
                            <ClearEntireCostmap name="ClearLocalCostmap"/>
                        </RecoveryNode>
                        <!-- Alternative navigation if primary fails -->
                        <Sequence name="AlternativeNavigation">
                            <ComputePathToPose goal="{goal}" path="{alt_path}" planner_id="dwb_core"/>
                            <RecoveryNode number_of_retries="3" name="AltExecution">
                                <NavigateWithSteps path="{alt_path}"/>
                                <ClearEntireCostmap name="ClearAltCostmap"/>
                            </RecoveryNode>
                        </Sequence>
                    </Fallback>
                </Sequence>
                <!-- Emergency navigation if planning fails -->
                <Sequence name="EmergencyNavigation">
                    <Fallback name="EmergencyFallback">
                        <NavigateToPose goal="{goal}" controller_server="emergency_controller"/>
                        <ReactiveSequence name="EmergencyRecovery">
                            <GoalReached goal="{goal}" tolerance="2.0"/>
                            <Wait wait_duration="5.0"/>
                        </ReactiveSequence>
                    </Fallback>
                </Sequence>
            </Fallback>
        </ReactiveSequence>
    </BehaviorTree>

    <!-- Emergency and safety behaviors -->
    <BehaviorTree ID="EmergencyTree">
        <ReactiveSequence name="EmergencySequence">
            <CheckEmergencyStop condition="{emergency_stop}"/>
            <EmergencyStopBehavior/>
            <Fallback name="RecoveryFallback">
                <BalanceRecoveryNode recovery_type="emergency"/>
                <Wait wait_duration="1.0"/>
            </Fallback>
        </ReactiveSequence>
    </BehaviorTree>
</root>
```

## Configuration Parameters

### Behavior Tree Configuration File

```yaml
# Behavior tree configuration for humanoid navigation
behavior_tree:
  ros__parameters:
    # Default behavior tree XML file
    default_bt_xml_filename: "humanoid_navigation_tree.xml"

    # Enable behavior tree logging
    enable_bt_tree_logging: true
    enable_bt_node_logging: true

    # Blackboard cleanup parameters
    blackboard_cleanup_service: "cleanup_blackboard"

    # Server parameters
    enable_groot_monitoring: true
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667

    # Recovery parameters
    number_of_recoveries: 5
    enable_recovery: true

    # Humanoid-specific parameters
    step_execution_timeout: 5.0
    balance_check_frequency: 10.0
    gait_adaptation_enabled: true
    recovery_step_size: 0.3
    max_recovery_attempts: 3

# Server configuration for behavior tree nodes
bt_navigator:
  ros__parameters:
    use_sim_time: False
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_bt_xml_filename: "humanoid_navigation_tree.xml"
    plugin_lib_names: [
      "bt_navigator/wait",
      "bt_navigator/spin",
      "bt_navigator/back_up",
      "bt_navigator/unicycle_controller",
      "nav2_compute_path_to_pose_action_bt_node",
      "nav2_follow_path_action_bt_node",
      "nav2_back_up_action_bt_node",
      "nav2_spin_action_bt_node",
      "nav2_wait_action_bt_node",
      "nav2_clear_costmap_service_bt_node",
      "nav2_is_path_valid_condition_bt_node",
      "nav2_is_stuck_condition_bt_node",
      "nav2_is_goal_reached_condition_bt_node",
      "nav2_is_path_blocked_condition_bt_node",
      "nav2_reinitialize_global_localization_service_bt_node",
      "nav2_rate_controller_bt_node",
      "nav2_distance_controller_bt_node",
      "nav2_speed_controller_bt_node",
      "nav2_truncate_path_action_bt_node",
      "nav2_goal_updater_node_bt_node",
      "nav2_recovery_node_bt_node",
      "nav2_pipeline_sequence_bt_node",
      "nav2_round_robin_node_bt_node",
      "nav2_transform_available_condition_bt_node",
      "nav2_time_expired_condition_bt_node",
      "nav2_distance_traveled_condition_bt_node",
      "humanoid_balance_check_condition",
      "humanoid_step_execution_action",
      "humanoid_balance_recovery_action"
    ]
```

## Humanoid-Specific Nodes Implementation

### Balance Check Condition Node

```cpp
// Condition node to check robot balance status
class BalanceCheckCondition : public BT::ConditionNode
{
public:
    BalanceCheckCondition(const std::string& name, const BT::NodeConfiguration& config)
        : BT::ConditionNode(name, config)
    {
        node_ = rclcpp::Node::make_shared("balance_check_condition");
        balance_subscriber_ = node_->create_subscription<std_msgs::msg::Bool>(
            "/balance_status", 10,
            std::bind(&BalanceCheckCondition::balanceCallback, this, std::placeholders::_1)
        );
    }

    static BT::PortsList providedPorts()
    {
        return {
            BT::OutputPort<bool>("balance_status", "Current balance status")
        };
    }

private:
    BT::NodeStatus tick() override
    {
        // Update balance status in blackboard if port is provided
        if (getOutput<bool>("balance_status", balance_status_)) {
            setOutput("balance_status", balance_status_);
        }

        // Return status based on balance
        return balance_status_ ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    }

    void balanceCallback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        balance_status_ = msg->data;
    }

    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr balance_subscriber_;
    bool balance_status_ = true;
};
```

### Gait Adaptation Node

```cpp
// Action node for gait adaptation based on terrain
class GaitAdaptationNode : public BT::StatefulActionNode
{
public:
    GaitAdaptationNode(const std::string& name, const BT::NodeConfiguration& config)
        : BT::StatefulActionNode(name, config)
    {
        node_ = rclcpp::Node::make_shared("gait_adaptation_node");
        gait_publisher_ = node_->create_publisher<std_msgs::msg::String>("/gait_command", 10);
        terrain_subscriber_ = node_->create_subscription<std_msgs::msg::String>(
            "/terrain_classification", 10,
            std::bind(&GaitAdaptationNode::terrainCallback, this, std::placeholders::_1)
        );
    }

    static BT::PortsList providedPorts()
    {
        return {
            BT::InputPort<std::string>("preferred_gait", "Preferred gait type")
        };
    }

private:
    BT::NodeStatus onStart() override
    {
        std::string preferred_gait = "walking";
        if (getInput<std::string>("preferred_gait", preferred_gait)) {
            RCLCPP_INFO(node_->get_logger(), "Adapting to preferred gait: %s", preferred_gait.c_str());
        }

        // Determine appropriate gait based on terrain and preferences
        std::string selected_gait = determineGait(preferred_gait, current_terrain_);

        // Publish gait command
        auto gait_msg = std_msgs::msg::String();
        gait_msg.data = selected_gait;
        gait_publisher_->publish(gait_msg);

        gait_adaptation_start_time_ = node_->get_clock()->now();
        gait_adapted_ = false;

        return BT::NodeStatus::RUNNING;
    }

    BT::NodeStatus onRunning() override
    {
        // Check if gait adaptation has been confirmed
        if (gait_adapted_) {
            return BT::NodeStatus::SUCCESS;
        }

        // Check for timeout
        auto elapsed = node_->get_clock()->now() - gait_adaptation_start_time_;
        if (elapsed.seconds() > 3.0) { // 3 second timeout
            return BT::NodeStatus::FAILURE;
        }

        return BT::NodeStatus::RUNNING;
    }

    void onHalted() override
    {
        RCLCPP_INFO(node_->get_logger(), "Gait adaptation halted");
    }

    std::string determineGait(const std::string& preferred_gait, const std::string& terrain)
    {
        if (terrain == "rough" || terrain == "uneven") {
            return "careful";
        } else if (terrain == "smooth" && preferred_gait == "fast") {
            return "fast";
        } else {
            return "walking";
        }
    }

    void terrainCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        current_terrain_ = msg->data;
    }

    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr gait_publisher_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr terrain_subscriber_;

    std::string current_terrain_ = "unknown";
    rclcpp::Time gait_adaptation_start_time_;
    bool gait_adapted_ = false;
};
```

## Integration with Navigation System

### Behavior Tree Integration Example

```python
# Example Python node integrating behavior tree with navigation
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateWithRecovery
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import time

class HumanoidBTNavigator(Node):
    def __init__(self):
        super().__init__('humanoid_bt_navigator')

        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateWithRecovery, 'navigate_with_recovery')

        # Publishers and subscribers
        self.balance_sub = self.create_subscription(
            Bool, '/balance_status', self.balance_callback, 10
        )
        self.gait_pub = self.create_publisher(
            String, '/gait_command', 10
        )

        # Navigation parameters
        self.declare_parameter('max_navigation_time', 300.0)  # 5 minutes
        self.declare_parameter('balance_threshold', 0.8)
        self.declare_parameter('recovery_enabled', True)

        self.current_balance = 1.0
        self.navigation_in_progress = False

    def balance_callback(self, msg):
        """
        Update current balance status
        """
        self.current_balance = msg.data

    def navigate_to_pose(self, pose):
        """
        Navigate to specified pose using behavior tree
        """
        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return False

        # Create goal
        goal_msg = NavigateWithRecovery.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = self.get_parameter('default_bt_xml_filename').value

        # Send goal
        self.navigation_in_progress = True
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.navigation_result_callback)

        return True

    def navigation_result_callback(self, future):
        """
        Handle navigation result
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Navigation goal rejected')
            self.navigation_in_progress = False
            return

        self.get_logger().info('Navigation goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_complete_callback)

    def navigation_complete_callback(self, future):
        """
        Handle navigation completion
        """
        result = future.result().result
        self.get_logger().info(f'Navigation completed with result: {result}')
        self.navigation_in_progress = False

    def adapt_gait_for_terrain(self, terrain_type):
        """
        Adapt gait based on terrain type
        """
        gait_map = {
            'rough': 'careful',
            'smooth': 'walking',
            'stairs': 'cautious',
            'ramp': 'controlled'
        }

        selected_gait = gait_map.get(terrain_type, 'walking')

        gait_msg = String()
        gait_msg.data = selected_gait
        self.gait_pub.publish(gait_msg)

        self.get_logger().info(f'Adapted gait to: {selected_gait} for terrain: {terrain_type}')
```

## Best Practices

### Tree Design Best Practices

1. **Modularity**: Design behavior trees with modular components that can be reused
2. **Error Handling**: Include proper error handling and recovery mechanisms
3. **Balance Safety**: Always prioritize balance and safety in navigation decisions
4. **Performance**: Optimize tree structure to minimize computational overhead
5. **Debugging**: Enable logging and monitoring for easier debugging

### Configuration Best Practices

1. **Parameter Tuning**: Carefully tune parameters for your specific humanoid robot
2. **Validation**: Test behavior trees in simulation before real robot deployment
3. **Monitoring**: Implement monitoring to track tree execution and performance
4. **Fallbacks**: Always include fallback behaviors for safety-critical operations

## Troubleshooting

### Common Issues

- **Tree loops**: Ensure proper termination conditions to prevent infinite loops
- **Balance failures**: Implement proper balance checking and recovery
- **Step execution timeouts**: Adjust timeout parameters based on robot capabilities
- **Memory issues**: Monitor memory usage with complex behavior trees

### Debugging Strategies

```bash
# Monitor behavior tree execution
ros2 run nav2_msgs bt_tree_logger

# Visualize tree with Groot
ros2 run groot_bt groot_monitor

# Check navigation logs
ros2 param set bt_navigator enable_bt_tree_logging true
```

This comprehensive guide provides the configuration and implementation details needed for creating behavior trees specifically adapted for humanoid robot navigation using Nav2.