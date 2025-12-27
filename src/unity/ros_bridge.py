"""
ROS Bridge for Unity Integration

This module provides the interface between ROS 2 and Unity for robot state synchronization.
It includes functions to receive robot state data from ROS 2 topics and format it for Unity.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
import json
import threading
from typing import Dict, List, Optional, Callable
import time


class UnityROSBridge(Node):
    """
    ROS 2 node that bridges data between ROS 2 and Unity
    """

    def __init__(self):
        super().__init__('unity_ros_bridge')

        # Robot state storage
        self.robot_states = {
            'joint_states': {},
            'poses': {},
            'velocities': {},
            'tf_transforms': {}
        }

        # Callback functions for Unity integration
        self.unity_callbacks = {}

        # Create subscribers
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.tf_subscriber = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )

        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Timer for publishing Unity updates
        self.timer = self.create_timer(0.05, self.publish_to_unity)  # 20 Hz update rate

        self.get_logger().info('Unity ROS Bridge initialized')

    def joint_state_callback(self, msg: JointState):
        """Handle joint state messages from ROS 2"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                position = msg.position[i] if i < len(msg.position) else 0.0
                velocity = msg.velocity[i] if i < len(msg.velocity) else 0.0
                effort = msg.effort[i] if i < len(msg.effort) else 0.0

                self.robot_states['joint_states'][name] = {
                    'position': position,
                    'velocity': velocity,
                    'effort': effort,
                    'timestamp': time.time()
                }

        # Trigger Unity callback if registered
        if 'joint_states' in self.unity_callbacks:
            self.unity_callbacks['joint_states'](self.robot_states['joint_states'])

    def tf_callback(self, msg: TFMessage):
        """Handle transform messages from ROS 2"""
        for transform in msg.transforms:
            child_frame = transform.child_frame_id
            transform_data = {
                'translation': {
                    'x': transform.transform.translation.x,
                    'y': transform.transform.translation.y,
                    'z': transform.transform.translation.z
                },
                'rotation': {
                    'x': transform.transform.rotation.x,
                    'y': transform.transform.rotation.y,
                    'z': transform.transform.rotation.z,
                    'w': transform.transform.rotation.w
                },
                'timestamp': time.time()
            }

            self.robot_states['tf_transforms'][child_frame] = transform_data

        # Trigger Unity callback if registered
        if 'tf_transforms' in self.unity_callbacks:
            self.unity_callbacks['tf_transforms'](self.robot_states['tf_transforms'])

    def cmd_vel_callback(self, msg: Twist):
        """Handle velocity commands"""
        velocity_data = {
            'linear': {
                'x': msg.linear.x,
                'y': msg.linear.y,
                'z': msg.linear.z
            },
            'angular': {
                'x': msg.angular.x,
                'y': msg.angular.y,
                'z': msg.angular.z
            },
            'timestamp': time.time()
        }

        self.robot_states['velocities']['cmd_vel'] = velocity_data

        # Trigger Unity callback if registered
        if 'cmd_vel' in self.unity_callbacks:
            self.unity_callbacks['cmd_vel'](velocity_data)

    def register_unity_callback(self, callback_type: str, callback_func: Callable):
        """Register a callback function for Unity integration"""
        self.unity_callbacks[callback_type] = callback_func

    def get_robot_state(self, component: str = None) -> Dict:
        """Get current robot state, optionally for a specific component"""
        if component:
            return self.robot_states.get(component, {})
        return self.robot_states

    def get_joint_positions(self) -> Dict[str, float]:
        """Get current joint positions"""
        positions = {}
        for joint_name, joint_data in self.robot_states['joint_states'].items():
            positions[joint_name] = joint_data['position']
        return positions

    def get_robot_pose(self, frame: str = 'base_link') -> Optional[Dict]:
        """Get robot pose for a specific frame"""
        return self.robot_states['tf_transforms'].get(frame)

    def publish_to_unity(self):
        """Publish updated robot state to Unity (placeholder for actual implementation)"""
        # In a real implementation, this would send data to Unity via:
        # 1. TCP/UDP socket
        # 2. Shared memory
        # 3. ROS bridge like rosbridge_suite
        # 4. Custom protocol

        # For now, we'll just log the state
        joint_positions = self.get_joint_positions()
        if joint_positions:
            self.get_logger().debug(f'Joint positions: {list(joint_positions.keys())}')

    def format_for_unity(self, data: Dict) -> str:
        """Format data for Unity consumption"""
        return json.dumps(data, separators=(',', ':'))


def create_unity_message(robot_state: Dict) -> str:
    """
    Format robot state data for Unity consumption

    Args:
        robot_state: Dictionary containing robot state information

    Returns:
        JSON string formatted for Unity
    """
    unity_format = {
        'timestamp': time.time(),
        'robot_data': {
            'joint_states': {},
            'poses': robot_state.get('tf_transforms', {}),
            'velocities': robot_state.get('velocities', {})
        }
    }

    # Format joint states for Unity
    for joint_name, joint_data in robot_state.get('joint_states', {}).items():
        unity_format['robot_data']['joint_states'][joint_name] = {
            'position': joint_data['position'],
            'velocity': joint_data['velocity'],
            'effort': joint_data['effort']
        }

    return json.dumps(unity_format, separators=(',', ':'))


class UnityIntegrationManager:
    """
    Manager class to handle Unity-specific integration tasks
    """

    def __init__(self):
        self.ros_bridge = None
        self.unity_connected = False
        self.robot_model_mapping = {}

    def connect_to_ros(self):
        """Initialize connection to ROS 2 network"""
        if not rclpy.ok():
            rclpy.init()

        self.ros_bridge = UnityROSBridge()

        # Start ROS spinning in a separate thread
        self.ros_thread = threading.Thread(target=self._spin_ros)
        self.ros_thread.daemon = True
        self.ros_thread.start()

        self.unity_connected = True
        return True

    def _spin_ros(self):
        """Run ROS spinning in a separate thread"""
        rclpy.spin(self.ros_bridge)

    def get_robot_state_for_unity(self) -> str:
        """Get formatted robot state for Unity"""
        if not self.ros_bridge:
            return json.dumps({'error': 'ROS bridge not initialized'})

        robot_state = self.ros_bridge.get_robot_state()
        return create_unity_message(robot_state)

    def register_robot_model_mapping(self, ros_name: str, unity_name: str):
        """Register mapping between ROS joint names and Unity object names"""
        self.robot_model_mapping[ros_name] = unity_name

    def get_unity_joint_mapping(self, ros_joint_names: List[str]) -> Dict[str, str]:
        """Get Unity names for given ROS joint names"""
        mapping = {}
        for ros_name in ros_joint_names:
            unity_name = self.robot_model_mapping.get(ros_name, ros_name)  # Default to same name
            mapping[ros_name] = unity_name
        return mapping


# Example usage
def main():
    """Example usage of the Unity ROS Bridge"""
    if not rclpy.ok():
        rclpy.init()

    # Create bridge instance
    bridge = UnityROSBridge()

    # Create integration manager
    unity_manager = UnityIntegrationManager()
    unity_manager.ros_bridge = bridge

    # Register robot model mappings (ROS joint name -> Unity object name)
    unity_manager.register_robot_model_mapping('hip_joint', 'HipJoint')
    unity_manager.register_robot_model_mapping('knee_joint', 'KneeJoint')
    unity_manager.register_robot_model_mapping('ankle_joint', 'AnkleJoint')
    unity_manager.register_robot_model_mapping('shoulder_joint', 'ShoulderJoint')
    unity_manager.register_robot_model_mapping('elbow_joint', 'ElbowJoint')
    unity_manager.register_robot_model_mapping('wrist_joint', 'WristJoint')

    # Example of getting robot state formatted for Unity
    def unity_update_callback():
        unity_state = unity_manager.get_robot_state_for_unity()
        print(f"Unity state: {unity_state[:100]}...")  # Print first 100 chars

    # Set up periodic updates
    import time
    try:
        while True:
            unity_update_callback()
            time.sleep(0.1)  # Update at 10 Hz
    except KeyboardInterrupt:
        print("Shutting down Unity ROS Bridge")
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()