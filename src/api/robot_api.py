#!/usr/bin/env python3

"""
Robot Control API for Gazebo-Unity Integration

This module provides REST API endpoints for controlling robots in simulation
and managing robot states between Gazebo and Unity.
"""

from flask import Flask, request, jsonify
import subprocess
import json
import os
from datetime import datetime


class RobotAPI:
    """
    API class for controlling robots in Gazebo simulation.
    """

    def __init__(self):
        self.app = Flask(__name__)
        self.robots = {}  # Store robot information
        self.robot_states = {}  # Store current robot states

        # Register API routes
        self.register_routes()

    def register_routes(self):
        """Register all API routes."""
        # Robot management endpoints
        self.app.add_url_rule('/api/robots', 'list_robots', self.list_robots, methods=['GET'])
        self.app.add_url_rule('/api/robots', 'create_robot', self.create_robot, methods=['POST'])
        self.app.add_url_rule('/api/robot/<robot_id>', 'get_robot', self.get_robot, methods=['GET'])
        self.app.add_url_rule('/api/robot/<robot_id>', 'update_robot', self.update_robot, methods=['PUT'])
        self.app.add_url_rule('/api/robot/<robot_id>', 'delete_robot', self.delete_robot, methods=['DELETE'])

        # Joint control endpoints
        self.app.add_url_rule('/api/robot/<robot_id>/joints', 'get_joint_positions', self.get_joint_positions, methods=['GET'])
        self.app.add_url_rule('/api/robot/<robot_id>/joints', 'set_joint_positions', self.set_joint_positions, methods=['POST'])
        self.app.add_url_rule('/api/robot/<robot_id>/joints/trajectory', 'send_trajectory', self.send_trajectory, methods=['POST'])

        # Robot movement endpoints
        self.app.add_url_rule('/api/robot/<robot_id>/move', 'move_robot', self.move_robot, methods=['POST'])
        self.app.add_url_rule('/api/robot/<robot_id>/pose', 'get_robot_pose', self.get_robot_pose, methods=['GET'])
        self.app.add_url_rule('/api/robot/<robot_id>/pose', 'set_robot_pose', self.set_robot_pose, methods=['POST'])

        # Robot configuration endpoints
        self.app.add_url_rule('/api/robot/<robot_id>/config', 'get_robot_config', self.get_robot_config, methods=['GET'])
        self.app.add_url_rule('/api/robot/<robot_id>/config', 'update_robot_config', self.update_robot_config, methods=['PUT'])

    def list_robots(self):
        """List all available robots."""
        try:
            robot_list = []
            for robot_id, robot_info in self.robots.items():
                robot_list.append({
                    'id': robot_id,
                    'name': robot_info.get('name', robot_id),
                    'type': robot_info.get('type', 'unknown'),
                    'status': robot_info.get('status', 'inactive'),
                    'last_updated': robot_info.get('last_updated', None)
                })

            return jsonify({'robots': robot_list})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def create_robot(self):
        """Create a new robot in the simulation."""
        try:
            data = request.get_json()
            robot_id = data.get('id', f"robot_{int(datetime.now().timestamp())}")
            name = data.get('name', f"Robot_{robot_id}")
            robot_type = data.get('type', 'humanoid')
            urdf_path = data.get('urdf_path', '')
            initial_pose = data.get('initial_pose', {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0})

            # Validate inputs
            if not urdf_path:
                return jsonify({'error': 'urdf_path is required'}), 400

            if not os.path.exists(urdf_path):
                return jsonify({'error': f'URDF file not found: {urdf_path}'}), 404

            # Create robot entry
            self.robots[robot_id] = {
                'id': robot_id,
                'name': name,
                'type': robot_type,
                'urdf_path': urdf_path,
                'initial_pose': initial_pose,
                'status': 'active',
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }

            # Initialize robot state
            self.robot_states[robot_id] = {
                'pose': initial_pose,
                'joint_positions': {},
                'joint_velocities': {},
                'joint_efforts': {}
            }

            return jsonify({
                'robot_id': robot_id,
                'name': name,
                'type': robot_type,
                'status': 'active',
                'created_at': self.robots[robot_id]['created_at']
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_robot(self, robot_id):
        """Get information about a specific robot."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            robot_info = self.robots[robot_id].copy()
            # Add current state information
            if robot_id in self.robot_states:
                robot_info['current_state'] = self.robot_states[robot_id]

            return jsonify(robot_info)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def update_robot(self, robot_id):
        """Update robot configuration."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            data = request.get_json()
            # Update allowed fields
            updatable_fields = ['name', 'type', 'status']
            for field in updatable_fields:
                if field in data:
                    self.robots[robot_id][field] = data[field]

            self.robots[robot_id]['last_updated'] = datetime.now().isoformat()

            return jsonify({
                'robot_id': robot_id,
                'updated_fields': data,
                'last_updated': self.robots[robot_id]['last_updated']
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def delete_robot(self, robot_id):
        """Delete a robot from the simulation."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            # Remove robot and its state
            del self.robots[robot_id]
            if robot_id in self.robot_states:
                del self.robot_states[robot_id]

            return jsonify({
                'robot_id': robot_id,
                'status': 'deleted'
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_joint_positions(self, robot_id):
        """Get current joint positions for a robot."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            # In a real implementation, this would get data from ROS/Gazebo
            # For simulation, return sample data based on robot type
            if robot_id in self.robot_states and 'joint_positions' in self.robot_states[robot_id]:
                joint_positions = self.robot_states[robot_id]['joint_positions']
            else:
                # Default joint positions based on robot type
                if self.robots[robot_id]['type'] == 'humanoid':
                    joint_positions = {
                        'left_shoulder': 0.0,
                        'left_elbow': 0.0,
                        'right_shoulder': 0.0,
                        'right_elbow': 0.0,
                        'left_hip': 0.0,
                        'left_knee': 0.0,
                        'right_hip': 0.0,
                        'right_knee': 0.0
                    }
                else:
                    joint_positions = {}

            response = {
                'robot_id': robot_id,
                'timestamp': datetime.now().isoformat(),
                'joint_positions': joint_positions
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def set_joint_positions(self, robot_id):
        """Set joint positions for a robot."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            data = request.get_json()
            joint_positions = data.get('joint_positions', {})
            duration = data.get('duration', 1.0)

            # Validate joint positions
            if not isinstance(joint_positions, dict):
                return jsonify({'error': 'joint_positions must be a dictionary'}), 400

            # Update robot state
            if robot_id not in self.robot_states:
                self.robot_states[robot_id] = {}

            self.robot_states[robot_id]['joint_positions'] = joint_positions
            self.robot_states[robot_id]['last_updated'] = datetime.now().isoformat()

            response = {
                'robot_id': robot_id,
                'success': True,
                'executed_at': datetime.now().isoformat(),
                'joint_positions': joint_positions,
                'duration': duration,
                'errors': []
            }

            # In a real implementation, this would send commands to ROS/Gazebo
            # For now, we just return success

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def send_trajectory(self, robot_id):
        """Send a trajectory command to the robot."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            data = request.get_json()
            joint_names = data.get('joint_names', [])
            points = data.get('points', [])
            duration = data.get('duration', 5.0)

            if not joint_names or not points:
                return jsonify({'error': 'joint_names and points are required'}), 400

            # Validate trajectory points
            for point in points:
                if 'positions' not in point or len(point['positions']) != len(joint_names):
                    return jsonify({'error': 'Each point must have positions matching joint_names length'}), 400

            # In a real implementation, this would send trajectory to ROS
            # For simulation, just return success
            response = {
                'robot_id': robot_id,
                'success': True,
                'joint_names': joint_names,
                'points_count': len(points),
                'duration': duration,
                'executed_at': datetime.now().isoformat()
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def move_robot(self, robot_id):
        """Move the robot to a specific position."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            data = request.get_json()
            target_position = data.get('position', {})
            target_orientation = data.get('orientation', {})

            if not target_position:
                return jsonify({'error': 'position is required'}), 400

            # Update robot pose in state
            if robot_id not in self.robot_states:
                self.robot_states[robot_id] = {}

            new_pose = self.robot_states[robot_id].get('pose', {}).copy()
            new_pose.update(target_position)
            if target_orientation:
                new_pose.update(target_orientation)

            self.robot_states[robot_id]['pose'] = new_pose
            self.robot_states[robot_id]['last_updated'] = datetime.now().isoformat()

            response = {
                'robot_id': robot_id,
                'success': True,
                'target_position': target_position,
                'target_orientation': target_orientation,
                'executed_at': datetime.now().isoformat()
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_robot_pose(self, robot_id):
        """Get the current pose of the robot."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            if robot_id in self.robot_states and 'pose' in self.robot_states[robot_id]:
                pose = self.robot_states[robot_id]['pose']
            else:
                # Default pose
                pose = {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0}

            response = {
                'robot_id': robot_id,
                'timestamp': datetime.now().isoformat(),
                'pose': pose
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def set_robot_pose(self, robot_id):
        """Set the pose of the robot."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            data = request.get_json()
            new_pose = data.get('pose', {})

            if not new_pose:
                return jsonify({'error': 'pose is required'}), 400

            # Update robot pose in state
            if robot_id not in self.robot_states:
                self.robot_states[robot_id] = {}

            self.robot_states[robot_id]['pose'] = new_pose
            self.robot_states[robot_id]['last_updated'] = datetime.now().isoformat()

            response = {
                'robot_id': robot_id,
                'success': True,
                'new_pose': new_pose,
                'executed_at': datetime.now().isoformat()
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_robot_config(self, robot_id):
        """Get robot configuration."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            robot_config = {
                'id': robot_id,
                'name': self.robots[robot_id].get('name'),
                'type': self.robots[robot_id].get('type'),
                'urdf_path': self.robots[robot_id].get('urdf_path'),
                'initial_pose': self.robots[robot_id].get('initial_pose'),
                'joints': self.get_robot_joints(robot_id)
            }

            return jsonify(robot_config)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def update_robot_config(self, robot_id):
        """Update robot configuration."""
        try:
            if robot_id not in self.robots:
                return jsonify({'error': 'Robot not found'}), 404

            data = request.get_json()
            updatable_fields = ['name', 'urdf_path']

            for field in updatable_fields:
                if field in data:
                    self.robots[robot_id][field] = data[field]

            self.robots[robot_id]['last_updated'] = datetime.now().isoformat()

            return jsonify({
                'robot_id': robot_id,
                'updated_fields': list(data.keys()),
                'last_updated': self.robots[robot_id]['last_updated']
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_robot_joints(self, robot_id):
        """Get joint information for a robot (helper method)."""
        # In a real implementation, this would parse the URDF
        # For simulation, return sample joints based on robot type
        if self.robots[robot_id]['type'] == 'humanoid':
            return [
                {'name': 'left_shoulder', 'type': 'revolute', 'limits': {'lower': -1.57, 'upper': 1.57}},
                {'name': 'left_elbow', 'type': 'revolute', 'limits': {'lower': -1.57, 'upper': 1.57}},
                {'name': 'right_shoulder', 'type': 'revolute', 'limits': {'lower': -1.57, 'upper': 1.57}},
                {'name': 'right_elbow', 'type': 'revolute', 'limits': {'lower': -1.57, 'upper': 1.57}},
                {'name': 'left_hip', 'type': 'revolute', 'limits': {'lower': -1.57, 'upper': 1.57}},
                {'name': 'left_knee', 'type': 'revolute', 'limits': {'lower': -1.57, 'upper': 1.57}},
                {'name': 'right_hip', 'type': 'revolute', 'limits': {'lower': -1.57, 'upper': 1.57}},
                {'name': 'right_knee', 'type': 'revolute', 'limits': {'lower': -1.57, 'upper': 1.57}}
            ]
        else:
            return []

    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the Flask application."""
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function to run the Robot API."""
    api = RobotAPI()
    print("Starting Robot Control API...")
    print("Available endpoints:")
    print("  GET    /api/robots                    - List all robots")
    print("  POST   /api/robots                    - Create a new robot")
    print("  GET    /api/robot/<id>                - Get robot info")
    print("  POST   /api/robot/<id>/joints         - Set joint positions")
    print("  GET    /api/robot/<id>/joints         - Get joint positions")
    print("  POST   /api/robot/<id>/move           - Move robot")

    api.run(debug=True)


if __name__ == '__main__':
    main()