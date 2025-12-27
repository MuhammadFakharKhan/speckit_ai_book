#!/usr/bin/env python3

"""
Simulation Management API for Gazebo-Unity Integration

This module provides REST API endpoints for managing Gazebo simulations
and coordinating with Unity visualization.
"""

from flask import Flask, request, jsonify
import subprocess
import json
import os
import signal
import psutil
from datetime import datetime


class SimulationAPI:
    """
    API class for managing Gazebo simulations and Unity integration.
    """

    def __init__(self):
        self.app = Flask(__name__)
        self.simulations = {}  # Store active simulations
        self.gazebo_processes = {}  # Track Gazebo processes

        # Register API routes
        self.register_routes()

    def register_routes(self):
        """Register all API routes."""
        # Simulation management endpoints
        self.app.add_url_rule('/api/simulation/start', 'start_simulation', self.start_simulation, methods=['POST'])
        self.app.add_url_rule('/api/simulation/stop', 'stop_simulation', self.stop_simulation, methods=['POST'])
        self.app.add_url_rule('/api/simulations', 'list_simulations', self.list_simulations, methods=['GET'])
        self.app.add_url_rule('/api/simulation/<simulation_id>', 'get_simulation', self.get_simulation, methods=['GET'])
        self.app.add_url_rule('/api/simulation/<simulation_id>', 'update_simulation', self.update_simulation, methods=['PUT'])

        # Robot control endpoints
        self.app.add_url_rule('/api/robot/<robot_id>/joints', 'set_joint_positions', self.set_joint_positions, methods=['POST'])
        self.app.add_url_rule('/api/robot/<robot_id>/joints', 'get_joint_positions', self.get_joint_positions, methods=['GET'])

        # Sensor data endpoints
        self.app.add_url_rule('/api/robot/<robot_id>/sensors', 'get_all_sensors', self.get_all_sensors, methods=['GET'])
        self.app.add_url_rule('/api/robot/<robot_id>/sensors/<sensor_type>', 'get_sensor_data', self.get_sensor_data, methods=['GET'])

        # Unity integration endpoints
        self.app.add_url_rule('/api/unity/scenes/load', 'load_unity_scene', self.load_unity_scene, methods=['POST'])
        self.app.add_url_rule('/api/unity/visualization/update', 'update_visualization', self.update_visualization, methods=['POST'])

        # Simulation profile endpoints
        self.app.add_url_rule('/api/simulation/profiles', 'list_profiles', self.list_profiles, methods=['GET'])
        self.app.add_url_rule('/api/simulation/profiles', 'create_profile', self.create_profile, methods=['POST'])

    def start_simulation(self):
        """Start a Gazebo simulation environment."""
        try:
            data = request.get_json()
            simulation_id = data.get('simulation_id', f"sim_{int(datetime.now().timestamp())}")
            world_file = data.get('world_file', 'basic_humanoid.sdf')
            profile = data.get('profile', 'default')
            robot_models = data.get('robot_models', [])
            real_time_factor = data.get('real_time_factor', 1.0)

            # Validate inputs
            if not world_file:
                return jsonify({'error': 'world_file is required'}), 400

            # Construct Gazebo command
            world_path = f"examples/gazebo/worlds/{world_file}"
            if not os.path.exists(world_path):
                return jsonify({'error': f'World file not found: {world_path}'}), 404

            # Start Gazebo process
            cmd = ['gz', 'sim', '-r', world_path]
            process = subprocess.Popen(cmd)

            # Store simulation info
            self.simulations[simulation_id] = {
                'id': simulation_id,
                'status': 'running',
                'start_time': datetime.now().isoformat(),
                'world_file': world_file,
                'profile': profile,
                'robot_models': robot_models,
                'real_time_factor': real_time_factor,
                'gazebo_pid': process.pid,
                'gazebo_port': 0,  # Gazebo doesn't use a specific port for this
                'ros_bridge_port': 9090  # Default rosbridge port
            }

            self.gazebo_processes[simulation_id] = process

            return jsonify({
                'simulation_id': simulation_id,
                'status': 'running',
                'start_time': self.simulations[simulation_id]['start_time'],
                'gazebo_pid': process.pid,
                'ros_bridge_port': 9090
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def stop_simulation(self):
        """Stop a running Gazebo simulation."""
        try:
            data = request.get_json()
            simulation_id = data.get('simulation_id')

            if not simulation_id or simulation_id not in self.simulations:
                return jsonify({'error': 'Simulation ID not found'}), 404

            # Get the process
            if simulation_id in self.gazebo_processes:
                process = self.gazebo_processes[simulation_id]
                try:
                    # Terminate the process
                    process.terminate()
                    process.wait(timeout=5)  # Wait up to 5 seconds
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    process.kill()

                # Remove from tracking
                del self.gazebo_processes[simulation_id]

            # Update simulation status
            self.simulations[simulation_id]['status'] = 'stopped'
            self.simulations[simulation_id]['stop_time'] = datetime.now().isoformat()

            return jsonify({
                'simulation_id': simulation_id,
                'status': 'stopped',
                'stop_time': self.simulations[simulation_id]['stop_time']
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def list_simulations(self):
        """List all available simulation environments."""
        try:
            active_sims = []
            for sim_id, sim_info in self.simulations.items():
                # Check if process is still running
                if sim_id in self.gazebo_processes:
                    process = self.gazebo_processes[sim_id]
                    if process.poll() is not None:  # Process has terminated
                        sim_info['status'] = 'stopped'
                        del self.gazebo_processes[sim_id]

                active_sims.append({
                    'id': sim_info['id'],
                    'name': sim_info['world_file'].replace('.sdf', ''),
                    'description': f"Simulation running {sim_info['world_file']}",
                    'world_file_path': f"examples/gazebo/worlds/{sim_info['world_file']}",
                    'supported_profiles': ['default', 'high_fidelity', 'performance'],
                    'robot_models': sim_info['robot_models'],
                    'status': sim_info['status']
                })

            return jsonify({'simulations': active_sims})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_simulation(self, simulation_id):
        """Get details for a specific simulation."""
        try:
            if simulation_id not in self.simulations:
                return jsonify({'error': 'Simulation not found'}), 404

            sim_info = self.simulations[simulation_id].copy()
            # Remove sensitive information
            if 'gazebo_pid' in sim_info:
                del sim_info['gazebo_pid']

            return jsonify(sim_info)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def set_joint_positions(self, robot_id):
        """Set joint positions for a robot."""
        try:
            data = request.get_json()
            joint_positions = data.get('joint_positions', {})
            duration = data.get('duration', 1.0)

            # In a real implementation, this would communicate with ROS
            # For simulation, we'll just return success
            response = {
                'success': True,
                'executed_at': datetime.now().isoformat(),
                'robot_id': robot_id,
                'joint_positions': joint_positions,
                'errors': []
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_joint_positions(self, robot_id):
        """Get current joint positions for a robot."""
        try:
            # In a real implementation, this would get data from ROS
            # For simulation, return sample data
            sample_positions = {
                'left_shoulder': 0.0,
                'left_elbow': 0.0,
                'right_shoulder': 0.0,
                'right_elbow': 0.0,
                'left_hip': 0.0,
                'left_knee': 0.0,
                'right_hip': 0.0,
                'right_knee': 0.0
            }

            response = {
                'robot_id': robot_id,
                'timestamp': datetime.now().isoformat(),
                'joint_positions': sample_positions
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_all_sensors(self, robot_id):
        """Get all sensor data from a robot."""
        try:
            # Simulate sensor data
            sensor_data = {
                'robot_id': robot_id,
                'timestamp': datetime.now().isoformat(),
                'sensors': {
                    'camera': {
                        'topic': '/camera/image_raw',
                        'data_available': True,
                        'resolution': [640, 480]
                    },
                    'lidar': {
                        'topic': '/lidar/scan',
                        'data_available': True,
                        'range_count': 360
                    },
                    'imu': {
                        'topic': '/imu/data',
                        'data_available': True,
                        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
                    }
                }
            }

            return jsonify(sensor_data)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_sensor_data(self, robot_id, sensor_type):
        """Get specific sensor data from a robot."""
        try:
            # Simulate sensor-specific data
            if sensor_type == 'camera':
                data = {
                    'robot_id': robot_id,
                    'sensor_type': sensor_type,
                    'timestamp': datetime.now().isoformat(),
                    'data': 'simulated_camera_data',
                    'resolution': [640, 480]
                }
            elif sensor_type == 'lidar':
                data = {
                    'robot_id': robot_id,
                    'sensor_type': sensor_type,
                    'timestamp': datetime.now().isoformat(),
                    'data': [1.0] * 360,  # Simulated distance readings
                    'range_count': 360
                }
            elif sensor_type == 'imu':
                data = {
                    'robot_id': robot_id,
                    'sensor_type': sensor_type,
                    'timestamp': datetime.now().isoformat(),
                    'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                    'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'linear_acceleration': {'x': 0.0, 'y': 0.0, 'z': -9.81}
                }
            else:
                return jsonify({'error': f'Unknown sensor type: {sensor_type}'}), 404

            return jsonify(data)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def load_unity_scene(self):
        """Load a Unity visualization scene."""
        try:
            data = request.get_json()
            scene_name = data.get('scene_name')
            simulation_id = data.get('simulation_id')
            robot_config = data.get('robot_config', {})

            if not scene_name:
                return jsonify({'error': 'scene_name is required'}), 400

            # Simulate loading a Unity scene
            response = {
                'scene_id': f"scene_{scene_name}_{int(datetime.now().timestamp())}",
                'status': 'loaded',
                'scene_name': scene_name,
                'simulation_id': simulation_id,
                'visualization_url': f"http://localhost:3000/unity/{scene_name}",
                'loaded_at': datetime.now().isoformat()
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def update_visualization(self):
        """Update visualization with current robot state."""
        try:
            data = request.get_json()
            scene_id = data.get('scene_id')
            robot_states = data.get('robot_states', [])

            if not scene_id:
                return jsonify({'error': 'scene_id is required'}), 400

            # Simulate updating Unity visualization
            response = {
                'success': True,
                'updated_at': datetime.now().isoformat(),
                'scene_id': scene_id,
                'robot_states_updated': len(robot_states)
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def list_profiles(self):
        """Get available simulation profiles."""
        try:
            profiles = [
                {
                    'name': 'default',
                    'description': 'Balanced performance and quality',
                    'physics_settings': {
                        'engine': 'ode',
                        'time_step': 0.001,
                        'real_time_factor': 1.0
                    },
                    'visual_settings': {
                        'quality': 'medium',
                        'rendering': 'realistic'
                    },
                    'hardware_requirements': {
                        'cpu_cores': 4,
                        'memory_gb': 8,
                        'gpu': 'Integrated'
                    },
                    'use_case': 'General simulation'
                },
                {
                    'name': 'high_fidelity',
                    'description': 'High quality physics and visuals',
                    'physics_settings': {
                        'engine': 'ode',
                        'time_step': 0.0005,
                        'real_time_factor': 0.5
                    },
                    'visual_settings': {
                        'quality': 'high',
                        'rendering': 'photorealistic'
                    },
                    'hardware_requirements': {
                        'cpu_cores': 8,
                        'memory_gb': 16,
                        'gpu': 'Dedicated'
                    },
                    'use_case': 'Research and validation'
                },
                {
                    'name': 'performance',
                    'description': 'Optimized for speed over quality',
                    'physics_settings': {
                        'engine': 'ode',
                        'time_step': 0.01,
                        'real_time_factor': 2.0
                    },
                    'visual_settings': {
                        'quality': 'low',
                        'rendering': 'fast'
                    },
                    'hardware_requirements': {
                        'cpu_cores': 2,
                        'memory_gb': 4,
                        'gpu': 'Integrated'
                    },
                    'use_case': 'Fast iteration and testing'
                }
            ]

            return jsonify({'profiles': profiles})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def create_profile(self):
        """Create a new simulation profile."""
        try:
            data = request.get_json()
            name = data.get('name')
            description = data.get('description', '')
            physics_settings = data.get('physics_settings', {})
            visual_settings = data.get('visual_settings', {})
            hardware_requirements = data.get('hardware_requirements', {})
            use_case = data.get('use_case', 'Custom profile')

            if not name:
                return jsonify({'error': 'name is required'}), 400

            # In a real implementation, this would save the profile
            profile = {
                'name': name,
                'description': description,
                'physics_settings': physics_settings,
                'visual_settings': visual_settings,
                'hardware_requirements': hardware_requirements,
                'use_case': use_case
            }

            return jsonify({'profile': profile})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application."""
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function to run the Simulation API."""
    api = SimulationAPI()
    print("Starting Simulation Management API...")
    print("Available endpoints:")
    print("  POST   /api/simulation/start   - Start a simulation")
    print("  POST   /api/simulation/stop    - Stop a simulation")
    print("  GET    /api/simulations        - List all simulations")
    print("  POST   /api/robot/<id>/joints  - Control robot joints")
    print("  GET    /api/robot/<id>/sensors - Get sensor data")
    print("  POST   /api/unity/scenes/load  - Load Unity scene")

    api.run(debug=True)


if __name__ == '__main__':
    main()