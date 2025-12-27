#!/usr/bin/env python3

"""
Sensor Data API for Gazebo-Unity Integration

This module provides REST API endpoints for accessing sensor data
from robots in Gazebo simulation.
"""

from flask import Flask, request, jsonify
import json
import os
from datetime import datetime
import random


class SensorAPI:
    """
    API class for accessing sensor data from simulated robots.
    """

    def __init__(self):
        self.app = Flask(__name__)
        self.sensors = {}  # Store sensor information
        self.sensor_data = {}  # Store latest sensor data
        self.robot_sensors = {}  # Map robots to their sensors

        # Register API routes
        self.register_routes()

    def register_routes(self):
        """Register all API routes."""
        # Sensor management endpoints
        self.app.add_url_rule('/api/robots/<robot_id>/sensors', 'list_robot_sensors', self.list_robot_sensors, methods=['GET'])
        self.app.add_url_rule('/api/robots/<robot_id>/sensors', 'add_robot_sensor', self.add_robot_sensor, methods=['POST'])
        self.app.add_url_rule('/api/sensors', 'list_sensors', self.list_sensors, methods=['GET'])

        # Individual sensor data endpoints
        self.app.add_url_rule('/api/robot/<robot_id>/sensors', 'get_all_sensor_data', self.get_all_sensor_data, methods=['GET'])
        self.app.add_url_rule('/api/robot/<robot_id>/sensors/<sensor_type>', 'get_sensor_data_by_type', self.get_sensor_data_by_type, methods=['GET'])
        self.app.add_url_rule('/api/sensor/<sensor_id>/data', 'get_sensor_data_by_id', self.get_sensor_data_by_id, methods=['GET'])

        # Sensor configuration endpoints
        self.app.add_url_rule('/api/sensor/<sensor_id>/config', 'get_sensor_config', self.get_sensor_config, methods=['GET'])
        self.app.add_url_rule('/api/sensor/<sensor_id>/config', 'update_sensor_config', self.update_sensor_config, methods=['PUT'])

        # Sensor status endpoints
        self.app.add_url_rule('/api/sensor/<sensor_id>/status', 'get_sensor_status', self.get_sensor_status, methods=['GET'])

    def list_robot_sensors(self, robot_id):
        """List all sensors for a specific robot."""
        try:
            if robot_id not in self.robot_sensors:
                return jsonify({'sensors': []})

            sensor_list = []
            for sensor_id in self.robot_sensors[robot_id]:
                if sensor_id in self.sensors:
                    sensor_info = self.sensors[sensor_id].copy()
                    # Remove sensitive data
                    if 'raw_config' in sensor_info:
                        del sensor_info['raw_config']
                    sensor_list.append(sensor_info)

            return jsonify({'sensors': sensor_list})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def add_robot_sensor(self, robot_id):
        """Add a sensor to a robot."""
        try:
            data = request.get_json()
            sensor_type = data.get('type')
            sensor_name = data.get('name', f"{sensor_type}_{int(datetime.now().timestamp())}")
            topic = data.get('topic', f"/{robot_id}/{sensor_type}")
            config = data.get('config', {})

            if not sensor_type:
                return jsonify({'error': 'sensor type is required'}), 400

            # Generate sensor ID
            sensor_id = f"{robot_id}_{sensor_type}_{sensor_name}"

            # Create sensor entry
            self.sensors[sensor_id] = {
                'id': sensor_id,
                'name': sensor_name,
                'type': sensor_type,
                'robot_id': robot_id,
                'topic': topic,
                'config': config,
                'raw_config': data,  # Keep original config for reference
                'status': 'active',
                'created_at': datetime.now().isoformat()
            }

            # Add to robot's sensor list
            if robot_id not in self.robot_sensors:
                self.robot_sensors[robot_id] = []
            self.robot_sensors[robot_id].append(sensor_id)

            # Initialize sensor data
            self.sensor_data[sensor_id] = self.generate_sensor_data(sensor_type, config)

            return jsonify({
                'sensor_id': sensor_id,
                'robot_id': robot_id,
                'type': sensor_type,
                'name': sensor_name,
                'status': 'active',
                'created_at': datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def list_sensors(self):
        """List all sensors across all robots."""
        try:
            sensor_list = []
            for sensor_id, sensor_info in self.sensors.items():
                # Remove sensitive data
                clean_info = sensor_info.copy()
                if 'raw_config' in clean_info:
                    del clean_info['raw_config']
                sensor_list.append(clean_info)

            return jsonify({'sensors': sensor_list})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_all_sensor_data(self, robot_id):
        """Get data from all sensors on a robot."""
        try:
            if robot_id not in self.robot_sensors:
                return jsonify({'error': 'Robot has no sensors'}), 404

            sensor_data = {}
            for sensor_id in self.robot_sensors[robot_id]:
                if sensor_id in self.sensor_data:
                    sensor_info = self.sensors[sensor_id]
                    sensor_type = sensor_info['type']
                    sensor_data[sensor_type] = {
                        'id': sensor_id,
                        'type': sensor_type,
                        'data': self.sensor_data[sensor_id],
                        'timestamp': datetime.now().isoformat(),
                        'topic': sensor_info.get('topic', f'/{robot_id}/{sensor_type}')
                    }

            response = {
                'robot_id': robot_id,
                'timestamp': datetime.now().isoformat(),
                'sensors': sensor_data
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_sensor_data_by_type(self, robot_id, sensor_type):
        """Get data from sensors of a specific type on a robot."""
        try:
            if robot_id not in self.robot_sensors:
                return jsonify({'error': 'Robot has no sensors'}), 404

            # Find sensors of the specified type
            matching_sensors = []
            for sensor_id in self.robot_sensors[robot_id]:
                if self.sensors[sensor_id]['type'] == sensor_type:
                    matching_sensors.append(sensor_id)

            if not matching_sensors:
                return jsonify({'error': f'No {sensor_type} sensors found for robot {robot_id}'}), 404

            # Get data from the first matching sensor (could be extended to return all)
            sensor_id = matching_sensors[0]
            sensor_config = self.sensors[sensor_id].get('config', {})

            # Generate or retrieve sensor data
            data = self.generate_sensor_data(sensor_type, sensor_config)
            self.sensor_data[sensor_id] = data

            response = {
                'robot_id': robot_id,
                'sensor_id': sensor_id,
                'sensor_type': sensor_type,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'topic': self.sensors[sensor_id].get('topic', f'/{robot_id}/{sensor_type}')
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_sensor_data_by_id(self, sensor_id):
        """Get data from a specific sensor by ID."""
        try:
            if sensor_id not in self.sensors:
                return jsonify({'error': 'Sensor not found'}), 404

            sensor_info = self.sensors[sensor_id].copy()
            sensor_type = sensor_info['type']
            sensor_config = sensor_info.get('config', {})

            # Generate or retrieve sensor data
            data = self.generate_sensor_data(sensor_type, sensor_config)
            self.sensor_data[sensor_id] = data

            response = {
                'sensor_id': sensor_id,
                'robot_id': sensor_info['robot_id'],
                'sensor_type': sensor_type,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'topic': sensor_info.get('topic', f'/{sensor_info["robot_id"]}/{sensor_type}')
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_sensor_config(self, sensor_id):
        """Get configuration for a specific sensor."""
        try:
            if sensor_id not in self.sensors:
                return jsonify({'error': 'Sensor not found'}), 404

            sensor_info = self.sensors[sensor_id].copy()
            # Remove sensitive data
            if 'raw_config' in sensor_info:
                del sensor_info['raw_config']

            return jsonify(sensor_info)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def update_sensor_config(self, sensor_id):
        """Update configuration for a specific sensor."""
        try:
            if sensor_id not in self.sensors:
                return jsonify({'error': 'Sensor not found'}), 404

            data = request.get_json()
            updatable_fields = ['config', 'topic', 'status']

            for field in updatable_fields:
                if field in data:
                    self.sensors[sensor_id][field] = data[field]

            self.sensors[sensor_id]['last_updated'] = datetime.now().isoformat()

            return jsonify({
                'sensor_id': sensor_id,
                'updated_fields': list(data.keys()),
                'last_updated': self.sensors[sensor_id]['last_updated']
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_sensor_status(self, sensor_id):
        """Get status for a specific sensor."""
        try:
            if sensor_id not in self.sensors:
                return jsonify({'error': 'Sensor not found'}), 404

            # In a real implementation, this would check if the sensor is publishing data
            # For simulation, return a status based on whether we have recent data
            has_recent_data = sensor_id in self.sensor_data

            status = {
                'sensor_id': sensor_id,
                'status': 'active' if has_recent_data else 'inactive',
                'robot_id': self.sensors[sensor_id]['robot_id'],
                'type': self.sensors[sensor_id]['type'],
                'has_recent_data': has_recent_data,
                'last_data_time': self.sensor_data.get(sensor_id, {}).get('timestamp', None),
                'timestamp': datetime.now().isoformat()
            }

            return jsonify(status)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def generate_sensor_data(self, sensor_type, config):
        """Generate simulated sensor data based on sensor type."""
        timestamp = datetime.now().isoformat()

        if sensor_type == 'camera':
            # Generate camera data
            width = config.get('width', 640)
            height = config.get('height', 480)
            format = config.get('format', 'rgb8')

            return {
                'timestamp': timestamp,
                'width': width,
                'height': height,
                'format': format,
                'data_available': True,
                'encoding': 'base64'  # In real implementation, this would be the actual image data
            }

        elif sensor_type == 'lidar' or sensor_type == 'laser':
            # Generate LIDAR data
            samples = config.get('samples', 360)
            min_range = config.get('min_range', 0.1)
            max_range = config.get('max_range', 30.0)

            # Simulate some readings (in a real scenario, this would come from Gazebo)
            ranges = []
            for i in range(samples):
                # Add some variation to make it realistic
                distance = random.uniform(min_range, max_range)
                ranges.append(round(distance, 3))

            return {
                'timestamp': timestamp,
                'ranges': ranges,
                'min_range': min_range,
                'max_range': max_range,
                'samples': samples,
                'angle_min': config.get('angle_min', -3.14159),
                'angle_max': config.get('angle_max', 3.14159),
                'angle_increment': config.get('angle_increment', 0.01745)  # ~1 degree
            }

        elif sensor_type == 'imu':
            # Generate IMU data
            return {
                'timestamp': timestamp,
                'orientation': {
                    'x': random.uniform(-1, 1),
                    'y': random.uniform(-1, 1),
                    'z': random.uniform(-1, 1),
                    'w': random.uniform(0, 1)
                },
                'angular_velocity': {
                    'x': random.uniform(-0.1, 0.1),
                    'y': random.uniform(-0.1, 0.1),
                    'z': random.uniform(-0.1, 0.1)
                },
                'linear_acceleration': {
                    'x': random.uniform(-0.5, 0.5),
                    'y': random.uniform(-0.5, 0.5),
                    'z': random.uniform(-10, -9)  # gravity
                }
            }

        elif sensor_type == 'gps':
            # Generate GPS data
            return {
                'timestamp': timestamp,
                'latitude': config.get('latitude', 0.0) + random.uniform(-0.001, 0.001),
                'longitude': config.get('longitude', 0.0) + random.uniform(-0.001, 0.001),
                'altitude': config.get('altitude', 0.0) + random.uniform(-1, 1),
                'position_covariance': [0.0] * 9
            }

        elif sensor_type == 'force_torque':
            # Generate force/torque data
            return {
                'timestamp': timestamp,
                'force': {
                    'x': random.uniform(-10, 10),
                    'y': random.uniform(-10, 10),
                    'z': random.uniform(-10, 10)
                },
                'torque': {
                    'x': random.uniform(-1, 1),
                    'y': random.uniform(-1, 1),
                    'z': random.uniform(-1, 1)
                }
            }

        else:
            # Generic sensor data
            return {
                'timestamp': timestamp,
                'type': sensor_type,
                'data': 'simulated_data',
                'status': 'ok'
            }

    def run(self, host='0.0.0.0', port=5002, debug=False):
        """Run the Flask application."""
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function to run the Sensor API."""
    api = SensorAPI()
    print("Starting Sensor Data API...")
    print("Available endpoints:")
    print("  GET    /api/robots/<id>/sensors           - List robot sensors")
    print("  POST   /api/robots/<id>/sensors          - Add sensor to robot")
    print("  GET    /api/robot/<id>/sensors           - Get all sensor data")
    print("  GET    /api/robot/<id>/sensors/<type>    - Get sensor data by type")
    print("  GET    /api/sensor/<id>/data             - Get specific sensor data")

    api.run(debug=True)


if __name__ == '__main__':
    main()