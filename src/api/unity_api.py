#!/usr/bin/env python3

"""
Unity Visualization API for Gazebo-Unity Integration

This module provides REST API endpoints for Unity visualization
and human-robot interaction concepts.
"""

from flask import Flask, request, jsonify
import json
import os
from datetime import datetime


class UnityAPI:
    """
    API class for Unity visualization and human-robot interaction.
    """

    def __init__(self):
        self.app = Flask(__name__)
        self.scenes = {}  # Store scene information
        self.visualizations = {}  # Store visualization states
        self.interactions = {}  # Store interaction elements

        # Register API routes
        self.register_routes()

    def register_routes(self):
        """Register all API routes."""
        # Scene management endpoints
        self.app.add_url_rule('/api/unity/scenes', 'list_scenes', self.list_scenes, methods=['GET'])
        self.app.add_url_rule('/api/unity/scenes', 'create_scene', self.create_scene, methods=['POST'])
        self.app.add_url_rule('/api/unity/scenes/<scene_id>', 'get_scene', self.get_scene, methods=['GET'])
        self.app.add_url_rule('/api/unity/scenes/<scene_id>', 'update_scene', self.update_scene, methods=['PUT'])
        self.app.add_url_rule('/api/unity/scenes/<scene_id>', 'delete_scene', self.delete_scene, methods=['DELETE'])

        # Visualization management endpoints
        self.app.add_url_rule('/api/unity/visualization', 'list_visualizations', self.list_visualizations, methods=['GET'])
        self.app.add_url_rule('/api/unity/visualization', 'create_visualization', self.create_visualization, methods=['POST'])
        self.app.add_url_rule('/api/unity/visualization/<vis_id>', 'get_visualization', self.get_visualization, methods=['GET'])
        self.app.add_url_rule('/api/unity/visualization/<vis_id>', 'update_visualization', self.update_visualization, methods=['PUT'])
        self.app.add_url_rule('/api/unity/visualization/<vis_id>', 'delete_visualization', self.delete_visualization, methods=['DELETE'])

        # Robot state synchronization endpoints
        self.app.add_url_rule('/api/unity/robot/<robot_id>/state', 'get_robot_state', self.get_robot_state, methods=['GET'])
        self.app.add_url_rule('/api/unity/robot/<robot_id>/state', 'update_robot_state', self.update_robot_state, methods=['POST'])
        self.app.add_url_rule('/api/unity/robot/<robot_id>/sync', 'sync_robot_state', self.sync_robot_state, methods=['POST'])

        # Interaction element endpoints
        self.app.add_url_rule('/api/unity/interactions', 'list_interactions', self.list_interactions, methods=['GET'])
        self.app.add_url_rule('/api/unity/interactions', 'create_interaction', self.create_interaction, methods=['POST'])
        self.app.add_url_rule('/api/unity/interactions/<interaction_id>', 'get_interaction', self.get_interaction, methods=['GET'])
        self.app.add_url_rule('/api/unity/interactions/<interaction_id>', 'update_interaction', self.update_interaction, methods=['PUT'])
        self.app.add_url_rule('/api/unity/interactions/<interaction_id>', 'delete_interaction', self.delete_interaction, methods=['DELETE'])

        # UI element endpoints
        self.app.add_url_rule('/api/unity/ui/elements', 'list_ui_elements', self.list_ui_elements, methods=['GET'])
        self.app.add_url_rule('/api/unity/ui/elements', 'create_ui_element', self.create_ui_element, methods=['POST'])
        self.app.add_url_rule('/api/unity/ui/elements/<element_id>', 'get_ui_element', self.get_ui_element, methods=['GET'])

        # Connection endpoints
        self.app.add_url_rule('/api/unity/connection', 'get_connection_status', self.get_connection_status, methods=['GET'])
        self.app.add_url_rule('/api/unity/connection', 'establish_connection', self.establish_connection, methods=['POST'])

    def list_scenes(self):
        """List all available Unity scenes."""
        try:
            scene_list = []
            for scene_id, scene_info in self.scenes.items():
                scene_list.append({
                    'id': scene_id,
                    'name': scene_info.get('name'),
                    'description': scene_info.get('description'),
                    'status': scene_info.get('status'),
                    'created_at': scene_info.get('created_at')
                })

            return jsonify({'scenes': scene_list})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def create_scene(self):
        """Create a new Unity scene."""
        try:
            data = request.get_json()
            scene_id = data.get('id', f"scene_{int(datetime.now().timestamp())}")
            name = data.get('name', f"Scene_{scene_id}")
            description = data.get('description', '')
            config = data.get('config', {})

            # Validate inputs
            if not name:
                return jsonify({'error': 'name is required'}), 400

            # Create scene entry
            self.scenes[scene_id] = {
                'id': scene_id,
                'name': name,
                'description': description,
                'config': config,
                'status': 'created',
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }

            # Initialize visualization for this scene
            self.visualizations[scene_id] = {
                'scene_id': scene_id,
                'robot_states': {},
                'camera_config': config.get('camera_config', {}),
                'lighting_config': config.get('lighting_config', {}),
                'environment_config': config.get('environment_config', {}),
                'last_updated': datetime.now().isoformat()
            }

            response = {
                'scene_id': scene_id,
                'name': name,
                'status': 'created',
                'created_at': self.scenes[scene_id]['created_at']
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_scene(self, scene_id):
        """Get details for a specific Unity scene."""
        try:
            if scene_id not in self.scenes:
                return jsonify({'error': 'Scene not found'}), 404

            scene_info = self.scenes[scene_id].copy()
            # Add visualization information
            if scene_id in self.visualizations:
                scene_info['visualization'] = self.visualizations[scene_id]

            return jsonify(scene_info)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def update_scene(self, scene_id):
        """Update a Unity scene configuration."""
        try:
            if scene_id not in self.scenes:
                return jsonify({'error': 'Scene not found'}), 404

            data = request.get_json()
            updatable_fields = ['name', 'description', 'config', 'status']

            for field in updatable_fields:
                if field in data:
                    self.scenes[scene_id][field] = data[field]

            self.scenes[scene_id]['last_updated'] = datetime.now().isoformat()

            return jsonify({
                'scene_id': scene_id,
                'updated_fields': list(data.keys()),
                'last_updated': self.scenes[scene_id]['last_updated']
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def delete_scene(self, scene_id):
        """Delete a Unity scene."""
        try:
            if scene_id not in self.scenes:
                return jsonify({'error': 'Scene not found'}), 404

            # Remove scene and its visualization
            del self.scenes[scene_id]
            if scene_id in self.visualizations:
                del self.visualizations[scene_id]

            return jsonify({
                'scene_id': scene_id,
                'status': 'deleted'
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def list_visualizations(self):
        """List all visualization states."""
        try:
            visualization_list = []
            for vis_id, vis_info in self.visualizations.items():
                visualization_list.append({
                    'id': vis_id,
                    'scene_id': vis_info.get('scene_id'),
                    'robot_count': len(vis_info.get('robot_states', {})),
                    'last_updated': vis_info.get('last_updated')
                })

            return jsonify({'visualizations': visualization_list})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def create_visualization(self):
        """Create a new visualization state."""
        try:
            data = request.get_json()
            vis_id = data.get('id', f"vis_{int(datetime.now().timestamp())}")
            scene_id = data.get('scene_id')
            robot_states = data.get('robot_states', {})
            camera_config = data.get('camera_config', {})
            lighting_config = data.get('lighting_config', {})

            if not scene_id:
                return jsonify({'error': 'scene_id is required'}), 400

            # Create visualization entry
            self.visualizations[vis_id] = {
                'id': vis_id,
                'scene_id': scene_id,
                'robot_states': robot_states,
                'camera_config': camera_config,
                'lighting_config': lighting_config,
                'environment_config': data.get('environment_config', {}),
                'last_updated': datetime.now().isoformat()
            }

            response = {
                'visualization_id': vis_id,
                'scene_id': scene_id,
                'status': 'created',
                'created_at': datetime.now().isoformat()
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_visualization(self, vis_id):
        """Get a specific visualization state."""
        try:
            if vis_id not in self.visualizations:
                return jsonify({'error': 'Visualization not found'}), 404

            visualization_info = self.visualizations[vis_id].copy()
            return jsonify(visualization_info)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def update_visualization(self, vis_id):
        """Update a visualization state."""
        try:
            if vis_id not in self.visualizations:
                return jsonify({'error': 'Visualization not found'}), 404

            data = request.get_json()
            updatable_fields = ['robot_states', 'camera_config', 'lighting_config', 'environment_config']

            for field in updatable_fields:
                if field in data:
                    self.visualizations[vis_id][field] = data[field]

            self.visualizations[vis_id]['last_updated'] = datetime.now().isoformat()

            return jsonify({
                'visualization_id': vis_id,
                'updated_fields': list(data.keys()),
                'last_updated': self.visualizations[vis_id]['last_updated']
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def delete_visualization(self, vis_id):
        """Delete a visualization state."""
        try:
            if vis_id not in self.visualizations:
                return jsonify({'error': 'Visualization not found'}), 404

            del self.visualizations[vis_id]

            return jsonify({
                'visualization_id': vis_id,
                'status': 'deleted'
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_robot_state(self, robot_id):
        """Get the current robot state for Unity visualization."""
        try:
            # Look for robot state in all visualizations
            for vis_id, vis_info in self.visualizations.items():
                if robot_id in vis_info.get('robot_states', {}):
                    robot_state = vis_info['robot_states'][robot_id]
                    response = {
                        'robot_id': robot_id,
                        'state': robot_state,
                        'timestamp': datetime.now().isoformat(),
                        'visualization_id': vis_id
                    }
                    return jsonify(response)

            # If not found in any visualization, return default state
            default_state = {
                'position': {'x': 0, 'y': 0, 'z': 0},
                'rotation': {'x': 0, 'y': 0, 'z': 0, 'w': 1},
                'joint_positions': {},
                'status': 'default'
            }

            response = {
                'robot_id': robot_id,
                'state': default_state,
                'timestamp': datetime.now().isoformat(),
                'visualization_id': None
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def update_robot_state(self, robot_id):
        """Update the robot state for Unity visualization."""
        try:
            data = request.get_json()
            scene_id = data.get('scene_id')
            new_state = data.get('state', {})

            if not scene_id:
                return jsonify({'error': 'scene_id is required'}), 400

            # Find the visualization for this scene
            vis_id = scene_id
            if vis_id not in self.visualizations:
                # Create a new visualization if it doesn't exist
                self.visualizations[vis_id] = {
                    'scene_id': scene_id,
                    'robot_states': {},
                    'camera_config': {},
                    'lighting_config': {},
                    'environment_config': {},
                    'last_updated': datetime.now().isoformat()
                }

            # Update robot state
            self.visualizations[vis_id]['robot_states'][robot_id] = new_state
            self.visualizations[vis_id]['last_updated'] = datetime.now().isoformat()

            response = {
                'robot_id': robot_id,
                'scene_id': scene_id,
                'updated_state': new_state,
                'timestamp': datetime.now().isoformat()
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def sync_robot_state(self, robot_id):
        """Synchronize robot state between Gazebo and Unity."""
        try:
            data = request.get_json()
            gazebo_state = data.get('gazebo_state', {})
            unity_scene_id = data.get('unity_scene_id')

            if not unity_scene_id:
                return jsonify({'error': 'unity_scene_id is required'}), 400

            # Convert Gazebo coordinates to Unity coordinates if needed
            # Gazebo: X forward, Y left, Z up
            # Unity: X right, Y up, Z forward
            unity_state = self.convert_coordinates_gazebo_to_unity(gazebo_state)

            # Update visualization
            vis_id = unity_scene_id
            if vis_id not in self.visualizations:
                self.visualizations[vis_id] = {
                    'scene_id': unity_scene_id,
                    'robot_states': {},
                    'camera_config': {},
                    'lighting_config': {},
                    'environment_config': {},
                    'last_updated': datetime.now().isoformat()
                }

            self.visualizations[vis_id]['robot_states'][robot_id] = unity_state
            self.visualizations[vis_id]['last_updated'] = datetime.now().isoformat()

            response = {
                'robot_id': robot_id,
                'unity_scene_id': unity_scene_id,
                'synced_state': unity_state,
                'timestamp': datetime.now().isoformat(),
                'conversion_applied': True
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def convert_coordinates_gazebo_to_unity(self, gazebo_state):
        """Convert coordinates from Gazebo to Unity coordinate system."""
        unity_state = gazebo_state.copy()

        # Convert position: Gazebo (X forward, Y left, Z up) -> Unity (X right, Y up, Z forward)
        if 'position' in unity_state:
            pos = unity_state['position']
            unity_state['position'] = {
                'x': pos.get('y', 0),    # Gazebo Y -> Unity X
                'y': pos.get('z', 0),    # Gazebo Z -> Unity Y
                'z': pos.get('x', 0)     # Gazebo X -> Unity Z
            }

        # Convert rotation quaternion if present
        if 'rotation' in unity_state:
            rot = unity_state['rotation']
            # This is a simplified conversion; full quaternion conversion would be more complex
            unity_state['rotation'] = {
                'x': rot.get('x', 0),
                'y': rot.get('y', 0),
                'z': rot.get('z', 0),
                'w': rot.get('w', 1)
            }

        # Convert joint positions if present (no conversion needed for joint angles)
        if 'joint_positions' in unity_state:
            # Joint positions are typically in radians and don't need coordinate conversion
            pass

        return unity_state

    def list_interactions(self):
        """List all interaction elements."""
        try:
            interaction_list = []
            for interaction_id, interaction_info in self.interactions.items():
                interaction_list.append({
                    'id': interaction_id,
                    'name': interaction_info.get('name'),
                    'type': interaction_info.get('type'),
                    'scene_id': interaction_info.get('scene_id'),
                    'status': interaction_info.get('status')
                })

            return jsonify({'interactions': interaction_list})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def create_interaction(self):
        """Create a new interaction element."""
        try:
            data = request.get_json()
            interaction_id = data.get('id', f"interaction_{int(datetime.now().timestamp())}")
            name = data.get('name')
            interaction_type = data.get('type')
            scene_id = data.get('scene_id')
            config = data.get('config', {})

            if not name or not interaction_type or not scene_id:
                return jsonify({'error': 'name, type, and scene_id are required'}), 400

            # Create interaction entry
            self.interactions[interaction_id] = {
                'id': interaction_id,
                'name': name,
                'type': interaction_type,
                'scene_id': scene_id,
                'config': config,
                'status': 'active',
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }

            response = {
                'interaction_id': interaction_id,
                'name': name,
                'type': interaction_type,
                'scene_id': scene_id,
                'status': 'active',
                'created_at': datetime.now().isoformat()
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_interaction(self, interaction_id):
        """Get details for a specific interaction element."""
        try:
            if interaction_id not in self.interactions:
                return jsonify({'error': 'Interaction not found'}), 404

            interaction_info = self.interactions[interaction_id].copy()
            return jsonify(interaction_info)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def update_interaction(self, interaction_id):
        """Update an interaction element."""
        try:
            if interaction_id not in self.interactions:
                return jsonify({'error': 'Interaction not found'}), 404

            data = request.get_json()
            updatable_fields = ['name', 'config', 'status']

            for field in updatable_fields:
                if field in data:
                    self.interactions[interaction_id][field] = data[field]

            self.interactions[interaction_id]['last_updated'] = datetime.now().isoformat()

            return jsonify({
                'interaction_id': interaction_id,
                'updated_fields': list(data.keys()),
                'last_updated': self.interactions[interaction_id]['last_updated']
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def delete_interaction(self, interaction_id):
        """Delete an interaction element."""
        try:
            if interaction_id not in self.interactions:
                return jsonify({'error': 'Interaction not found'}), 404

            del self.interactions[interaction_id]

            return jsonify({
                'interaction_id': interaction_id,
                'status': 'deleted'
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def list_ui_elements(self):
        """List all UI elements."""
        try:
            # In a real implementation, this would return UI elements
            ui_elements = [
                {
                    'id': 'robot_control_panel',
                    'type': 'panel',
                    'position': {'x': 10, 'y': 10},
                    'size': {'width': 300, 'height': 200},
                    'visible': True
                },
                {
                    'id': 'status_display',
                    'type': 'text',
                    'position': {'x': 320, 'y': 20},
                    'content': 'Robot Status: Idle',
                    'visible': True
                }
            ]

            return jsonify({'ui_elements': ui_elements})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def create_ui_element(self):
        """Create a new UI element."""
        try:
            data = request.get_json()
            element_id = data.get('id', f"ui_element_{int(datetime.now().timestamp())}")
            element_type = data.get('type', 'panel')
            position = data.get('position', {'x': 0, 'y': 0})
            size = data.get('size', {'width': 100, 'height': 50})
            content = data.get('content', '')

            # Create UI element entry
            ui_element = {
                'id': element_id,
                'type': element_type,
                'position': position,
                'size': size,
                'content': content,
                'visible': data.get('visible', True),
                'created_at': datetime.now().isoformat()
            }

            # In a real implementation, this would store the UI element
            # For now, we'll just return the created element

            return jsonify(ui_element)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_ui_element(self, element_id):
        """Get a specific UI element."""
        try:
            # In a real implementation, this would return the stored UI element
            # For now, return a sample element
            ui_element = {
                'id': element_id,
                'type': 'panel',
                'position': {'x': 50, 'y': 50},
                'size': {'width': 200, 'height': 100},
                'content': f'UI Element: {element_id}',
                'visible': True,
                'last_updated': datetime.now().isoformat()
            }

            return jsonify(ui_element)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_connection_status(self):
        """Get Unity connection status."""
        try:
            # In a real implementation, this would check the actual connection
            # For simulation, return connected status
            status = {
                'connected': True,
                'connection_type': 'websocket',
                'unity_version': '2022.3 LTS',
                'ros_bridge_connected': True,
                'last_heartbeat': datetime.now().isoformat(),
                'message_queue_size': 0
            }

            return jsonify(status)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def establish_connection(self):
        """Establish connection to Unity."""
        try:
            data = request.get_json()
            unity_address = data.get('unity_address', 'ws://localhost:8080')
            ros_bridge_address = data.get('ros_bridge_address', 'ws://localhost:9090')

            # In a real implementation, this would establish the actual connection
            # For simulation, return success
            connection_info = {
                'status': 'connected',
                'unity_address': unity_address,
                'ros_bridge_address': ros_bridge_address,
                'connection_timestamp': datetime.now().isoformat(),
                'supported_protocols': ['json', 'bson'],
                'compression_enabled': True
            }

            return jsonify(connection_info)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def run(self, host='0.0.0.0', port=5003, debug=False):
        """Run the Flask application."""
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function to run the Unity API."""
    api = UnityAPI()
    print("Starting Unity Visualization API...")
    print("Available endpoints:")
    print("  GET    /api/unity/scenes                     - List Unity scenes")
    print("  POST   /api/unity/scenes                     - Create Unity scene")
    print("  GET    /api/unity/visualization              - List visualizations")
    print("  POST   /api/unity/robot/<id>/state          - Update robot state")
    print("  POST   /api/unity/robot/<id>/sync           - Sync robot state")
    print("  GET    /api/unity/connection                 - Get connection status")

    api.run(debug=True)


if __name__ == '__main__':
    main()