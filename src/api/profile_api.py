"""
Simulation Profile Management API

REST API for managing simulation profiles with CRUD operations,
validation, and integration with the simulation profile manager.
"""

from flask import Flask, request, jsonify
from typing import Dict, Any, Optional
import os
import sys
import logging

# Add src directory to path to import simulation modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from simulation.profile_manager import SimulationProfileManager, SimulationProfile, ProfileType


def create_profile_api():
    """Create and configure the profile management API"""
    app = Flask(__name__)

    # Initialize the simulation profile manager
    profile_manager = SimulationProfileManager()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    @app.route('/api/profiles', methods=['GET'])
    def get_profiles():
        """Get list of all available simulation profiles"""
        try:
            profile_names = profile_manager.list_profiles()
            profiles_data = []

            for name in profile_names:
                profile = profile_manager.get_profile(name)
                if profile:
                    profile_data = {
                        'name': profile.name,
                        'type': profile.profile_type.value,
                        'description': profile.description,
                        'use_case': profile.use_case,
                        'physics_settings': {
                            'engine': profile.physics_settings.engine,
                            'time_step': profile.physics_settings.time_step,
                            'real_time_factor': profile.physics_settings.real_time_factor
                        },
                        'visual_quality': {
                            'rendering_quality': profile.visual_quality.rendering_quality,
                            'shadows': profile.visual_quality.shadows
                        }
                    }
                    profiles_data.append(profile_data)

            return jsonify({
                'success': True,
                'profiles': profiles_data,
                'count': len(profiles_data)
            }), 200

        except Exception as e:
            logger.error(f"Error getting profiles: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve profiles',
                'message': str(e)
            }), 500

    @app.route('/api/profiles/<profile_name>', methods=['GET'])
    def get_profile(profile_name):
        """Get a specific simulation profile by name"""
        try:
            profile = profile_manager.get_profile(profile_name)
            if not profile:
                return jsonify({
                    'success': False,
                    'error': 'Profile not found',
                    'profile_name': profile_name
                }), 404

            profile_data = {
                'name': profile.name,
                'type': profile.profile_type.value,
                'description': profile.description,
                'use_case': profile.use_case,
                'physics_settings': {
                    'engine': profile.physics_settings.engine,
                    'gravity_x': profile.physics_settings.gravity_x,
                    'gravity_y': profile.physics_settings.gravity_y,
                    'gravity_z': profile.physics_settings.gravity_z,
                    'time_step': profile.physics_settings.time_step,
                    'real_time_factor': profile.physics_settings.real_time_factor
                },
                'visual_quality': {
                    'rendering_quality': profile.visual_quality.rendering_quality,
                    'shadows': profile.visual_quality.shadows,
                    'reflections': profile.visual_quality.reflections,
                    'anti_aliasing': profile.visual_quality.anti_aliasing
                },
                'hardware_requirements': {
                    'min_cpu_cores': profile.hardware_requirements.min_cpu_cores,
                    'min_ram_gb': profile.hardware_requirements.min_ram_gb,
                    'min_gpu_vram_gb': profile.hardware_requirements.min_gpu_vram_gb,
                    'recommended_cpu_cores': profile.hardware_requirements.recommended_cpu_cores,
                    'recommended_ram_gb': profile.hardware_requirements.recommended_ram_gb,
                    'recommended_gpu_vram_gb': profile.hardware_requirements.recommended_gpu_vram_gb
                }
            }

            return jsonify({
                'success': True,
                'profile': profile_data
            }), 200

        except Exception as e:
            logger.error(f"Error getting profile {profile_name}: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve profile',
                'message': str(e)
            }), 500

    @app.route('/api/profiles', methods=['POST'])
    def create_profile():
        """Create a new simulation profile"""
        try:
            data = request.get_json()

            # Validate required fields
            required_fields = ['name', 'type', 'physics_settings', 'visual_quality', 'hardware_requirements']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'success': False,
                        'error': f'Missing required field: {field}'
                    }), 400

            # Validate profile type
            try:
                profile_type = ProfileType(data['type'])
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid profile type. Valid types: {[pt.value for pt in ProfileType]}'
                }), 400

            # Create physics settings
            physics_data = data['physics_settings']
            from simulation.profile_manager import PhysicsSettings
            physics_settings = PhysicsSettings(
                engine=physics_data.get('engine', 'ode'),
                gravity_x=physics_data.get('gravity_x', 0.0),
                gravity_y=physics_data.get('gravity_y', 0.0),
                gravity_z=physics_data.get('gravity_z', -9.81),
                time_step=physics_data.get('time_step', 0.001),
                real_time_factor=physics_data.get('real_time_factor', 1.0)
            )

            # Create visual quality settings
            visual_data = data['visual_quality']
            from simulation.profile_manager import VisualQuality
            visual_quality = VisualQuality(
                rendering_quality=visual_data.get('rendering_quality', 'medium'),
                shadows=visual_data.get('shadows', True),
                reflections=visual_data.get('reflections', False),
                anti_aliasing=visual_data.get('anti_aliasing', 'fxaa')
            )

            # Create hardware requirements
            hardware_data = data['hardware_requirements']
            from simulation.profile_manager import HardwareRequirements
            hardware_requirements = HardwareRequirements(
                min_cpu_cores=hardware_data.get('min_cpu_cores', 4),
                min_ram_gb=hardware_data.get('min_ram_gb', 8),
                min_gpu_vram_gb=hardware_data.get('min_gpu_vram_gb', 2),
                recommended_cpu_cores=hardware_data.get('recommended_cpu_cores', 8),
                recommended_ram_gb=hardware_data.get('recommended_ram_gb', 16),
                recommended_gpu_vram_gb=hardware_data.get('recommended_gpu_vram_gb', 4)
            )

            # Create the profile
            profile = SimulationProfile(
                name=data['name'],
                profile_type=profile_type,
                physics_settings=physics_settings,
                visual_quality=visual_quality,
                hardware_requirements=hardware_requirements,
                description=data.get('description', ''),
                use_case=data.get('use_case', '')
            )

            # Add to manager
            success = profile_manager.add_profile(profile)
            if not success:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create profile'
                }), 500

            logger.info(f"Created profile: {profile.name}")

            return jsonify({
                'success': True,
                'profile': {
                    'name': profile.name,
                    'type': profile.profile_type.value,
                    'description': profile.description
                }
            }), 201

        except Exception as e:
            logger.error(f"Error creating profile: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Failed to create profile',
                'message': str(e)
            }), 500

    @app.route('/api/profiles/<profile_name>', methods=['PUT'])
    def update_profile(profile_name):
        """Update an existing simulation profile"""
        try:
            # Check if profile exists
            existing_profile = profile_manager.get_profile(profile_name)
            if not existing_profile:
                return jsonify({
                    'success': False,
                    'error': 'Profile not found',
                    'profile_name': profile_name
                }), 404

            data = request.get_json()

            # For updates, we can modify only the fields that are provided
            profile_data = {
                'name': data.get('name', existing_profile.name),
                'type': data.get('type', existing_profile.profile_type.value),
                'description': data.get('description', existing_profile.description),
                'use_case': data.get('use_case', existing_profile.use_case),
                'physics_settings': data.get('physics_settings', {
                    'engine': existing_profile.physics_settings.engine,
                    'gravity_x': existing_profile.physics_settings.gravity_x,
                    'gravity_y': existing_profile.physics_settings.gravity_y,
                    'gravity_z': existing_profile.physics_settings.gravity_z,
                    'time_step': existing_profile.physics_settings.time_step,
                    'real_time_factor': existing_profile.physics_settings.real_time_factor
                }),
                'visual_quality': data.get('visual_quality', {
                    'rendering_quality': existing_profile.visual_quality.rendering_quality,
                    'shadows': existing_profile.visual_quality.shadows,
                    'reflections': existing_profile.visual_quality.reflections,
                    'anti_aliasing': existing_profile.visual_quality.anti_aliasing
                }),
                'hardware_requirements': data.get('hardware_requirements', {
                    'min_cpu_cores': existing_profile.hardware_requirements.min_cpu_cores,
                    'min_ram_gb': existing_profile.hardware_requirements.min_ram_gb,
                    'min_gpu_vram_gb': existing_profile.hardware_requirements.min_gpu_vram_gb,
                    'recommended_cpu_cores': existing_profile.hardware_requirements.recommended_cpu_cores,
                    'recommended_ram_gb': existing_profile.hardware_requirements.recommended_ram_gb,
                    'recommended_gpu_vram_gb': existing_profile.hardware_requirements.recommended_gpu_vram_gb
                })
            }

            # Validate profile type if provided
            try:
                profile_type = ProfileType(profile_data['type'])
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid profile type. Valid types: {[pt.value for pt in ProfileType]}'
                }), 400

            # Create updated profile with new settings
            from simulation.profile_manager import PhysicsSettings, VisualQuality, HardwareRequirements
            physics_settings = PhysicsSettings(
                engine=profile_data['physics_settings'].get('engine', 'ode'),
                gravity_x=profile_data['physics_settings'].get('gravity_x', 0.0),
                gravity_y=profile_data['physics_settings'].get('gravity_y', 0.0),
                gravity_z=profile_data['physics_settings'].get('gravity_z', -9.81),
                time_step=profile_data['physics_settings'].get('time_step', 0.001),
                real_time_factor=profile_data['physics_settings'].get('real_time_factor', 1.0)
            )

            visual_quality = VisualQuality(
                rendering_quality=profile_data['visual_quality'].get('rendering_quality', 'medium'),
                shadows=profile_data['visual_quality'].get('shadows', True),
                reflections=profile_data['visual_quality'].get('reflections', False),
                anti_aliasing=profile_data['visual_quality'].get('anti_aliasing', 'fxaa')
            )

            hardware_requirements = HardwareRequirements(
                min_cpu_cores=profile_data['hardware_requirements'].get('min_cpu_cores', 4),
                min_ram_gb=profile_data['hardware_requirements'].get('min_ram_gb', 8),
                min_gpu_vram_gb=profile_data['hardware_requirements'].get('min_gpu_vram_gb', 2),
                recommended_cpu_cores=profile_data['hardware_requirements'].get('recommended_cpu_cores', 8),
                recommended_ram_gb=profile_data['hardware_requirements'].get('recommended_ram_gb', 16),
                recommended_gpu_vram_gb=profile_data['hardware_requirements'].get('recommended_gpu_vram_gb', 4)
            )

            updated_profile = SimulationProfile(
                name=profile_data['name'],
                profile_type=profile_type,
                physics_settings=physics_settings,
                visual_quality=visual_quality,
                hardware_requirements=hardware_requirements,
                description=profile_data['description'],
                use_case=profile_data['use_case']
            )

            # Add the updated profile (this will overwrite the existing one)
            success = profile_manager.add_profile(updated_profile)
            if not success:
                return jsonify({
                    'success': False,
                    'error': 'Failed to update profile'
                }), 500

            logger.info(f"Updated profile: {updated_profile.name}")

            return jsonify({
                'success': True,
                'profile': {
                    'name': updated_profile.name,
                    'type': updated_profile.profile_type.value,
                    'description': updated_profile.description
                }
            }), 200

        except Exception as e:
            logger.error(f"Error updating profile {profile_name}: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Failed to update profile',
                'message': str(e)
            }), 500

    @app.route('/api/profiles/<profile_name>', methods=['DELETE'])
    def delete_profile(profile_name):
        """Delete a simulation profile (placeholder - in real implementation would need to check if profile is in use)"""
        try:
            # In a real implementation, we would check if the profile is currently in use
            # and prevent deletion if it is. For this example, we'll just return a message
            # indicating this limitation.

            profile = profile_manager.get_profile(profile_name)
            if not profile:
                return jsonify({
                    'success': False,
                    'error': 'Profile not found',
                    'profile_name': profile_name
                }), 404

            # In a real implementation, you might have logic here to check if the profile
            # is currently being used by any active simulations before deletion

            return jsonify({
                'success': False,
                'error': 'Delete operation not implemented in this example',
                'message': 'Profile deletion requires checking for active usage in real implementation',
                'profile_name': profile_name
            }), 405  # Method not allowed for now

        except Exception as e:
            logger.error(f"Error deleting profile {profile_name}: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Failed to delete profile',
                'message': str(e)
            }), 500

    @app.route('/api/profiles/<profile_name>/apply', methods=['POST'])
    def apply_profile(profile_name):
        """Apply a simulation profile to the current simulation environment"""
        try:
            success = profile_manager.apply_profile(profile_name)
            if not success:
                return jsonify({
                    'success': False,
                    'error': 'Failed to apply profile',
                    'profile_name': profile_name
                }), 404

            logger.info(f"Applied profile: {profile_name}")

            return jsonify({
                'success': True,
                'message': f'Profile {profile_name} applied successfully',
                'profile_name': profile_name
            }), 200

        except Exception as e:
            logger.error(f"Error applying profile {profile_name}: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Failed to apply profile',
                'message': str(e)
            }), 500

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Endpoint not found'
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

    return app


if __name__ == '__main__':
    # For testing purposes
    app = create_profile_api()
    app.run(debug=True, host='0.0.0.0', port=5001)