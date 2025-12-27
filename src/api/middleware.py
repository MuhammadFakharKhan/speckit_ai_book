"""
API Middleware for Validation and Error Handling

Provides validation, error handling, and other middleware functionality
for the simulation API endpoints.
"""

from functools import wraps
from typing import Dict, Any, Callable, Optional
import json
import logging
import time
from flask import request, jsonify, g
import re


class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class ValidationMiddleware:
    """Middleware for request validation"""

    @staticmethod
    def validate_json_schema(schema: Dict[str, Any]):
        """
        Decorator to validate request JSON against a schema
        """
        def decorator(f: Callable):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not request.is_json:
                    return jsonify({
                        'success': False,
                        'error': 'Content-Type must be application/json'
                    }), 400

                json_data = request.get_json()
                if json_data is None:
                    return jsonify({
                        'success': False,
                        'error': 'Request body must be valid JSON'
                    }), 400

                try:
                    ValidationMiddleware._validate_json_data(json_data, schema)
                except ValidationError as e:
                    return jsonify({
                        'success': False,
                        'error': e.message,
                        'field': e.field
                    }), 400

                return f(*args, **kwargs)
            return decorated_function
        return decorator

    @staticmethod
    def _validate_json_data(data: Dict[str, Any], schema: Dict[str, Any]):
        """Validate JSON data against schema"""
        required_fields = schema.get('required', [])
        properties = schema.get('properties', {})

        # Check required fields
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}", field)

        # Validate each property
        for field, value in data.items():
            if field in properties:
                field_schema = properties[field]
                ValidationMiddleware._validate_field(field, value, field_schema)

    @staticmethod
    def _validate_field(field_name: str, value: Any, schema: Dict[str, Any]):
        """Validate a single field against its schema"""
        field_type = schema.get('type')
        field_format = schema.get('format')
        field_pattern = schema.get('pattern')

        # Type validation
        if field_type == 'string':
            if not isinstance(value, str):
                raise ValidationError(f"Field '{field_name}' must be a string", field_name)
        elif field_type == 'number':
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Field '{field_name}' must be a number", field_name)
        elif field_type == 'integer':
            if not isinstance(value, int):
                raise ValidationError(f"Field '{field_name}' must be an integer", field_name)
        elif field_type == 'boolean':
            if not isinstance(value, bool):
                raise ValidationError(f"Field '{field_name}' must be a boolean", field_name)
        elif field_type == 'object':
            if not isinstance(value, dict):
                raise ValidationError(f"Field '{field_name}' must be an object", field_name)
        elif field_type == 'array':
            if not isinstance(value, list):
                raise ValidationError(f"Field '{field_name}' must be an array", field_name)

        # Format validation
        if field_format == 'email' and field_type == 'string':
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, value):
                raise ValidationError(f"Field '{field_name}' must be a valid email", field_name)

        # Pattern validation
        if field_pattern and field_type == 'string':
            if not re.match(field_pattern, value):
                raise ValidationError(f"Field '{field_name}' does not match required pattern", field_name)

        # Additional validations
        if 'minimum' in schema and field_type in ['number', 'integer']:
            minimum = schema['minimum']
            if value < minimum:
                raise ValidationError(f"Field '{field_name}' must be at least {minimum}", field_name)

        if 'maximum' in schema and field_type in ['number', 'integer']:
            maximum = schema['maximum']
            if value > maximum:
                raise ValidationError(f"Field '{field_name}' must be at most {maximum}", field_name)

        if 'minLength' in schema and field_type == 'string':
            min_length = schema['minLength']
            if len(value) < min_length:
                raise ValidationError(f"Field '{field_name}' must be at least {min_length} characters", field_name)

        if 'maxLength' in schema and field_type == 'string':
            max_length = schema['maxLength']
            if len(value) > max_length:
                raise ValidationError(f"Field '{field_name}' must be at most {max_length} characters", field_name)

        if 'enum' in schema:
            enum_values = schema['enum']
            if value not in enum_values:
                raise ValidationError(f"Field '{field_name}' must be one of {enum_values}", field_name)

    @staticmethod
    def validate_path_parameter(param_name: str, validator_func: Callable[[str], bool], error_message: str = None):
        """
        Decorator to validate path parameters
        """
        def decorator(f: Callable):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                param_value = kwargs.get(param_name)
                if param_value is None:
                    return jsonify({
                        'success': False,
                        'error': f'Path parameter {param_name} is required'
                    }), 400

                if not validator_func(param_value):
                    error_msg = error_message or f'Invalid value for parameter {param_name}'
                    return jsonify({
                        'success': False,
                        'error': error_msg
                    }), 400

                return f(*args, **kwargs)
            return decorated_function
        return decorator


class ErrorHandlingMiddleware:
    """Middleware for centralized error handling"""

    @staticmethod
    def handle_errors(app):
        """Register error handlers with the Flask app"""

        @app.errorhandler(ValidationError)
        def handle_validation_error(error):
            return jsonify({
                'success': False,
                'error': error.message,
                'field': error.field
            }), 400

        @app.errorhandler(ValueError)
        def handle_value_error(error):
            return jsonify({
                'success': False,
                'error': 'Invalid value provided',
                'message': str(error)
            }), 400

        @app.errorhandler(KeyError)
        def handle_key_error(error):
            return jsonify({
                'success': False,
                'error': 'Missing required parameter',
                'key': str(error)
            }), 400

        @app.errorhandler(Exception)
        def handle_general_error(error):
            # Log the error for debugging
            app.logger.error(f"Unhandled exception: {str(error)}", exc_info=True)

            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            }), 500

    @staticmethod
    def log_requests(f: Callable):
        """Decorator to log API requests"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()

            # Log request
            app_logger = logging.getLogger('api')
            app_logger.info(f"REQUEST: {request.method} {request.path} - {request.remote_addr}")

            try:
                result = f(*args, **kwargs)

                # Calculate response time
                response_time = time.time() - start_time

                # Log successful response
                if isinstance(result, tuple):
                    response, status_code = result[0], result[1] if len(result) > 1 else 200
                else:
                    response, status_code = result, 200

                app_logger.info(f"RESPONSE: {request.path} - Status: {status_code} - Time: {response_time:.3f}s")

                return result
            except Exception as e:
                # Log error response
                response_time = time.time() - start_time
                app_logger.error(f"ERROR: {request.path} - {str(e)} - Time: {response_time:.3f}s")
                raise

        return decorated_function


class RateLimitMiddleware:
    """Simple rate limiting middleware"""

    def __init__(self):
        self.requests = {}
        self.window_size = 60  # 60 seconds
        self.max_requests = 100  # max requests per window

    def check_rate_limit(self, f: Callable):
        """Decorator to check rate limit"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr

            # Initialize client data if not exists
            if client_ip not in self.requests:
                self.requests[client_ip] = []

            # Clean old requests
            current_time = time.time()
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < self.window_size
            ]

            # Check if rate limit exceeded
            if len(self.requests[client_ip]) >= self.max_requests:
                return jsonify({
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {self.max_requests} requests per {self.window_size} seconds'
                }), 429

            # Add current request
            self.requests[client_ip].append(current_time)

            return f(*args, **kwargs)
        return decorated_function


class AuthenticationMiddleware:
    """Basic authentication middleware"""

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}

    def require_auth(self, f: Callable):
        """Decorator to require authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check for API key in header
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({
                    'success': False,
                    'error': 'Authorization header required'
                }), 401

            # Extract API key (format: "Bearer <api_key>" or just "<api_key>")
            if auth_header.startswith('Bearer '):
                api_key = auth_header[7:].strip()
            else:
                api_key = auth_header.strip()

            # Validate API key
            if api_key not in self.api_keys.values():
                return jsonify({
                    'success': False,
                    'error': 'Invalid API key'
                }), 401

            # Add user info to request context
            for user, key in self.api_keys.items():
                if key == api_key:
                    g.current_user = user
                    break

            return f(*args, **kwargs)
        return decorated_function


# Example schemas for validation
SIMULATION_PROFILE_SCHEMA = {
    'required': ['name', 'type', 'physics_settings', 'visual_quality', 'hardware_requirements'],
    'properties': {
        'name': {
            'type': 'string',
            'minLength': 1,
            'maxLength': 50,
            'pattern': r'^[a-zA-Z0-9_-]+$'
        },
        'type': {
            'type': 'string',
            'enum': ['high_fidelity', 'performance', 'education']
        },
        'description': {
            'type': 'string',
            'maxLength': 500
        },
        'use_case': {
            'type': 'string',
            'maxLength': 500
        },
        'physics_settings': {
            'type': 'object',
            'required': ['engine', 'time_step', 'real_time_factor'],
            'properties': {
                'engine': {
                    'type': 'string',
                    'enum': ['ode', 'bullet', 'dart']
                },
                'time_step': {
                    'type': 'number',
                    'minimum': 0.0001,
                    'maximum': 0.01
                },
                'real_time_factor': {
                    'type': 'number',
                    'minimum': 0.1,
                    'maximum': 10.0
                }
            }
        },
        'visual_quality': {
            'type': 'object',
            'required': ['rendering_quality'],
            'properties': {
                'rendering_quality': {
                    'type': 'string',
                    'enum': ['low', 'medium', 'high', 'ultra']
                },
                'shadows': {
                    'type': 'boolean'
                },
                'reflections': {
                    'type': 'boolean'
                }
            }
        },
        'hardware_requirements': {
            'type': 'object',
            'properties': {
                'min_cpu_cores': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 64
                },
                'min_ram_gb': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 512
                }
            }
        }
    }
}


def setup_middleware(app):
    """Set up all middleware for the Flask app"""
    # Initialize rate limiter
    rate_limiter = RateLimitMiddleware()

    # Register error handlers
    ErrorHandlingMiddleware.handle_errors(app)

    # Add logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s'
    )

    # Wrap app routes with middleware as needed
    # This would typically be done by decorating specific routes
    return app


# Example usage
if __name__ == "__main__":
    from flask import Flask

    app = Flask(__name__)

    # Setup middleware
    app = setup_middleware(app)

    @app.route('/test-validation', methods=['POST'])
    @ValidationMiddleware.validate_json_schema(SIMULATION_PROFILE_SCHEMA)
    @ErrorHandlingMiddleware.log_requests
    def test_validation():
        data = request.get_json()
        return jsonify({
            'success': True,
            'received': data
        })

    @app.route('/test-rate-limit')
    @rate_limiter.check_rate_limit
    def test_rate_limit():
        return jsonify({
            'success': True,
            'message': 'Rate limit test passed'
        })

    app.run(debug=True)