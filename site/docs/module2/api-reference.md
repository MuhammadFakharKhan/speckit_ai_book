# API Reference: Module 2 - The Digital Twin (Gazebo & Unity)

This document provides a comprehensive reference for the REST APIs available in Module 2, which focuses on digital twin simulation using Gazebo for physics and Unity for visualization.

## Base URL

All API endpoints are relative to: `http://localhost:5001/api/`

## Authentication

The API does not require authentication for development purposes. In production environments, authentication should be implemented using tokens or OAuth.

## Common Response Format

All API responses follow this structure:

```json
{
  "success": true,
  "data": { ... },
  "message": "Optional message"
}
```

For errors:

```json
{
  "success": false,
  "error": "Error message",
  "message": "Detailed error information"
}
```

## Simulation Profile Management API

### GET /profiles

Get a list of all available simulation profiles.

**Response:**
```json
{
  "success": true,
  "profiles": [
    {
      "name": "education",
      "type": "education",
      "description": "Balanced simulation for educational purposes",
      "use_case": "Use for learning and educational demonstrations",
      "physics_settings": {
        "engine": "ode",
        "time_step": 0.001,
        "real_time_factor": 1.0
      },
      "visual_quality": {
        "rendering_quality": "medium",
        "shadows": true
      }
    }
  ],
  "count": 1
}
```

### GET `/profiles/{profile_name}`

Get details of a specific simulation profile.

**Path Parameters:**
- `profile_name` (string): Name of the profile to retrieve

**Response:**
```json
{
  "success": true,
  "profile": {
    "name": "education",
    "type": "education",
    "description": "Balanced simulation for educational purposes",
    "use_case": "Use for learning and educational demonstrations",
    "physics_settings": {
      "engine": "ode",
      "gravity_x": 0.0,
      "gravity_y": 0.0,
      "gravity_z": -9.81,
      "time_step": 0.001,
      "real_time_factor": 1.0
    },
    "visual_quality": {
      "rendering_quality": "medium",
      "shadows": true,
      "reflections": false,
      "anti_aliasing": "fxaa"
    },
    "hardware_requirements": {
      "min_cpu_cores": 4,
      "min_ram_gb": 8,
      "min_gpu_vram_gb": 1,
      "recommended_cpu_cores": 6,
      "recommended_ram_gb": 12,
      "recommended_gpu_vram_gb": 2
    }
  }
}
```

### POST /profiles

Create a new simulation profile.

**Request Body:**
```json
{
  "name": "custom_profile",
  "type": "performance",
  "description": "Custom performance profile",
  "use_case": "Use for high-performance simulation",
  "physics_settings": {
    "engine": "ode",
    "time_step": 0.002,
    "real_time_factor": 2.0
  },
  "visual_quality": {
    "rendering_quality": "low",
    "shadows": false,
    "reflections": false,
    "anti_aliasing": "fxaa"
  },
  "hardware_requirements": {
    "min_cpu_cores": 2,
    "min_ram_gb": 4,
    "min_gpu_vram_gb": 1
  }
}
```

**Response:**
```json
{
  "success": true,
  "profile": {
    "name": "custom_profile",
    "type": "performance",
    "description": "Custom performance profile"
  }
}
```

### PUT `/profiles/{profile_name}`

Update an existing simulation profile.

**Path Parameters:**
- `profile_name` (string): Name of the profile to update

**Request Body:** Same format as POST but only specified fields are updated.

### DELETE `/profiles/{profile_name}`

Delete a simulation profile.

> **Note:** Profile deletion is not implemented in this example to prevent accidental deletion of profiles in use. In a production system, this would include checks to ensure the profile is not currently in use.

### POST `/profiles/{profile_name}/apply`

Apply a simulation profile to the current simulation environment.

**Path Parameters:**
- `profile_name` (string): Name of the profile to apply

**Response:**
```json
{
  "success": true,
  "message": "Profile education applied successfully",
  "profile_name": "education"
}
```

## Simulation Management API

### GET /simulations

Get a list of available simulations.

**Response:**
```json
{
  "success": true,
  "simulations": [
    {
      "name": "humanoid_basic",
      "type": "physics",
      "description": "Basic humanoid physics simulation",
      "path": "examples/gazebo/worlds/basic_humanoid.sdf",
      "status": "available"
    }
  ],
  "count": 1
}
```

### POST /simulation/start

Start a simulation.

**Request Body:**
```json
{
  "simulation_name": "humanoid_basic",
  "profile_name": "education"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Simulation started successfully",
  "session_id": "sim_12345"
}
```

### POST /simulation/stop

Stop a running simulation.

**Request Body:**
```json
{
  "session_id": "sim_12345"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Simulation stopped successfully"
}
```

## Robot Control API

### GET `/robot/{robot_id}/joints`

Get current joint states for a robot.

**Path Parameters:**
- `robot_id` (string): ID of the robot

**Response:**
```json
{
  "success": true,
  "joint_states": {
    "hip_joint": {
      "position": 0.0,
      "velocity": 0.0,
      "effort": 0.0
    },
    "knee_joint": {
      "position": 0.0,
      "velocity": 0.0,
      "effort": 0.0
    }
  }
}
```

### POST `/robot/{robot_id}/joints`

Control robot joints.

**Path Parameters:**
- `robot_id` (string): ID of the robot

**Request Body:**
```json
{
  "joint_commands": {
    "hip_joint": 0.5,
    "knee_joint": -0.2
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Joint commands sent successfully"
}
```

## Sensor Data API

### GET `/robot/{robot_id}/sensors`

Get sensor data from a robot.

**Path Parameters:**
- `robot_id` (string): ID of the robot

**Response:**
```json
{
  "success": true,
  "sensors": {
    "camera": {
      "topic": "/camera/image_raw",
      "data": "base64_encoded_image_data",
      "timestamp": 1678886400
    },
    "lidar": {
      "topic": "/lidar/scan",
      "ranges": [1.0, 1.5, 2.0, ...],
      "timestamp": 1678886400
    }
  }
}
```

## Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request - Invalid input parameters |
| 404 | Not Found - Requested resource does not exist |
| 405 | Method Not Allowed - Requested method is not supported |
| 500 | Internal Server Error - Something went wrong on the server |

## Example Usage

### Using cURL

```bash
# Get all simulation profiles
curl -X GET http://localhost:5001/api/profiles

# Get specific profile
curl -X GET http://localhost:5001/api/profiles/education

# Apply a profile
curl -X POST http://localhost:5001/api/profiles/education/apply
```

### Using Python requests

```python
import requests

# Get all profiles
response = requests.get('http://localhost:5001/api/profiles')
profiles = response.json()

# Apply a profile
response = requests.post('http://localhost:5001/api/profiles/education/apply')
result = response.json()
print(result['message'])
```

## Rate Limiting

API endpoints are not rate-limited in the development environment. In production, rate limiting should be implemented to prevent abuse.