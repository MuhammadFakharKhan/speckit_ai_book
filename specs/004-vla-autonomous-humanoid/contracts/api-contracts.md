# API Contracts: Vision-Language-Action (VLA) System

## Voice Recognition Service

### POST /voice/process
**Description**: Process incoming voice data and return transcribed text with confidence score

**Request**:
```json
{
  "audio_data": "base64_encoded_audio",
  "language": "en-US",
  "sample_rate": 16000
}
```

**Response**:
```json
{
  "id": "uuid",
  "text": "transcribed text from audio",
  "confidence": 0.92,
  "language": "en-US",
  "timestamp": "2025-12-26T10:00:00Z",
  "intent": "navigation",
  "parameters": {
    "destination": "kitchen"
  }
}
```

**Error Responses**:
- 400: Invalid audio format
- 422: Audio too short or too noisy
- 500: Speech recognition service error

## Cognitive Planning Service

### POST /plan/generate
**Description**: Generate action plan from natural language command

**Request**:
```json
{
  "command_id": "uuid",
  "text": "go to the kitchen and bring me the red cup",
  "robot_capabilities": ["navigation", "manipulation", "perception"],
  "environment_context": {
    "robot_position": {"x": 0, "y": 0, "z": 0},
    "known_locations": ["kitchen", "living_room", "bedroom"]
  }
}
```

**Response**:
```json
{
  "plan_id": "uuid",
  "original_command": "go to the kitchen and bring me the red cup",
  "action_sequence_id": "uuid",
  "subtasks": [
    {
      "id": "uuid",
      "type": "navigation",
      "description": "navigate to kitchen",
      "parameters": {
        "destination": "kitchen"
      }
    },
    {
      "id": "uuid",
      "type": "perception",
      "description": "locate red cup in kitchen",
      "parameters": {
        "object_type": "cup",
        "color": "red"
      }
    }
  ],
  "estimated_duration": 120,
  "confidence": 0.85
}
```

**Error Responses**:
- 400: Invalid command format
- 422: Command cannot be executed with available capabilities
- 500: Planning service error

## Action Execution Service

### POST /action/execute
**Description**: Execute an action sequence on the robot

**Request**:
```json
{
  "action_sequence_id": "uuid",
  "steps": [
    {
      "id": "uuid",
      "type": "navigation",
      "parameters": {
        "destination": {"x": 5.0, "y": 3.0, "z": 0.0}
      },
      "timeout": 60
    }
  ],
  "robot_id": "robot_1"
}
```

**Response**:
```json
{
  "execution_id": "uuid",
  "status": "executing",
  "progress": 0.0,
  "estimated_completion": "2025-12-26T10:02:30Z"
}
```

### GET /action/status/{execution_id}
**Description**: Get the current status of an action execution

**Response**:
```json
{
  "execution_id": "uuid",
  "status": "completed",
  "progress": 1.0,
  "completed_steps": 3,
  "total_steps": 3,
  "execution_log": [
    {
      "step_id": "uuid",
      "status": "completed",
      "timestamp": "2025-12-26T10:00:30Z",
      "result": "Successfully navigated to kitchen"
    }
  ]
}
```

**Error Responses**:
- 404: Execution ID not found
- 500: Execution service error

## Simulation Service

### POST /simulation/start
**Description**: Start a simulation environment for testing

**Request**:
```json
{
  "environment_name": "home_layout_1",
  "robot_type": "humanoid_a",
  "initial_position": {"x": 0, "y": 0, "z": 0}
}
```

**Response**:
```json
{
  "simulation_id": "uuid",
  "status": "running",
  "robot_endpoint": "ws://localhost:9090",
  "visualization_url": "http://localhost:8080"
}
```

## ROS 2 Command Interface

### Message Types

**VoiceCommand.msg**:
```
string text
float32 confidence
string language
string intent
KeyValue[] parameters
time timestamp
```

**ActionStep.msg**:
```
string id
string type  # navigation, perception, manipulation
KeyValue[] parameters
duration timeout
string[] dependencies
```

**ActionSequence.msg**:
```
string id
ActionStep[] steps
string status  # pending, executing, completed, failed
duration estimated_duration
float32 priority
```