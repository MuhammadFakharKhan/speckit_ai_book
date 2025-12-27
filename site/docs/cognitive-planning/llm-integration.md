---
title: LLM Integration for Cognitive Planning
description: Documentation on integrating Large Language Models for cognitive planning in VLA systems
sidebar_position: 2
tags: [vla, llm, cognitive-planning, ai-integration, task-decomposition]
---

# LLM Integration for Cognitive Planning

## Overview

Large Language Model (LLM) integration forms the core intelligence mechanism for cognitive planning in the Vision-Language-Action (VLA) system. This integration leverages the advanced reasoning, natural language understanding, and knowledge capabilities of LLMs to transform high-level natural language commands into structured, executable action plans for humanoid robots.

## LLM Integration Architecture

### System Architecture

The LLM integration follows a modular architecture that separates the language model interface from the planning logic:

```
Natural Language Input → LLM Interface → Prompt Engineering → LLM Processing → Response Parsing → Plan Generation → Validation
```

This architecture ensures flexibility in LLM selection while maintaining consistent planning interfaces.

### Core Components

#### 1. LLM Interface Layer

The LLM interface provides a standardized way to interact with different language models:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json

class LLMInterface(ABC):
    """
    Abstract interface for LLM integration
    """
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from LLM
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text
        """
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Get token count for text
        """
        pass

class OpenAILLMInterface(LLMInterface):
    """
    OpenAI API implementation of LLM interface
    """
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using OpenAI API
        """
        params = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1000),
            'response_format': kwargs.get('response_format', {'type': 'text'})
        }

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI embeddings API
        """
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def get_token_count(self, text: str) -> int:
        """
        Estimate token count using tiktoken
        """
        import tiktoken
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))

class LocalLLMInterface(LLMInterface):
    """
    Local LLM implementation using Hugging Face models
    """
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        from transformers import pipeline
        self.generator = pipeline('text-generation', model=model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using local model
        """
        max_length = kwargs.get('max_tokens', 200)
        temperature = kwargs.get('temperature', 0.7)

        response = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=50256  # Default for GPT-2
        )

        return response[0]['generated_text'][len(prompt):]
```

#### 2. Prompt Engineering System

The prompt engineering system creates optimized prompts for planning tasks:

```python
class PromptEngineer:
    """
    System for creating and optimizing prompts for cognitive planning
    """
    def __init__(self):
        self.templates = self._load_planning_templates()
        self.context_builder = ContextBuilder()

    def create_planning_prompt(self, task_description: str, context: Dict[str, Any],
                             planning_type: str = "task_decomposition") -> str:
        """
        Create optimized prompt for cognitive planning
        """
        template = self.templates.get(planning_type, self.templates['default'])

        # Build comprehensive context
        full_context = self.context_builder.build_context(context)

        # Format prompt with context and task
        prompt = template.format(
            task_description=task_description,
            context=full_context,
            current_datetime=self._get_current_datetime(),
            robot_capabilities=full_context.get('robot_capabilities', {}),
            environment_state=full_context.get('environment_state', {}),
            safety_constraints=full_context.get('safety_constraints', {})
        )

        return prompt

    def _load_planning_templates(self) -> Dict[str, str]:
        """
        Load planning-specific prompt templates
        """
        return {
            'task_decomposition': """You are an expert cognitive planning system for humanoid robots. Your task is to decompose complex natural language commands into executable subtasks.

Context Information:
- Robot Capabilities: {robot_capabilities}
- Environment State: {environment_state}
- Safety Constraints: {safety_constraints}

Current Time: {current_datetime}

Task to Decompose: {task_description}

Please decompose this task into a sequence of specific, executable subtasks. Each subtask should be:
1. Specific and unambiguous
2. Executable by the robot
3. Include all necessary parameters
4. Consider safety and feasibility

Provide your response in the following JSON format:
{{
    "task_summary": "...",
    "decomposition_confidence": 0.0-1.0,
    "subtasks": [
        {{
            "id": "unique_id",
            "description": "Clear description of the subtask",
            "type": "navigation|manipulation|perception|communication|wait",
            "parameters": {{
                "location": "...",
                "object": "...",
                "action": "...",
                "duration": 0.0
            }},
            "dependencies": ["other_task_id"],
            "success_criteria": ["condition1", "condition2"],
            "estimated_duration": 0.0,
            "priority": 0.0-1.0
        }}
    ],
    "overall_constraints": ["constraint1", "constraint2"],
    "fallback_strategies": [
        {{
            "trigger": "failure_condition",
            "strategy": "alternative_approach"
        }}
    ]
}}""",

            'action_sequencing': """You are an expert action sequencing system for humanoid robots. Given the following subtasks, create an optimal execution sequence considering dependencies, resource availability, and safety.

Context Information:
- Robot Capabilities: {robot_capabilities}
- Environment State: {environment_state}
- Safety Constraints: {safety_constraints}

Current Time: {current_datetime}

Subtasks to Sequence:
{task_description}

Please sequence these subtasks optimally, considering:
1. Dependencies between tasks
2. Resource availability
3. Safety requirements
4. Efficiency of execution

Provide your response in the following JSON format:
{{
    "sequencing_confidence": 0.0-1.0,
    "execution_sequence": [
        {{
            "task_id": "...",
            "execution_order": 0,
            "estimated_start_time": "...",
            "estimated_duration": 0.0,
            "required_resources": ["resource1", "resource2"]
        }}
    ],
    "optimization_reasoning": "Explanation of sequencing decisions",
    "safety_considerations": ["consideration1", "consideration2"]
}}""",

            'plan_validation': """You are an expert plan validation system for humanoid robots. Validate the following plan for feasibility, safety, and completeness.

Context Information:
- Robot Capabilities: {robot_capabilities}
- Environment State: {environment_state}
- Safety Constraints: {safety_constraints}

Current Time: {current_datetime}

Plan to Validate:
{task_description}

Please validate this plan and identify any issues related to:
1. Feasibility given robot capabilities
2. Safety violations
3. Missing steps or information
4. Resource conflicts
5. Temporal inconsistencies

Provide your response in the following JSON format:
{{
    "is_valid": true|false,
    "validation_confidence": 0.0-1.0,
    "issues": [
        {{
            "type": "feasibility|safety|completeness|resource|temporal",
            "severity": "critical|high|medium|low",
            "description": "Issue description",
            "suggestion": "Suggested fix"
        }}
    ],
    "overall_assessment": "Summary of validation results",
    "risk_level": "low|medium|high"
}}""",

            'default': """You are an expert cognitive planning system for humanoid robots. Analyze the following task and provide a structured plan.

Context: {context}
Task: {task_description}

Provide a detailed analysis and plan."""
        }

    def _get_current_datetime(self) -> str:
        """
        Get current datetime for context
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

#### 3. Response Parser

The response parser converts LLM responses into structured planning data:

```python
import json
import re
from typing import Union

class ResponseParser:
    """
    Parse LLM responses into structured planning data
    """
    def __init__(self):
        self.json_fixer = JSONFixer()

    def parse_planning_response(self, llm_response: str, expected_format: str = "task_decomposition"):
        """
        Parse LLM response into structured planning data
        """
        try:
            # Try to extract JSON from response
            json_content = self._extract_json_from_response(llm_response)

            if json_content:
                parsed_data = json.loads(json_content)
                return self._validate_parsed_data(parsed_data, expected_format)
            else:
                # If no JSON found, try to parse as text
                return self._parse_text_response(llm_response, expected_format)

        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed_json = self.json_fixer.fix_json(llm_response)
            if fixed_json:
                try:
                    parsed_data = json.loads(fixed_json)
                    return self._validate_parsed_data(parsed_data, expected_format)
                except json.JSONDecodeError:
                    pass

            # If all parsing fails, return text response
            return {
                'raw_response': llm_response,
                'parsed_successfully': False,
                'error': 'Could not parse LLM response into structured format'
            }

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """
        Extract JSON content from LLM response
        """
        # Look for JSON between ```json and ``` or ``` and ```
        json_pattern = r'```(?:json)?\s*({.*?})\s*```'
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            return match.group(1)

        # Look for JSON object directly
        json_obj_pattern = r'\{.*\}'
        matches = re.findall(json_obj_pattern, response, re.DOTALL)

        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

        return None

    def _validate_parsed_data(self, data: Dict, expected_format: str) -> Dict:
        """
        Validate parsed data against expected format
        """
        validation_result = {
            'parsed_successfully': True,
            'data': data,
            'validation_issues': [],
            'confidence': 0.8  # Default confidence
        }

        if expected_format == "task_decomposition":
            validation_result['validation_issues'] = self._validate_task_decomposition(data)
        elif expected_format == "action_sequencing":
            validation_result['validation_issues'] = self._validate_action_sequencing(data)
        elif expected_format == "plan_validation":
            validation_result['validation_issues'] = self._validate_plan_validation(data)

        # Calculate confidence based on validation issues
        if validation_result['validation_issues']:
            validation_result['confidence'] = max(0.1, 0.8 - len(validation_result['validation_issues']) * 0.1)

        return validation_result

    def _validate_task_decomposition(self, data: Dict) -> List[str]:
        """
        Validate task decomposition data
        """
        issues = []

        if 'subtasks' not in data:
            issues.append("Missing 'subtasks' field in task decomposition")
            return issues

        for i, subtask in enumerate(data['subtasks']):
            if not isinstance(subtask, dict):
                issues.append(f"Subtask {i} is not a dictionary")
                continue

            required_fields = ['id', 'description', 'type']
            for field in required_fields:
                if field not in subtask:
                    issues.append(f"Subtask {i} missing required field: {field}")

            if subtask.get('type') not in ['navigation', 'manipulation', 'perception', 'communication', 'wait']:
                issues.append(f"Subtask {i} has invalid type: {subtask.get('type')}")

        return issues

    def _validate_action_sequencing(self, data: Dict) -> List[str]:
        """
        Validate action sequencing data
        """
        issues = []

        if 'execution_sequence' not in data:
            issues.append("Missing 'execution_sequence' field in action sequencing")
            return issues

        for i, step in enumerate(data['execution_sequence']):
            if not isinstance(step, dict):
                issues.append(f"Execution step {i} is not a dictionary")
                continue

            required_fields = ['task_id', 'execution_order']
            for field in required_fields:
                if field not in step:
                    issues.append(f"Execution step {i} missing required field: {field}")

        return issues

    def _validate_plan_validation(self, data: Dict) -> List[str]:
        """
        Validate plan validation data
        """
        issues = []

        required_fields = ['is_valid', 'issues']
        for field in required_fields:
            if field not in data:
                issues.append(f"Plan validation missing required field: {field}")

        return issues

    def _parse_text_response(self, response: str, expected_format: str) -> Dict:
        """
        Parse text response when JSON is not available
        """
        return {
            'raw_response': response,
            'parsed_successfully': False,
            'error': f'Could not extract structured {expected_format} data from text response',
            'confidence': 0.3  # Low confidence for text-only parsing
        }
```

## Planning-Specific LLM Integration

### Task Decomposition Integration

The system uses specialized prompting for task decomposition:

```python
class TaskDecompositionLLM:
    """
    LLM integration specifically for task decomposition
    """
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.prompt_engineer = PromptEngineer()
        self.response_parser = ResponseParser()

    def decompose_task(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompose a task using LLM
        """
        # Create optimized prompt
        prompt = self.prompt_engineer.create_planning_prompt(
            task_description,
            context,
            planning_type="task_decomposition"
        )

        # Generate response from LLM
        llm_response = self.llm.generate(
            prompt,
            temperature=0.3,  # Lower temperature for consistency
            max_tokens=1500
        )

        # Parse the response
        parsed_result = self.response_parser.parse_planning_response(
            llm_response,
            expected_format="task_decomposition"
        )

        # Add metadata
        parsed_result['original_prompt'] = prompt
        parsed_result['llm_response'] = llm_response
        parsed_result['decomposition_timestamp'] = self._get_current_timestamp()

        return parsed_result

    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp for logging
        """
        from datetime import datetime
        return datetime.now().isoformat()

    def batch_decompose_tasks(self, tasks: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose multiple tasks efficiently
        """
        results = []

        for task in tasks:
            result = self.decompose_task(task['description'], context)
            result['task_id'] = task.get('id', 'unknown')
            results.append(result)

        return results
```

### Action Sequencing Integration

The system uses specialized prompting for action sequencing:

```python
class ActionSequencingLLM:
    """
    LLM integration specifically for action sequencing
    """
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.prompt_engineer = PromptEngineer()
        self.response_parser = ResponseParser()

    def sequence_actions(self, subtasks: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sequence actions using LLM
        """
        # Convert subtasks to string format for LLM
        tasks_str = json.dumps(subtasks, indent=2)

        # Create optimized prompt
        prompt = self.prompt_engineer.create_planning_prompt(
            tasks_str,
            context,
            planning_type="action_sequencing"
        )

        # Generate response from LLM
        llm_response = self.llm.generate(
            prompt,
            temperature=0.2,  # Very low temperature for consistency
            max_tokens=1000
        )

        # Parse the response
        parsed_result = self.response_parser.parse_planning_response(
            llm_response,
            expected_format="action_sequencing"
        )

        # Add metadata
        parsed_result['original_prompt'] = prompt
        parsed_result['llm_response'] = llm_response
        parsed_result['sequencing_timestamp'] = self._get_current_timestamp()

        return parsed_result

    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp for logging
        """
        from datetime import datetime
        return datetime.now().isoformat()
```

### Plan Validation Integration

The system uses specialized prompting for plan validation:

```python
class PlanValidationLLM:
    """
    LLM integration specifically for plan validation
    """
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.prompt_engineer = PromptEngineer()
        self.response_parser = ResponseParser()

    def validate_plan(self, plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a plan using LLM
        """
        # Convert plan to string format for LLM
        plan_str = json.dumps(plan, indent=2)

        # Create optimized prompt
        prompt = self.prompt_engineer.create_planning_prompt(
            plan_str,
            context,
            planning_type="plan_validation"
        )

        # Generate response from LLM
        llm_response = self.llm.generate(
            prompt,
            temperature=0.1,  # Very low temperature for consistency
            max_tokens=1200
        )

        # Parse the response
        parsed_result = self.response_parser.parse_planning_response(
            llm_response,
            expected_format="plan_validation"
        )

        # Add metadata
        parsed_result['original_prompt'] = prompt
        parsed_result['llm_response'] = llm_response
        parsed_result['validation_timestamp'] = self._get_current_timestamp()

        return parsed_result

    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp for logging
        """
        from datetime import datetime
        return datetime.now().isoformat()
```

## Context Integration

### Environmental Context

The LLM integration incorporates environmental context:

```python
class ContextBuilder:
    """
    Build comprehensive context for LLM planning
    """
    def __init__(self):
        self.location_resolver = LocationResolver()
        self.object_context = ObjectContextProvider()

    def build_context(self, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive context from base context
        """
        full_context = base_context.copy()

        # Add resolved locations
        if 'known_locations' in full_context:
            full_context['resolved_locations'] = self._resolve_locations(
                full_context['known_locations']
            )

        # Add object context
        if 'visible_objects' in full_context:
            full_context['object_context'] = self.object_context.get_object_context(
                full_context['visible_objects']
            )

        # Add temporal context
        full_context['temporal_context'] = self._get_temporal_context()

        # Add safety context
        full_context['safety_context'] = self._get_safety_context()

        return full_context

    def _resolve_locations(self, locations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve location coordinates and accessibility
        """
        resolved = {}
        for name, location in locations.items():
            resolved[name] = {
                'coordinates': location.get('coordinates'),
                'accessible': location.get('accessible', True),
                'navigation_difficulty': location.get('difficulty', 'normal'),
                'description': location.get('description', '')
            }
        return resolved

    def _get_temporal_context(self) -> Dict[str, Any]:
        """
        Get temporal context information
        """
        from datetime import datetime
        now = datetime.now()

        return {
            'current_time': now.strftime('%H:%M:%S'),
            'current_day': now.strftime('%A'),
            'current_date': now.strftime('%Y-%m-%d'),
            'time_of_day': self._get_time_of_day(now),
            'operational_hours': self._get_operational_hours()
        }

    def _get_time_of_day(self, dt: datetime) -> str:
        """
        Determine time of day
        """
        hour = dt.hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    def _get_operational_hours(self) -> Dict[str, str]:
        """
        Get operational hours context
        """
        return {
            'start': '08:00',
            'end': '20:00',
            'current_in_operational_hours': True  # Would be determined dynamically
        }

    def _get_safety_context(self) -> Dict[str, Any]:
        """
        Get safety-related context
        """
        return {
            'safety_zones': ['no_go', 'caution', 'safe'],
            'emergency_procedures': ['stop_immediately', 'return_to_base'],
            'risk_assessment': 'normal'
        }
```

### Robot State Context

The integration incorporates robot state information:

```python
class RobotStateContextProvider:
    """
    Provide robot state context for LLM planning
    """
    def __init__(self):
        self.state_monitor = RobotStateMonitor()

    def get_robot_context(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive robot state context
        """
        return {
            'capabilities': self._get_robot_capabilities(robot_state),
            'current_status': self._get_current_status(robot_state),
            'resource_availability': self._get_resource_availability(robot_state),
            'performance_metrics': self._get_performance_metrics(robot_state),
            'constraint_limits': self._get_constraint_limits(robot_state)
        }

    def _get_robot_capabilities(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract robot capabilities from state
        """
        return {
            'navigation': robot_state.get('navigation_capability', False),
            'manipulation': robot_state.get('manipulation_capability', False),
            'perception': robot_state.get('perception_capability', False),
            'communication': robot_state.get('communication_capability', False),
            'locomotion_types': robot_state.get('locomotion_types', ['walking']),
            'max_speed': robot_state.get('max_speed', 1.0),
            'payload_capacity': robot_state.get('payload_capacity', 5.0)  # kg
        }

    def _get_current_status(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get current robot status
        """
        return {
            'position': robot_state.get('position', {'x': 0.0, 'y': 0.0, 'z': 0.0}),
            'battery_level': robot_state.get('battery_level', 1.0),  # 0.0-1.0
            'operational': robot_state.get('operational', True),
            'current_task': robot_state.get('current_task', 'idle'),
            'error_status': robot_state.get('error_status', 'none'),
            'active_modules': robot_state.get('active_modules', [])
        }

    def _get_resource_availability(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get resource availability information
        """
        battery_level = robot_state.get('battery_level', 1.0)

        return {
            'battery_available': battery_level > 0.2,  # 20% threshold
            'estimated_runtime': self._calculate_runtime(battery_level),
            'compute_resources': robot_state.get('compute_available', True),
            'manipulator_available': robot_state.get('manipulator_status', 'available') == 'available',
            'navigation_available': robot_state.get('navigation_status', 'available') == 'available'
        }

    def _calculate_runtime(self, battery_level: float) -> float:
        """
        Calculate estimated runtime based on battery level
        """
        # Assume 2 hours runtime at 100% battery
        return battery_level * 2.0 * 60 * 60  # Convert to seconds

    def _get_performance_metrics(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get performance metrics for planning context
        """
        return {
            'navigation_accuracy': robot_state.get('navigation_accuracy', 0.95),
            'manipulation_success_rate': robot_state.get('manipulation_success_rate', 0.90),
            'task_completion_rate': robot_state.get('task_completion_rate', 0.85),
            'average_task_duration': robot_state.get('avg_task_duration', 120.0)  # seconds
        }

    def _get_constraint_limits(self, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get constraint limits for planning
        """
        return {
            'max_navigation_distance': robot_state.get('max_navigation_distance', 50.0),  # meters
            'max_manipulation_weight': robot_state.get('max_manipulation_weight', 5.0),  # kg
            'max_operation_time': robot_state.get('max_operation_time', 8.0),  # hours
            'safety_buffer_distance': robot_state.get('safety_buffer', 0.5)  # meters
        }
```

## Error Handling and Fallback Strategies

### LLM Error Handling

The system implements comprehensive error handling for LLM interactions:

```python
import time
import random
from enum import Enum

class LLMErrorType(Enum):
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    CONTENT_FILTERED = "content_filtered"
    RATE_LIMITED = "rate_limited"
    CONTEXT_OVERFLOW = "context_overflow"
    PARSING_ERROR = "parsing_error"

class LLMErrorHandler:
    """
    Handle errors from LLM interactions with appropriate fallbacks
    """
    def __init__(self):
        self.fallback_strategies = self._initialize_fallback_strategies()

    def handle_llm_error(self, error: Exception, error_type: LLMErrorType,
                        original_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle LLM error and attempt recovery
        """
        strategy = self.fallback_strategies.get(error_type, self._default_fallback)

        try:
            return strategy(error, original_request, context)
        except Exception as fallback_error:
            # If fallback also fails, return safe default
            return self._safe_default_response(error, error_type)

    def _initialize_fallback_strategies(self) -> Dict[LLMErrorType, callable]:
        """
        Initialize fallback strategies for different error types
        """
        return {
            LLMErrorType.API_ERROR: self._handle_api_error,
            LLMErrorType.TIMEOUT_ERROR: self._handle_timeout_error,
            LLMErrorType.CONTENT_FILTERED: self._handle_content_filtered,
            LLMErrorType.RATE_LIMITED: self._handle_rate_limited,
            LLMErrorType.CONTEXT_OVERFLOW: self._handle_context_overflow,
            LLMErrorType.PARSING_ERROR: self._handle_parsing_error
        }

    def _handle_api_error(self, error, original_request, context):
        """
        Handle API-related errors
        """
        # Log the error
        print(f"API Error: {str(error)}")

        # Try with reduced context
        reduced_context = self._reduce_context_size(context)
        return self._retry_with_context(original_request, reduced_context)

    def _handle_timeout_error(self, error, original_request, context):
        """
        Handle timeout errors
        """
        print(f"Timeout Error: {str(error)}")

        # Retry with higher temperature (simpler response)
        return self._retry_with_params(original_request, context, temperature=0.9)

    def _handle_content_filtered(self, error, original_request, context):
        """
        Handle content filtering errors
        """
        print(f"Content Filtered: {str(error)}")

        # Retry with safer prompt
        safer_prompt = self._make_prompt_safer(original_request)
        return self._retry_with_prompt(safer_prompt, context)

    def _handle_rate_limited(self, error, original_request, context):
        """
        Handle rate limiting errors
        """
        print(f"Rate Limited: {str(error)}")

        # Wait and retry
        time.sleep(random.uniform(1, 3))  # Random backoff
        return self._retry_with_context(original_request, context)

    def _handle_context_overflow(self, error, original_request, context):
        """
        Handle context overflow errors
        """
        print(f"Context Overflow: {str(error)}")

        # Significantly reduce context
        reduced_context = self._aggressively_reduce_context(context)
        return self._retry_with_context(original_request, reduced_context)

    def _handle_parsing_error(self, error, original_request, context):
        """
        Handle parsing errors
        """
        print(f"Parsing Error: {str(error)}")

        # Try with different response format
        return self._retry_with_format(original_request, context, format_type="text")

    def _default_fallback(self, error, original_request, context):
        """
        Default fallback for unhandled errors
        """
        print(f"Unhandled LLM Error: {str(error)}")
        return self._safe_default_response(error, LLMErrorType.API_ERROR)

    def _safe_default_response(self, original_error, error_type):
        """
        Return a safe default response when all else fails
        """
        return {
            'success': False,
            'error_type': error_type.value,
            'error_message': str(original_error),
            'fallback_used': True,
            'suggestion': 'Please try rephrasing your request or breaking it into smaller tasks',
            'confidence': 0.1
        }

    def _reduce_context_size(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reduce context size by removing less critical information
        """
        reduced = context.copy()

        # Remove detailed object descriptions if present
        if 'object_context' in reduced and isinstance(reduced['object_context'], dict):
            for obj_key in reduced['object_context']:
                if isinstance(reduced['object_context'][obj_key], dict):
                    # Keep only essential properties
                    essential_props = ['type', 'position', 'id']
                    obj_context = reduced['object_context'][obj_key]
                    reduced['object_context'][obj_key] = {
                        k: v for k, v in obj_context.items() if k in essential_props
                    }

        return reduced

    def _aggressively_reduce_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggressively reduce context to minimum viable
        """
        return {
            'robot_capabilities': context.get('robot_capabilities', {}),
            'environment_state': context.get('environment_state', {})
        }

    def _make_prompt_safer(self, original_prompt: str) -> str:
        """
        Make prompt safer to avoid content filtering
        """
        # Remove any potentially sensitive instructions
        safe_prompt = original_prompt.replace("harmful", "safe").replace("dangerous", "safe")
        return safe_prompt

    def _retry_with_context(self, original_request: str, new_context: Dict[str, Any]):
        """
        Retry the original request with new context
        """
        # This would typically call back to the original planning function
        # with the new context
        pass

    def _retry_with_params(self, original_request: str, context: Dict[str, Any], **params):
        """
        Retry with different parameters
        """
        pass

    def _retry_with_prompt(self, new_prompt: str, context: Dict[str, Any]):
        """
        Retry with new prompt
        """
        pass

    def _retry_with_format(self, original_request: str, context: Dict[str, Any], format_type: str):
        """
        Retry with different response format
        """
        pass
```

## Performance Optimization

### Caching Strategies

The system implements caching to improve performance:

```python
from functools import lru_cache
import hashlib
import time

class LLMCache:
    """
    Cache for LLM responses to improve performance
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}

    @lru_cache(maxsize=1000)
    def get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Generate cache key for prompt and parameters
        """
        cache_input = f"{prompt}_{str(sorted(params.items()))}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response if still valid
        """
        if cache_key in self.cache:
            response, timestamp = self.cache[cache_key]

            # Check if response is still valid (not expired)
            if time.time() - timestamp < self.ttl_seconds:
                self.access_times[cache_key] = time.time()
                return response
            else:
                # Remove expired entry
                del self.cache[cache_key]
                del self.access_times[cache_key]

        return None

    def set(self, cache_key: str, response: Dict[str, Any]):
        """
        Set cached response
        """
        # Check if cache is at max size
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]

        self.cache[cache_key] = (response, time.time())
        self.access_times[cache_key] = time.time()

    def invalidate(self, cache_key: str = None):
        """
        Invalidate specific cache entry or all cache
        """
        if cache_key and cache_key in self.cache:
            del self.cache[cache_key]
            del self.access_times[cache_key]
        elif cache_key is None:
            # Clear entire cache
            self.cache.clear()
            self.access_times.clear()
```

### Rate Limiting

The system implements rate limiting to manage API usage:

```python
import time
from collections import deque

class RateLimiter:
    """
    Rate limiter for LLM API calls
    """
    def __init__(self, max_calls: int = 10, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times = deque()

    def can_make_call(self) -> bool:
        """
        Check if a call can be made within rate limits
        """
        current_time = time.time()

        # Remove calls that are outside the time window
        while self.call_times and current_time - self.call_times[0] > self.time_window:
            self.call_times.popleft()

        # Check if we're under the limit
        return len(self.call_times) < self.max_calls

    def record_call(self):
        """
        Record that a call has been made
        """
        self.call_times.append(time.time())

    def wait_if_needed(self):
        """
        Wait if rate limit would be exceeded
        """
        if not self.can_make_call():
            sleep_time = self.time_window - (time.time() - self.call_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
```

## Integration with VLA Pipeline

### Planning Pipeline Integration

The LLM integration connects with the broader VLA pipeline:

```python
class PlanningPipelineIntegrator:
    """
    Integrate LLM planning with the broader VLA pipeline
    """
    def __init__(self, llm_interface: LLMInterface):
        self.task_decomposer = TaskDecompositionLLM(llm_interface)
        self.action_sequencer = ActionSequencingLLM(llm_interface)
        self.plan_validator = PlanValidationLLM(llm_interface)
        self.error_handler = LLMErrorHandler()
        self.cache = LLMCache()

    def generate_plan(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete plan through the LLM integration pipeline
        """
        try:
            # Step 1: Decompose task
            decomposition_result = self.task_decomposer.decompose_task(
                task_description,
                context
            )

            if not decomposition_result.get('parsed_successfully', False):
                raise Exception(f"Task decomposition failed: {decomposition_result.get('error')}")

            # Step 2: Sequence actions
            sequencing_result = self.action_sequencer.sequence_actions(
                decomposition_result['data']['subtasks'],
                context
            )

            if not sequencing_result.get('parsed_successfully', False):
                raise Exception(f"Action sequencing failed: {sequencing_result.get('error')}")

            # Step 3: Validate plan
            validation_result = self.plan_validator.validate_plan(
                {
                    'subtasks': decomposition_result['data']['subtasks'],
                    'sequence': sequencing_result['data']['execution_sequence']
                },
                context
            )

            # Combine results
            final_plan = {
                'task_description': task_description,
                'decomposition': decomposition_result,
                'sequencing': sequencing_result,
                'validation': validation_result,
                'overall_confidence': self._calculate_overall_confidence([
                    decomposition_result.get('confidence', 0.5),
                    sequencing_result.get('confidence', 0.5),
                    validation_result.get('data', {}).get('validation_confidence', 0.5)
                ]),
                'is_executable': validation_result.get('data', {}).get('is_valid', False),
                'generated_at': self._get_current_timestamp()
            }

            return final_plan

        except Exception as e:
            # Handle errors using the error handler
            error_result = self.error_handler.handle_llm_error(
                e,
                LLMErrorType.API_ERROR,  # Default error type
                task_description,
                context
            )

            return {
                'task_description': task_description,
                'success': False,
                'error_handling_result': error_result,
                'generated_at': self._get_current_timestamp()
            }

    def _calculate_overall_confidence(self, confidence_scores: List[float]) -> float:
        """
        Calculate overall confidence from multiple scores
        """
        if not confidence_scores:
            return 0.5

        # Use weighted average with validation result having higher weight
        weights = [0.3, 0.3, 0.4]  # decomposition, sequencing, validation
        weighted_sum = sum(score * weight for score, weight in zip(confidence_scores, weights))
        return weighted_sum

    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()
```

## Best Practices

### Prompt Engineering Best Practices

1. **Clear Instructions**: Provide explicit instructions about expected output format
2. **Context Provision**: Include relevant context to guide LLM decisions
3. **Examples**: Provide examples of desired output when possible
4. **Constraint Specification**: Clearly specify constraints and requirements
5. **Iterative Refinement**: Start with simple prompts and add complexity gradually

### Error Handling Best Practices

1. **Graceful Degradation**: Provide fallbacks when LLM fails
2. **Comprehensive Logging**: Log all LLM interactions for debugging
3. **Rate Limiting**: Implement proper rate limiting to avoid API issues
4. **Caching**: Cache common requests to improve performance
5. **Validation**: Always validate LLM outputs before use

### Performance Optimization

1. **Context Management**: Carefully manage context size to stay within limits
2. **Parallel Processing**: Process independent tasks in parallel when possible
3. **Caching**: Implement intelligent caching for common planning tasks
4. **Model Selection**: Choose appropriate models for different planning needs
5. **Monitoring**: Monitor usage and performance metrics

## Security Considerations

### Input Sanitization

The system sanitizes inputs before sending to LLMs:

```python
import re

class InputSanitizer:
    """
    Sanitize inputs before sending to LLM
    """
    def __init__(self):
        self.sensitive_patterns = [
            r'\b(password|secret|key|token|api_key)\b',
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{16}\b',  # Credit card pattern
        ]

    def sanitize_input(self, input_text: str) -> str:
        """
        Sanitize input text
        """
        sanitized = input_text

        # Remove sensitive information
        for pattern in self.sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)

        # Limit length to prevent prompt injection
        if len(sanitized) > 10000:  # 10k character limit
            sanitized = sanitized[:10000]

        return sanitized
```

## Future Enhancements

### Advanced Integration Features

- **Multi-Model Coordination**: Coordinate between different specialized models
- **Learning-Based Adaptation**: Adapt prompting based on execution outcomes
- **Collaborative Planning**: Plan coordination across multiple robots
- **Predictive Planning**: Anticipate future needs and plan accordingly

## Conclusion

LLM integration provides the cognitive intelligence for the VLA system's planning capabilities. By carefully engineering prompts, handling errors gracefully, and optimizing performance, the system can effectively transform natural language commands into executable robot plans. The modular architecture ensures flexibility in LLM selection while maintaining consistent planning interfaces.

For implementation details, refer to the specific cognitive planning components including [Task Decomposition](./task-decomposition.md), [Action Sequencing](./action-sequencing.md), and [Validation](./validation.md).