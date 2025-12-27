"""
Simulation Example Integrator for Docusaurus Documentation

Integrates simulation examples into Docusaurus documentation by
embedding code, images, and interactive elements.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class SimulationExampleIntegrator:
    """Integrates simulation examples into Docusaurus documentation"""

    def __init__(self, docs_dir: str = "site/docs", examples_dir: str = "examples/gazebo"):
        self.docs_dir = Path(docs_dir)
        self.examples_dir = Path(examples_dir)
        self.integration_markers = {
            'code_example': r'<!--\s*CODE_EXAMPLE:\s*(.*?)\s*-->',
            'image_example': r'<!--\s*IMAGE_EXAMPLE:\s*(.*?)\s*-->',
            'simulation_demo': r'<!--\s*SIMULATION_DEMO:\s*(.*?)\s*-->',
            'api_demo': r'<!--\s*API_DEMO:\s*(.*?)\s*-->'
        }

    def find_documentation_files(self) -> List[Path]:
        """Find all documentation files that may contain integration markers"""
        doc_files = []
        for root, dirs, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith(('.md', '.mdx')):
                    doc_files.append(Path(root) / file)
        return doc_files

    def integrate_examples(self) -> Dict[str, List[str]]:
        """Integrate examples into all documentation files"""
        results = {
            'processed': [],
            'errors': [],
            'warnings': []
        }

        doc_files = self.find_documentation_files()

        for doc_file in doc_files:
            try:
                updated_content = self._process_file(doc_file)
                if updated_content:
                    # Write updated content back to file
                    with open(doc_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    results['processed'].append(str(doc_file))
            except Exception as e:
                error_msg = f"{doc_file}: {str(e)}"
                results['errors'].append(error_msg)
                print(f"Error processing {doc_file}: {e}")

        return results

    def _process_file(self, doc_file: Path) -> Optional[str]:
        """Process a single documentation file"""
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Process each type of integration marker
        for marker_type, pattern in self.integration_markers.items():
            content = self._replace_marker_content(content, marker_type, pattern)

        # Return updated content if it changed, otherwise None
        return content if content != original_content else None

    def _replace_marker_content(self, content: str, marker_type: str, pattern: str) -> str:
        """Replace content for a specific marker type"""
        matches = re.findall(pattern, content)

        for match in matches:
            try:
                replacement = self._generate_replacement(marker_type, match.strip())
                if replacement:
                    # Replace the marker with the generated content
                    marker_full = f"<!-- {marker_type.upper()}: {match.strip()} -->"
                    content = content.replace(marker_full, replacement)
            except Exception as e:
                print(f"Error replacing {marker_type} marker '{match}': {e}")

        return content

    def _generate_replacement(self, marker_type: str, identifier: str) -> Optional[str]:
        """Generate replacement content for a marker"""
        if marker_type == 'code_example':
            return self._generate_code_example(identifier)
        elif marker_type == 'image_example':
            return self._generate_image_example(identifier)
        elif marker_type == 'simulation_demo':
            return self._generate_simulation_demo(identifier)
        elif marker_type == 'api_demo':
            return self._generate_api_demo(identifier)
        return None

    def _generate_code_example(self, identifier: str) -> Optional[str]:
        """Generate code example content"""
        # Look for Python files in examples directory
        example_path = self.examples_dir / identifier

        # If identifier is just a filename, look for it in common locations
        if not example_path.exists():
            # Try different extensions
            for ext in ['.py', '.launch.py', '.yaml', '.sdf', '.urdf']:
                test_path = self.examples_dir / f"{identifier}{ext}"
                if test_path.exists():
                    example_path = test_path
                    break

        if not example_path.exists():
            # Try to find in subdirectories
            for root, dirs, files in os.walk(self.examples_dir):
                for file in files:
                    if file.startswith(identifier):
                        example_path = Path(root) / file
                        break

        if example_path.exists():
            try:
                with open(example_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Determine language for syntax highlighting
                ext = example_path.suffix.lower()
                lang_map = {
                    '.py': 'python',
                    '.launch.py': 'python',
                    '.yaml': 'yaml',
                    '.yml': 'yaml',
                    '.sdf': 'xml',
                    '.urdf': 'xml',
                    '.xml': 'xml',
                    '.json': 'json',
                    '.cpp': 'cpp',
                    '.c': 'c',
                    '.js': 'javascript',
                    '.ts': 'typescript'
                }
                language = lang_map.get(ext, 'text')

                return f"```{language}\n{code}\n```"
            except Exception as e:
                print(f"Error reading code example {identifier}: {e}")
                return f"<!-- Error loading code example: {identifier} -->"

        return f"<!-- Code example not found: {identifier} -->"

    def _generate_image_example(self, identifier: str) -> Optional[str]:
        """Generate image example content"""
        # Look for image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']

        # Try to find image in static assets or examples
        for ext in image_extensions:
            image_path = Path("site/static/img") / f"{identifier}{ext}"
            if image_path.exists():
                return f"![{identifier}](@site/static/img/{identifier}{ext})\n\n"

        # Try in examples directory
        for ext in image_extensions:
            image_path = self.examples_dir / f"{identifier}{ext}"
            if image_path.exists():
                # Copy to static assets if not already there
                static_path = Path("site/static/img") / f"{identifier}{ext}"
                static_path.parent.mkdir(parents=True, exist_ok=True)

                if not static_path.exists():
                    import shutil
                    shutil.copy2(image_path, static_path)

                return f"![{identifier}](@site/static/img/{identifier}{ext})\n\n"

        return f"<!-- Image not found: {identifier} -->\n\n"

    def _generate_simulation_demo(self, identifier: str) -> Optional[str]:
        """Generate simulation demo content"""
        # Create a simulation demo with code and explanation
        demo_content = f"""
### Simulation Demo: {identifier}

This simulation demonstrates key concepts related to {identifier.replace('_', ' ').title()}.

```python
# Example simulation code for {identifier}
import rclpy
from rclpy.node import Node

class {identifier.title().replace('_', '')}Demo(Node):
    def __init__(self):
        super().__init__('simulation_demo')
        self.get_logger().info('Starting {identifier} simulation demo')

    def run_demo(self):
        # Simulation logic would go here
        self.get_logger().info('Running {identifier} demo')

def main():
    rclpy.init()
    demo = {identifier.title().replace('_', '')}Demo()
    demo.run_demo()
    rclpy.shutdown()
```

**Key Concepts:**
- How to set up the simulation environment
- Parameter configuration
- Running the simulation
- Validating results

"""
        return demo_content

    def _generate_api_demo(self, identifier: str) -> Optional[str]:
        """Generate API demo content"""
        # Create an API demo with example requests
        demo_content = f"""
### API Demo: {identifier}

The following example shows how to interact with the simulation API for {identifier}.

```bash
# Get simulation profiles
curl -X GET http://localhost:5001/api/profiles
```

```python
import requests

# Python example
response = requests.get('http://localhost:5001/api/profiles')
profiles = response.json()
print(f"Available profiles: {len(profiles['profiles'])}")
```

**API Endpoint:** `/api/{identifier}`

**Method:** `GET`

**Response:**
```json
{{
  "success": true,
  "data": [/* {identifier} data */],
  "count": 0
}}
```

"""
        return demo_content

    def create_example_documentation(self, example_name: str, description: str, code_path: str = None) -> str:
        """Create a complete example documentation page"""
        example_content = f"""# {example_name.title().replace('_', ' ')} Example

{description}

## Overview

This example demonstrates how to use the simulation framework for {example_name.replace('_', ' ')}.

"""

        # Add code example if provided
        if code_path:
            code_content = self._generate_code_example(code_path)
            if code_content and not code_content.startswith("<!--"):
                example_content += f"""
## Code Implementation

{code_content}
"""

        # Add simulation demo
        example_content += f"""
## Simulation Demo

<!-- SIMULATION_DEMO: {example_name} -->

"""

        # Add API demo
        example_content += f"""
## API Integration

<!-- API_DEMO: {example_name} -->

"""

        # Add usage instructions
        example_content += f"""
## Running the Example

To run this example:

1. Make sure Gazebo Garden is installed
2. Source your ROS 2 environment
3. Navigate to the examples directory
4. Run the simulation:

```bash
cd examples/gazebo
# Command to run the {example_name} example would go here
```

## Key Concepts Demonstrated

- Concept 1
- Concept 2
- Concept 3

"""

        return example_content

    def integrate_examples_into_docs(self):
        """Integrate all simulation examples into documentation"""
        print("Integrating simulation examples into documentation...")

        # Process all documentation files
        results = self.integrate_examples()

        print(f"Processed {len(results['processed'])} files")
        if results['errors']:
            print(f"Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")

        if results['warnings']:
            print(f"Warnings: {len(results['warnings'])}")
            for warning in results['warnings']:
                print(f"  - {warning}")

        return results


def setup_example_integrator():
    """Set up the example integrator with common examples"""
    integrator = SimulationExampleIntegrator()

    # Create some example documentation files
    examples = [
        ("basic_physics", "Basic physics simulation example", "basic_physics_example.py"),
        ("sensor_demo", "Sensor simulation demonstration", "sensor_demo.py"),
        ("joint_control", "Joint control example", "joint_control.py")
    ]

    docs_path = Path("site/docs/module2/examples")
    docs_path.mkdir(parents=True, exist_ok=True)

    for example_name, description, code_file in examples:
        content = integrator.create_example_documentation(example_name, description, code_file)
        example_file = docs_path / f"{example_name}.md"

        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Created example documentation: {example_file}")

    return integrator


if __name__ == "__main__":
    # Set up example integrator
    integrator = setup_example_integrator()

    # Integrate examples into documentation
    results = integrator.integrate_examples_into_docs()

    print("\nIntegration complete!")