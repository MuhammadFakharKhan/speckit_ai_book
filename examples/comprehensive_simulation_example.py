"""
Comprehensive Simulation Example: Humanoid Robot Digital Twin

This example demonstrates a complete digital twin simulation system
integrating Gazebo physics, Unity visualization, ROS 2 communication,
and API control for a humanoid robot.
"""

import sys
import os
import time
import threading
from typing import Dict, List, Optional
import json
import requests

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.profile_manager import SimulationProfileManager
from unity.ros_bridge import UnityROSBridge, UnityIntegrationManager
from unity.state_synchronizer import GazeboUnityStateSynchronizer, RobotState, JointState, Transform
from unity.ui_elements import InteractionUIManager
from api.profile_api import create_profile_api


class ComprehensiveSimulationExample:
    """
    Comprehensive example combining all simulation concepts
    """

    def __init__(self):
        # Initialize components
        self.profile_manager = SimulationProfileManager()
        self.ros_bridge = UnityROSBridge()
        self.unity_manager = UnityIntegrationManager()
        self.state_synchronizer = GazeboUnityStateSynchronizer()
        self.ui_manager = InteractionUIManager()

        # Set up ROS integration
        self.unity_manager.ros_bridge = self.ros_bridge

        # Robot joint names
        self.joint_names = [
            "hip_joint", "knee_joint", "ankle_joint",
            "shoulder_joint", "elbow_joint", "wrist_joint"
        ]

        # Set up UI
        self.ui_manager.setup_default_interaction_ui(self.joint_names)

        # API server
        self.api_server = None

    def setup_simulation_profile(self):
        """Set up the simulation profile"""
        print("Setting up simulation profile...")

        # Apply education profile for balanced performance
        success = self.profile_manager.apply_profile("education")
        if success:
            print("✓ Simulation profile applied successfully")
        else:
            print("✗ Failed to apply simulation profile")

        return success

    def setup_ros_bridge(self):
        """Set up ROS bridge connection"""
        print("Setting up ROS bridge...")

        try:
            # Connect to ROS network
            self.unity_manager.connect_to_ros()
            print("✓ ROS bridge connected successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to connect ROS bridge: {e}")
            return False

    def setup_state_synchronization(self):
        """Set up state synchronization between Gazebo and Unity"""
        print("Setting up state synchronization...")

        try:
            # Start synchronization
            self.state_synchronizer.start_synchronization()
            print("✓ State synchronization started")
            return True
        except Exception as e:
            print(f"✗ Failed to start state synchronization: {e}")
            return False

    def create_sample_robot_state(self) -> RobotState:
        """Create a sample robot state for demonstration"""
        import random

        # Create joint states
        joint_states = {}
        for joint_name in self.joint_names:
            joint_states[joint_name] = JointState(
                name=joint_name,
                position=random.uniform(-1.0, 1.0),
                velocity=random.uniform(-0.1, 0.1),
                effort=random.uniform(-10.0, 10.0),
                timestamp=time.time()
            )

        # Create transforms
        transforms = {
            'base_link': Transform(
                position=(0.0, 0.0, 0.5),
                rotation=(0.0, 0.0, 0.0, 1.0),
                timestamp=time.time()
            )
        }

        # Create velocities
        velocities = {
            'base_link': (0.1, 0.0, 0.0)
        }

        return RobotState(
            joint_states=joint_states,
            transforms=transforms,
            velocities=velocities,
            timestamp=time.time(),
            simulation_time=time.time()
        )

    def run_simulation_loop(self, duration: int = 30):
        """Run the main simulation loop"""
        print(f"Starting simulation loop for {duration} seconds...")

        start_time = time.time()
        iteration = 0

        while time.time() - start_time < duration:
            iteration += 1

            # Create sample robot state
            sample_state = self.create_sample_robot_state()

            # Update gazebo state (simulating Gazebo providing state data)
            self.state_synchronizer.update_gazebo_state(sample_state)

            # Get synchronized Unity state
            unity_state = self.state_synchronizer.get_unity_state()
            if unity_state:
                print(f"Iteration {iteration}: Unity state updated with {len(unity_state.joint_states)} joints")

            # Update UI with current joint positions
            for joint_name in self.joint_names[:3]:  # Update first 3 joints as example
                if joint_name in sample_state.joint_states:
                    pos = sample_state.joint_states[joint_name].position
                    self.ui_manager.update_joint_slider(joint_name, pos)

            # Small delay to simulate real-time operation
            time.sleep(0.1)

        print(f"✓ Simulation loop completed after {iteration} iterations")

    def demonstrate_api_control(self):
        """Demonstrate API-based control"""
        print("Demonstrating API control...")

        # Start API server in a separate thread
        import threading

        def start_api_server():
            app = create_profile_api()
            app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()

        time.sleep(2)  # Wait for server to start

        try:
            # Test API endpoints
            response = requests.get('http://localhost:5001/api/profiles')
            if response.status_code == 200:
                profiles = response.json()
                print(f"✓ Retrieved {profiles.get('count', 0)} profiles via API")
            else:
                print(f"✗ API request failed with status {response.status_code}")

            # Test applying profile via API
            response = requests.post('http://localhost:5001/api/profiles/education/apply')
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("✓ Profile applied successfully via API")
                else:
                    print(f"✗ API profile application failed: {result.get('error')}")
            else:
                print(f"✗ API profile application failed with status {response.status_code}")

        except Exception as e:
            print(f"✗ API demonstration failed: {e}")

    def demonstrate_comprehensive_integration(self):
        """Demonstrate comprehensive integration of all components"""
        print("\n" + "="*60)
        print("COMPREHENSIVE SIMULATION INTEGRATION DEMONSTRATION")
        print("="*60)

        # Step 1: Setup simulation profile
        profile_ok = self.setup_simulation_profile()

        # Step 2: Setup ROS bridge
        ros_ok = self.setup_ros_bridge()

        # Step 3: Setup state synchronization
        sync_ok = self.setup_state_synchronization()

        # Step 4: Demonstrate API control
        self.demonstrate_api_control()

        # Step 5: Run simulation loop
        if profile_ok and ros_ok and sync_ok:
            self.run_simulation_loop(duration=10)  # Shorter demo run
        else:
            print("✗ Skipping simulation loop due to setup failures")

        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")

        # Stop state synchronization
        self.state_synchronizer.stop_synchronization()

        # Destroy ROS node
        if hasattr(self.ros_bridge, 'destroy_node'):
            self.ros_bridge.destroy_node()


def create_comprehensive_example_documentation():
    """Create documentation for the comprehensive example"""
    doc_content = """
# Comprehensive Simulation Example

This example demonstrates the complete integration of all simulation components:

## Components Demonstrated

1. **Simulation Profile Management**: Configuring physics and visualization parameters
2. **ROS Bridge**: Connecting Gazebo simulation to Unity visualization
3. **State Synchronization**: Real-time synchronization between systems
4. **API Control**: Programmatic control via REST API
5. **UI Integration**: Interactive controls for robot manipulation

## Architecture

The comprehensive example follows this architecture:

```
[User Input] → [UI Controls] → [API Server] → [Profile Manager]
                                        ↓
[Unity Visual] ← [State Sync] ← [Gazebo Sim] ← [ROS Bridge]
                                        ↓
                                [Robot Control]
```

## Key Features

- Real-time state synchronization between Gazebo and Unity
- API-based profile management and control
- Interactive UI for robot manipulation
- Comprehensive error handling and validation
- Performance-optimized communication

## Running the Example

```bash
python examples/comprehensive_simulation_example.py
```

## Expected Output

- API server running on port 5001
- Simulation profile applied
- State synchronization established
- Sample robot states generated and synchronized
- API endpoints tested and validated
"""

    with open("examples/comprehensive_simulation_example.md", "w") as f:
        f.write(doc_content)

    print("✓ Comprehensive example documentation created")


def main():
    """Main function to run the comprehensive simulation example"""
    print("Starting Comprehensive Simulation Example")
    print("This example demonstrates integration of all simulation components:")
    print("- Gazebo physics simulation")
    print("- Unity visualization")
    print("- ROS 2 communication")
    print("- API control")
    print("- UI interaction")
    print()

    # Create documentation
    create_comprehensive_example_documentation()

    # Create and run example
    example = ComprehensiveSimulationExample()

    try:
        example.demonstrate_comprehensive_integration()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError running simulation: {e}")
    finally:
        example.cleanup()
        print("✓ Cleanup completed")


if __name__ == "__main__":
    main()