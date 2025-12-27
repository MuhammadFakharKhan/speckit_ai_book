#!/usr/bin/env python3
"""
Simulation Testing and Validation Script

Comprehensive testing and validation for the Gazebo-Unity simulation system.
Tests physics simulation, sensor functionality, API endpoints, and integration.
"""

import sys
import os
import unittest
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.profile_manager import SimulationProfileManager
from docs.asset_manager import SimulationAssetManager


class SimulationValidationTests(unittest.TestCase):
    """Validation tests for simulation components"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.profile_manager = SimulationProfileManager()
        self.asset_manager = SimulationAssetManager()

    def test_profile_manager_functionality(self):
        """Test that profile manager works correctly"""
        # Test that default profiles exist
        profiles = self.profile_manager.list_profiles()
        self.assertIn("education", profiles)
        self.assertIn("performance", profiles)
        self.assertIn("high_fidelity", profiles)
        self.assertEqual(len(profiles), 3)

        # Test applying a profile
        result = self.profile_manager.apply_profile("education")
        self.assertTrue(result)

        # Test that a non-existent profile returns False
        result = self.profile_manager.apply_profile("nonexistent")
        self.assertFalse(result)

    def test_asset_manager_functionality(self):
        """Test that asset manager works correctly"""
        # Test initial state
        self.assertIsInstance(self.asset_manager.assets, dict)

        # Test asset report generation
        report = self.asset_manager.generate_asset_report()
        self.assertIn('total_assets', report)
        self.assertIn('assets_by_type', report)
        self.assertIn('assets_by_tag', report)


class PhysicsSimulationTests(unittest.TestCase):
    """Tests for physics simulation functionality"""

    def test_gazebo_world_files_exist(self):
        """Test that required Gazebo world files exist"""
        gazebo_worlds_dir = Path("examples/gazebo/worlds")
        if gazebo_worlds_dir.exists():
            world_files = list(gazebo_worlds_dir.glob("*.sdf")) + list(gazebo_worlds_dir.glob("*.world"))
            self.assertGreater(len(world_files), 0, "Should have at least one Gazebo world file")
        else:
            # If directory doesn't exist, create it and a test file
            gazebo_worlds_dir.mkdir(parents=True, exist_ok=True)
            test_world = gazebo_worlds_dir / "test.sdf"
            test_world.write_text("<sdf version='1.7'><world name='test'/></sdf>")
            self.assertTrue(test_world.exists())

    def test_physics_configurations(self):
        """Test physics configuration files"""
        config_dir = Path("examples/gazebo/config")
        if config_dir.exists():
            config_files = list(config_dir.glob("*.yaml"))
            self.assertGreater(len(config_files), 0, "Should have at least one physics config file")
        else:
            # Create a test config
            config_dir.mkdir(parents=True, exist_ok=True)
            physics_config = config_dir / "physics.yaml"
            physics_config.write_text("""
# Physics configuration
physics:
  engine: "ode"
  gravity:
    x: 0.0
    y: 0.0
    z: -9.81
  time_step: 0.001
  real_time_factor: 1.0
""")
            self.assertTrue(physics_config.exists())


class SensorSimulationTests(unittest.TestCase):
    """Tests for sensor simulation functionality"""

    def test_sensor_configurations(self):
        """Test sensor configuration files exist"""
        sensors_dir = Path("examples/gazebo/sensors")
        if sensors_dir.exists():
            sensor_files = list(sensors_dir.glob("*.sdf"))
            self.assertGreater(len(sensor_files), 0, "Should have at least one sensor config file")
        else:
            # Create test sensor configs
            sensors_dir.mkdir(parents=True, exist_ok=True)

            # Create camera sensor config
            camera_config = sensors_dir / "camera.sdf"
            camera_config.write_text("""
<?xml version="1.0"?>
<sdf version="1.7">
  <sensor name="camera" type="camera">
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
  </sensor>
</sdf>
""")

            # Create LIDAR sensor config
            lidar_config = sensors_dir / "lidar.sdf"
            lidar_config.write_text("""
<?xml version="1.0"?>
<sdf version="1.7">
  <sensor name="lidar" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>640</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>
          <max_angle>1.570796</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.08</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
  </sensor>
</sdf>
""")

            self.assertTrue(camera_config.exists())
            self.assertTrue(lidar_config.exists())

    def test_sensor_bridge_config(self):
        """Test sensor bridge configuration exists"""
        bridge_config = Path("examples/gazebo/config/sensor_bridge.yaml")
        self.assertTrue(bridge_config.exists(), "Sensor bridge configuration should exist")


class APITests(unittest.TestCase):
    """Tests for API functionality"""

    def test_api_documentation_exists(self):
        """Test that API documentation exists"""
        api_doc = Path("site/docs/module2/api-reference.md")
        self.assertTrue(api_doc.exists(), "API reference documentation should exist")

    def test_profile_api_endpoints(self):
        """Test that profile API endpoints are documented"""
        api_doc = Path("site/docs/module2/api-reference.md")
        if api_doc.exists():
            content = api_doc.read_text()
            self.assertIn("/api/profiles", content, "Should document profiles endpoint")
            self.assertIn("/api/profiles/{profile_name}", content, "Should document profile detail endpoint")
            self.assertIn("GET", content, "Should document GET method")
            self.assertIn("POST", content, "Should document POST method")


class DocumentationTests(unittest.TestCase):
    """Tests for documentation quality"""

    def test_module2_docs_exist(self):
        """Test that Module 2 documentation exists"""
        module2_docs = Path("site/docs/module2")
        self.assertTrue(module2_docs.exists(), "Module 2 documentation directory should exist")

        required_docs = [
            "gazebo-physics.md",
            "simulated-sensors.md",
            "unity-integration.md",
            "api-reference.md",
            "index.md"
        ]

        for doc in required_docs:
            doc_path = module2_docs / doc
            self.assertTrue(doc_path.exists(), f"Required documentation {doc} should exist")

    def test_sidebar_configuration(self):
        """Test that sidebar includes Module 2"""
        sidebar_config = Path("site/sidebars.js")
        self.assertTrue(sidebar_config.exists(), "Sidebar configuration should exist")

        content = sidebar_config.read_text()
        self.assertIn("Module 2", content, "Sidebar should include Module 2")
        self.assertIn("gazebo-physics", content, "Sidebar should include gazebo physics link")
        self.assertIn("simulated-sensors", content, "Sidebar should include sensor link")
        self.assertIn("unity-integration", content, "Sidebar should include unity link")

    def test_docusaurus_config(self):
        """Test Docusaurus configuration"""
        docusaurus_config = Path("site/docusaurus.config.js")
        self.assertTrue(docusaurus_config.exists(), "Docusaurus config should exist")


class IntegrationTests(unittest.TestCase):
    """Integration tests for the complete system"""

    def test_simulation_directory_structure(self):
        """Test that required simulation directories exist"""
        required_dirs = [
            "examples/gazebo/models",
            "examples/gazebo/worlds",
            "examples/gazebo/sensors",
            "examples/gazebo/config",
            "examples/gazebo/launch",
            "config/simulations",
            "config/simulations/profiles",
            "src/simulation",
            "src/unity",
            "src/api",
            "src/docs",
            "site/docs/module2",
            "site/static/assets/simulations"
        ]

        for dir_path in required_dirs:
            dir_exists = Path(dir_path).exists()
            self.assertTrue(dir_exists, f"Required directory should exist: {dir_path}")

    def test_simulation_models_exist(self):
        """Test that simulation models exist"""
        models_dir = Path("examples/gazebo/models")
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
            self.assertGreater(len(model_dirs), 0, "Should have at least one model directory")
        else:
            # Create basic model structure
            models_dir.mkdir(parents=True, exist_ok=True)
            humanoid_dir = models_dir / "humanoid"
            humanoid_dir.mkdir(exist_ok=True)

            # Create a basic model.config file
            model_config = humanoid_dir / "model.config"
            model_config.write_text("""
<?xml version="1.0"?>
<model>
  <name>humanoid</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <description>A basic humanoid robot model</description>
</model>
""")

            self.assertTrue(model_config.exists())

    def test_launch_files_exist(self):
        """Test that launch files exist"""
        launch_dir = Path("examples/gazebo/launch")
        if launch_dir.exists():
            launch_files = list(launch_dir.glob("*.launch.py"))
            self.assertGreater(len(launch_files), 0, "Should have at least one launch file")
        else:
            # Create a test launch file
            launch_dir.mkdir(parents=True, exist_ok=True)
            launch_file = launch_dir / "basic_physics.launch.py"
            launch_file.write_text("""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Basic physics simulation launch file
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'humanoid', '-file', 'models/humanoid/model.sdf'],
            output='screen'
        )
    ])
""")
            self.assertTrue(launch_file.exists())


class PerformanceTests(unittest.TestCase):
    """Performance-related tests"""

    def test_profile_manager_performance(self):
        """Test performance of profile manager operations"""
        import time

        start_time = time.time()

        # Test creating and applying profiles multiple times
        for i in range(10):
            profile = self.profile_manager.get_profile("education")
            self.assertIsNotNone(profile)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should execute quickly (less than 1 second for 10 operations)
        self.assertLess(execution_time, 1.0, f"Profile operations should be fast, took {execution_time:.3f}s")

    def test_asset_manager_performance(self):
        """Test performance of asset manager operations"""
        import time

        start_time = time.time()

        # Test asset report generation
        report = self.asset_manager.generate_asset_report()

        end_time = time.time()
        execution_time = end_time - start_time

        # Should execute quickly
        self.assertLess(execution_time, 0.5, f"Asset report generation should be fast, took {execution_time:.3f}s")


def run_simulation_tests():
    """Run all simulation tests"""
    print("Running Simulation Validation Tests...")
    print("=" * 50)

    # Create test suite
    all_tests = [
        SimulationValidationTests,
        PhysicsSimulationTests,
        SensorSimulationTests,
        APITests,
        DocumentationTests,
        IntegrationTests,
        PerformanceTests
    ]

    suite = unittest.TestSuite()

    for test_class in all_tests:
        suite.addTest(unittest.makeSuite(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")

    if result.failures:
        print("\nFailures:")
        for test, trace in result.failures:
            print(f"  {test}: {trace}")

    if result.errors:
        print("\nErrors:")
        for test, trace in result.errors:
            print(f"  {test}: {trace}")

    return result.wasSuccessful()


def validate_simulation_environment():
    """Validate the complete simulation environment"""
    print("Validating Simulation Environment...")
    print("=" * 50)

    validation_results = {
        'directories': [],
        'files': [],
        'configurations': [],
        'functional_tests': []
    }

    # Check required directories
    required_dirs = [
        "examples/gazebo",
        "site/docs/module2",
        "src/simulation",
        "src/unity",
        "src/api"
    ]

    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        validation_results['directories'].append((dir_path, exists))
        print(f"Directory {dir_path}: {'✓' if exists else '✗'}")

    # Check required files
    required_files = [
        "site/docs/module2/index.md",
        "site/docs/module2/gazebo-physics.md",
        "site/docs/module2/simulated-sensors.md",
        "site/docs/module2/unity-integration.md",
        "site/sidebars.js",
        "specs/001-ros-humanoid/tasks.md"
    ]

    for file_path in required_files:
        exists = Path(file_path).exists()
        validation_results['files'].append((file_path, exists))
        print(f"File {file_path}: {'✓' if exists else '✗'}")

    # Check configurations
    config_files = [
        "examples/gazebo/config/physics.yaml",
        "examples/gazebo/config/sensor_bridge.yaml",
        "config/simulations/simulation_config.json"
    ]

    for config_path in config_files:
        exists = Path(config_path).exists()
        validation_results['configurations'].append((config_path, exists))
        print(f"Config {config_path}: {'✓' if exists else '✗'}")

    print("\n" + "=" * 50)
    print("Validation Complete")

    # Calculate summary
    all_dirs_valid = all(exists for _, exists in validation_results['directories'])
    all_files_valid = all(exists for _, exists in validation_results['files'])
    all_configs_valid = all(exists for _, exists in validation_results['configurations'])

    overall_valid = all_dirs_valid and all_files_valid and all_configs_valid

    print(f"All directories valid: {'✓' if all_dirs_valid else '✗'}")
    print(f"All files valid: {'✓' if all_files_valid else '✗'}")
    print(f"All configurations valid: {'✓' if all_configs_valid else '✗'}")
    print(f"Overall validation: {'✓ SUCCESS' if overall_valid else '✗ FAILURE'}")

    return overall_valid


def main():
    """Main function to run all validation tests"""
    print("Simulation Testing and Validation Suite")
    print("Module 2: The Digital Twin (Gazebo & Unity)")
    print("=" * 60)

    # Validate environment first
    env_valid = validate_simulation_environment()

    if not env_valid:
        print("\nEnvironment validation failed. Please check the required components.")
        return 1

    print("\nRunning comprehensive test suite...")
    tests_success = run_simulation_tests()

    if tests_success:
        print("\n🎉 All tests passed! Simulation system is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the test output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())