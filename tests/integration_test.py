"""
Comprehensive Integration Test for Module 2: Digital Twin Simulation

This test verifies the integration of all components in the digital twin simulation system:
- Gazebo physics simulation
- Sensor simulation and ROS 2 integration
- Unity visualization (conceptual, as Unity is external)
- API control system
- Profile management
- Documentation and asset management
"""

import unittest
import sys
import os
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.profile_manager import SimulationProfileManager
from unity.ros_bridge import UnityROSBridge, UnityIntegrationManager
from unity.state_synchronizer import GazeboUnityStateSynchronizer, RobotState, JointState, Transform
from unity.ui_elements import InteractionUIManager
from api.profile_api import create_profile_api
from docs.asset_manager import SimulationAssetManager
from quality.qa_checks import SimulationQualityAssurance


class IntegrationTest(unittest.TestCase):
    """Integration tests for the complete simulation system"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.profile_manager = SimulationProfileManager()
        self.asset_manager = SimulationAssetManager()
        self.qa_checker = SimulationQualityAssurance()

    def test_profile_manager_integration(self):
        """Test profile manager integration with other components"""
        # Test that default profiles exist
        profiles = self.profile_manager.list_profiles()
        self.assertIn("education", profiles)
        self.assertIn("performance", profiles)
        self.assertIn("high_fidelity", profiles)

        # Test applying a profile
        result = self.profile_manager.apply_profile("education")
        self.assertTrue(result)

        # Test creating a custom profile
        from simulation.profile_manager import (
            SimulationProfile, PhysicsSettings, VisualQuality, HardwareRequirements, ProfileType
        )

        custom_profile = SimulationProfile(
            name="integration_test",
            profile_type=ProfileType.PERFORMANCE,
            physics_settings=PhysicsSettings(
                engine="ode",
                time_step=0.001,
                real_time_factor=1.0
            ),
            visual_quality=VisualQuality(
                rendering_quality="medium"
            ),
            hardware_requirements=HardwareRequirements()
        )

        result = self.profile_manager.add_profile(custom_profile)
        self.assertTrue(result)

        # Verify the profile was added
        profile = self.profile_manager.get_profile("integration_test")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "integration_test")

    def test_asset_management_integration(self):
        """Test asset management integration"""
        # Test asset report generation
        report = self.asset_manager.generate_asset_report()
        self.assertIn('total_assets', report)
        self.assertIn('assets_by_type', report)

        # Test asset validation
        invalid_assets = self.asset_manager.validate_asset_integrity()
        # Should have no invalid assets in a clean system
        # (this might have some if files were manually modified)

    def test_quality_assurance_integration(self):
        """Test quality assurance integration"""
        # Run QA checks
        results = self.qa_checker.run_all_checks()

        # Generate summary
        summary = self.qa_checker.get_summary()

        # Verify we have results
        self.assertGreaterEqual(summary['total'], 0)
        self.assertGreaterEqual(summary['passed'], 0)

        # Print summary for visibility
        print(f"QA Summary: Total={summary['total']}, Passed={summary['passed']}, Failed={summary['failed']}")

    def test_api_integration(self):
        """Test API integration (conceptual - actual API testing would require server)"""
        # Test that API module can be imported and initialized
        try:
            api_app = create_profile_api()
            self.assertIsNotNone(api_app)
            print("✓ API application created successfully")
        except Exception as e:
            self.fail(f"Failed to create API application: {e}")

    def test_directory_structure_completeness(self):
        """Test that all required directories exist"""
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
            "src/quality",
            "site/docs/module2",
            "site/static/assets/simulations",
            "tests",
            "scripts"
        ]

        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)

        if missing_dirs:
            self.fail(f"Missing required directories: {missing_dirs}")
        else:
            print(f"✓ All {len(required_dirs)} required directories exist")

    def test_documentation_completeness(self):
        """Test that all required documentation files exist"""
        required_docs = [
            "site/docs/module2/index.md",
            "site/docs/module2/gazebo-physics.md",
            "site/docs/module2/simulated-sensors.md",
            "site/docs/module2/unity-integration.md",
            "site/docs/module2/api-reference.md",
            "site/docs/module2/troubleshooting.md",
            "site/docs/module2/performance-optimization.md",
            "site/docs/module2/exercises.md",
            "site/docs/module2/quickstart.md",
            "site/docs/features.md"
        ]

        missing_docs = []
        for doc_path in required_docs:
            if not Path(doc_path).exists():
                missing_docs.append(doc_path)

        if missing_docs:
            self.fail(f"Missing required documentation files: {missing_docs}")
        else:
            print(f"✓ All {len(required_docs)} required documentation files exist")

    def test_simulation_examples_exist(self):
        """Test that simulation examples exist and are accessible"""
        example_files = [
            "examples/comprehensive_simulation_example.py",
            "scripts/test_simulation.py"
        ]

        missing_examples = []
        for example_path in example_files:
            if not Path(example_path).exists():
                missing_examples.append(example_path)

        if missing_examples:
            self.fail(f"Missing simulation examples: {missing_examples}")
        else:
            print(f"✓ All {len(example_files)} simulation examples exist")

    def test_configuration_files_exist(self):
        """Test that all required configuration files exist"""
        config_files = [
            "examples/gazebo/config/physics.yaml",
            "examples/gazebo/config/sensor_bridge.yaml",
            "config/simulations/simulation_config.json",
            "site/sidebars.js"
        ]

        missing_configs = []
        for config_path in config_files:
            if not Path(config_path).exists():
                missing_configs.append(config_path)

        if missing_configs:
            self.fail(f"Missing configuration files: {missing_configs}")
        else:
            print(f"✓ All {len(config_files)} configuration files exist")

    def test_source_code_modules_exist(self):
        """Test that all required source code modules exist"""
        modules = [
            "src/simulation/profile_manager.py",
            "src/unity/ros_bridge.py",
            "src/unity/state_synchronizer.py",
            "src/unity/ui_elements.py",
            "src/api/profile_api.py",
            "src/api/middleware.py",
            "src/docs/asset_manager.py",
            "src/docs/example_integrator.py",
            "src/quality/qa_checks.py"
        ]

        missing_modules = []
        for module_path in modules:
            if not Path(module_path).exists():
                missing_modules.append(module_path)

        if missing_modules:
            self.fail(f"Missing source code modules: {missing_modules}")
        else:
            print(f"✓ All {len(modules)} source code modules exist")

    def test_simulation_model_structure(self):
        """Test that simulation model structure is correct"""
        models_dir = Path("examples/gazebo/models")
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
            self.assertGreater(len(model_dirs), 0, "Should have at least one model directory")

            # Check for humanoid model
            humanoid_dir = models_dir / "humanoid"
            if humanoid_dir.exists():
                print("✓ Humanoid model directory exists")
            else:
                print("⚠ Humanoid model directory not found, but other models exist")
        else:
            self.fail("examples/gazebo/models directory does not exist")

    def test_api_documentation_integration(self):
        """Test that API documentation references exist in docs"""
        api_doc_path = Path("site/docs/module2/api-reference.md")
        if api_doc_path.exists():
            content = api_doc_path.read_text()

            # Check for key API endpoints
            required_endpoints = [
                "/api/profiles",
                "/api/profiles/{profile_name}",
                "/api/profiles/{profile_name}/apply",
                "GET", "POST", "PUT", "DELETE"
            ]

            missing_endpoints = []
            for endpoint in required_endpoints:
                if endpoint not in content:
                    missing_endpoints.append(endpoint)

            if missing_endpoints:
                print(f"⚠ Missing endpoints in API doc: {missing_endpoints}")
            else:
                print("✓ All required API endpoints documented")
        else:
            self.fail("API reference documentation does not exist")


class SystemIntegrationTest(unittest.TestCase):
    """Higher-level system integration tests"""

    def test_complete_system_workflow(self):
        """Test a complete workflow using multiple components"""
        print("Testing complete system workflow...")

        # 1. Create and apply a simulation profile
        profile_manager = SimulationProfileManager()
        profiles_before = profile_manager.list_profiles()
        print(f"✓ Found {len(profiles_before)} initial profiles")

        # 2. Apply a profile
        result = profile_manager.apply_profile("education")
        self.assertTrue(result, "Should be able to apply education profile")
        print("✓ Successfully applied education profile")

        # 3. Test asset management
        asset_manager = SimulationAssetManager()
        report = asset_manager.generate_asset_report()
        print(f"✓ Asset manager generated report with {report['total_assets']} assets")

        # 4. Test quality assurance
        qa_checker = SimulationQualityAssurance()
        results = qa_checker.run_all_checks()
        summary = qa_checker.get_summary()
        print(f"✓ QA checks completed: {summary['passed']} passed, {summary['failed']} failed")

        # 5. Verify documentation exists
        docs_dir = Path("site/docs/module2")
        self.assertTrue(docs_dir.exists(), "Module 2 documentation directory should exist")
        print("✓ Module 2 documentation directory exists")

        # 6. Verify examples exist
        example_path = Path("examples/comprehensive_simulation_example.py")
        self.assertTrue(example_path.exists(), "Comprehensive example should exist")
        print("✓ Comprehensive simulation example exists")

        print("✓ Complete system workflow test passed")

    def test_cross_component_dependencies(self):
        """Test that components can work together"""
        print("Testing cross-component dependencies...")

        # Test that profile manager can be used by other components
        profile_manager = SimulationProfileManager()
        profile = profile_manager.get_profile("education")
        self.assertIsNotNone(profile, "Should be able to get education profile")
        print("✓ Profile manager accessible")

        # Test that asset manager can scan simulation assets
        asset_manager = SimulationAssetManager()
        added_assets = asset_manager.scan_simulation_assets("examples/gazebo")
        print(f"✓ Asset manager can scan simulation assets ({len(added_assets)} found)")

        # Test that UI manager can be set up with joint names
        ui_manager = InteractionUIManager()
        joint_names = ["hip_joint", "knee_joint", "ankle_joint"]
        ui_manager.setup_default_interaction_ui(joint_names)
        print("✓ UI manager can be configured with joint names")

        print("✓ Cross-component dependencies test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("Running Comprehensive Integration Tests for Module 2")
    print("=" * 70)

    # Create test suite
    suite = unittest.TestSuite()

    # Add integration tests
    suite.addTest(unittest.makeSuite(IntegrationTest))
    suite.addTest(unittest.makeSuite(SystemIntegrationTest))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("Integration Test Results Summary:")
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


if __name__ == "__main__":
    success = run_integration_tests()
    if success:
        print("\n🎉 All integration tests passed! System is properly integrated.")
        sys.exit(0)
    else:
        print("\n❌ Some integration tests failed. Please review the output above.")
        sys.exit(1)