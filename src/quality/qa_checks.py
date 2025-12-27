"""
Simulation Quality Assurance Checks

Comprehensive quality assurance checks for the Gazebo-Unity simulation system.
Validates simulation components, configurations, and integration quality.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import subprocess
import re


@dataclass
class QAResult:
    """Result of a quality assurance check"""
    name: str
    passed: bool
    message: str
    severity: str  # 'critical', 'error', 'warning', 'info'
    details: Optional[Dict] = None


class SimulationQualityAssurance:
    """Quality assurance system for simulation components"""

    def __init__(self):
        self.results: List[QAResult] = []
        self.checks_run = 0

    def run_all_checks(self) -> List[QAResult]:
        """Run all quality assurance checks"""
        print("Running Simulation Quality Assurance Checks...")
        print("=" * 60)

        # Run all check methods
        methods = [
            self.check_directory_structure,
            self.check_file_integrity,
            self.check_configurations,
            self.check_documentation,
            self.check_api_endpoints,
            self.check_simulation_models,
            self.check_launch_files,
            self.check_code_quality,
            self.check_dependencies
        ]

        for method in methods:
            try:
                result = method()
                if isinstance(result, list):
                    self.results.extend(result)
                elif isinstance(result, QAResult):
                    self.results.append(result)
            except Exception as e:
                error_result = QAResult(
                    name=method.__name__,
                    passed=False,
                    message=f"Check failed with exception: {str(e)}",
                    severity="error"
                )
                self.results.append(error_result)

        return self.results

    def check_directory_structure(self) -> List[QAResult]:
        """Check that required directories exist"""
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

        results = []
        for dir_path in required_dirs:
            path = Path(dir_path)
            exists = path.exists()
            results.append(QAResult(
                name=f"Directory exists: {dir_path}",
                passed=exists,
                message=f"Directory {dir_path} {'exists' if exists else 'missing'}",
                severity="critical" if not exists else "info"
            ))

        return results

    def check_file_integrity(self) -> List[QAResult]:
        """Check integrity of critical files"""
        critical_files = [
            "specs/001-ros-humanoid/tasks.md",
            "site/docs/module2/index.md",
            "site/docs/module2/gazebo-physics.md",
            "site/docs/module2/simulated-sensors.md",
            "site/docs/module2/unity-integration.md",
            "site/docs/module2/api-reference.md",
            "src/simulation/profile_manager.py",
            "src/unity/ros_bridge.py",
            "src/unity/state_synchronizer.py",
            "src/unity/ui_elements.py",
            "src/api/profile_api.py",
            "src/api/middleware.py",
            "scripts/test_simulation.py"
        ]

        results = []
        for file_path in critical_files:
            path = Path(file_path)
            exists = path.exists()
            size_ok = path.stat().st_size > 0 if exists else False

            results.append(QAResult(
                name=f"File integrity: {file_path}",
                passed=exists and size_ok,
                message=f"File {file_path} {'OK' if exists and size_ok else 'missing or empty'}",
                severity="critical" if not exists or not size_ok else "info"
            ))

        return results

    def check_configurations(self) -> List[QAResult]:
        """Check configuration files for validity"""
        config_files = [
            "examples/gazebo/config/physics.yaml",
            "examples/gazebo/config/sensor_bridge.yaml",
            "config/simulations/simulation_config.json"
        ]

        results = []
        for config_path in config_files:
            path = Path(config_path)
            if path.exists():
                try:
                    if path.suffix.lower() in ['.yaml', '.yml']:
                        with open(path, 'r') as f:
                            yaml.safe_load(f)
                        results.append(QAResult(
                            name=f"YAML config valid: {config_path}",
                            passed=True,
                            message=f"YAML configuration {config_path} is valid",
                            severity="info"
                        ))
                    elif path.suffix.lower() == '.json':
                        with open(path, 'r') as f:
                            json.load(f)
                        results.append(QAResult(
                            name=f"JSON config valid: {config_path}",
                            passed=True,
                            message=f"JSON configuration {config_path} is valid",
                            severity="info"
                        ))
                except Exception as e:
                    results.append(QAResult(
                        name=f"Config validation failed: {config_path}",
                        passed=False,
                        message=f"Configuration {config_path} validation failed: {str(e)}",
                        severity="error"
                    ))
            else:
                results.append(QAResult(
                    name=f"Config file missing: {config_path}",
                    passed=False,
                    message=f"Configuration file {config_path} is missing",
                    severity="critical"
                ))

        return results

    def check_documentation(self) -> List[QAResult]:
        """Check documentation quality and completeness"""
        docs_dir = Path("site/docs/module2")
        if not docs_dir.exists():
            return [QAResult(
                name="Documentation directory",
                passed=False,
                message="Module 2 documentation directory missing",
                severity="critical"
            )]

        required_docs = [
            "index.md",
            "gazebo-physics.md",
            "simulated-sensors.md",
            "unity-integration.md",
            "api-reference.md",
            "troubleshooting.md",
            "performance-optimization.md"
        ]

        results = []
        for doc in required_docs:
            doc_path = docs_dir / doc
            exists = doc_path.exists()
            results.append(QAResult(
                name=f"Documentation exists: {doc}",
                passed=exists,
                message=f"Documentation {doc} {'exists' if exists else 'missing'}",
                severity="critical" if not exists else "info"
            ))

        # Check sidebar integration
        sidebar_path = Path("site/sidebars.js")
        if sidebar_path.exists():
            content = sidebar_path.read_text()
            has_module2 = "Module 2" in content
            results.append(QAResult(
                name="Sidebar integration",
                passed=has_module2,
                message=f"Module 2 {'integrated' if has_module2 else 'not integrated'} in sidebar",
                severity="critical" if not has_module2 else "info"
            ))
        else:
            results.append(QAResult(
                name="Sidebar configuration",
                passed=False,
                message="Sidebar configuration file missing",
                severity="critical"
            ))

        return results

    def check_api_endpoints(self) -> List[QAResult]:
        """Check API implementation"""
        api_files = [
            "src/api/profile_api.py",
            "src/api/middleware.py"
        ]

        results = []
        for api_file in api_files:
            path = Path(api_file)
            if path.exists():
                content = path.read_text()
                has_flask = "from flask import Flask" in content
                has_routes = "app.route" in content
                results.append(QAResult(
                    name=f"API implementation: {api_file}",
                    passed=has_flask and has_routes,
                    message=f"API file {api_file} {'has' if has_flask and has_routes else 'missing'} Flask implementation",
                    severity="critical" if not (has_flask and has_routes) else "info"
                ))
            else:
                results.append(QAResult(
                    name=f"API file missing: {api_file}",
                    passed=False,
                    message=f"API file {api_file} is missing",
                    severity="critical"
                ))

        return results

    def check_simulation_models(self) -> List[QAResult]:
        """Check simulation model files"""
        models_dir = Path("examples/gazebo/models")
        if not models_dir.exists():
            return [QAResult(
                name="Models directory",
                passed=False,
                message="Simulation models directory missing",
                severity="critical"
            )]

        results = []
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        results.append(QAResult(
            name="Simulation models",
            passed=len(model_dirs) > 0,
            message=f"Found {len(model_dirs)} model directories",
            severity="critical" if len(model_dirs) == 0 else "info"
        ))

        # Check for model.config files
        for model_dir in model_dirs:
            config_file = model_dir / "model.config"
            exists = config_file.exists()
            results.append(QAResult(
                name=f"Model config exists: {model_dir.name}",
                passed=exists,
                message=f"Model config for {model_dir.name} {'exists' if exists else 'missing'}",
                severity="warning" if not exists else "info"
            ))

        return results

    def check_launch_files(self) -> List[QAResult]:
        """Check launch files"""
        launch_dir = Path("examples/gazebo/launch")
        if not launch_dir.exists():
            return [QAResult(
                name="Launch directory",
                passed=False,
                message="Launch files directory missing",
                severity="critical"
            )]

        launch_files = list(launch_dir.glob("*.launch.py"))
        results = [QAResult(
            name="Launch files",
            passed=len(launch_files) > 0,
            message=f"Found {len(launch_files)} launch files",
            severity="critical" if len(launch_files) == 0 else "info"
        )]

        # Check launch file content
        for launch_file in launch_files:
            content = launch_file.read_text()
            has_launch_description = "LaunchDescription" in content
            has_generate_launch_description = "generate_launch_description" in content
            results.append(QAResult(
                name=f"Launch file structure: {launch_file.name}",
                passed=has_launch_description and has_generate_launch_description,
                message=f"Launch file {launch_file.name} {'has' if has_launch_description and has_generate_launch_description else 'missing'} proper structure",
                severity="error" if not (has_launch_description and has_generate_launch_description) else "info"
            ))

        return results

    def check_code_quality(self) -> List[QAResult]:
        """Check code quality and standards"""
        python_files = list(Path("src").rglob("*.py")) + list(Path("scripts").rglob("*.py")) + list(Path("tests").rglob("*.py"))

        results = []
        for py_file in python_files:
            try:
                content = py_file.read_text()
                has_docstring = '"""' in content or "'''" in content
                has_imports = "import" in content or "from" in content
                results.append(QAResult(
                    name=f"Code quality: {py_file}",
                    passed=has_docstring and has_imports,
                    message=f"Python file {py_file} {'meets' if has_docstring and has_imports else 'does not meet'} basic quality standards",
                    severity="warning" if not (has_docstring and has_imports) else "info"
                ))
            except Exception as e:
                results.append(QAResult(
                    name=f"Code check failed: {py_file}",
                    passed=False,
                    message=f"Could not check code quality for {py_file}: {str(e)}",
                    severity="error"
                ))

        return results

    def check_dependencies(self) -> List[QAResult]:
        """Check for dependency files"""
        dependency_files = [
            "package.json",
            "requirements.txt",  # We'll check if this should exist
            "pyproject.toml"
        ]

        results = []
        for dep_file in dependency_files:
            path = Path(dep_file)
            exists = path.exists()
            results.append(QAResult(
                name=f"Dependency file: {dep_file}",
                passed=exists,
                message=f"Dependency file {dep_file} {'exists' if exists else 'missing'}",
                severity="warning" if not exists else "info"
            ))

        return results

    def generate_report(self) -> str:
        """Generate a QA report"""
        if not self.results:
            return "No quality assurance checks have been run."

        report = []
        report.append("SIMULATION QUALITY ASSURANCE REPORT")
        report.append("=" * 60)
        report.append(f"Total checks run: {len(self.results)}")
        report.append("")

        # Count results by severity
        critical_count = sum(1 for r in self.results if r.severity == "critical" and not r.passed)
        error_count = sum(1 for r in self.results if r.severity == "error" and not r.passed)
        warning_count = sum(1 for r in self.results if r.severity == "warning" and not r.passed)
        passed_count = sum(1 for r in self.results if r.passed)

        report.append(f"Critical failures: {critical_count}")
        report.append(f"Errors: {error_count}")
        report.append(f"Warnings: {warning_count}")
        report.append(f"Passed: {passed_count}")
        report.append("")

        # Group results by severity
        for severity in ["critical", "error", "warning", "info"]:
            severity_results = [r for r in self.results if r.severity == severity]
            if severity_results:
                report.append(f"{severity.upper()} RESULTS:")
                for result in severity_results:
                    status = "✓" if result.passed else "✗"
                    report.append(f"  {status} {result.name}: {result.message}")
                report.append("")

        return "\n".join(report)

    def get_summary(self) -> Dict[str, int]:
        """Get summary of QA results"""
        summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "critical": sum(1 for r in self.results if r.severity == "critical" and not r.passed),
            "errors": sum(1 for r in self.results if r.severity == "error" and not r.passed),
            "warnings": sum(1 for r in self.results if r.severity == "warning" and not r.passed)
        }
        return summary


def run_quality_assurance():
    """Run the quality assurance checks"""
    qa = SimulationQualityAssurance()
    results = qa.run_all_checks()

    # Print report
    print(qa.generate_report())

    # Print summary
    summary = qa.get_summary()
    print("SUMMARY:")
    print(f"  Total checks: {summary['total']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Critical issues: {summary['critical']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")

    # Return success status (no critical failures or errors)
    return summary['critical'] == 0 and summary['errors'] == 0


if __name__ == "__main__":
    success = run_quality_assurance()
    if success:
        print("\n✓ Quality assurance checks passed!")
        sys.exit(0)
    else:
        print("\n✗ Quality assurance checks revealed issues!")
        sys.exit(1)