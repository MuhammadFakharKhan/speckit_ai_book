"""
Simulation Profile Management System

Manages different simulation profiles (high_fidelity, performance, education)
with configurable physics settings, visual quality, and hardware requirements.
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ProfileType(Enum):
    HIGH_FIDELITY = "high_fidelity"
    PERFORMANCE = "performance"
    EDUCATION = "education"


@dataclass
class PhysicsSettings:
    engine: str  # ode, bullet, dart
    gravity_x: float = 0.0
    gravity_y: float = 0.0
    gravity_z: float = -9.81
    time_step: float = 0.001
    real_time_factor: float = 1.0


@dataclass
class VisualQuality:
    rendering_quality: str  # low, medium, high, ultra
    shadows: bool = True
    reflections: bool = False
    anti_aliasing: str = "fxaa"  # fxaa, taa, msaa


@dataclass
class HardwareRequirements:
    min_cpu_cores: int = 4
    min_ram_gb: int = 8
    min_gpu_vram_gb: int = 2
    recommended_cpu_cores: int = 8
    recommended_ram_gb: int = 16
    recommended_gpu_vram_gb: int = 4


@dataclass
class SimulationProfile:
    name: str
    profile_type: ProfileType
    physics_settings: PhysicsSettings
    visual_quality: VisualQuality
    hardware_requirements: HardwareRequirements
    description: str = ""
    use_case: str = ""


class SimulationProfileManager:
    def __init__(self, profiles_dir: str = "config/simulations/profiles"):
        self.profiles_dir = profiles_dir
        self.profiles: Dict[str, SimulationProfile] = {}

        # Create profiles directory if it doesn't exist
        os.makedirs(profiles_dir, exist_ok=True)

        # Load default profiles
        self._load_default_profiles()

    def _load_default_profiles(self):
        """Load default simulation profiles"""
        # High fidelity profile
        high_fidelity_profile = SimulationProfile(
            name="high_fidelity",
            profile_type=ProfileType.HIGH_FIDELITY,
            physics_settings=PhysicsSettings(
                engine="bullet",
                time_step=0.0005,
                real_time_factor=0.5
            ),
            visual_quality=VisualQuality(
                rendering_quality="high",
                shadows=True,
                reflections=True,
                anti_aliasing="taa"
            ),
            hardware_requirements=HardwareRequirements(
                min_cpu_cores=8,
                min_ram_gb=16,
                min_gpu_vram_gb=4,
                recommended_cpu_cores=16,
                recommended_ram_gb=32,
                recommended_gpu_vram_gb=8
            ),
            description="High-fidelity simulation for research and validation",
            use_case="Use for detailed physics simulation and sensor validation"
        )

        # Performance profile
        performance_profile = SimulationProfile(
            name="performance",
            profile_type=ProfileType.PERFORMANCE,
            physics_settings=PhysicsSettings(
                engine="ode",
                time_step=0.002,
                real_time_factor=2.0
            ),
            visual_quality=VisualQuality(
                rendering_quality="medium",
                shadows=False,
                reflections=False,
                anti_aliasing="fxaa"
            ),
            hardware_requirements=HardwareRequirements(
                min_cpu_cores=4,
                min_ram_gb=8,
                min_gpu_vram_gb=2,
                recommended_cpu_cores=8,
                recommended_ram_gb=16,
                recommended_gpu_vram_gb=4
            ),
            description="Performance-optimized simulation for rapid testing",
            use_case="Use for fast iteration and testing on standard hardware"
        )

        # Education profile
        education_profile = SimulationProfile(
            name="education",
            profile_type=ProfileType.EDUCATION,
            physics_settings=PhysicsSettings(
                engine="ode",
                time_step=0.001,
                real_time_factor=1.0
            ),
            visual_quality=VisualQuality(
                rendering_quality="medium",
                shadows=True,
                reflections=False,
                anti_aliasing="fxaa"
            ),
            hardware_requirements=HardwareRequirements(
                min_cpu_cores=4,
                min_ram_gb=8,
                min_gpu_vram_gb=1,
                recommended_cpu_cores=6,
                recommended_ram_gb=12,
                recommended_gpu_vram_gb=2
            ),
            description="Balanced simulation for educational purposes",
            use_case="Use for learning and educational demonstrations"
        )

        self.add_profile(high_fidelity_profile)
        self.add_profile(performance_profile)
        self.add_profile(education_profile)

    def add_profile(self, profile: SimulationProfile) -> bool:
        """Add a new simulation profile"""
        try:
            self.profiles[profile.name] = profile
            self._save_profile(profile)
            return True
        except Exception as e:
            print(f"Error adding profile {profile.name}: {e}")
            return False

    def get_profile(self, name: str) -> Optional[SimulationProfile]:
        """Get a simulation profile by name"""
        return self.profiles.get(name)

    def list_profiles(self) -> List[str]:
        """List all available profile names"""
        return list(self.profiles.keys())

    def _save_profile(self, profile: SimulationProfile):
        """Save a profile to JSON file"""
        profile_file = os.path.join(self.profiles_dir, f"{profile.name}.json")
        profile_dict = {
            "name": profile.name,
            "profile_type": profile.profile_type.value,
            "physics_settings": asdict(profile.physics_settings),
            "visual_quality": asdict(profile.visual_quality),
            "hardware_requirements": asdict(profile.hardware_requirements),
            "description": profile.description,
            "use_case": profile.use_case
        }

        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile_dict, f, indent=2)

    def load_profile_from_file(self, profile_name: str) -> bool:
        """Load a profile from a JSON file"""
        profile_file = os.path.join(self.profiles_dir, f"{profile_name}.json")

        if not os.path.exists(profile_file):
            return False

        try:
            with open(profile_file, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)

            # Create profile objects from the loaded data
            physics_settings = PhysicsSettings(**profile_data["physics_settings"])
            visual_quality = VisualQuality(**profile_data["visual_quality"])
            hardware_requirements = HardwareRequirements(**profile_data["hardware_requirements"])

            profile_type = ProfileType(profile_data["profile_type"])

            profile = SimulationProfile(
                name=profile_data["name"],
                profile_type=profile_type,
                physics_settings=physics_settings,
                visual_quality=visual_quality,
                hardware_requirements=hardware_requirements,
                description=profile_data.get("description", ""),
                use_case=profile_data.get("use_case", "")
            )

            self.profiles[profile.name] = profile
            return True
        except Exception as e:
            print(f"Error loading profile {profile_name}: {e}")
            return False

    def apply_profile(self, profile_name: str) -> bool:
        """Apply a simulation profile (placeholder for actual implementation)"""
        profile = self.get_profile(profile_name)
        if not profile:
            print(f"Profile {profile_name} not found")
            return False

        print(f"Applying simulation profile: {profile.name}")
        print(f"  Physics Engine: {profile.physics_settings.engine}")
        print(f"  Time Step: {profile.physics_settings.time_step}s")
        print(f"  Real Time Factor: {profile.physics_settings.real_time_factor}")
        print(f"  Visual Quality: {profile.visual_quality.rendering_quality}")

        # In a real implementation, this would configure the simulation environment
        # according to the profile settings

        return True


# Example usage
if __name__ == "__main__":
    manager = SimulationProfileManager()

    print("Available profiles:", manager.list_profiles())

    # Apply education profile as an example
    manager.apply_profile("education")