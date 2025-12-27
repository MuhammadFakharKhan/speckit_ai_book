---
title: Synthetic Dataset Creation Workflows
description: Complete workflows for creating synthetic datasets with domain randomization for humanoid robotics applications
sidebar_position: 8
tags: [synthetic-data, dataset, workflow, domain-randomization, training]
---

# Synthetic Dataset Creation Workflows

## Introduction

This document outlines complete workflows for creating synthetic datasets using Isaac Sim, with a focus on domain randomization techniques that improve the synthetic-to-real transfer for humanoid robotics applications. These workflows are designed to generate high-quality datasets suitable for training perception and navigation systems.

## Complete Dataset Creation Pipeline

### Basic Pipeline Structure

```python
# Complete synthetic dataset creation pipeline
import omni
from omni.isaac.kit import SimulationApp
from omni.isaac.synthetic_utils import SyntheticDataHelper
import numpy as np
import os
from datetime import datetime
import json

class SyntheticDatasetPipeline:
    def __init__(self, config_path=None):
        # Initialize simulation app
        self.app = SimulationApp({"headless": False})

        # Initialize synthetic data helper
        self.sd_helper = SyntheticDataHelper()

        # Load configuration
        self.config = self.load_config(config_path) if config_path else self.default_config()

        # Setup output directory
        self.output_dir = self.setup_output_directory()

        # Initialize domain randomization
        self.domain_randomizer = DomainRandomizer(self.config)

        # Initialize scene manager
        self.scene_manager = SceneManager(self.config)

    def load_config(self, config_path):
        """
        Load configuration from file
        """
        with open(config_path, 'r') as f:
            return json.load(f)

    def default_config(self):
        """
        Default configuration for synthetic dataset creation
        """
        return {
            "dataset": {
                "name": "humanoid_robot_dataset",
                "description": "Synthetic dataset for humanoid robotics",
                "version": "1.0.0"
            },
            "rendering": {
                "resolution": [1920, 1080],
                "format": "png",
                "pixel_samples": 16,
                "max_surface_bounces": 4
            },
            "domain_randomization": {
                "enabled": True,
                "material_randomization": True,
                "lighting_randomization": True,
                "camera_randomization": False,
                "object_placement_randomization": True
            },
            "data_types": {
                "rgb": True,
                "depth": True,
                "segmentation": True,
                "normals": False,
                "flow": False
            },
            "collection_parameters": {
                "num_frames": 10000,
                "frame_interval": 1,
                "scene_changes_per_reset": 5,
                "reset_interval": 100
            }
        }

    def setup_output_directory(self):
        """
        Setup output directory with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"datasets/{self.config['dataset']['name']}_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/segmentation", exist_ok=True)
        os.makedirs(f"{output_dir}/metadata", exist_ok=True)

        return output_dir

    def run_dataset_creation(self):
        """
        Main pipeline for dataset creation
        """
        print(f"Starting dataset creation: {self.config['dataset']['name']}")

        # Initialize simulation
        self.initialize_simulation()

        # Main collection loop
        frame_count = 0
        scene_change_count = 0

        for i in range(self.config['collection_parameters']['num_frames']):
            # Randomize scene if needed
            if i % self.config['collection_parameters']['reset_interval'] == 0:
                self.randomize_scene()
                scene_change_count += 1

            # Capture frame
            self.capture_frame(frame_count)
            frame_count += 1

            # Step simulation
            self.app.update()

            # Progress reporting
            if i % 100 == 0:
                print(f"Captured {i}/{self.config['collection_parameters']['num_frames']} frames")

        # Finalize dataset
        self.finalize_dataset()

        print(f"Dataset creation completed. Output saved to: {self.output_dir}")

        return self.output_dir

    def initialize_simulation(self):
        """
        Initialize the simulation environment
        """
        # Setup stage
        stage = omni.usd.get_context().get_stage()

        # Configure rendering settings
        self.configure_rendering()

        # Load initial scene
        self.scene_manager.load_initial_scene()

        # Setup synthetic data capture
        self.setup_synthetic_data_capture()

    def configure_rendering(self):
        """
        Configure rendering settings for synthetic data
        """
        # Set resolution
        resolution = self.config['rendering']['resolution']
        self.sd_helper.set_resolution(resolution)

        # Configure rendering quality
        self.sd_helper.set_pixel_samples(self.config['rendering']['pixel_samples'])
        self.sd_helper.set_max_surface_bounces(self.config['rendering']['max_surface_bounces'])

    def setup_synthetic_data_capture(self):
        """
        Setup synthetic data capture based on configuration
        """
        # Enable data types as specified in config
        if self.config['data_types']['rgb']:
            self.sd_helper.enable_rgb_output(True)

        if self.config['data_types']['depth']:
            self.sd_helper.enable_depth_output(True)

        if self.config['data_types']['segmentation']:
            self.sd_helper.enable_segmentation_output(True)

        if self.config['data_types']['normals']:
            self.sd_helper.enable_normals_output(True)

        if self.config['data_types']['flow']:
            self.sd_helper.enable_flow_output(True)

    def randomize_scene(self):
        """
        Randomize scene based on domain randomization configuration
        """
        if self.config['domain_randomization']['enabled']:
            self.domain_randomizer.randomize_current_scene()

    def capture_frame(self, frame_number):
        """
        Capture a single frame and save to dataset
        """
        # Capture RGB
        if self.config['data_types']['rgb']:
            rgb_data = self.sd_helper.get_rgb_data()
            rgb_path = f"{self.output_dir}/rgb/frame_{frame_number:06d}.png"
            self.sd_helper.save_rgb_image(rgb_data, rgb_path)

        # Capture depth
        if self.config['data_types']['depth']:
            depth_data = self.sd_helper.get_depth_data()
            depth_path = f"{self.output_dir}/depth/frame_{frame_number:06d}.png"
            self.sd_helper.save_depth_image(depth_data, depth_path)

        # Capture segmentation
        if self.config['data_types']['segmentation']:
            seg_data = self.sd_helper.get_segmentation_data()
            seg_path = f"{self.output_dir}/segmentation/frame_{frame_number:06d}.png"
            self.sd_helper.save_segmentation_image(seg_data, seg_path)

        # Save metadata
        metadata = {
            "frame_number": frame_number,
            "timestamp": datetime.now().isoformat(),
            "domain_randomization_params": self.domain_randomizer.get_current_params(),
            "camera_pose": self.get_camera_pose(),
            "lighting_conditions": self.get_lighting_conditions()
        }

        metadata_path = f"{self.output_dir}/metadata/frame_{frame_number:06d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_camera_pose(self):
        """
        Get current camera pose for metadata
        """
        # Implementation depends on camera setup
        return {"position": [0, 0, 0], "rotation": [0, 0, 0, 1]}

    def get_lighting_conditions(self):
        """
        Get current lighting conditions for metadata
        """
        # Implementation depends on lighting setup
        return {"intensity": 1.0, "color": [1.0, 1.0, 1.0]}

    def finalize_dataset(self):
        """
        Finalize dataset creation and save summary
        """
        # Save dataset configuration
        config_path = f"{self.output_dir}/config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        # Save dataset summary
        summary = {
            "dataset_name": self.config['dataset']['name'],
            "total_frames": self.config['collection_parameters']['num_frames'],
            "data_types": self.config['data_types'],
            "domain_randomization": self.config['domain_randomization'],
            "creation_date": datetime.now().isoformat(),
            "output_directory": self.output_dir
        }

        summary_path = f"{self.output_dir}/summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

# Example usage
if __name__ == "__main__":
    pipeline = SyntheticDatasetPipeline()
    output_dir = pipeline.run_dataset_creation()
    print(f"Dataset created at: {output_dir}")
```

## Domain Randomization Implementation

### Domain Randomizer Class

```python
# Domain randomization implementation
import random
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf

class DomainRandomizer:
    def __init__(self, config):
        self.config = config
        self.stage = None
        self.current_params = {}

    def set_stage(self, stage):
        """
        Set the USD stage for randomization
        """
        self.stage = stage

    def randomize_current_scene(self):
        """
        Apply domain randomization to current scene
        """
        if self.config['domain_randomization']['material_randomization']:
            self.randomize_materials()

        if self.config['domain_randomization']['lighting_randomization']:
            self.randomize_lighting()

        if self.config['domain_randomization']['object_placement_randomization']:
            self.randomize_object_placement()

        if self.config['domain_randomization']['camera_randomization']:
            self.randomize_camera_parameters()

    def randomize_materials(self):
        """
        Randomize materials in the scene
        """
        # Find all materials in the scene
        materials = self.find_materials_in_scene()

        for material_path in materials:
            material_prim = self.stage.GetPrimAtPath(material_path)

            # Randomize diffuse color
            if random.random() < 0.8:  # 80% chance to randomize
                diffuse_color = self.randomize_color()
                self.set_material_property(material_prim, "diffuse_color", diffuse_color)

            # Randomize roughness
            if random.random() < 0.6:  # 60% chance to randomize
                roughness = random.uniform(0.1, 0.9)
                self.set_material_property(material_prim, "roughness", roughness)

            # Randomize metallic
            if random.random() < 0.3:  # 30% chance to randomize
                metallic = random.uniform(0.0, 1.0)
                self.set_material_property(material_prim, "metallic", metallic)

            # Store parameters for metadata
            self.current_params[f"material_{material_path}"] = {
                "diffuse_color": diffuse_color if 'diffuse_color' in locals() else None,
                "roughness": roughness if 'roughness' in locals() else None,
                "metallic": metallic if 'metallic' in locals() else None
            }

    def randomize_lighting(self):
        """
        Randomize lighting conditions in the scene
        """
        # Find all lights in the scene
        lights = self.find_lights_in_scene()

        for light_path in lights:
            light_prim = self.stage.GetPrimAtPath(light_path)

            # Randomize light intensity
            intensity_factor = random.uniform(0.5, 2.0)  # 0.5x to 2x original
            original_intensity = self.get_light_property(light_prim, "intensity")
            new_intensity = original_intensity * intensity_factor
            self.set_light_property(light_prim, "intensity", new_intensity)

            # Randomize light color
            if random.random() < 0.7:  # 70% chance to randomize color
                color_temperature = random.uniform(3000, 8000)  # Kelvin
                color = self.color_temperature_to_rgb(color_temperature)
                self.set_light_property(light_prim, "color", color)

            # Store parameters for metadata
            self.current_params[f"light_{light_path}"] = {
                "intensity_factor": intensity_factor,
                "color_temperature": color_temperature if 'color_temperature' in locals() else None
            }

    def randomize_object_placement(self):
        """
        Randomize placement of objects in the scene
        """
        # Find objects that can be randomly placed
        movable_objects = self.find_movable_objects()

        for obj_path in movable_objects:
            obj_prim = self.stage.GetPrimAtPath(obj_path)

            # Get current position
            current_pos = self.get_object_position(obj_prim)

            # Apply random offset
            random_offset = [
                random.uniform(-0.5, 0.5),  # X offset
                random.uniform(-0.5, 0.5),  # Y offset
                random.uniform(-0.2, 0.2)   # Z offset
            ]

            new_pos = [
                current_pos[0] + random_offset[0],
                current_pos[1] + random_offset[1],
                current_pos[2] + random_offset[2]
            ]

            # Set new position
            self.set_object_position(obj_prim, new_pos)

            # Store parameters for metadata
            self.current_params[f"object_{obj_path}"] = {
                "position_offset": random_offset
            }

    def find_materials_in_scene(self):
        """
        Find all material paths in the current scene
        """
        materials = []
        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "Material":
                materials.append(prim.GetPath())
        return materials

    def find_lights_in_scene(self):
        """
        Find all light paths in the current scene
        """
        lights = []
        for prim in self.stage.Traverse():
            if prim.GetTypeName() in ["DistantLight", "DomeLight", "RectLight", "SphereLight"]:
                lights.append(prim.GetPath())
        return lights

    def find_movable_objects(self):
        """
        Find objects that can be randomly moved
        """
        movable_objects = []
        for prim in self.stage.Traverse():
            # Consider objects that are not part of the robot or fixed environment
            prim_name = prim.GetName().lower()
            if (prim.IsA(UsdGeom.Xform) and
                "robot" not in prim_name and
                "ground" not in prim_name and
                "floor" not in prim_name):
                movable_objects.append(prim.GetPath())
        return movable_objects

    def randomize_color(self):
        """
        Generate a random color
        """
        return [random.random(), random.random(), random.random()]

    def color_temperature_to_rgb(self, color_temperature):
        """
        Convert color temperature in Kelvin to RGB
        """
        # Simplified algorithm for color temperature to RGB
        temp = color_temperature / 100

        if temp <= 66:
            red = 255
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            red = max(0, min(255, red))

        if temp <= 66:
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)

        green = max(0, min(255, green))

        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
            blue = max(0, min(255, blue))

        return [red/255, green/255, blue/255]

    def set_material_property(self, material_prim, property_name, value):
        """
        Set a material property
        """
        # Find the shader inside the material
        for child in material_prim.GetChildren():
            if child.GetTypeName() == "Shader":
                shader = child
                # Set the property on the shader
                input_path = f"inputs:{property_name}"
                shader.CreateInput(input_path, Sdf.ValueTypeNames.Color3f if property_name == "diffuse_color" else Sdf.ValueTypeNames.Float)
                shader.GetInput(input_path).Set(value)
                break

    def set_light_property(self, light_prim, property_name, value):
        """
        Set a light property
        """
        attr_name = f"inputs:{property_name}" if property_name in ["color"] else property_name
        attr = light_prim.GetAttribute(attr_name)
        if attr:
            attr.Set(value)

    def get_light_property(self, light_prim, property_name):
        """
        Get a light property
        """
        attr_name = f"inputs:{property_name}" if property_name in ["color"] else property_name
        attr = light_prim.GetAttribute(attr_name)
        if attr:
            return attr.Get()
        return 1.0  # Default value

    def get_object_position(self, obj_prim):
        """
        Get object position
        """
        xform = UsdGeom.Xformable(obj_prim)
        transform_ops = xform.GetOrderedXformOps()

        for op in transform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op.Get()

        return [0, 0, 0]  # Default position

    def set_object_position(self, obj_prim, position):
        """
        Set object position
        """
        xform = UsdGeom.Xformable(obj_prim)
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(position))

    def get_current_params(self):
        """
        Get current domain randomization parameters
        """
        return self.current_params.copy()
```

## Scene Management

### Scene Manager Class

```python
# Scene management for synthetic dataset creation
import random
import os
from pxr import Usd, UsdGeom, Sdf

class SceneManager:
    def __init__(self, config):
        self.config = config
        self.stage = None
        self.scene_templates = self.load_scene_templates()

    def load_scene_templates(self):
        """
        Load available scene templates
        """
        # In a real implementation, this would load from files
        return [
            "indoor_office.usd",
            "outdoor_park.usd",
            "laboratory.usd",
            "warehouse.usd",
            "apartment.usd"
        ]

    def set_stage(self, stage):
        """
        Set the USD stage for scene management
        """
        self.stage = stage

    def load_initial_scene(self):
        """
        Load the initial scene for dataset creation
        """
        # For this example, we'll create a simple scene programmatically
        # In practice, you'd load from a USD file
        self.create_basic_scene()

    def create_basic_scene(self):
        """
        Create a basic scene for humanoid robot dataset
        """
        # Create world
        world_prim = UsdGeom.Xform.Define(self.stage, "/World")

        # Create ground plane
        ground_prim = UsdGeom.Plane.Define(self.stage, "/World/Ground")
        ground_prim.CreateSizeAttr(20.0)

        # Create simple humanoid robot (simplified representation)
        robot_prim = UsdGeom.Xform.Define(self.stage, "/World/Robot")

        # Robot body parts
        torso_prim = UsdGeom.Capsule.Define(self.stage, "/World/Robot/Torso")
        torso_prim.CreateRadiusAttr(0.15)
        torso_prim.CreateHeightAttr(0.6)

        head_prim = UsdGeom.Sphere.Define(self.stage, "/World/Robot/Head")
        head_prim.CreateRadiusAttr(0.12)
        # Position head above torso
        UsdGeom.XformCommonAPI(head_prim.GetPrim()).SetTranslate((0, 0.5, 0))

        # Create simple environment objects
        self.create_environment_objects()

    def create_environment_objects(self):
        """
        Create environment objects for the scene
        """
        # Create some obstacles/objects
        for i in range(5):
            obj_name = f"Obstacle_{i}"
            obj_prim = UsdGeom.Cube.Define(self.stage, f"/World/{obj_name}")
            obj_prim.CreateSizeAttr(0.5)

            # Position randomly in the scene
            x_pos = random.uniform(-5, 5)
            y_pos = random.uniform(-5, 5)
            z_pos = 0.25  # Half the cube size

            UsdGeom.XformCommonAPI(obj_prim.GetPrim()).SetTranslate((x_pos, y_pos, z_pos))

    def change_scene(self):
        """
        Change to a different scene configuration
        """
        # This would typically load a different scene template
        # For this example, we'll just randomize the current scene
        self.randomize_current_scene_objects()

    def randomize_current_scene_objects(self):
        """
        Randomize objects in the current scene
        """
        # Move existing objects to new positions
        for prim in self.stage.Traverse():
            if prim.GetName().startswith("Obstacle_"):
                # Randomize position
                x_pos = random.uniform(-8, 8)
                y_pos = random.uniform(-8, 8)
                z_pos = 0.25  # Keep on ground

                UsdGeom.XformCommonAPI(prim).SetTranslate((x_pos, y_pos, z_pos))

                # Randomize size slightly
                size_attr = prim.GetAttribute("size")
                if size_attr:
                    original_size = size_attr.Get()
                    new_size = original_size * random.uniform(0.8, 1.2)
                    size_attr.Set(new_size)
```

## Dataset Validation and Quality Assurance

### Validation Pipeline

```python
# Dataset validation and quality assurance
import cv2
import numpy as np
from PIL import Image
import json
import os

class DatasetValidator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.stats = {}

    def validate_dataset(self):
        """
        Validate the entire dataset
        """
        print("Starting dataset validation...")

        # Validate structure
        structure_valid = self.validate_dataset_structure()

        # Validate individual samples
        sample_stats = self.validate_samples()

        # Generate quality metrics
        quality_metrics = self.calculate_quality_metrics()

        # Generate validation report
        report = {
            "structure_valid": structure_valid,
            "sample_count": sample_stats["total_samples"],
            "missing_samples": sample_stats["missing_samples"],
            "quality_metrics": quality_metrics,
            "validation_passed": structure_valid and sample_stats["missing_samples"] == 0
        }

        # Save validation report
        self.save_validation_report(report)

        return report

    def validate_dataset_structure(self):
        """
        Validate the dataset directory structure
        """
        required_dirs = ["rgb", "depth", "segmentation", "metadata"]

        for dir_name in required_dirs:
            dir_path = os.path.join(self.dataset_path, dir_name)
            if not os.path.exists(dir_path):
                print(f"Missing required directory: {dir_path}")
                return False

        return True

    def validate_samples(self):
        """
        Validate individual samples in the dataset
        """
        # Get list of all frame numbers
        rgb_files = [f for f in os.listdir(os.path.join(self.dataset_path, "rgb")) if f.endswith(".png")]
        frame_numbers = [int(f.split("_")[1].split(".")[0]) for f in rgb_files]

        total_samples = len(frame_numbers)
        missing_samples = 0

        for frame_num in frame_numbers:
            # Check if all required files exist for this frame
            rgb_path = os.path.join(self.dataset_path, "rgb", f"frame_{frame_num:06d}.png")
            depth_path = os.path.join(self.dataset_path, "depth", f"frame_{frame_num:06d}.png")
            seg_path = os.path.join(self.dataset_path, "segmentation", f"frame_{frame_num:06d}.png")
            meta_path = os.path.join(self.dataset_path, "metadata", f"frame_{frame_num:06d}.json")

            for path in [rgb_path, depth_path, seg_path, meta_path]:
                if not os.path.exists(path):
                    print(f"Missing file for frame {frame_num}: {path}")
                    missing_samples += 1
                    break

        return {
            "total_samples": total_samples,
            "missing_samples": missing_samples
        }

    def calculate_quality_metrics(self):
        """
        Calculate quality metrics for the dataset
        """
        # Calculate metrics on a sample of frames
        sample_size = min(100, len(os.listdir(os.path.join(self.dataset_path, "rgb"))))

        rgb_quality = []
        depth_quality = []
        seg_quality = []

        # Sample frames for quality assessment
        frame_files = os.listdir(os.path.join(self.dataset_path, "rgb"))
        sample_frames = random.sample(frame_files, sample_size)

        for frame_file in sample_frames:
            frame_num = int(frame_file.split("_")[1].split(".")[0])

            # Load RGB image and assess quality
            rgb_path = os.path.join(self.dataset_path, "rgb", f"frame_{frame_num:06d}.png")
            rgb_img = cv2.imread(rgb_path)
            if rgb_img is not None:
                rgb_quality.append(self.assess_image_quality(rgb_img))

            # Load depth image and assess quality
            depth_path = os.path.join(self.dataset_path, "depth", f"frame_{frame_num:06d}.png")
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                depth_quality.append(self.assess_depth_quality(depth_img))

            # Load segmentation image and assess quality
            seg_path = os.path.join(self.dataset_path, "segmentation", f"frame_{frame_num:06d}.png")
            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            if seg_img is not None:
                seg_quality.append(self.assess_segmentation_quality(seg_img))

        return {
            "rgb_quality_avg": np.mean(rgb_quality) if rgb_quality else 0,
            "depth_quality_avg": np.mean(depth_quality) if depth_quality else 0,
            "seg_quality_avg": np.mean(seg_quality) if seg_quality else 0,
            "sample_size": sample_size
        }

    def assess_image_quality(self, img):
        """
        Assess RGB image quality (simplified)
        """
        # Calculate basic quality metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Focus measure using Laplacian variance
        focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness measure
        brightness = np.mean(gray)

        # Contrast measure
        contrast = np.std(gray)

        # Combine metrics (weights can be adjusted)
        quality_score = (focus_measure * 0.4 + brightness * 0.3 + contrast * 0.3) / 1000

        return min(quality_score, 1.0)  # Normalize to 0-1 range

    def assess_depth_quality(self, depth_img):
        """
        Assess depth image quality
        """
        # Check for valid depth values (not all zeros or invalid)
        valid_pixels = np.count_nonzero(depth_img > 0)
        total_pixels = depth_img.size

        valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0

        # Additional depth quality checks can be added here
        return valid_ratio

    def assess_segmentation_quality(self, seg_img):
        """
        Assess segmentation image quality
        """
        # Check for proper segmentation (multiple unique values)
        unique_values = len(np.unique(seg_img))

        # Should have more than just background (0) and possibly unlabeled (255)
        seg_quality = min(unique_values / 10.0, 1.0)  # Normalize

        return seg_quality

    def save_validation_report(self, report):
        """
        Save validation report to file
        """
        report_path = os.path.join(self.dataset_path, "validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Validation report saved to: {report_path}")
```

## Configuration Files

### Example Configuration File

```json
{
  "dataset": {
    "name": "humanoid_robot_perception_dataset",
    "description": "Synthetic dataset for training perception models for humanoid robots",
    "version": "1.0.0",
    "author": "Isaac Sim User",
    "license": "MIT"
  },
  "rendering": {
    "resolution": [1280, 720],
    "format": "png",
    "pixel_samples": 16,
    "max_surface_bounces": 4,
    "enable_denoiser": true
  },
  "domain_randomization": {
    "enabled": true,
    "material_randomization": {
      "enabled": true,
      "probability": 0.8,
      "diffuse_color_range": [[0.1, 0.1, 0.1], [1.0, 1.0, 1.0]],
      "roughness_range": [0.1, 0.9],
      "metallic_range": [0.0, 1.0]
    },
    "lighting_randomization": {
      "enabled": true,
      "probability": 0.7,
      "intensity_range": [0.5, 2.0],
      "color_temperature_range": [3000, 8000]
    },
    "object_placement_randomization": {
      "enabled": true,
      "probability": 0.9,
      "position_jitter": [0.5, 0.5, 0.2],
      "rotation_jitter": [5, 5, 5]
    },
    "camera_randomization": {
      "enabled": false,
      "position_jitter": [0.1, 0.1, 0.05],
      "rotation_jitter": [1, 1, 1]
    }
  },
  "data_types": {
    "rgb": true,
    "depth": true,
    "segmentation": true,
    "normals": false,
    "flow": false
  },
  "collection_parameters": {
    "num_frames": 5000,
    "frame_interval": 1,
    "scene_changes_per_reset": 5,
    "reset_interval": 100,
    "capture_frequency": 30
  },
  "validation": {
    "enabled": true,
    "quality_thresholds": {
      "rgb_quality": 0.5,
      "depth_valid_ratio": 0.8,
      "seg_diversity": 0.3
    }
  }
}
```

## Best Practices and Guidelines

### Dataset Creation Best Practices

1. **Planning Phase**:
   - Define the specific perception task the dataset will address
   - Determine the required scene complexity and variation
   - Plan for adequate domain randomization coverage

2. **Scene Design**:
   - Create diverse but representative environments
   - Include relevant objects and obstacles for humanoid navigation
   - Consider lighting conditions that match target deployment

3. **Domain Randomization**:
   - Start with conservative randomization ranges
   - Gradually expand ranges based on validation results
   - Monitor synthetic-to-real transfer performance

4. **Quality Assurance**:
   - Implement automated validation checks
   - Sample dataset regularly during creation
   - Validate with target perception models

### Performance Considerations

- **Render Quality vs Speed**: Balance rendering quality with dataset creation speed
- **Memory Management**: Monitor GPU and system memory usage
- **Storage Requirements**: Plan for large storage needs (10k+ images per dataset)
- **Parallel Processing**: Consider multi-process approaches for faster generation

This comprehensive workflow provides a complete pipeline for creating synthetic datasets with domain randomization for humanoid robotics applications, including validation and quality assurance measures.