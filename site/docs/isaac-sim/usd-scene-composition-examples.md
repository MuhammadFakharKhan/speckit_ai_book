---
title: USD Scene Composition Examples
description: Practical examples and validation steps for USD-based scene composition in Isaac Sim for humanoid robotics
sidebar_position: 7
tags: [usd, scene-composition, examples, isaac-sim, humanoid]
---

# USD Scene Composition Examples

## Introduction

This document provides practical examples of USD scene composition specifically for humanoid robotics applications in Isaac Sim. These examples demonstrate how to create complex, modular scenes that are suitable for both simulation and synthetic data generation.

## Basic Scene Structure

### Simple Humanoid Scene Example

```usda
# Example: Basic humanoid robot scene
# File: humanoid_scene.usda

#usda 1.0
(
    customLayerData = {
        string creator = "Isaac Sim"
        string description = "Basic humanoid robot scene for simulation"
    }
)

def Xform "World"
{
    # Robot definition
    def Xform "Robot"
    {
        # Robot body
        def Xform "Body"
        {
            # Torso
            def Capsule "Torso"
            {
                float radius = 0.15
                float height = 0.6
                rel material:binding = </Materials/RobotBodyMaterial>
            }

            # Head
            def Sphere "Head"
            {
                float radius = 0.12
                rel material:binding = </Materials/RobotHeadMaterial>
            }
        }

        # Left leg
        def Xform "LeftLeg"
        {
            def Capsule "Thigh"
            {
                float radius = 0.08
                float height = 0.4
                rel material:binding = </Materials/RobotBodyMaterial>
            }
            def Capsule "Shin"
            {
                float radius = 0.07
                float height = 0.4
                rel material:binding = </Materials/RobotBodyMaterial>
            }
        }

        # Right leg
        def Xform "RightLeg"
        {
            def Capsule "Thigh"
            {
                float radius = 0.08
                float height = 0.4
                rel material:binding = </Materials/RobotBodyMaterial>
            }
            def Capsule "Shin"
            {
                float radius = 0.07
                float height = 0.4
                rel material:binding = </Materials/RobotBodyMaterial>
            }
        }
    }

    # Environment
    def Xform "Environment"
    {
        # Ground plane
        def Plane "Ground"
        {
            float size = 10.0
            rel material:binding = </Materials/GroundMaterial>
        }

        # Obstacles
        def Xform "Obstacles"
        {
            def Cube "Cube1"
            {
                float size = 0.5
                rel material:binding = </Materials/ObstacleMaterial>
            }
        }
    }
}
```

## Modular Scene Composition

### Robot Asset Definition

```usda
# Example: Modular robot asset definition
# File: robot_asset.usda

#usda 1.0
(
    customLayerData = {
        string creator = "Isaac Sim"
        string description = "Modular humanoid robot asset"
    }
)

def Xform "HumanoidRobot"
{
    # Robot articulation
    def Xform "Articulation"
    {
        # Hip joint
        def Xform "Hip"
        {
            # Upper body
            def Xform "UpperBody"
            {
                # Spine
                def Xform "Spine"
                {
                    # Head
                    def Xform "Head"
                    {
                        def Sphere "HeadGeometry"
                        {
                            float radius = 0.12
                        }
                    }
                }

                # Left arm
                def Xform "LeftArm"
                {
                    def Xform "Shoulder"
                    {
                        def Xform "UpperArm"
                        {
                            def Xform "Elbow"
                            {
                                def Xform "LowerArm"
                                {
                                    def Xform "Wrist"
                                    {
                                        def Sphere "Hand"
                                        {
                                            float radius = 0.05
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

### Environment Assets

```usda
# Example: Modular environment assets
# File: environment_assets.usda

#usda 1.0
(
    customLayerData = {
        string creator = "Isaac Sim"
        string description = "Modular environment assets"
    }
)

# Indoor environment template
def Xform "IndoorEnvironment"
{
    # Floor
    def Plane "Floor"
    {
        float size = 20.0
    }

    # Walls
    def Xform "Walls"
    {
        def Cube "WallFront"
        {
            float3 size = (20.0, 3.0, 0.2)
        }
        def Cube "WallBack"
        {
            float3 size = (20.0, 3.0, 0.2)
        }
        def Cube "WallLeft"
        {
            float3 size = (0.2, 3.0, 20.0)
        }
        def Cube "WallRight"
        {
            float3 size = (0.2, 3.0, 20.0)
        }
    }

    # Furniture
    def Xform "Furniture"
    {
        def Xform "Table"
        {
            def Cube "TableTop"
            {
                float3 size = (1.0, 0.8, 0.8)
            }
            def Cube "Leg1"
            {
                float3 size = (0.05, 0.7, 0.05)
            }
            # ... additional legs
        }
    }
}
```

## Advanced Scene Composition

### Scene Variants

```usda
# Example: Scene with variants for different configurations
# File: scene_with_variants.usda

#usda 1.0
(
    customLayerData = {
        string creator = "Isaac Sim"
        string description = "Scene with variants for different configurations"
    }
)

def Xform "RobotScene"
{
    # Robot with variants
    def Xform "Robot"
    {
        # Define variant sets
        variantSet "robotConfig" = {
            "basic" = {
                def Sphere "Body"
                {
                    float radius = 0.1
                }
            }
            "detailed" = {
                def Xform "DetailedBody"
                {
                    def Capsule "Torso" { float radius = 0.15; float height = 0.6; }
                    def Sphere "Head" { float radius = 0.12; }
                }
            }
            "articulated" = {
                def Xform "ArticulatedBody"
                {
                    def Xform "UpperBody" { }
                    def Xform "LowerBody" { }
                }
            }
        }
    }

    # Environment with variants
    def Xform "Environment"
    {
        variantSet "environmentType" = {
            "indoor" = {
                def Xform "IndoorEnv" { }
            }
            "outdoor" = {
                def Xform "OutdoorEnv" { }
            }
            "laboratory" = {
                def Xform "LabEnv" { }
            }
        }
    }
}
```

### Composition Arcs

```usda
# Example: Using composition arcs for scene assembly
# File: main_scene.usda

#usda 1.0
(
    customLayerData = {
        string creator = "Isaac Sim"
        string description = "Main scene using composition arcs"
    }
)

def Xform "World"
{
    # Reference robot asset
    def "Robot" (
        references = @./robot_asset.usda@</HumanoidRobot>
    )
    {
        # Robot-specific overrides
        add xformOp:translate = (0, 0, 0)
    }

    # Reference environment asset
    def "Environment" (
        references = @./environment_assets.usda@</IndoorEnvironment>
    )
    {
        # Environment-specific overrides
        add xformOp:translate = (0, 0, 0)
    }

    # Add specific scene elements
    def Xform "SceneSpecific"
    {
        # Elements specific to this scene instance
        def Sphere "LightBall"
        {
            float radius = 0.1
        }
    }
}
```

## Validation and Testing

### Scene Validation Script

```python
# Example validation script for USD scenes
import omni
from pxr import Usd, UsdGeom, Sdf
import numpy as np

def validate_usd_scene(usd_path):
    """
    Validate a USD scene for humanoid robotics simulation
    """
    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Could not open USD file: {usd_path}")

    validation_results = {
        "basic_structure": check_basic_structure(stage),
        "robot_valid": validate_robot_structure(stage),
        "environment_valid": validate_environment_structure(stage),
        "physics_properties": validate_physics_properties(stage),
        "rendering_properties": validate_rendering_properties(stage),
        "simulation_compatibility": validate_simulation_compatibility(stage)
    }

    return validation_results

def check_basic_structure(stage):
    """
    Check basic scene structure
    """
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        return False

    # Check for required elements
    robot_prims = [p for p in stage.Traverse() if "Robot" in p.GetName() or "robot" in p.GetName()]
    environment_prims = [p for p in stage.Traverse() if "Environment" in p.GetName() or "env" in p.GetName()]

    return len(robot_prims) > 0 and len(environment_prims) > 0

def validate_robot_structure(stage):
    """
    Validate robot structure for humanoid simulation
    """
    robot_prims = [p for p in stage.Traverse() if "Robot" in p.GetName() or "robot" in p.GetName()]

    if not robot_prims:
        return False

    # Check for humanoid-specific components
    has_body = any("Body" in p.GetName() or "body" in p.GetName() for p in stage.Traverse())
    has_legs = any("Leg" in p.GetName() or "leg" in p.GetName() for p in stage.Traverse())
    has_head = any("Head" in p.GetName() or "head" in p.GetName() for p in stage.Traverse())

    return has_body and (has_legs or has_head)

def validate_environment_structure(stage):
    """
    Validate environment structure
    """
    # Check for ground/floor
    ground_prims = [p for p in stage.Traverse()
                   if any(ground_type in p.GetName().lower()
                         for ground_type in ["ground", "floor", "plane", "terrain"])]

    return len(ground_prims) > 0

def validate_physics_properties(stage):
    """
    Validate physics properties for simulation
    """
    # Check for collision geometries
    collision_prims = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdGeom.Mesh) or prim.HasAPI(UsdGeom.Capsule) or prim.HasAPI(UsdGeom.Sphere):
            # Check if collision API is applied
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_prims.append(prim)

    return len(collision_prims) > 0

def validate_rendering_properties(stage):
    """
    Validate rendering properties
    """
    # Check for materials
    material_prims = [p for p in stage.Traverse() if p.GetTypeName() == "Material"]

    # Check for cameras
    camera_prims = [p for p in stage.Traverse() if p.HasAPI(UsdGeom.Camera)]

    return len(material_prims) > 0 and len(camera_prims) > 0

def validate_simulation_compatibility(stage):
    """
    Validate compatibility with Isaac Sim simulation
    """
    # Check for required schemas and APIs
    has_xform = any(p for p in stage.Traverse() if p.HasAPI(UsdGeom.Xform))
    has_meshes = any(p for p in stage.Traverse() if p.HasAPI(UsdGeom.Mesh))

    return has_xform and has_meshes

# Example usage
if __name__ == "__main__":
    usd_path = "/path/to/scene.usd"
    results = validate_usd_scene(usd_path)
    print("Validation results:", results)
```

## Domain Randomization in USD

### Material Randomization

```usda
# Example: USD with material variations for domain randomization
# File: scene_with_material_variants.usda

#usda 1.0
(
    customLayerData = {
        string creator = "Isaac Sim"
        string description = "Scene with material variants for domain randomization"
    }
)

def Xform "World"
{
    # Robot with material variants
    def Xform "Robot"
    {
        variantSet "robotMaterial" = {
            "matte" = {
                def Material "RobotMaterial"
                {
                    def Shader "PreviewSurface"
                    {
                        color3f inputs:diffuseColor = (0.8, 0.8, 0.8)
                        float inputs:metallic = 0.0
                        float inputs:roughness = 0.8
                    }
                }
            }
            "shiny" = {
                def Material "RobotMaterial"
                {
                    def Shader "PreviewSurface"
                    {
                        color3f inputs:diffuseColor = (0.7, 0.7, 0.9)
                        float inputs:metallic = 0.9
                        float inputs:roughness = 0.1
                    }
                }
            }
            "colorful" = {
                def Material "RobotMaterial"
                {
                    def Shader "PreviewSurface"
                    {
                        color3f inputs:diffuseColor = (0.2, 0.8, 0.5)
                        float inputs:metallic = 0.2
                        float inputs:roughness = 0.4
                    }
                }
            }
        }
    }

    # Environment with lighting variants
    def Xform "Environment"
    {
        variantSet "lightingConfig" = {
            "bright" = {
                def DistantLight "KeyLight"
                {
                    float intensity = 1000
                    color3f color = (1.0, 1.0, 1.0)
                }
            }
            "dim" = {
                def DistantLight "KeyLight"
                {
                    float intensity = 300
                    color3f color = (0.8, 0.8, 0.9)
                }
            }
            "warm" = {
                def DistantLight "KeyLight"
                {
                    float intensity = 700
                    color3f color = (1.0, 0.9, 0.8)
                }
            }
        }
    }
}
```

## Python API Examples

### Creating Scenes Programmatically

```python
# Example: Creating USD scenes programmatically
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import omni
import carb

def create_humanoid_scene_programmatically(output_path):
    """
    Create a humanoid scene programmatically using USD Python API
    """
    # Create new stage
    stage = Usd.Stage.CreateNew(output_path)

    # Create world Xform
    world_prim = UsdGeom.Xform.Define(stage, "/World")

    # Create humanoid robot
    robot_prim = UsdGeom.Xform.Define(stage, "/World/Robot")

    # Create robot body
    torso_prim = UsdGeom.Capsule.Define(stage, "/World/Robot/Torso")
    torso_prim.CreateRadiusAttr(0.15)
    torso_prim.CreateHeightAttr(0.6)

    # Create head
    head_prim = UsdGeom.Sphere.Define(stage, "/World/Robot/Head")
    head_prim.CreateRadiusAttr(0.12)

    # Position head above torso
    head_prim.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0.5, 0))

    # Create legs
    left_leg_prim = UsdGeom.Xform.Define(stage, "/World/Robot/LeftLeg")
    right_leg_prim = UsdGeom.Xform.Define(stage, "/World/Robot/RightLeg")

    left_thigh = UsdGeom.Capsule.Define(stage, "/World/Robot/LeftLeg/Thigh")
    left_thigh.CreateRadiusAttr(0.08)
    left_thigh.CreateHeightAttr(0.4)

    left_shin = UsdGeom.Capsule.Define(stage, "/World/Robot/LeftLeg/Shin")
    left_shin.CreateRadiusAttr(0.07)
    left_shin.CreateHeightAttr(0.4)
    left_shin.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, -0.4, 0))

    # Create environment
    ground_prim = UsdGeom.Plane.Define(stage, "/World/Ground")
    ground_prim.CreateSizeAttr(10.0)

    # Add basic material
    material = UsdShade.Material.Define(stage, "/World/Materials/RobotMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/Materials/RobotMaterial/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.8, 0.8))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Bind material to robot
    binding_api = UsdShade.MaterialBindingAPI.Apply(torso_prim.GetPrim())
    binding_api.Bind(material)

    # Save the stage
    stage.GetRootLayer().Save()

    return stage

# Example usage
stage = create_humanoid_scene_programmatically("./humanoid_scene.usd")
```

## Isaac Sim Integration Examples

### Loading and Using USD Scenes

```python
# Example: Loading and using USD scenes in Isaac Sim
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import numpy as np

def load_and_setup_scene(usd_path):
    """
    Load a USD scene and set up for Isaac Sim simulation
    """
    # Get the USD context
    usd_context = omni.usd.get_context()

    # Open the stage
    usd_context.open_stage(usd_path)

    # Wait for stage to load
    import carb
    carb.log_info(f"Stage loaded: {usd_path}")

    # Set up the Isaac Sim world
    world = World(stage_units_in_meters=1.0)

    # Add the scene to the world
    add_reference_to_stage(usd_path=usd_path, prim_path="/World")

    # Reset the world to apply changes
    world.reset()

    return world

def setup_scene_for_synthetic_data(usd_path):
    """
    Set up scene specifically for synthetic data generation
    """
    # Load the scene
    world = load_and_setup_scene(usd_path)

    # Configure for synthetic data generation
    from omni.synthetic_utils import SyntheticDataHelper

    # Initialize synthetic data helper
    sd_helper = SyntheticDataHelper()

    # Configure output paths and formats
    sd_helper.set_output_directory("/path/to/output/dataset")
    sd_helper.set_resolution((1920, 1080))

    # Enable required data types
    sd_helper.enable_rgb_output(True)
    sd_helper.enable_depth_output(True)
    sd_helper.enable_segmentation_output(True)

    return world, sd_helper

# Example usage
world, sd_helper = setup_scene_for_synthetic_data("./humanoid_scene.usd")
```

## Best Practices

### Scene Organization

- **Modularity**: Create reusable scene components
- **Naming conventions**: Use consistent naming for prims and paths
- **Layer management**: Use composition arcs to manage complexity
- **Version control**: Track USD files in version control systems

### Performance Considerations

- **Geometry complexity**: Balance detail with performance
- **Instance referencing**: Use instancing for repeated objects
- **LOD systems**: Implement level-of-detail for complex scenes
- **Memory management**: Monitor memory usage for large scenes

### Validation Workflow

1. **Structure validation**: Verify scene hierarchy and components
2. **Physics validation**: Check collision geometries and properties
3. **Rendering validation**: Verify materials and lighting
4. **Simulation validation**: Test scene in Isaac Sim environment

## Troubleshooting

### Common Issues

- **Invalid references**: Ensure referenced files exist and paths are correct
- **Missing schemas**: Verify required USD schemas are applied
- **Transform issues**: Check for proper transformation hierarchies
- **Material binding**: Ensure materials are properly bound to geometries

### Debugging Strategies

- **USD viewer**: Use usdview to inspect scene structure
- **Isaac Sim logs**: Check Isaac Sim logs for loading errors
- **Validation scripts**: Run validation scripts to identify issues
- **Incremental loading**: Load scene components incrementally to isolate problems

This comprehensive guide provides practical examples and validation steps for USD scene composition in Isaac Sim, specifically tailored for humanoid robotics applications.