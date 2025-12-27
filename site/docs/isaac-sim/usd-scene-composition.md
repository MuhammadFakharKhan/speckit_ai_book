---
title: USD Scene Composition
description: Creating and composing scenes using Universal Scene Description (USD) in Isaac Sim for humanoid robotics applications
sidebar_position: 3
tags: [usd, scene-composition, isaac-sim, humanoid]
---

# USD Scene Composition

## Introduction

Universal Scene Description (USD) is the foundation of scene composition in Isaac Sim. USD provides a powerful, scalable framework for creating, assembling, and editing 3D scenes. Understanding USD is crucial for creating effective simulation environments for humanoid robots.

## USD Fundamentals

### Core Concepts

- **Prims**: The basic building blocks of USD scenes
- **Properties**: Attributes of prims such as position, rotation, scale
- **Relationships**: Connections between prims
- **Variants**: Different versions of the same prim

### USD Composition Arcs

USD uses composition arcs to build complex scenes from simpler components:

- **References**: Include content from other USD files
- **Payloads**: Lazy-loaded references for performance
- **Inherits**: Share scene graph structure
- **Specializes**: Override specific aspects of referenced content

## Creating Humanoid Robot Scenes

### Robot Setup

When creating scenes with humanoid robots, consider:

- Robot articulation and joint configurations
- Collision geometry for physics simulation
- Visual mesh for rendering
- Sensor placements for perception tasks

### Environment Design

Design environments that are suitable for humanoid navigation:

- Appropriate floor surfaces and friction
- Obstacles at various heights relevant to bipedal navigation
- Doorways and passages suitable for humanoid dimensions
- Lighting conditions that match real-world scenarios

## USD Best Practices

### Modularity

- Create reusable scene components
- Use USD packages for common assets
- Organize scenes hierarchically
- Version control USD files appropriately

### Performance

- Use payloads for large scene components
- Implement level-of-detail (LOD) for complex assets
- Minimize the number of unique materials
- Use instancing for repeated objects

## Example: Simple Humanoid Scene

```usd
# Example USD scene with a humanoid robot
def Xform "World"
{
    def Xform "Robot"
    {
        # Robot body and joints
        def Xform "Body"
        {
            # Body components
        }

        def Xform "LeftLeg"
        {
            # Left leg components
        }

        def Xform "RightLeg"
        {
            # Right leg components
        }
    }

    def Xform "Environment"
    {
        # Environment components
    }
}
```

## Integration with Isaac Sim

USD scenes integrate seamlessly with Isaac Sim's physics and rendering systems:

- Physics properties can be defined in USD
- Materials and shaders are preserved
- Animation and rigging data is maintained
- Sensor placements are configurable

## Validation and Testing

Always validate your USD scenes:

- Check for proper hierarchy and transforms
- Verify collision geometry
- Test physics simulation behavior
- Ensure rendering quality meets requirements