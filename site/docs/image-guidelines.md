---
title: Image Guidelines
description: Guidelines for using images in the ROS 2 for Humanoid Robotics documentation
sidebar_position: 999
---

# Image Guidelines

## Image Optimization

When adding images to the documentation, please follow these optimization guidelines:

1. **File Formats**:
   - Use SVG for diagrams, logos, and vector graphics
   - Use WebP for photographs (with PNG fallback)
   - Use PNG for graphics with transparency
   - Use JPG for photographs with many colors

2. **File Sizes**:
   - Keep image file sizes under 500KB when possible
   - Compress images using tools like ImageOptim, TinyPNG, or similar
   - For photographs, aim for 80% quality as a good balance between size and quality

3. **Dimensions**:
   - Use appropriate dimensions for the content
   - Don't use oversized images and scale them down with CSS
   - Consider responsive design for different screen sizes

## Alt Text

Always include descriptive alt text for accessibility:

```markdown
![Description of the image content](@site/static/img/image-name.png)
```

## Image Placement

Place images in the `static/img/` directory and reference them using:

```markdown
![Alt text](/img/image-name.png)
```

Or using the Docusaurus syntax:

```markdown
![Alt text](/img/image-name.svg)
```

## SVG Guidelines

For SVG images:
- Use simple, clean vector graphics
- Keep file sizes small by minimizing path complexity
- Ensure they scale well at different sizes
- Test rendering across different browsers

## Icon Usage Guidelines

For using Docusaurus icons in robotics documentation:

### 1. Robotics-Specific Icons
- Use `icon-ros` class for ROS 2 related content
- Use `icon-isaac` class for Isaac ecosystem content
- Use `icon-nav2` class for Navigation 2 content
- Use `icon-robot` class for general robotics content
- Use `icon-navigation`, `icon-perception`, `icon-manipulation`, and `icon-planning` for specific robotics domains

### 2. Icon Implementation
- Use CSS classes for simple emoji-based icons: `<span className="icon-ros">🔄</span>`
- Use React components for more complex icons: `<ROSIcon size="md" />`
- Always provide text alternatives for accessibility
- Use consistent icon sizing (sm, md, lg, xl)

### 3. Best Practices
- Use icons to enhance, not replace, clear text descriptions
- Maintain consistency in icon usage throughout the documentation
- Choose icons that clearly relate to the content they represent
- Ensure icons are accessible to users with visual impairments