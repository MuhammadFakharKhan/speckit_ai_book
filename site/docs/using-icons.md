---
title: Using Icons in Documentation
description: How to use Docusaurus icons in the ROS 2 for Humanoid Robotics documentation
sidebar_position: 998
---

# Developer Guide: Using Icons in Robotics Documentation

This guide provides developers with instructions on how to use Docusaurus icons in the ROS 2 for Humanoid Robotics documentation.

## Available Robotics Icons

The following robotics-specific icons are available for use in the documentation:

### ROS 2 Icons
- <span className="icon-ros">🔄</span> ROS 2 related content
- <span className="icon-ros2">🔄2</span> ROS 2 specific features

### Isaac Ecosystem Icons
- <span className="icon-isaac">💡</span> Isaac related content
- <span className="icon-nav2">🧭</span> Navigation 2 (Nav2) content

### General Robotics Icons
- <span className="icon-robot">🤖</span> General robotics content
- <span className="icon-navigation">🧭</span> Navigation content
- <span className="icon-perception">👁️</span> Perception content
- <span className="icon-manipulation"> gripper</span> Manipulation content
- <span className="icon-planning">📋</span> Planning content

## How to Use Icons

### CSS Class Approach (Simple)
To use these icons in your documentation, you can add the appropriate CSS class to any HTML element:

```markdown
<span className="icon-ros">🔄</span> This section covers ROS 2 concepts
```

### React Component Approach (Advanced)
For more complex implementations, you can use React components:

```jsx
import { ROSIcon } from '@site/src/components/icons';

<ROSIcon size="md" /> This section covers ROS 2 concepts
```

The component approach supports different sizes:
- `size="sm"` for small icons
- `size="md"` for medium icons (default)
- `size="lg"` for large icons
- `size="xl"` for extra-large icons

## Developer Implementation Guide

### 1. Adding New Icons
To add new icons to the documentation:

1. Create a new icon component in `src/components/icons/`
2. Add the icon to the `src/components/icons/index.js` export file
3. Update the CSS to include any necessary styling
4. Document the new icon in this guide

### 2. Creating New Icon Components
When creating new icon components, follow this pattern:

```jsx
import React from 'react';

const NewIcon = ({ size = 'md', className, ...props }) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const sizeClass = sizes[size] || sizes.md;
  const combinedClassName = `${sizeClass} ${className || ''}`;

  return (
    <svg
      className={combinedClassName}
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      {/* SVG path definition here */}
    </svg>
  );
};

export default NewIcon;
```

### 3. Styling Icons with CSS
For CSS-based icons, add new classes to `src/css/custom.css`:

```css
.icon-new-icon::before {
  content: "your-icon-content";
  display: inline-block;
  margin-right: 5px;
}
```

## Best Practices for Developers

1. **Consistency**: Use the same icon for similar concepts throughout the documentation
2. **Accessibility**: Always provide text alternatives for icons
3. **Context**: Use icons to enhance, not replace, clear text descriptions
4. **Relevance**: Choose icons that clearly relate to the content they represent
5. **Performance**: Optimize SVG icons for minimal file size
6. **Maintainability**: Keep icon components organized in the icons directory

## Example Usage

Here are some examples of how icons can be used effectively:

### Navigation Section
<span className="icon-navigation">🧭</span> **Navigation in Humanoid Robots**: This section covers path planning, localization, and navigation algorithms for humanoid robots using Nav2.

### Perception Section
<span className="icon-perception">👁️</span> **Perception Systems**: This section covers sensor processing, object detection, and environment understanding for humanoid robots.

### Manipulation Section
<span className="icon-manipulation"> gripper</span> **Manipulation Tasks**: This section covers grasping, manipulation planning, and end-effector control for humanoid robots.

## Troubleshooting

### Icons Not Displaying
- Check that the CSS class name matches exactly
- Verify the icon component is properly imported
- Ensure the icon component file exists and exports correctly
- Check browser console for any error messages

### Accessibility Issues
- Always provide alternative text for screen readers
- Use appropriate color contrast for visibility
- Test with accessibility tools to ensure compliance

## Testing Icons

To test that icons are working properly:
1. Run the development server: `npm run start`
2. Navigate to pages where icons are used
3. Verify icons display correctly in different browsers
4. Test with accessibility tools
5. Check that icons scale properly at different sizes