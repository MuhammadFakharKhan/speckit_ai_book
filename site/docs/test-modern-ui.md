---
title: Test Modern UI Features
sidebar_position: 1
description: Testing the new modern UI features
toc: true
---

# Test Modern UI Features

This page demonstrates the new modern UI features that have been implemented.

## Navigation & Layout

- Breadcrumb navigation is now available at the top of documentation pages
- A dropdown menu has been added to the navbar for easy access to different modules
- Enhanced footer with more relevant links for the ROS 2 community

## Visual Design

The site now features:
- Modern card-based layouts
- Improved typography with better hierarchy
- Enhanced color palette with better accessibility
- Smooth hover effects and transitions
- Depth and shadow effects for visual hierarchy

## Interactive Elements

### Collapsible Code Blocks

Here's an example of how code blocks can be made collapsible:

```python
# This is a sample ROS 2 Python node
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

### Callout Components

:::note
This is a note callout component with a modern design.
:::

:::tip
This is a tip callout component.
:::

:::caution
This is a caution callout component.
:::

:::danger
This is a danger callout component.
:::

## Table of Contents

On the right side of this page, you should see a table of contents sidebar that highlights the current section as you scroll.

## Reading Progress

As you scroll down this page, you should see a progress indicator at the top of the screen showing how far you've read.

## Back to Top Button

As you scroll down, a "Back to Top" button should appear in the bottom right corner.