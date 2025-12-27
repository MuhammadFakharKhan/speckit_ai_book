# Performance Optimization Guide: Module 2 - Digital Twin Simulation

This guide provides strategies and techniques for optimizing the performance of your Gazebo-Unity digital twin simulation environment.

## Understanding Performance Metrics

### Key Performance Indicators (KPIs)

- **Real-time Factor (RTF)**: Ratio of simulation time to real time (1.0 = real-time, >1.0 = faster than real-time)
- **Simulation Update Rate**: Frequency of physics simulation updates (typically 100-1000 Hz)
- **Rendering Frame Rate**: Visual frame rate in Unity and Gazebo (target: 30-60 FPS)
- **CPU Usage**: Percentage of CPU resources consumed by simulation
- **Memory Usage**: RAM consumption by simulation components
- **Network Latency**: Delay in communication between simulation components

### Performance Monitoring Tools

#### Gazebo Performance Monitoring
```bash
# Monitor Gazebo statistics
gz stats

# Monitor specific topics
gz topic -i -t /stats

# Monitor system resources
htop
```

#### ROS 2 Performance Monitoring
```bash
# Monitor topic bandwidth
ros2 topic hz /joint_states

# Monitor node performance
ros2 run topological_navigation performance_monitor
```

## Physics Simulation Optimization

### Time Step Optimization

The physics time step is critical for both accuracy and performance:

```yaml
# physics.yaml - Optimal time step settings
physics:
  # For high-fidelity simulation: 0.0005-0.001
  # For performance: 0.002-0.005
  time_step: 0.001
  real_time_factor: 1.0
```

**Guidelines**:
- Smaller time steps: More accurate but slower
- Larger time steps: Faster but less stable
- Always test stability with your specific models

### Solver Optimization

Adjust solver parameters for better performance:

```yaml
# physics.yaml - Solver settings
physics:
  solver:
    # For performance: 10-50 iterations
    # For accuracy: 100-200 iterations
    iterations: 50
    # Error reduction parameter
    erp: 0.2
    # Constraint force mixing
    cfm: 1e-5
```

### Collision Geometry Optimization

Simplify collision geometry for better performance:

```xml
<!-- Model SDF with optimized collision geometry -->
<model name="optimized_robot">
  <link name="base_link">
    <!-- Use simple shapes instead of complex meshes -->
    <collision name="collision">
      <geometry>
        <!-- Use box instead of complex mesh -->
        <box>
          <size>0.5 0.5 0.5</size>
        </box>
      </geometry>
    </collision>
  </link>
</model>
```

**Best Practices**:
- Use primitive shapes (box, sphere, cylinder) instead of meshes
- Combine multiple simple shapes instead of complex meshes
- Use the minimum number of collision elements needed

## Sensor Simulation Optimization

### Camera Optimization

```yaml
# Sensor configuration with optimized settings
sensor:
  camera:
    # Reduce resolution for performance
    image_width: 640  # Reduce from 1280
    image_height: 480 # Reduce from 960
    # Lower update rate if not needed
    update_rate: 30   # Reduce from 60
```

### LIDAR Optimization

```yaml
# LIDAR sensor optimization
sensor:
  lidar:
    # Reduce number of rays for performance
    horizontal_samples: 320  # Reduce from 640
    vertical_samples: 1      # Use 1 for 2D LIDAR
    # Increase resolution for performance
    resolution: 2            # Increase from 1
```

### Sensor Bridge Optimization

Optimize the ROS 2 to Gazebo bridge:

```yaml
# sensor_bridge.yaml - Optimized settings
sensor_bridge:
  ros__parameters:
    # Reduce queue sizes to prevent memory buildup
    qos_sensor_data:
      history: "keep_last"
      depth: 5        # Reduce from 10
      reliability: "best_effort"  # Use for performance
```

## Unity Visualization Optimization

### Asset Optimization

#### Model Optimization
- Use Level of Detail (LOD) systems
- Reduce polygon count for distant objects
- Use occlusion culling
- Optimize textures (compress, reduce size)

#### Script Optimization
```csharp
// Unity C# script - Optimized update loop
public class OptimizedRobotController : MonoBehaviour
{
    // Cache component references
    private Transform cachedTransform;
    private Rigidbody cachedRigidbody;

    // Use fixed time intervals for updates
    private float lastUpdateTime;
    private const float UPDATE_INTERVAL = 0.033f; // ~30 Hz

    void Start()
    {
        cachedTransform = transform;
        cachedRigidbody = GetComponent<Rigidbody>();
    }

    void Update()
    {
        float currentTime = Time.time;
        if (currentTime - lastUpdateTime >= UPDATE_INTERVAL)
        {
            UpdateRobotPosition();
            lastUpdateTime = currentTime;
        }
    }
}
```

### Rendering Optimization

- Use occlusion culling
- Implement frustum culling
- Use texture atlasing
- Reduce overdraw
- Use efficient shaders

## Simulation Profile Management

### Profile Configuration

Use different profiles for different use cases:

```python
# Profile optimization example
from simulation.profile_manager import SimulationProfileManager

manager = SimulationProfileManager()

# For development and testing
dev_profile = manager.get_profile("performance")
manager.apply_profile("performance")

# For detailed analysis
analysis_profile = manager.get_profile("high_fidelity")
manager.apply_profile("high_fidelity")
```

### API-Based Profile Switching

```bash
# Switch to performance profile via API
curl -X POST http://localhost:5001/api/profiles/performance/apply

# Switch to high-fidelity profile
curl -X POST http://localhost:5001/api/profiles/high_fidelity/apply
```

## Hardware Considerations

### CPU Optimization

- **Physics Simulation**: CPU-intensive, benefits from multiple cores
- **ROS 2 Nodes**: Generally lightweight but can scale with complexity
- **Optimization**: Use dedicated cores for physics, limit thread contention

### GPU Optimization

- **Gazebo Rendering**: GPU-intensive for visualization
- **Unity Rendering**: GPU-intensive for advanced visualization
- **Optimization**: Ensure adequate VRAM and modern graphics drivers

### Memory Management

Monitor and optimize memory usage:

```bash
# Monitor memory usage
free -h
cat /proc/meminfo

# Check specific process memory
ps aux | grep -E "(gz|ros|Unity)"
```

## Network Optimization

### Communication Efficiency

- Use appropriate QoS settings
- Reduce message frequency where possible
- Compress large data (images, point clouds)
- Use efficient serialization formats

### Bridge Optimization

```yaml
# Optimize ros-gz bridge
bridge_config:
  - ros_topic_name: "/compressed_image"
    gz_topic_name: "/camera/image/compressed"
    ros_type_name: "sensor_msgs/msg/CompressedImage"
    gz_type_name: "gz.msgs.CompressedImage"
    direction: "GZ_TO_ROS"
    # Use compressed format for performance
```

## Code Optimization Techniques

### Efficient Data Structures

```python
# Optimized data handling for simulation
import numpy as np
from collections import deque

class OptimizedSimulationData:
    def __init__(self, buffer_size=100):
        # Use numpy arrays for numerical computations
        self.joint_positions = np.zeros(6, dtype=np.float32)
        self.joint_velocities = np.zeros(6, dtype=np.float32)

        # Use deque for efficient append/pop operations
        self.position_history = deque(maxlen=buffer_size)
        self.velocity_history = deque(maxlen=buffer_size)

    def update_joints(self, positions, velocities):
        # Vectorized operations for efficiency
        np.copyto(self.joint_positions, positions)
        np.copyto(self.joint_velocities, velocities)

        # Efficient history tracking
        self.position_history.append(np.copy(positions))
        self.velocity_history.append(np.copy(velocities))
```

### Asynchronous Processing

```python
# Asynchronous sensor data processing
import asyncio
import concurrent.futures

class AsyncSensorProcessor:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def process_sensor_data(self, sensor_data):
        # Process data asynchronously
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._process_blocking_operation,
            sensor_data
        )
        return result

    def _process_blocking_operation(self, data):
        # Heavy computation happens in separate thread
        # to avoid blocking the main simulation loop
        processed = self.heavy_computation(data)
        return processed
```

## Monitoring and Profiling

### Performance Profiling Tools

#### Python Profiling
```python
import cProfile
import pstats

def profile_simulation():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run simulation code
    run_simulation()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

#### System Monitoring
```bash
# Monitor CPU usage per process
top -p $(pgrep -d',' gz),$(pgrep -d',' ros)

# Monitor network usage
iftop -i lo  # For local communication

# Monitor disk I/O
iotop
```

## Best Practices Summary

### General Optimization Principles

1. **Profile Before Optimizing**: Always measure performance before making changes
2. **Incremental Improvements**: Make small changes and measure impact
3. **Use Appropriate Tools**: Select optimization strategies based on bottlenecks
4. **Balance Quality vs Performance**: Choose settings appropriate for use case

### Common Performance Patterns

- **Physics**: Optimize collision geometry and time steps
- **Sensors**: Reduce resolution and update rates where acceptable
- **Visualization**: Use LOD and culling techniques
- **Communication**: Optimize message frequency and size
- **Memory**: Use efficient data structures and manage buffers

### Performance Testing Workflow

1. Establish baseline performance metrics
2. Identify bottlenecks using profiling tools
3. Apply targeted optimizations
4. Measure improvement quantitatively
5. Repeat for other components

## Troubleshooting Performance Issues

### Common Performance Problems

**Issue**: Simulation runs slower than real-time
- **Check**: Physics time step, solver iterations, collision complexity
- **Solution**: Increase time step, reduce iterations, simplify collisions

**Issue**: High CPU usage
- **Check**: Update rates, number of objects, script efficiency
- **Solution**: Reduce update rates, optimize scripts, reduce complexity

**Issue**: High memory usage
- **Check**: Buffer sizes, history length, data storage
- **Solution**: Optimize buffer sizes, implement data cleanup

By following these optimization strategies, you can achieve the right balance between simulation fidelity and performance for your specific use case. Remember to always validate that optimizations don't compromise the accuracy required for your application.