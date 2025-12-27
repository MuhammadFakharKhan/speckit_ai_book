# Troubleshooting Guide: Module 2 - Digital Twin Simulation

This guide provides solutions to common issues encountered when setting up and running the Gazebo-Unity digital twin simulation environment.

## Common Installation Issues

### ROS 2 Installation Problems

**Issue**: `ros2` command not found after installation
- **Solution**:
  1. Verify ROS 2 installation: `ls /opt/ros/humble/`
  2. Source the setup file: `source /opt/ros/humble/setup.bash`
  3. Add to your shell profile: `echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc`

**Issue**: Permission denied when installing ROS 2 packages
- **Solution**: Use proper installation commands with sudo where needed:
  ```bash
  sudo apt update
  sudo apt install ros-humble-desktop
  ```

### Gazebo Installation Issues

**Issue**: Gazebo Garden doesn't start or crashes immediately
- **Solution**:
  1. Check graphics drivers: `lspci | grep -i vga`
  2. Ensure proper X11 forwarding if using Docker
  3. Try running with software rendering: `export LIBGL_ALWAYS_SOFTWARE=1`

**Issue**: Gazebo plugins fail to load
- **Solution**:
  1. Verify Gazebo installation: `gz --versions`
  2. Check plugin paths: `echo $GAZEBO_PLUGIN_PATH`
  3. Install missing dependencies: `sudo apt install ros-humble-gazebo-*`

### Unity Installation Issues

**Issue**: Unity Hub or Unity Editor fails to install
- **Solution**:
  1. Ensure system meets minimum requirements
  2. Check available disk space (minimum 20GB recommended)
  3. Disable antivirus temporarily during installation
  4. Use Unity LTS version (2022.3.x) for stability

## Simulation Runtime Issues

### Physics Simulation Problems

**Issue**: Robot falls through the ground or exhibits unstable physics
- **Solution**:
  1. Check physics configuration in `examples/gazebo/config/physics.yaml`
  2. Reduce time step (try 0.0005 instead of 0.001)
  3. Adjust solver parameters (iterations, erp, cfm)
  4. Verify model collision geometry

**Issue**: Simulation runs too slowly
- **Solution**:
  1. Switch to performance profile: `POST /api/profiles/performance/apply`
  2. Reduce physics update rate in configuration
  3. Simplify collision geometry
  4. Reduce solver iterations

### Sensor Simulation Issues

**Issue**: Sensor data is not publishing or shows zero values
- **Solution**:
  1. Check sensor bridge configuration: `examples/gazebo/config/sensor_bridge.yaml`
  2. Verify topic names match between Gazebo and ROS 2
  3. Check sensor plugin is properly attached to model
  4. Verify sensor parameters (update rate, range, etc.)

**Issue**: Camera images are black or distorted
- **Solution**:
  1. Check camera sensor configuration in model SDF
  2. Verify image format and resolution settings
  3. Check lighting in the simulation environment
  4. Adjust camera parameters (fov, clipping planes)

### ROS 2 Communication Issues

**Issue**: ROS 2 nodes cannot communicate with Gazebo
- **Solution**:
  1. Verify both use same RMW implementation: `echo $RMW_IMPLEMENTATION`
  2. Check network configuration and firewall settings
  3. Ensure both ROS 2 and Gazebo use same domain ID
  4. Verify ros-gz bridge is running: `ros2 run ros_gz_bridge parameter_bridge`

**Issue**: Joint states not updating or incorrect
- **Solution**:
  1. Check joint state publisher configuration
  2. Verify joint names match between URDF and controller
  3. Check controller configuration and parameters
  4. Ensure proper TF tree is published

## Unity Integration Issues

### Connection Problems

**Issue**: Unity cannot connect to ROS 2 network
- **Solution**:
  1. Install Unity ROS TCP Connector package
  2. Ensure ROS IP and port are correctly configured in Unity
  3. Check firewall settings for Unity application
  4. Verify ROS bridge is running: `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`

**Issue**: Robot state does not synchronize between Gazebo and Unity
- **Solution**:
  1. Check synchronization frequency and parameters
  2. Verify transform coordinate system compatibility
  3. Check for timing issues or dropped messages
  4. Adjust interpolation settings in synchronizer

### Visualization Issues

**Issue**: Robot appears distorted or with incorrect proportions in Unity
- **Solution**:
  1. Verify scale factor between Gazebo and Unity coordinate systems
  2. Check model import settings in Unity
  3. Verify joint mapping between ROS and Unity
  4. Check for unit conversion issues (meters vs. Unity units)

## API and Development Issues

### API Endpoints Not Working

**Issue**: API server fails to start
- **Solution**:
  1. Check if port 5001 is available: `netstat -tuln | grep 5001`
  2. Verify Flask and required dependencies are installed
  3. Check for import errors in the API modules
  4. Run with debug mode to see detailed error messages

**Issue**: API requests return 404 or 500 errors
- **Solution**:
  1. Verify correct endpoint URLs (check API documentation)
  2. Check request format and content-type headers
  3. Ensure API server is running and accessible
  4. Check server logs for specific error details

### Documentation Issues

**Issue**: Docusaurus site fails to build
- **Solution**:
  1. Check Node.js version (18+ required): `node --version`
  2. Clean and reinstall dependencies: `rm -rf node_modules && npm install`
  3. Clear Docusaurus cache: `npx docusaurus clear`
  4. Check for syntax errors in Markdown files

## Performance Optimization

### Slow Simulation Performance

**Issue**: Simulation runs below real-time factor
- **Solution**:
  1. Apply performance profile via API: `POST /api/profiles/performance/apply`
  2. Reduce physics complexity (simpler collision meshes)
  3. Lower sensor update rates
  4. Reduce solver iterations and constraint parameters

### High Memory Usage

**Issue**: Simulation consumes excessive memory
- **Solution**:
  1. Reduce simulation world complexity
  2. Limit history buffer sizes for trajectories
  3. Reduce image resolution for cameras
  4. Use compressed image transport

## Development Environment Issues

### Python Environment Problems

**Issue**: Python packages not found or import errors
- **Solution**:
  1. Create and activate virtual environment
  2. Install required packages: `pip install -r requirements.txt`
  3. Verify Python path includes src directory
  4. Check for conflicting package versions

### Build and Compilation Issues

**Issue**: Compilation errors when building custom plugins
- **Solution**:
  1. Verify proper ROS 2 workspace setup
  2. Check package.xml and CMakeLists.txt files
  3. Ensure correct ROS 2 package dependencies
  4. Source ROS 2 environment before building: `source install/setup.bash`

## Network and Communication Issues

### Docker Container Issues

**Issue**: Gazebo or ROS 2 doesn't work properly in Docker
- **Solution**:
  1. Enable X11 forwarding: `xhost +local:docker` and `--env="DISPLAY"`
  2. Use host network mode: `--network=host`
  3. Mount necessary volumes for shared memory
  4. Check GPU passthrough for rendering

### Multi-Machine Setup Issues

**Issue**: Cannot connect simulation components across machines
- **Solution**:
  1. Ensure all machines are on same network
  2. Configure ROS_DOMAIN_ID consistently
  3. Check firewall rules for required ports
  4. Verify network discovery protocols

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs**: Look at detailed error messages in console output
2. **Verify versions**: Ensure compatible versions of ROS 2, Gazebo, and Unity
3. **Search existing issues**: Check GitHub repositories for similar problems
4. **Create minimal reproduction**: Simplify the problem to isolate the issue
5. **Ask for help**: Provide detailed information about your setup and the specific error

### Useful Commands for Debugging

```bash
# Check ROS 2 network
ros2 topic list
ros2 service list
ros2 node list

# Check Gazebo status
gz topic -l
gz stats

# Check system resources
htop
df -h
free -h

# Check API server
curl -X GET http://localhost:5001/api/profiles
```

### Log Files Locations

- ROS 2 logs: `~/.ros/log/`
- Gazebo logs: `~/.gz/gazebo/log/`
- Unity logs: `~/Library/Logs/Unity/` (macOS) or `C:\Users\%USERNAME%\AppData\LocalLow\` (Windows)
- Docusaurus logs: Console output when running `npm start`

Remember to restart services after making configuration changes, and always test changes incrementally to isolate issues.