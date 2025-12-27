from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os


def generate_launch_description():
    """Launch Gazebo with a basic humanoid robot model."""

    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='basic_humanoid.sdf',
        description='Choose one of the world files from `/examples/gazebo/worlds`'
    )

    # Define paths using absolute paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up to project root
    world_path = os.path.join(project_root, 'examples', 'gazebo', 'worlds', 'basic_humanoid.sdf')
    urdf_path = os.path.join(project_root, 'examples', 'gazebo', 'models', 'humanoid', 'basic_humanoid.urdf')

    # Launch Gazebo with the world
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path],
        output='screen'
    )

    # Launch robot state publisher to publish URDF
    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_desc
        }]
    )

    # Launch joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_gui': False,
            'rate': 50.0
        }]
    )

    # Declare additional launch arguments
    use_gui_arg = DeclareLaunchArgument(
        'use_gui',
        default_value='false',
        description='Whether to use joint_state_publisher_gui'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'
    )

    # Launch joint state publisher GUI (optional)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[{'use_gui': LaunchConfiguration('use_gui')}]
    )

    return LaunchDescription([
        world_arg,
        use_gui_arg,
        use_rviz_arg,
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui
    ])