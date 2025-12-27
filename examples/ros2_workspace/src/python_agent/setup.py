from setuptools import find_packages, setup

package_name = 'python_agent'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name,
            ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Agent',
    maintainer_email='agent@todo.todo',
    description='Python agent nodes for ROS 2 humanoid robotics',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_publisher = python_agent.joint_publisher:main',
            'agent_node = python_agent.agent_node:main',
        ],
    },
)