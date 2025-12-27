from setuptools import find_packages, setup

package_name = 'controller_node'

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
    maintainer='Controller',
    maintainer_email='controller@todo.todo',
    description='Controller nodes for ROS 2 humanoid robotics',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_subscriber = controller_node.joint_subscriber:main',
            'controller_node = controller_node.controller_node:main',
        ],
    },
)