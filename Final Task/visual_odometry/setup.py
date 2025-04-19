from setuptools import setup

package_name = 'visual_odometry'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
	    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
	    ('share/' + package_name, ['package.xml']),
	    ('share/' + package_name + '/launch', ['launch/visual_odometry_launch.py']),
	    ('lib/' + package_name, ['visual_odometry/visual_odometry_node.py']),  # Ensure this is correct
	],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='ROS2 package for visual odometry using monocular camera images',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visual_odometry_node = visual_odometry.visual_odometry_node:main',
        ],
    },
)
