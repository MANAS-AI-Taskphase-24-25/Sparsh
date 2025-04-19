from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_dir',
            #default_value='/media/sparsh/Windows/Mystudymaterial/Manas/FinalTask/dataset/images/mav0/cam0/data1/',
            default_value="/media/sparsh/Windows/Mystudymaterial/data_odometry_gray/dataset/sequences/02/image_0/",
            description='Path to directory containing image files'
        ),
        DeclareLaunchArgument(
            'focal',
            default_value='718.856',
            description='Camera focal length in pixels'
        ),
        DeclareLaunchArgument(
            'pp_x',
            default_value='607.1928',
            description='Camera principal point x-coordinate'
        ),
        DeclareLaunchArgument(
            'pp_y',
            default_value='185.2157',
            description='Camera principal point y-coordinate'
        ),
        DeclareLaunchArgument(
            'show_trajectory',
            default_value='true',
            description='Whether to display the trajectory window'
        ),
        DeclareLaunchArgument(
            'frame_rate',
            default_value='10.0',
            description='Rate at which to process images (Hz)'
        ),
        DeclareLaunchArgument(
            'path_offset_x',
            default_value='0.0',
            description='X offset for path visualization in RViz'
        ),
        DeclareLaunchArgument(
            'path_offset_z',
            default_value='0.0',
            description='Z offset for path visualization in RViz'
        ),
        Node(
            package='visual_odometry',
            executable='visual_odometry_node',
            name='visual_odometry_node',
            output='screen',
            parameters=[
                {'image_dir': LaunchConfiguration('image_dir')},
                {'focal': LaunchConfiguration('focal')},
                {'pp_x': LaunchConfiguration('pp_x')},
                {'pp_y': LaunchConfiguration('pp_y')},
                {'show_trajectory': LaunchConfiguration('show_trajectory')},
                {'frame_rate': LaunchConfiguration('frame_rate')},
                {'path_offset_x': LaunchConfiguration('path_offset_x')},
                {'path_offset_z': LaunchConfiguration('path_offset_z')},
            ]
        ),
    ])