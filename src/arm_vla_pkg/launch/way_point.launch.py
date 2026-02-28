import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
# IMPORT THIS: Necessary to stop ROS from trying to read URDF as YAML
from launch_ros.parameter_descriptions import ParameterValue 

def generate_launch_description():
    
    # 1. SETUP
    pkg_name = 'bot_description' # Make sure this matches your package name
    xacro_file = os.path.join(get_package_share_directory(pkg_name), 'urdf', 'bot_gz.urdf.xacro')
    
    # 2. PROCESS XACRO (THE FIX IS HERE)
    # We wrap the Command in ParameterValue(..., value_type=str)
    robot_description_content = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)
    
    robot_description = {'robot_description': robot_description_content}

    # 3. NODES
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    node_joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        # arguments=['-d', os.path.join(get_package_share_directory(pkg_name), 'config', 'view_bot.rviz')]
    )

    return LaunchDescription([
        node_robot_state_publisher,
        node_joint_state_publisher_gui,
        node_rviz
    ])