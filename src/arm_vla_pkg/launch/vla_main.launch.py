#!/usr/bin/env python3
"""
VLA Complete System Launch
Based on working final_arm.launch.py + VLA nodes
"""
import os
import yaml
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def generate_launch_description():
    
    # SETUP PATHS
    pkg_gazebo = FindPackageShare('akabot_gazebo')
    pkg_description = FindPackageShare('akabot_description')
    pkg_moveit = FindPackageShare('akabot_moveit_config')
    pkg_ros_ign_gazebo = FindPackageShare('ros_ign_gazebo')

    world_path = PathJoinSubstitution([pkg_gazebo, 'worlds', 'pick_and_place_balls.world'])
    srdf_file = PathJoinSubstitution([pkg_moveit, 'srdf', 'akabot.srdf'])
    robot_description_file = PathJoinSubstitution([pkg_description, 'urdf', 'akabot_gz.urdf.xacro'])
    
    # LOAD CONFIGURATIONS
    robot_description_content = Command(
        [PathJoinSubstitution([FindExecutable(name='xacro')]), ' ', robot_description_file]
    )
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}
    
    robot_description_semantic_content = Command(
        [PathJoinSubstitution([FindExecutable(name='xacro')]), ' ', srdf_file]
    )
    robot_description_semantic = {'robot_description_semantic': ParameterValue(robot_description_semantic_content, value_type=str)}
    
    kinematics_config = load_yaml('akabot_moveit_config', 'config/kinematics.yaml')
    moveit_controllers = load_yaml('akabot_moveit_config', 'config/moveit_controllers.yaml')

    # ROBOT STATE PUBLISHER
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}]
    )

    # LAUNCH GAZEBO
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_ros_ign_gazebo, 'launch', 'ign_gazebo.launch.py'])
        ),
        launch_arguments={'gz_args': ['-r -v4 ', world_path]}.items(),
    )

    # SPAWN ROBOT
    spawn_robot = Node(
        package='ros_ign_gazebo',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'akabot',
            '-x', '-0.155216', '-y', '-0.056971', '-z', '1.05', '-Y', '0.016798'
        ],
        output='screen'
    )

    # BRIDGES
    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/tf_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/ee_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
        ],
        output='screen'
    )

    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/tf_camera/image_raw', '/ee_camera/image_raw'],
        output='screen'
    )

    # CONTROLLERS
    joint_state_broadcaster = TimerAction(
        period=5.0,
        actions=[Node(package='controller_manager', executable='spawner',
                      arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'])]
    )
    
    arm_controller = TimerAction(
        period=8.0,
        actions=[Node(package='controller_manager', executable='spawner',
                      arguments=['akabot_arm_controller', '--controller-manager', '/controller_manager'])]
    )
    
    hand_controller = TimerAction(
        period=12.0,
        actions=[Node(package='controller_manager', executable='spawner',
                      arguments=['hand_controller', '--controller-manager', '/controller_manager'])]
    )

    # MOVEIT
    move_group_node = TimerAction(
        period=15.0,
        actions=[Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                robot_description,
                robot_description_semantic,
                kinematics_config,
                moveit_controllers,
                {
                    'use_sim_time': True,
                    'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
                    'moveit_manage_controllers': True
                }
            ],
            arguments=['--ros-args', '--log-level', 'info']
        )]
    )

    # VISION SYSTEM (publish_tf from akabot_control)
    # Increased delay to allow system stabilization
    publish_tf = TimerAction(
        period=21.0,
        actions=[Node(
            package='akabot_control',
            executable='publish_tf',
            name='publish_tf',
            output='screen',
            parameters=[{'use_sim_time': True}]
        )]
    )

    # ====================
    # VLA PIPELINE NODES
    # ====================
    
    # VLA Vision Node (TF → JSON converter)
    # Wait extra time for publish_tf to start detecting objects
    vla_vision = TimerAction(
        period=30.0,  # Increased from 25 to 30
        actions=[Node(
            package='arm_vla_pkg',
            executable='vision_node',
            name='vla_vision',
            output='screen',
            parameters=[{'use_sim_time': True}]
        )]
    )
    
    # VLA Brain Node (Ollama LLM)
    vla_brain = TimerAction(
        period=31.0,  # Increased from 26 to 31
        actions=[Node(
            package='arm_vla_pkg',
            executable='brain_node',
            name='vla_brain',
            output='screen',
            parameters=[{'use_sim_time': True}]
        )]
    )
    
    # VLA Action Node (MoveIt execution)
    vla_action = TimerAction(
        period=32.0,  # Increased from 27 to 32
        actions=[Node(
            package='arm_vla_pkg',
            executable='action_node',
            name='vla_action',
            output='screen',
            parameters=[{'use_sim_time': True}]
        )]
    )
    
    # VLA Speech Node (voice input) - Optional, start manually if needed
    # Commented out by default since it requires microphone
    # vla_speech = TimerAction(
    #     period=28.0,
    #     actions=[Node(
    #         package='arm_vla_pkg',
    #         executable='speech_node',
    #         name='vla_speech',
    #         output='screen',
    #         parameters=[{'use_sim_time': True}]
    #     )]
    # )

    return LaunchDescription([
        # Core System (from final_arm.launch.py)
        gazebo_launch,
        robot_state_publisher,
        spawn_robot,
        parameter_bridge,
        image_bridge,
        joint_state_broadcaster,
        arm_controller,
        hand_controller,
        move_group_node,
        publish_tf,
        
        # VLA Pipeline
        vla_vision,
        vla_brain,
        vla_action,
        # vla_speech,  # Uncomment to enable voice control
    ])
