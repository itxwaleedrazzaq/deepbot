import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
    IncludeLaunchDescription,
)
from launch.substitutions import (
    LaunchConfiguration,
    Command,
    PathJoinSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import (
    get_package_share_path,
    get_package_share_directory,
)


def generate_launch_description():

    # ---------------------------------------------------------
    # Launch arguments
    # ---------------------------------------------------------

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='double.world',
        description='Gazebo world to use'
    )

    ai_model_arg = DeclareLaunchArgument(
        'ai_model',
        default_value='NAC',
        description='AI model to use'
    )

    seed_arg = DeclareLaunchArgument(
        'seed',
        default_value='0',
        description='Random seed'
    )

    world = LaunchConfiguration('world')
    ai_model = LaunchConfiguration('ai_model')
    seed = LaunchConfiguration('seed')

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    controller_config = os.path.join(
        get_package_share_directory('deepbot_bringup'),
        'config',
        'diffdrive_params.yaml'
    )

    urdf_path = os.path.join(
        get_package_share_path('deepbot_bringup'),
        'description',
        'main.urdf.xacro'
    )

    world_path = PathJoinSubstitution([
        FindPackageShare('deepbot_bringup'),
        'worlds',
        world
    ])

    robot_description = ParameterValue(
        Command(['xacro ', urdf_path]),
        value_type=str
    )

    # ---------------------------------------------------------
    # AI test node
    # ---------------------------------------------------------

    node_test_ai = Node(
        package='deepbot_test',
        executable='test_ai_node_seed',
        parameters=[
            {'model': ai_model},
            {'seed': seed},
        ],
        output='screen'
    )

    delayed_node_test_ai = TimerAction(
        period=2.0,
        actions=[node_test_ai]
    )

    # ---------------------------------------------------------
    # Trajectory recorder
    # ---------------------------------------------------------

    node_recorder = Node(
        package='deepbot_test',
        executable='trajectory_recorder_node_seed',
        parameters=[
            {'world': world},
            {'model': ai_model},
            {'seed': seed},
        ],
        output='screen'
    )

    delayed_node_recorder = TimerAction(
        period=10.0,
        actions=[node_recorder]
    )

    # ---------------------------------------------------------
    # Robot state publisher
    # ---------------------------------------------------------

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {
                'robot_description': robot_description,
                'use_sim_time': True
            }
        ],
        output='screen'
    )

    delayed_robot_state_publisher = TimerAction(
        period=15.0,
        actions=[node_robot_state_publisher]
    )

    # ---------------------------------------------------------
    # Joint state publisher
    # ---------------------------------------------------------

    node_joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[
            {'use_sim_time': True}
        ],
        output='screen'
    )

    delayed_joint_state_publisher = TimerAction(
        period=15.0,
        actions=[node_joint_state_publisher]
    )

    # ---------------------------------------------------------
    # Gazebo
    # ---------------------------------------------------------

    node_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ]),
        launch_arguments={
            'world': world_path
        }.items()
    )

    delayed_node_gazebo = TimerAction(
        period=20.0,
        actions=[node_gazebo]
    )

    # ---------------------------------------------------------
    # Spawn robot
    # ---------------------------------------------------------

    node_spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic',
            'robot_description',
            '-entity',
            'diffdrive_gazebo_bot'
        ],
        output='screen'
    )

    delayed_node_spawn_entity = TimerAction(
        period=20.0,
        actions=[node_spawn_entity]
    )

    # ---------------------------------------------------------
    # Controller manager
    # ---------------------------------------------------------

    node_controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robot_description},
            controller_config
        ],
        output='screen'
    )

    delayed_controller_manager = TimerAction(
        period=10.0,
        actions=[node_controller_manager]
    )

    # ---------------------------------------------------------
    # Controllers
    # ---------------------------------------------------------

    node_cont_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'diff_cont',
            'joint_broad'
        ],
        output='screen'
    )

    delayed_diff_spawners = TimerAction(
        period=12.0,
        actions=[node_cont_spawner]
    )

    # ---------------------------------------------------------
    # Launch description
    # ---------------------------------------------------------

    return LaunchDescription([
        world_arg,
        ai_model_arg,
        seed_arg,

        delayed_node_recorder,
        delayed_node_test_ai,
        delayed_node_gazebo,
        delayed_joint_state_publisher,

        delayed_node_spawn_entity,
        delayed_controller_manager,
        delayed_diff_spawners,

        
    ])