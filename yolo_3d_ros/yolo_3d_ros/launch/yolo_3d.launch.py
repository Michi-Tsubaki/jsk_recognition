from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("yolo_3d_ros"))
    default_model = share / "models" / "yolo26m-seg.pt"
    config = LaunchConfiguration("config_file")
    input_topic = LaunchConfiguration("input_topic")
    model = LaunchConfiguration("model")
    device = LaunchConfiguration("device")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=str(share / "config" / "yolo_3d.yaml"),
            ),
            DeclareLaunchArgument(
                "input_topic",
                default_value="/camera/camera/depth/color/points",
            ),
            DeclareLaunchArgument("model", default_value=str(default_model)),
            DeclareLaunchArgument("device", default_value=""),
            Node(
                package="yolo_3d_ros",
                executable="yolo_3d_node",
                name="yolo_3d",
                output="screen",
                parameters=[
                    config,
                    {
                        "input_topic": input_topic,
                        "model": model,
                        "device": device,
                    },
                ],
            ),
        ]
    )
