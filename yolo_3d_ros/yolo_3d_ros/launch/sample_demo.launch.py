from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("yolo_3d_ros"))
    default_model = share / "models" / "yolo26m-seg.pt"
    model = LaunchConfiguration("model")
    device = LaunchConfiguration("device")
    sequence = LaunchConfiguration("sequence")

    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value=str(default_model)),
            DeclareLaunchArgument("device", default_value=""),
            DeclareLaunchArgument(
                "sequence",
                default_value=str(share / "docs" / "sample_rgbd_3s.npz"),
            ),
            Node(
                package="yolo_3d_ros",
                executable="sample_pointcloud_publisher",
                name="sample_pointcloud_publisher",
                output="screen",
                parameters=[
                    {
                        "sequence_path": sequence,
                        "topic": "/camera/depth/points",
                        "loop": True,
                    }
                ],
            ),
            Node(
                package="yolo_3d_ros",
                executable="yolo_3d_node",
                name="yolo_3d",
                output="screen",
                parameters=[
                    str(share / "config" / "yolo_3d.yaml"),
                    {
                        "input_topic": "/camera/depth/points",
                        "model": model,
                        "device": device,
                        "min_points": 30,
                    },
                ],
            ),
        ]
    )
