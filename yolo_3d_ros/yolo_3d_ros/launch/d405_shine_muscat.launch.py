from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch import LaunchDescription

# COCO-Seg class IDs available in the stock Ultralytics YOLO segmentation model.
# grape, muscat, blueberry, and raspberry are not COCO classes.
SHINE_MUSCAT_PROXY_CLASSES = [
    46,  # banana
    47,  # apple
    49,  # orange
    50,  # broccoli
    58,  # potted plant
]


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("yolo_3d_ros"))
    default_model = share / "models" / "yolo26m-seg.pt"

    config = LaunchConfiguration("config_file")
    input_topic = LaunchConfiguration("input_topic")
    model = LaunchConfiguration("model")
    device = LaunchConfiguration("device")
    confidence = LaunchConfiguration("confidence")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=str(share / "config" / "yolo_3d.yaml"),
            ),
            DeclareLaunchArgument(
                "input_topic",
                default_value="/camera/camera/depth/color/points",
                description="Ordered D405 PointCloud2 topic aligned with color.",
            ),
            DeclareLaunchArgument("model", default_value=str(default_model)),
            DeclareLaunchArgument("device", default_value=""),
            DeclareLaunchArgument(
                "confidence",
                default_value="0.15",
                description="Lower than the generic default to catch weak grape-like detections.",
            ),
            Node(
                package="yolo_3d_ros",
                executable="yolo_3d_node",
                name="yolo_3d_shine_muscat",
                output="screen",
                parameters=[
                    config,
                    {
                        "input_topic": input_topic,
                        "model": model,
                        "device": device,
                        "confidence": ParameterValue(confidence, value_type=float),
                        "classes": SHINE_MUSCAT_PROXY_CLASSES,
                    },
                ],
            ),
        ]
    )
