"""ROS-installed integration checks for generated messages and PointCloud2 layouts."""

import numpy as np
from std_msgs.msg import Header
from yolo_3d_msgs.msg import SegmentedObject3DArray

from yolo_3d_ros.core import SegmentedInstance
from yolo_3d_ros.pointcloud import pointcloud2_to_array
from yolo_3d_ros.ros_messages import bounding_box3d_array, build_output_messages


def _instance(index: int, class_id: int, name: str) -> SegmentedInstance:
    points = np.array(
        [[0.1 + index, 0.2, 1.0], [0.2 + index, 0.3, 1.1]],
        dtype=np.float32,
    )
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    return SegmentedInstance(
        class_id=class_id,
        class_name=name,
        score=0.8,
        instance_index=index,
        points_xyz=points,
        colors_rgb=colors,
        position=np.median(points, axis=0).astype(np.float32),
        position_variance=np.var(points, axis=0).astype(np.float32),
        bbox_center=np.mean(points, axis=0).astype(np.float32),
        bbox_size=np.ptp(points, axis=0).clip(1.0e-4).astype(np.float32),
        pixel_mask=np.ones((2, 2), dtype=bool),
        source_point_count=2,
    )


def test_detection_and_segmented_cloud_arrays_have_identical_order() -> None:
    header = Header()
    header.frame_id = "camera_color_optical_frame"
    header.stamp.sec = 12
    header.stamp.nanosec = 34
    instances = [_instance(0, 0, "person"), _instance(1, 5, "bus")]

    detections, segmented, merged = build_output_messages(header, instances)

    assert isinstance(segmented, SegmentedObject3DArray)
    assert len(detections.detections) == len(segmented.objects) == 2
    assert detections.detections[0].results[0].hypothesis.class_id == "person"
    assert detections.detections[0].id == ""
    assert segmented.objects[1].detection.results[0].hypothesis.class_id == "bus"
    assert segmented.objects[0].point_cloud.width == 2
    assert merged.width == 4
    assert [field.name for field in merged.fields] == [
        "x",
        "y",
        "z",
        "rgb",
        "instance_id",
        "class_id",
        "confidence",
    ]

    # The per-object PointCloud2 payload reconstructs to the actual segmented
    # NumPy point/color arrays, rather than to a zero-masked full camera image.
    xyz, rgb = pointcloud2_to_array(segmented.objects[0].point_cloud)
    assert xyz.shape == rgb.shape == (1, 2, 3)
    np.testing.assert_allclose(xyz.reshape(-1, 3), instances[0].points_xyz)
    np.testing.assert_array_equal(rgb.reshape(-1, 3), instances[0].colors_rgb)

    position = detections.detections[0].results[0].pose.pose.position
    np.testing.assert_allclose(
        [position.x, position.y, position.z],
        instances[0].position,
    )

    boxes = bounding_box3d_array(header, instances)
    assert boxes.header.frame_id == header.frame_id
    assert len(boxes.boxes) == 2
    np.testing.assert_allclose(
        [
            boxes.boxes[1].center.position.x,
            boxes.boxes[1].center.position.y,
            boxes.boxes[1].center.position.z,
        ],
        instances[1].bbox_center,
    )
