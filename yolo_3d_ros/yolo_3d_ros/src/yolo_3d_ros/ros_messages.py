"""Construction of ROS messages from ROS-independent segmentation results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
from vision_msgs.msg import (
    BoundingBox3DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from yolo_3d_msgs.msg import SegmentedObject3D, SegmentedObject3DArray

from yolo_3d_ros.core import SegmentedInstance


def _packed_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb32 = np.asarray(rgb, dtype=np.uint32)
    return (rgb32[:, 0] << 16) | (rgb32[:, 1] << 8) | rgb32[:, 2]


def colored_pointcloud2(header, xyz: np.ndarray, rgb: np.ndarray) -> PointCloud2:
    """Create a compact unorganized XYZRGB PointCloud2."""

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must contain the same number of points")

    packed = np.empty(
        xyz.shape[0],
        dtype=np.dtype(
            {
                "names": ["x", "y", "z", "rgb"],
                "formats": ["<f4", "<f4", "<f4", "<u4"],
                "offsets": [0, 4, 8, 12],
                "itemsize": 16,
            }
        ),
    )
    packed["x"], packed["y"], packed["z"] = xyz.T
    packed["rgb"] = _packed_rgb(rgb)

    message = PointCloud2()
    message.header = header
    message.height = 1
    message.width = xyz.shape[0]
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        # PCL convention: RGB bits are carried in a field declared FLOAT32.
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = message.point_step * message.width
    message.is_dense = True
    message.data = packed.tobytes()
    return message


def detection3d(header, instance: SegmentedInstance) -> Detection3D:
    """Convert one extracted instance to a standards-based 3D detection.

    ``Detection3D.id`` is intentionally left empty because this package performs
    per-frame segmentation, not temporal object tracking.
    """

    detection = Detection3D()
    detection.header = header

    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = instance.class_name
    hypothesis.hypothesis.score = float(instance.score)
    hypothesis.pose.pose.position.x = float(instance.position[0])
    hypothesis.pose.pose.position.y = float(instance.position[1])
    hypothesis.pose.pose.position.z = float(instance.position[2])
    hypothesis.pose.pose.orientation.w = 1.0
    # Pose covariance is left at its all-zero default: this package estimates a
    # representative point, but does not estimate statistical pose uncertainty.
    detection.results = [hypothesis]

    detection.bbox.center.position.x = float(instance.bbox_center[0])
    detection.bbox.center.position.y = float(instance.bbox_center[1])
    detection.bbox.center.position.z = float(instance.bbox_center[2])
    detection.bbox.center.orientation.w = 1.0
    detection.bbox.size.x = float(instance.bbox_size[0])
    detection.bbox.size.y = float(instance.bbox_size[1])
    detection.bbox.size.z = float(instance.bbox_size[2])
    return detection


def merged_labeled_pointcloud2(header, instances: Sequence[SegmentedInstance]) -> PointCloud2:
    """Merge all object points and retain instance/class/score fields per point."""

    total = sum(instance.points_xyz.shape[0] for instance in instances)
    dtype = np.dtype(
        {
            "names": [
                "x",
                "y",
                "z",
                "rgb",
                "instance_id",
                "class_id",
                "confidence",
            ],
            "formats": ["<f4", "<f4", "<f4", "<u4", "<u4", "<i4", "<f4"],
            "offsets": [0, 4, 8, 12, 16, 20, 24],
            "itemsize": 28,
        }
    )
    packed = np.empty(total, dtype=dtype)
    cursor = 0
    for output_index, instance in enumerate(instances):
        size = instance.points_xyz.shape[0]
        target = packed[cursor : cursor + size]
        target["x"], target["y"], target["z"] = instance.points_xyz.T
        target["rgb"] = _packed_rgb(instance.colors_rgb)
        target["instance_id"] = output_index
        target["class_id"] = instance.class_id
        target["confidence"] = instance.score
        cursor += size

    message = PointCloud2()
    message.header = header
    message.height = 1
    message.width = total
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="instance_id", offset=16, datatype=PointField.UINT32, count=1),
        PointField(name="class_id", offset=20, datatype=PointField.INT32, count=1),
        PointField(name="confidence", offset=24, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 28
    message.row_step = message.point_step * message.width
    message.is_dense = True
    message.data = packed.tobytes()
    return message


def build_output_messages(
    header,
    instances: Sequence[SegmentedInstance],
) -> tuple[Detection3DArray, SegmentedObject3DArray, PointCloud2]:
    detections = Detection3DArray()
    detections.header = header
    segmented = SegmentedObject3DArray()
    segmented.header = header

    for instance in instances:
        detection = detection3d(header, instance)
        detections.detections.append(detection)
        item = SegmentedObject3D()
        item.detection = detection
        item.point_cloud = colored_pointcloud2(
            header,
            instance.points_xyz,
            instance.colors_rgb,
        )
        segmented.objects.append(item)

    return detections, segmented, merged_labeled_pointcloud2(header, instances)


def bounding_box3d_array(header, instances: Sequence[SegmentedInstance]) -> BoundingBox3DArray:
    boxes = BoundingBox3DArray()
    boxes.header = header

    for instance in instances:
        box = detection3d(header, instance).bbox
        boxes.boxes.append(box)

    return boxes


def bgr_image_message(header, image: np.ndarray) -> Image:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("debug image must have shape (H,W,3)")
    message = Image()
    message.header = header
    message.height = image.shape[0]
    message.width = image.shape[1]
    message.encoding = "bgr8"
    message.is_bigendian = False
    message.step = image.shape[1] * 3
    message.data = image.tobytes()
    return message
