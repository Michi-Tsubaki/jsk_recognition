"""Conversion utilities for organized ``sensor_msgs/PointCloud2`` messages.

The implementation intentionally does not depend on ``ros_numpy``.  It understands
row padding, little/big endian messages, packed ``rgb``/``rgba`` fields, and separate
``r``/``g``/``b`` fields.  This also keeps the conversion logic unit-testable without
requiring a ROS installation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PointCloudFormatError(ValueError):
    """Raised when a PointCloud2 message cannot be interpreted as organized XYZRGB."""


# sensor_msgs/msg/PointField constants, duplicated to keep this module ROS-independent.
_POINT_FIELD_DTYPES: dict[int, str] = {
    1: "i1",  # INT8
    2: "u1",  # UINT8
    3: "i2",  # INT16
    4: "u2",  # UINT16
    5: "i4",  # INT32
    6: "u4",  # UINT32
    7: "f4",  # FLOAT32
    8: "f8",  # FLOAT64
}


def _structured_dtype(fields: list[Any], point_step: int, is_bigendian: bool) -> np.dtype:
    endian = ">" if is_bigendian else "<"
    names: list[str] = []
    formats: list[Any] = []
    offsets: list[int] = []

    for field in fields:
        datatype = int(field.datatype)
        if datatype not in _POINT_FIELD_DTYPES:
            raise PointCloudFormatError(
                f"Unsupported PointField datatype {datatype} for field {field.name!r}"
            )
        count = int(getattr(field, "count", 1))
        scalar = np.dtype(endian + _POINT_FIELD_DTYPES[datatype])
        field_format: Any = scalar if count == 1 else (scalar, (count,))
        names.append(str(field.name))
        formats.append(field_format)
        offsets.append(int(field.offset))

    try:
        return np.dtype(
            {
                "names": names,
                "formats": formats,
                "offsets": offsets,
                "itemsize": int(point_step),
            }
        )
    except (TypeError, ValueError) as exc:
        raise PointCloudFormatError(f"Invalid PointCloud2 field layout: {exc}") from exc


def _packed_rgb_to_uint8(values: np.ndarray, is_bigendian: bool) -> np.ndarray:
    """Decode PCL-style 0x00RRGGBB values stored as FLOAT32 or UINT32."""

    endian = ">" if is_bigendian else "<"
    if values.dtype.kind == "f" and values.dtype.itemsize == 4:
        packed = np.array(values, dtype=np.dtype(endian + "f4"), copy=True).view(
            np.dtype(endian + "u4")
        )
    else:
        packed = np.asarray(values, dtype=np.dtype(endian + "u4"))

    # NumPy transparently converts non-native endian integers for arithmetic.
    r = ((packed >> 16) & 0xFF).astype(np.uint8)
    g = ((packed >> 8) & 0xFF).astype(np.uint8)
    b = (packed & 0xFF).astype(np.uint8)
    return np.stack((r, g, b), axis=-1)


def pointcloud2_to_array(pointcloud2: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert an organized PointCloud2 message to ``xyz`` and ``rgb`` arrays.

    Args:
        pointcloud2: A ``sensor_msgs.msg.PointCloud2`` instance, or a duck-typed
            object exposing the same fields.

    Returns:
        ``(xyz, rgb)`` where ``xyz`` has shape ``(height, width, 3)`` and dtype
        ``float32``, while ``rgb`` has the same shape and dtype ``uint8`` in RGB
        channel order.

    Raises:
        PointCloudFormatError: If XYZ or color fields are missing, dimensions are
            invalid, or the serialized buffer is too short.

    Notes:
        A point is considered invalid if *any* XYZ component is NaN or infinity.
        Its XYZ and RGB values are both replaced by zero before returning.
    """

    height = int(pointcloud2.height)
    width = int(pointcloud2.width)
    point_step = int(pointcloud2.point_step)
    row_step = int(pointcloud2.row_step) or point_step * width

    if height <= 0 or width <= 0:
        raise PointCloudFormatError(
            f"PointCloud2 dimensions must be positive, got {width}x{height}"
        )
    if point_step <= 0:
        raise PointCloudFormatError(f"point_step must be positive, got {point_step}")
    if row_step < point_step * width:
        raise PointCloudFormatError(
            f"row_step={row_step} is shorter than width*point_step={point_step * width}"
        )

    raw = memoryview(bytes(pointcloud2.data))
    required_bytes = row_step * height
    if len(raw) < required_bytes:
        raise PointCloudFormatError(
            f"PointCloud2 data has {len(raw)} bytes, expected at least {required_bytes}"
        )

    dtype = _structured_dtype(
        list(pointcloud2.fields), point_step, bool(pointcloud2.is_bigendian)
    )
    cloud = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=raw,
        strides=(row_step, point_step),
    )
    field_names = set(dtype.names or ())

    missing_xyz = {"x", "y", "z"} - field_names
    if missing_xyz:
        raise PointCloudFormatError(f"Missing required XYZ field(s): {sorted(missing_xyz)}")

    xyz = np.stack(
        (
            np.asarray(cloud["x"], dtype=np.float32),
            np.asarray(cloud["y"], dtype=np.float32),
            np.asarray(cloud["z"], dtype=np.float32),
        ),
        axis=-1,
    )

    if "rgb" in field_names:
        rgb = _packed_rgb_to_uint8(cloud["rgb"], bool(pointcloud2.is_bigendian))
    elif "rgba" in field_names:
        rgb = _packed_rgb_to_uint8(cloud["rgba"], bool(pointcloud2.is_bigendian))
    elif {"r", "g", "b"}.issubset(field_names):
        rgb = np.stack(
            tuple(
                np.clip(np.asarray(cloud[channel]), 0, 255).astype(np.uint8)
                for channel in ("r", "g", "b")
            ),
            axis=-1,
        )
    else:
        raise PointCloudFormatError(
            "Missing color data: expected packed 'rgb'/'rgba' or separate 'r', 'g', 'b' fields"
        )

    invalid = ~np.isfinite(xyz).all(axis=2)
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    xyz[invalid] = 0.0
    rgb[invalid] = 0
    return xyz, rgb
