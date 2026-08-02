import struct
from dataclasses import dataclass

import numpy as np
import pytest

from yolo_3d_ros.pointcloud import PointCloudFormatError, pointcloud2_to_array


@dataclass
class Field:
    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass
class Cloud:
    height: int
    width: int
    fields: list[Field]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes


def packed_rgb_float(r: int, g: int, b: int, endian: str = "<") -> float:
    packed = (r << 16) | (g << 8) | b
    return struct.unpack(endian + "f", struct.pack(endian + "I", packed))[0]


def test_packed_rgb_with_row_padding_and_nonfinite_cleanup() -> None:
    fields = [
        Field("x", 0, 7),
        Field("y", 4, 7),
        Field("z", 8, 7),
        Field("rgb", 12, 7),
    ]
    rows = []
    values = [
        [(1.0, 2.0, 3.0, (10, 20, 30)), (4.0, 5.0, 6.0, (40, 50, 60))],
        [(np.inf, 8.0, 9.0, (70, 80, 90)), (10.0, 11.0, 12.0, (1, 2, 3))],
    ]
    for row in values:
        payload = b"".join(
            struct.pack("<ffff", x, y, z, packed_rgb_float(*color))
            for x, y, z, color in row
        )
        rows.append(payload + b"PAD!")
    cloud = Cloud(2, 2, fields, False, 16, 36, b"".join(rows))

    xyz, rgb = pointcloud2_to_array(cloud)
    assert xyz.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, 0], [10, 20, 30])
    np.testing.assert_array_equal(rgb[1, 1], [1, 2, 3])
    np.testing.assert_array_equal(xyz[1, 0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(rgb[1, 0], [0, 0, 0])


def test_separate_rgb_fields() -> None:
    fields = [
        Field("x", 0, 7),
        Field("y", 4, 7),
        Field("z", 8, 7),
        Field("r", 12, 2),
        Field("g", 13, 2),
        Field("b", 14, 2),
    ]
    point = struct.pack("<fffBBB", 1.0, 2.0, 3.0, 4, 5, 6) + b"\x00"
    cloud = Cloud(1, 1, fields, False, 16, 16, point)
    xyz, rgb = pointcloud2_to_array(cloud)
    np.testing.assert_array_equal(xyz[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(rgb[0, 0], [4, 5, 6])


def test_big_endian_packed_rgba() -> None:
    fields = [
        Field("x", 0, 7),
        Field("y", 4, 7),
        Field("z", 8, 7),
        Field("rgba", 12, 6),
    ]
    packed = (255 << 24) | (9 << 16) | (8 << 8) | 7
    point = struct.pack(">fffI", 1.0, 2.0, 3.0, packed)
    cloud = Cloud(1, 1, fields, True, 16, 16, point)
    xyz, rgb = pointcloud2_to_array(cloud)
    np.testing.assert_array_equal(xyz[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(rgb[0, 0], [9, 8, 7])


def test_nan_in_any_coordinate_zeros_xyz_and_rgb() -> None:
    fields = [
        Field("x", 0, 7),
        Field("y", 4, 7),
        Field("z", 8, 7),
        Field("rgb", 12, 7),
    ]
    point = struct.pack("<ffff", 1.0, np.nan, 3.0, packed_rgb_float(9, 8, 7))
    cloud = Cloud(1, 1, fields, False, 16, 16, point)
    xyz, rgb = pointcloud2_to_array(cloud)
    np.testing.assert_array_equal(xyz[0, 0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])


def test_missing_color_raises_clear_error() -> None:
    fields = [Field("x", 0, 7), Field("y", 4, 7), Field("z", 8, 7)]
    cloud = Cloud(1, 1, fields, False, 12, 12, struct.pack("<fff", 1.0, 2.0, 3.0))
    with pytest.raises(PointCloudFormatError, match="Missing color"):
        pointcloud2_to_array(cloud)


def test_short_serialized_buffer_raises_clear_error() -> None:
    fields = [
        Field("x", 0, 7),
        Field("y", 4, 7),
        Field("z", 8, 7),
        Field("rgb", 12, 7),
    ]
    cloud = Cloud(2, 2, fields, False, 16, 32, b"\x00" * 63)
    with pytest.raises(PointCloudFormatError, match="expected at least 64"):
        pointcloud2_to_array(cloud)
