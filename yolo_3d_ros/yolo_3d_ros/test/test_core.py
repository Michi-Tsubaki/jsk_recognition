import numpy as np

from yolo_3d_ros.core import ExtractionConfig, extract_segmented_instances


def make_grid(height=6, width=8):
    rows, cols = np.indices((height, width), dtype=np.float32)
    xyz = np.stack(
        ((cols - width / 2) * 0.01, (rows - height / 2) * 0.01, np.ones_like(rows)),
        axis=-1,
    )
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 100
    rgb[..., 1] = 150
    rgb[..., 2] = 200
    return xyz, rgb


def test_extracts_compact_points_and_geometry():
    xyz, rgb = make_grid()
    masks = np.zeros((1, 6, 8), dtype=np.float32)
    masks[0, 1:5, 2:7] = 1.0
    objects = extract_segmented_instances(
        xyz,
        rgb,
        masks,
        np.array([5]),
        np.array([0.9]),
        {5: "bus"},
        ExtractionConfig(min_points=2, bbox_percentile=0.0, max_points_per_object=0),
    )
    assert len(objects) == 1
    obj = objects[0]
    assert obj.class_name == "bus"
    assert obj.points_xyz.shape == (20, 3)
    np.testing.assert_allclose(obj.position[2], 1.0)
    np.testing.assert_allclose(obj.bbox_size[2], 1.0e-4)
    assert obj.source_point_count == 20


def test_invalid_and_out_of_range_depth_are_excluded():
    xyz, rgb = make_grid(4, 4)
    xyz[0, 0] = np.nan
    xyz[0, 1, 2] = 99.0
    xyz[0, 2] = 0.0
    masks = np.ones((1, 4, 4), dtype=np.float32)
    objects = extract_segmented_instances(
        xyz,
        rgb,
        masks,
        np.array([0]),
        np.array([1.0]),
        ["person"],
        ExtractionConfig(min_points=1, max_depth_m=5.0, max_points_per_object=0),
    )
    assert len(objects) == 1
    assert objects[0].source_point_count == 13


def test_returns_empty_when_mask_has_too_few_valid_points():
    xyz, rgb = make_grid(3, 3)
    masks = np.zeros((1, 3, 3), dtype=np.float32)
    masks[0, 1, 1] = 1.0
    objects = extract_segmented_instances(
        xyz,
        rgb,
        masks,
        np.array([0]),
        np.array([0.5]),
        ["person"],
        ExtractionConfig(min_points=2),
    )
    assert objects == []


def test_resizes_model_mask_to_camera_resolution():
    xyz, rgb = make_grid(8, 10)
    masks = np.ones((1, 4, 5), dtype=np.float32)
    objects = extract_segmented_instances(
        xyz,
        rgb,
        masks,
        np.array([0]),
        np.array([0.5]),
        ["person"],
        ExtractionConfig(min_points=1, max_points_per_object=0),
    )
    assert objects[0].source_point_count == 80


def test_deterministic_downsampling():
    xyz, rgb = make_grid(20, 20)
    masks = np.ones((1, 20, 20), dtype=np.float32)
    cfg = ExtractionConfig(min_points=1, max_points_per_object=25, random_seed=42)
    first = extract_segmented_instances(
        xyz, rgb, masks, np.array([0]), np.array([1.0]), ["person"], cfg
    )[0]
    second = extract_segmented_instances(
        xyz, rgb, masks, np.array([0]), np.array([1.0]), ["person"], cfg
    )[0]
    assert first.points_xyz.shape[0] == 25
    np.testing.assert_array_equal(first.points_xyz, second.points_xyz)
