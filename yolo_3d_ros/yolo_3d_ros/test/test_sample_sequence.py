import hashlib
from pathlib import Path

import numpy as np

from yolo_3d_ros.core import ExtractionConfig, extract_segmented_instances
from yolo_3d_ros.sample_data import depth_to_xyz, load_rgbd_sequence

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "docs" / "sample_rgbd_3s.npz"

COLOR_SOURCE = ROOT / "docs" / "sample_source" / "frame-000000.color.jpg"
DEPTH_SOURCE = ROOT / "docs" / "sample_source" / "frame-000000.depth.png"
EXPECTED_COLOR_SHA256 = "c9b3b20cab6472f7f31402a7e3ba3274ccf892940d9de3d584dff60babb99d07"
EXPECTED_DEPTH_SHA256 = "b8dbee19bfebde9b57bee5dfbb1226dc12d59d0b1af43e74204bd82c2329008e"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _smoke_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    with np.load(SAMPLE, allow_pickle=False) as data:
        masks = np.asarray(data["smoke_masks"], dtype=np.float32)
        class_ids = np.asarray(data["smoke_class_ids"], dtype=np.int64)
        scores = np.asarray(data["smoke_scores"], dtype=np.float32)
        names = np.asarray(data["smoke_names"]).astype(str)
    name_map = {
        int(class_id): str(name)
        for class_id, name in zip(class_ids, names, strict=True)
    }
    return masks, class_ids, scores, name_map


def test_bundled_sequence_is_exactly_three_seconds() -> None:
    sequence = load_rgbd_sequence(SAMPLE)
    assert sequence.frame_count == 30
    assert sequence.fps == 10.0
    assert sequence.duration_seconds == 3.0
    assert sequence.depth_m.shape == (30, 240, 320)
    assert sequence.rgb_frame(0).shape == (240, 320, 3)
    assert "BundleFusion copyroom" in sequence.source


def test_bundled_provenance_metadata_and_source_hashes() -> None:
    with np.load(SAMPLE, allow_pickle=False) as data:
        assert str(np.asarray(data["source_license"]).item()) == "CC-BY-NC-SA-4.0"
        assert str(np.asarray(data["source_color_url"]).item()).endswith(
            "/pic/frame-000000.color.jpg"
        )
        assert str(np.asarray(data["source_depth_url"]).item()).endswith(
            "/pic/frame-000000.depth.png"
        )
        assert str(np.asarray(data["source_color_sha256"]).item()) == EXPECTED_COLOR_SHA256
        assert str(np.asarray(data["source_depth_sha256"]).item()) == EXPECTED_DEPTH_SHA256
        translations = np.asarray(data["depth_translation_m"], dtype=np.float32)

    assert translations.shape == (30,)
    np.testing.assert_allclose(translations[0], 0.0, atol=1.0e-7)
    assert float(translations.max()) > 0.019
    assert float(translations.min()) < -0.019
    assert _sha256(COLOR_SOURCE) == EXPECTED_COLOR_SHA256
    assert _sha256(DEPTH_SOURCE) == EXPECTED_DEPTH_SHA256


def test_all_30_frames_run_through_mask_to_3d_geometry() -> None:
    sequence = load_rgbd_sequence(SAMPLE)
    masks, class_ids, scores, names = _smoke_metadata()
    config = ExtractionConfig(min_points=30, max_points_per_object=500, random_seed=7)
    positions_over_time: list[np.ndarray] = []

    for frame_index in range(sequence.frame_count):
        rgb = sequence.rgb_frame(frame_index)
        xyz = depth_to_xyz(sequence.depth_m[frame_index], sequence.intrinsics)
        assert xyz.shape == rgb.shape
        assert np.isfinite(xyz).all()
        assert np.count_nonzero(xyz[..., 2]) > 0

        objects = extract_segmented_instances(
            xyz,
            rgb,
            masks,
            class_ids,
            scores,
            names,
            config,
        )
        assert len(objects) == 3
        assert [obj.class_name for obj in objects] == [
            "fixture-bin-left",
            "fixture-bin-center",
            "fixture-bin-right",
        ]
        assert all(obj.points_xyz.shape == (500, 3) for obj in objects)
        assert all(obj.colors_rgb.shape == (500, 3) for obj in objects)
        assert all(np.isfinite(obj.position).all() for obj in objects)
        assert all(np.all(obj.bbox_size > 0.0) for obj in objects)
        positions_over_time.append(np.stack([obj.position for obj in objects]))

    positions = np.stack(positions_over_time)
    # The generator applies a deterministic ±20 mm camera-Z translation. This makes
    # the 3 s replay non-identical while preserving its public registered RGB-D source.
    assert np.all(np.ptp(positions[..., 2], axis=0) > 0.035)
    np.testing.assert_allclose(positions[0], positions[15], atol=1.0e-6)
    assert np.all(positions[7, :, 2] > positions[0, :, 2] + 0.018)


def test_bundled_evidence_files_match_the_fixture() -> None:
    console = (ROOT / "docs" / "sample_result_console.txt").read_text(encoding="utf-8")
    assert "fixture-bin-left" in console
    assert "fixture-bin-center" in console
    assert "fixture-bin-right" in console
    assert (ROOT / "docs" / "images" / "sample_rgb_segmentation.png").stat().st_size > 10_000
    assert (ROOT / "docs" / "images" / "sample_3d_result.png").stat().st_size > 10_000


def test_empty_prediction_produces_no_objects() -> None:
    sequence = load_rgbd_sequence(SAMPLE)
    rgb = sequence.rgb_frame(0)
    xyz = depth_to_xyz(sequence.depth_m[0], sequence.intrinsics)
    objects = extract_segmented_instances(
        xyz,
        rgb,
        np.empty((0, rgb.shape[0], rgb.shape[1]), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.float32),
        {},
        ExtractionConfig(),
    )
    assert objects == []
