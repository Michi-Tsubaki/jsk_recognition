"""Load compact RGB-D test sequences and unproject them to organized XYZ arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class RGBDSequence:
    """A compact RGB-D sequence stored in one NPZ file."""

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    fps: float
    source: str

    @property
    def frame_count(self) -> int:
        return int(self.depth_m.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps

    def rgb_frame(self, index: int) -> np.ndarray:
        if self.rgb.ndim == 3:
            return self.rgb
        return self.rgb[index]


def load_rgbd_sequence(path: str | Path) -> RGBDSequence:
    """Load and validate an RGB-D replay from a compressed NPZ file."""

    with np.load(Path(path), allow_pickle=False) as data:
        rgb = np.asarray(data["rgb"], dtype=np.uint8)
        depth_m = np.asarray(data["depth_m"], dtype=np.float32)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).item())
        source = str(np.asarray(data["source"]).item())
    if depth_m.ndim != 3:
        raise ValueError(f"depth_m must have shape (N,H,W), got {depth_m.shape}")
    if rgb.ndim not in (3, 4) or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must have shape (H,W,3) or (N,H,W,3), got {rgb.shape}")
    if rgb.shape[-3:-1] != depth_m.shape[-2:]:
        raise ValueError("RGB and depth resolutions do not match")
    if rgb.ndim == 4 and rgb.shape[0] != depth_m.shape[0]:
        raise ValueError("RGB and depth frame counts do not match")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3,3), got {intrinsics.shape}")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    return RGBDSequence(rgb, depth_m, intrinsics, fps, source)


def depth_to_xyz(depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject a metric depth image into an organized HxWx3 XYZ array."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D image")
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3,3)")
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("focal lengths must be positive")

    rows, cols = np.indices(depth.shape, dtype=np.float32)
    xyz = np.empty((*depth.shape, 3), dtype=np.float32)
    xyz[..., 2] = depth
    xyz[..., 0] = (cols - cx) * depth / fx
    xyz[..., 1] = (rows - cy) * depth / fy
    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    xyz[invalid] = 0.0
    return xyz


def bundlefusion_fixture_metadata(
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic masks used only for the bundled geometry smoke test.

    The polygons describe three bin-like image regions in the public BundleFusion
    copyroom frame. They are deliberately not presented as neural-network output.
    """

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    reference_width = 320.0
    reference_height = 240.0
    polygons = [
        [(96, 48), (171, 44), (184, 61), (183, 153), (165, 179), (121, 180), (106, 151), (101, 91)],
        [
            (158, 34),
            (228, 18),
            (255, 28),
            (267, 50),
            (262, 136),
            (244, 157),
            (178, 157),
            (168, 131),
            (168, 65),
        ],
        [(262, 0), (319, 0), (319, 159), (288, 165), (259, 145), (242, 104), (249, 52)],
    ]

    masks = np.zeros((3, height, width), dtype=np.uint8)
    for index, polygon in enumerate(polygons):
        scaled = np.asarray(
            [
                (
                    int(round(x * width / reference_width)),
                    int(round(y * height / reference_height)),
                )
                for x, y in polygon
            ],
            dtype=np.int32,
        )
        scaled[:, 0] = np.clip(scaled[:, 0], 0, width - 1)
        scaled[:, 1] = np.clip(scaled[:, 1], 0, height - 1)
        cv2.fillPoly(masks[index], [scaled], 1)

    # Keep the fixture masks disjoint. Real YOLO masks are handled independently and
    # are allowed to overlap.
    right = masks[2].astype(bool)
    center = masks[1].astype(bool) & ~right
    left = masks[0].astype(bool) & ~(center | right)
    masks_float = np.stack((left, center, right)).astype(np.float32)
    class_ids = np.array([1000, 1001, 1002], dtype=np.int64)
    scores = np.array([0.99, 0.98, 0.97], dtype=np.float32)
    names = np.asarray(
        ["fixture-bin-left", "fixture-bin-center", "fixture-bin-right"],
        dtype="<U24",
    )
    return masks_float, class_ids, scores, names
