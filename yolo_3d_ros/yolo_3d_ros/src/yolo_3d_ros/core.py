"""ROS-independent 2D-mask to 3D-instance extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Controls conversion of segmentation masks into compact object point clouds."""

    mask_threshold: float = 0.5
    min_points: int = 100
    min_depth_m: float = 0.05
    max_depth_m: float = 10.0
    bbox_percentile: float = 1.0
    max_points_per_object: int = 25_000
    random_seed: int = 0

    def validate(self) -> None:
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        if self.min_points < 1:
            raise ValueError("min_points must be >= 1")
        if not 0.0 <= self.min_depth_m < self.max_depth_m:
            raise ValueError("depth limits must satisfy 0 <= min_depth_m < max_depth_m")
        if not 0.0 <= self.bbox_percentile < 50.0:
            raise ValueError("bbox_percentile must be in [0, 50)")
        if self.max_points_per_object < 0:
            raise ValueError("max_points_per_object must be >= 0")


@dataclass(slots=True)
class SegmentedInstance:
    """One mask-selected object in the input camera frame."""

    class_id: int
    class_name: str
    score: float
    instance_index: int
    points_xyz: np.ndarray
    colors_rgb: np.ndarray
    position: np.ndarray
    position_variance: np.ndarray
    bbox_center: np.ndarray
    bbox_size: np.ndarray
    pixel_mask: np.ndarray
    source_point_count: int


def _class_name(class_id: int, names: Mapping[int, str] | Sequence[str]) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _resize_masks(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N,H,W), got {masks.shape}")
    if masks.shape[1:] == (height, width):
        return masks.astype(np.float32, copy=False)

    resized = np.empty((masks.shape[0], height, width), dtype=np.float32)
    for index, mask in enumerate(masks):
        resized[index] = cv2.resize(
            mask.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return resized


def extract_segmented_instances(
    xyz: np.ndarray,
    rgb: np.ndarray,
    masks: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    names: Mapping[int, str] | Sequence[str],
    config: ExtractionConfig | None = None,
) -> list[SegmentedInstance]:
    """Apply instance masks to organized XYZ/RGB arrays and estimate 3D geometry.

    The returned point clouds are compact ``N x 3`` arrays.  Position is the median
    of all valid mask-selected points.  The axis-aligned bounding box uses configurable
    lower/upper percentiles to suppress isolated depth outliers.
    """

    cfg = config or ExtractionConfig()
    cfg.validate()

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"xyz must have shape (H,W,3), got {xyz.shape}")
    if rgb.shape != xyz.shape:
        raise ValueError(f"rgb must have the same shape as xyz, got {rgb.shape} vs {xyz.shape}")

    height, width, _ = xyz.shape
    masks = _resize_masks(masks, height, width)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    count = min(masks.shape[0], class_ids.size, scores.size)
    if count == 0:
        return []

    finite = np.isfinite(xyz).all(axis=2)
    nonzero = np.linalg.norm(xyz, axis=2) > np.finfo(np.float32).eps
    depth_ok = (xyz[..., 2] >= cfg.min_depth_m) & (xyz[..., 2] <= cfg.max_depth_m)
    valid_xyz = finite & nonzero & depth_ok

    instances: list[SegmentedInstance] = []
    for instance_index in range(count):
        pixel_mask = masks[instance_index] >= cfg.mask_threshold
        selected = pixel_mask & valid_xyz
        source_point_count = int(np.count_nonzero(selected))
        if source_point_count < cfg.min_points:
            continue

        points = np.ascontiguousarray(xyz[selected], dtype=np.float32)
        colors = np.ascontiguousarray(rgb[selected], dtype=np.uint8)

        position = np.median(points, axis=0).astype(np.float32)
        variance = np.var(points, axis=0, dtype=np.float64).astype(np.float32)
        if cfg.bbox_percentile > 0.0:
            lower, upper = np.percentile(
                points,
                [cfg.bbox_percentile, 100.0 - cfg.bbox_percentile],
                axis=0,
            )
        else:
            lower = points.min(axis=0)
            upper = points.max(axis=0)
        bbox_center = ((lower + upper) * 0.5).astype(np.float32)
        bbox_size = np.maximum(upper - lower, 1.0e-4).astype(np.float32)

        if cfg.max_points_per_object and points.shape[0] > cfg.max_points_per_object:
            rng = np.random.default_rng(cfg.random_seed + instance_index)
            chosen = rng.choice(
                points.shape[0],
                size=cfg.max_points_per_object,
                replace=False,
            )
            chosen.sort()
            points = np.ascontiguousarray(points[chosen])
            colors = np.ascontiguousarray(colors[chosen])

        class_id = int(class_ids[instance_index])
        instances.append(
            SegmentedInstance(
                class_id=class_id,
                class_name=_class_name(class_id, names),
                score=float(scores[instance_index]),
                instance_index=instance_index,
                points_xyz=points,
                colors_rgb=colors,
                position=position,
                position_variance=variance,
                bbox_center=bbox_center,
                bbox_size=bbox_size,
                pixel_mask=np.ascontiguousarray(pixel_mask),
                source_point_count=source_point_count,
            )
        )
    return instances
