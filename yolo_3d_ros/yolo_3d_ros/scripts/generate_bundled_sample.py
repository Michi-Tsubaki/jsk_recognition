#!/usr/bin/env python3
"""Regenerate README evidence images from the bundled 3-second RGB-D replay.

The NPZ contains measured BundleFusion RGB/depth plus a documented sinusoidal
camera-Z dither and deterministic fixture masks. The fixture masks exercise the
mask-to-3D geometry path without neural-network weights; they are not YOLO output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from yolo_3d_ros.core import (  # noqa: E402
    ExtractionConfig,
    SegmentedInstance,
    extract_segmented_instances,
)
from yolo_3d_ros.sample_data import depth_to_xyz, load_rgbd_sequence  # noqa: E402


def _load_fixture_metadata(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    with np.load(path, allow_pickle=False) as data:
        masks = np.asarray(data["smoke_masks"], dtype=np.float32)
        class_ids = np.asarray(data["smoke_class_ids"], dtype=np.int64)
        scores = np.asarray(data["smoke_scores"], dtype=np.float32)
        names = np.asarray(data["smoke_names"]).astype(str)
    name_map = {
        int(class_id): str(name)
        for class_id, name in zip(class_ids, names, strict=True)
    }
    return masks, class_ids, scores, name_map


def _bbox_edges(instance: SegmentedInstance) -> list[tuple[np.ndarray, np.ndarray]]:
    half = instance.bbox_size / 2.0
    lower = instance.bbox_center - half
    upper = instance.bbox_center + half
    corners = np.array(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [lower[0], upper[1], upper[2]],
            [upper[0], upper[1], upper[2]],
        ],
        dtype=np.float32,
    )
    pairs = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return [(corners[start], corners[end]) for start, end in pairs]


def _write_rgb_result(
    rgb: np.ndarray,
    masks: np.ndarray,
    instances: list[SegmentedInstance],
    output: Path,
) -> None:
    overlay = rgb.copy().astype(np.float32)
    overlay_colors = np.array(
        [[65, 105, 225], [255, 196, 0], [40, 190, 90]],
        dtype=np.float32,
    )
    for mask, overlay_color in zip(masks.astype(bool), overlay_colors, strict=True):
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * overlay_color
    frame = cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    scale = 3
    frame = cv2.resize(
        frame,
        (rgb.shape[1] * scale, rgb.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    for index, instance in enumerate(instances, start=1):
        contours, _ = cv2.findContours(
            instance.pixel_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            frame,
            [contour * scale for contour in contours],
            -1,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        rows, cols = np.nonzero(instance.pixel_mask)
        anchor = (int(np.median(cols) * scale), int(np.median(rows) * scale))
        cv2.circle(frame, anchor, 20, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, anchor, 18, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(index),
            (anchor[0] - 7, anchor[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    panel_width = 600
    canvas = np.full(
        (frame.shape[0], frame.shape[1] + panel_width, 3),
        24,
        dtype=np.uint8,
    )
    canvas[:, : frame.shape[1]] = frame
    text_x = frame.shape[1] + 24
    cv2.putText(
        canvas,
        "Public RGB-D replay: mask-to-3D result",
        (text_x, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "fixture masks only; not neural-network output",
        (text_x, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    cursor_y = 142
    for index, instance in enumerate(instances, start=1):
        x, y, z = instance.position
        lines = [
            f"{index}. {instance.class_name}",
            f"   xyz=({x:.3f}, {y:.3f}, {z:.3f}) m",
            f"   selected={instance.source_point_count:,} points",
            f"   published={len(instance.points_xyz):,} points",
        ]
        for line_index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (text_x, cursor_y + 26 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.49,
                (245, 245, 245) if line_index == 0 else (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
        cursor_y += 130

    cv2.putText(
        canvas,
        "30 frames / 10 Hz / 3.0 s; measured depth + documented Z dither",
        (text_x, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (165, 165, 165),
        1,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Failed to write {output}")


def _write_3d_result(instances: list[SegmentedInstance], output: Path) -> None:
    figure = plt.figure(figsize=(10.2, 6.4))
    axis = figure.add_subplot(111, projection="3d")
    for index, instance in enumerate(instances, start=1):
        step = max(1, instance.points_xyz.shape[0] // 3000)
        points = instance.points_xyz[::step]
        colors = instance.colors_rgb[::step] / 255.0
        axis.scatter(points[:, 0], points[:, 2], -points[:, 1], s=1.2, c=colors)
        for start, end in _bbox_edges(instance):
            axis.plot(
                [start[0], end[0]],
                [start[2], end[2]],
                [-start[1], -end[1]],
                linewidth=1.1,
                color="black",
            )
        x, y, z = instance.position
        axis.text(x, z, -y, str(index), fontsize=11, weight="bold")

    axis.set_xlabel("camera x [m]")
    axis.set_ylabel("camera z [m]")
    axis.set_zlabel("-camera y [m]")
    axis.set_title("Mask-selected colored points, median positions, and 3D AABBs")
    axis.view_init(elev=18, azim=-68)
    axis.set_box_aspect((1.5, 2.3, 1.0))
    legend_lines = []
    for index, instance in enumerate(instances, start=1):
        x, y, z = instance.position
        legend_lines.append(
            f"{index}: {instance.class_name:<20} ({x:.3f}, {y:.3f}, {z:.3f}) m"
        )
    figure.text(
        0.02,
        0.90,
        "\n".join(legend_lines),
        va="top",
        fontsize=9,
        family="monospace",
    )
    figure.text(
        0.5,
        0.02,
        "Measured registered RGB-D frame; deterministic fixture masks; no YOLO inference",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _write_result_text(instances: list[SegmentedInstance]) -> None:
    result_lines = [
        "Bundled public RGB-D geometry smoke-test result (frame 0)",
        "The masks are deterministic test fixtures, not neural-network inference.",
    ]
    for instance in instances:
        x, y, z = instance.position
        result_lines.append(
            f"{instance.class_name}: score={instance.score:.3f}, "
            f"position=({x:.4f}, {y:.4f}, {z:.4f}) m, "
            f"source_points={instance.source_point_count}, "
            f"published_points={len(instance.points_xyz)}"
        )

    console_text = "\n".join(result_lines) + "\n"
    (PACKAGE_ROOT / "docs" / "sample_result_console.txt").write_text(
        console_text,
        encoding="utf-8",
    )
    markdown_lines = [
        "# Deterministic sample output",
        "",
        "Generated by `scripts/generate_bundled_sample.py` from replay frame 0.",
        "The masks are geometry-test fixtures, not YOLO inference.",
        "",
        "```text",
        *result_lines[2:],
        "```",
        "",
    ]
    (PACKAGE_ROOT / "docs" / "SAMPLE_OUTPUT.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )
    print(console_text, end="")


def main() -> None:
    fixture_path = PACKAGE_ROOT / "docs" / "sample_rgbd_3s.npz"
    sequence = load_rgbd_sequence(fixture_path)
    masks, class_ids, scores, names = _load_fixture_metadata(fixture_path)
    rgb = sequence.rgb_frame(0)
    xyz = depth_to_xyz(sequence.depth_m[0], sequence.intrinsics)
    instances = extract_segmented_instances(
        xyz,
        rgb,
        masks,
        class_ids,
        scores,
        names,
        ExtractionConfig(min_points=30, max_points_per_object=5000),
    )
    if len(instances) != masks.shape[0]:
        raise RuntimeError(f"Expected {masks.shape[0]} fixture objects, got {len(instances)}")

    images = PACKAGE_ROOT / "docs" / "images"
    _write_rgb_result(rgb, masks, instances, images / "sample_rgb_segmentation.png")
    _write_3d_result(instances, images / "sample_3d_result.png")
    _write_result_text(instances)


if __name__ == "__main__":
    main()
