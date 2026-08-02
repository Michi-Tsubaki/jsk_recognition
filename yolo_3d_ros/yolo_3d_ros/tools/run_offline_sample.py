#!/usr/bin/env python3
"""Run Ultralytics segmentation and mask-to-3D extraction without ROS.

The default input is the first frame of the bundled 3-second RGB-D replay. A
custom replay with the same NPZ schema can be supplied through ``--sequence``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from yolo_3d_ros.core import ExtractionConfig, extract_segmented_instances  # noqa: E402
from yolo_3d_ros.inference import YOLOSegmenter  # noqa: E402
from yolo_3d_ros.sample_data import depth_to_xyz, load_rgbd_sequence  # noqa: E402


def _load_rgbd_frame(sequence_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    sequence = load_rgbd_sequence(sequence_path)
    rgb = sequence.rgb_frame(0)
    xyz = depth_to_xyz(sequence.depth_m[0], sequence.intrinsics)
    return rgb, xyz, sequence.source


def _installed_default_model() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("yolo_3d_ros")) / "models" / "yolo26m-seg.pt"
    except (ImportError, LookupError):
        return PACKAGE_ROOT / "models" / "yolo26m-seg.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--sequence",
        type=Path,
        default=PACKAGE_ROOT / "docs" / "sample_rgbd_3s.npz",
        help="RGB-D NPZ replay; frame 0 is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "docs" / "images" / "generated_offline_yolo_result.png",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    model_path = args.model or _installed_default_model()
    if not model_path.is_file():
        raise SystemExit(
            f"Model checkpoint not found: {model_path}. Build yolo_3d_ros first, "
            "or pass --model /absolute/path/to/model.pt."
        )

    rgb, xyz, source = _load_rgbd_frame(args.sequence)
    segmenter = YOLOSegmenter(
        str(model_path),
        args.confidence,
        0.7,
        640,
        args.device,
        None,
    )
    prediction = segmenter.predict(rgb, annotate=True)
    instances = extract_segmented_instances(
        xyz,
        rgb,
        prediction.masks,
        prediction.class_ids,
        prediction.scores,
        prediction.names,
        ExtractionConfig(min_points=30, max_points_per_object=5000),
    )
    if not instances:
        raise SystemExit(
            "No valid 3D objects were produced. The bundled copyroom frame contains "
            "objects outside the standard COCO classes, so a pretrained COCO model may "
            "legitimately return no detections."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12, 5.5))
    ax_image = figure.add_subplot(1, 2, 1)
    if prediction.annotated_bgr is not None:
        ax_image.imshow(cv2.cvtColor(prediction.annotated_bgr, cv2.COLOR_BGR2RGB))
    else:
        ax_image.imshow(rgb)
    ax_image.set_title("Ultralytics instance segmentation")
    ax_image.axis("off")

    ax_3d = figure.add_subplot(1, 2, 2, projection="3d")
    for instance in instances:
        step = max(1, instance.points_xyz.shape[0] // 1500)
        points = instance.points_xyz[::step]
        colors = instance.colors_rgb[::step] / 255.0
        ax_3d.scatter(points[:, 0], points[:, 2], -points[:, 1], s=1, c=colors)
        x, y, z = instance.position
        ax_3d.text(x, z, -y, instance.class_name)
    ax_3d.set_xlabel("camera x [m]")
    ax_3d.set_ylabel("camera z [m]")
    ax_3d.set_zlabel("-camera y [m]")
    ax_3d.set_title("Mask-selected compact 3D points")
    figure.suptitle(source, fontsize=9)
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)
    plt.close(figure)

    print(f"Wrote {args.output}")
    for instance in instances:
        x, y, z = instance.position
        print(
            f"{instance.class_name}: score={instance.score:.3f}, "
            f"position=({x:.4f}, {y:.4f}, {z:.4f}) m, "
            f"source_points={instance.source_point_count}, "
            f"published_points={len(instance.points_xyz)}"
        )


if __name__ == "__main__":
    main()
