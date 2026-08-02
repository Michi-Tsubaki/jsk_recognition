#!/usr/bin/env python3
"""Convert WGISD COCO polygon annotations to Ultralytics YOLO segmentation."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def _merge_polygons(polygons: list[list[float]]) -> list[float] | None:
    if not polygons:
        return None
    if len(polygons) == 1:
        return polygons[0]

    try:
        import numpy as np
        from ultralytics.data.converter import merge_multi_segment
    except ImportError:
        return max(polygons, key=_polygon_area)

    merged = merge_multi_segment(polygons)
    if not merged:
        return max(polygons, key=_polygon_area)
    return np.concatenate(merged, axis=0).reshape(-1).tolist()


def _polygon_area(points: list[float]) -> float:
    if len(points) < 6 or len(points) % 2:
        return 0.0
    area = 0.0
    pairs = list(zip(points[0::2], points[1::2], strict=True))
    for index, (x0, y0) in enumerate(pairs):
        x1, y1 = pairs[(index + 1) % len(pairs)]
        area += float(x0) * float(y1) - float(x1) * float(y0)
    return abs(area) * 0.5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare WGISD for YOLO segmentation fine-tuning."
    )
    parser.add_argument(
        "--wgisd-root",
        type=Path,
        required=True,
        help="Path to a local clone of https://github.com/thsant/wgisd.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the YOLO segmentation dataset will be written.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking them into the output dataset.",
    )
    parser.add_argument(
        "--class-name",
        default="grape",
        help="Single class name used for all WGISD grape-variety categories.",
    )
    return parser.parse_args()


def _write_image(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def _normalise_polygon(points: list[float], width: int, height: int) -> list[float] | None:
    if len(points) < 6 or len(points) % 2:
        return None

    normalised: list[float] = []
    for index in range(0, len(points), 2):
        x = min(max(float(points[index]), 0.0), float(width))
        y = min(max(float(points[index + 1]), 0.0), float(height))
        normalised.extend((x / width, y / height))
    return normalised


def _convert_split(
    coco_json: Path,
    image_root: Path,
    output_dir: Path,
    split: str,
    copy_images: bool,
) -> tuple[int, int]:
    data = json.loads(coco_json.read_text())
    images = {image["id"]: image for image in data["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    image_count = 0
    annotation_count = 0
    for image_id, image in sorted(images.items()):
        file_name = image["file_name"]
        source_image = image_root / file_name
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing WGISD image: {source_image}")

        _write_image(
            source_image,
            output_dir / "images" / split / file_name,
            copy_images,
        )

        label_path = output_dir / "labels" / split / f"{Path(file_name).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            polygon = _merge_polygons(annotation.get("segmentation", []))
            if polygon is None:
                continue
            normalised = _normalise_polygon(
                polygon,
                width=int(image["width"]),
                height=int(image["height"]),
            )
            if normalised is None:
                continue
            values = " ".join(f"{value:.6f}" for value in normalised)
            lines.append(f"0 {values}")
            annotation_count += 1
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        image_count += 1

    return image_count, annotation_count


def main() -> None:
    args = _parse_args()
    wgisd_root = args.wgisd_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    coco_dir = wgisd_root / "coco_annotations"
    image_root = wgisd_root / "data"
    if not coco_dir.is_dir() or not image_root.is_dir():
        raise FileNotFoundError(
            "Expected WGISD directories 'coco_annotations' and 'data' under "
            f"{wgisd_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_images, train_annotations = _convert_split(
        coco_dir / "train_polygons_instances.json",
        image_root,
        output_dir,
        "train",
        args.copy_images,
    )
    val_images, val_annotations = _convert_split(
        coco_dir / "test_polygons_instances.json",
        image_root,
        output_dir,
        "val",
        args.copy_images,
    )

    dataset_yaml = output_dir / "wgisd_grape_seg.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {args.class_name}",
                "",
            ]
        )
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# WGISD YOLO Segmentation Dataset",
                "",
                "Generated from the Embrapa WGISD COCO polygon annotations.",
                "All grape-variety categories are mapped to one `grape` class.",
                "",
                "Source: https://github.com/thsant/wgisd",
                "License: Creative Commons Attribution-NonCommercial 4.0 International",
                "",
            ]
        )
    )

    print(f"Wrote {dataset_yaml}")
    print(f"train: {train_images} images, {train_annotations} polygons")
    print(f"val: {val_images} images, {val_annotations} polygons")


if __name__ == "__main__":
    main()
