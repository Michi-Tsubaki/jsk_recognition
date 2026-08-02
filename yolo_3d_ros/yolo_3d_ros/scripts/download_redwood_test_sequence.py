#!/usr/bin/env python3
"""Download Open3D's 5-frame Redwood RGB-D sample and make a 3 s compact sequence."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

URL = (
    "https://github.com/isl-org/open3d_downloads/releases/download/"
    "20220201-data/SampleRedwoodRGBDImages.zip"
)


def _find_files(root: Path, suffix: str, parent_name: str) -> list[Path]:
    return sorted(
        path for path in root.rglob(f"*{suffix}") if parent_name.lower() in path.parent.name.lower()
    )


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination_resolved not in target.parents and target != destination_resolved:
            raise RuntimeError(f"Unsafe path in ZIP archive: {member.filename}")
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test/data/redwood_rgbd_3s.npz"),
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="yolo3d_redwood_") as temp_dir:
        temp = Path(temp_dir)
        archive = temp / "SampleRedwoodRGBDImages.zip"
        print(f"Downloading {URL}")
        with urllib.request.urlopen(URL) as response, archive.open("wb") as output:  # noqa: S310
            shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as zf:
            _safe_extract(zf, temp / "extracted")
        extracted = temp / "extracted"

        colors = _find_files(extracted, ".jpg", "color")[:5]
        depths = _find_files(extracted, ".png", "depth")[:5]
        if len(colors) != 5 or len(depths) != 5:
            raise RuntimeError(
                f"Expected five color/depth frames, found {len(colors)} and {len(depths)}"
            )

        intrinsic_files = list(extracted.rglob("camera_primesense.json"))
        intrinsic_files.extend(extracted.rglob("camera_intrinsic.json"))
        fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
        original_width, original_height = 640, 480
        if intrinsic_files:
            payload = json.loads(intrinsic_files[0].read_text())
            matrix = payload.get("intrinsic_matrix", [])
            if len(matrix) == 9:
                fx, fy, cx, cy = matrix[0], matrix[4], matrix[6], matrix[7]
            original_width = int(payload.get("width", original_width))
            original_height = int(payload.get("height", original_height))

        scale_x = args.width / original_width
        scale_y = args.height / original_height
        intrinsics = np.array(
            [
                [fx * scale_x, 0.0, cx * scale_x],
                [0.0, fy * scale_y, cy * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        rgb_frames = []
        depth_frames = []
        repeats = max(1, round(args.fps * 3.0 / len(colors)))
        for color_path, depth_path in zip(colors, depths, strict=True):
            bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if bgr is None or depth_mm is None:
                raise RuntimeError(f"Failed to read {color_path} or {depth_path}")
            rgb = cv2.cvtColor(
                cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            depth_m = cv2.resize(
                depth_mm,
                (args.width, args.height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32) / 1000.0
            for _ in range(repeats):
                rgb_frames.append(rgb)
                depth_frames.append(depth_m)

        target_frames = round(args.fps * 3.0)
        rgb_array = np.asarray(rgb_frames[:target_frames], dtype=np.uint8)
        depth_array = np.asarray(depth_frames[:target_frames], dtype=np.float32)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            rgb=rgb_array,
            depth_m=depth_array,
            intrinsics=intrinsics,
            fps=np.float32(args.fps),
            source=np.array("Open3D SampleRedwoodRGBDImages (Redwood living-room1)"),
        )
        print(f"Wrote {args.output} ({len(depth_array) / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
