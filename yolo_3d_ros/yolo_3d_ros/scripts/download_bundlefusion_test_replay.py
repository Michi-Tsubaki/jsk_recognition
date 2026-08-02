#!/usr/bin/env python3
"""Create the bundled 3 s RGB-D replay from a public BundleFusion frame.

The RGB and metric depth are measured data. To exercise temporal PointCloud2
transport without bundling a large sequence, a documented sinusoidal camera-Z
translation is applied to valid depth pixels over 30 frames. Deterministic
fixture masks are stored only for geometry smoke tests; they are not YOLO output.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from yolo_3d_ros.sample_data import bundlefusion_fixture_metadata  # noqa: E402

COLOR_URL = (
    "https://raw.githubusercontent.com/chaowang15/RGBDPlaneDetection/"
    "master/pic/frame-000000.color.jpg"
)
DEPTH_URL = (
    "https://raw.githubusercontent.com/chaowang15/RGBDPlaneDetection/"
    "master/pic/frame-000000.depth.png"
)
SOURCE_DESCRIPTION = (
    "BundleFusion copyroom frame-000000 RGB-D example distributed by "
    "chaowang15/RGBDPlaneDetection; 30-frame replay with deterministic "
    "camera-Z translation"
)
SOURCE_LICENSE = "CC-BY-NC-SA-4.0"
EXPECTED_COLOR_SHA256 = "c9b3b20cab6472f7f31402a7e3ba3274ccf892940d9de3d584dff60babb99d07"
EXPECTED_DEPTH_SHA256 = "b8dbee19bfebde9b57bee5dfbb1226dc12d59d0b1af43e74204bd82c2329008e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "yolo_3d_ros sample downloader"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        destination.write_bytes(response.read())


def create_sequence(
    color_path: Path,
    depth_path: Path,
    output: Path,
    width: int,
    height: int,
    fps: float,
    motion_amplitude_m: float,
) -> None:
    """Convert one registered RGB-D pair into a compact 3-second replay."""

    _verify_source(color_path, EXPECTED_COLOR_SHA256)
    _verify_source(depth_path, EXPECTED_DEPTH_SHA256)

    bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth_mm is None:
        raise RuntimeError("Failed to read the RGB-D source frame")
    if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
        raise RuntimeError(f"Expected a 16UC1 depth image, got {depth_mm.dtype} {depth_mm.shape}")
    if bgr.shape[:2] != depth_mm.shape:
        raise RuntimeError(
            f"Color/depth resolutions differ: {bgr.shape[:2]} versus {depth_mm.shape}"
        )

    original_height, original_width = depth_mm.shape
    rgb = cv2.cvtColor(
        cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    base_depth_m = (
        cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        / 1000.0
    )

    # The source example implementation uses fx=fy=583, cx=320, cy=240 at 640x480.
    scale_x = width / original_width
    scale_y = height / original_height
    intrinsics = np.array(
        [
            [583.0 * scale_x, 0.0, 320.0 * scale_x],
            [0.0, 583.0 * scale_y, 240.0 * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    frame_count = max(1, round(fps * 3.0))
    phase = 2.0 * np.pi * np.arange(frame_count, dtype=np.float32) / frame_count
    depth_translation_m = motion_amplitude_m * np.sin(phase)
    depth_sequence = np.repeat(base_depth_m[np.newaxis, ...], frame_count, axis=0)
    valid = depth_sequence > 0.0
    depth_sequence += depth_translation_m[:, np.newaxis, np.newaxis] * valid

    masks, class_ids, scores, names = bundlefusion_fixture_metadata(height, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rgb=rgb,
        depth_m=depth_sequence.astype(np.float32),
        intrinsics=intrinsics,
        fps=np.float32(fps),
        source=np.array(SOURCE_DESCRIPTION),
        source_license=np.array(SOURCE_LICENSE),
        source_color_url=np.array(COLOR_URL),
        source_depth_url=np.array(DEPTH_URL),
        source_color_sha256=np.array(EXPECTED_COLOR_SHA256),
        source_depth_sha256=np.array(EXPECTED_DEPTH_SHA256),
        depth_translation_m=depth_translation_m.astype(np.float32),
        smoke_masks=masks,
        smoke_class_ids=class_ids,
        smoke_scores=scores,
        smoke_names=names,
    )

    measured = base_depth_m[base_depth_m > 0.0]
    print(
        f"Wrote {output}: {frame_count} frames at {fps:.1f} Hz, "
        f"measured depth={measured.size:,}, range={measured.min():.3f}-{measured.max():.3f} m, "
        f"camera-Z dither=±{motion_amplitude_m:.3f} m"
    )


def _copy_sources(color_path: Path, depth_path: Path, source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(color_path, source_dir / "frame-000000.color.jpg")
    shutil.copy2(depth_path, source_dir / "frame-000000.depth.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "docs" / "sample_rgbd_3s.npz",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PACKAGE_ROOT / "docs" / "sample_source",
        help="Directory in which the verified original 640x480 source pair is retained.",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--motion-amplitude-m", type=float, default=0.02)
    parser.add_argument("--color-file", type=Path)
    parser.add_argument("--depth-file", type=Path)
    args = parser.parse_args()

    if (args.color_file is None) != (args.depth_file is None):
        parser.error("--color-file and --depth-file must be supplied together")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0.0:
        parser.error("width, height, and fps must be positive")
    if args.motion_amplitude_m < 0.0:
        parser.error("--motion-amplitude-m must be non-negative")

    if args.color_file is not None and args.depth_file is not None:
        _copy_sources(args.color_file, args.depth_file, args.source_dir)
        create_sequence(
            args.color_file,
            args.depth_file,
            args.output,
            args.width,
            args.height,
            args.fps,
            args.motion_amplitude_m,
        )
        return

    with tempfile.TemporaryDirectory(prefix="yolo3d_bundlefusion_") as temporary:
        root = Path(temporary)
        color_path = root / "frame-000000.color.jpg"
        depth_path = root / "frame-000000.depth.png"
        print(f"Downloading {COLOR_URL}")
        _download(COLOR_URL, color_path)
        print(f"Downloading {DEPTH_URL}")
        _download(DEPTH_URL, depth_path)
        _copy_sources(color_path, depth_path, args.source_dir)
        create_sequence(
            color_path,
            depth_path,
            args.output,
            args.width,
            args.height,
            args.fps,
            args.motion_amplitude_m,
        )


if __name__ == "__main__":
    main()
