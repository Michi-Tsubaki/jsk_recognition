# Sample data provenance and construction

`sample_rgbd_3s.npz` is the compact organized RGB-D replay used by the pure-Python
tests and by `sample_pointcloud_publisher`.

## Public source pair

The archive retains the verified, original 640 x 480 files:

- `docs/sample_source/frame-000000.color.jpg`
- `docs/sample_source/frame-000000.depth.png`

They were obtained from:

- <https://github.com/chaowang15/RGBDPlaneDetection/tree/master/pic>
- color: <https://raw.githubusercontent.com/chaowang15/RGBDPlaneDetection/master/pic/frame-000000.color.jpg>
- depth: <https://raw.githubusercontent.com/chaowang15/RGBDPlaneDetection/master/pic/frame-000000.depth.png>

The redistribution repository identifies the pair as `frame-000000` from the
BundleFusion `copyroom` sequence. BundleFusion documents color frames as 24-bit JPG,
depth frames as 16-bit PNG in millimetres, and invalid depth as zero.

Source SHA-256 values:

```text
c9b3b20cab6472f7f31402a7e3ba3274ccf892940d9de3d584dff60babb99d07  frame-000000.color.jpg
b8dbee19bfebde9b57bee5dfbb1226dc12d59d0b1af43e74204bd82c2329008e  frame-000000.depth.png
```

## Deterministic 3-second replay

`scripts/download_bundlefusion_test_replay.py` performs the conversion:

1. Verify both original files against the SHA-256 values above.
2. Resize RGB from 640 x 480 to 320 x 240 with area interpolation.
3. Resize depth with nearest-neighbour interpolation and convert millimetres to metres.
4. Scale the fixture intrinsics used by the redistribution implementation,
   `fx=fy=583`, `cx=320`, `cy=240`, to:

   ```text
   fx=fy=291.5, cx=160.0, cy=120.0
   ```

5. Create 30 frames at 10 Hz, yielding exactly 3.0 seconds.
6. For frame index `i` in `[0, 29]`, apply the following documented camera-Z test
   perturbation only where the measured depth `z0(u,v)` is non-zero:

   ```text
   dz(i) = 0.020 sin(2 pi i / 30) [m]
   z_i(u,v) = z0(u,v) + dz(i), when z0(u,v) > 0
   z_i(u,v) = 0,                 otherwise
   ```

The source is one registered measured RGB-D frame, not a native 30-frame camera
recording. The temporal perturbation makes the 3-second publisher and all 30-frame
processing paths testable without redistributing the 520 MB `copyroom` sequence.
This distinction is intentionally explicit in the sample output documentation.

## Fixture masks

The NPZ also stores three deterministic polygon masks covering bin-like regions in
the public RGB frame. They are used only to test mask resizing, depth filtering,
3D position estimation, AABB estimation, compact point arrays, and 30-frame replay.
They are **not Ultralytics predictions**.

The actual-model path is separate:

```bash
uv run --active --no-sync python tools/run_offline_sample.py \
  --sequence docs/sample_rgbd_3s.npz \
  --model "$(ros2 pkg prefix yolo_3d_ros)/share/yolo_3d_ros/models/yolo26m-seg.pt" \
  --device cpu
```

A generic pretrained model is not guaranteed to recognize an object in this
particular frame; the deterministic masks keep the geometry regression test stable.

## Stored NPZ entries

```text
rgb                   uint8    (240, 320, 3)
depth_m               float32  (30, 240, 320)
intrinsics             float32  (3, 3)
fps                    float32  scalar = 10.0
source                 unicode  scalar
source_license         unicode  scalar
source_color_url       unicode  scalar
source_depth_url       unicode  scalar
source_color_sha256    unicode  scalar
source_depth_sha256    unicode  scalar
depth_translation_m    float32  (30,)
smoke_masks            float32  (3, 240, 320)
smoke_class_ids        int64    (3,)
smoke_scores           float32  (3,)
smoke_names            unicode  (3,)
```

Current NPZ SHA-256:

```text
bd712bc37d5f76665318a6e0127fd082f5490a3d6f54d86b51ab5580dd063ace  sample_rgbd_3s.npz
```

## Regeneration

Re-download, verify, and recreate the replay:

```bash
cd yolo_3d_ros
PYTHONPATH=src python3 scripts/download_bundlefusion_test_replay.py
```

Regenerate the fixture-based console output and README screenshots from the existing
NPZ, without changing the NPZ:

```bash
PYTHONPATH=src python3 scripts/generate_bundled_sample.py
```

An optional helper can fetch Open3D's five-frame Redwood sample and expand it to a
3-second replay for additional local experiments:

```bash
PYTHONPATH=src python3 scripts/download_redwood_test_sequence.py \
  --output test/data/redwood_rgbd_3s.npz
```

## License

BundleFusion RGB-D scanning data is released under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0. The source pair, NPZ, and derived
screenshots are therefore separate from this package's AGPL-3.0-only code license.
