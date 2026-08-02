# Verification report

## Executed in the artifact-generation environment

Target runtime is ROS 2 Jazzy / Python 3.12. The artifact-generation container used
Python 3.13.5 for the ROS-independent checks below.

The following command completed successfully:

```text
PYTHONPATH=src python3 -m pytest -q \
  test/test_core.py \
  test/test_pointcloud_conversion.py \
  test/test_inference.py \
  test/test_model_stage.py \
  test/test_sample_sequence.py

24 passed
```

The executed tests cover:

- PointCloud2 packed `rgb` and `rgba` decoding.
- Separate `r/g/b` fields.
- Little- and big-endian layouts.
- Row padding and serialized-buffer size validation.
- Replacement of any non-finite XYZ point and its RGB value with zero.
- Ultralytics adapter RGB-to-BGR conversion and extraction of masks, class IDs,
  confidence values, names, and debug images using a fake model without weights.
- Empty-prediction behavior at the inference/core boundary.
- YOLO-mask resizing to the organized camera resolution.
- Minimum valid-point and camera-depth filtering.
- Median 3D position and percentile AABB extraction.
- Deterministic compact point-cloud downsampling.
- The bundled 320 x 240, 10 Hz, 30-frame, 3.0-second public RGB-D replay.
- Deterministic ±20 mm temporal camera-Z variation across the replay.
- Three fixture masks producing compact object point/color arrays with valid metric
  positions on all 30 frames.
- Reuse of an already downloaded model only when its SHA-256 matches.
- Removal and replacement of a corrupt cached model through the `curl` path.
- Fallback download through the `wget` path.
- Offline model staging through `YOLO_3D_MODEL_SOURCE`.
- Rejection and removal of a staged model with the wrong SHA-256.

A CMake build/install smoke test was also executed without ROS by supplying minimal
stub package-config files. It configured the real `CMakeLists.txt`, ran the
`download_yolo_3d_model` target with an offline checkpoint fixture, and verified that
`cmake --install` created:

```text
share/yolo_3d_ros/models/yolo26m-seg.pt
```

The installed bytes matched the source fixture exactly. This validates the ordering
between the custom `ALL` target and the install rule used by a normal CMake/colcon
build.

Additional checks executed during artifact generation:

- Python syntax compilation for package modules, executable scripts, tools, launch
  files, and tests.
- XML, YAML, TOML, NumPy archive, source-asset hash, shell syntax, and archive
  integrity checks.
- Deterministic regeneration of the README RGB overlay and 3D point-cloud/AABB
  screenshots.

## Deferred to ROS 2 Jazzy CI or the target machine

The artifact-generation container does not include `/opt/ros/jazzy`, a RealSense
camera, an X11/Wayland Open3D display, or unrestricted outbound DNS. Therefore the
following are configured but are not claimed as locally executed:

- A real `colcon build` against ROS 2 Jazzy and rosidl generation for `yolo_3d_msgs`.
- `test/test_ros_messages.py`, including Jazzy serialization of each compact
  per-object `PointCloud2` and its order relative to `Detection3DArray`.
- RealSense driver integration and hardware QoS compatibility.
- Full download of the 54,750,385-byte official checkpoint from GitHub during the
  local test run. The fixed URL, size, and SHA-256 were checked against the official
  release/model metadata; downloader behavior itself was tested with controlled
  `curl` and `wget` executables.
- Actual YOLO26 neural-network inference.
- RViz and interactive Open3D rendering.
- Execution of the GitHub-hosted `industrial_ci` workflow itself.

`.github/workflows/ros2_jazzy.yml` supplies the ROS 2 Jazzy build/test path and
keeps `YOLO_3D_DOWNLOAD_MODEL=ON`, so the hosted build exercises the real network
asset download, hash verification, installation, and ROS tests.
