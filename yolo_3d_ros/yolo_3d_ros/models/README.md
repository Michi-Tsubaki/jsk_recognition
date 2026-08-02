# Installed model directory

`colcon build --packages-up-to yolo_3d_ros --symlink-install` downloads the pinned
Ultralytics YOLO26 segmentation checkpoint into the package build tree and installs
it as:

```text
share/yolo_3d_ros/models/yolo26m-seg.pt
```

Default source:

```text
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt
SHA-256: 16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5
Size: 54,750,385 bytes
```

The checkpoint is not licensed under this repository's Apache-2.0 license. It is an
Ultralytics asset subject to the Ultralytics AGPL-3.0 or Enterprise licensing terms.
See `../licenses/ULTRALYTICS_AGPL-3.0.txt`.
