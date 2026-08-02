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


## WGISD grape checkpoint

`wgisd_grape_seg.pt` is a YOLO11n segmentation checkpoint fine-tuned for one class:

```text
0: grape
```

It was trained for 20 epochs from `yolo11n-seg.pt` using WGISD COCO polygon
annotations converted by `tools/prepare_wgisd_yolo_seg.py`.

Training summary:

```text
train images: 110
validation images: 27
box mAP50: 0.787
mask mAP50: 0.781
```

This checkpoint is not licensed under this repository's Apache-2.0 license. The
base Ultralytics checkpoint is subject to Ultralytics licensing terms, and WGISD is
licensed under Creative Commons Attribution-NonCommercial 4.0 International.
