# yolo_3d_ros

[![ROS 2 Jazzy CI](https://github.com/Michi-Tsubaki/jsk_recognition/actions/workflows/ros2_jazzy.yml/badge.svg?branch=ros2)](https://github.com/Michi-Tsubaki/jsk_recognition/actions/workflows/ros2_jazzy.yml)

ROS 2 packages for running Ultralytics YOLO instance segmentation on `sensor_msgs/msg/PointCloud2` stream and publishing 3D detections, segmented point clouds and bounding boxes.

## Example (Recognition of a Grape)

###  Point Cloud 2 (Rviz2) ▶ BoundingBox3DArray (Rviz2)
<img src="figs/grape-rviz.png" width="45%"/> <img src="figs/arrow.png" width="2%"/> <img src="figs/grape-rviz-bbox.png" width="45%"/>

## Setup

Please note that if `uv` (a python package manager) is not installed on your environment, `uv` will be automatically installed when you run `colcon build`.

```bash
mkdir -p <path to your colcon workspace>/src
cd <path to your colcon workspace>/src
git clone https://github.com/Michi-Tsubaki/jsk_recognition.git -b ros2
cd ..
vcs import src < src/jsk_recognition/yolo_3d_ros/jazzy.repos
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep update
rosdep install --from-paths src -iry
cd src/jsk_recognition/yolo_3d_ros
uv sync
```

This creates `yolo_3d_ros/.venv` and installs the Python dependencies there.

```bash
cd <path to your colcon workspace>
colcon build --packages-up-to yolo_3d_ros --symlink-install
source install/setup.bash
```

`colcon build` downloads `yolo26m-seg.pt`, verifies its SHA-256 hash and installs it to
```text
<path to your colcon workspace>/install/yolo_3d_ros/share/yolo_3d_ros/models/yolo26m-seg.pt
```


## Run With RealSense

- Start RealSense with **aligned** depth and **ordered point clouds**.

```bash
sudo apt update
sudo apt install ros-jazzy-realsense2-camera
source /opt/ros/jazzy/setup.bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true pointcloud.enable:=true pointcloud.ordered_pc:=true
```

- Run YOLO 3D in another terminal.

```bash
source /opt/ros/jazzy/setup.bash
source <path to your colcon workspace>/install/setup.bash

ros2 launch yolo_3d_ros yolo_3d.launch.py input_topic:=/camera/depth/color/points
```

- Use `device:=cpu` to force CPU inference.
```bash
ros2 launch yolo_3d_ros yolo_3d.launch.py input_topic:=/camera/depth/color/points device:=cpu
```

For a D405 aimed at shine muscat or grape bunches, use the WGISD fine-tuned grape
segmentation model:

```bash
ros2 launch yolo_3d_ros d405_shine_muscat_finetuned.launch.py \
  input_topic:=/camera/camera/depth/color/points
```

The fine-tuned model publishes only class `0: grape`. For the stock COCO-trained
YOLO segmentation model, `grape`, `muscat`, `blueberry`, and `raspberry` are not
available class names. The proxy launch below filters with the closest available
COCO classes, ORed as `banana`, `apple`, `orange`, `broccoli`, and `potted plant`.

```bash
ros2 launch yolo_3d_ros d405_shine_muscat.launch.py \
  input_topic:=/camera/camera/depth/color/points
```

- Use another checkpoint only when needed, especially when you finetune the model,

```bash
ros2 launch yolo_3d_ros yolo_3d.launch.py input_topic:=/camera/depth/color/points model:=<absolute path to the finetuned model.pt>
```


## WGISD Grape Fine-Tuning

The committed `models/wgisd_grape_seg.pt` checkpoint was fine-tuned from
`yolo11n-seg.pt` on Embrapa WGISD COCO polygon annotations converted to one
`grape` class.

Recreate the dataset from a local WGISD clone:

```bash
git clone https://github.com/thsant/wgisd.git /tmp/wgisd
yolo_3d_ros/.venv/bin/python yolo_3d_ros/tools/prepare_wgisd_yolo_seg.py \
  --wgisd-root /tmp/wgisd \
  --output-dir /tmp/wgisd_yolo_seg
```

Run the same short fine-tuning job:

```bash
yolo_3d_ros/.venv/bin/python yolo_3d_ros/tools/train_wgisd_grape_seg.py \
  --data /tmp/wgisd_yolo_seg/wgisd_grape_seg.yaml \
  --model yolo11n-seg.pt \
  --epochs 20 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --project /tmp/wgisd_runs \
  --name wgisd_grape_seg \
  --output-model yolo_3d_ros/models/wgisd_grape_seg.pt \
  --exist-ok
```

This run converted 110 train images and 27 validation images. The final validation
metrics were box mAP50 0.787 and mask mAP50 0.781 on WGISD's test polygon split.
WGISD is CC BY-NC 4.0, so the fine-tuned checkpoint is separate from this
package's AGPL-3.0-only code license and should be treated as non-commercial.


## Topics

| Topic | Type |
|---|---|
| `/yolo_3d/detections` | `vision_msgs/msg/Detection3DArray` |
| `/yolo_3d/segmented_objects` | `yolo_3d_msgs/msg/SegmentedObject3DArray` |
| `/yolo_3d/segmented_points` | `sensor_msgs/msg/PointCloud2` |
| `/yolo_3d/bounding_boxes` | `vision_msgs/msg/BoundingBox3DArray` |
| `/yolo_3d/debug_image` | `sensor_msgs/msg/Image` |


## License

The yolo_3d_ros packages are AGPL-3.0-only. The Ultralytics Python package and `yolo26m-seg.pt` are governed by Ultralytics AGPL-3.0 License terms.
