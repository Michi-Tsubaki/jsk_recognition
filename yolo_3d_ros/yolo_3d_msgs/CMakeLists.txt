cmake_minimum_required(VERSION 3.16)
project(yolo_3d_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SegmentedObject3D.msg"
  "msg/SegmentedObject3DArray.msg"
  DEPENDENCIES sensor_msgs std_msgs vision_msgs
)

ament_export_dependencies(rosidl_default_runtime)

install(FILES
  LICENSE
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
