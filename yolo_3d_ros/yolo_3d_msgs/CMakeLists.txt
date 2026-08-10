# Fall back to ROS1 when $ROS_VERSION is unset (pre-ros_environment distros,
# e.g. indigo); every ROS2 environment always sets it via ros_environment.
if("$ENV{ROS_VERSION}" STREQUAL "")
  set(ENV{ROS_VERSION} 1)
endif()

if("$ENV{ROS_VERSION}" EQUAL 1)
  cmake_minimum_required(VERSION 2.8.12)
elseif("$ENV{ROS_VERSION}" EQUAL 2)
  cmake_minimum_required(VERSION 3.16)
else()
  message(FATAL_ERROR "Unsupported ROS_VERSION: $ENV{ROS_VERSION}")
endif()

project(yolo_3d_msgs)

if("$ENV{ROS_VERSION}" EQUAL 1)
  find_package(catkin QUIET)
  if(catkin_FOUND)
    catkin_package()
  endif()
  message(STATUS "${PROJECT_NAME} is a ROS2 package; skipping ROS1 build.")
  return()
endif()

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
