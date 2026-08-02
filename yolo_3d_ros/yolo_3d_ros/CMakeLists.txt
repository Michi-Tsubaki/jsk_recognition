cmake_minimum_required(VERSION 3.16)
project(yolo_3d_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(yolo_3d_msgs REQUIRED)
find_program(UV_EXECUTABLE NAMES uv REQUIRED)

set(
  YOLO_3D_VENV_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/.venv"
  CACHE PATH
  "Virtual environment used by the installed yolo_3d_ros executables"
)
set(YOLO_3D_VENV_PYTHON "${YOLO_3D_VENV_DIR}/bin/python")

add_custom_target(prepare_yolo_3d_python_env ALL
  COMMAND "${UV_EXECUTABLE}" venv
    --allow-existing
    --system-site-packages
    --python-preference only-system
    "${YOLO_3D_VENV_DIR}"
  COMMAND "${UV_EXECUTABLE}" sync
    --project "${CMAKE_CURRENT_SOURCE_DIR}"
    --python "${YOLO_3D_VENV_PYTHON}"
    --group test
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Preparing yolo_3d_ros Python environment with uv"
  VERBATIM
)

# Pin the default checkpoint to an immutable Ultralytics release asset and verify it
# before installation. The checkpoint remains subject to Ultralytics' own license.
set(YOLO_3D_MODEL_FILENAME "yolo26m-seg.pt")
set(
  YOLO_3D_MODEL_URL
  "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt"
  CACHE STRING
  "URL used to download the default YOLO segmentation checkpoint"
)
set(
  YOLO_3D_MODEL_SHA256
  "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
  CACHE STRING
  "Expected SHA-256 for the default YOLO segmentation checkpoint"
)
option(
  YOLO_3D_DOWNLOAD_MODEL
  "Download and install the pinned YOLO checkpoint during the build"
  ON
)
set(
  YOLO_3D_MODEL_SOURCE
  ""
  CACHE FILEPATH
  "Optional local yolo26m-seg.pt used instead of downloading (for offline builds)"
)

set(YOLO_3D_MODEL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/models")
set(
  YOLO_3D_MODEL_BUILD_PATH
  "${YOLO_3D_MODEL_BUILD_DIR}/${YOLO_3D_MODEL_FILENAME}"
)

if(YOLO_3D_MODEL_SOURCE OR YOLO_3D_DOWNLOAD_MODEL)
  find_program(CURL_EXECUTABLE NAMES curl)
  find_program(WGET_EXECUTABLE NAMES wget)

  if(NOT YOLO_3D_MODEL_SOURCE AND NOT CURL_EXECUTABLE AND NOT WGET_EXECUTABLE)
    message(FATAL_ERROR
      "YOLO_3D_DOWNLOAD_MODEL=ON requires curl or wget. Install curl, or pass "
      "-DYOLO_3D_DOWNLOAD_MODEL=OFF "
      "-DYOLO_3D_MODEL_SOURCE=/absolute/path/to/yolo26m-seg.pt"
    )
  endif()

  add_custom_target(download_yolo_3d_model ALL
    BYPRODUCTS "${YOLO_3D_MODEL_BUILD_PATH}"
    COMMAND "${CMAKE_COMMAND}"
      "-DMODEL_URL=${YOLO_3D_MODEL_URL}"
      "-DMODEL_SOURCE=${YOLO_3D_MODEL_SOURCE}"
      "-DMODEL_PATH=${YOLO_3D_MODEL_BUILD_PATH}"
      "-DMODEL_SHA256=${YOLO_3D_MODEL_SHA256}"
      "-DCURL_EXECUTABLE=${CURL_EXECUTABLE}"
      "-DWGET_EXECUTABLE=${WGET_EXECUTABLE}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/stage_model.cmake"
    COMMENT
      "Staging ${YOLO_3D_MODEL_FILENAME} for yolo_3d_ros"
    VERBATIM
  )

  install(
    FILES "${YOLO_3D_MODEL_BUILD_PATH}"
    DESTINATION "share/${PROJECT_NAME}/models"
  )
else()
  message(WARNING
    "No default YOLO model will be installed because "
    "YOLO_3D_DOWNLOAD_MODEL=OFF and YOLO_3D_MODEL_SOURCE is empty. "
    "The model launch argument must point to an existing checkpoint."
  )
endif()

ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR src/${PROJECT_NAME}
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/scripts")
set(YOLO_3D_SCRIPT_WRAPPERS)
foreach(script_name IN ITEMS
    open3d_object_viewer
    sample_pointcloud_publisher
    segmented_object_listener
    yolo_3d_node
)
  set(SCRIPT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/scripts/${script_name}")
  configure_file(
    cmake/python_entrypoint.in
    "${CMAKE_CURRENT_BINARY_DIR}/scripts/${script_name}"
    @ONLY
  )
  execute_process(
    COMMAND chmod 755 "${CMAKE_CURRENT_BINARY_DIR}/scripts/${script_name}"
  )
  list(APPEND YOLO_3D_SCRIPT_WRAPPERS "${CMAKE_CURRENT_BINARY_DIR}/scripts/${script_name}")
endforeach()

install(PROGRAMS
  ${YOLO_3D_SCRIPT_WRAPPERS}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  docs
  launch
  models
  rviz
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  LICENSE
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_pytest_test(test_core test/test_core.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  ament_add_pytest_test(test_pointcloud_conversion test/test_pointcloud_conversion.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  ament_add_pytest_test(test_inference test/test_inference.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  ament_add_pytest_test(test_model_stage test/test_model_stage.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  ament_add_pytest_test(test_sample_sequence test/test_sample_sequence.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  ament_add_pytest_test(test_ros_messages test/test_ros_messages.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
endif()

ament_package()
