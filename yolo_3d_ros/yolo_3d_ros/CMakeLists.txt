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

project(yolo_3d_ros)

if("$ENV{ROS_VERSION}" EQUAL 1)
  find_package(catkin QUIET)
  if(catkin_FOUND)
    catkin_package()
  endif()
  message(STATUS "${PROJECT_NAME} is a ROS2 package. Skipping ROS1 build.")
  return()
endif()

set(_YOLO_3D_PYTHON3_CANDIDATE "${Python3_EXECUTABLE}")
if(NOT _YOLO_3D_PYTHON3_CANDIDATE)
  find_program(_YOLO_3D_PYTHON3_CANDIDATE NAMES python3)
endif()
if(_YOLO_3D_PYTHON3_CANDIDATE)
  execute_process(
    COMMAND "${_YOLO_3D_PYTHON3_CANDIDATE}" -c "import catkin_pkg"
    RESULT_VARIABLE _YOLO_3D_PYTHON3_HAS_CATKIN_PKG
    OUTPUT_QUIET
    ERROR_QUIET
  )
else()
  set(_YOLO_3D_PYTHON3_HAS_CATKIN_PKG 1)
endif()
if(NOT _YOLO_3D_PYTHON3_HAS_CATKIN_PKG EQUAL 0)
  find_program(_YOLO_3D_SYSTEM_PYTHON3 NAMES python3 HINTS /usr/bin /bin NO_DEFAULT_PATH)
  if(_YOLO_3D_SYSTEM_PYTHON3)
    execute_process(
      COMMAND "${_YOLO_3D_SYSTEM_PYTHON3}" -c "import catkin_pkg"
      RESULT_VARIABLE _YOLO_3D_SYSTEM_PYTHON3_HAS_CATKIN_PKG
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(_YOLO_3D_SYSTEM_PYTHON3_HAS_CATKIN_PKG EQUAL 0)
      set(Python3_EXECUTABLE "${_YOLO_3D_SYSTEM_PYTHON3}"
        CACHE FILEPATH "Python interpreter used by ament_cmake" FORCE
      )
      message(STATUS
        "Using ${Python3_EXECUTABLE} for ament_cmake because "
        "${_YOLO_3D_PYTHON3_CANDIDATE} cannot import catkin_pkg"
      )
    endif()
  endif()
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(yolo_3d_msgs REQUIRED)

option(
  YOLO_3D_INSTALL_UV_IF_MISSING
  "Install uv with Astral's installer if uv is not found"
  ON
)

find_program(UV_EXECUTABLE NAMES uv)
if(NOT UV_EXECUTABLE)
  if(NOT YOLO_3D_INSTALL_UV_IF_MISSING)
    message(FATAL_ERROR
      "uv was not found. Install uv, or enable YOLO_3D_INSTALL_UV_IF_MISSING."
    )
  endif()

  find_program(CURL_EXECUTABLE NAMES curl)
  if(NOT CURL_EXECUTABLE)
    message(FATAL_ERROR
      "uv was not found and curl is required to install it. "
      "Install curl or install uv manually."
    )
  endif()

  message(STATUS "uv was not found; installing uv with Astral's installer")
  execute_process(
    COMMAND sh -c "\"${CURL_EXECUTABLE}\" -LsSf https://astral.sh/uv/install.sh | sh"
    RESULT_VARIABLE UV_INSTALL_RESULT
  )
  if(NOT UV_INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Failed to install uv with Astral's installer "
      "(exit code: ${UV_INSTALL_RESULT})."
    )
  endif()

  unset(UV_EXECUTABLE CACHE)
  find_program(UV_EXECUTABLE
    NAMES uv
    HINTS "$ENV{UV_INSTALL_DIR}" "$ENV{XDG_BIN_HOME}" "$ENV{HOME}/.local/bin"
  )
  if(NOT UV_EXECUTABLE)
    message(FATAL_ERROR
      "uv was installed, but the uv executable was not found. "
      "Add the uv install directory to PATH and rerun CMake."
    )
  endif()
endif()

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
    yolo_3d_node
)
  configure_file(
    "scripts/${script_name}"
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
  launch
  models
  rviz
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
  ament_add_pytest_test(test_ros_messages test/test_ros_messages.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
endif()

ament_package()
