cmake_minimum_required(VERSION 3.16)

foreach(required_variable MODEL_PATH MODEL_SHA256)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} must be provided")
  endif()
endforeach()

string(TOLOWER "${MODEL_SHA256}" expected_sha256)
get_filename_component(model_directory "${MODEL_PATH}" DIRECTORY)
file(MAKE_DIRECTORY "${model_directory}")

# The target runs on every build, but a valid cached file is reused. This keeps
# incremental colcon builds fast while still detecting a corrupted partial file.
if(EXISTS "${MODEL_PATH}")
  file(SHA256 "${MODEL_PATH}" existing_sha256)
  string(TOLOWER "${existing_sha256}" existing_sha256)
  if(existing_sha256 STREQUAL expected_sha256)
    file(SIZE "${MODEL_PATH}" existing_size)
    message(STATUS
      "[yolo_3d_ros] Reusing verified model: ${MODEL_PATH} (${existing_size} bytes)"
    )
    return()
  endif()

  message(WARNING
    "[yolo_3d_ros] Removing model with unexpected SHA-256: ${MODEL_PATH}\n"
    "  expected: ${expected_sha256}\n"
    "  actual:   ${existing_sha256}"
  )
  file(REMOVE "${MODEL_PATH}")
endif()

set(partial_path "${MODEL_PATH}.part")
file(REMOVE "${partial_path}")

if(DEFINED MODEL_SOURCE AND NOT "${MODEL_SOURCE}" STREQUAL "")
  if(NOT EXISTS "${MODEL_SOURCE}")
    message(FATAL_ERROR "Local model source does not exist: ${MODEL_SOURCE}")
  endif()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy "${MODEL_SOURCE}" "${partial_path}"
    RESULT_VARIABLE copy_result
    ERROR_VARIABLE copy_error
  )
  if(NOT copy_result EQUAL 0)
    file(REMOVE "${partial_path}")
    message(FATAL_ERROR
      "Failed to stage local YOLO model from ${MODEL_SOURCE}: ${copy_error}"
    )
  endif()
  set(stage_method "local file ${MODEL_SOURCE}")
else()
  if(NOT DEFINED MODEL_URL OR "${MODEL_URL}" STREQUAL "")
    message(FATAL_ERROR "MODEL_URL must be provided when MODEL_SOURCE is empty")
  endif()

  if(DEFINED CURL_EXECUTABLE
      AND NOT "${CURL_EXECUTABLE}" STREQUAL ""
      AND EXISTS "${CURL_EXECUTABLE}")
    execute_process(
      COMMAND "${CURL_EXECUTABLE}"
        --fail
        --location
        --silent
        --show-error
        --retry 5
        --retry-all-errors
        --retry-delay 2
        --connect-timeout 30
        --max-time 900
        --output "${partial_path}"
        "${MODEL_URL}"
      RESULT_VARIABLE download_result
      ERROR_VARIABLE download_error
    )
    set(stage_method "curl")
  elseif(DEFINED WGET_EXECUTABLE
         AND NOT "${WGET_EXECUTABLE}" STREQUAL ""
         AND EXISTS "${WGET_EXECUTABLE}")
    execute_process(
      COMMAND "${WGET_EXECUTABLE}"
        --tries=5
        --timeout=30
        "--output-document=${partial_path}"
        "${MODEL_URL}"
      RESULT_VARIABLE download_result
      ERROR_VARIABLE download_error
    )
    set(stage_method "wget")
  else()
    message(FATAL_ERROR
      "Neither curl nor wget is available. Install curl, or build with "
      "-DYOLO_3D_DOWNLOAD_MODEL=OFF "
      "-DYOLO_3D_MODEL_SOURCE=/absolute/path/to/yolo26m-seg.pt"
    )
  endif()

  if(NOT download_result EQUAL 0)
    file(REMOVE "${partial_path}")
    message(FATAL_ERROR
      "Failed to download ${MODEL_URL} with ${stage_method}: ${download_error}"
    )
  endif()
endif()

if(NOT EXISTS "${partial_path}")
  message(FATAL_ERROR "Model staging did not create ${partial_path}")
endif()

file(SHA256 "${partial_path}" actual_sha256)
string(TOLOWER "${actual_sha256}" actual_sha256)
if(NOT actual_sha256 STREQUAL expected_sha256)
  file(REMOVE "${partial_path}")
  message(FATAL_ERROR
    "Downloaded/staged YOLO model failed SHA-256 verification.\n"
    "  expected: ${expected_sha256}\n"
    "  actual:   ${actual_sha256}"
  )
endif()

file(SIZE "${partial_path}" model_size)
file(RENAME "${partial_path}" "${MODEL_PATH}")
message(STATUS
  "[yolo_3d_ros] Staged verified model with ${stage_method}: "
  "${MODEL_PATH} (${model_size} bytes)"
)
