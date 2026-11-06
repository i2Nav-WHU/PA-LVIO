cmake_minimum_required(VERSION 3.10)
project(common)

include_directories(
    ${PROJECT_SOURCE_DIR}/../)

add_library(${PROJECT_NAME} SHARED
    logging.cc
    misc.cc
    rotation.cc
    timecost.cc)

## Libraries

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# abseil
find_package(absl REQUIRED)
target_link_libraries(${PROJECT_NAME}
    absl::strings
    absl::time
    absl::log
    absl::base
    absl::log_initialize
    absl::log_flags
    absl::log_globals
    absl::log_internal_globals)
