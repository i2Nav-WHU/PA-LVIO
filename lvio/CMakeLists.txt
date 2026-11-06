cmake_minimum_required(VERSION 3.10)
project(lvio)

# Build LVIO library
include_directories(
    ${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE LVIO_SOURCE
    lvio/lvio.cc
    lvio/optimizer.cc)
add_library(lvio SHARED
    ${LVIO_SOURCE})

## Subdirectory

# fileio
add_subdirectory(fileio)
target_link_libraries(${PROJECT_NAME} fileio)

# preintegration
add_subdirectory(preintegration)
target_link_libraries(${PROJECT_NAME} preintegration)

# visual
add_subdirectory(visual)
target_link_libraries(${PROJECT_NAME} visual)

# lidar
add_subdirectory(lidar)
target_link_libraries(${PROJECT_NAME} lidar)

# factors
add_subdirectory(factors)
target_link_libraries(${PROJECT_NAME} factors)

# common
add_subdirectory(common)
target_link_libraries(${PROJECT_NAME} common)

## Libraries

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# yaml-cpp
find_package(yaml-cpp REQUIRED)
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})

# Ceres
find_package(Ceres REQUIRED)
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

# TBB
find_package(TBB REQUIRED)
target_link_libraries(${PROJECT_NAME}
    TBB::tbb)

# abseil
find_package(absl REQUIRED)
target_link_libraries(${PROJECT_NAME}
    absl::strings
    absl::str_format)
