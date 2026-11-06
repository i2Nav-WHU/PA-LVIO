cmake_minimum_required(VERSION 3.10)
project(lidar)

include_directories(
    ${PROJECT_SOURCE_DIR}/../)

add_library(${PROJECT_NAME} SHARED
    pointcloud.cc
    lidar_map.cc)
target_link_libraries(${PROJECT_NAME}
    stdc++fs)

## Subdirectory

# ikd-Tree
add_subdirectory(ikd-Tree)
target_link_libraries(${PROJECT_NAME} ikd-Tree)

# common
target_link_libraries(${PROJECT_NAME} common)

## Libraries

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# yaml-cpp
find_package(yaml-cpp REQUIRED)
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})

# TBB
find_package(TBB REQUIRED)
target_link_libraries(${PROJECT_NAME}
    TBB::tbb)

# abseil
find_package(absl REQUIRED)
target_link_libraries(${PROJECT_NAME}
    absl::strings
    absl::str_format)
