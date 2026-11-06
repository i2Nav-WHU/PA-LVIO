cmake_minimum_required(VERSION 3.10)
project(pa_lvio)

## Build Options

set(CMAKE_CXX_STANDARD 17)
if (CMAKE_BUILD_TYPE STREQUAL Release)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
endif()

## pa_lvio

# lvio Library
add_subdirectory(lvio)

## Executable

# ROS interface
find_package(catkin REQUIRED COMPONENTS
    message_generation roscpp std_msgs pcl_conversions livox_ros_driver)
catkin_package(
    LIBRARIES
    CATKIN_DEPENDS roscpp std_msgs pcl_conversions livox_ros_driver)
include_directories(
    ${catkin_INCLUDE_DIRS})

# pa_lvio ROS
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/lvio)
add_executable(${PROJECT_NAME}_ros
    ROS/main.cc
    ROS/fusion.cc
    ROS/visual/visual_drawer_rviz.cc
    ROS/lidar/lidar_converter.cc
    ROS/lidar/lidar_viewer_rviz.cc)

target_link_libraries(${PROJECT_NAME}_ros
    ${catkin_LIBRARIES}
    lvio
    stdc++fs)

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}_ros ${PCL_LIBRARIES})

# abseil
find_package(absl REQUIRED)
target_link_libraries(${PROJECT_NAME}_ros
    absl::strings
    absl::time)
