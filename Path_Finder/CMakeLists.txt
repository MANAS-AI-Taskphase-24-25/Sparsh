cmake_minimum_required(VERSION 3.5)
project(Path_Finder)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(pathfinder src/pathfinder.cpp)
ament_target_dependencies(pathfinder rclcpp nav_msgs geometry_msgs)

install(TARGETS
  pathfinder
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
