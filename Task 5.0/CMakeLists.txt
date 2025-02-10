cmake_minimum_required(VERSION 3.5)
project(ros2_chat)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(chat_publisher src/chat_publisher.cpp)
ament_target_dependencies(chat_publisher rclcpp std_msgs)

add_executable(chat_subscriber src/chat_subscriber.cpp)
ament_target_dependencies(chat_subscriber rclcpp std_msgs)

install(TARGETS
  chat_publisher
  chat_subscriber
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

