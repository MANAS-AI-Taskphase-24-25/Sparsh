#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <iostream>

class ChatPublisher : public rclcpp::Node {
public:
    ChatPublisher() : Node("chat_publisher") {
        publisher_ = this->create_publisher<std_msgs::msg::String>("chat_topic", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&ChatPublisher::publish_message, this));
    }

private:
    void publish_message() {
        std::string input;
        std::cout << "Enter message: ";
        std::getline(std::cin, input);

        auto message = std_msgs::msg::String();
        message.data = input;
        publisher_->publish(message);
        RCLCPP_INFO(this->get_logger(), "Sent: '%s'", input.c_str());
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ChatPublisher>());
    rclcpp::shutdown();
    return 0;
}
