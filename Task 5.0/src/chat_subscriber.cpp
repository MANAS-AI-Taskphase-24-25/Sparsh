#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class ChatSubscriber : public rclcpp::Node {
public:
    ChatSubscriber() : Node("chat_subscriber") {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "chat_topic", 10,
            std::bind(&ChatSubscriber::message_callback, this, std::placeholders::_1));
    }

private:
    void message_callback(const std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received: '%s'", msg->data.c_str());
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ChatSubscriber>());
    rclcpp::shutdown();
    return 0;
}
