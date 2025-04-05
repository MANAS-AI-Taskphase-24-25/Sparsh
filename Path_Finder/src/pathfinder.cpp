#include <climits>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <queue>
#include <vector>
#include <cmath>
#include <unordered_map>

struct AStarNode {
    int x, y, g, h;
    AStarNode(int x, int y, int g, int h) : x(x), y(y), g(g), h(h) {}

    int getF() const { return g + h; }
};


struct CompareNodes {
    bool operator()(const AStarNode& a, const AStarNode& b) {
        return a.getF() > b.getF();
    }
};

class AStarPathPlanner : public rclcpp::Node {
public:
    AStarPathPlanner() : Node("a_star_path_planner") {
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10, std::bind(&AStarPathPlanner::mapCallback, this, std::placeholders::_1));

        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/path", 10);
    }

private:
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    nav_msgs::msg::OccupancyGrid::SharedPtr map_;
    std::vector<bool> grid_;
    int width_, height_;

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr map) {
        map_ = map;
        width_ = map_->info.width;
        height_ = map_->info.height;
        grid_.assign(map_->data.begin(), map_->data.end());
        runAStar();
    }

    void runAStar() {
        std::priority_queue<AStarNode, std::vector<AStarNode>, CompareNodes> openSet;
        std::unordered_map<int, int> came_from;

        int startX = 0, startY = 0, goalX = width_ - 1, goalY = height_ - 1;
        AStarNode startNode(startX, startY, 0, calculateHeuristic(startX, startY, goalX, goalY));
        openSet.push(startNode);

        std::vector<int> g_score(width_ * height_, INT_MAX);
        g_score[startY * width_ + startX] = 0;

        const int moves[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

        while (!openSet.empty()) {
            AStarNode current = openSet.top();
            openSet.pop();

            if (current.x == goalX && current.y == goalY) {
                reconstructPathAndPublish(came_from, current.x, current.y);
                return;
            }

            for (const auto& move : moves) {
                int nextX = current.x + move[0], nextY = current.y + move[1];

                if (!isValidPosition(nextX, nextY)) continue;

                int nextIndex = nextY * width_ + nextX;
                int newG = g_score[current.y * width_ + current.x] + 1;

                if (newG < g_score[nextIndex]) {
                    g_score[nextIndex] = newG;
                    came_from[nextIndex] = current.y * width_ + current.x;
                    openSet.push(AStarNode(nextX, nextY, newG, calculateHeuristic(nextX, nextY, goalX, goalY)));
                }
            }
        }

        RCLCPP_WARN(this->get_logger(), "No valid path found.");
    }

    bool isValidPosition(int x, int y) const {
        return x >= 0 && x < width_ && y >= 0 && y < height_ && grid_[y * width_ + x] == 0;
    }

    int calculateHeuristic(int x, int y, int goalX, int goalY) {
        return std::abs(x - goalX) + std::abs(y - goalY);  // Manhattan distance (faster)
    }

    void reconstructPathAndPublish(const std::unordered_map<int, int>& came_from, int goalX, int goalY) {
        auto path_msg = std::make_shared<nav_msgs::msg::Path>();
        path_msg->header.stamp = this->now();
        path_msg->header.frame_id = "map";

        int index = goalY * width_ + goalX;
        while (came_from.find(index) != came_from.end()) {
            int x = index % width_;
            int y = index / width_;
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header = path_msg->header;
            pose_stamped.pose.position.x = x * map_->info.resolution + map_->info.origin.position.x;
            pose_stamped.pose.position.y = y * map_->info.resolution + map_->info.origin.position.y;
            path_msg->poses.push_back(pose_stamped);

            index = came_from.at(index);
        }

        std::reverse(path_msg->poses.begin(), path_msg->poses.end());
        path_pub_->publish(*path_msg);

        RCLCPP_INFO(this->get_logger(), "Published path with %zu waypoints", path_msg->poses.size());
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AStarPathPlanner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
