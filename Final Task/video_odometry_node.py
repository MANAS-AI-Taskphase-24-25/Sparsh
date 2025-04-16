import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf_transformations import quaternion_from_euler
from nav_msgs.msg import Odometry


class VideoOdometry(Node):
    def __init__(self):
        super().__init__('video_odometry_node')

        # Publishers
        self.path_publisher = self.create_publisher(Path, '/vo_path', 10)
        self.image_publisher = self.create_publisher(Image, '/vo_image', 10)
        self.odom_publisher = self.create_publisher(Odometry, '/vo_odom', 10)


        # Path message setup
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

        # Video capture
        self.video_path = '/home/sparsh/ros2_tp/src/environment/utils/video.mp4'
        self.cap = cv2.VideoCapture(self.video_path)

        # Feature detector
        self.feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.prev_gray = None
        self.prev_pts = None
        self.x, self.y, self.theta = 0.0, 0.0, 0.0  # Initial position and orientation

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.05, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video.")
            self.cap.release()
            return

        #frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return

        next_pts, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        good_prev = self.prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) >= 6:
            M, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)

            if M is not None:
                dx, dy = M[0, 2], M[1, 2]
                dtheta = np.arctan2(M[1, 0], M[0, 0])

                scale = 0.1  # Empirical scaling factor
                self.x += scale * (np.cos(self.theta) * dx - np.sin(self.theta) * dy)
                self.y += scale * (np.sin(self.theta) * dx + np.cos(self.theta) * dy)
                self.theta += dtheta

                qx, qy, qz, qw = quaternion_from_euler(0, 0, self.theta)

                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = self.x
                pose.pose.position.y = self.y
                pose.pose.position.z = 0.0
                pose.pose.orientation.x = qx
                pose.pose.orientation.y = qy
                pose.pose.orientation.z = qz
                pose.pose.orientation.w = qw

                self.path_msg.header.stamp = pose.header.stamp
                self.path_msg.poses.append(pose)
                self.path_publisher.publish(self.path_msg)
                odom = Odometry()
                odom.header.stamp = pose.header.stamp
                odom.header.frame_id = 'map'
                odom.child_frame_id = 'base_link'

                # Position
                odom.pose.pose.position.x = self.x
                odom.pose.pose.position.y = self.y
                odom.pose.pose.position.z = 0.0
                odom.pose.pose.orientation.x = qx
                odom.pose.pose.orientation.y = qy
                odom.pose.pose.orientation.z = qz
                odom.pose.pose.orientation.w = qw

                # No velocity data in this example (can be estimated if needed)
                odom.twist.twist.linear.x = 0.0
                odom.twist.twist.linear.y = 0.0
                odom.twist.twist.linear.z = 0.0
                odom.twist.twist.angular.x = 0.0
                odom.twist.twist.angular.y = 0.0
                odom.twist.twist.angular.z = 0.0

                self.odom_publisher.publish(odom)


                # Publish frame
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                img_msg.header.stamp = pose.header.stamp
                img_msg.header.frame_id = "camera_frame"
                self.image_publisher.publish(img_msg)

        # Update for next iteration
        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)


def main(args=None):
    rclpy.init(args=args)
    node = VideoOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

