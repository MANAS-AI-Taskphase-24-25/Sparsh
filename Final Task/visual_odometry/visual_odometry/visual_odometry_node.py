import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Pose, Twist, PoseStamped
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import math
from tf_transformations import quaternion_from_matrix
import glob

MAX_FRAME = 25761
MIN_NUM_FEAT = 2200

class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odometry_node')
        
        # Parameters
        self.declare_parameter('image_dir', '')
        self.declare_parameter('focal', 718.856)
        self.declare_parameter('pp_x', 607.1928)
        self.declare_parameter('pp_y', 185.2157)
        self.declare_parameter('show_trajectory', False)
        self.declare_parameter('frame_rate', 10.0)
        self.declare_parameter('path_offset_x', 0.0)
        self.declare_parameter('path_offset_z', 0.0)
        
        self.image_dir = self.get_parameter('image_dir').get_parameter_value().string_value
        self.focal = self.get_parameter('focal').get_parameter_value().double_value
        self.pp = (self.get_parameter('pp_x').get_parameter_value().double_value,
                  self.get_parameter('pp_y').get_parameter_value().double_value)
        self.show_trajectory = self.get_parameter('show_trajectory').get_parameter_value().bool_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().double_value
        self.path_offset_x = self.get_parameter('path_offset_x').get_parameter_value().double_value
        self.path_offset_z = self.get_parameter('path_offset_z').get_parameter_value().double_value
        
        # Validate image directory
        if not os.path.isdir(self.image_dir):
            self.get_logger().error(f'Image directory {self.image_dir} does not exist')
            rclpy.shutdown()
            return
        
        # Get sorted list of image files
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        if not self.image_files:
            self.get_logger().error(f'No PNG images found in {self.image_dir}')
            rclpy.shutdown()
            return
        
        # ROS2 components
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.path_pub = self.create_publisher(Path, '/path', 10)
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)
        
        # Visual odometry variables
        self.prev_img = None
        self.prev_pts = None
        self.R_f = None
        self.t_f = None
        self.rotation_buffer = []
        self.frame_count = 20
        self.path = Path()
        self.path.header.frame_id = 'odom'
        
        # Trajectory visualization
        if self.show_trajectory:
            self.traj = np.zeros((1200, 1200, 3), dtype=np.uint8)
            self.font = cv2.FONT_HERSHEY_PLAIN
        
        # Timer for processing images
        self.timer = self.create_timer(1.0 / self.frame_rate, self.process_image)
        
        self.get_logger().info(f'Initialized with {len(self.image_files)} images from {self.image_dir}')
    
    def feature_detection(self, img):
        fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        keypoints = fast.detect(img, None)
        return cv2.KeyPoint_convert(keypoints)
    
    def feature_tracking(self, img1, img2, points1):
        lk_params = dict(winSize=(21, 21),
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        points2, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None, **lk_params)
        points1_good = points1[status.flatten() == 1]
        points2_good = points2[status.flatten() == 1]
        return points1_good, points2_good
    
    def process_image(self):
        if self.frame_count >= len(self.image_files) or self.frame_count >= MAX_FRAME:
            self.get_logger().info('Reached maximum frame count or end of images, stopping.')
            self.timer.cancel()
            if self.show_trajectory:
                cv2.destroyAllWindows()
            return

        # Read current image
        img_path = self.image_files[self.frame_count]
        curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if curr_img is None:
            self.get_logger().error(f'Failed to read image: {img_path}')
            self.frame_count += 1
            return

        # Publish image
        img_msg = self.bridge.cv2_to_imgmsg(curr_img, encoding='mono8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera'
        self.image_pub.publish(img_msg)

        if self.prev_img is None:
            self.prev_img = curr_img
            self.prev_pts = self.feature_detection(curr_img)
            self.R_f = np.eye(3)
            self.t_f = np.zeros((3, 1))
            self.rotation_buffer = [self.R_f]
            self.frame_count += 1
            return

        p0, p1 = self.feature_tracking(self.prev_img, curr_img, self.prev_pts)
        E, mask = cv2.findEssentialMat(p1, p0, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, p1, p0, focal=self.focal, pp=self.pp)
        
        self.rotation_buffer.append(self.R_f.copy())
        if len(self.rotation_buffer) > 5:
            self.rotation_buffer.pop(0)

        R_ref = self.rotation_buffer[0]
        R_delta = np.dot(self.R_f, R_ref.T)
        angle_rad = np.arccos(np.clip((np.trace(R_delta) - 1) / 2.0, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
    
        scale = 1.0
        if angle_deg > 130.0:
            self.get_logger().warn(f'Skipping frame {self.frame_count} due to large rotation angle: {angle_deg:.2f}°')
            self.prev_img = curr_img
            self.prev_pts = self.feature_detection(curr_img)
            self.frame_count += 1
            return
        self.get_logger().info(f'Frame {self.frame_count}: ΔRotation = {angle_deg:.2f}°, Scale = {scale}, t = {t.flatten()}')
        # Example: artificially scale the angle of rotation
        angle_scale_factor = 1  # You can tweak this to 1.5, 2.0, etc.

        # Convert R to axis-angle
        rvec, _ = cv2.Rodrigues(R)
        rvec = rvec * angle_scale_factor  # Scale the rotation vector
        R, _ = cv2.Rodrigues(rvec)  # Convert back to rotation matrix

        if scale > 0.0:
            self.t_f = self.t_f + scale * np.dot(self.R_f, t)
            self.R_f = np.dot(R, self.R_f)

        self.get_logger().info(f't_f: x={self.t_f[0][0]:.2f}, y={self.t_f[1][0]:.2f}, z={self.t_f[2][0]:.2f}')

        if len(p0) < MIN_NUM_FEAT:
            self.prev_pts = self.feature_detection(self.prev_img)
            self.prev_pts, _ = self.feature_tracking(self.prev_img, curr_img, self.prev_pts)
        else:
            self.prev_pts = p1

        # ---- Coordinate transform: OpenCV to ROS ----
        t_cv = self.t_f.flatten()
        t_ros = np.array([
            t_cv[2],     # x ← z (forward)
            -t_cv[0],    # y ← -x (right → left)
            t_cv[1]      # z ← y (down → up)
        ])

        # ---- Odometry message ----
        odom_msg = Odometry()
        odom_msg.header.stamp = img_msg.header.stamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        homog = np.eye(4)
        homog[:3, :3] = self.R_f
        homog[:3, 3] = self.t_f.flatten()
        q = quaternion_from_matrix(homog)

        odom_msg.pose.pose.position.x = float(t_ros[0])
        odom_msg.pose.pose.position.y = float(t_ros[1])
        odom_msg.pose.pose.position.z = float(t_ros[2])
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        odom_msg.twist.twist = Twist()
        self.odom_pub.publish(odom_msg)

        # ---- TF transform ----
        tf_msg = TFMessage()
        transform = TransformStamped()
        transform.header.stamp = img_msg.header.stamp
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = float(t_ros[0])
        transform.transform.translation.y = float(t_ros[1])
        transform.transform.translation.z = float(t_ros[2])
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]
        tf_msg.transforms.append(transform)
        self.tf_pub.publish(tf_msg)

        # ---- Path publishing (also in x-y plane of ROS) ----
        pose = PoseStamped()
        pose.header.stamp = img_msg.header.stamp
        pose.header.frame_id = 'odom'
        pose.pose.position.x = float(t_ros[0]) + self.path_offset_x
        pose.pose.position.y = float(t_ros[1])
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0  # Identity quaternion
        self.path.poses.append(pose)
        self.path.header.stamp = img_msg.header.stamp
        self.path_pub.publish(self.path)

        self.get_logger().info(f'Path pose: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}, z={pose.pose.position.z:.2f}')

        self.prev_img = curr_img
        self.frame_count += 1

        # if self.show_trajectory:
        #     x = int(self.t_f[0]) + 500
        #     y = int(-self.t_f[2]) + 500
        #     cv2.circle(self.traj, (x, y), 1, (255, 0, 0), 2)
        #     text = f"x = {self.t_f[0][0]:.2f}m y = {self.t_f[1][0]:.2f}m z = {self.t_f[2][0]:.2f}m"
        #     cv2.rectangle(self.traj, (10, 30), (550, 50), (0, 0, 0), -1)
        #     cv2.putText(self.traj, text, (20, 45), self.font, 1, (255, 255, 255), 1)
        #     cv2.imshow("Trajectory", self.traj)
        #     if cv2.waitKey(1) == 27:
        #         cv2.destroyAllWindows()

    
    def destroy_node(self):
        if self.show_trajectory:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()