import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import yaml
import os
from rclpy.qos import qos_profile_sensor_data

# ROS2 패키지 내부 config 경로 robust하게 찾기 위한 임포트
from ament_index_python.packages import get_package_share_directory

class LidarToCamProj(Node):
    def __init__(self):
        super().__init__('lidar2cam_projector')

        # === robust하게 config 경로 찾기 ===
        pkg_share = get_package_share_directory('lidar2cam_projector')
        int_file = os.path.join(pkg_share, 'config', 'fisheye_calib.yaml')
        ext_file = os.path.join(pkg_share, 'config', 'extrinsic.yaml')

        # === 내부 파라미터 로드 ===
        with open(int_file) as f:
            calib = yaml.safe_load(f)
        self.K = np.array(calib['K'])
        self.D = np.array(calib['D']) if 'D' in calib else None

        # === 외부 파라미터 로드 ===
        with open(ext_file) as f:
            ext = yaml.safe_load(f)
        self.R = np.array(ext['R'])        # (3,3)
        self.t = np.array(ext['t']).reshape((3,1))  # (3,1)

        self.bridge = CvBridge()
        self.last_scan_points = None
        self.img = None

        # 구독 설정
        self.create_subscription(Image, '/image_raw', self.img_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)

    def scan_cb(self, msg):
        # /scan → (N,3)
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        mask = (ranges > msg.range_min) & (ranges < msg.range_max)
        x = ranges[mask] * np.cos(angles[mask])
        y = ranges[mask] * np.sin(angles[mask])
        points = np.stack([x, y, np.zeros_like(x)], axis=1)  # (N,3)
        self.last_scan_points = points

    def img_cb(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.last_scan_points is not None:
            proj_img = self.overlay_lidar_on_img(self.img.copy(), self.last_scan_points)
            cv2.imshow('LiDAR on Camera', proj_img)
            cv2.waitKey(1)

    def overlay_lidar_on_img(self, img, lidar_points):
        cam_pts = (self.R @ lidar_points.T + self.t).T  # (N,3)
        img_pts, _ = cv2.projectPoints(
            cam_pts, np.zeros(3), np.zeros(3), self.K, self.D if self.D is not None else None
        )
        img_pts = img_pts.squeeze()
        for pt in img_pts:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        return img

def main():
    rclpy.init()
    node = LidarToCamProj()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
