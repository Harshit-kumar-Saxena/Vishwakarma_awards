#!/usr/bin/env python3
"""
Ball TF Publisher Node
1. Detects white balls on a blue plate.
2. Converts 2D pixels to 3D world coordinates.
3. Publishes coordinate transforms (TF) for each ball.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_ros import TransformBroadcaster, LookupException, ConnectivityException, ExtrapolationException

class BallTFPublisher(Node):
    def __init__(self):
        super().__init__('publish_tf')
        
        self.bridge = CvBridge()
        
        # --- TF Infrastructure ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # --- Camera Subscribers ---
        self.camera_model = None
        
        self.create_subscription(
            CameraInfo, 
            '/tf_camera/camera_info', 
            self.info_callback, 
            qos_profile_sensor_data
        )
        self.create_subscription(
            Image, 
            '/tf_camera/image_raw', 
            self.image_callback, 
            qos_profile_sensor_data
        )
        
        # --- Configuration ---
        self.start_time = self.get_clock().now()
        self.warmup_duration = 5.0
        
        # --- CRITICAL FIX: RELATIVE HEIGHT ---
        # Gazebo Ball Z (1.06) - Gazebo Robot Z (1.05) = 0.01m
        # The ball is roughly at the same height as the robot base.
        self.ball_z_height = 0.01
        
        self.get_logger().info(f'Ball TF Publisher initialized. Waiting {self.warmup_duration}s for sim stability...')

    def info_callback(self, msg):
        """Store camera intrinsic parameters"""
        self.camera_model = msg

    def image_callback(self, msg):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed < self.warmup_duration:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            processed_img, balls_2d = self.detect_balls_on_plate(cv_image)
            
            # Sort balls by X-coordinate
            balls_2d.sort(key=lambda b: b[0])
            
            if self.camera_model is not None:
                for i, (u, v, radius) in enumerate(balls_2d):
                    self.publish_transform(u, v, i, msg.header)
            else:
                self.get_logger().warn('Waiting for Camera Info...', throttle_duration_sec=2.0)

            # Visualization
            cv2.imshow("Ball TF Publisher", processed_img)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'CV Error: {str(e)}')

    def detect_balls_on_plate(self, image):
        detected_balls = [] 
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # --- Step A: Find Blue Plate ---
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask_plate = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((5, 5), np.uint8)
        mask_plate = cv2.morphologyEx(mask_plate, cv2.MORPH_OPEN, kernel)
        mask_plate = cv2.morphologyEx(mask_plate, cv2.MORPH_CLOSE, kernel)

        plate_contours, _ = cv2.findContours(mask_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_plate_contour = None
        if plate_contours:
            largest_plate_contour = max(plate_contours, key=cv2.contourArea)
            cv2.drawContours(image, [largest_plate_contour], -1, (255, 0, 0), 2) 

        # --- Step B: Find White Objects ---
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

        white_contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if white_contours and largest_plate_contour is not None:
            for contour in white_contours:
                area = cv2.contourArea(contour)
                if area > 50:
                    ((x, y), radius) = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    
                    dist = cv2.pointPolygonTest(largest_plate_contour, center, False)
                    if dist >= 0:
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter == 0: continue
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        
                        if circularity > 0.7:
                            detected_balls.append((int(x), int(y), radius))
                            cv2.circle(image, center, int(radius), (0, 255, 0), 2)
                            cv2.circle(image, center, 3, (0, 0, 255), -1)
                            cv2.putText(image, f"Ball", (center[0]+10, center[1]), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image, detected_balls

    def publish_transform(self, u, v, ball_id, header):
        # 1. Camera Intrinsics
        k = self.camera_model.k
        fx, fy = k[0], k[4]
        cx, cy = k[2], k[5]
        
        # 2. 3D Ray in Camera Frame
        ray_x = (u - cx) / fx
        ray_y = (v - cy) / fy
        ray_z = 1.0
        ray_camera = np.array([ray_x, ray_y, ray_z])

        # 3. Get Camera Pose
        # We use base_link so the coordinates are relative to the robot base (which is at 0,0,0 in TF)
        target_frame = 'base_link' 
        source_frame = 'tf_camera_optical_link' # Manual override for Gazebo frame name
        
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            
            # Position
            t_vec = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            
            # Rotation
            q = trans.transform.rotation
            q_w, q_x, q_y, q_z = q.w, q.x, q.y, q.z
            R_mat = np.array([
                [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
                [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
                [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
            ])
            
            # 4. Transform Ray to Target Frame
            ray_world = np.dot(R_mat, ray_camera)
            
            # 5. Ray-Plane Intersection
            if abs(ray_world[2]) < 1e-6: return 
            
            # Now calculating using relative height (0.01) vs camera relative height (~0.8)
            alpha = (self.ball_z_height - t_vec[2]) / ray_world[2]
            
            # Check: Camera (~0.8) > Ball (0.01). Ray Z is negative.
            # (0.01 - 0.8) / (-) = (+) --> This should now pass!
            if alpha < 0: return 
            
            intersection = t_vec + (alpha * ray_world)
            
            # 6. Broadcast
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = target_frame
            t.child_frame_id = f'ball_{ball_id}'
            
            t.transform.translation.x = intersection[0]
            t.transform.translation.y = intersection[1]
            t.transform.translation.z = intersection[2]
            t.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(t)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'TF Error: Could not transform {source_frame} to {target_frame}: {e}', throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = BallTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()