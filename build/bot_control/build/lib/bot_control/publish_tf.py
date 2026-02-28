#!/usr/bin/env python3
"""
Ball TF Publisher with Transform Caching
Publishes ball positions as TF frames, with caching to handle camera occlusion
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_ros import TransformBroadcaster
import struct

class BallTFPublisher(Node):
    def __init__(self):
        super().__init__('publish_tf')
        self.bridge = CvBridge()
        
        # TF infrastructure
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Sensor data
        self.latest_pointcloud = None
        self.pointcloud_width = 640
        self.pointcloud_height = 480
        self.camera_info = None
        
        # Transform caching (key feature for occlusion handling)
        self.cached_transforms = {}  # {frame_name: (x, y, z, timestamp)}
        self.transform_timeout = 10.0  # Cache lifetime in seconds
        
        # Statistics
        self.pc_count = 0
        self.img_count = 0
        self.last_ball_count = 0
        
        # Startup delay to let Gazebo stabilize
        self.start_time = self.get_clock().now()
        self.warmup_duration = 5.0
        
        # QoS for sensor topics (best effort for simulation)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.pc_sub = self.create_subscription(
            PointCloud2, '/tf_camera/points', 
            self.pointcloud_callback, sensor_qos
        )
        
        self.img_sub = self.create_subscription(
            Image, '/tf_camera/image_raw', 
            self.image_callback, sensor_qos
        )
        
        self.info_sub = self.create_subscription(
            CameraInfo, '/tf_camera/camera_info',
            self.info_callback, sensor_qos
        )
        
        # Status logging
        self.status_timer = self.create_timer(5.0, self.print_status)
        
        # Continuous TF publishing at 10 Hz (prevents frame expiration during occlusion)
        self.tf_publish_timer = self.create_timer(0.1, self.publish_cached_transforms)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Ball TF Publisher WITH CACHING Started")
        self.get_logger().info("=" * 60)

    def print_status(self):
        """Periodic status report"""
        self.get_logger().info(
            f"Status: {self.img_count} imgs | {self.pc_count} clouds | "
            f"{len(self.cached_transforms)} cached | {self.last_ball_count} detected"
        )

    def info_callback(self, msg):
        if self.camera_info is None:
            self.get_logger().info("✓ Camera info received")
        self.camera_info = msg

    def pointcloud_callback(self, msg):
        """Store latest point cloud for depth lookup"""
        self.pc_count += 1
        
        if self.pc_count == 1:
            self.get_logger().info(f"✓ Point cloud ready: {msg.width}x{msg.height}")
        
        self.latest_pointcloud = msg
        self.pointcloud_width = msg.width
        self.pointcloud_height = msg.height

    def publish_cached_transforms(self):
        """
        Continuously publish cached transforms at 10 Hz
        
        Why this is necessary:
        - TF frames expire if not published regularly
        - When arm moves between camera and object, vision is occluded
        - Cache ensures transforms remain available during occlusion
        - 10 Hz rate keeps TF tree fresh for motion planning
        """
        current_time = self.get_clock().now()
        
        # Remove expired transforms
        expired = [name for name, (_, _, _, ts) in self.cached_transforms.items()
                   if (current_time - ts).nanoseconds / 1e9 > self.transform_timeout]
        
        for name in expired:
            del self.cached_transforms[name]
        
        # Publish all active cached transforms
        for frame_name, (x, y, z, _) in self.cached_transforms.items():
            t_msg = TransformStamped()
            t_msg.header.stamp = current_time.to_msg()
            t_msg.header.frame_id = 'base_link'
            t_msg.child_frame_id = frame_name
            
            t_msg.transform.translation.x = float(x)
            t_msg.transform.translation.y = float(y)
            t_msg.transform.translation.z = float(z)
            
            # Identity rotation (objects don't rotate)
            t_msg.transform.rotation.w = 1.0
            t_msg.transform.rotation.x = 0.0
            t_msg.transform.rotation.y = 0.0
            t_msg.transform.rotation.z = 0.0
            
            self.tf_broadcaster.sendTransform(t_msg)

    def update_cached_transform(self, frame_name, x, y, z):
        """Update transform cache with new detection"""
        current_time = self.get_clock().now()
        
        if frame_name not in self.cached_transforms:
            self.get_logger().info(f"🎯 Cached: {frame_name} at ({x:.3f}, {y:.3f}, {z:.3f})")
        
        self.cached_transforms[frame_name] = (x, y, z, current_time)

    def get_3d_point_from_cloud_robust(self, u, v, search_radius=5):
        """
        Extract 3D point from point cloud with robustness
        
        Strategy:
        1. Sample multiple pixels around detection center
        2. Filter invalid points (NaN, inf, out-of-range depth)
        3. Use median to reject outliers
        
        Why median instead of mean:
        - Point clouds often have outliers at object edges
        - Median is robust to these outliers
        - Provides more stable depth estimate
        """
        if self.latest_pointcloud is None:
            return None
        
        valid_points = []
        
        # Sample pattern: center + surrounding pixels
        offsets = [(0, 0)]
        for r in range(1, search_radius + 1):
            offsets.extend([
                (r, 0), (-r, 0), (0, r), (0, -r),
                (r, r), (-r, -r), (r, -r), (-r, r)
            ])
        
        for du, dv in offsets:
            test_u = u + du
            test_v = v + dv
            
            if test_u < 0 or test_u >= self.pointcloud_width or \
               test_v < 0 or test_v >= self.pointcloud_height:
                continue
            
            try:
                # Extract XYZ from point cloud data
                point_step = self.latest_pointcloud.point_step
                row_step = self.latest_pointcloud.row_step
                offset = test_v * row_step + test_u * point_step
                
                data = self.latest_pointcloud.data
                x = struct.unpack_from('f', data, offset + 0)[0]
                y = struct.unpack_from('f', data, offset + 4)[0]
                z = struct.unpack_from('f', data, offset + 8)[0]
                
                # Validate point
                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                    continue
                if np.isinf(x) or np.isinf(y) or np.isinf(z):
                    continue
                if z > 5.0 or z < 0.01:  # Sanity check on depth
                    continue
                
                valid_points.append((x, y, z))
                
            except Exception:
                continue
        
        # Return median of valid points
        if valid_points:
            x_vals = [p[0] for p in valid_points]
            y_vals = [p[1] for p in valid_points]
            z_vals = [p[2] for p in valid_points]
            
            return (float(np.median(x_vals)), 
                    float(np.median(y_vals)), 
                    float(np.median(z_vals)))
        
        return None

    def transform_optical_to_base_link(self, point_optical, timestamp):
        """
        Transform point from camera optical frame to base_link frame
        
        Uses TF tree to handle camera mounting position and orientation
        """
        try:
            source_frame = self.latest_pointcloud.header.frame_id
            target_frame = 'base_link'
            
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Extract transform components
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z
            
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            
            # Convert quaternion to rotation matrix
            R = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)
            
            # Apply transform: p_base = R * p_optical + t
            p_optical = np.array([point_optical[0], point_optical[1], point_optical[2]])
            p_base = R @ p_optical + np.array([tx, ty, tz])
            
            return (float(p_base[0]), float(p_base[1]), float(p_base[2]))
            
        except Exception:
            return None

    def quaternion_to_rotation_matrix(self, x, y, z, w):
        """Convert quaternion to 3x3 rotation matrix"""
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        return R

    def image_callback(self, msg):
        """
        Detect balls and update cached transforms
        
        Detection pipeline:
        1. Find brown table (defines valid region)
        2. Detect white balls within table region
        3. Get 3D position from point cloud
        4. Transform to base_link frame
        5. Update cache (not direct publish - cache handles publishing)
        """
        self.img_count += 1

        # Skip warmup period
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed < self.warmup_duration:
            return

        # Wait for sensor data
        if self.latest_pointcloud is None or self.camera_info is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            kernel = np.ones((5, 5), np.uint8)
            timestamp = rclpy.time.Time.from_msg(msg.header.stamp)

            # Detect table region
            lower_brown = np.array([10, 100, 20])
            upper_brown = np.array([20, 255, 200])
            mask_plate = cv2.inRange(hsv, lower_brown, upper_brown)
            mask_plate = cv2.morphologyEx(mask_plate, cv2.MORPH_OPEN, kernel)
            mask_plate = cv2.morphologyEx(mask_plate, cv2.MORPH_CLOSE, kernel)

            plate_contours, _ = cv2.findContours(
                mask_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            largest_plate_contour = None
            if plate_contours:
                largest_plate_contour = max(plate_contours, key=cv2.contourArea)
                cv2.drawContours(cv_image, [largest_plate_contour], -1, (139, 69, 19), 2)

            # Detect white balls
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 50, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

            white_contours, _ = cv2.findContours(
                mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            ball_id = 0
            balls_2d = []

            if white_contours and largest_plate_contour is not None:
                for cnt in white_contours:
                    area = cv2.contourArea(cnt)
                    if area < 50:  # Minimum area threshold
                        continue

                    ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                    cx, cy = int(x), int(y)

                    # Ball must be inside table region
                    dist = cv2.pointPolygonTest(largest_plate_contour, (cx, cy), False)
                    if dist < 0:
                        continue

                    # Circularity check (reject non-circular objects)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue

                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity < 0.7:
                        continue

                    balls_2d.append((cx, cy, radius))

                # Sort by X coordinate for consistent naming
                balls_2d.sort(key=lambda b: b[0])

                # Process each detected ball
                for (cx, cy, radius) in balls_2d:
                    frame_name = f'ball_{ball_id}'
                    
                    # Get 3D position from point cloud
                    point_optical = self.get_3d_point_from_cloud_robust(cx, cy, search_radius=5)
                    
                    if point_optical is not None:
                        # Transform to base_link
                        point_base = self.transform_optical_to_base_link(point_optical, timestamp)
                        
                        if point_base is not None:
                            x, y, z = point_base
                            z = z - 0.001  # Small offset correction
                            
                            # Update cache (cache will handle publishing)
                            self.update_cached_transform(frame_name, x, y, z)
                            
                            # Visualization
                            cv2.circle(cv_image, (cx, cy), int(radius), (0, 255, 0), 2)
                            cv2.circle(cv_image, (cx, cy), 3, (0, 0, 255), -1)
                            cv2.putText(cv_image, frame_name, (cx + 10, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            ball_id += 1

            self.last_ball_count = ball_id

            # Display status
            status_text = f"Detected: {ball_id} | Cached: {len(self.cached_transforms)}"
            cv2.putText(cv_image, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Ball Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BallTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()