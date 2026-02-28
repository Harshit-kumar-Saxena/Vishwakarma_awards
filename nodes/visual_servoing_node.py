#!/usr/bin/env python3
"""
Visual Servoing Node - Hybrid Closed-Loop Control
Combines TF camera (coarse) + End-effector camera (fine) positioning
Publishes correction commands to action_node
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import json


class VisualServoNode(Node):
    def __init__(self):
        super().__init__('visual_servo_node')
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Publishers
        self.correction_pub = self.create_publisher(Point, '/vla/position_correction', 10)
        self.status_pub = self.create_publisher(String, '/vla/servo_status', 10)
        
        # Subscribers
        self.ee_camera_sub = self.create_subscription(
            Image, '/ee_camera/image_raw', self.ee_camera_callback, 10
        )
        
        # State
        self.latest_ee_image = None
        self.servoing_active = False
        self.target_color_range = None  # Will be set dynamically
        
        # Servoing parameters
        self.IMAGE_CENTER_X = 320  # 640/2
        self.IMAGE_CENTER_Y = 240  # 480/2
        self.POSITION_TOLERANCE = 15  # pixels
        self.MAX_CORRECTION = 0.02  # meters (2cm max correction per iteration)
        
        # Pixel-to-meter conversion (approximate, depends on camera height)
        # At gripper level (~0.05m from object), rough estimate:
        self.PIXEL_TO_METER_X = 0.0001  # Tune this based on your camera FOV
        self.PIXEL_TO_METER_Y = 0.0001
        
        # Color detection parameters (HSV ranges)
        self.color_ranges = {
            'white': {
                'lower': np.array([0, 0, 200]),
                'upper': np.array([180, 30, 255])
            },
            'yellow': {
                'lower': np.array([20, 100, 100]),
                'upper': np.array([30, 255, 255])
            },
            'red': {
                'lower': np.array([0, 100, 100]),
                'upper': np.array([10, 255, 255])
            }
        }
        
        self.get_logger().info('👁️ Visual Servoing Node ready')
    
    def ee_camera_callback(self, msg):
        """Store latest end-effector camera image"""
        try:
            self.latest_ee_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
    
    def detect_object_in_image(self, image, color='white'):
        """
        Detect object (ball) in image using color segmentation
        Returns: (found, center_x, center_y, area)
        """
        if image is None:
            return False, 0, 0, 0
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get color range
        color_range = self.color_ranges.get(color, self.color_ranges['white'])
        
        # Create mask
        mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        
        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0, 0, 0
        
        # Get largest contour (assumed to be the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Filter by minimum area (avoid noise)
        if area < 100:  # Adjust based on expected ball size
            return False, 0, 0, 0
        
        # Get centroid
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return False, 0, 0, 0
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        return True, cx, cy, area
    
    def compute_correction(self, detected_x, detected_y):
        """
        Compute XY correction needed to center object
        Returns: (correction_x, correction_y) in meters
        """
        # Calculate pixel error
        error_x = detected_x - self.IMAGE_CENTER_X
        error_y = detected_y - self.IMAGE_CENTER_Y
        
        # Convert to meters (in gripper frame)
        # Note: Image X axis may be reversed relative to robot frame
        correction_x = -error_x * self.PIXEL_TO_METER_X  # Negative because image X is flipped
        correction_y = error_y * self.PIXEL_TO_METER_Y
        
        # Limit maximum correction
        correction_x = np.clip(correction_x, -self.MAX_CORRECTION, self.MAX_CORRECTION)
        correction_y = np.clip(correction_y, -self.MAX_CORRECTION, self.MAX_CORRECTION)
        
        return correction_x, correction_y
    
    def is_centered(self, detected_x, detected_y):
        """Check if object is centered within tolerance"""
        error_x = abs(detected_x - self.IMAGE_CENTER_X)
        error_y = abs(detected_y - self.IMAGE_CENTER_Y)
        
        return error_x < self.POSITION_TOLERANCE and error_y < self.POSITION_TOLERANCE
    
    def visual_servo_step(self, target_color='white', max_iterations=10):
        """
        Perform one visual servoing sequence
        Returns: (success, final_correction)
        """
        if self.latest_ee_image is None:
            self.get_logger().warn('⚠️ No end-effector camera image available')
            return False, (0, 0)
        
        self.get_logger().info(f'🎯 Starting visual servoing for {target_color} object')
        
        iteration = 0
        total_correction_x = 0
        total_correction_y = 0
        
        while iteration < max_iterations:
            # Detect object
            found, cx, cy, area = self.detect_object_in_image(self.latest_ee_image, target_color)
            
            if not found:
                self.get_logger().warn(f'❌ Object not found in iteration {iteration}')
                return False, (total_correction_x, total_correction_y)
            
            self.get_logger().info(f'📍 Detected at ({cx}, {cy}), area={area}')
            
            # Check if centered
            if self.is_centered(cx, cy):
                self.get_logger().info(f'✅ Object centered after {iteration} iterations')
                return True, (total_correction_x, total_correction_y)
            
            # Compute correction
            corr_x, corr_y = self.compute_correction(cx, cy)
            
            # Publish correction
            correction_msg = Point()
            correction_msg.x = corr_x
            correction_msg.y = corr_y
            correction_msg.z = 0.0
            self.correction_pub.publish(correction_msg)
            
            # Accumulate corrections
            total_correction_x += corr_x
            total_correction_y += corr_y
            
            self.get_logger().info(f'🔧 Correction: X={corr_x:.4f}m, Y={corr_y:.4f}m')
            
            # Wait for movement (handled by action_node)
            rclpy.spin_once(self, timeout_sec=0.5)
            
            iteration += 1
        
        self.get_logger().warn(f'⏰ Max iterations reached')
        return False, (total_correction_x, total_correction_y)
    
    def draw_debug_image(self, image, detected_x, detected_y, found):
        """Draw debug visualization"""
        if image is None:
            return None
        
        debug_img = image.copy()
        
        # Draw center crosshair
        cv2.line(debug_img, (self.IMAGE_CENTER_X - 20, self.IMAGE_CENTER_Y),
                 (self.IMAGE_CENTER_X + 20, self.IMAGE_CENTER_Y), (0, 255, 0), 2)
        cv2.line(debug_img, (self.IMAGE_CENTER_X, self.IMAGE_CENTER_Y - 20),
                 (self.IMAGE_CENTER_X, self.IMAGE_CENTER_Y + 20), (0, 255, 0), 2)
        
        # Draw tolerance circle
        cv2.circle(debug_img, (self.IMAGE_CENTER_X, self.IMAGE_CENTER_Y),
                   self.POSITION_TOLERANCE, (0, 255, 0), 1)
        
        # Draw detected object
        if found:
            cv2.circle(debug_img, (detected_x, detected_y), 10, (0, 0, 255), -1)
            cv2.line(debug_img, (self.IMAGE_CENTER_X, self.IMAGE_CENTER_Y),
                     (detected_x, detected_y), (255, 0, 0), 2)
            
            # Show error
            error_x = detected_x - self.IMAGE_CENTER_X
            error_y = detected_y - self.IMAGE_CENTER_Y
            cv2.putText(debug_img, f'Error: ({error_x}, {error_y})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_img


def main(args=None):
    rclpy.init(args=args)
    node = VisualServoNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()