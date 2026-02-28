#!/usr/bin/env python3
"""
Visual Servoing Calibration Tool
Helps tune:
1. Color detection thresholds
2. Pixel-to-meter conversion factors
3. Camera orientation corrections
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class VisualServoCalibration(Node):
    def __init__(self):
        super().__init__('visual_servo_calibration')
        
        self.bridge = CvBridge()
        self.latest_image = None
        
        # Trackbar values
        self.h_low = 0
        self.h_high = 180
        self.s_low = 0
        self.s_high = 255
        self.v_low = 200
        self.v_high = 255
        
        # Subscribe to end-effector camera
        self.create_subscription(
            Image, '/ee_camera/image_raw', self.image_callback, 10
        )
        
        self.get_logger().info('🔧 Visual Servoing Calibration Tool')
        self.get_logger().info('Press "q" to quit, "s" to save values')
        
        # Create windows
        cv2.namedWindow('Original')
        cv2.namedWindow('Mask')
        cv2.namedWindow('Controls')
        
        # Create trackbars
        cv2.createTrackbar('H Low', 'Controls', self.h_low, 180, self.on_trackbar)
        cv2.createTrackbar('H High', 'Controls', self.h_high, 180, self.on_trackbar)
        cv2.createTrackbar('S Low', 'Controls', self.s_low, 255, self.on_trackbar)
        cv2.createTrackbar('S High', 'Controls', self.s_high, 255, self.on_trackbar)
        cv2.createTrackbar('V Low', 'Controls', self.v_low, 255, self.on_trackbar)
        cv2.createTrackbar('V High', 'Controls', self.v_high, 255, self.on_trackbar)
        
        # Timer for visualization
        self.create_timer(0.033, self.visualize)  # ~30Hz
    
    def on_trackbar(self, val):
        """Trackbar callback"""
        self.h_low = cv2.getTrackbarPos('H Low', 'Controls')
        self.h_high = cv2.getTrackbarPos('H High', 'Controls')
        self.s_low = cv2.getTrackbarPos('S Low', 'Controls')
        self.s_high = cv2.getTrackbarPos('S High', 'Controls')
        self.v_low = cv2.getTrackbarPos('V Low', 'Controls')
        self.v_high = cv2.getTrackbarPos('V High', 'Controls')
    
    def image_callback(self, msg):
        """Store latest image"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
    
    def visualize(self):
        """Visualize detection with current parameters"""
        if self.latest_image is None:
            return
        
        image = self.latest_image.copy()
        h, w = image.shape[:2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask
        lower = np.array([self.h_low, self.s_low, self.v_low])
        upper = np.array([self.h_high, self.s_high, self.v_high])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw on original image
        display = image.copy()
        
        # Draw image center crosshair
        center_x, center_y = w // 2, h // 2
        cv2.line(display, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 0), 2)
        cv2.line(display, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 0), 2)
        cv2.circle(display, (center_x, center_y), 20, (0, 255, 0), 1)
        
        # Draw detected objects
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    # Draw contour
                    cv2.drawContours(display, [contour], -1, (0, 255, 255), 2)
                    
                    # Get centroid
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Draw centroid
                        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                        
                        # Draw line from center to object
                        cv2.line(display, (center_x, center_y), (cx, cy), (255, 0, 0), 1)
                        
                        # Calculate pixel error
                        error_x = cx - center_x
                        error_y = cy - center_y
                        
                        # Display info
                        info_text = f'Area: {int(area)} | Error: ({error_x}, {error_y})'
                        cv2.putText(display, info_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Estimate correction (you'll need to tune these)
                        PIXEL_TO_METER_X = 0.00015
                        PIXEL_TO_METER_Y = 0.00015
                        corr_x = -error_x * PIXEL_TO_METER_X
                        corr_y = error_y * PIXEL_TO_METER_Y
                        
                        corr_text = f'Correction: X={corr_x:.4f}m, Y={corr_y:.4f}m'
                        cv2.putText(display, corr_text, (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show parameter values
        param_text = f'HSV: [{self.h_low},{self.h_high}] [{self.s_low},{self.s_high}] [{self.v_low},{self.v_high}]'
        cv2.putText(display, param_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show images
        cv2.imshow('Original', display)
        cv2.imshow('Mask', mask_clean)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quitting...')
            rclpy.shutdown()
        elif key == ord('s'):
            self.save_parameters()
    
    def save_parameters(self):
        """Save calibrated parameters"""
        self.get_logger().info('=' * 60)
        self.get_logger().info('CALIBRATED COLOR DETECTION PARAMETERS')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f"'lower': np.array([{self.h_low}, {self.s_low}, {self.v_low}]),")
        self.get_logger().info(f"'upper': np.array([{self.h_high}, {self.s_high}, {self.v_high}])")
        self.get_logger().info('=' * 60)
        self.get_logger().info('Copy these values to your visual servoing node!')
        self.get_logger().info('=' * 60)


def main(args=None):
    rclpy.init(args=args)
    node = VisualServoCalibration()
    
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