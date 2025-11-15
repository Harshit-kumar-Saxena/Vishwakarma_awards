#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info('Vision Node (Real Camera) Started')
        
        self.bridge = CvBridge()
        
        # Subscribe to the robot's camera
        # Ensure your Gazebo bridge or camera plugin publishes this topic
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw', 
            self.image_callback,
            10)

        self.state_pub = self.create_publisher(String, '/world_state', 10)

        # HSV Range for the ball (Tune this for your Gazebo ball color!)
        # Example: White/Grey ball
        self.lower_color = np.array([0, 0, 200])
        self.upper_color = np.array([180, 30, 255])

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2

        # Detect Ball
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_info = "ball:missing"
        
        if contours:
            # Find largest contour
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            if radius > 5: # Filter noise
                # We send PIXEL coordinates relative to the image center
                # offset_x < 0 means ball is to the left
                # offset_y < 0 means ball is above center
                offset_x = int(x - center_x)
                offset_y = int(y - center_y)
                
                ball_info = f"ball_loc:[{offset_x},{offset_y}],ball_radius:{int(radius)}"
                
                # Draw for debug
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.line(frame, (center_x, center_y), (int(x), int(y)), (255, 0, 0), 2)

        # Publish what we see
        msg = String()
        msg.data = f"{ball_info}, image_w:{w}, image_h:{h}"
        self.state_pub.publish(msg)

        cv2.imshow("Robot Vision", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()