#!/usr/bin/env python3
"""
Visual Servoing Test Script
Step-by-step testing and validation
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
import time


class VisualServoTester(Node):
    def __init__(self):
        super().__init__('visual_servo_tester')
        
        self.bridge = CvBridge()
        self.action_pub = self.create_publisher(String, '/vla/action_command', 10)
        
        # Subscribe to camera for verification
        self.create_subscription(
            Image, '/ee_camera/image_raw', self.camera_callback, 10
        )
        
        self.camera_active = False
        
        self.get_logger().info('🧪 Visual Servoing Test Script')
    
    def camera_callback(self, msg):
        """Check if camera is working"""
        if not self.camera_active:
            self.camera_active = True
            self.get_logger().info('✅ End-effector camera is active')
    
    def test_simple_pick(self):
        """Test a simple pick-and-place with known positions"""
        self.get_logger().info('=' * 60)
        self.get_logger().info('TEST 1: Simple Pick-and-Place with Visual Servoing')
        self.get_logger().info('=' * 60)
        
        # Check camera
        time.sleep(2.0)
        if not self.camera_active:
            self.get_logger().error('❌ End-effector camera not active!')
            self.get_logger().info('Check if /ee_camera/image_raw topic is publishing')
            return
        
        # Send test command
        action = {
            'action': 'pick_and_place',
            'object_id': 'ball_0',
            'pick': [0.25, 0.15, 0.015],
            'place': [0.40, -0.20, 0.01]
        }
        
        self.get_logger().info(f'📤 Sending action: {action["object_id"]}')
        
        msg = String()
        msg.data = json.dumps(action)
        self.action_pub.publish(msg)
        
        self.get_logger().info('⏳ Action sent. Watch the robot...')
        self.get_logger().info('Expected behavior:')
        self.get_logger().info('  1. Move to hover position')
        self.get_logger().info('  2. TF-based correction (if object visible)')
        self.get_logger().info('  3. Descend to approach height')
        self.get_logger().info('  4. Visual servoing adjustments (watch for small movements)')
        self.get_logger().info('  5. Final grasp and place')
    
    def print_instructions(self):
        """Print testing instructions"""
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('VISUAL SERVOING TESTING GUIDE')
        self.get_logger().info('=' * 60)
        self.get_logger().info('')
        self.get_logger().info('STEP 1: Verify Camera Feed')
        self.get_logger().info('  Run: ros2 run rqt_image_view rqt_image_view')
        self.get_logger().info('  Select topic: /ee_camera/image_raw')
        self.get_logger().info('  → You should see gripper view')
        self.get_logger().info('')
        self.get_logger().info('STEP 2: Calibrate Color Detection')
        self.get_logger().info('  Run: ros2 run bot_vla visual_servo_calibration.py')
        self.get_logger().info('  → Adjust HSV sliders until ball is white in mask')
        self.get_logger().info('  → Press "s" to save parameters')
        self.get_logger().info('  → Update color ranges in action_node_hybrid.py')
        self.get_logger().info('')
        self.get_logger().info('STEP 3: Tune Pixel-to-Meter Conversion')
        self.get_logger().info('  Method: Place ball at known position, measure pixel error')
        self.get_logger().info('  → Move gripper over ball using manual commands')
        self.get_logger().info('  → Note pixel error in calibration tool')
        self.get_logger().info('  → Calculate: PIXEL_TO_METER = known_error_m / pixel_error')
        self.get_logger().info('  → Update values in action_node_hybrid.py')
        self.get_logger().info('')
        self.get_logger().info('STEP 4: Test Visual Servoing')
        self.get_logger().info('  → Run full pick-and-place test')
        self.get_logger().info('  → Watch console for "Visual servoing" messages')
        self.get_logger().info('  → Check if gripper makes small corrective movements')
        self.get_logger().info('')
        self.get_logger().info('STEP 5: Fine-tune Parameters')
        self.get_logger().info('  If undercorrecting: increase PIXEL_TO_METER values')
        self.get_logger().info('  If overcorrecting: decrease PIXEL_TO_METER values')
        self.get_logger().info('  If oscillating: increase POSITION_TOLERANCE')
        self.get_logger().info('  If too slow: decrease MAX_SERVO_ITERATIONS')
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('COMMON ISSUES & SOLUTIONS')
        self.get_logger().info('=' * 60)
        self.get_logger().info('')
        self.get_logger().info('Issue: Object not detected in camera')
        self.get_logger().info('  → Check camera is pointing at workspace')
        self.get_logger().info('  → Adjust HSV color ranges')
        self.get_logger().info('  → Ensure good lighting')
        self.get_logger().info('')
        self.get_logger().info('Issue: Corrections in wrong direction')
        self.get_logger().info('  → Flip signs in compute_visual_correction()')
        self.get_logger().info('  → Camera orientation may be different')
        self.get_logger().info('')
        self.get_logger().info('Issue: TF occlusion (arm blocks camera)')
        self.get_logger().info('  → This is expected! Visual servoing takes over')
        self.get_logger().info('  → End-effector camera provides close-range feedback')
        self.get_logger().info('')
        self.get_logger().info('Issue: Gripper still misses by 1-2cm')
        self.get_logger().info('  → Increase MAX_SERVO_ITERATIONS')
        self.get_logger().info('  → Decrease POSITION_TOLERANCE')
        self.get_logger().info('  → Check gripper tip offset still accurate')
        self.get_logger().info('')
        self.get_logger().info('=' * 60)


def main(args=None):
    rclpy.init(args=args)
    node = VisualServoTester()
    
    # Print instructions
    node.print_instructions()
    
    # Wait for user input
    node.get_logger().info('')
    input('Press ENTER to run test pick-and-place...')
    
    # Run test
    node.test_simple_pick()
    
    # Keep spinning
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()