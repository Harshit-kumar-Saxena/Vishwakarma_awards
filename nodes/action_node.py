#!/usr/bin/env python3
"""
VLA Action Node with Hybrid Visual Servoing
Implements closed-loop control using TF + end-effector camera
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from tf_transformations import quaternion_from_euler
from tf2_ros import TransformListener, Buffer
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import time
import threading


class HybridActionNode(Node):
    def __init__(self):
        super().__init__('hybrid_action_node')
        
        self.service_cb_group = ReentrantCallbackGroup()
        self.action_cb_group = ReentrantCallbackGroup()
        
        self.joint_names = [
            'top_plate_joint',
            'lower_arm_joint',
            'upper_arm_joint',
            'wrist_joint',
            'claw_base_joint'
        ]
        
        self.current_joint_state = None
        self.ik_service_ready = False
        self.latest_ee_image = None
        self.bridge = CvBridge()
        
        # TF listener for position verification
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10,
            callback_group=self.service_cb_group
        )
        
        self.ee_camera_sub = self.create_subscription(
            Image, '/ee_camera/image_raw', self.ee_camera_callback, 10,
            callback_group=self.service_cb_group
        )
        
        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik',
            callback_group=self.service_cb_group
        )
        
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory,
            '/bot_arm_controller/follow_joint_trajectory',
            callback_group=self.action_cb_group
        )
        
        self.hand_client = ActionClient(
            self, FollowJointTrajectory,
            '/hand_controller/follow_joint_trajectory',
            callback_group=self.action_cb_group
        )
        
        # Heights
        self.HOVER_Z = 0.14
        self.PICK_Z = 0.0
        self.GRAB_Z = 0.0
        self.PLATE_HOVER_Z = 0.15
        self.PLATE_PLACE_Z = 0.12
        
        # Initial offsets (will be refined by closed-loop)
        self.REACH_OFFSET_X = -0.025
        self.REACH_OFFSET_Y = -0.014
        
        # Visual servoing parameters
        self.IMAGE_CENTER_X = 320
        self.IMAGE_CENTER_Y = 240
        self.POSITION_TOLERANCE = 20  # pixels
        self.PIXEL_TO_METER_X = 0.00015
        self.PIXEL_TO_METER_Y = 0.00015
        self.MAX_SERVO_ITERATIONS = 8
        
        # IK parameters
        self.GRIPPER_DOWN_PITCH = 1.57
        self.ORIENTATION_TOLERANCE = 6.28
        self.ORIENTATION_WEIGHT = 0.01
        
        self.create_subscription(
            String, '/vla/action_command', self.action_callback, 10,
            callback_group=self.action_cb_group
        )
        
        self.status_pub = self.create_publisher(String, '/vla/action_status', 10)
        
        # Wait for services
        self.init_thread = threading.Thread(target=self.wait_for_services, daemon=True)
        self.init_thread.start()
        
        self.get_logger().info('🔄 Hybrid Action Node initializing...')
    
    def wait_for_services(self):
        """Wait for required services"""
        retry_count = 0
        while not self.ik_client.wait_for_service(timeout_sec=2.0) and retry_count < 15:
            retry_count += 1
        
        if self.ik_client.service_is_ready():
            self.ik_service_ready = True
            self.get_logger().info('✅ IK service ready')
        
        self.trajectory_client.wait_for_server(timeout_sec=10.0)
        self.hand_client.wait_for_server(timeout_sec=10.0)
        self.get_logger().info('✅ Hybrid Action Node ready')
    
    def joint_state_callback(self, msg):
        self.current_joint_state = msg
    
    def ee_camera_callback(self, msg):
        """Store latest end-effector camera image"""
        try:
            self.latest_ee_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def action_callback(self, msg):
        """Process action commands"""
        try:
            action = json.loads(msg.data)
            self.get_logger().info(f'🎯 Executing: {action["object_id"]}')
            
            if not self.ik_service_ready:
                self.get_logger().error('IK service not ready!')
                return
            
            if action['action'] == 'pick_and_place':
                success = self.execute_pick_and_place_hybrid(
                    action['object_id'],
                    action['pick'],
                    action['place']
                )
                
                if success:
                    self.get_logger().info(f'✅ Completed: {action["object_id"]}')
                    self.publish_status('success', action['object_id'])
                else:
                    self.get_logger().error(f'❌ Failed: {action["object_id"]}')
                    self.publish_status('failed', action['object_id'])
                
        except Exception as e:
            self.get_logger().error(f'Action error: {e}')

    def execute_pick_and_place_hybrid(self, object_id, pick_pos, place_pos):
        """
        Execute pick-and-place with hybrid visual servoing:
        1. TF-based coarse positioning
        2. End-effector camera fine adjustment
        3. Grasp with verification
        """
        try:
            # Determine object color for visual servoing
            target_color = self.get_object_color(object_id)
            
            # ========== STAGE 1: COARSE POSITIONING (TF Camera) ==========
            self.get_logger().info('📡 Stage 1: TF-based coarse positioning')
            
            # Apply initial offsets
            adjusted_x = pick_pos[0] + self.REACH_OFFSET_X
            adjusted_y = pick_pos[1] + self.REACH_OFFSET_Y
            
            # 1. Open gripper
            self.get_logger().info('✋ Opening gripper')
            self.set_gripper(-0.3)
            time.sleep(2.0)
            
            # 2. Move to hover position
            self.get_logger().info('⬆️ Moving to hover')
            if not self.reach_pose_accurate(adjusted_x, adjusted_y, self.HOVER_Z):
                return False
            time.sleep(0.5)
            
            # 3. TF-based position verification and correction
            corrected_x, corrected_y = self.tf_position_correction(
                object_id, adjusted_x, adjusted_y
            )
            
            if corrected_x is not None:
                self.get_logger().info(f'🔧 TF correction applied: ΔX={corrected_x-adjusted_x:.3f}, ΔY={corrected_y-adjusted_y:.3f}')
                adjusted_x, adjusted_y = corrected_x, corrected_y
                
                # Move to corrected hover position
                if not self.reach_pose_accurate(adjusted_x, adjusted_y, self.HOVER_Z):
                    return False
                time.sleep(0.5)
            
            # ========== STAGE 2: FINE POSITIONING (End-Effector Camera) ==========
            self.get_logger().info('👁️ Stage 2: Visual servoing fine adjustment')
            
            # Descend to approach height (where camera can see object)
            approach_height = self.PICK_Z + 0.03  # 3cm above pick
            if not self.reach_pose_accurate(adjusted_x, adjusted_y, approach_height):
                return False
            time.sleep(1.0)  # Let camera stabilize
            
            # Visual servoing loop
            servo_success, final_x, final_y = self.visual_servo_approach(
                adjusted_x, adjusted_y, approach_height, target_color
            )
            
            if servo_success:
                self.get_logger().info('✅ Visual servoing successful')
                adjusted_x, adjusted_y = final_x, final_y
            else:
                self.get_logger().warn('⚠️ Visual servoing incomplete, using current position')
            
            # ========== STAGE 3: PRECISE GRASP ==========
            self.get_logger().info('🎯 Stage 3: Final grasp')
            
            # Descend to PICK height
            if not self.reach_pose_accurate(adjusted_x, adjusted_y, self.PICK_Z):
                return False
            time.sleep(0.5)
            
            # Final descent to GRAB
            if not self.reach_pose_accurate(adjusted_x, adjusted_y, self.GRAB_Z):
                return False
            time.sleep(0.5)
            
            # Close gripper
            self.get_logger().info('🤏 Closing gripper')
            self.set_gripper(-0.05)
            time.sleep(2.0)
            
            # Verify grasp (optional - can check if object still visible)
            
            # ========== STAGE 4: PLACE ==========
            self.get_logger().info('📦 Stage 4: Place object')
            
            # Lift
            if not self.reach_pose_accurate(adjusted_x, adjusted_y, self.HOVER_Z):
                return False
            time.sleep(0.5)
            
            # Move to place position
            if not self.reach_pose_accurate(place_pos[0], place_pos[1], self.PLATE_HOVER_Z):
                if not self.reach_pose_accurate(place_pos[0], place_pos[1], self.PLATE_HOVER_Z - 0.03):
                    return False
            time.sleep(0.5)
            
            # Descend to place
            self.reach_pose_accurate(place_pos[0], place_pos[1], self.PLATE_PLACE_Z)
            time.sleep(0.5)
            
            # Release
            self.get_logger().info('✋ Releasing')
            self.set_gripper(-0.3)
            time.sleep(1.5)
            
            # Retract
            self.reach_pose_accurate(place_pos[0], place_pos[1], self.HOVER_Z)
            time.sleep(0.5)
            
            self.get_logger().info('✅ Mission complete')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')
            return False
    
    def tf_position_correction(self, object_id, current_x, current_y, max_error=0.05):
        """
        Use TF to verify actual object position and correct if needed
        Returns: (corrected_x, corrected_y) or (None, None) if correction not possible
        """
        try:
            # Try to get current object transform
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                object_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            actual_x = transform.transform.translation.x
            actual_y = transform.transform.translation.y
            
            error_x = actual_x - current_x
            error_y = actual_y - current_y
            error_mag = (error_x**2 + error_y**2)**0.5
            
            self.get_logger().info(f'📊 TF verification: error={error_mag:.3f}m')
            
            if error_mag > 0.005:  # More than 5mm error
                if error_mag < max_error:
                    self.get_logger().info(f'🔧 Applying TF correction: X+={error_x:.3f}, Y+={error_y:.3f}')
                    return actual_x, actual_y
                else:
                    self.get_logger().warn(f'⚠️ TF error too large: {error_mag:.3f}m')
            
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed (expected if arm blocks camera): {e}')
        
        return None, None
    
    def visual_servo_approach(self, start_x, start_y, height, target_color):
        """
        Perform visual servoing to center object in end-effector camera
        Returns: (success, final_x, final_y)
        """
        if self.latest_ee_image is None:
            self.get_logger().warn('⚠️ No camera image available')
            return False, start_x, start_y
        
        current_x = start_x
        current_y = start_y
        
        for iteration in range(self.MAX_SERVO_ITERATIONS):
            # Detect object in image
            found, cx, cy, area = self.detect_object_in_image(self.latest_ee_image, target_color)
            
            if not found:
                self.get_logger().warn(f'❌ Object not visible (iteration {iteration})')
                if iteration == 0:
                    return False, current_x, current_y
                else:
                    return True, current_x, current_y  # Use last good position
            
            self.get_logger().info(f'📍 Detected at ({cx}, {cy}), area={area}')
            
            # Check if centered
            if self.is_centered(cx, cy):
                self.get_logger().info(f'✅ Centered in {iteration} iterations')
                return True, current_x, current_y
            
            # Compute correction
            corr_x, corr_y = self.compute_visual_correction(cx, cy)
            
            # Apply correction
            current_x += corr_x
            current_y += corr_y
            
            self.get_logger().info(f'🔧 Correction: ΔX={corr_x:.4f}m, ΔY={corr_y:.4f}m')
            
            # Move to corrected position
            if not self.reach_pose_accurate(current_x, current_y, height):
                return False, current_x, current_y
            
            time.sleep(0.3)  # Let camera stabilize
        
        self.get_logger().warn('⏰ Max servo iterations reached')
        return False, current_x, current_y
    
    def detect_object_in_image(self, image, color='white'):
        """Detect object in image using color segmentation"""
        if image is None:
            return False, 0, 0, 0
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color ranges (tune for your objects)
        color_ranges = {
            'white': (np.array([0, 0, 87]), np.array([180, 50, 255])),
            'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
        }
        
        lower, upper = color_ranges.get(color, color_ranges['white'])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0, 0, 0
        
        # Largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 100:
            return False, 0, 0, 0
        
        # Centroid
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return False, 0, 0, 0
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        return True, cx, cy, area
    
    def compute_visual_correction(self, detected_x, detected_y):
        """Convert pixel error to meter correction"""
        error_x = detected_x - self.IMAGE_CENTER_X
        error_y = detected_y - self.IMAGE_CENTER_Y
        
        # IMPORTANT: Adjust signs based on your camera orientation!
        # You may need to flip these based on testing
        correction_x = -error_x * self.PIXEL_TO_METER_X
        correction_y = error_y * self.PIXEL_TO_METER_Y
        
        # Limit corrections
        correction_x = np.clip(correction_x, -0.02, 0.02)
        correction_y = np.clip(correction_y, -0.02, 0.02)
        
        return correction_x, correction_y
    
    def is_centered(self, detected_x, detected_y):
        """Check if object is centered"""
        error_x = abs(detected_x - self.IMAGE_CENTER_X)
        error_y = abs(detected_y - self.IMAGE_CENTER_Y)
        return error_x < self.POSITION_TOLERANCE and error_y < self.POSITION_TOLERANCE
    
    def get_object_color(self, object_id):
        """Determine object color from ID"""
        if 'ball' in object_id.lower():
            return 'white'
        elif 'yellow' in object_id.lower():
            return 'yellow'
        return 'white'
    
    # ========== IK and Motion Functions (same as before) ==========
    
    def create_pose(self, x, y, z, roll=0.0, pitch=None, yaw=0.0):
        if pitch is None:
            pitch = self.GRIPPER_DOWN_PITCH
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        p.position.z = float(z)
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        p.orientation.x = qx
        p.orientation.y = qy
        p.orientation.z = qz
        p.orientation.w = qw
        return p
    
    def compute_ik(self, target_pose):
        if not self.ik_service_ready:
            return None
        
        req = GetPositionIK.Request()
        ps = PoseStamped()
        ps.header.frame_id = 'base_link'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = target_pose
        
        req.ik_request.group_name = 'bot_arm'
        req.ik_request.pose_stamped = ps
        req.ik_request.ik_link_name = 'claw_base'
        req.ik_request.timeout = Duration(sec=5, nanosec=0)
        
        constraints = Constraints()
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = ps.header
        orientation_constraint.link_name = req.ik_request.ik_link_name
        orientation_constraint.orientation = target_pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = self.ORIENTATION_TOLERANCE
        orientation_constraint.absolute_y_axis_tolerance = self.ORIENTATION_TOLERANCE
        orientation_constraint.absolute_z_axis_tolerance = self.ORIENTATION_TOLERANCE
        orientation_constraint.weight = self.ORIENTATION_WEIGHT
        constraints.orientation_constraints.append(orientation_constraint)
        req.ik_request.constraints = constraints
        
        if self.current_joint_state is not None:
            rs = RobotState()
            rs.joint_state = self.current_joint_state
            req.ik_request.robot_state = rs
        else:
            req.ik_request.robot_state = RobotState()
        
        try:
            future = self.ik_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
            
            if not future.done():
                return None
            
            response = future.result()
            if response is None or response.error_code.val != 1:
                return None
            
            solution = response.solution.joint_state
            positions = []
            for name in self.joint_names:
                if name in solution.name:
                    idx = solution.name.index(name)
                    positions.append(solution.position[idx])
                else:
                    return None
            return positions
            
        except Exception:
            return None
    
    def execute_joint_trajectory(self, joint_positions, duration=5.0):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        traj.points.append(point)
        goal.trajectory = traj
        
        send_goal_future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)
        
        if not send_goal_future.done():
            return False
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            return False
        
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=max(10.0, duration * 4.0))
        
        return result_future.done() and result_future.result() is not None
    
    def move_to_pose(self, target_pose, duration=6.0):
        joints = self.compute_ik(target_pose)
        if joints:
            return self.execute_joint_trajectory(joints, duration)
        return False

    def reach_pose_accurate(self, x, y, z):
        pose = self.create_pose(x, y, z, pitch=self.GRIPPER_DOWN_PITCH)
        if self.move_to_pose(pose, duration=6.0):
            return True
        
        for yaw_offset in [0.2, -0.2, 0.4, -0.4]:
            pose = self.create_pose(x, y, z, pitch=self.GRIPPER_DOWN_PITCH, yaw=yaw_offset)
            if self.move_to_pose(pose, duration=6.0):
                return True
        
        for pitch in [1.40, 1.70, 1.20, 1.00]:
            pose = self.create_pose(x, y, z, pitch=pitch)
            if self.move_to_pose(pose, duration=6.0):
                return True
        
        self.get_logger().error(f'❌ IK failed at ({x:.3f}, {y:.3f}, {z:.3f})')
        return False

    def set_gripper(self, val):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['right_claw_joint', 'left_claw_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [float(val), float(-val)]
        pt.time_from_start = Duration(sec=1, nanosec=0)
        goal.trajectory.points = [pt]

        try:
            send_goal_future = self.hand_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=3.0)
            if send_goal_future.done():
                goal_handle = send_goal_future.result()
                if goal_handle.accepted:
                    result_future = goal_handle.get_result_async()
                    rclpy.spin_until_future_complete(self, result_future, timeout_sec=3.0)
        except Exception as e:
            self.get_logger().warn(f'Gripper error: {e}')

    def publish_status(self, status, object_id):
        msg = String()
        msg.data = json.dumps({'status': status, 'object_id': object_id})
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HybridActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()