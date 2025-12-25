#!/usr/bin/env python3
"""
VLA Action Node - Improved with Better Orientation Handling
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from tf_transformations import quaternion_from_euler
import json
import time
import threading
import math


class VLAActionNode(Node):
    def __init__(self):
        super().__init__('vla_action')
        
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
        
        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10,
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
        
        # ============================================================
        # CALIBRATION PARAMETERS - ADJUST THESE
        # ============================================================
        
        # Heights (adjust based on your robot's reachability)
        self.HOVER_Z = 0.25
        self.PICK_Z = 0.020
        self.GRAB_Z = 0.018
        self.PLATE_HOVER_Z = 0.27
        self.PLATE_PLACE_Z = 0.22
        
        # Gripper tip offsets - CALIBRATE WITH calibration_helper.py
        self.REACH_OFFSET_X = -0.019  # START HERE: Run calibration to find correct values
        self.REACH_OFFSET_Y = -0.012
        
        # IK orientation parameters
        self.GRIPPER_DOWN_PITCH = 1.57  # 90 degrees (gripper pointing straight down)
        self.ORIENTATION_TOLERANCE = 6.28  # Full rotation freedom
        self.ORIENTATION_WEIGHT = 0.01  # Very low weight = position priority
        
        # ============================================================
        
        self.create_subscription(
            String, '/vla/action_command', self.action_callback, 10,
            callback_group=self.action_cb_group
        )
        
        self.status_pub = self.create_publisher(String, '/vla/action_status', 10)
        
        # Wait for services in background thread
        self.init_thread = threading.Thread(target=self.wait_for_services, daemon=True)
        self.init_thread.start()
        
        self.get_logger().info('🔄 VLA Action Node initializing...')
        self.get_logger().info(f'📍 Offsets: X={self.REACH_OFFSET_X:.3f}, Y={self.REACH_OFFSET_Y:.3f}')
    
    def wait_for_services(self):
        """Wait for required services to be available"""
        retry_count = 0
        while not self.ik_client.wait_for_service(timeout_sec=2.0) and retry_count < 15:
            retry_count += 1
        
        if self.ik_client.service_is_ready():
            self.ik_service_ready = True
            self.get_logger().info('✅ IK service ready')
        
        self.trajectory_client.wait_for_server(timeout_sec=10.0)
        self.hand_client.wait_for_server(timeout_sec=10.0)
        self.get_logger().info('✅ Action servers ready')
        self.get_logger().info('✅ VLA Action Node ready')
    
    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def action_callback(self, msg):
        """Process incoming action commands from brain node"""
        try:
            action = json.loads(msg.data)
            self.get_logger().info(f'🎯 Executing: {action["object_id"]}')
            
            if not self.ik_service_ready:
                self.get_logger().error('IK service not ready!')
                return
            
            if action['action'] == 'pick_and_place':
                success = self.execute_pick_and_place(
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

    def execute_pick_and_place(self, object_id, pick_pos, place_pos):
        """Execute complete pick-and-place sequence with improved accuracy"""
        try:
            # Apply gripper tip offset compensation
            adjusted_x = pick_pos[0] + self.REACH_OFFSET_X
            adjusted_y = pick_pos[1] + self.REACH_OFFSET_Y
            
            self.get_logger().info(f'🎯 Target: ({pick_pos[0]:.3f}, {pick_pos[1]:.3f}, {pick_pos[2]:.3f})')
            self.get_logger().info(f'🔧 Adjusted: ({adjusted_x:.3f}, {adjusted_y:.3f}, {pick_pos[2]:.3f})')
            
            # 1. Open gripper
            self.get_logger().info('✋ Opening gripper')
            self.set_gripper(-0.3)
            time.sleep(2.0)
            
            # 2. Workspace validation
            if not self.is_within_workspace(adjusted_x, adjusted_y, self.PICK_Z):
                return False
            
            # 3. Approach at PICK height
            self.get_logger().info('⬇️ Approaching')
            if not self.reach_pose_accurate(adjusted_x, adjusted_y, self.PICK_Z):
                self.get_logger().error('Pick approach failed')
                return False
            time.sleep(0.5)
            
            # 4. Descend to GRAB
            self.get_logger().info('🎯 Descending to grab')
            self.reach_pose_accurate(adjusted_x, adjusted_y, self.GRAB_Z)
            time.sleep(0.5)
            
            # 5. Close gripper
            self.get_logger().info('🤏 Closing gripper')
            self.set_gripper(-0.05)
            time.sleep(2.0)
            
            # 6. Lift to HOVER
            self.get_logger().info('⬆️ Lifting')
            if not self.reach_pose_accurate(adjusted_x, adjusted_y, self.HOVER_Z):
                self.get_logger().error('Failed to lift')
                return False
            time.sleep(0.5)
            
            # 7. Move to place HOVER
            self.get_logger().info('🚚 Moving to place position')
            if not self.reach_pose_accurate(place_pos[0], place_pos[1], self.PLATE_HOVER_Z):
                if not self.reach_pose_accurate(place_pos[0], place_pos[1], self.PLATE_HOVER_Z - 0.03):
                    return False
            time.sleep(0.5)
            
            # 8. Descend to PLACE
            self.get_logger().info('⬇️ Descending to place')
            self.reach_pose_accurate(place_pos[0], place_pos[1], self.PLATE_PLACE_Z)
            time.sleep(0.5)
            
            # 9. Release gripper
            self.get_logger().info('✋ Releasing')
            self.set_gripper(-0.3)
            time.sleep(1.5)
            
            # 10. Retract
            self.get_logger().info('⬆️ Retracting')
            self.reach_pose_accurate(place_pos[0], place_pos[1], self.HOVER_Z)
            time.sleep(0.5)
            
            self.get_logger().info('✅ Mission complete')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')
            return False
    
    def create_pose(self, x, y, z, roll=0.0, pitch=None, yaw=0.0):
        """Create a Pose with gripper pointing down"""
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
        """Compute IK with relaxed orientation constraints"""
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
        
        # Very relaxed orientation constraints
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
        
        # Include current joint state
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
        """Execute joint trajectory"""
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
        """Compute IK and execute trajectory"""
        joints = self.compute_ik(target_pose)
        if joints:
            return self.execute_joint_trajectory(joints, duration)
        return False

    def reach_pose_accurate(self, x, y, z):
        """
        Reach pose with systematic orientation search
        Now tries downward-pointing gripper first (most common for pick/place)
        """
        # Try standard downward orientation first
        pose = self.create_pose(x, y, z, pitch=self.GRIPPER_DOWN_PITCH)
        if self.move_to_pose(pose, duration=6.0):
            return True
        
        # Try slight variations in yaw
        for yaw_offset in [0.2, -0.2, 0.4, -0.4]:
            pose = self.create_pose(x, y, z, pitch=self.GRIPPER_DOWN_PITCH, yaw=yaw_offset)
            if self.move_to_pose(pose, duration=6.0):
                return True
        
        # Try pitch variations
        for pitch in [1.40, 1.70, 1.20, 1.00]:
            pose = self.create_pose(x, y, z, pitch=pitch)
            if self.move_to_pose(pose, duration=6.0):
                return True
        
        self.get_logger().error(f'❌ IK failed at ({x:.3f}, {y:.3f}, {z:.3f})')
        return False

    def is_within_workspace(self, x, y, z):
        """Check if position is reachable"""
        max_reach = 0.40
        distance = (x**2 + y**2)**0.5
        
        if distance > max_reach:
            self.get_logger().error(f'Target too far: {distance:.3f}m > {max_reach}m')
            return False
        
        if z < 0.01 or z > 0.30:
            self.get_logger().error(f'Z={z:.3f}m outside valid range')
            return False
        
        return True

    def set_gripper(self, val):
        """Control gripper (-0.3=open, 0.0=closed)"""
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
        """Publish action completion status"""
        msg = String()
        msg.data = json.dumps({'status': status, 'object_id': object_id})
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VLAActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()