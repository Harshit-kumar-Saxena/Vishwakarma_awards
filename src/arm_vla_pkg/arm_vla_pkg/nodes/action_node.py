#!/usr/bin/env python3
"""
VLA Action Node - FIXED AsyncIO handling
Executes pick-and-place from brain_node JSON commands
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


class VLAActionNode(Node):
    def __init__(self):
        super().__init__('vla_action_node')
        
        # Separate callback groups for concurrent execution
        self.service_cb_group = ReentrantCallbackGroup()
        self.action_cb_group = ReentrantCallbackGroup()
        
        # Joint configuration
        self.joint_names = [
            'top_plate_joint',
            'lower_arm_joint',
            'upper_arm_joint',
            'wrist_joint',
            'claw_base_joint'
        ]
        
        # State tracking
        self.current_joint_state = None
        self.joint_state_received = False
        self.ik_service_ready = False
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            10,
            callback_group=self.service_cb_group
        )
        
        # IK service client
        self.ik_client = self.create_client(
            GetPositionIK, 
            '/compute_ik',
            callback_group=self.service_cb_group
        )
        
        # Trajectory action clients
        self.trajectory_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/akabot_arm_controller/follow_joint_trajectory',
            callback_group=self.action_cb_group
        )
        
        self.hand_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/hand_controller/follow_joint_trajectory',
            callback_group=self.action_cb_group
        )
        
        # Heights (proven values)
        self.HOVER_Z = 0.25
        self.PICK_Z = 0.20
        self.GRAB_Z = 0.10
        self.PLATE_HOVER_Z = 0.25
        self.PLATE_PLACE_Z = 0.15
        
        # Subscriber for action commands
        self.create_subscription(
            String, 
            '/vla/action_command', 
            self.action_callback, 
            10,
            callback_group=self.action_cb_group
        )
        
        # Status publisher
        self.status_pub = self.create_publisher(String, '/vla/action_status', 10)
        
        # Wait for services in background thread
        self.init_thread = threading.Thread(target=self.wait_for_services, daemon=True)
        self.init_thread.start()
        
        self.get_logger().info('🔄 VLA Action Node initializing...')
    
    def wait_for_services(self):
        """Wait for all required services in background"""
        self.get_logger().info('Waiting for IK service...')
        
        # Wait for IK service
        retry_count = 0
        while not self.ik_client.wait_for_service(timeout_sec=2.0) and retry_count < 15:
            retry_count += 1
            self.get_logger().warn(f'IK service not ready ({retry_count}/15)...')
        
        if self.ik_client.service_is_ready():
            self.ik_service_ready = True
            self.get_logger().info('✅ IK service ready')
        else:
            self.get_logger().error('❌ IK service timeout!')
        
        # Wait for action servers
        self.get_logger().info('Waiting for action servers...')
        self.trajectory_client.wait_for_server(timeout_sec=10.0)
        self.hand_client.wait_for_server(timeout_sec=10.0)
        
        self.get_logger().info('✅ VLA Action Node ready')
    
    def joint_state_callback(self, msg):
        """Store current joint state"""
        self.current_joint_state = msg
        self.joint_state_received = True

    def action_callback(self, msg):
        """Execute pick-and-place from JSON command"""
        try:
            action = json.loads(msg.data)
            self.get_logger().info(f'🎯 Executing: {action["action"]} for {action["object_id"]}')
            
            # Wait for IK service if not ready
            if not self.ik_service_ready:
                self.get_logger().warn('Waiting for IK service to be ready...')
                time.sleep(5.0)
                if not self.ik_service_ready:
                    self.get_logger().error('IK service still not ready!')
                    return
            
            if action['action'] == 'pick_and_place':
                success = self.execute_pick_and_place(
                    action['object_id'],
                    action['pick'],
                    action['place']
                )
                
                if success:
                    self.get_logger().info(f'✅ Successfully completed {action["object_id"]}')
                    self.publish_status('success', action['object_id'])
                else:
                    self.get_logger().error(f'❌ Failed to execute {action["object_id"]}')
                    self.publish_status('failed', action['object_id'])
            else:
                self.get_logger().warn(f'Unknown action: {action["action"]}')
                
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON received')
        except KeyError as e:
            self.get_logger().error(f'Missing key in action JSON: {e}')
        except Exception as e:
            self.get_logger().error(f'Action execution error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def execute_pick_and_place(self, object_id, pick_pos, place_pos):
        """Full pick-and-place sequence"""
        try:
            # 1. Move home
            self.get_logger().info('📍 Moving to home...')
            if not self.move_to_named_target('home'):
                self.get_logger().warn('Home position not reached, continuing...')
            time.sleep(0.5)
            
            # 2. Open gripper
            self.get_logger().info('✋ Opening gripper...')
            self.set_gripper(-0.5)
            time.sleep(1.0)
            
            # 3. Check workspace
            if not self.is_within_workspace(pick_pos[0], pick_pos[1], self.HOVER_Z):
                self.get_logger().error('Pick position outside reachable workspace!')
                return False
            
            # 4. Move to pick hover
            self.get_logger().info(f'⬆️ Moving to pick hover: {pick_pos[:2]} + {self.HOVER_Z}')
            if not self.try_reach_pose_with_fallback(pick_pos[0], pick_pos[1], self.HOVER_Z):
                self.get_logger().error('Failed to reach hover position!')
                return False
            
            # 5. Intermediate approach
            self.get_logger().info(f'⬇️ Approaching pick position...')
            if not self.try_reach_pose_with_fallback(pick_pos[0], pick_pos[1], self.PICK_Z):
                self.get_logger().warn('Intermediate approach failed, trying higher')
                if not self.try_reach_pose_with_fallback(pick_pos[0], pick_pos[1], self.PICK_Z + 0.05):
                    self.get_logger().error('Failed approach!')
                    return False
            
            # 6. Final descend & grab
            self.get_logger().info(f'🎯 Descending to grab: {pick_pos}')
            if self.try_reach_pose_with_fallback(pick_pos[0], pick_pos[1], self.GRAB_Z):
                self.get_logger().info('🤏 Closing gripper...')
                self.set_gripper(0.0)
                time.sleep(1.5)
            else:
                self.get_logger().warn('Final descend unreachable, closing at approach height')
                self.set_gripper(0.0)
                time.sleep(1.5)
            
            # 7. Lift object
            self.get_logger().info(f'⬆️ Lifting object...')
            if not self.try_reach_pose_with_fallback(pick_pos[0], pick_pos[1], self.HOVER_Z):
                self.get_logger().error('Failed to lift!')
                return False
            
            # 8. Move to place hover
            self.get_logger().info(f'🚁 Moving to place hover: {place_pos[:2]} + {self.PLATE_HOVER_Z}')
            if not self.try_reach_pose_with_fallback(place_pos[0], place_pos[1], self.PLATE_HOVER_Z):
                self.get_logger().error('Failed to reach place hover!')
                return False
            
            # 9. Descend to place
            self.get_logger().info(f'⬇️ Descending to place: {place_pos}')
            if not self.try_reach_pose_with_fallback(place_pos[0], place_pos[1], self.PLATE_PLACE_Z):
                self.get_logger().warn('Failed exact place, releasing at hover')
            
            # 10. Release
            self.get_logger().info('✋ Releasing object...')
            self.set_gripper(-0.5)
            time.sleep(1.5)
            
            # 11. Retract
            self.get_logger().info(f'⬆️ Retracting...')
            self.try_reach_pose_with_fallback(place_pos[0], place_pos[1], self.HOVER_Z)
            
            # 12. Return home
            self.get_logger().info('🏠 Returning home...')
            self.move_to_named_target('home')
            
            self.get_logger().info(f'✅ Mission Complete for {object_id}!')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Pick-and-place failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def create_pose(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """Create Pose message"""
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
        """Compute IK for target pose - ASYNC with blocking"""
        if not self.ik_service_ready:
            self.get_logger().error('IK service not ready!')
            return None
        
        req = GetPositionIK.Request()
        ps = PoseStamped()
        ps.header.frame_id = 'base_link'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = target_pose
        
        req.ik_request.group_name = 'akabot_arm'
        req.ik_request.pose_stamped = ps
        req.ik_request.ik_link_name = 'claw_base'
        req.ik_request.timeout = Duration(sec=2, nanosec=0)
        
        # Relaxed orientation for 5-DOF
        constraints = Constraints()
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = ps.header
        orientation_constraint.link_name = req.ik_request.ik_link_name
        orientation_constraint.orientation = target_pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 6.28
        orientation_constraint.absolute_y_axis_tolerance = 6.28
        orientation_constraint.absolute_z_axis_tolerance = 6.28
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints.append(orientation_constraint)
        req.ik_request.constraints = constraints
        
        # Use current state as seed
        if self.current_joint_state is not None:
            rs = RobotState()
            rs.joint_state = self.current_joint_state
            req.ik_request.robot_state = rs
        else:
            req.ik_request.robot_state = RobotState()
        
        # ASYNC call with blocking
        try:
            future = self.ik_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
            
            if not future.done():
                self.get_logger().error('IK service timeout')
                return None
            
            response = future.result()
            
            if response is None:
                self.get_logger().error('IK service returned None')
                return None
            
            if response.error_code.val != 1:
                self.get_logger().warn(f'IK failed with error code: {response.error_code.val}')
                return None
            
            # Extract joint positions
            solution = response.solution.joint_state
            positions = []
            for name in self.joint_names:
                if name in solution.name:
                    idx = solution.name.index(name)
                    positions.append(solution.position[idx])
                else:
                    self.get_logger().error(f'Missing joint {name} in IK solution')
                    return None
            
            return positions
            
        except Exception as e:
            self.get_logger().error(f'IK call exception: {e}')
            return None
    
    def execute_joint_trajectory(self, joint_positions, duration=5.0):
        """Execute joint trajectory - ASYNC with blocking"""
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        
        traj.points.append(point)
        goal.trajectory = traj
        
        # Send goal asynchronously
        send_goal_future = self.trajectory_client.send_goal_async(goal)
        
        # Wait for goal to be accepted
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)
        
        if not send_goal_future.done():
            self.get_logger().error("Timeout waiting for goal acceptance")
            return False
        
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return False
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=max(10.0, duration * 4.0))
        
        if not result_future.done():
            self.get_logger().warn("Trajectory execution timeout")
            return False
        
        result = result_future.result()
        return result is not None
    
    def move_to_pose(self, target_pose, duration=6.0):
        """Move to Pose using IK + trajectory execution"""
        joints = self.compute_ik(target_pose)
        if joints:
            return self.execute_joint_trajectory(joints, duration)
        return False
    
    def move_to_named_target(self, name):
        """Move to named configuration"""
        targets = {
            'home':  [3.12, 1.5686, 0.0, 0.0, 0.0],
            'ready': [3.12, 2.182, -0.925, -1.1177, 0.0]
        }
        if name not in targets:
            return False
        return self.execute_joint_trajectory(targets[name], duration=7.0)

    def try_reach_pose_with_fallback(self, x, y, z, yaw=0.0):
        """Try multiple pitch angles for IK solutions"""
        pitch_list = [0.0, 0.15, 0.30, 0.45, 0.60, -0.15]
        
        for pitch in pitch_list:
            pose = self.create_pose(x=x, y=y, z=z, roll=0.0, pitch=pitch, yaw=yaw)
            if self.move_to_pose(pose, duration=6.0):
                return True
        
        self.get_logger().error(f'IK failed for all fallback angles at z={z:.3f}')
        return False

    def is_within_workspace(self, x, y, z):
        """Check if target is reachable"""
        max_reach = 0.45
        distance = (x**2 + y**2)**0.5
        
        if distance > max_reach:
            self.get_logger().error(f'Target ({x:.3f}, {y:.3f}) exceeds reach ({max_reach}m)')
            return False
        
        if z < 0.08 or z > 0.35:
            self.get_logger().error(f'Z={z:.3f}m outside safe range [0.08, 0.35]')
            return False
        
        return True

    def set_gripper(self, val):
        """Control gripper - ASYNC with blocking"""
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
            self.get_logger().warn(f'Gripper control issue: {e}')

    def publish_status(self, status, object_id):
        """Publish execution status"""
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