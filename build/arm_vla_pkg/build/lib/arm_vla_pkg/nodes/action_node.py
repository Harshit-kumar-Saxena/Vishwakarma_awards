#!/usr/bin/env python3
"""
Test Action Node - Waypoints Only (Robust Timing)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import json
import time
import threading

class WaypointActionNode(Node):
    def __init__(self):
        super().__init__('vla_action')
        
        # Use ReentrantCallbackGroup to allow parallel execution
        self.cb_group = ReentrantCallbackGroup()
        
        self.joint_names = [
            'top_plate_joint', 'lower_arm_joint', 'upper_arm_joint', 
            'wrist_joint', 'claw_base_joint'
        ]
        
        # ========== WAYPOINTS ==========
        
        # 1. Initial Position
        self.INITIAL_POSITION = [3.365, 1.636, -0.542, -1.700, 0.000]

        # 2. Waypoint 2
        self.WAYPOINT_2 = [3.365, 2.074, -1.479, -1.700, 0.000]
        
        # 3. Waypoint 3
        self.WAYPOINT_3 = [3.365, 1.725, -1.479, -1.570, 0.000]

        # 4. Waypoint 4 (The Pick Position)
        self.WAYPOINT_4 = [3.449, 1.434, -1.204, -1.296, 0.000]
        
        self.APPROACH_SEQUENCE = [
            self.WAYPOINT_2,
            self.WAYPOINT_3,
            self.WAYPOINT_4
        ]
        
        # Post-Pick Waypoints
        self.POST_PICK_1 = [3.449, 1.096, -0.358, 0.781, 0.00]
        self.POST_PICK_2 = [2.634, 1.096, -0.358, -1.424, 0.00]

        self.POST_PICK_SEQUENCE = [
            self.POST_PICK_1,
            self.POST_PICK_2
        ]
        
        self.current_joint_state = None
        
        # ROS Communication
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10, callback_group=self.cb_group)
        
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, '/bot_arm_controller/follow_joint_trajectory', callback_group=self.cb_group)
        
        self.hand_client = ActionClient(
            self, FollowJointTrajectory, '/hand_controller/follow_joint_trajectory', callback_group=self.cb_group)
        
        self.create_subscription(String, '/vla/action_command', self.action_callback, 10, callback_group=self.cb_group)
        
        # Start init in a thread
        threading.Thread(target=self.wait_for_controllers, daemon=True).start()
        self.get_logger().info('✅ Waypoint Test Node Initialized')

    def wait_for_controllers(self):
        self.trajectory_client.wait_for_server(timeout_sec=10.0)
        self.hand_client.wait_for_server(timeout_sec=10.0)
        self.get_logger().info('✅ Controllers Connected')

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def action_callback(self, msg):
        try:
            action = json.loads(msg.data)
            self.get_logger().info(f'🎯 COMMAND RECEIVED: {action["object_id"]}')
            threading.Thread(target=self.execute_waypoints).start()
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def execute_waypoints(self):
        self.get_logger().info('🚀 STARTED: Executing Waypoint Sequence')

        # 1. MOVE TO INITIAL
        self.get_logger().info('1. Moving to Initial Position')
        if not self.move_to_joints(self.INITIAL_POSITION, 6.0): return

        # 2. OPEN GRIPPER 
        self.get_logger().info('2. Opening Gripper')
        self.set_gripper(-0.5) 

        # 3. EXECUTE APPROACH SEQUENCE
        self.get_logger().info('3. Moving through Sequence')
        for i, wp in enumerate(self.APPROACH_SEQUENCE):
            self.get_logger().info(f'   -> Moving to Waypoint {i+2}')
            if not self.move_to_joints(wp, 6.0): 
                self.get_logger().error(f'❌ Failed at Waypoint {i+2}')
                return
            time.sleep(0.5)

        # === CRITICAL FIX: STOP COMPLETELY BEFORE GRIPPING ===
        self.get_logger().info('   🛑 STOPPING ARM (Waiting 3s for stability)...')
        time.sleep(3.0) 

        # 4. CLOSE GRIPPER
        self.get_logger().info('4. Closing Gripper (Simulated Pick)')
        # This function is now BLOCKING - it won't return until gripper is done
        if not self.set_gripper(-0.1): 
            self.get_logger().error('❌ Gripper failed to close properly')
            return
        
        # === CRITICAL FIX: ENSURE GRIP IS SECURE ===
        self.get_logger().info('   🔒 Grip Secure. Waiting 2s before lifting...')
        time.sleep(2.0)

        # 5. EXECUTE POST-PICK SEQUENCE
        self.get_logger().info('5. Moving through Post-Pick Sequence')
        for i, wp in enumerate(self.POST_PICK_SEQUENCE):
            self.get_logger().info(f'   -> Moving to Post-Pick Waypoint {i+1}')
            if not self.move_to_joints(wp, 6.0): 
                self.get_logger().error(f'❌ Failed at Post-Pick Waypoint {i+1}')
                return
            time.sleep(1.0) # Move slower with object

        self.get_logger().info('✅ SEQUENCE COMPLETE')
        time.sleep(1.0)
        self.get_logger().info('2. Opening Gripper')
        self.set_gripper(-0.3) 

    # ================= HELPERS =================

    def wait_for_future(self, future, timeout_sec):
        """Waits for a future without blocking the main thread spinner"""
        start = time.time()
        while not future.done():
            time.sleep(0.05)
            if time.time() - start > timeout_sec: return False
        return True

    def move_to_joints(self, positions, duration):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        goal.trajectory.points = [point]

        self.get_logger().info(f'      Sending goal (Duration: {duration}s)...')
        f = self.trajectory_client.send_goal_async(goal)
        if not self.wait_for_future(f, 2.0): 
            self.get_logger().error('      Goal send failed/timeout')
            return False
        
        gh = f.result()
        if not gh.accepted: 
            self.get_logger().error('      Goal rejected')
            return False
        
        rf = gh.get_result_async()
        # Wait for movement to finish + 5 second buffer
        if not self.wait_for_future(rf, duration + 5.0):
            self.get_logger().warn(f"      ⚠️ Move timed out, continuing...")
            return True
        return True

    def set_gripper(self, val):
        """
        UPDATED: Now waits for the gripper action to fully complete
        """
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['right_claw_joint', 'left_claw_joint']
        pt = JointTrajectoryPoint()
        
        # Correct directions: Right (neg), Left (pos)
        pt.positions = [float(val), float(abs(val))] 
        pt.time_from_start = Duration(sec=4, nanosec=0) # Give it 4 seconds to close smoothly
        goal.trajectory.points = [pt]

        self.get_logger().info(f'      Setting gripper to {val}...')
        
        # 1. Send Goal
        f = self.hand_client.send_goal_async(goal)
        if not self.wait_for_future(f, 2.0):
            self.get_logger().error('      Gripper goal send failed')
            return False

        gh = f.result()
        if not gh.accepted:
            self.get_logger().error('      Gripper goal rejected')
            return False

        # 2. Wait for Result (BLOCKING with LONG TIMEOUT)
        rf = gh.get_result_async()
        if not self.wait_for_future(rf, 15.0): # Wait up to 15 seconds for simulation to catch up
            self.get_logger().warn('      Gripper action timed out (but continuing)')
            return True
            
        return True

def main(args=None):
    rclpy.init(args=args)
    node = WaypointActionNode()
    
    # REQUIRED for Threading
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()