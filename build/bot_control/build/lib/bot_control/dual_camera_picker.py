#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import threading
import time
import math
import sys
import os

# --- FIX: Handle Import for both Launch (Package) and Manual Run (Local) ---
try:
    from bot_control.bot_controller import botController
except ImportError:
    # If running manually from the folder, add current dir to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from bot_controller import botController

class DualCameraPicker(Node):
    def __init__(self):
        super().__init__('dual_camera_picker')

        # Controller node (runs in background)
        self.controller = botController()
        
        # Hand Action Client
        self.hand_client = ActionClient(self.controller, FollowJointTrajectory, '/hand_controller/follow_joint_trajectory')

        # --- HARDCODED TARGET (Ball 2) ---
        self.TARGET_X = 0.22
        self.TARGET_Y = 0.081

        # --- HEIGHTS (Meters relative to Robot Base) ---
        self.HOVER_Z = 0.027       # Safe travel height
        self.PICK_Z = 0.024        # Approach
        self.GRAB_Z = 0.05        # Exact Ball Height (0.045m rounded up)
        
        self.PLATE_HOVER_Z = 0.50
        self.PLATE_PLACE_Z = 0.40

        self.get_logger().info("Dual Camera Picker Initialized (DIRECT MODE).")

    def set_gripper(self, val, wait_for_result=True, timeout=5.0):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['right_claw_joint', 'left_claw_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [float(val), float(-float(val))]
        pt.time_from_start.sec = 1
        goal.trajectory.points = [pt]

        if not self.hand_client.wait_for_server(timeout_sec=timeout):
            self.get_logger().warn("Hand server not available")
            return

        send_fut = self.hand_client.send_goal_async(goal)
        start = time.time()
        while not send_fut.done():
            if time.time() - start > timeout:
                break
            time.sleep(0.02)
        
        if wait_for_result:
            time.sleep(1.0)

    def try_reach_pose(self, x, y, z):
        yaw = math.atan2(y, x)
        
        # Try pitch angles: Vertical (1.57) -> Angled (0.8)
        pitch_list = [1.57, 1.3, 1.0, 0.8, 0.5]  
        
        for pitch in pitch_list:
            self.get_logger().info(f"Trying IK: X={x:.3f} Y={y:.3f} Z={z:.3f} | Pitch={pitch:.2f}")
            pose = self.controller.create_pose(x, y, z, roll=0.0, pitch=pitch, yaw=yaw)
            
            if self.controller.move_to_pose(pose, duration=4.0):
                self.get_logger().info(f"✅ Reached target at pitch={pitch:.2f}")
                return True
        
        self.get_logger().error(f"❌ IK FAILED for ({x:.3f}, {y:.3f}, {z:.3f})")
        return False

    def run_logic(self):
        self.get_logger().info("Waiting for setup (2s)...")
        time.sleep(2.0)

        self.get_logger().info(f"📍 TARGET: X={self.TARGET_X}, Y={self.TARGET_Y}")

        # 1. MOVE HOME & OPEN GRIPPER
        self.get_logger().info("🏠 Moving Home & Opening Gripper...")
        self.controller.move_to_named_target('home')
        self.set_gripper(-0.5)

        # 2. HOVER (High Safety Move)
        self.get_logger().info("⬆️  Moving to Hover...")
        if not self.try_reach_pose(self.TARGET_X, self.TARGET_Y, self.HOVER_Z):
            return

        # 3. APPROACH
        self.get_logger().info(f"⬇️  Descending to Approach (Z={self.PICK_Z})...")
        self.try_reach_pose(self.TARGET_X, self.TARGET_Y, self.PICK_Z)

        # 4. GRAB (Final Descent)
        self.get_logger().info(f"🎯 Descending to GRAB (Z={self.GRAB_Z})...")
        if not self.try_reach_pose(self.TARGET_X, self.TARGET_Y, self.GRAB_Z):
             self.get_logger().warn("Could not reach exact Grab Z, attempting grip here...")

        # 5. CLOSE GRIPPER
        self.get_logger().info("✊ Closing Gripper...")
        self.set_gripper(-0.1)
        time.sleep(1.0) 

        # 6. LIFT
        self.get_logger().info("⬆️  Lifting...")
        self.try_reach_pose(self.TARGET_X, self.TARGET_Y, self.HOVER_Z)

        # 7. PLACE
        plate_x = self.TARGET_X
        plate_y = self.TARGET_Y - 0.20 
        
        self.get_logger().info(f"🚚 Placing at X={plate_x:.3f}, Y={plate_y:.3f}")

        self.try_reach_pose(plate_x, plate_y, self.PLATE_HOVER_Z)
        self.try_reach_pose(plate_x, plate_y, self.PLATE_PLACE_Z)
        self.set_gripper(-0.5) # Release
        self.try_reach_pose(plate_x, plate_y, self.HOVER_Z)

        # 8. FINISH
        self.controller.move_to_named_target('home')
        self.get_logger().info("✅ Mission Complete!")

def main(args=None):
    rclpy.init(args=args)
    picker = DualCameraPicker()
    
    # Run spinner in thread
    spinner = threading.Thread(target=rclpy.spin, args=(picker.controller,), daemon=True)
    spinner.start()
    
    try:
        picker.run_logic()
    except KeyboardInterrupt:
        pass
    finally:
        picker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()