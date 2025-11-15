#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
import json
import time

class ActionNode(Node):
    def __init__(self):
        super().__init__('action_node')
        
        # 1. Connect to the Arm Controller
        self.arm_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/akabot_arm_controller/follow_joint_trajectory'
        )
        
        # 2. Connect to the Hand Controller
        self.gripper_client = ActionClient(
            self, 
            GripperCommand, 
            '/hand_controller/gripper_cmd'
        )
        
        # 3. Subscribe to Brain commands
        self.create_subscription(String, '/arm_command', self.command_callback, 10)
        
        # --- This is the *exact* order from your YAML file ---
        self.arm_joint_names_ordered = [
            'top_plate_joint', 'lower_arm_joint', 'upper_arm_joint', 
            'wrist_joint', 'claw_base_joint'
        ]
        
        self.get_logger().info("Action Node (ROS2 Control) Ready")

    def command_callback(self, msg):
        command = msg.data
        
        if command.startswith("MOVE_JOINTS:"):
            try:
                # Parse the JSON object "{"top_plate_joint": 0.1, ...}"
                json_str = command.split(":", 1)[1]
                target_joints_dict = json.loads(json_str)
                
                self.move_arm(target_joints_dict)
                    
            except Exception as e:
                self.get_logger().error(f"Failed to parse joints JSON: {e}")

        elif command == "CLOSE_GRIPPER":
            self.get_logger().info("Executing: CLOSE_GRIPPER")
            self.move_gripper(0.0) # Closed (Check your URDF limits)
        elif command == "OPEN_GRIPPER":
            self.get_logger().info("Executing: OPEN_GRIPPER")
            self.move_gripper(0.04) # Open (Check your URDF limits)

    def move_arm(self, target_joints_dict):
        if not self.arm_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Arm controller not available")
            return

        goal_msg = FollowJointTrajectory.Goal()
        
        # Set the joint names in the *exact* order the controller expects
        goal_msg.trajectory.joint_names = self.arm_joint_names_ordered
        
        # Build the positions list in that same exact order
        point_positions = []
        try:
            for name in self.arm_joint_names_ordered:
                point_positions.append(target_joints_dict[name])
        except KeyError as e:
            self.get_logger().error(f"Ollama did not send angle for joint: {e}")
            return
            
        point = JointTrajectoryPoint()
        point.positions = point_positions
        point.time_from_start.sec = 1 # Move quickly (1 second)
        
        goal_msg.trajectory.points = [point]
        
        self.get_logger().info(f"Sending 5-joint goal to arm controller.")
        self.arm_client.send_goal_async(goal_msg)

    def move_gripper(self, position):
        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Gripper controller not available")
            return
            
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = 10.0 # Example effort
        
        self.gripper_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()