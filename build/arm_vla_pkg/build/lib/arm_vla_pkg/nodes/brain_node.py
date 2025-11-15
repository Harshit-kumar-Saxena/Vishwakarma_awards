#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import ollama
import json
import time

# Name of your Ollama model
BRAIN_MODEL = 'my_arm_brain' 

# --- CRITICAL PROMPT UPDATE ---
# We now tell the LLM the *exact* joint names.
# We also give it commands for the gripper.
SYSTEM_PROMPT = """
You are a Visual Servoing Robot Arm Controller.
Your goal is to center the 'ball' (x=0, y=0) and get close (radius > 60).

You receive:
1. "vision": A string describing what the camera sees.
2. "arm_joints": A JSON object of the 5 arm joints and their CURRENT radian positions.
   {"top_plate_joint": 0.0, "lower_arm_joint": 0.0, ...}
3. "gripper_joint": The current position of the 'right_claw_joint'. (0.0 is closed).

Your output MUST be ONE of the following:
1. A JSON object with NEW target positions for the 5 arm joints.
   Example: {"top_plate_joint": 0.1, "lower_arm_joint": -0.5, "upper_arm_joint": 1.2, "wrist_joint": 0.0, "claw_base_joint": 0.5}
2. A single command string: "OPEN_GRIPPER"
3. A single command string: "CLOSE_GRIPPER"
4. A single command string: "WAIT"

Rules for moving the arm:
- To move the ball LEFT (ball.x is NEGATIVE), INCREASE 'top_plate_joint'.
- To move the ball RIGHT (ball.x is POSITIVE), DECREASE 'top_plate_joint'.
- To move the ball UP (ball.y is NEGATIVE), DECREASE 'lower_arm_joint' or 'upper_arm_joint'.
- To move the ball DOWN (ball.y is POSITIVE), INCREASE 'lower_arm_joint' or 'upper_arm_joint'.
- 'wrist_joint' and 'claw_base_joint' control the end effector orientation.
- ONLY output new joint angles if you need to move the arm.
- If the ball is centered (x~0, y~0) and radius > 60, output "CLOSE_GRIPPER".
"""

class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        self.get_logger().info("Brain Node (Name-Aware) Started")
        
        # Data containers
        self.latest_vision = "ball:missing"
        # We now use a dictionary to store joint states
        self.arm_joint_names = [
            'top_plate_joint', 'lower_arm_joint', 'upper_arm_joint', 
            'wrist_joint', 'claw_base_joint'
        ]
        self.gripper_joint_name = 'right_claw_joint'
        
        self.current_arm_joints = {name: 0.0 for name in self.arm_joint_names}
        self.current_gripper_joint = 0.0
        self.processing = False

        # Subscribe to Vision
        self.create_subscription(String, '/world_state', self.vision_callback, 10)
        
        # Subscribe to Real Robot Joints
        self.create_subscription(JointState, '/joint_states', self.joints_callback, 10)

        # Publisher to Action Node
        self.command_pub = self.create_publisher(String, '/arm_command', 10)

        # Control Loop Timer
        self.create_timer(0.5, self.control_loop) # 2 Hz

    def vision_callback(self, msg):
        self.latest_vision = msg.data

    def joints_callback(self, msg):
        # This is now much smarter. It finds the joints by name.
        for i, name in enumerate(msg.name):
            if name in self.current_arm_joints:
                self.current_arm_joints[name] = msg.position[i]
            elif name == self.gripper_joint_name:
                self.current_gripper_joint = msg.position[i]

    def control_loop(self):
        if self.processing: return
        if "ball:missing" in self.latest_vision:
            self.get_logger().info("Waiting for ball...")
            return

        self.processing = True
        
        # 1. Construct Prompt
        prompt = f"""
        Current Status:
        - vision: "{self.latest_vision}"
        - arm_joints: {json.dumps(self.current_arm_joints)}
        - gripper_joint: {self.current_gripper_joint:.3f}
        
        What is your next single command (JSON object for joints, or string command)?
        """

        try:
            # 2. Ask Ollama
            response = ollama.chat(
                model=BRAIN_MODEL, 
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ]
            )
            command_text = response['message']['content'].strip()
            
            # 3. Clean and Publish
            if command_text.startswith("{"):
                # This is a JSON joint command
                self.get_logger().info(f"Ollama suggests joints: {command_text}")
                msg = String()
                msg.data = f"MOVE_JOINTS:{command_text}"
                self.command_pub.publish(msg)
            elif "OPEN_GRIPPER" in command_text:
                self.get_logger().info(f"Ollama suggests: OPEN_GRIPPER")
                msg = String()
                msg.data = "OPEN_GRIPPER"
                self.command_pub.publish(msg)
            elif "CLOSE_GRIPPER" in command_text:
                self.get_logger().info(f"Ollama suggests: CLOSE_GRIPPER")
                msg = String()
                msg.data = "CLOSE_GRIPPER"
                self.command_pub.publish(msg)
            else:
                # This includes "WAIT"
                pass # Do nothing

        except Exception as e:
            self.get_logger().error(f"Brain Error: {e}")
        
        self.processing = False

def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()