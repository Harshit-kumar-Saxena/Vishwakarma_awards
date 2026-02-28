#!/usr/bin/env python3
"""
Waypoint Recorder for Gazebo/RViz
Captures robot arm positions and saves them as waypoints

Usage:
    ros2 run arm_vla waypoint_recorder.py
    
Then either:
    1. Manually move the arm in Gazebo and press 's' to save
    2. Click positions in RViz (if interactive markers enabled)
    3. Use keyboard commands to record current position
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import json
import sys
import termios
import tty
import select
from datetime import datetime


class WaypointRecorder(Node):
    def __init__(self):
        super().__init__('waypoint_recorder')
        
        # Joint names for your robot
        self.joint_names = [
            'top_plate_joint',
            'lower_arm_joint',
            'upper_arm_joint',
            'wrist_joint',
            'claw_base_joint'
        ]
        
        self.gripper_joints = ['right_claw_joint', 'left_claw_joint']
        
        # Storage
        self.waypoints = []
        self.current_joint_state = None
        
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Timer for keyboard input
        self.timer = self.create_timer(0.1, self.check_keyboard)
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('📍 WAYPOINT RECORDER STARTED')
        self.get_logger().info('=' * 60)
        self.print_instructions()
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "=" * 60)
        print("COMMANDS:")
        print("  's' - Save current position as waypoint")
        print("  'p' - Print all waypoints")
        print("  'd' - Delete last waypoint")
        print("  'w' - Write waypoints to file")
        print("  'l' - Load waypoints from file")
        print("  'c' - Clear all waypoints")
        print("  'h' - Show this help")
        print("  'q' - Quit")
        print("=" * 60)
        print("\nManually move your robot arm in Gazebo, then press 's' to save!\n")
    
    def joint_callback(self, msg):
        """Store latest joint state"""
        self.current_joint_state = msg
    
    def check_keyboard(self):
        """Check for keyboard input"""
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            self.handle_command(key)
    
    def handle_command(self, key):
        """Handle keyboard commands"""
        if key == 's':
            self.save_waypoint()
        elif key == 'p':
            self.print_waypoints()
        elif key == 'd':
            self.delete_last_waypoint()
        elif key == 'w':
            self.write_waypoints_to_file()
        elif key == 'l':
            self.load_waypoints_from_file()
        elif key == 'c':
            self.clear_waypoints()
        elif key == 'h':
            self.print_instructions()
        elif key == 'q':
            self.write_waypoints_to_file()
            self.get_logger().info('Quitting...')
            rclpy.shutdown()
    
    def save_waypoint(self):
        """Save current robot position as a waypoint"""
        if self.current_joint_state is None:
            self.get_logger().warn('⚠️  No joint state data available yet!')
            return
        
        # Extract arm joint positions
        arm_joints = {}
        gripper_joints = {}
        
        for name in self.joint_names:
            if name in self.current_joint_state.name:
                idx = self.current_joint_state.name.index(name)
                arm_joints[name] = float(self.current_joint_state.position[idx])
        
        for name in self.gripper_joints:
            if name in self.current_joint_state.name:
                idx = self.current_joint_state.name.index(name)
                gripper_joints[name] = float(self.current_joint_state.position[idx])
        
        if len(arm_joints) != len(self.joint_names):
            self.get_logger().error(f'❌ Could not read all joint positions!')
            return
        
        # Create waypoint
        waypoint = {
            'id': len(self.waypoints),
            'name': f'waypoint_{len(self.waypoints)}',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'joint_positions': arm_joints,
            'gripper_positions': gripper_joints
        }
        
        self.waypoints.append(waypoint)
        
        self.get_logger().info(f'✅ Saved Waypoint #{len(self.waypoints)}')
        self.print_waypoint_compact(waypoint)
    
    def print_waypoint_compact(self, wp):
        """Print waypoint in compact format"""
        joints = wp['joint_positions']
        print(f"    Joints: [", end='')
        for i, name in enumerate(self.joint_names):
            print(f"{joints[name]:.3f}", end='')
            if i < len(self.joint_names) - 1:
                print(", ", end='')
        print("]")
        
        if wp['gripper_positions']:
            print(f"    Gripper: {wp['gripper_positions']['right_claw_joint']:.3f}")
    
    def print_waypoints(self):
        """Print all saved waypoints"""
        if not self.waypoints:
            self.get_logger().info('📝 No waypoints saved yet')
            return
        
        print("\n" + "=" * 60)
        print(f"SAVED WAYPOINTS ({len(self.waypoints)} total)")
        print("=" * 60)
        
        for wp in self.waypoints:
            print(f"\nWaypoint {wp['id']}: {wp['name']}")
            print(f"  Time: {wp['timestamp']}")
            self.print_waypoint_compact(wp)
        
        print("=" * 60 + "\n")
    
    def delete_last_waypoint(self):
        """Delete the last waypoint"""
        if not self.waypoints:
            self.get_logger().warn('⚠️  No waypoints to delete')
            return
        
        deleted = self.waypoints.pop()
        self.get_logger().info(f'🗑️  Deleted Waypoint #{deleted["id"]}')
    
    def clear_waypoints(self):
        """Clear all waypoints"""
        count = len(self.waypoints)
        self.waypoints = []
        self.get_logger().info(f'🗑️  Cleared {count} waypoints')
    
    def write_waypoints_to_file(self):
        """Write waypoints to JSON file"""
        if not self.waypoints:
            self.get_logger().warn('⚠️  No waypoints to save')
            return
        
        filename = f'waypoints_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # Also create a Python-friendly format
        py_filename = f'waypoints_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
        
        try:
            # Write JSON
            with open(filename, 'w') as f:
                json.dump({
                    'waypoints': self.waypoints,
                    'robot': 'bot_arm',
                    'joint_names': self.joint_names
                }, f, indent=2)
            
            # Write Python format
            with open(py_filename, 'w') as f:
                f.write("#!/usr/bin/env python3\n")
                f.write('"""\n')
                f.write(f'Waypoints recorded on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write('"""\n\n')
                f.write('# Joint names\n')
                f.write(f'JOINT_NAMES = {self.joint_names}\n\n')
                f.write('# Waypoints as list of joint positions\n')
                f.write('WAYPOINTS = [\n')
                
                for wp in self.waypoints:
                    joints = wp['joint_positions']
                    joint_list = [joints[name] for name in self.joint_names]
                    f.write(f'    {joint_list},  # {wp["name"]}\n')
                
                f.write(']\n\n')
                
                # Add gripper positions
                f.write('# Gripper positions for each waypoint\n')
                f.write('GRIPPER_POSITIONS = [\n')
                for wp in self.waypoints:
                    if wp['gripper_positions']:
                        grip = wp['gripper_positions']['right_claw_joint']
                        f.write(f'    {grip:.3f},  # {wp["name"]}\n')
                    else:
                        f.write(f'    0.0,  # {wp["name"]} (no gripper data)\n')
                f.write(']\n')
            
            self.get_logger().info(f'💾 Saved {len(self.waypoints)} waypoints to:')
            self.get_logger().info(f'   - {filename}')
            self.get_logger().info(f'   - {py_filename}')
            
        except Exception as e:
            self.get_logger().error(f'❌ Failed to write file: {e}')
    
    def load_waypoints_from_file(self):
        """Load waypoints from JSON file"""
        print("\nEnter filename to load (or press Enter for latest): ", end='', flush=True)
        
        # This is a simple implementation - for production you'd want better input handling
        self.get_logger().info('⚠️  File loading from stdin not implemented in this version')
        self.get_logger().info('    Use the JSON file directly in your code instead')


def main(args=None):
    # Setup terminal for character-by-character input
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        rclpy.init(args=args)
        
        # Set terminal to raw mode
        tty.setcbreak(sys.stdin.fileno())
        
        node = WaypointRecorder()
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()