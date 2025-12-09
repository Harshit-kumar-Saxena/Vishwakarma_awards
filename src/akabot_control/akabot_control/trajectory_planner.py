#!/usr/bin/env python3
"""
Trajectory Planner for 5-DOF Akabot Arm
Uses analytical IK and MoveIt for trajectory planning

Joint Structure:
1. top_plate_joint (revolute, Z-axis) - Rotation
2. lower_arm_joint (revolute, X-axis) - Shoulder pitch
3. upper_arm_joint (revolute, X-axis) - Elbow pitch  
4. wrist_joint (revolute, X-axis) - Wrist pitch
5. claw_base_joint (revolute, Y-axis) - Wrist yaw
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ArmKinematics:
    """Arm dimensions from URDF"""
    # Link lengths (approximate from geometry)
    L1 = 0.0  # base_plate to lower_arm (vertical offset)
    L2 = 0.13  # lower_arm length (X direction)
    L3 = 0.185  # upper_arm length  (main arm segment)
    L4 = 0.065  # wrist length
    L5 = 0.07   # claw_base to gripper tip
    
    # Z offset from base
    Z_BASE = 0.048  # top_plate height
    
    # Joint limits (from URDF)
    JOINT_LIMITS = {
        'top_plate_joint': (1.56, 4.68),      # Z-rotation
        'lower_arm_joint': (0.0, 3.12),       # X-rotation (shoulder)
        'upper_arm_joint': (-1.7, 1.7),       # X-rotation (elbow)
        'wrist_joint': (-1.7, 1.7),           # X-rotation (wrist)
        'claw_base_joint': (-3.17, 0.0)       # Y-rotation (yaw)
    }


class TrajectoryPlanner:
    """Plan and execute trajectories for the 5-DOF arm"""
    
    def __init__(self):
        self.kinematics = ArmKinematics()
        self.current_pose = None
        self.home_pose = {
            'top_plate_joint': 3.14,      # 180 degrees
            'lower_arm_joint': 1.57,      # 90 degrees (arm up)
            'upper_arm_joint': 0.0,       # Neutral
            'wrist_joint': 0.0,           # Neutral
            'claw_base_joint': -1.57      # Neutral
        }
    
    def forward_kinematics(self, joint_values: dict) -> Tuple[float, float, float]:
        """
        Calculate end-effector position from joint angles
        Returns (x, y, z) position
        """
        # Extract joint angles
        theta1 = joint_values.get('top_plate_joint', 0)      # Z rotation
        theta2 = joint_values.get('lower_arm_joint', 0)      # Shoulder
        theta3 = joint_values.get('upper_arm_joint', 0)      # Elbow
        theta4 = joint_values.get('wrist_joint', 0)          # Wrist
        
        # Z coordinate (height)
        z = self.kinematics.Z_BASE
        z += self.kinematics.L2 * math.sin(theta2)
        z += self.kinematics.L3 * math.sin(theta2 + theta3)
        z += self.kinematics.L4 * math.sin(theta2 + theta3 + theta4)
        
        # Distance in XY plane
        r = self.kinematics.L2 * math.cos(theta2)
        r += self.kinematics.L3 * math.cos(theta2 + theta3)
        r += self.kinematics.L4 * math.cos(theta2 + theta3 + theta4)
        
        # Apply XY rotation (top_plate_joint)
        x = r * math.cos(theta1)
        y = r * math.sin(theta1)
        
        return (x, y, z)
    
    def inverse_kinematics(self, target_x: float, target_y: float, target_z: float) -> Optional[dict]:
        """
        Calculate joint angles for target end-effector position
        Uses analytical 5-DOF solution
        
        Args:
            target_x, target_y, target_z: Target position in meters
            
        Returns:
            Dictionary of joint values, or None if unreachable
        """
        # Step 1: Calculate base rotation (top_plate_joint)
        base_rotation = math.atan2(target_y, target_x)
        
        # Step 2: Calculate distance in XY plane
        r_xy = math.sqrt(target_x**2 + target_y**2)
        
        # Step 3: Calculate Z-relative position
        z_rel = target_z - self.kinematics.Z_BASE
        
        # Step 4: Solve 2-link planar arm (L3, L4 segments)
        # This is the main arm in the vertical plane
        arm_length = math.sqrt(r_xy**2 + z_rel**2)
        
        # Check if target is reachable
        max_reach = self.kinematics.L3 + self.kinematics.L4 + self.kinematics.L2
        if arm_length > max_reach:
            return None
        
        # Use simple geometric IK
        # Angle to target from shoulder
        alpha = math.atan2(z_rel, r_xy)
        
        # Calculate using law of cosines
        try:
            cos_angle = (self.kinematics.L3**2 + arm_length**2 - self.kinematics.L4**2) / \
                       (2 * self.kinematics.L3 * arm_length + 1e-6)
            
            if abs(cos_angle) > 1.0:
                return None
            
            beta = math.acos(cos_angle)
            shoulder_angle = alpha + beta
            
            # Elbow angle
            cos_elbow = (self.kinematics.L3**2 + self.kinematics.L4**2 - arm_length**2) / \
                       (2 * self.kinematics.L3 * self.kinematics.L4 + 1e-6)
            
            if abs(cos_elbow) > 1.0:
                return None
            
            elbow_angle = math.acos(cos_elbow) - math.pi
            
            # Build joint dictionary
            joints = {
                'top_plate_joint': base_rotation,
                'lower_arm_joint': shoulder_angle,
                'upper_arm_joint': elbow_angle,
                'wrist_joint': 0.0,  # Keep wrist neutral
                'claw_base_joint': -1.57  # Keep neutral
            }
            
            # Check joint limits
            if self._check_joint_limits(joints):
                return joints
            else:
                return None
                
        except (ValueError, ZeroDivisionError):
            return None
    
    def _check_joint_limits(self, joints: dict) -> bool:
        """Check if joint values are within limits"""
        for joint_name, angle in joints.items():
            if joint_name in self.kinematics.JOINT_LIMITS:
                min_lim, max_lim = self.kinematics.JOINT_LIMITS[joint_name]
                if angle < min_lim or angle > max_lim:
                    return False
        return True
    
    def plan_approach_trajectory(self, target_x: float, target_y: float, target_z: float,
                                approach_distance: float = 0.05) -> Optional[List[dict]]:
        """
        Plan trajectory: home -> hover -> approach -> target
        
        Args:
            target_x, target_y, target_z: Target position
            approach_distance: Distance to stop before target for precision
            
        Returns:
            List of waypoint joint configurations
        """
        waypoints = []
        
        # Waypoint 1: Home pose
        waypoints.append(self.home_pose.copy())
        
        # Waypoint 2: Hover above target (high Z)
        hover_z = max(target_z + 0.15, 0.35)
        hover_ik = self.inverse_kinematics(target_x, target_y, hover_z)
        if hover_ik:
            waypoints.append(hover_ik)
        else:
            return None
        
        # Waypoint 3: Approach pose (slightly above target)
        approach_z = max(target_z + approach_distance, 0.20)
        approach_ik = self.inverse_kinematics(target_x, target_y, approach_z)
        if approach_ik:
            waypoints.append(approach_ik)
        else:
            return None
        
        return waypoints
    
    def plan_place_trajectory(self, target_x: float, target_y: float, target_z: float) -> Optional[List[dict]]:
        """
        Plan trajectory to place object in bowl
        
        Args:
            target_x, target_y, target_z: Target bowl position
            
        Returns:
            List of waypoint joint configurations
        """
        waypoints = []
        
        # Approach the bowl location
        approach_z = target_z + 0.10
        approach_ik = self.inverse_kinematics(target_x, target_y, approach_z)
        if approach_ik:
            waypoints.append(approach_ik)
        else:
            return None
        
        # Lower into bowl
        lower_z = target_z + 0.01
        lower_ik = self.inverse_kinematics(target_x, target_y, lower_z)
        if lower_ik:
            waypoints.append(lower_ik)
        else:
            return None
        
        return waypoints
    
    def plan_return_trajectory(self) -> List[dict]:
        """
        Plan return to home position
        
        Returns:
            List of waypoint joint configurations
        """
        return [self.home_pose.copy()]
    
    def interpolate_trajectory(self, start: dict, end: dict, num_steps: int = 20) -> List[dict]:
        """
        Interpolate between two joint configurations
        
        Args:
            start: Starting joint configuration
            end: Ending joint configuration
            num_steps: Number of interpolation steps
            
        Returns:
            List of interpolated configurations
        """
        trajectory = []
        
        for i in range(num_steps + 1):
            alpha = i / num_steps
            waypoint = {}
            
            for joint in start.keys():
                start_val = start[joint]
                end_val = end[joint]
                
                # Linear interpolation
                waypoint[joint] = start_val + alpha * (end_val - start_val)
            
            trajectory.append(waypoint)
        
        return trajectory
    
    def smooth_trajectory(self, waypoints: List[dict], step_size: int = 5) -> List[dict]:
        """
        Smooth trajectory by interpolating between waypoints
        
        Args:
            waypoints: List of key waypoints
            step_size: Number of interpolation steps between waypoints
            
        Returns:
            Smoothed trajectory
        """
        if len(waypoints) < 2:
            return waypoints
        
        smooth_trajectory = []
        smooth_trajectory.append(waypoints[0])
        
        for i in range(len(waypoints) - 1):
            interp = self.interpolate_trajectory(waypoints[i], waypoints[i+1], step_size)
            smooth_trajectory.extend(interp[1:])  # Skip first (already added)
        
        return smooth_trajectory


if __name__ == '__main__':
    planner = TrajectoryPlanner()
    
    # Example: Plan to reach position (0.15, 0.05, 0.25)
    target = (0.15, 0.05, 0.25)
    
    # Test IK
    joint_config = planner.inverse_kinematics(*target)
    if joint_config:
        print(f"IK Solution for {target}:")
        for joint, angle in joint_config.items():
            print(f"  {joint}: {math.degrees(angle):.2f}°")
        
        # Verify with FK
        fk_result = planner.forward_kinematics(joint_config)
        print(f"FK Verification: {fk_result}")
    else:
        print(f"Target {target} is unreachable")
    
    # Test trajectory planning
    traj = planner.plan_approach_trajectory(*target)
    if traj:
        print(f"\nTrajectory has {len(traj)} waypoints")
    else:
        print("Could not plan trajectory")
