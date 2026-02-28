# Autonomous Pick-and-Place Robotic Arm (VLA-Powered) 

This project features an intelligent 5-DOF robotic arm simulation designed for autonomous pick-and-place tasks. By integrating a **Vision-Language-Action** (VLA) model powered by Ollama (Llama 3B), the system interprets natural language instructions to execute precise operations within a high-fidelity virtual environment.
```text
vishwakarma_awards/
├── modelfiles/                # VLA model definitions (e.g., arm_brain.modelfile)
├── src/                       # Source code packages
|   ├── assets/                    # Project media (images, GIFs, demo videos)
│   ├── arm_vla_pkg/           # VLA logic nodes (brain, vision, action)
│   ├── bot_control/           # Hardware/Simulation control nodes
│   ├── bot_description/       # URDF and 3D mesh files
│   ├── bot_gazebo/            # Gazebo simulation worlds and launch files
│   └── bot_moveit_config/     # MoveIt 2 motion planning configuration
├── .gitignore                 # Files to ignore (build, install, log)
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```
---

## Project Media
<img src="assets/robotic arm.jpeg" width="600" alt="AMR Mapping"> 

### Demo Video
[![Watch the Demo]](https://drive.google.com/drive/folders/10Yyzd4tVNzO6YSYjObQKvspG1mxJ9qfd?usp=sharing)
---

## 🚀 Key Features
- **VLA Integration**: Uses a high-level "Brain Node" to interpret visual data and language instructions.
- **Spatial Reasoning**: Dual-camera system (Eye-in-Hand and Static) for 6D pose estimation and obstacle avoidance.
- **MoveIt 2 Support**: Optimized motion planning for smooth, collision-free trajectories.

---

## 🛠️ System Requirements

### Software
- **ROS 2**: Humble Hawksbill
- **Motion Planning**: MoveIt 2
- **VLA Framework**: TinyVLA or custom OpenVLA implementation. 
- **Rviz2 and Ign Gazebo**: for Visiualzation
---

## 🚦 Usage Guide

### 1. Build the Workspace
```text
colcon build --symlink-install
source install/setup.bash

Start VLA Brain
ros2 launch arm_vla_pkg vla_main.launch.py
```
📄 License
This project is licensed under the MIT License.
