---
sidebar_position: 3
---

# Tools and Software Requirements: Physical AI & Humanoid Robotics Course

## Overview

This document provides comprehensive information about the software tools, hardware requirements, and development environments needed for each week of the Physical AI & Humanoid Robotics course. The requirements are organized by week to enable progressive setup and to accommodate different institutional contexts.

## Week 1: ROS 2 Fundamentals and Architecture

### Software Requirements

#### Core Dependencies
- **Operating System**: Ubuntu 20.04 LTS or Ubuntu 22.04 LTS
- **ROS 2 Distribution**: Humble Hawksbill (latest patch version)
- **Programming Language**: Python 3.8+ (system Python or conda environment)
- **Development Environment**: VS Code with ROS extension pack or preferred IDE

#### Installation Commands
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install ROS 2 Humble
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop-full
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2 environment
source /opt/ros/humble/setup.bash
```

#### Development Tools
- **Build System**: Colcon for ROS 2 package building
- **Package Manager**: APT for system packages, pip for Python packages
- **Version Control**: Git with recommended configuration
- **Debugging Tools**: rqt, ros2 command line tools, Python debugger

#### Recommended VS Code Extensions
- ROS (for VS Code)
- Python
- C/C++
- GitLens
- Docker

### Hardware Requirements
- **Processor**: Multi-core CPU (Intel i5 or equivalent AMD)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 50GB free disk space
- **Network**: Stable internet connection for package installation

### Setup Verification
```bash
# Verify ROS 2 installation
ros2 --version

# Test basic ROS 2 functionality
ros2 run turtlesim turtlesim_node &
ros2 run turtlesim turtle_teleop_key

# Check available packages
ros2 pkg list | grep std_msgs
```

## Week 2: Advanced ROS 2 Concepts and Navigation

### Additional Software Requirements

#### Navigation Stack
```bash
# Install navigation packages
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
sudo apt install ros-humble-dwb-core
sudo apt install ros-humble-robot-localization
sudo apt install ros-humble-slam-toolbox
```

#### Simulation Tools
```bash
# Gazebo simulation (will be installed in Week 4, but can be pre-installed)
sudo apt install ros-humble-gazebo-ros-pkgs
```

#### Configuration Files
Create navigation configuration files in your workspace:
- Costmap configuration (local and global)
- Controller configuration (DWA, MPC, etc.)
- Planner configuration (NavFn, GlobalPlanner, etc.)

### Development Environment Enhancements
- **Visualization**: RViz2 with navigation plugins
- **Simulation**: Gazebo Garden for testing navigation
- **Debugging**: Navigation-specific tools and visualizers

### Setup Verification
```bash
# Verify navigation installation
ros2 launch nav2_bringup tb3_simulation_launch.py

# Test navigation components
ros2 run nav2_util lifecycle_bringup

# Check available navigation parameters
ros2 param list
```

## Week 3: Sensor Fusion and System Debugging

### Additional Software Requirements

#### Sensor Processing Libraries
```bash
# Install sensor fusion packages
sudo apt install ros-humble-robot-localization
sudo apt install ros-humble-hector-sensors-description
sudo apt install ros-humble-imu-tools
sudo apt install ros-humble-laser-filters
sudo apt install ros-humble-robot-state-publisher
```

#### Filtering and Estimation
```bash
# Kalman filtering libraries
sudo apt install ros-humble-filter-interface
sudo apt install ros-humble-filter-base
sudo apt install ros-humble-velocity-controllers
```

#### Debugging and Profiling Tools
```bash
# Performance analysis
sudo apt install ros-humble-rqt-top
sudo apt install ros-humble-rqt-plot
sudo apt install ros-humble-rqt-graph
sudo apt install ros-humble-rqt-topic

# Memory and CPU monitoring
sudo apt install htop
sudo apt install nethogs
```

### Hardware Requirements for Week 3
- **Additional Sensors**: IMU, LiDAR, Camera (for real hardware testing)
- **Processing Power**: Enhanced CPU for real-time filtering
- **Memory**: Additional 4GB RAM if processing multiple sensors

### Setup Verification
```bash
# Test sensor fusion
ros2 launch robot_localization ekf.launch.py

# Verify filter performance
ros2 run rqt_plot rqt_plot

# Check sensor data flow
ros2 topic echo /imu/data sensor_msgs/Imu
```

## Week 4: Gazebo Simulation and Physics Modeling

### Software Requirements

#### Gazebo Installation
```bash
# Install Gazebo Garden (recommended version)
sudo apt install ros-humble-gazebo-ros
sudo apt install ros-humble-gazebo-plugins
sudo apt install ros-humble-gazebo-dev
sudo apt install ros-humble-gazebo-msgs
sudo apt install ros-humble-gazebo-ros-pkgs

# Install Gazebo standalone (if not already installed)
wget https://osrf-distributions.s3.amazonaws.com/gazebo/releases/gazebo-ros_pkgs-ubuntu-jammy-20230801.133320.tar.bz2
# Follow Gazebo installation guide for your specific version
```

#### Physics and Modeling Tools
```bash
# Physics engines and modeling
sudo apt install libgazebo11-dev
sudo apt install libsdformat-dev
sudo apt install ros-humble-urdf
sudo apt install ros-humble-xacro
sudo apt install ros-humble-joint-state-publisher
```

#### Visualization and Editing
```bash
# Model editing and visualization
sudo apt install meshlab
sudo apt install blender
sudo apt install openscad
```

### Hardware Requirements for Simulation
- **Graphics**: Dedicated GPU (NVIDIA RTX series recommended)
- **Memory**: 16GB RAM minimum, 32GB recommended for complex scenes
- **Storage**: Additional 20GB for Gazebo models and environments
- **CPU**: Multi-core processor with good single-core performance

### Setup Verification
```bash
# Launch Gazebo
gazebo --version

# Test basic simulation
gazebo

# Launch ROS 2 integration
ros2 launch gazebo_ros gazebo.launch.py

# Test with TurtleBot3 (if available)
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

## Week 5: Advanced Simulation and Reality Transfer

### Additional Software Requirements

#### Advanced Simulation Tools
```bash
# Domain randomization tools
pip install domain-randomization-tools

# Synthetic data generation
sudo apt install ros-humble-synthetic-data
sudo apt install ros-humble-ignition-tools
```

#### Computer Vision Enhancement
```bash
# OpenCV with Gazebo integration
sudo apt install ros-humble-vision-opencv
sudo apt install ros-humble-cv-bridge
sudo apt install ros-humble-image-transport
sudo apt install python3-opencv
```

#### Performance Analysis
```bash
# Simulation performance tools
sudo apt install ros-humble-gazebo-performance
sudo apt install ros-humble-simulation-stats
```

### Setup Verification
```bash
# Test domain randomization
ros2 run domain_randomization domain_randomization_node

# Verify synthetic data generation
ros2 run synthetic_data generate_dataset

# Check simulation performance
ros2 run gazebo_performance monitor_performance
```

## Week 6: NVIDIA Isaac Platform and Perception Pipelines

### Software Requirements

#### NVIDIA GPU Drivers and CUDA
```bash
# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535

# Verify GPU is detected
nvidia-smi

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-11-8
```

#### Isaac ROS Packages
```bash
# Install Isaac ROS packages
sudo apt install ros-humble-isaac-ros-common
sudo apt install ros-humble-isaac-ros-dnn-inference
sudo apt install ros-humble-isaac-ros-image-pipeline
sudo apt install ros-humble-isaac-ros-visual-slam
sudo apt install ros-humble-isaac-ros-apriltag
sudo apt install ros-humble-isaac-ros-people-segmentation
sudo apt install ros-humble-isaac-ros-segmentation
```

#### Docker and Containerization
```bash
# Install Docker and NVIDIA Container Toolkit
sudo apt install docker.io
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Deep Learning Frameworks
```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional AI libraries
pip install transformers
pip install opencv-python
pip install numpy
pip install scipy
```

### Hardware Requirements for AI
- **GPU**: NVIDIA RTX 3080 or higher (4090 recommended)
- **Memory**: 32GB RAM minimum for large models
- **Storage**: SSD with 50GB free space for models and datasets
- **Cooling**: Adequate cooling for GPU-intensive workloads

### Setup Verification
```bash
# Verify CUDA installation
nvcc --version
nvidia-smpi

# Test Isaac ROS packages
ros2 run isaac_ros_image_proc image_format_converter

# Test Docker GPU support
docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Verify Isaac perception packages
ros2 launch isaac_ros_apriltag isaac_ros_apriltag.launch.py
```

## Week 7: Neural Network Optimization and Path Planning

### Additional Software Requirements

#### Optimization Tools
```bash
# TensorRT installation
sudo apt install tensorrt
sudo apt install python3-libnvinfer
sudo apt install python3-libnvinfer-dev

# Optimization libraries
pip install tensorrt
pip install onnx
pip install onnxruntime
pip install onnxruntime-gpu
```

#### Path Planning Libraries
```bash
# OMPL (Open Motion Planning Library)
sudo apt install ros-humble-ompl
sudo apt install libompl-dev

# Custom path planning tools
sudo apt install ros-humble-nav2-planners
sudo apt install ros-humble-dwb-core
```

#### Performance Analysis Tools
```bash
# Profiling tools
sudo apt install ros-humble-performance-test-tools
sudo apt install nvtop  # GPU monitoring
```

### Setup Verification
```bash
# Test TensorRT installation
python3 -c "import tensorrt as trt; print(trt.__version__)"

# Test ONNX runtime
python3 -c "import onnxruntime as ort; print(ort.__version__)"

# Test path planning
ros2 run ompl ompl_planning_example
```

## Week 8: Manipulation Control and GPU Optimization

### Additional Software Requirements

#### Manipulation Packages
```bash
# MoveIt 2 for manipulation
sudo apt install ros-humble-moveit
sudo apt install ros-humble-moveit-visual-tools
sudo apt install ros-humble-moveit-resources
sudo apt install ros-humble-ros2-control
sudo apt install ros-humble-ros2-controllers
```

#### Kinematics Solvers
```bash
# KDL and other kinematics libraries
sudo apt install ros-humble-orocos-kdl
sudo apt install ros-humble-kdl-parser
sudo apt install ros-humble-tf2-kdl
```

#### GPU Optimization Tools
```bash
# NVIDIA optimization tools
sudo apt install nvidia-ml-py3
sudo apt install nvidia-nsight-systems
```

### Setup Verification
```bash
# Test MoveIt installation
ros2 launch moveit_resources demo.launch.py

# Test kinematics
ros2 run kdl_parser test_inertia_rpy

# Check GPU optimization
nvidia-smi dmon -s u -d 1
```

## Week 9: Isaac Perception System Integration

### Additional Software Requirements

#### Complete Isaac Stack
```bash
# Additional Isaac packages for integration
sudo apt install ros-humble-isaac-ros-gems
sudo apt install ros-humble-isaac-ros-buffers
sudo apt install ros-humble-isaac-ros-nitros
sudo apt install ros-humble-isaac-ros-color-correction
```

#### Integration Testing Tools
```bash
# Testing and validation
sudo apt install ros-humble-test-tools
sudo apt install ros-humble-launch-testing
sudo apt install ros-humble-ros-testing
```

### Setup Verification
```bash
# Test complete Isaac perception pipeline
ros2 launch isaac_ros_apriltag isaac_ros_apriltag.launch.py

# Verify integration
ros2 run isaac_ros_test isaac_ros_integration_test
```

## Week 10: Multimodal Embeddings and Instruction Following

### Software Requirements

#### Natural Language Processing
```bash
# Transformers and NLP libraries
pip install transformers
pip install tokenizers
pip install datasets
pip install evaluate
pip install accelerate
pip install torch

# NLTK and spaCy
pip install nltk
pip install spacy
python -m spacy download en_core_web_sm
```

#### Multimodal Libraries
```bash
# Multimodal processing
pip install clip @ https://github.com/openai/CLIP/archive/d50d76d.tar.gz
pip install sentence-transformers
pip install huggingface_hub
```

### Setup Verification
```bash
# Test NLP installation
python3 -c "import transformers; print(transformers.__version__)"

# Test CLIP installation
python3 -c "import clip; print('CLIP installed successfully')"
```

## Week 11: Embodied Language Models and Action Grounding

### Additional Software Requirements

#### Advanced NLP Tools
```bash
# Advanced language models
pip install openai
pip install langchain
pip install torch
pip install torchvision
pip install torchaudio
```

#### Vision-Language Integration
```bash
# Vision-language models
pip install lavis  # From salesforce-lavis
pip install pytorchvideo
```

### Setup Verification
```bash
# Test embodied language model
python3 -c "import torch; print('PyTorch ready for embodied models')"
```

## Week 12: Voice Command Interpretation and NLP Mapping

### Software Requirements

#### Speech Recognition
```bash
# Speech processing libraries
pip install speechrecognition
pip install pyaudio
pip install sounddevice
pip install webrtcvad

# Advanced ASR
pip install transformers
pip install datasets
pip install evaluate
```

#### Audio Processing
```bash
# Audio libraries
sudo apt install portaudio19-dev
sudo apt install python3-pyaudio
sudo apt install sox
sudo apt install libsox-dev
```

### Setup Verification
```bash
# Test audio input
python3 -c "import pyaudio; p = pyaudio.PyAudio(); print('Audio ready')"

# Test speech recognition
python3 -c "import speech_recognition as sr; print('Speech recognition ready')"
```

## Week 13: Capstone Integration and Evaluation

### Complete System Requirements

#### All Previous Tools
- All software from Weeks 1-12
- Complete ROS 2 workspace with all packages
- Full Isaac ROS installation
- All AI and ML frameworks

#### Evaluation Tools
```bash
# Performance evaluation
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn

# Reporting tools
pip install jupyter
pip install nbformat
```

### Hardware Requirements for Capstone
- **Full Robot Platform**: TurtleBot3 or equivalent
- **NVIDIA Jetson**: For edge deployment testing
- **Sensors**: Full sensor suite for perception
- **Network**: Stable connection for remote operations

## Alternative Setup Scenarios

### Cloud-Based Development
For institutions without sufficient local hardware:

#### AWS/GCP Setup
- **EC2 Instance**: p3.xlarge or p4d.24xlarge with GPU support
- **Docker**: Pre-configured containers with all dependencies
- **Remote Access**: VS Code Remote SSH or Jupyter Lab

#### Pre-built Containers
```dockerfile
FROM osrf/ros:humble-desktop-full
# Install all course dependencies
# Configure for course-specific requirements
```

### Minimal Hardware Setup
For budget-conscious implementations:

#### Simulation-Only Mode
- Focus on Gazebo simulation without physical hardware
- Use cloud GPUs for AI model training
- Emphasize software development skills

#### Component-Based Learning
- Rotate hardware access among student groups
- Use shared lab equipment
- Emphasize theory and simulation

## Troubleshooting Common Issues

### ROS 2 Installation Issues
```bash
# If ROS 2 packages are not found
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# If there are permission issues with Docker
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

### GPU Issues
```bash
# Check GPU status
nvidia-smi

# If CUDA is not working
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Network and Firewall Issues
```bash
# ROS 2 network configuration
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
```

## Maintenance and Updates

### Regular Maintenance Tasks
- **Weekly**: Update system packages
- **Monthly**: Update ROS 2 and Isaac packages
- **Quarterly**: Review and update curriculum tools

### Version Management
- Pin specific versions for course stability
- Test updates in development environment first
- Maintain backup configurations

## Institutional Setup Guide

### Lab Configuration
For setting up a robotics lab for this course:

#### Network Configuration
- VLAN for robot communication
- DHCP for automatic IP assignment
- Firewall rules for ROS 2 communication

#### Hardware Setup
- Workstations with appropriate GPUs
- Robot charging stations
- Network switches for robot connectivity
- Security cameras for monitoring

#### Software Management
- Centralized package management
- Student workspace templates
- Automated backup systems
- Access control for different user levels

## Next Steps

Continue with [Lab and Assignment Descriptions](./labs-assignments.md) to explore detailed hands-on activities and projects for each week of the course.

## References

[All sources will be cited in the References section at the end of the book, following APA format]