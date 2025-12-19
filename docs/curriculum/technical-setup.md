---
sidebar_position: 13
---

# Technical Setup Guide: Physical AI & Humanoid Robotics Course

## Overview

This technical setup guide provides comprehensive instructions for configuring all hardware and software components required for the Physical AI & Humanoid Robotics course. The guide covers system requirements, installation procedures, configuration steps, and troubleshooting procedures to ensure a smooth and consistent experience for both instructors and students.

## System Requirements

### Minimum Hardware Requirements

#### Student Workstation
- **Processor**: Intel i5 or equivalent AMD processor (4+ cores)
- **Memory**: 16GB RAM (32GB recommended)
- **Storage**: 500GB SSD (1TB recommended)
- **Graphics**: NVIDIA GPU with CUDA support (RTX 3060 or equivalent)
- **Network**: Gigabit Ethernet, WiFi 802.11ac
- **Operating System**: Ubuntu 20.04 LTS or Ubuntu 22.04 LTS

#### Recommended Hardware Specifications
- **Processor**: Intel i7/i9 or equivalent AMD Ryzen processor (8+ cores)
- **Memory**: 32GB RAM (64GB for advanced projects)
- **Storage**: 1TB NVMe SSD
- **Graphics**: NVIDIA RTX 3080/4080 or RTX A4000 (12GB+ VRAM)
- **Network**: Gigabit Ethernet preferred
- **Display**: 24" 1080p monitor (4K preferred)

### Robot Hardware Requirements

#### Primary Robot Platform (TurtleBot3 Compatible)
- **Chassis**: Differential drive mobile robot
- **Processor**: ARM-based single-board computer (Raspberry Pi 4 or Jetson Nano)
- **Sensors**:
  - 360° LiDAR (2D mapping)
  - RGB-D camera (3D perception)
  - IMU (orientation and motion)
  - Wheel encoders (odometry)
- **Connectivity**: WiFi/BT for ROS 2 communication
- **Battery**: 2+ hours operation time
- **Payload**: 1kg minimum capacity

#### Alternative Platforms
- **Simulation-Only Setup**: For students without hardware access
- **Custom Platforms**: Support for various robot configurations
- **Cloud Robotics**: Remote access to robot platforms

### Lab Infrastructure Requirements

#### Network Infrastructure
- **Router**: Enterprise-grade with QoS support
- **Switches**: Managed gigabit switches
- **Access Points**: Enterprise WiFi with multiple SSIDs
- **Bandwidth**: 100Mbps+ internet connection
- **Security**: WPA3 encryption, guest network isolation

#### Physical Infrastructure
- **Lab Tables**: Adjustable height, cable management
- **Power**: Multiple outlets per station, UPS backup
- **Lighting**: LED with adjustable intensity
- **Ventilation**: Adequate for electronics operation
- **Safety**: Fire suppression, first aid stations

## Software Installation Guide

### Ubuntu System Setup

#### Fresh Ubuntu Installation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install basic development tools
sudo apt install build-essential cmake git wget curl unzip -y

# Install Python development tools
sudo apt install python3-dev python3-pip python3-setuptools python3-wheel -y

# Install essential utilities
sudo apt install htop tmux vim nano tree -y

# Install network tools
sudo apt install net-tools nmap iperf3 -y

# Install multimedia tools
sudo apt install vlc ffmpeg gimp inkscape -y
```

#### ROS 2 Humble Installation
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install -y software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop-full -y

# Install ROS 2 development tools
sudo apt install python3-colcon-common-extensions -y
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2 environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### Verification
```bash
# Verify ROS 2 installation
ros2 --version

# Test basic functionality
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_py listener

# Check available packages
ros2 pkg list | head -20
```

### NVIDIA GPU Setup

#### Driver Installation
```bash
# Check current driver
nvidia-smi

# If no driver or old version, install latest:
sudo apt install nvidia-driver-535 -y

# Reboot system
sudo reboot
```

#### CUDA Toolkit Installation
```bash
# Download CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt -y install cuda-toolkit-11-8

# Add to environment
echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

#### Verification
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi
```

### Isaac ROS Installation

#### Install Isaac ROS Packages
```bash
# Install Isaac ROS common packages
sudo apt install ros-humble-isaac-ros-common -y

# Install Isaac ROS perception packages
sudo apt install ros-humble-isaac-ros-dnn-inference -y
sudo apt install ros-humble-isaac-ros-image-pipeline -y
sudo apt install ros-humble-isaac-ros-visual-slam -y
sudo apt install ros-humble-isaac-ros-apriltag -y
sudo apt install ros-humble-isaac-ros-people-segmentation -y
sudo apt install ros-humble-isaac-ros-segmentation -y
sudo apt install ros-humble-isaac-ros-gems -y
sudo apt install ros-humble-isaac-ros-buffers -y
sudo apt install ros-humble-isaac-ros-nitros -y

# Install Isaac ROS tools
sudo apt install ros-humble-isaac-ros-dev-tools -y
```

#### Verification
```bash
# Verify Isaac ROS installation
ros2 run isaac_ros_image_proc image_format_converter --ros-args --help

# Test Isaac launch
ros2 launch isaac_ros_apriltag isaac_ros_apriltag.launch.py
```

### Gazebo Installation

#### Gazebo Garden Installation
```bash
# Add Gazebo repository
sudo curl -fsSL https://get.garden.gazebosim.org | sudo sh -

# Install Gazebo Garden
sudo apt update
sudo apt install gz-garden -y

# Install ROS 2 Gazebo packages
sudo apt install ros-humble-gazebo-ros ros-humble-gazebo-plugins ros-humble-gazebo-dev -y
sudo apt install ros-humble-gazebo-msgs ros-humble-gazebo-ros-pkgs -y
```

#### Verification
```bash
# Test Gazebo
gz sim

# Test ROS 2 integration
ros2 launch gazebo_ros gazebo.launch.py
```

### Development Environment Setup

#### Visual Studio Code Installation
```bash
# Download VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg

sudo apt update
sudo apt install apt-transport-https code -y

# Install ROS extension
code --install-extension ms-iot.vscode-ros
code --install-extension ms-python.python
code --install-extension ms-vscode.cpptools
code --install-extension twxs.cmake
```

#### Python Development Environment
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Initialize conda
conda init bash
source ~/.bashrc

# Create robotics environment
conda create -n robotics python=3.10 -y
conda activate robotics

# Install essential Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install opencv-python
pip install numpy scipy matplotlib
pip install pandas jupyter
pip install pyyaml
pip install black flake8
pip install pytest pytest-cov
```

#### Docker and NVIDIA Container Toolkit
```bash
# Install Docker
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release -y

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Course-Specific Package Installation

#### Robotics Libraries
```bash
# Install navigation stack
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup -y
sudo apt install ros-humble-dwb-core ros-humble-robot-localization -y
sudo apt install ros-humble-slam-toolbox ros-humble-interactive-markers -y

# Install manipulation packages
sudo apt install ros-humble-moveit ros-humble-moveit-visual-tools -y
sudo apt install ros-humble-moveit-resources ros-humble-ros2-control -y
sudo apt install ros-humble-ros2-controllers -y

# Install perception packages
sudo apt install ros-humble-vision-opencv ros-humble-cv-bridge -y
sudo apt install ros-humble-image-transport ros-humble-compressed-image-transport -y
sudo apt install ros-humble-depth-image-proc ros-humble-image-proc -y

# Install simulation packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins -y
sudo apt install ros-humble-ros-gz ros-humble-ros-gz-interfaces -y
```

#### AI and Machine Learning Libraries
```bash
# Activate conda environment
conda activate robotics

# Install additional ML packages
pip install tensorrt
pip install onnx onnxruntime-gpu
pip install tensorflow
pip install torchmetrics
pip install transformers
pip install sentence-transformers
pip install clip @ https://github.com/openai/CLIP/archive/d50d76d.tar.gz
pip install lavis  # For vision-language models
```

### Workspace Setup

#### Create ROS 2 Workspace
```bash
# Create workspace
mkdir -p ~/robotics_ws/src
cd ~/robotics_ws

# Build workspace
colcon build --symlink-install

# Source workspace
echo "source ~/robotics_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### Course Package Structure
```bash
# Create course packages
cd ~/robotics_ws/src

# Create basic course package
ros2 pkg create --build-type ament_python robot_course_pkg

# Create perception package
ros2 pkg create --build-type ament_python perception_course_pkg

# Create navigation package
ros2 pkg create --build-type ament_python navigation_course_pkg

# Create vla package
ros2 pkg create --build-type ament_python vla_course_pkg

# Build packages
cd ~/robotics_ws
colcon build --packages-select robot_course_pkg perception_course_pkg navigation_course_pkg vla_course_pkg
```

## Hardware Setup Guide

### Robot Platform Assembly

#### TurtleBot3 Setup
```bash
# Install TurtleBot3 packages
sudo apt install ros-humble-turtlebot3 ros-humble-turtlebot3-msgs ros-humble-turtlebot3-simulations -y

# Set environment variables
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc  # or waffle
source ~/.bashrc
```

#### Custom Robot Setup
For custom robots, create URDF models and launch files:

```xml
<!-- Create robot.urdf.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="custom_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

### Sensor Integration

#### Camera Setup
```bash
# Install camera drivers
sudo apt install ros-humble-camera-calibration ros-humble-image-view -y
sudo apt install ros-humble-usb-cam ros-humble-v4l2-camera -y

# Test camera
ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video0
```

#### LiDAR Setup
```bash
# Install LiDAR drivers
sudo apt install ros-humble-ydlidar-ros2-driver ros-humble-rplidar-ros -y

# Test LiDAR
ros2 launch ydlidar_ros2_driver ydlidar_ros2_driver.launch.py
```

### Network Configuration

#### Robot Communication Setup
```bash
# Configure network for robot communication
# Edit /etc/hosts to add robot IP aliases
echo "192.168.1.100 turtlebot3" | sudo tee -a /etc/hosts

# Set ROS 2 environment for multi-robot
echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
```

## Testing and Verification

### Basic System Tests

#### ROS 2 Basic Test
```bash
# Test ROS 2 communication
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_cpp listener &

# Test service communication
ros2 run demo_nodes_cpp add_two_ints_server &
ros2 run demo_nodes_cpp add_two_ints_client &

# Check topics
ros2 topic list
ros2 service list
```

#### Isaac ROS Test
```bash
# Test Isaac ROS image processing
ros2 run isaac_ros_image_proc image_format_converter --ros-args --help

# Test Isaac perception
ros2 launch isaac_ros_apriltag isaac_ros_apriltag.launch.py
```

#### Gazebo Test
```bash
# Launch Gazebo with TurtleBot3
ros2 launch turtlebot3_gazebo empty_world.launch.py

# Launch navigation in simulation
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Performance Tests

#### GPU Performance Test
```bash
# Test CUDA performance
nvidia-smi

# Test PyTorch GPU
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"

# Test Isaac ROS GPU acceleration
ros2 run isaac_ros_image_proc image_format_converter --ros-args -p encoding_desired:=rgb8
```

#### System Performance Test
```bash
# Monitor system resources
htop

# Test network performance
iperf3 -s  # On server
iperf3 -c <server_ip>  # On client

# Test ROS 2 performance
ros2 topic hz /camera/image_raw
```

## Troubleshooting Guide

### Common Installation Issues

#### ROS 2 Installation Issues
**Problem**: Package not found
```bash
# Solution: Update package lists
sudo apt update
sudo apt upgrade

# Check ROS 2 repository configuration
apt policy ros-humble-desktop-full
```

**Problem**: Permission denied
```bash
# Solution: Check user permissions
sudo usermod -a -G dialout $USER
sudo usermod -a -G docker $USER
# Log out and log back in
```

#### GPU Issues
**Problem**: CUDA not recognized
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall CUDA if needed
sudo apt remove --purge nvidia-*
sudo apt autoremove
sudo apt autoclean
# Then reinstall from scratch
```

**Problem**: Isaac ROS packages not working
```bash
# Check NVIDIA driver and CUDA
nvidia-smi
nvcc --version

# Check Isaac ROS installation
apt policy ros-humble-isaac-ros-common

# Verify GPU access
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Robot Communication Issues

#### Network Problems
**Problem**: Cannot connect to robot
```bash
# Check network connectivity
ping <robot_ip>

# Check ROS 2 domain
echo $ROS_DOMAIN_ID

# Check firewall
sudo ufw status
```

#### Sensor Issues
**Problem**: Camera not detected
```bash
# Check device
ls /dev/video*

# Test with v4l2
v4l2-ctl --list-devices

# Check permissions
sudo chmod 666 /dev/video0
```

**Problem**: LiDAR not working
```bash
# Check device
ls /dev/ttyUSB*

# Check permissions
sudo chmod 666 /dev/ttyUSB0

# Test serial communication
sudo apt install screen
screen /dev/ttyUSB0 115200
```

### Performance Issues

#### Slow Simulation
**Problem**: Gazebo running slowly
```bash
# Check GPU acceleration
nvidia-smi

# Reduce simulation complexity
# Use simpler robot models
# Reduce physics update rate

# Monitor system resources
htop
```

#### High CPU Usage
**Problem**: ROS 2 nodes consuming too much CPU
```bash
# Monitor CPU usage
htop

# Check node performance
ros2 run topicos topico

# Optimize code
# Reduce update rates
# Use efficient algorithms
```

### Software Compatibility Issues

#### Version Conflicts
**Problem**: Package version conflicts
```bash
# Check versions
ros2 --version
python3 --version
gcc --version

# Use virtual environments
conda create -n robotics_env python=3.10
conda activate robotics_env
```

#### Dependency Issues
**Problem**: Missing dependencies
```bash
# Check dependencies
ldd /path/to/binary

# Install missing packages
sudo apt install <missing_package>

# Use conda for Python dependencies
conda install <package_name>
```

## Maintenance and Updates

### Regular Maintenance

#### System Updates
```bash
# Weekly system maintenance
sudo apt update && sudo apt upgrade -y
sudo apt autoremove && sudo apt autoclean

# Monthly ROS 2 workspace cleanup
cd ~/robotics_ws
rm -rf build/ install/ log/
colcon build --symlink-install
```

#### Package Updates
```bash
# Update ROS 2 packages
sudo apt update
sudo apt list --upgradable
sudo apt upgrade ros-humble-*

# Update Python packages
pip list --outdated
pip install --upgrade <package_name>

# Update conda environment
conda update --all
```

### Backup and Recovery

#### System Backup
```bash
# Create system backup
sudo tar -czf ~/backup_$(date +%Y%m%d).tar.gz \
  --exclude=/home/*/.cache \
  --exclude=/var/cache/apt \
  --exclude=/tmp \
  /

# Backup ROS 2 workspace
cd ~
tar -czf robotics_ws_backup_$(date +%Y%m%d).tar.gz robotics_ws/
```

#### Workspace Recovery
```bash
# Restore workspace
cd ~
tar -xzf robotics_ws_backup_$(date +%Y%m%d).tar.gz

# Rebuild workspace
cd ~/robotics_ws
colcon build --symlink-install
source install/setup.bash
```

## Lab Setup Guide

### Classroom Configuration

#### Station Setup
- **Computer**: Meet minimum requirements above
- **Monitor**: 24" 1080p or better
- **Keyboard/Mouse**: Ergonomic, wireless preferred
- **Robot Station**: Clear space for robot operation
- **Networking**: Ethernet cable for reliable connection
- **Power**: Surge protector for all equipment

#### Safety Setup
- **Emergency Procedures**: Posted instructions
- **First Aid Kit**: Easily accessible
- **Fire Extinguisher**: Appropriate type for electronics
- **Emergency Contacts**: Posted phone numbers
- **Cable Management**: Organized, safe routing

### Equipment Management

#### Inventory Tracking
- **Serial Numbers**: Record all equipment
- **Condition Reports**: Regular equipment checks
- **Maintenance Schedule**: Preventive maintenance calendar
- **Replacement Parts**: Stock common components
- **Warranty Information**: Track warranty periods

#### Checkout Procedures
- **Student Accounts**: Individual equipment checkout
- **Damage Reports**: Document equipment condition
- **Due Dates**: Clear return deadlines
- **Late Fees**: Consequences for overdue equipment
- **Replacement Costs**: Fee structure for lost items

## Quality Assurance

### Pre-Use Checks

#### Daily Checks
- [ ] **Power**: All equipment powers on
- [ ] **Network**: Network connectivity verified
- [ ] **Software**: All required software loads
- [ ] **Sensors**: All sensors detect properly
- [ ] **Safety**: Emergency procedures accessible

#### Weekly Checks
- [ ] **System Updates**: Apply security patches
- [ ] **Performance**: Run performance tests
- [ ] **Backups**: Verify backup integrity
- [ ] **Cleaning**: Clean equipment and workspace
- [ ] **Inventory**: Verify equipment counts

### Performance Monitoring

#### System Monitoring
```bash
# Monitor system performance
htop  # CPU and memory usage
nvidia-smi  # GPU usage
iotop  # Disk I/O
iftop  # Network traffic

# Monitor ROS 2 performance
ros2 run topicos topico
ros2 topic hz /topic_name
```

#### Usage Statistics
```bash
# Track system usage
# Log file sizes and growth
# Monitor resource consumption
# Track error rates and patterns
# Record performance metrics
```

## Security Considerations

### Network Security
- **Firewall**: Configure for robotics applications
- **Encryption**: Secure robot communications
- **Access Control**: Limit network access
- **Monitoring**: Track network activity
- **Updates**: Apply security patches

### Data Security
- **Privacy**: Protect student data
- **Encryption**: Secure sensitive information
- **Access**: Limit data access rights
- **Backup**: Secure backup procedures
- **Disposal**: Secure data destruction

## Support Resources

### Documentation
- **User Manuals**: Hardware and software guides
- **Quick Start Guides**: Step-by-step instructions
- **Troubleshooting**: Common problem solutions
- **FAQ**: Frequently asked questions
- **Video Tutorials**: Screencast demonstrations

### Help Desk
- **Hours**: Regular support hours
- **Contact**: Multiple contact methods
- **Escalation**: Problem escalation procedures
- **Training**: Support staff training
- **Metrics**: Track support effectiveness

## International Considerations

### Regional Requirements
- **Standards**: Local electrical and safety standards
- **Regulations**: Import/export regulations
- **Languages**: Multi-language support
- **Currencies**: Pricing in local currencies
- **Support**: Local support availability

### Cultural Adaptations
- **Interfaces**: Cultural sensitivity in UI
- **Examples**: Culturally appropriate examples
- **Standards**: Local measurement systems
- **Colors**: Culturally appropriate color schemes
- **Symbols**: Culturally appropriate symbols

## Conclusion

This technical setup guide provides comprehensive instructions for configuring the Physical AI & Humanoid Robotics course environment. Following these procedures will ensure a consistent, reliable, and secure environment for both instructors and students.

Regular maintenance, monitoring, and updates are essential for keeping the system running smoothly. The troubleshooting guide provides solutions for common issues, and the quality assurance procedures help maintain high standards.

For ongoing support and updates, maintain contact with the course development team and participate in the community of educators using this curriculum.

## Appendices

### Appendix A: Installation Scripts
Complete installation scripts for automated setup

### Appendix B: Hardware Specifications
Detailed specifications for all recommended hardware

### Appendix C: Software Licenses
License information for all required software

### Appendix D: Network Diagrams
Network topology diagrams for lab setup

## Next Steps

After completing the technical setup, proceed with [Testing and Validation Guide](./testing-validation.md) to verify that all components are properly configured and functioning as expected.

## References

[All sources will be cited in the References section at the end of the book, following APA format]