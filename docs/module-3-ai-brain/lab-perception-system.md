---
sidebar_position: 8
---

# Lab: Isaac Perception System

## Learning Objectives

By completing this lab, you will be able to:

1. Implement a complete GPU-accelerated perception pipeline using Isaac ROS
2. Integrate multiple perception modules (detection, segmentation, tracking)
3. Optimize the perception system for real-time performance
4. Validate perception outputs against ground truth data
5. Deploy the perception system on edge hardware

## Lab Overview

In this hands-on lab, you will build a complete perception system using NVIDIA Isaac ROS packages. The system will process camera images to detect objects, segment the scene, and track objects across frames. You'll optimize the system for real-time performance on edge hardware and validate its accuracy.

### Prerequisites

Before starting this lab, ensure you have:

- NVIDIA GPU with CUDA support (RTX series recommended)
- JetPack 5.0+ installed on Jetson platform (if deploying on edge)
- ROS 2 Humble Hawksbill installed
- Isaac ROS packages installed
- Docker and NVIDIA Container Toolkit configured

### Lab Duration

This lab should take approximately 4-6 hours to complete, depending on your familiarity with ROS 2 and Isaac.

## Setting Up the Development Environment

### Option 1: Using Isaac ROS Docker Containers (Recommended)

First, pull the necessary Isaac ROS containers:

```bash
# Pull Isaac ROS base container
docker pull nvcr.io/nvidia/isaac_ros:galactic-ros-dev

# Pull specific perception packages
docker pull nvcr.io/nvidia/isaac_ros:isaac_ros_detectnet
docker pull nvcr.io/nvidia/isaac_ros:isaac_ros_segmentation
docker pull nvcr.io/nvidia/isaac_ros:isaac_ros_image_pipeline
```

Create a Docker container with GPU support:

```bash
# Run Isaac ROS development container
docker run --gpus all \
    --rm -it \
    --network host \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --env DISPLAY=$DISPLAY \
    --env TERM=xterm-256color \
    --env QT_X11_NO_MITSHM=1 \
    --privileged \
    --name isaac_perception_lab \
    nvcr.io/nvidia/isaac_ros:galactic-ros-dev

# Inside the container, set up your workspace
mkdir -p ~/isaac_ws/src
cd ~/isaac_ws
colcon build --symlink-install
source install/setup.bash
```

### Option 2: Native Installation

For native installation on Ubuntu:

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install -y cuda-toolkit-11-8

# Install Isaac ROS packages
sudo apt-get install -y ros-galactic-isaac-ros-common
sudo apt-get install -y ros-galactic-isaac-ros-dnn-inference
sudo apt-get install -y ros-galactic-isaac-ros-image-pipeline
```

## Creating the Perception System Package

Let's create a new ROS 2 package for our perception system:

```bash
cd ~/isaac_ws/src
ros2 pkg create --build-type ament_python isaac_perception_system
cd isaac_perception_system
```

Create the main perception node:

```python
# File: isaac_perception_system/isaac_perception_system/perception_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import cv2
import numpy as np
from cv_bridge import CvBridge
import message_filters

class IsaacPerceptionSystem(Node):
    def __init__(self):
        super().__init__('isaac_perception_system')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.detection_publisher = self.create_publisher(
            Detection2DArray, 'perception/detections', 10
        )

        self.segmentation_publisher = self.create_publisher(
            Image, 'perception/segmentation', 10
        )

        # Subscribers using message filters for synchronization
        self.image_sub = message_filters.Subscriber(
            self, Image, 'camera/image_raw'
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, 'camera/camera_info'
        )

        # Synchronize image and camera info
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.process_synchronized_data)

        # Performance tracking
        self.frame_count = 0
        self.start_time = self.get_clock().now()

        self.get_logger().info('Isaac Perception System initialized')

    def process_synchronized_data(self, image_msg, info_msg):
        """Process synchronized image and camera info data"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Process image through perception pipeline
            detections, segmentation = self.run_perception_pipeline(cv_image)

            # Publish results
            if detections is not None:
                self.publish_detections(detections, image_msg.header)

            if segmentation is not None:
                self.publish_segmentation(segmentation, image_msg.header)

            # Track performance
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # Log every 30 frames
                current_time = self.get_clock().now()
                elapsed = (current_time - self.start_time).nanoseconds / 1e9
                fps = self.frame_count / elapsed
                self.get_logger().info(f'Processing at {fps:.2f} FPS')

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

    def run_perception_pipeline(self, image):
        """Run the complete perception pipeline"""
        # This is a simplified version - in practice, you'd use Isaac ROS packages
        # For this lab, we'll simulate the perception pipeline

        # 1. Object Detection (simulated)
        detections = self.simulate_object_detection(image)

        # 2. Semantic Segmentation (simulated)
        segmentation = self.simulate_segmentation(image)

        return detections, segmentation

    def simulate_object_detection(self, image):
        """Simulate object detection using Isaac-compatible format"""
        # In a real implementation, this would use Isaac ROS detectnet
        # For simulation, we'll create some sample detections
        height, width = image.shape[:2]

        # Create sample detections (in practice, this comes from neural network)
        sample_detections = [
            {
                'class_name': 'person',
                'confidence': 0.85,
                'bbox': [width//4, height//4, width//2, height//2],  # [x, y, w, h]
                'center': [width//2, height//2]
            },
            {
                'class_name': 'car',
                'confidence': 0.78,
                'bbox': [width//3, height//3, width//3, height//3],
                'center': [width//2, 2*height//3]
            }
        ]

        return sample_detections

    def simulate_segmentation(self, image):
        """Simulate semantic segmentation"""
        # In a real implementation, this would use Isaac ROS segmentation
        # For simulation, we'll create a simple segmentation mask
        height, width = image.shape[:2]

        # Create a simple segmentation mask
        segmentation_mask = np.zeros((height, width), dtype=np.uint8)

        # Add some regions (simulated segmentation)
        cv2.rectangle(segmentation_mask, (width//4, height//4), (3*width//4, 3*height//4), 1, -1)  # Person region
        cv2.circle(segmentation_mask, (width//2, 3*height//4), width//6, 2, -1)  # Car region

        return segmentation_mask

    def publish_detections(self, detections, header):
        """Publish detections in vision_msgs format"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for det in detections:
            detection = Detection2D()

            # Set bounding box
            bbox = detection.bbox
            bbox.center.x = det['center'][0]
            bbox.center.y = det['center'][1]
            bbox.size_x = det['bbox'][2]
            bbox.size_y = det['bbox'][3]

            # Set ID and confidence
            detection.results = []  # This would be populated with actual detection results

            detection_array.detections.append(detection)

        self.detection_publisher.publish(detection_array)

    def publish_segmentation(self, segmentation_mask, header):
        """Publish segmentation result"""
        segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation_mask, encoding='mono8')
        segmentation_msg.header = header
        self.segmentation_publisher.publish(segmentation_msg)

def main(args=None):
    rclpy.init(args=args)

    perception_node = IsaacPerceptionSystem()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down perception system')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Now let's create the actual Isaac ROS implementation:

```python
# File: isaac_perception_system/isaac_perception_system/isaac_perception_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import cv2
import numpy as np
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class IsaacPerceptionNode(Node):
    def __init__(self):
        super().__init__('isaac_perception_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.detection_publisher = self.create_publisher(
            Detection2DArray, 'isaac_perception/detections', 10
        )

        # Set up QoS for camera data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.image_sub = message_filters.Subscriber(
            self, Image, 'camera/image_rect_color', qos_profile=qos_profile
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, 'camera/camera_info', qos_profile=qos_profile
        )

        # Synchronize image and camera info
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub], queue_size=5, slop=0.1
        )
        self.sync.registerCallback(self.process_perception_data)

        # Performance tracking
        self.frame_count = 0
        self.start_time = self.get_clock().now()

        # Isaac-specific parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', '/models/detectnet/resnet18_detector.trt'),
                ('input_width', 224),
                ('input_height', 224),
                ('confidence_threshold', 0.5),
                ('max_objects', 10)
            ]
        )

        self.get_logger().info('Isaac Perception Node initialized')

    def process_perception_data(self, image_msg, info_msg):
        """Process perception data using Isaac ROS packages"""
        try:
            # In a real Isaac implementation, this would connect to Isaac ROS
            # perception packages like detectnet, segmentation, etc.
            # For this lab, we'll simulate the Isaac pipeline

            # Track performance
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                current_time = self.get_clock().now()
                elapsed = (current_time - self.start_time).nanoseconds / 1e9
                fps = self.frame_count / elapsed
                self.get_logger().info(f'Isaac perception running at {fps:.2f} FPS')

            # For simulation purposes, create sample detections
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            detections = self.create_sample_detections(cv_image, image_msg.header)

            # Publish results
            self.detection_publisher.publish(detections)

        except Exception as e:
            self.get_logger().error(f'Error in perception pipeline: {e}')

    def create_sample_detections(self, image, header):
        """Create sample detections for demonstration"""
        height, width = image.shape[:2]
        detection_array = Detection2DArray()
        detection_array.header = header

        # Create sample detections
        for i in range(3):  # Create 3 sample detections
            detection = Detection2D()

            # Set random bounding box
            x = np.random.randint(0, width // 2)
            y = np.random.randint(0, height // 2)
            w = np.random.randint(width // 4, width // 2)
            h = np.random.randint(height // 4, height // 2)

            detection.bbox.center.x = x + w // 2
            detection.bbox.center.y = y + h // 2
            detection.bbox.size_x = w
            detection.bbox.size_y = h

            # Add detection result
            from vision_msgs.msg import ObjectHypothesisWithPose
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = f'object_{i}'
            hypothesis.hypothesis.score = 0.8 + np.random.random() * 0.2  # 0.8-1.0
            detection.results.append(hypothesis)

            detection_array.detections.append(detection)

        return detection_array

def main(args=None):
    rclpy.init(args=args)

    perception_node = IsaacPerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down Isaac perception node')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating the Launch File

Create a launch file to start the perception system with Isaac ROS packages:

```xml
<!-- File: isaac_perception_system/launch/perception_system.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/models/detectnet/resnet18_detector.trt',
        description='Path to the TensorRT model'
    )

    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Confidence threshold for detections'
    )

    # Get launch configurations
    model_path = LaunchConfiguration('model_path')
    confidence_threshold = LaunchConfiguration('confidence_threshold')

    # Create composable node container
    perception_container = ComposableNodeContainer(
        name='perception_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # Image format conversion for Isaac
            ComposableNode(
                package='isaac_ros_image_proc',
                plugin='isaac_ros::ImageFormatConverterNode',
                name='image_format_converter',
                parameters=[{
                    'encoding_desired': 'rgb8',
                    'image_width': 640,
                    'image_height': 480
                }],
                remappings=[
                    ('image_raw', 'camera/image_raw'),
                    ('image', 'camera/image_rect_color')
                ]
            ),
            # Isaac DetectNet for object detection
            ComposableNode(
                package='isaac_ros_detectnet',
                plugin='nvidia::isaac_ros::dnn_inference::ImageEncoderNode',
                name='detectnet_encoder',
                parameters=[{
                    'model_path': model_path,
                    'input_tensor_names': ['input'],
                    'output_tensor_names': ['output'],
                    'model_input_width': 224,
                    'model_input_height': 224,
                    'model_input_channel': 3,
                    'confidence_threshold': confidence_threshold
                }],
                remappings=[
                    ('encoded_tensor', 'detectnet/encoded_tensor'),
                    ('image', 'camera/image_rect_color')
                ]
            ),
            # Custom perception processing node
            ComposableNode(
                package='isaac_perception_system',
                plugin='isaac_perception_system.IsaacPerceptionNode',
                name='isaac_perception_node',
                parameters=[{
                    'model_path': model_path,
                    'confidence_threshold': confidence_threshold
                }],
                remappings=[
                    ('camera/image_rect_color', 'camera/image_rect_color'),
                    ('camera/camera_info', 'camera/camera_info'),
                    ('isaac_perception/detections', 'perception/detections')
                ]
            )
        ],
        output='screen'
    )

    return LaunchDescription([
        model_path_arg,
        confidence_threshold_arg,
        perception_container
    ])
```

## Creating the Isaac Perception Pipeline

Now let's create a more complete implementation that integrates Isaac ROS packages:

```python
# File: isaac_perception_system/isaac_perception_system/complete_perception_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from std_msgs.msg import Header, Float32
from builtin_interfaces.msg import Duration
import cv2
import numpy as np
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import deque
import time

class CompleteIsaacPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('complete_isaac_perception_pipeline')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.detection_publisher = self.create_publisher(
            Detection2DArray, 'isaac_perception/detections', 10
        )

        self.performance_publisher = self.create_publisher(
            Float32, 'isaac_perception/performance', 10
        )

        # Set up QoS for camera data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers with synchronization
        self.image_sub = message_filters.Subscriber(
            self, Image, 'camera/image_rect_color', qos_profile=qos_profile
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, 'camera/camera_info', qos_profile=qos_profile
        )

        # Synchronize image and camera info
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub], queue_size=5, slop=0.1
        )
        self.sync.registerCallback(self.process_perception_pipeline)

        # Performance tracking
        self.frame_times = deque(maxlen=30)  # Track last 30 frame times
        self.frame_count = 0
        self.last_performance_report = time.time()

        # Perception parameters
        self.confidence_threshold = 0.5
        self.max_objects = 20
        self.iou_threshold = 0.3  # Intersection over Union threshold for NMS

        # Object tracking (simple implementation for this lab)
        self.tracked_objects = {}
        self.next_object_id = 0

        # Isaac-specific parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', '/models/detectnet/resnet18_detector.trt'),
                ('input_width', 224),
                ('input_height', 224),
                ('confidence_threshold', 0.5),
                ('max_objects', 20),
                ('iou_threshold', 0.3),
                ('enable_tracking', True),
                ('tracking_max_age', 10),
                ('tracking_min_hits', 3)
            ]
        )

        # Get parameters
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.max_objects = self.get_parameter('max_objects').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.enable_tracking = self.get_parameter('enable_tracking').value

        self.get_logger().info('Complete Isaac Perception Pipeline initialized')

    def process_perception_pipeline(self, image_msg, info_msg):
        """Process the complete perception pipeline"""
        start_time = time.time()

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Run Isaac-based perception (simulated in this lab)
            detections = self.run_isaac_perception(cv_image)

            # Apply non-maximum suppression to remove duplicate detections
            detections = self.non_maximum_suppression(detections, self.iou_threshold)

            # Apply object tracking if enabled
            if self.enable_tracking:
                detections = self.update_object_tracking(detections)

            # Publish results
            self.publish_detections(detections, image_msg.header)

            # Track performance
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)

            # Calculate and publish performance metrics
            self.publish_performance_metrics()

        except Exception as e:
            self.get_logger().error(f'Error in perception pipeline: {e}')

    def run_isaac_perception(self, image):
        """Simulate Isaac ROS perception pipeline"""
        # In a real implementation, this would interface with Isaac ROS packages
        # For this lab, we'll simulate the perception with realistic timing

        height, width = image.shape[:2]

        # Simulate realistic detection times
        time.sleep(0.02)  # Simulate 20ms processing time

        # Create realistic detections
        detections = []

        # Add some random detections based on image content
        num_detections = np.random.poisson(2)  # Average of 2 detections per frame

        for i in range(min(num_detections, self.max_objects)):
            # Random bounding box
            x = np.random.randint(0, width // 2)
            y = np.random.randint(0, height // 2)
            w = np.random.randint(width // 8, width // 3)
            h = np.random.randint(height // 8, height // 3)

            # Ensure bounding box is within image bounds
            x = min(x, width - w)
            y = min(y, height - h)

            # Random confidence (above threshold)
            confidence = self.confidence_threshold + np.random.random() * (1.0 - self.confidence_threshold)

            detection = {
                'bbox': [x, y, w, h],
                'confidence': confidence,
                'class_id': np.random.choice(['person', 'car', 'bicycle', 'traffic_sign']),
                'center': [x + w//2, y + h//2]
            }

            detections.append(detection)

        return detections

    def non_maximum_suppression(self, detections, iou_threshold):
        """Apply non-maximum suppression to remove overlapping detections"""
        if not detections:
            return detections

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        # Apply NMS
        keep = []
        for detection in detections:
            overlap = False
            for kept in keep:
                # Calculate IoU
                iou = self.calculate_iou(detection['bbox'], kept['bbox'])
                if iou > iou_threshold:
                    overlap = True
                    break

            if not overlap:
                keep.append(detection)

        return keep

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update_object_tracking(self, detections):
        """Simple object tracking to maintain consistent IDs across frames"""
        # This is a simplified tracking implementation
        # In a real system, you'd use more sophisticated tracking like SORT or Deep SORT

        updated_detections = []

        for detection in detections:
            # Find the closest existing tracked object
            best_match = None
            best_distance = float('inf')

            for obj_id, obj_info in self.tracked_objects.items():
                # Calculate distance to last known position
                dist = np.sqrt(
                    (detection['center'][0] - obj_info['center'][0])**2 +
                    (detection['center'][1] - obj_info['center'][1])**2
                )

                if dist < best_distance and dist < 50:  # 50 pixel threshold
                    best_distance = dist
                    best_match = obj_id

            if best_match is not None:
                # Update existing object
                self.tracked_objects[best_match]['center'] = detection['center']
                self.tracked_objects[best_match]['bbox'] = detection['bbox']
                self.tracked_objects[best_match]['confidence'] = detection['confidence']
                self.tracked_objects[best_match]['last_seen'] = time.time()

                # Add tracking ID to detection
                detection['track_id'] = best_match
            else:
                # Create new tracked object
                new_id = self.next_object_id
                self.tracked_objects[new_id] = {
                    'center': detection['center'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'last_seen': time.time(),
                    'class_id': detection['class_id']
                }
                detection['track_id'] = new_id
                self.next_object_id += 1

            updated_detections.append(detection)

        # Remove old tracked objects that haven't been seen recently
        current_time = time.time()
        objects_to_remove = []
        for obj_id, obj_info in self.tracked_objects.items():
            if current_time - obj_info['last_seen'] > 1.0:  # 1 second timeout
                objects_to_remove.append(obj_id)

        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]

        return updated_detections

    def publish_detections(self, detections, header):
        """Publish detections in vision_msgs format"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for det in detections:
            detection_2d = Detection2D()

            # Set bounding box
            detection_2d.bbox.center.x = det['center'][0]
            detection_2d.bbox.center.y = det['center'][1]
            detection_2d.bbox.size_x = det['bbox'][2]
            detection_2d.bbox.size_y = det['bbox'][3]

            # Add detection result with confidence
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det.get('class_id', 'unknown')
            hypothesis.hypothesis.score = det['confidence']
            detection_2d.results.append(hypothesis)

            # Store tracking ID as a custom field (in a real system, you'd use a custom message)
            detection_2d.bbox.center.z = det.get('track_id', -1)  # Using z as tracking ID storage

            detection_array.detections.append(detection_2d)

        self.detection_publisher.publish(detection_array)

    def publish_performance_metrics(self):
        """Publish performance metrics"""
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            # Publish FPS
            fps_msg = Float32()
            fps_msg.data = float(fps)
            self.performance_publisher.publish(fps_msg)

            # Log performance periodically
            current_time = time.time()
            if current_time - self.last_performance_report > 5.0:  # Every 5 seconds
                self.get_logger().info(
                    f'Perception performance: {fps:.2f} FPS, '
                    f'avg frame time: {avg_frame_time*1000:.2f} ms, '
                    f'tracked objects: {len(self.tracked_objects)}'
                )
                self.last_performance_report = current_time

def main(args=None):
    rclpy.init(args=args)

    perception_pipeline = CompleteIsaacPerceptionPipeline()

    try:
        rclpy.spin(perception_pipeline)
    except KeyboardInterrupt:
        perception_pipeline.get_logger().info('Shutting down perception pipeline')
    finally:
        perception_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating the Setup Script

Let's create a setup script to install dependencies and build the package:

```python
# File: isaac_perception_system/setup.py
from setuptools import setup

package_name = 'isaac_perception_system'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/perception_system.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Isaac Perception System for Robotics',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'isaac_perception_node = isaac_perception_system.complete_perception_pipeline:main',
            'isaac_perception_sim = isaac_perception_system.perception_node:main',
        ],
    },
)
```

```xml
<!-- File: isaac_perception_system/package.xml -->
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>isaac_perception_system</name>
  <version>1.0.0</version>
  <description>Isaac Perception System for Robotics</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>vision_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>std_msgs</depend>
  <depend>builtin_interfaces</depend>
  <depend>cv_bridge</end>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Testing the Perception System

Create a test script to validate the perception system:

```python
# File: isaac_perception_system/test/test_perception_system.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import threading
import time

class TestIsaacPerceptionSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_perception_system')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        # Create publisher for test images
        self.image_publisher = self.node.create_publisher(
            Image, 'camera/image_rect_color', 10
        )

        # Create subscriber for detections
        self.detections_received = []
        self.detection_subscription = self.node.create_subscription(
            'isaac_perception/detections',
            lambda msg: self.detections_received.append(msg)
        )

        self.bridge = CvBridge()

    def test_basic_perception_pipeline(self):
        """Test that the perception pipeline processes images and produces detections"""
        # Create a simple test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ros_image = self.bridge.cv2_to_imgmsg(test_image, encoding='bgr8')

        # Publish test image
        self.image_publisher.publish(ros_image)

        # Wait for detections
        timeout = time.time() + 5.0  # 5 second timeout
        while time.time() < timeout and len(self.detections_received) == 0:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Check that detections were received
        self.assertGreater(len(self.detections_received), 0,
                          "No detections received from perception system")

        # Check that detections have expected structure
        detection = self.detections_received[0]
        self.assertIsNotNone(detection.header)
        self.assertGreater(len(detection.detections), 0,
                          "Detections array is empty")

    def test_performance_metrics(self):
        """Test that performance metrics are published"""
        # Implementation would check performance metrics
        pass

    def test_tracking_functionality(self):
        """Test that object tracking works across frames"""
        # Implementation would test tracking consistency
        pass

def main():
    unittest.main()

if __name__ == '__main__':
    main()
```

## Running the Complete Perception System

Now let's create instructions for running the complete system:

```bash
#!/bin/bash
# File: isaac_perception_system/run_perception_system.sh

# Script to run the Isaac Perception System

echo "Setting up Isaac Perception System..."

# Source ROS 2 environment
source /opt/ros/galactic/setup.bash

# Source workspace
cd ~/isaac_ws
source install/setup.bash

# Build the workspace if not already built
colcon build --packages-select isaac_perception_system

# Source again after build
source install/setup.bash

echo "Starting Isaac Perception System..."

# Run the perception system
ros2 launch isaac_perception_system perception_system.launch.py

echo "Isaac Perception System finished."
```

## Validation and Testing

Create a validation script to test the perception system:

```python
# File: isaac_perception_system/scripts/validate_perception_system.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import time
from std_msgs.msg import Float32

class PerceptionValidator(Node):
    def __init__(self):
        super().__init__('perception_validator')

        self.bridge = CvBridge()
        self.detection_count = 0
        self.performance_metrics = []

        # Publishers
        self.image_publisher = self.create_publisher(Image, 'camera/image_rect_color', 10)

        # Subscribers
        self.detection_subscription = self.create_subscription(
            Detection2DArray,
            'isaac_perception/detections',
            self.detection_callback,
            10
        )

        self.performance_subscription = self.create_subscription(
            Float32,
            'isaac_perception/performance',
            self.performance_callback,
            10
        )

        # Timer for sending test images
        self.timer = self.create_timer(0.1, self.publish_test_image)  # 10 Hz
        self.test_start_time = time.time()
        self.test_duration = 30  # Test for 30 seconds

    def publish_test_image(self):
        """Publish a test image for perception system"""
        current_time = time.time()
        if current_time - self.test_start_time > self.test_duration:
            self.get_logger().info('Test duration completed. Shutting down.')
            rclpy.shutdown()
            return

        # Create a synthetic test image with known objects
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Add some synthetic objects (circles and rectangles)
        cv2.circle(image, (100, 100), 30, (255, 0, 0), -1)  # Blue circle
        cv2.rectangle(image, (200, 200), (300, 300), (0, 255, 0), -1)  # Green rectangle

        ros_image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'camera_frame'

        self.image_publisher.publish(ros_image)

    def detection_callback(self, msg):
        """Handle incoming detections"""
        self.detection_count += len(msg.detections)
        self.get_logger().info(f'Received {len(msg.detections)} detections, total: {self.detection_count}')

    def performance_callback(self, msg):
        """Handle performance metrics"""
        self.performance_metrics.append(msg.data)
        if len(self.performance_metrics) % 10 == 0:  # Log every 10 metrics
            avg_fps = np.mean(self.performance_metrics[-10:]) if self.performance_metrics else 0
            self.get_logger().info(f'Average FPS (last 10): {avg_fps:.2f}')

def main(args=None):
    rclpy.init(args=args)

    validator = PerceptionValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Validation interrupted by user')
    finally:
        # Print final statistics
        if validator.performance_metrics:
            avg_fps = np.mean(validator.performance_metrics)
            std_fps = np.std(validator.performance_metrics)
            validator.get_logger().info(f'Final Results:')
            validator.get_logger().info(f'  Average FPS: {avg_fps:.2f} ± {std_fps:.2f}')
            validator.get_logger().info(f'  Total detections: {validator.detection_count}')

        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Deployment on Edge Hardware

Create a deployment script for Jetson platforms:

```bash
#!/bin/bash
# File: isaac_perception_system/deploy_to_jetson.sh

# Deployment script for Isaac Perception System on Jetson

echo "Deploying Isaac Perception System to Jetson..."

# Set Jetson-specific parameters
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all

# Optimize for Jetson
sudo nvpmodel -m 0  # Set to maximum performance mode
sudo jetson_clocks  # Lock clocks to maximum frequency

# Build for Jetson architecture
cd ~/isaac_ws
colcon build --packages-select isaac_perception_system --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the workspace
source install/setup.bash

# Run with Jetson-optimized parameters
echo "Starting Isaac Perception System on Jetson..."
ros2 launch isaac_perception_system perception_system.launch.py \
    model_path:=/models/jetson/resnet18_detector.trt \
    confidence_threshold:=0.6

echo "Deployment completed."
```

## Summary and Next Steps

This lab provided hands-on experience with creating a complete Isaac perception system. You learned to:

1. Set up the Isaac ROS development environment
2. Create a perception pipeline with detection, segmentation, and tracking
3. Optimize the system for real-time performance
4. Validate the system with test data
5. Deploy the system on edge hardware

### Key Takeaways

- Isaac ROS provides GPU-accelerated perception packages that significantly outperform CPU-based alternatives
- Proper synchronization of sensor data is crucial for accurate perception
- Performance optimization involves balancing computational load with real-time constraints
- Validation and testing are essential for robust perception systems

### Further Enhancements

Consider these improvements for your perception system:

1. **Advanced Tracking**: Implement more sophisticated tracking algorithms like Deep SORT
2. **Multi-Sensor Fusion**: Integrate LiDAR data with camera data for better 3D perception
3. **Model Optimization**: Use TensorRT optimization for better inference performance
4. **Adaptive Processing**: Adjust processing parameters based on scene complexity

## References

[All sources will be cited in the References section at the end of the book, following APA format]