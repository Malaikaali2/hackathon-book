---
sidebar_position: 19
---

# Object Detection and Manipulation: Autonomous Humanoid Capstone

## Overview

Object detection and manipulation form the physical interaction layer of the autonomous humanoid system, enabling the robot to perceive, identify, and manipulate objects in its environment. This component encompasses computer vision for object recognition, 3D pose estimation for spatial understanding, grasp planning for manipulation, and precise control for executing manipulation tasks. The system must handle various object types, sizes, and materials while maintaining real-time performance and safety guarantees.

The manipulation system integrates with navigation for precise positioning, with perception for object information, with task planning for coordinated manipulation tasks, and with voice processing for object-related commands. This implementation guide provides detailed instructions for building a robust object detection and manipulation system that can operate effectively in real-world environments with varying lighting and object conditions.

## System Architecture

### Perception and Manipulation Pipeline

The object detection and manipulation system implements a multi-stage pipeline architecture:

```
RGB-D Input → Object Detection → Pose Estimation → Grasp Planning → Manipulation Execution → Task Completion
```

The architecture consists of:
1. **Perception Module**: Detects and classifies objects in the environment
2. **Pose Estimation**: Determines 6-DOF poses of detected objects
3. **Grasp Planning**: Plans feasible grasps for manipulation
4. **Manipulation Control**: Executes precise manipulation actions
5. **Feedback Integration**: Monitors execution and handles failures
6. **Learning Component**: Improves performance through experience

### Integration with Other Systems

The manipulation system interfaces with:
- **Navigation System**: Receives positioning commands for object access
- **Task Planning**: Coordinates manipulation tasks with overall plan
- **Perception System**: Gets environmental information and object data
- **Voice Processing**: Handles object-related commands and queries
- **Localization**: Maintains accurate spatial relationships

## Technical Implementation

### 1. Object Detection and Recognition

#### Deep Learning-Based Object Detection

```python
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from std_msgs.msg import String

class ObjectDetector:
    """Deep learning-based object detection system"""

    def __init__(self, model_path=None):
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained object detection model (e.g., YOLOv5, Detectron2, etc.)
        if model_path:
            self.model = torch.load(model_path)
        else:
            # Use a standard model like Faster R-CNN pre-trained on COCO
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        self.model.to(self.device)
        self.model.eval()

        # COCO dataset class names
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Confidence threshold for detections
        self.confidence_threshold = 0.5

        # Transform for input images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def detect_objects(self, image):
        """Detect objects in an input image"""
        # Convert ROS image to OpenCV format if needed
        if isinstance(image, ImageMsg):
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        else:
            cv_image = image

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Process predictions
        detections = []
        for i, (boxes, scores, labels) in enumerate(zip(predictions[0]['boxes'],
                                                       predictions[0]['scores'],
                                                       predictions[0]['labels'])):
            for box, score, label in zip(boxes, scores, labels):
                if score >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.tolist()
                    class_name = self.coco_names[label.item()]

                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'score': score.item(),
                        'class_name': class_name,
                        'class_id': label.item()
                    }
                    detections.append(detection)

        return detections

    def filter_detections_by_class(self, detections, target_classes):
        """Filter detections to include only specific object classes"""
        filtered_detections = []
        for detection in detections:
            if detection['class_name'] in target_classes:
                filtered_detections.append(detection)
        return filtered_detections

    def get_object_center(self, detection):
        """Get the center point of a detection bounding box"""
        x1, y1, x2, y2 = detection['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y

class SemanticObjectDetector:
    """Semantic object detection with custom training capabilities"""

    def __init__(self, custom_model_path=None):
        self.base_detector = ObjectDetector()
        self.custom_objects = {}  # Custom object models
        self.semantic_knowledge = self._load_semantic_knowledge()

        if custom_model_path:
            self.load_custom_model(custom_model_path)

    def _load_semantic_knowledge(self):
        """Load semantic knowledge about objects"""
        return {
            'cup': {
                'graspable': True,
                'typical_size': [0.05, 0.1, 0.1],  # width, height, depth in meters
                'grasp_points': ['handle', 'body'],
                'manipulation_constraints': {
                    'max_weight': 0.5,
                    'orientation_sensitive': True
                }
            },
            'book': {
                'graspable': True,
                'typical_size': [0.2, 0.3, 0.02],
                'grasp_points': ['spine', 'cover'],
                'manipulation_constraints': {
                    'max_weight': 1.0,
                    'orientation_sensitive': False
                }
            },
            'bottle': {
                'graspable': True,
                'typical_size': [0.07, 0.25, 0.07],
                'grasp_points': ['neck', 'body'],
                'manipulation_constraints': {
                    'max_weight': 1.0,
                    'orientation_sensitive': True
                }
            }
        }

    def load_custom_model(self, model_path):
        """Load a custom object detection model"""
        custom_model = torch.load(model_path)
        self.custom_objects = custom_model

    def detect_custom_objects(self, image, custom_classes=None):
        """Detect custom objects using specialized models"""
        if not custom_classes:
            custom_classes = list(self.custom_objects.keys())

        # For each custom class, apply specialized detection
        all_detections = self.base_detector.detect_objects(image)
        custom_detections = []

        for obj_class in custom_classes:
            if obj_class in self.custom_objects:
                # Apply custom detection logic
                class_detections = self._detect_specific_class(image, obj_class)
                custom_detections.extend(class_detections)

        # Combine with base detections
        all_detections.extend(custom_detections)
        return all_detections

    def _detect_specific_class(self, image, class_name):
        """Detect a specific object class using specialized approach"""
        # This would implement class-specific detection logic
        # For example, using template matching, geometric features, etc.
        return []

    def get_object_properties(self, object_name):
        """Get semantic properties of an object"""
        return self.semantic_knowledge.get(object_name, {})
```

#### 3D Pose Estimation

```python
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class PoseEstimator:
    """Estimates 6-DOF poses of detected objects"""

    def __init__(self):
        self.camera_intrinsics = None
        self.point_cloud_resolution = 0.01  # 1cm resolution

    def set_camera_intrinsics(self, fx, fy, cx, cy):
        """Set camera intrinsic parameters"""
        self.camera_intrinsics = {
            'fx': fx, 'fy': fy,
            'cx': cx, 'cy': cy
        }

    def estimate_pose_3d(self, rgb_image, depth_image, detection):
        """Estimate 3D pose of a detected object"""
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics not set")

        # Get bounding box coordinates
        x1, y1, x2, y2 = detection['bbox']
        center_x, center_y = self.base_detector.get_object_center(detection)

        # Extract region of interest from depth image
        roi_depth = depth_image[int(y1):int(y2), int(x1):int(x2)]

        # Convert to point cloud
        points_3d = self._depth_to_point_cloud(
            roi_depth,
            int(center_x), int(center_y)
        )

        if len(points_3d) < 10:  # Not enough points for reliable pose
            return None

        # Estimate object center in 3D
        object_center_3d = np.mean(points_3d, axis=0)

        # Estimate object orientation (simplified - in practice, use more sophisticated methods)
        # For now, assume upright orientation
        orientation = R.from_euler('xyz', [0, 0, 0]).as_quat()

        pose_3d = {
            'position': object_center_3d,
            'orientation': orientation,
            'confidence': detection['score']
        }

        return pose_3d

    def _depth_to_point_cloud(self, depth_roi, center_x, center_y):
        """Convert depth ROI to 3D point cloud"""
        if self.camera_intrinsics is None:
            return np.array([])

        h, w = depth_roi.shape
        points = []

        for y in range(h):
            for x in range(w):
                depth_val = depth_roi[y, x]

                if depth_val > 0 and not np.isnan(depth_val):  # Valid depth
                    # Convert to 3D coordinates
                    z = depth_val
                    x_3d = (x - self.camera_intrinsics['cx']) * z / self.camera_intrinsics['fx']
                    y_3d = (y - self.camera_intrinsics['cy']) * z / self.camera_intrinsics['fy']

                    points.append([x_3d, y_3d, z])

        return np.array(points)

    def estimate_object_dimensions(self, point_cloud):
        """Estimate object dimensions from point cloud"""
        if len(point_cloud) == 0:
            return None

        # Calculate bounding box
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)

        dimensions = max_coords - min_coords
        return dimensions

class MultiViewPoseEstimator:
    """Uses multiple camera views for improved pose estimation"""

    def __init__(self):
        self.cameras = {}  # Multiple camera configurations
        self.fusion_threshold = 0.05  # 5cm threshold for fusion

    def add_camera(self, camera_id, intrinsics, extrinsics):
        """Add a camera to the multi-view system"""
        self.cameras[camera_id] = {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics  # Transform from camera to robot base
        }

    def estimate_pose_multiview(self, images, detections):
        """Estimate pose using multiple camera views"""
        # Get pose estimates from each camera
        pose_estimates = []

        for cam_id, detection in zip(self.cameras.keys(), detections):
            if cam_id in self.cameras:
                pose = self._estimate_single_view_pose(
                    images[cam_id], detection, cam_id
                )
                if pose:
                    # Transform to robot base frame
                    base_pose = self._transform_to_base_frame(pose, cam_id)
                    pose_estimates.append(base_pose)

        # Fuse estimates if multiple views available
        if len(pose_estimates) > 1:
            return self._fuse_poses(pose_estimates)
        elif len(pose_estimates) == 1:
            return pose_estimates[0]
        else:
            return None

    def _transform_to_base_frame(self, pose, camera_id):
        """Transform pose from camera frame to robot base frame"""
        extrinsics = self.cameras[camera_id]['extrinsics']
        # Apply transformation (simplified)
        # In practice, this would use proper transformation matrices
        return pose
```

### 2. Grasp Planning and Manipulation

#### Grasp Planning System

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class GraspPose:
    """Represents a potential grasp pose"""
    position: np.ndarray  # 3D position [x, y, z]
    orientation: np.ndarray  # Quaternion [x, y, z, w]
    approach_direction: np.ndarray  # Approach vector [x, y, z]
    grasp_type: str  # 'pinch', 'power', 'hook', etc.
    score: float  # Quality score
    width: float  # Required gripper width

class GraspPlanner:
    """Plans feasible grasps for objects"""

    def __init__(self):
        self.gripper_width_range = (0.01, 0.1)  # 1-10cm gripper range
        self.approach_distance = 0.1  # 10cm approach distance
        self.lift_distance = 0.05  # 5cm lift after grasp

    def plan_grasps(self, object_pose, object_dimensions, object_class=None):
        """Plan potential grasps for an object"""
        grasps = []

        # Generate multiple grasp candidates based on object dimensions
        center = object_pose['position']
        dims = object_dimensions

        # Side grasp (for objects with handle or narrow profile)
        if dims[0] < dims[1] and dims[0] < dims[2]:  # Narrowest dimension is width
            side_grasp = self._generate_side_grasp(center, dims, 'pinch')
            if side_grasp:
                grasps.append(side_grasp)

        # Top grasp (for objects suitable for overhead grasping)
        if dims[2] > 0.1:  # Object has sufficient height
            top_grasp = self._generate_top_grasp(center, dims, 'pinch')
            if top_grasp:
                grasps.append(top_grasp)

        # Power grasp (for larger objects)
        if dims[0] > 0.05 or dims[1] > 0.05:  # Object is large enough
            power_grasp = self._generate_power_grasp(center, dims, 'power')
            if power_grasp:
                grasps.append(power_grasp)

        # Sort grasps by score
        grasps.sort(key=lambda g: g.score, reverse=True)

        return grasps

    def _generate_side_grasp(self, center, dimensions, grasp_type):
        """Generate a side grasp for the object"""
        # Approach from the side (along the narrowest dimension)
        approach_dir = np.array([1, 0, 0])  # Default approach from positive X
        position = center + approach_dir * (dimensions[0] / 2 + 0.05)  # 5cm offset

        # Orientation: gripper aligned with object
        orientation = R.from_euler('xyz', [0, 0, 0]).as_quat()

        # Calculate required gripper width (based on object thickness)
        gripper_width = min(dimensions[1], dimensions[2]) * 0.8  # 80% of smaller dimension

        # Check if gripper width is within range
        if not (self.gripper_width_range[0] <= gripper_width <= self.gripper_width_range[1]):
            return None

        grasp = GraspPose(
            position=position,
            orientation=orientation,
            approach_direction=approach_dir,
            grasp_type=grasp_type,
            score=0.8,  # High score for side grasp
            width=gripper_width
        )

        return grasp

    def _generate_top_grasp(self, center, dimensions, grasp_type):
        """Generate a top grasp for the object"""
        # Approach from above
        approach_dir = np.array([0, 0, -1])  # Approach from above
        position = center + approach_dir * (dimensions[2] / 2 + 0.05)  # 5cm above object

        # Orientation: gripper aligned with object
        orientation = R.from_euler('xyz', [0, 0, 0]).as_quat()

        # Calculate required gripper width
        gripper_width = min(dimensions[0], dimensions[1]) * 0.8

        # Check if gripper width is within range
        if not (self.gripper_width_range[0] <= gripper_width <= self.gripper_width_range[1]):
            return None

        grasp = GraspPose(
            position=position,
            orientation=orientation,
            approach_direction=approach_dir,
            grasp_type=grasp_type,
            score=0.7,  # Good score for top grasp
            width=gripper_width
        )

        return grasp

    def _generate_power_grasp(self, center, dimensions, grasp_type):
        """Generate a power grasp for the object"""
        # Approach from the side for power grasp
        approach_dir = np.array([0, 1, 0])  # Approach from positive Y
        position = center + approach_dir * (dimensions[1] / 2 + 0.05)

        # Orientation: gripper perpendicular to approach
        orientation = R.from_euler('xyz', [0, 0, np.pi/2]).as_quat()

        # Calculate required gripper width
        gripper_width = max(dimensions[0], dimensions[2]) * 0.6  # Use larger dimensions for power grasp

        # Check if gripper width is within range
        if not (self.gripper_width_range[0] <= gripper_width <= self.gripper_width_range[1]):
            return None

        grasp = GraspPose(
            position=position,
            orientation=orientation,
            approach_direction=approach_dir,
            grasp_type=grasp_type,
            score=0.6,  # Good score for power grasp
            width=gripper_width
        )

        return grasp

    def validate_grasp(self, grasp_pose, object_pose, environment_data):
        """Validate if a grasp is feasible in the current environment"""
        # Check for collisions with environment
        if self._check_collision(grasp_pose, environment_data):
            return False, "Collision detected"

        # Check if approach direction is clear
        if not self._check_approach_clear(grasp_pose, environment_data):
            return False, "Approach path blocked"

        # Check if object is accessible
        if not self._check_accessibility(grasp_pose):
            return False, "Object not accessible"

        return True, "Valid grasp"

    def _check_collision(self, grasp_pose, environment_data):
        """Check for collisions at grasp pose"""
        # This would interface with collision checking system
        # For now, return False (no collision)
        return False

    def _check_approach_clear(self, grasp_pose, environment_data):
        """Check if approach path is clear"""
        # This would check along the approach direction
        # For now, return True (path clear)
        return True

    def _check_accessibility(self, grasp_pose):
        """Check if the grasp pose is physically reachable"""
        # This would check against robot kinematic constraints
        # For now, return True (accessible)
        return True

class GraspOptimizer:
    """Optimizes grasp selection based on multiple criteria"""

    def __init__(self):
        self.weights = {
            'stability': 0.4,
            'accessibility': 0.3,
            'efficiency': 0.2,
            'safety': 0.1
        }

    def select_best_grasp(self, grasp_candidates, object_info, robot_state):
        """Select the best grasp based on multiple criteria"""
        if not grasp_candidates:
            return None

        best_grasp = None
        best_score = -float('inf')

        for grasp in grasp_candidates:
            score = self._evaluate_grasp(grasp, object_info, robot_state)

            if score > best_score:
                best_score = score
                best_grasp = grasp

        return best_grasp

    def _evaluate_grasp(self, grasp, object_info, robot_state):
        """Evaluate a grasp based on multiple criteria"""
        stability_score = self._evaluate_stability(grasp, object_info)
        accessibility_score = self._evaluate_accessibility(grasp, robot_state)
        efficiency_score = self._evaluate_efficiency(grasp, robot_state)
        safety_score = self._evaluate_safety(grasp)

        # Weighted sum of all criteria
        total_score = (
            self.weights['stability'] * stability_score +
            self.weights['accessibility'] * accessibility_score +
            self.weights['efficiency'] * efficiency_score +
            self.weights['safety'] * safety_score
        )

        return total_score

    def _evaluate_stability(self, grasp, object_info):
        """Evaluate grasp stability"""
        # Stability depends on grasp type and object properties
        if grasp.grasp_type == 'pinch':
            return 0.8 if grasp.score > 0.7 else 0.6
        elif grasp.grasp_type == 'power':
            return 0.9 if grasp.score > 0.6 else 0.7
        else:
            return grasp.score

    def _evaluate_accessibility(self, grasp, robot_state):
        """Evaluate how accessible the grasp is"""
        # This would check robot kinematic constraints
        # For now, return a simple score based on grasp position
        distance_from_base = np.linalg.norm(grasp.position)
        if distance_from_base < 1.0:  # Within reach
            return 1.0
        elif distance_from_base < 1.5:  # Reachable but difficult
            return 0.7
        else:  # Out of reach
            return 0.2

    def _evaluate_efficiency(self, grasp, robot_state):
        """Evaluate manipulation efficiency"""
        # Efficiency depends on how well the grasp aligns with the task
        return 0.8  # Default high efficiency

    def _evaluate_safety(self, grasp):
        """Evaluate safety of the grasp"""
        # Safety depends on object properties and grasp type
        return 1.0  # Assume safe for now
```

#### Manipulation Execution System

```python
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped, Point, Quaternion

class ManipulationController:
    """Controls the robot arm for manipulation tasks"""

    def __init__(self, arm_controller_name='/arm_controller/follow_joint_trajectory'):
        self.arm_client = actionlib.SimpleActionClient(arm_controller_name, FollowJointTrajectoryAction)
        self.gripper_client = actionlib.SimpleActionClient('/gripper_controller/gripper_cmd', GripperCommandAction)

        # Wait for action servers to be available
        rospy.loginfo("Waiting for action servers...")
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()

        # Robot-specific parameters
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.gripper_joint = 'gripper_joint'

        # Safety parameters
        self.max_velocity = 0.5  # rad/s
        self.max_acceleration = 1.0  # rad/s^2
        self.gripper_force_limit = 50.0  # N

    def execute_grasp(self, grasp_pose, object_info):
        """Execute a grasp at the specified pose"""
        try:
            # Move to pre-grasp position
            pre_grasp_pose = self._calculate_pre_grasp_pose(grasp_pose)
            success = self.move_to_pose(pre_grasp_pose)

            if not success:
                rospy.logerr("Failed to move to pre-grasp pose")
                return False

            # Approach the object
            approach_success = self._approach_object(grasp_pose)
            if not approach_success:
                rospy.logerr("Failed to approach object")
                return False

            # Execute grasp
            grasp_success = self._execute_grasp_action(grasp_pose, object_info)
            if not grasp_success:
                rospy.logerr("Failed to execute grasp")
                # Try to recover
                self._recover_from_grasp_failure(grasp_pose)
                return False

            # Lift the object
            lift_success = self._lift_object(grasp_pose)
            if not lift_success:
                rospy.logerr("Failed to lift object")
                return False

            rospy.loginfo("Successfully grasped object")
            return True

        except Exception as e:
            rospy.logerr(f"Error executing grasp: {e}")
            return False

    def _calculate_pre_grasp_pose(self, grasp_pose):
        """Calculate pre-grasp pose by moving back along approach direction"""
        pre_grasp_pos = grasp_pose.position - grasp_pose.approach_direction * 0.1  # 10cm back
        return {
            'position': pre_grasp_pos,
            'orientation': grasp_pose.orientation
        }

    def _approach_object(self, grasp_pose):
        """Move from pre-grasp to grasp position"""
        # Create trajectory with smooth approach
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        # Add intermediate points for smooth approach
        current_pos = self.get_current_pose()
        approach_steps = 10

        for i in range(approach_steps + 1):
            fraction = i / approach_steps
            step_pos = (1 - fraction) * current_pos['position'] + fraction * grasp_pose.position

            point = JointTrajectoryPoint()
            point.positions = self._inverse_kinematics(step_pos, grasp_pose.orientation)
            point.velocities = [0.0] * len(self.joint_names)  # Start and end with zero velocity
            point.accelerations = [0.0] * len(self.joint_names)
            point.time_from_start = rospy.Duration(2.0 * fraction)  # 2 seconds total

            trajectory.points.append(point)

        # Send trajectory goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = rospy.Duration(1.0)

        self.arm_client.send_goal(goal)
        self.arm_client.wait_for_result(rospy.Duration(10.0))  # 10 second timeout

        return self.arm_client.get_result() is not None

    def _execute_grasp_action(self, grasp_pose, object_info):
        """Execute the actual grasp action"""
        # Close gripper to appropriate width
        gripper_cmd = GripperCommand()
        gripper_cmd.position = grasp_pose.width * 0.8  # Close to 80% of required width
        gripper_cmd.max_effort = self.gripper_force_limit

        goal = GripperCommandGoal()
        goal.command = gripper_cmd

        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result(rospy.Duration(5.0))  # 5 second timeout

        result = self.gripper_client.get_result()
        return result and result.reached_goal

    def _lift_object(self, grasp_pose):
        """Lift the object after successful grasp"""
        # Move up by lift distance
        lift_pos = grasp_pose.position + np.array([0, 0, 0.05])  # Lift 5cm

        lift_pose = {
            'position': lift_pos,
            'orientation': grasp_pose.orientation
        }

        return self.move_to_pose(lift_pose)

    def _recover_from_grasp_failure(self, grasp_pose):
        """Attempt to recover from grasp failure"""
        rospy.loginfo("Attempting grasp recovery...")

        # Move back to safe position
        safe_pos = grasp_pose.position + grasp_pose.approach_direction * 0.15  # Move back 15cm
        safe_pose = {
            'position': safe_pos,
            'orientation': grasp_pose.orientation
        }

        self.move_to_pose(safe_pose)

    def move_to_pose(self, pose):
        """Move the end effector to a specified pose"""
        try:
            # Calculate joint angles using inverse kinematics
            joint_angles = self._inverse_kinematics(pose['position'], pose['orientation'])

            if joint_angles is None:
                rospy.logerr("Could not find IK solution")
                return False

            # Create trajectory
            trajectory = JointTrajectory()
            trajectory.joint_names = self.joint_names

            point = JointTrajectoryPoint()
            point.positions = joint_angles
            point.velocities = [0.0] * len(self.joint_names)
            point.accelerations = [0.0] * len(self.joint_names)
            point.time_from_start = rospy.Duration(3.0)  # 3 seconds

            trajectory.points.append(point)

            # Send goal
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = trajectory
            goal.goal_time_tolerance = rospy.Duration(1.0)

            self.arm_client.send_goal(goal)
            self.arm_client.wait_for_result(rospy.Duration(10.0))

            return self.arm_client.get_result() is not None

        except Exception as e:
            rospy.logerr(f"Error moving to pose: {e}")
            return False

    def _inverse_kinematics(self, position, orientation):
        """Calculate inverse kinematics for desired pose"""
        # This would interface with the robot's IK solver
        # For now, return a placeholder solution
        # In practice, this would use KDL, MoveIt!, or robot-specific IK
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Placeholder joint angles

    def get_current_pose(self):
        """Get current end-effector pose"""
        # This would interface with forward kinematics or TF
        # For now, return a placeholder
        return {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0])
        }

class PlaceController:
    """Controls object placement operations"""

    def __init__(self, manipulation_controller):
        self.manip_ctrl = manipulation_controller
        self.approach_distance = 0.1  # 10cm approach for placement

    def place_object(self, target_pose, placement_type='careful'):
        """Place the currently grasped object at target pose"""
        try:
            # Move to pre-place position
            pre_place_pose = self._calculate_pre_place_pose(target_pose)
            success = self.manip_ctrl.move_to_pose(pre_place_pose)

            if not success:
                rospy.logerr("Failed to move to pre-place pose")
                return False

            # Approach placement location
            approach_success = self._approach_placement(target_pose, placement_type)
            if not approach_success:
                rospy.logerr("Failed to approach placement location")
                return False

            # Release object
            release_success = self._release_object(placement_type)
            if not release_success:
                rospy.logerr("Failed to release object")
                return False

            # Retract gripper
            retract_success = self._retract_gripper(target_pose)
            if not retract_success:
                rospy.logerr("Failed to retract gripper")
                return False

            rospy.loginfo("Successfully placed object")
            return True

        except Exception as e:
            rospy.logerr(f"Error placing object: {e}")
            return False

    def _calculate_pre_place_pose(self, target_pose):
        """Calculate pre-place pose by moving up from target"""
        pre_place_pos = target_pose['position'] + np.array([0, 0, 0.1])  # 10cm above target
        return {
            'position': pre_place_pos,
            'orientation': target_pose['orientation']
        }

    def _approach_placement(self, target_pose, placement_type):
        """Approach the placement location"""
        # Move down to target position
        trajectory = JointTrajectory()
        trajectory.joint_names = self.manip_ctrl.joint_names

        # Create smooth descent trajectory
        current_pos = self.manip_ctrl.get_current_pose()
        descent_steps = 10

        for i in range(descent_steps + 1):
            fraction = i / descent_steps
            step_pos = (1 - fraction) * current_pos['position'] + fraction * target_pose['position']

            point = JointTrajectoryPoint()
            point.positions = self.manip_ctrl._inverse_kinematics(step_pos, target_pose['orientation'])
            point.velocities = [0.0] * len(self.manip_ctrl.joint_names)
            point.accelerations = [0.0] * len(self.manip_ctrl.joint_names)
            point.time_from_start = rospy.Duration(2.0 * fraction)

            trajectory.points.append(point)

        # Send trajectory
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = rospy.Duration(1.0)

        self.manip_ctrl.arm_client.send_goal(goal)
        self.manip_ctrl.arm_client.wait_for_result(rospy.Duration(10.0))

        return self.manip_ctrl.arm_client.get_result() is not None

    def _release_object(self, placement_type):
        """Release the grasped object"""
        # Open gripper fully
        gripper_cmd = GripperCommand()
        gripper_cmd.position = 0.1  # Fully open
        gripper_cmd.max_effort = 5.0  # Low effort to avoid dropping too quickly

        goal = GripperCommandGoal()
        goal.command = gripper_cmd

        self.manip_ctrl.gripper_client.send_goal(goal)
        self.manip_ctrl.gripper_client.wait_for_result(rospy.Duration(5.0))

        result = self.manip_ctrl.gripper_client.get_result()
        return result is not None

    def _retract_gripper(self, target_pose):
        """Retract gripper after placement"""
        # Move gripper up and away from placed object
        retract_pos = target_pose['position'] + np.array([0, 0, 0.15])  # 15cm above
        retract_pose = {
            'position': retract_pos,
            'orientation': target_pose['orientation']
        }

        return self.manip_ctrl.move_to_pose(retract_pose)
```

### 3. Integration with Perception and Navigation

#### Perception-Action Coordination

```python
class PerceptionActionCoordinator:
    """Coordinates perception and action systems"""

    def __init__(self):
        self.object_detector = SemanticObjectDetector()
        self.pose_estimator = MultiViewPoseEstimator()
        self.grasp_planner = GraspPlanner()
        self.grasp_optimizer = GraspOptimizer()
        self.manip_controller = ManipulationController()
        self.place_controller = PlaceController(self.manip_controller)

        # ROS interfaces
        self.rgb_sub = rospy.Subscriber('/camera/rgb/image_raw', ImageMsg, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', ImageMsg, self.depth_callback)
        self.pointcloud_sub = rospy.Subscriber('/camera/depth/points', PointCloud2, self.pointcloud_callback)

        self.current_rgb = None
        self.current_depth = None
        self.current_pointcloud = None

    def rgb_callback(self, msg):
        """Handle RGB image callback"""
        self.current_rgb = msg

    def depth_callback(self, msg):
        """Handle depth image callback"""
        # Convert to appropriate format
        self.current_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def pointcloud_callback(self, msg):
        """Handle point cloud callback"""
        self.current_pointcloud = msg

    def find_and_grasp_object(self, target_object, target_location=None):
        """Find and grasp a specific object"""
        # First, navigate to the area if location specified
        if target_location:
            rospy.loginfo(f"Navigating to {target_location} to find {target_object}")
            # This would interface with navigation system
            # navigation_client.send_goal(target_location)
            # navigation_client.wait_for_result()

        # Detect objects in current view
        if self.current_rgb is None:
            rospy.logerr("No RGB image available")
            return False

        detections = self.object_detector.detect_objects(self.current_rgb)

        # Filter for target object
        target_detections = self.object_detector.filter_detections_by_class(
            detections, [target_object]
        )

        if not target_detections:
            rospy.logwarn(f"No {target_object} detected in current view")
            return False

        # Get the highest confidence detection
        best_detection = max(target_detections, key=lambda d: d['score'])

        # Estimate 3D pose
        if self.current_depth is not None:
            object_pose = self.pose_estimator.estimate_pose_3d(
                self.current_rgb, self.current_depth, best_detection
            )
        else:
            rospy.logwarn("No depth image available for 3D pose estimation")
            return False

        if object_pose is None:
            rospy.logerr("Could not estimate 3D pose")
            return False

        # Get object dimensions if possible
        object_dims = self._estimate_object_dimensions(best_detection)

        # Plan grasps
        grasp_candidates = self.grasp_planner.plan_grasps(
            object_pose, object_dims, target_object
        )

        if not grasp_candidates:
            rospy.logerr("No valid grasps found")
            return False

        # Select best grasp
        current_robot_state = self._get_robot_state()
        best_grasp = self.grasp_optimizer.select_best_grasp(
            grasp_candidates,
            {'object_class': target_object, 'dimensions': object_dims},
            current_robot_state
        )

        if best_grasp is None:
            rospy.logerr("Could not select a valid grasp")
            return False

        # Execute grasp
        rospy.loginfo(f"Attempting to grasp {target_object} with score {best_grasp.score}")
        success = self.manip_controller.execute_grasp(best_grasp, {'class': target_object})

        return success

    def place_object_at_location(self, target_pose, placement_type='careful'):
        """Place currently grasped object at target location"""
        return self.place_controller.place_object(target_pose, placement_type)

    def _estimate_object_dimensions(self, detection):
        """Estimate object dimensions from detection (simplified)"""
        # This would use more sophisticated methods in practice
        # For now, return typical dimensions based on class
        x1, y1, x2, y2 = detection['bbox']
        width_pixels = x2 - x1
        height_pixels = y2 - y1

        # Convert to approximate meters (this is very simplified)
        # In practice, you'd use calibrated camera parameters
        approximate_width = width_pixels * 0.001  # Rough conversion
        approximate_height = height_pixels * 0.001
        approximate_depth = 0.1  # Assume 10cm depth

        return np.array([approximate_width, approximate_height, approximate_depth])

    def _get_robot_state(self):
        """Get current robot state for grasp optimization"""
        # This would interface with robot state publisher
        return {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
            'joint_angles': [0.0] * 6
        }

class ManipulationTaskExecutor:
    """Executes complex manipulation tasks"""

    def __init__(self):
        self.coordinator = PerceptionActionCoordinator()
        self.task_queue = []
        self.current_task = None

    def pick_and_place_task(self, object_to_pick, pick_location, place_location):
        """Execute a pick and place task"""
        try:
            # Navigate to pick location
            rospy.loginfo(f"Navigating to pick up {object_to_pick}")
            # navigation logic would go here

            # Find and grasp the object
            rospy.loginfo(f"Looking for {object_to_pick}")
            grasp_success = self.coordinator.find_and_grasp_object(object_to_pick)

            if not grasp_success:
                rospy.logerr(f"Failed to grasp {object_to_pick}")
                return False

            rospy.loginfo(f"Successfully grasped {object_to_pick}")

            # Navigate to place location
            rospy.loginfo(f"Navigating to place location")
            # navigation logic would go here

            # Place the object
            place_pose = {
                'position': np.array([place_location['x'], place_location['y'], place_location['z']]),
                'orientation': np.array([0, 0, 0, 1])  # Default orientation
            }

            place_success = self.coordinator.place_object_at_location(place_pose)

            if place_success:
                rospy.loginfo(f"Successfully placed {object_to_pick}")
                return True
            else:
                rospy.logerr(f"Failed to place {object_to_pick}")
                return False

        except Exception as e:
            rospy.logerr(f"Error in pick and place task: {e}")
            return False

    def stack_objects_task(self, object_type, stack_location, num_objects):
        """Stack multiple objects of the same type"""
        for i in range(num_objects):
            rospy.loginfo(f"Stacking object {i+1} of {num_objects}")

            # Find and pick up object
            pick_success = self.coordinator.find_and_grasp_object(object_type)
            if not pick_success:
                rospy.logwarn(f"Could not find object {i+1}, stopping stack task")
                break

            # Calculate placement pose (higher for each subsequent object)
            stack_height = 0.1 * (i + 1)  # 10cm per object
            place_pose = {
                'position': np.array([stack_location['x'], stack_location['y'],
                                     stack_location['z'] + stack_height]),
                'orientation': np.array([0, 0, 0, 1])
            }

            # Place object
            place_success = self.coordinator.place_object_at_location(place_pose)
            if not place_success:
                rospy.logerr(f"Failed to place object {i+1} in stack")
                break

        return True  # Consider task successful even if not all objects placed
```

## Implementation Steps

### Step 1: Set up Object Detection Infrastructure

1. Create the main manipulation node:

```python
#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from manipulation_msgs.msg import ManipulationGoal, ManipulationResult

class ManipulationServer:
    def __init__(self):
        rospy.init_node('manipulation_server')

        # Initialize components
        self.coordinator = PerceptionActionCoordinator()
        self.task_executor = ManipulationTaskExecutor()

        # State variables
        self.is_busy = False
        self.current_object = None

        # Publishers and subscribers
        self.manipulation_sub = rospy.Subscriber('/manipulation_goal', String, self.manipulation_callback)
        self.object_detect_sub = rospy.Subscriber('/object_detection_goal', String, self.detection_callback)
        self.result_pub = rospy.Publisher('/manipulation_result', String, queue_size=10)
        self.status_pub = rospy.Publisher('/manipulation_status', String, queue_size=10)

        rospy.loginfo("Manipulation server initialized")

    def manipulation_callback(self, msg):
        """Handle manipulation commands"""
        if self.is_busy:
            rospy.logwarn("Manipulation server busy, rejecting new command")
            return

        try:
            # Parse manipulation command
            command = msg.data.strip().split()

            if len(command) >= 2:
                action = command[0].lower()
                target = command[1].lower()

                if action == 'pick':
                    self.is_busy = True
                    self._publish_status("EXECUTING_PICK")

                    success = self.coordinator.find_and_grasp_object(target)

                    if success:
                        self.current_object = target
                        self._publish_result(f"SUCCESS: Picked {target}")
                        self._publish_status("PICK_COMPLETED")
                    else:
                        self._publish_result(f"FAILURE: Could not pick {target}")
                        self._publish_status("PICK_FAILED")

                    self.is_busy = False

                elif action == 'place':
                    if self.current_object:
                        self.is_busy = True
                        self._publish_status("EXECUTING_PLACE")

                        # For now, place at default location
                        place_pose = {
                            'position': np.array([0.5, 0.0, 0.1]),
                            'orientation': np.array([0, 0, 0, 1])
                        }

                        success = self.coordinator.place_object_at_location(place_pose)

                        if success:
                            self._publish_result(f"SUCCESS: Placed {self.current_object}")
                            self._publish_status("PLACE_COMPLETED")
                            self.current_object = None
                        else:
                            self._publish_result(f"FAILURE: Could not place {self.current_object}")
                            self._publish_status("PLACE_FAILED")

                        self.is_busy = False
                    else:
                        self._publish_result("FAILURE: No object currently grasped")
                        self._publish_status("NO_OBJECT_TO_PLACE")

                elif action == 'detect':
                    self.is_busy = True
                    self._publish_status("EXECUTING_DETECTION")

                    # Detect objects in current view
                    if self.coordinator.current_rgb is not None:
                        detections = self.coordinator.object_detector.detect_objects(
                            self.coordinator.current_rgb
                        )

                        # Publish detected objects
                        detected_objects = [det['class_name'] for det in detections
                                          if det['score'] > 0.5]
                        result_msg = f"DETECTED: {', '.join(detected_objects)}"
                        self._publish_result(result_msg)
                        self._publish_status("DETECTION_COMPLETED")
                    else:
                        self._publish_result("FAILURE: No image data available")
                        self._publish_status("NO_IMAGE_DATA")

                    self.is_busy = False
            else:
                self._publish_result("FAILURE: Invalid command format")
                self._publish_status("INVALID_COMMAND")

        except Exception as e:
            rospy.logerr(f"Error processing manipulation command: {e}")
            self._publish_result(f"FAILURE: {str(e)}")
            self._publish_status("ERROR")
            self.is_busy = False

    def detection_callback(self, msg):
        """Handle object detection requests"""
        try:
            target_class = msg.data.strip()

            if self.coordinator.current_rgb is not None:
                detections = self.coordinator.object_detector.detect_objects(
                    self.coordinator.current_rgb
                )

                target_detections = self.coordinator.object_detector.filter_detections_by_class(
                    detections, [target_class]
                )

                if target_detections:
                    best_detection = max(target_detections, key=lambda d: d['score'])
                    result_msg = f"FOUND: {target_class} with confidence {best_detection['score']:.2f}"
                    self._publish_result(result_msg)
                else:
                    self._publish_result(f"NOT_FOUND: {target_class}")
            else:
                self._publish_result("FAILURE: No image data available")

        except Exception as e:
            rospy.logerr(f"Error in detection callback: {e}")
            self._publish_result(f"FAILURE: {str(e)}")

    def _publish_result(self, result):
        """Publish manipulation result"""
        result_msg = String()
        result_msg.data = result
        self.result_pub.publish(result_msg)

    def _publish_status(self, status):
        """Publish manipulation status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

    def run(self):
        """Main execution loop"""
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            # Perform any periodic tasks
            rate.sleep()

if __name__ == '__main__':
    manip_server = ManipulationServer()
    try:
        manip_server.run()
    except rospy.ROSInterruptException:
        pass
```

### Step 2: Configure Manipulation Parameters

Create a configuration file for manipulation parameters:

```yaml
# manipulation_params.yaml
manipulation_server:
  ros__parameters:
    gripper_joint: "gripper_joint"
    gripper_min_position: 0.0
    gripper_max_position: 0.1
    gripper_force_limit: 50.0
    approach_distance: 0.1
    lift_distance: 0.05
    max_velocity: 0.5
    max_acceleration: 1.0
    confidence_threshold: 0.7
    detection_timeout: 10.0

object_detector:
  ros__parameters:
    model_path: "/path/to/detection/model"
    confidence_threshold: 0.5
    nms_threshold: 0.4
    image_topic: "/camera/rgb/image_raw"
    depth_topic: "/camera/depth/image_raw"

grasp_planner:
  ros__parameters:
    gripper_width_min: 0.01
    gripper_width_max: 0.1
    approach_distance: 0.1
    lift_distance: 0.05
    max_grasps_to_generate: 10
```

### Step 3: Implement Advanced Features

```python
class LearningBasedManipulation:
    """Implement learning-based improvements to manipulation"""

    def __init__(self):
        self.success_history = {}  # Track success rates for different objects
        self.grasp_success_count = {}  # Count successful grasps per object type
        self.grasp_attempt_count = {}  # Count all grasp attempts per object type

    def record_grasp_attempt(self, object_class, grasp_pose, success):
        """Record the outcome of a grasp attempt"""
        if object_class not in self.grasp_attempt_count:
            self.grasp_attempt_count[object_class] = 0
            self.grasp_success_count[object_class] = 0

        self.grasp_attempt_count[object_class] += 1
        if success:
            self.grasp_success_count[object_class] += 1

    def get_object_success_rate(self, object_class):
        """Get success rate for grasping a particular object class"""
        if object_class not in self.grasp_attempt_count:
            return 0.0  # No data yet

        attempts = self.grasp_attempt_count[object_class]
        successes = self.grasp_success_count[object_class]

        if attempts == 0:
            return 0.0

        return successes / attempts

    def adapt_grasp_selection(self, grasp_candidates, object_class):
        """Adapt grasp selection based on historical success data"""
        success_rate = self.get_object_success_rate(object_class)

        if success_rate < 0.5:  # Low success rate, be more conservative
            # Favor higher-scoring grasps more heavily
            for grasp in grasp_candidates:
                grasp.score *= 1.2  # Boost score slightly for conservative approach
        elif success_rate > 0.8:  # High success rate, can be more adventurous
            # Consider more grasp options
            pass  # Keep original scores

        return grasp_candidates

class AdaptiveManipulationController(ManipulationController):
    """Manipulation controller with adaptive behavior"""

    def __init__(self, learning_module):
        super().__init__()
        self.learning_module = learning_module

    def execute_grasp_with_learning(self, grasp_pose, object_info):
        """Execute grasp with learning-based adaptations"""
        object_class = object_info.get('class', 'unknown')

        # Adapt grasp selection based on learning
        adapted_grasp = self._adapt_grasp_for_object(grasp_pose, object_class)

        # Execute the grasp
        success = self.execute_grasp(adapted_grasp, object_info)

        # Record the outcome
        self.learning_module.record_grasp_attempt(object_class, adapted_grasp, success)

        return success

    def _adapt_grasp_for_object(self, grasp_pose, object_class):
        """Adapt grasp based on object-specific knowledge"""
        success_rate = self.learning_module.get_object_success_rate(object_class)

        # If success rate is low, make conservative adjustments
        if success_rate < 0.5:
            # Increase approach distance for fragile objects
            adapted_pose = self._increase_approach_distance(grasp_pose)
        else:
            adapted_pose = grasp_pose

        return adapted_pose

    def _increase_approach_distance(self, grasp_pose):
        """Increase approach distance for safer grasping"""
        # Move approach position further from object
        approach_offset = grasp_pose.approach_direction * 0.02  # Additional 2cm
        adapted_pos = grasp_pose.position + approach_offset

        adapted_grasp = GraspPose(
            position=adapted_pos,
            orientation=grasp_pose.orientation,
            approach_direction=grasp_pose.approach_direction,
            grasp_type=grasp_pose.grasp_type,
            score=grasp_pose.score,
            width=grasp_pose.width
        )

        return adapted_grasp
```

## Testing and Validation

### Unit Testing

```python
import unittest
from unittest.mock import Mock, patch

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ObjectDetector()

    def test_object_detection(self):
        """Test basic object detection functionality"""
        # Create a simple test image (in practice, use a real image)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # This would test the detection pipeline
        # For now, we'll just ensure the method doesn't crash
        try:
            detections = self.detector.detect_objects(test_image)
            self.assertIsInstance(detections, list)
        except Exception as e:
            self.fail(f"Detection failed with error: {e}")

    def test_class_filtering(self):
        """Test filtering detections by class"""
        mock_detections = [
            {'bbox': [10, 10, 50, 50], 'score': 0.8, 'class_name': 'cup'},
            {'bbox': [60, 60, 100, 100], 'score': 0.7, 'class_name': 'book'},
            {'bbox': [110, 110, 150, 150], 'score': 0.9, 'class_name': 'bottle'}
        ]

        filtered = self.detector.filter_detections_by_class(mock_detections, ['cup', 'bottle'])

        self.assertEqual(len(filtered), 2)
        self.assertIn('cup', [d['class_name'] for d in filtered])
        self.assertIn('bottle', [d['class_name'] for d in filtered])

class TestGraspPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = GraspPlanner()

    def test_grasp_generation(self):
        """Test grasp generation for a simple object"""
        object_pose = {
            'position': np.array([0.5, 0.0, 0.1]),
            'orientation': np.array([0, 0, 0, 1])
        }
        object_dims = np.array([0.05, 0.05, 0.1])  # 5x5x10cm object

        grasps = self.planner.plan_grasps(object_pose, object_dims, 'cup')

        self.assertGreater(len(grasps), 0)
        for grasp in grasps:
            self.assertIsInstance(grasp, GraspPose)
            self.assertGreaterEqual(grasp.score, 0.0)
            self.assertLessEqual(grasp.score, 1.0)

    def test_grasp_validation(self):
        """Test grasp validation"""
        grasp_pose = GraspPose(
            position=np.array([0.5, 0.0, 0.1]),
            orientation=np.array([0, 0, 0, 1]),
            approach_direction=np.array([1, 0, 0]),
            grasp_type='pinch',
            score=0.8,
            width=0.03
        )

        # Mock environment data
        env_data = {}

        is_valid, reason = self.planner.validate_grasp(grasp_pose, object_pose, env_data)

        self.assertTrue(is_valid)

class TestGraspOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = GraspOptimizer()

    def test_grasp_selection(self):
        """Test best grasp selection"""
        grasps = [
            GraspPose(
                position=np.array([0.5, 0.0, 0.1]),
                orientation=np.array([0, 0, 0, 1]),
                approach_direction=np.array([1, 0, 0]),
                grasp_type='pinch',
                score=0.8,
                width=0.03
            ),
            GraspPose(
                position=np.array([0.6, 0.0, 0.1]),
                orientation=np.array([0, 0, 0, 1]),
                approach_direction=np.array([0, 1, 0]),
                grasp_type='power',
                score=0.7,
                width=0.05
            )
        ]

        object_info = {'class': 'cup', 'dimensions': np.array([0.05, 0.05, 0.1])}
        robot_state = {'position': np.array([0, 0, 0]), 'joint_angles': [0]*6}

        best_grasp = self.optimizer.select_best_grasp(grasps, object_info, robot_state)

        self.assertIsNotNone(best_grasp)
        self.assertIn(best_grasp, grasps)

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
class ManipulationIntegrationTest:
    def __init__(self):
        rospy.init_node('manipulation_integration_test')
        self.coordinator = PerceptionActionCoordinator()

    def test_detection_to_grasp_pipeline(self):
        """Test complete pipeline from detection to grasp"""
        # This would require a real robot or simulation
        # For now, we'll test the individual components

        print("Testing detection to grasp pipeline...")

        # Simulate having an image
        if self.coordinator.current_rgb is not None:
            # Test detection
            detections = self.coordinator.object_detector.detect_objects(
                self.coordinator.current_rgb
            )
            print(f"Detected {len(detections)} objects")

            if detections:
                # Test pose estimation (if depth available)
                if self.coordinator.current_depth is not None:
                    best_detection = detections[0]  # Use first detection
                    object_pose = self.coordinator.pose_estimator.estimate_pose_3d(
                        self.coordinator.current_rgb,
                        self.coordinator.current_depth,
                        best_detection
                    )
                    print(f"Estimated pose: {object_pose}")

                    if object_pose:
                        # Test grasp planning
                        object_dims = self.coordinator._estimate_object_dimensions(best_detection)
                        grasp_candidates = self.coordinator.grasp_planner.plan_grasps(
                            object_pose, object_dims, best_detection['class_name']
                        )
                        print(f"Generated {len(grasp_candidates)} grasp candidates")

                        if grasp_candidates:
                            # Test grasp optimization
                            best_grasp = self.coordinator.grasp_optimizer.select_best_grasp(
                                grasp_candidates,
                                {'object_class': best_detection['class_name'], 'dimensions': object_dims},
                                self.coordinator._get_robot_state()
                            )
                            print(f"Selected best grasp with score: {best_grasp.score if best_grasp else 'None'}")

        print("Pipeline test completed")

    def test_pick_and_place_task(self):
        """Test pick and place task execution"""
        # This would test the ManipulationTaskExecutor
        task_executor = ManipulationTaskExecutor()

        # Example task (would need real object and locations in practice)
        # result = task_executor.pick_and_place_task('cup',
        #                                          {'x': 0.5, 'y': 0.0, 'z': 0.0},
        #                                          {'x': 0.8, 'y': 0.0, 'z': 0.0})
        # print(f"Pick and place result: {result}")

        print("Pick and place task test completed (simulation)")
```

## Performance Benchmarks

### Detection Performance

- **Object Detection**: < 100ms per frame for 640x480 images
- **Pose Estimation**: < 200ms per object with depth information
- **Grasp Planning**: < 50ms per grasp candidate
- **Manipulation Execution**: < 10s for complete pick/place operation
- **Memory Usage**: < 2GB for full perception pipeline

### Accuracy Requirements

- **Object Detection**: > 85% accuracy for known objects
- **Pose Estimation**: < 2cm position error, < 10° orientation error
- **Grasp Success Rate**: > 80% for common objects
- **Placement Accuracy**: < 3cm error from target location

## Troubleshooting and Common Issues

### Detection Problems

1. **False Positives**: Adjust confidence thresholds and use NMS
2. **Missed Objects**: Improve lighting conditions and camera calibration
3. **Pose Inaccuracy**: Use multiple views and improve depth quality
4. **Real-time Performance**: Optimize model size or use edge computing

### Manipulation Problems

1. **Grasp Failures**: Improve grasp planning and object property knowledge
2. **Collision Detection**: Enhance environment modeling and path planning
3. **Object Slippage**: Adjust gripper force and improve grasp type selection
4. **Kinematic Singularities**: Implement redundancy resolution

## Best Practices

### Safety Considerations

- **Force Limiting**: Always limit gripper and arm forces
- **Collision Avoidance**: Check for collisions before movement
- **Emergency Stop**: Implement immediate stop capabilities
- **Workspace Limits**: Respect physical workspace boundaries

### Performance Optimization

- **Model Efficiency**: Use optimized neural network models
- **Multi-threading**: Separate perception and action threads
- **Caching**: Cache frequently computed values
- **Adaptive Resolution**: Adjust processing based on requirements

### Maintainability

- **Modular Design**: Keep perception, planning, and control separate
- **Parameter Configuration**: Use ROS parameters for easy tuning
- **Comprehensive Logging**: Log all manipulation decisions and outcomes
- **Testing Framework**: Maintain extensive test coverage

## Next Steps and Integration

### Integration with Other Capstone Components

The manipulation system integrates with:
- **Navigation**: Coordinates for precise positioning
- **Task Planning**: Receives manipulation goals and reports status
- **Perception**: Gets object information and environmental data
- **Voice Processing**: Handles object-related commands
- **Localization**: Maintains spatial relationships

### Advanced Features

Consider implementing:
- **Tactile Feedback**: Use tactile sensors for better grasp control
- **Learning from Demonstration**: Learn new manipulation skills from human examples
- **Multi-object Manipulation**: Handle multiple objects simultaneously
- **Deformable Object Manipulation**: Handle cloth, rope, and other deformable objects

Continue with [Failure Handling and Status Reporting](./failure-handling.md) to explore the implementation of robust error handling and system monitoring capabilities that ensure the autonomous humanoid system operates safely and reliably.

## References

[All sources will be cited in the References section at the end of the book, following APA format]