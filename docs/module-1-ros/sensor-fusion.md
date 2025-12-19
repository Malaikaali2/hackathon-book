---
sidebar_position: 6
---

# Sensor Fusion Implementation Guide

## Overview

Sensor fusion is the process of combining data from multiple sensors to achieve improved accuracy and reliability compared to using a single sensor alone. In robotics, sensor fusion is critical for navigation, localization, mapping, and environmental perception.

## Key Concepts

### Types of Sensor Fusion

1. **Data-Level Fusion**: Combining raw sensor measurements
2. **Feature-Level Fusion**: Combining extracted features from sensors
3. **Decision-Level Fusion**: Combining decisions from individual sensors
4. **Hybrid Fusion**: Combining multiple fusion levels

### Common Sensor Types in Robotics

- **IMU (Inertial Measurement Unit)**: Provides acceleration, angular velocity, and orientation
- **GPS**: Provides global position (outdoor environments)
- **LIDAR**: Provides precise distance measurements for mapping and obstacle detection
- **Cameras**: Provide visual information for object recognition and scene understanding
- **Encoders**: Provide wheel rotation data for odometry
- **Ultrasonic sensors**: Provide short-range distance measurements

## Mathematical Foundations

### Kalman Filter

The Kalman filter is a mathematical method for estimating the state of a system from noisy measurements. It's optimal for linear systems with Gaussian noise.

**Prediction Step:**
- Predict state: `x_pred = F * x_prev + B * u`
- Predict covariance: `P_pred = F * P_prev * F^T + Q`

**Update Step:**
- Compute Kalman gain: `K = P_pred * H^T * (H * P_pred * H^T + R)^-1`
- Update state: `x_new = x_pred + K * (z - H * x_pred)`
- Update covariance: `P_new = (I - K * H) * P_pred`

### Extended Kalman Filter (EKF)

For non-linear systems, the Extended Kalman Filter linearizes the system around the current estimate.

### Particle Filter

A particle filter represents the probability distribution with a set of particles that evolve over time based on system dynamics and sensor measurements.

## ROS 2 Sensor Fusion Packages

### Robot Localization Package

The `robot_localization` package provides sensor fusion for robot state estimation using an EKF or UKF (Unscented Kalman Filter).

#### Configuration Example

```yaml
# ekf.yaml
ekf_filter_node:
  ros__parameters:
    # The frequency, in Hz, at which the filter will output a position estimate
    frequency: 30.0

    # The period, in seconds, at which the filter will output a position estimate
    sensor_timeout: 0.1

    # Whether to two-dimensional or not
    two_d_mode: true

    # Whether to broadcast the transform between the input and output frames
    transform_time_offset: 0.0

    # Whether to publish the acceleration state
    publish_acceleration: false

    # Whether to broadcast the transform between the input and output frames
    publish_tf: true

    # Map frame
    map_frame: map

    # Odometry frame (world-fixed)
    odom_frame: odom

    # Base frame (robot-fixed)
    base_link_frame: base_link

    # World frame for the transform between the input and output frames
    world_frame: odom

    # Sensor configuration
    odom0: /wheel/odometry
    odom0_config: [true,  true,  false,
                   false, false, false,
                   false, false, false,
                   false, false, true,
                   false, false, false]
    odom0_differential: false
    odom0_relative: false

    imu0: /imu/data
    imu0_config: [false, false, false,
                  true,  true,  true,
                  false, false, false,
                  true,  true,  true,
                  false, false, false]
    imu0_differential: false
    imu0_relative: true
    imu0_queue_size: 10
    imu0_pose_rejection_threshold: 0.8
    imu0_twist_rejection_threshold: 0.8
    imu0_linear_acceleration_rejection_threshold: 0.8
```

## Implementation Example: IMU and Odometry Fusion

### Creating a Sensor Fusion Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusionNode(Node):

    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            'imu/data',
            self.imu_callback,
            10)

        # Subscribe to odometry data
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)

        # Publisher for fused pose
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            'fused_pose',
            10)

        # Initialize state variables
        self.imu_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion
        self.odom_position = np.array([0.0, 0.0, 0.0])
        self.odom_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion
        self.fused_pose = PoseWithCovarianceStamped()

        # Covariance matrices (simplified)
        self.imu_cov = np.eye(4) * 0.01  # IMU covariance
        self.odom_cov = np.eye(4) * 0.1   # Odometry covariance

        self.get_logger().info('Sensor fusion node initialized')

    def imu_callback(self, msg):
        """Process IMU data"""
        # Extract orientation from IMU
        self.imu_orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Update covariance from IMU
        self.imu_cov = np.array(msg.orientation_covariance).reshape(3, 3)

    def odom_callback(self, msg):
        """Process odometry data and perform fusion"""
        # Extract position and orientation from odometry
        self.odom_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        self.odom_orientation = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        # Perform sensor fusion (simplified example using weighted average)
        fused_orientation = self.fuse_orientations(
            self.imu_orientation, self.odom_orientation
        )

        # Create and publish fused pose
        self.fused_pose.header.stamp = self.get_clock().now().to_msg()
        self.fused_pose.header.frame_id = 'map'

        # Set position from odometry (usually more reliable for position)
        self.fused_pose.pose.pose.position.x = msg.pose.pose.position.x
        self.fused_pose.pose.pose.position.y = msg.pose.pose.position.y
        self.fused_pose.pose.pose.position.z = msg.pose.pose.position.z

        # Set orientation from fusion
        self.fused_pose.pose.pose.orientation.x = fused_orientation[0]
        self.fused_pose.pose.pose.orientation.y = fused_orientation[1]
        self.fused_pose.pose.pose.orientation.z = fused_orientation[2]
        self.fused_pose.pose.pose.orientation.w = fused_orientation[3]

        # Set covariance (simplified)
        self.fused_pose.pose.covariance = self.calculate_fused_covariance()

        self.pose_pub.publish(self.fused_pose)

    def fuse_orientations(self, imu_quat, odom_quat, imu_weight=0.7, odom_weight=0.3):
        """
        Fuse orientations using weighted average of quaternions
        """
        # Normalize quaternions
        imu_quat = imu_quat / np.linalg.norm(imu_quat)
        odom_quat = odom_quat / np.linalg.norm(odom_quat)

        # Weighted average of quaternions
        fused_quat = imu_weight * imu_quat + odom_weight * odom_quat
        fused_quat = fused_quat / np.linalg.norm(fused_quat)

        return fused_quat

    def calculate_fused_covariance(self):
        """
        Calculate fused covariance matrix
        """
        # Simplified covariance fusion (in practice, use proper fusion methods)
        fused_cov = [0.0] * 36
        # Fill diagonal with some reasonable values
        fused_cov[0] = 0.01  # x position variance
        fused_cov[7] = 0.01  # y position variance
        fused_cov[14] = 0.01  # z position variance
        fused_cov[21] = 0.005  # x orientation variance
        fused_cov[28] = 0.005  # y orientation variance
        fused_cov[35] = 0.005  # z orientation variance

        return fused_cov

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()

    try:
        rclpy.spin(sensor_fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Fusion Techniques

### Multi-Sensor Fusion with LIDAR and Camera

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np

class MultiSensorFusionNode(Node):

    def __init__(self):
        super().__init__('multi_sensor_fusion_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Subscribe to different sensor types
        self.lidar_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10)

        self.camera_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10)

        # Publisher for fused data
        self.fused_pub = self.create_publisher(
            PointCloud2,  # Could be custom message type
            'fused_sensor_data',
            10)

        # Storage for sensor data
        self.lidar_data = None
        self.camera_image = None
        self.camera_timestamp = None
        self.lidar_timestamp = None

    def lidar_callback(self, msg):
        """Process LIDAR data"""
        self.lidar_data = msg.ranges
        self.lidar_timestamp = msg.header.stamp

        # If we have synchronized data, perform fusion
        if self.camera_image is not None:
            self.perform_lidar_camera_fusion()

    def camera_callback(self, msg):
        """Process camera data"""
        self.camera_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.camera_timestamp = msg.header.stamp

        # If we have synchronized data, perform fusion
        if self.lidar_data is not None:
            self.perform_lidar_camera_fusion()

    def perform_lidar_camera_fusion(self):
        """Perform fusion between LIDAR and camera data"""
        # This is a simplified example
        # In practice, you'd need calibration data and projection matrices

        # Project LIDAR points to camera frame
        # This requires extrinsic calibration between sensors
        projected_points = self.project_lidar_to_camera(
            self.lidar_data
        )

        # Combine with camera image features
        combined_data = self.combine_lidar_camera(
            projected_points,
            self.camera_image
        )

        # Publish fused result
        # In a real implementation, you'd create a proper message
        self.get_logger().info(f'Fused {len(projected_points)} LIDAR points with camera image')

    def project_lidar_to_camera(self, lidar_ranges):
        """Project LIDAR ranges to camera coordinate system"""
        # This would require:
        # 1. LIDAR to camera extrinsic calibration
        # 2. Camera intrinsic parameters
        # 3. Mathematical transformation

        # Simplified example - in reality, this would be complex
        points_3d = []
        for i, range_val in enumerate(lidar_ranges):
            if not np.isnan(range_val) and range_val > 0:
                # Convert polar to Cartesian coordinates
                angle = i * 0.01  # Assuming 0.01 radian increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                z = 0  # Assuming 2D LIDAR
                points_3d.append([x, y, z])

        return points_3d

    def combine_lidar_camera(self, lidar_points, camera_image):
        """Combine LIDAR points with camera image"""
        # In practice, this might involve:
        # - Colorizing LIDAR points based on camera image
        # - Object detection using both modalities
        # - Creating enhanced point clouds

        # For this example, we'll just return the data
        return {
            'lidar_points': lidar_points,
            'image_shape': camera_image.shape
        }

def main(args=None):
    rclpy.init(args=args)
    fusion_node = MultiSensorFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Fusion Best Practices

### 1. Data Synchronization
- Use ROS 2 message filters for time synchronization
- Implement proper timestamp handling
- Consider sensor delays and compensate accordingly

### 2. Calibration
- Perform extrinsic calibration between sensors
- Regularly validate calibration parameters
- Account for sensor mounting offsets

### 3. Covariance Management
- Properly set covariance matrices
- Update covariances based on sensor conditions
- Consider sensor-specific noise characteristics

### 4. Fault Detection
- Implement sensor health monitoring
- Detect and handle sensor failures gracefully
- Provide fallback mechanisms

### 5. Computational Efficiency
- Optimize algorithms for real-time performance
- Consider sensor data rates
- Implement data decimation when appropriate

## Common Fusion Algorithms

### 1. Kalman Filter Family
- **EKF**: Good for non-linear systems with moderate non-linearity
- **UKF**: Better for highly non-linear systems
- **Particle Filter**: Good for multi-modal distributions

### 2. Complementary Filter
- Simple fusion of sensors with complementary characteristics
- Good for IMU and other sensors with different frequency responses

### 3. Covariance Intersection
- Handles correlated uncertainties between sensors
- Useful when cross-covariances are unknown

## Performance Evaluation

### Metrics for Sensor Fusion
- **Accuracy**: How close the fused estimate is to the true value
- **Precision**: Consistency of the fused estimate
- **Robustness**: Performance under sensor failures or degraded conditions
- **Latency**: Time delay introduced by fusion process

### Testing Approaches
- Use ground truth data when available
- Cross-validation with different sensor subsets
- Stress testing under various environmental conditions

## References

[All sources will be cited in the References section at the end of the book, following APA format]