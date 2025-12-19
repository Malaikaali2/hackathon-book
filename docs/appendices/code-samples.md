---
sidebar_position: 33
---

# Code Samples Reference

## Overview

This comprehensive code samples reference provides practical implementations and examples for the concepts discussed throughout the Physical AI and Humanoid Robotics book. Each code sample is designed to be educational, practical, and directly applicable to real-world robotics development. The samples cover various aspects of robotics development including perception, planning, control, simulation, and deployment.

All code samples are provided in multiple languages where applicable (C++ and Python) and follow best practices for robotics development. Each sample includes detailed comments, error handling, and is designed to be easily integrated into larger robotics systems.

## ROS/ROS2 Fundamentals

### Basic Publisher/Subscriber Pattern

#### C++ Publisher Example
```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher() : Node("minimal_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello World: " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalPublisher>());
    rclcpp::shutdown();
    return 0;
}
```

#### Python Publisher Example
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### C++ Subscriber Example
```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class MinimalSubscriber : public rclcpp::Node
{
public:
    MinimalSubscriber() : Node("minimal_subscriber")
    {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic", 10,
            std::bind(&MinimalSubscriber::topic_callback, this, std::placeholders::_1));
    }

private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}
```

## Perception Systems

### Camera Data Processing

#### RGB-D Camera Processing
```python
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped

class RGBDProcessor(Node):
    def __init__(self):
        super().__init__('rgbd_processor')

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.info_callback, 10)

        # Store camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Processed data storage
        self.rgb_image = None
        self.depth_image = None

    def info_callback(self, msg):
        """Store camera intrinsic parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def rgb_callback(self, msg):
        """Process RGB image"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting RGB image: {e}')

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            # Convert depth image to meters
            depth_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)

            # Convert to float32 and scale to meters
            if depth_image.dtype == np.uint16:
                depth_image = depth_image.astype(np.float32) / 1000.0  # mm to meters
            elif depth_image.dtype == np.uint8:
                depth_image = depth_image.astype(np.float32)

            self.depth_image = depth_image

            # Process the combined RGB-D data
            if self.rgb_image is not None and self.depth_image is not None:
                self.process_rgbd_data()

        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')

    def process_rgbd_data(self):
        """Process combined RGB-D data for object detection"""
        # Example: Find objects in the scene
        height, width = self.rgb_image.shape[:2]

        # Convert to HSV for color-based segmentation
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)

        # Define color range for object detection (example: red objects)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Apply depth mask to filter objects at certain distances
        if self.depth_image is not None:
            # Filter objects within 1-2 meters
            depth_mask = (self.depth_image >= 1.0) & (self.depth_image <= 2.0)
            mask = mask & depth_mask

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process detected objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate center in 3D space
                center_x = x + w // 2
                center_y = y + h // 2

                if center_x < self.depth_image.shape[1] and center_y < self.depth_image.shape[0]:
                    depth = self.depth_image[center_y, center_x]

                    if depth > 0:  # Valid depth
                        # Convert pixel coordinates to 3D world coordinates
                        world_x, world_y, world_z = self.pixel_to_world(
                            center_x, center_y, depth)

                        self.get_logger().info(
                            f'Detected object at: ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})')

    def pixel_to_world(self, u, v, depth):
        """Convert pixel coordinates to world coordinates"""
        if self.camera_matrix is None:
            return 0, 0, depth

        # Camera intrinsic parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Convert to world coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return x, y, z

def main(args=None):
    rclpy.init(args=args)
    processor = RGBDProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### LiDAR Processing

#### Point Cloud Processing
```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/concave_hull.h>

class PointCloudProcessor : public rclcpp::Node
{
public:
    PointCloudProcessor() : Node("pointcloud_processor")
    {
        // Create subscriber for point cloud
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10,
            std::bind(&PointCloudProcessor::pointcloud_callback, this, std::placeholders::_1));

        // Create publisher for processed point cloud
        processed_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/processed_points", 10);

        // Create publisher for ground plane
        ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/ground_points", 10);
    }

private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert ROS message to PCL
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);

        // Convert to PointXYZ
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

        // Downsample the point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(cloud);
        voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
        voxel_filter.filter(*cloud_filtered);

        // Segment ground plane using SAC segmentation
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        // Create segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(100);
        seg.setDistanceThreshold(0.2); // 20cm threshold

        // Segment the largest planar component
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        // Extract ground points and non-ground points
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);

        // Extract ground
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*ground_cloud);

        // Extract non-ground
        extract.setNegative(true);
        extract.filter(*cloud_no_ground);

        // Publish ground points
        sensor_msgs::msg::PointCloud2 ground_msg;
        pcl::toPCLPointCloud2(*ground_cloud, pcl_pc2);
        pcl_conversions::fromPCL(pcl_pc2, ground_msg);
        ground_msg.header = msg->header;
        ground_pub_->publish(ground_msg);

        // Publish processed (non-ground) points
        sensor_msgs::msg::PointCloud2 processed_msg;
        pcl::toPCLPointCloud2(*cloud_no_ground, pcl_pc2);
        pcl_conversions::fromPCL(pcl_pc2, processed_msg);
        processed_msg.header = msg->header;
        processed_pub_->publish(processed_msg);

        RCLCPP_INFO(this->get_logger(),
            "Processed point cloud: %zu points, %zu ground points",
            cloud_no_ground->size(), ground_cloud->size());
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr processed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudProcessor>());
    rclcpp::shutdown();
    return 0;
}
```

## Navigation and Path Planning

### A* Path Planning Algorithm

```python
import numpy as np
import heapq
from typing import List, Tuple, Optional

class AStarPlanner:
    def __init__(self, occupancy_grid: np.ndarray, resolution: float = 1.0):
        """
        Initialize A* path planner

        Args:
            occupancy_grid: 2D numpy array where 0=free, 1=occupied
            resolution: Size of each grid cell in meters
        """
        self.grid = occupancy_grid
        self.resolution = resolution
        self.height, self.width = occupancy_grid.shape

        # 8-directional movement (including diagonals)
        self.movements = [
            (-1, -1), (-1, 0), (-1, 1),  # Up-left, Up, Up-right
            (0, -1),           (0, 1),   # Left, Right
            (1, -1),  (1, 0),  (1, 1)    # Down-left, Down, Down-right
        ]

        # Movement costs (diagonal = sqrt(2), straight = 1)
        self.movement_costs = [
            np.sqrt(2), 1, np.sqrt(2),  # Up-left, Up, Up-right
            1,          1,              # Left, Right
            np.sqrt(2), 1, np.sqrt(2)   # Down-left, Down, Down-right
        ]

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Euclidean) between two points"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def is_valid(self, x: int, y: int) -> bool:
        """Check if coordinates are valid and not occupied"""
        return (0 <= x < self.width and
                0 <= y < self.height and
                self.grid[y, x] == 0)  # 0 means free space

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using A* algorithm

        Args:
            start: Start position (x, y) in meters
            goal: Goal position (x, y) in meters

        Returns:
            List of (x, y) coordinates in meters, or None if no path found
        """
        # Convert meters to grid coordinates
        start_grid = (int(start[0] / self.resolution), int(start[1] / self.resolution))
        goal_grid = (int(goal[0] / self.resolution), int(goal[1] / self.resolution))

        # Validate start and goal positions
        if not self.is_valid(start_grid[0], start_grid[1]) or not self.is_valid(goal_grid[0], goal_grid[1]):
            print("Start or goal position is occupied or out of bounds")
            return None

        # Initialize open and closed sets
        open_set = [(0, start_grid)]  # (f_score, (x, y))
        heapq.heapify(open_set)

        came_from = {}  # For path reconstruction
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current)
                # Convert grid coordinates back to meters
                path_meters = [(x * self.resolution, y * self.resolution) for x, y in path]
                return path_meters

            for i, (dx, dy) in enumerate(self.movements):
                neighbor = (current[0] + dx, current[1] + dy)

                if not self.is_valid(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = g_score[current] + self.movement_costs[i]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("No path found")
        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

# Example usage
def example_usage():
    # Create a simple occupancy grid (0 = free, 1 = occupied)
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    planner = AStarPlanner(grid, resolution=0.5)

    start = (0.5, 0.5)  # Start at (0.5m, 0.5m)
    goal = (4.5, 9.5)   # Goal at (4.5m, 9.5m)

    path = planner.plan(start, goal)

    if path:
        print(f"Path found with {len(path)} waypoints:")
        for i, (x, y) in enumerate(path):
            print(f"  Waypoint {i}: ({x:.2f}, {y:.2f})")
    else:
        print("No path found")

if __name__ == "__main__":
    example_usage()
```

### DWA (Dynamic Window Approach) Local Planner

```python
import numpy as np
from typing import Tuple, List
import math

class DWAPlanner:
    def __init__(self, robot_radius: float = 0.3, max_speed: float = 1.0, min_speed: float = -0.5):
        """
        Initialize Dynamic Window Approach local planner

        Args:
            robot_radius: Robot radius for collision checking
            max_speed: Maximum linear speed
            min_speed: Minimum linear speed (can be negative for backward motion)
        """
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_yaw_rate = np.pi / 3  # Maximum angular velocity (60 degrees/s)
        self.max_accel = 1.0  # Maximum acceleration
        self.max_delta_yaw_rate = np.pi / 6  # Maximum angular acceleration
        self.dt = 0.1  # Time step for simulation
        self.predict_time = 3.0  # Prediction horizon
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0

    def plan(self, state: np.ndarray, goal: np.ndarray, obstacle_list: List[np.ndarray]) -> Tuple[float, float]:
        """
        Plan local trajectory using DWA

        Args:
            state: Current state [x, y, yaw, v, omega]
            goal: Goal position [x, y]
            obstacle_list: List of obstacles [[x, y], ...]

        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        # Calculate dynamic window
        window = self.calc_dynamic_window(state)

        # Evaluate trajectories in the window
        best_traj = None
        best_score = float('inf')

        # Discretize the search space
        v_samples = np.linspace(window[0], window[1], 10)
        omega_samples = np.linspace(window[2], window[3], 10)

        for v in v_samples:
            for omega in omega_samples:
                # Simulate trajectory
                traj = self.predict_trajectory(state, v, omega)

                # Calculate costs
                to_goal_cost = self.calc_to_goal_cost(traj, goal)
                speed_cost = self.calc_speed_cost(traj)
                obstacle_cost = self.calc_obstacle_cost(traj, obstacle_list)

                # Total cost (lower is better)
                total_cost = (self.to_goal_cost_gain * to_goal_cost +
                             self.speed_cost_gain * speed_cost +
                             self.obstacle_cost_gain * obstacle_cost)

                if total_cost < best_score:
                    best_score = total_cost
                    best_traj = [v, omega]

        if best_traj is None:
            # If no valid trajectory found, stop
            return 0.0, 0.0

        return best_traj[0], best_traj[1]

    def calc_dynamic_window(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate dynamic window based on current state and constraints

        Args:
            state: Current state [x, y, yaw, v, omega]

        Returns:
            Dynamic window [v_min, v_max, omega_min, omega_max]
        """
        vs = np.array([self.min_speed, self.max_speed,
                      -self.max_yaw_rate, self.max_yaw_rate])

        vd = np.array([state[3] - self.max_accel * self.dt,
                      state[3] + self.max_accel * self.dt,
                      state[4] - self.max_delta_yaw_rate * self.dt,
                      state[4] + self.max_delta_yaw_rate * self.dt])

        # Dynamic window is intersection of velocity space and acceleration space
        dw = np.array([max(vs[0], vd[0]), min(vs[1], vd[1]),
                      max(vs[2], vd[2]), min(vs[3], vd[3])])
        return dw

    def predict_trajectory(self, state: np.ndarray, v: float, omega: float) -> np.ndarray:
        """
        Predict trajectory for given velocity commands

        Args:
            state: Current state [x, y, yaw, v, omega]
            v: Linear velocity
            omega: Angular velocity

        Returns:
            Trajectory array [n_steps x 4] with [x, y, yaw, v]
        """
        state = state.copy()
        trajectory = np.zeros((int(self.predict_time / self.dt), 4))

        for i in range(len(trajectory)):
            state = self.motion(state, v, omega)
            trajectory[i] = state[:4]  # x, y, yaw, v

        return trajectory

    def motion(self, state: np.ndarray, v: float, omega: float) -> np.ndarray:
        """
        Motion model for robot

        Args:
            state: Current state [x, y, yaw, v, omega]
            v: Linear velocity
            omega: Angular velocity

        Returns:
            New state after dt time
        """
        state[0] += v * math.cos(state[2]) * self.dt  # x
        state[1] += v * math.sin(state[2]) * self.dt  # y
        state[2] += omega * self.dt  # yaw
        state[3] = v  # v
        state[4] = omega  # omega
        return state

    def calc_to_goal_cost(self, traj: np.ndarray, goal: np.ndarray) -> float:
        """Calculate cost related to distance to goal"""
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - traj[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        return cost

    def calc_speed_cost(self, traj: np.ndarray) -> float:
        """Calculate cost related to speed (prefer higher speeds)"""
        return abs(self.max_speed - traj[-1, 3])

    def calc_obstacle_cost(self, traj: np.ndarray, obstacle_list: List[np.ndarray]) -> float:
        """Calculate cost related to obstacles"""
        min_dist = float('inf')

        for point in traj:
            for obs in obstacle_list:
                dist = math.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                if dist <= self.robot_radius:
                    return float('inf')  # Collision
                min_dist = min(min_dist, dist)

        return 1.0 / min_dist if min_dist != float('inf') else float('inf')

# Example usage
def dwa_example():
    planner = DWAPlanner()

    # Initial state [x, y, yaw, v, omega]
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # Goal position
    goal = np.array([10.0, 10.0])

    # Obstacle positions
    obstacles = [
        np.array([5.0, 5.0]),
        np.array([7.0, 6.0]),
        np.array([3.0, 8.0])
    ]

    # Simulate navigation
    path = [state[:2].copy()]

    for _ in range(100):  # Maximum 100 steps
        v, omega = planner.plan(state, goal, obstacles)

        # Apply control and update state
        state = planner.motion(state, v, omega)
        path.append(state[:2].copy())

        # Check if reached goal
        dist_to_goal = np.linalg.norm(state[:2] - goal)
        if dist_to_goal < 0.5:  # Within 0.5m of goal
            print("Goal reached!")
            break

    print(f"Path length: {len(path)} steps")
    return np.array(path)

if __name__ == "__main__":
    path = dwa_example()
```

## Manipulation and Control

### Inverse Kinematics

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, List

class InverseKinematics:
    def __init__(self, dh_parameters: List[Tuple[float, float, float, float]]):
        """
        Initialize Inverse Kinematics solver using DH parameters

        Args:
            dh_parameters: List of DH parameters [(a, alpha, d, theta_offset), ...]
        """
        self.dh_params = dh_parameters
        self.n_joints = len(dh_params)

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Calculate forward kinematics

        Args:
            joint_angles: Joint angles in radians

        Returns:
            4x4 transformation matrix
        """
        T = np.eye(4)  # Identity matrix

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset

            # DH transformation matrix
            T_i = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])

            T = T @ T_i  # Matrix multiplication

        return T

    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Calculate geometric Jacobian matrix

        Args:
            joint_angles: Joint angles in radians

        Returns:
            6xN Jacobian matrix (linear and angular velocities)
        """
        n = len(joint_angles)
        J = np.zeros((6, n))

        # Get end-effector position and orientation
        T_end = self.forward_kinematics(joint_angles)
        end_pos = T_end[:3, 3]

        # Calculate Jacobian columns
        T_current = np.eye(4)

        for i in range(n):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset

            # Joint transformation
            T_i = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])

            # Calculate joint position
            joint_pos = T_current[:3, 3]

            # Calculate z-axis of joint frame
            z_axis = T_current[:3, 2]

            # Calculate Jacobian column
            if i < n-1:  # Not the last joint (usually rotational)
                # Linear velocity component
                J[:3, i] = np.cross(z_axis, end_pos - joint_pos)
                # Angular velocity component
                J[3:, i] = z_axis
            else:  # Last joint (could be prismatic)
                # Linear velocity component (prismatic joint)
                J[:3, i] = z_axis
                # Angular velocity component
                J[3:, i] = np.array([0, 0, 0])

            T_current = T_current @ T_i

        return J

    def inverse_kinematics(self, target_pose: np.ndarray, initial_joints: np.ndarray,
                          max_iterations: int = 1000, tolerance: float = 1e-4) -> Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics using Jacobian transpose method

        Args:
            target_pose: 4x4 target transformation matrix
            initial_joints: Initial joint angles
            max_iterations: Maximum number of iterations
            tolerance: Position/orientation tolerance

        Returns:
            Tuple of (joint_angles, success_flag)
        """
        joints = initial_joints.copy()

        for iteration in range(max_iterations):
            # Calculate current pose
            current_pose = self.forward_kinematics(joints)

            # Calculate error
            pos_error = target_pose[:3, 3] - current_pose[:3, 3]

            # Calculate orientation error (using rotation matrix difference)
            R_current = current_pose[:3, :3]
            R_target = target_pose[:3, :3]

            # Use logarithmic map for rotation error (simplified)
            R_error = R_target @ R_current.T
            rotation_error = self.rotation_matrix_to_axis_angle(R_error)

            # Combine position and orientation errors
            error = np.concatenate([pos_error, rotation_error])

            # Check convergence
            if np.linalg.norm(error) < tolerance:
                return joints, True

            # Calculate Jacobian
            J = self.jacobian(joints)

            # Update joints using Jacobian transpose method
            # Note: For better convergence, consider using damped least squares
            joints += 0.1 * J.T @ error  # Learning rate of 0.1

            # Apply joint limits if needed
            # joints = np.clip(joints, joint_limits_min, joint_limits_max)

        return joints, False  # Failed to converge

    def rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle representation"""
        # Extract angle
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

        if angle < 1e-6:  # Small angle approximation
            return np.array([0, 0, 0])

        # Extract axis
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))

        return angle * axis

# Example usage for a simple 3-DOF planar manipulator
def ik_example():
    # Define DH parameters for a 3-DOF planar manipulator
    # [a, alpha, d, theta_offset]
    dh_params = [
        (0.5, 0, 0, 0),    # Joint 1
        (0.4, 0, 0, 0),    # Joint 2
        (0.3, 0, 0, 0)     # Joint 3
    ]

    ik_solver = InverseKinematics(dh_params)

    # Target pose (simplified - just position in 2D plane)
    target_pose = np.eye(4)
    target_pose[0, 3] = 0.8  # x position
    target_pose[1, 3] = 0.6  # y position
    target_pose[2, 3] = 0.0  # z position

    # Initial joint angles
    initial_joints = np.array([0.0, 0.0, 0.0])

    # Solve inverse kinematics
    solution, success = ik_solver.inverse_kinematics(target_pose, initial_joints)

    if success:
        print(f"Solution found: {solution}")

        # Verify solution
        final_pose = ik_solver.forward_kinematics(solution)
        print(f"Final position: [{final_pose[0,3]:.3f}, {final_pose[1,3]:.3f}]")
        print(f"Target position: [{target_pose[0,3]:.3f}, {target_pose[1,3]:.3f}]")
    else:
        print("Failed to find solution")

if __name__ == "__main__":
    ik_example()
```

## AI and Machine Learning Integration

### Vision-Language-Action (VLA) System

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from transformers import CLIPProcessor, CLIPModel
import openai

class VisionLanguageActionSystem(nn.Module):
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize Vision-Language-Action system

        Args:
            clip_model_name: Name of the CLIP model to use
        """
        super().__init__()

        # Load pre-trained CLIP model for vision-language understanding
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Action prediction head
        self.action_predictor = nn.Sequential(
            nn.Linear(512, 256),  # CLIP vision embedding dimension
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),   # Action space dimension
            nn.Tanh()             # Normalize to [-1, 1] range
        )

        # Task planning module
        self.task_planner = nn.Sequential(
            nn.Linear(512 + 64, 256),  # CLIP text + action embedding
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)    # Task sequence length
        )

    def forward(self, images: torch.Tensor, text: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VLA system

        Args:
            images: Batch of images [B, C, H, W]
            text: List of text descriptions

        Returns:
            Dictionary with predictions
        """
        # Process images through CLIP vision encoder
        image_features = self.clip_model.get_image_features(pixel_values=images)

        # Process text through CLIP text encoder
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**inputs)

        # Predict actions based on visual input
        action_predictions = self.action_predictor(image_features)

        # Plan task sequence combining vision and language
        combined_features = torch.cat([image_features, action_predictions], dim=1)
        task_sequence = self.task_planner(combined_features)

        return {
            'action_predictions': action_predictions,
            'task_sequence': task_sequence,
            'image_features': image_features,
            'text_features': text_features
        }

    def process_command(self, image: np.ndarray, command: str) -> Dict[str, any]:
        """
        Process a natural language command with visual input

        Args:
            image: Input image (BGR format from OpenCV)
            command: Natural language command

        Returns:
            Dictionary with action and task information
        """
        # Convert image to PIL format for CLIP
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Process through VLA system
        inputs = self.clip_processor(
            text=[command],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.forward(
                images=inputs['pixel_values'],
                text=[command]
            )

        # Extract action predictions
        action_raw = outputs['action_predictions'][0].numpy()

        # Convert to meaningful robot actions
        robot_action = self.convert_to_robot_action(action_raw, command)

        return {
            'command': command,
            'robot_action': robot_action,
            'action_vector': action_raw,
            'confidence': float(torch.nn.functional.softmax(outputs['task_sequence'][0], dim=0).max())
        }

    def convert_to_robot_action(self, action_vector: np.ndarray, command: str) -> Dict[str, any]:
        """
        Convert action vector to meaningful robot actions

        Args:
            action_vector: Raw action vector from network
            command: Original command for context

        Returns:
            Dictionary with robot action parameters
        """
        # Normalize action vector to meaningful ranges
        # Assuming action_vector contains [dx, dy, dz, rx, ry, rz, gripper] for manipulator
        position_delta = action_vector[:3] * 0.5  # Scale to max 0.5m movement
        rotation_delta = action_vector[3:6] * 0.5  # Scale to max 0.5 rad rotation
        gripper_action = action_vector[6] if len(action_vector) > 6 else 0.0

        # Determine action type based on command
        if 'pick' in command.lower() or 'grasp' in command.lower():
            action_type = 'grasp'
        elif 'move' in command.lower() or 'go' in command.lower():
            action_type = 'navigate'
        elif 'place' in command.lower() or 'put' in command.lower():
            action_type = 'place'
        else:
            action_type = 'custom'

        return {
            'type': action_type,
            'position_delta': position_delta.tolist(),
            'rotation_delta': rotation_delta.tolist(),
            'gripper_action': float(gripper_action),
            'command_context': command
        }

# Advanced VLA with memory and planning
class AdvancedVLA(nn.Module):
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()

        # Base VLA system
        self.vla_base = VisionLanguageActionSystem(clip_model_name)

        # Memory module for context
        self.memory_encoder = nn.LSTM(512, 256, batch_first=True)
        self.context_attention = nn.MultiheadAttention(256, 8)

        # Sequential planning
        self.sequential_planner = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Sequence of 64 steps
        )

    def forward_with_memory(self, images: torch.Tensor, text: List[str],
                           previous_states: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with memory of previous states
        """
        # Get base VLA outputs
        base_outputs = self.vla_base(images, text)

        # Process memory if available
        if previous_states is not None:
            memory_out, _ = self.memory_encoder(previous_states)
            # Apply attention to focus on relevant memory
            attended_memory, _ = self.context_attention(
                base_outputs['image_features'].unsqueeze(0),
                memory_out, memory_out
            )
            attended_memory = attended_memory.squeeze(0)
        else:
            attended_memory = torch.zeros_like(base_outputs['image_features'])

        # Plan sequential actions
        combined_features = torch.cat([
            base_outputs['image_features'],
            attended_memory
        ], dim=1)

        sequential_plan = self.sequential_planner(combined_features)

        base_outputs['sequential_plan'] = sequential_plan
        base_outputs['memory_attention'] = attended_memory

        return base_outputs

# Example usage
def vla_example():
    # Initialize VLA system
    vla_system = VisionLanguageActionSystem()

    # Example: Process an image with a command
    # In practice, you would load a real image
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    command = "Pick up the red cup on the table"

    result = vla_system.process_command(sample_image, command)

    print(f"Command: {result['command']}")
    print(f"Action type: {result['robot_action']['type']}")
    print(f"Position delta: {result['robot_action']['position_delta']}")
    print(f"Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    # Import PIL for image processing
    from PIL import Image

    vla_example()
```

## Simulation Integration

### Gazebo ROS2 Interface

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs
import numpy as np
import math

class GazeboRobotInterface(Node):
    def __init__(self):
        super().__init__('gazebo_robot_interface')

        # Robot state
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.velocity = np.array([0.0, 0.0])        # linear, angular

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing odometry
        self.timer = self.create_timer(0.1, self.publish_odometry)  # 10 Hz

        self.get_logger().info('Gazebo Robot Interface initialized')

    def scan_callback(self, msg):
        """Process laser scan data"""
        # Example: Find minimum distance to obstacles
        min_distance = min(msg.ranges)

        if min_distance < 0.5:  # Less than 0.5m to obstacle
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')

    def image_callback(self, msg):
        """Process camera image data"""
        # Image processing would happen here
        # For now, just log that we received an image
        self.get_logger().info(f'Received image: {msg.width}x{msg.height}')

    def publish_odometry(self):
        """Publish odometry data"""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        # Set position
        msg.pose.pose.position.x = float(self.position[0])
        msg.pose.pose.position.y = float(self.position[1])
        msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        from tf_transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, self.position[2])
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        # Set velocities
        msg.twist.twist.linear.x = float(self.velocity[0])
        msg.twist.twist.angular.z = float(self.velocity[1])

        self.odom_pub.publish(msg)

        # Broadcast transform
        from geometry_msgs.msg import TransformStamped
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = float(self.position[0])
        t.transform.translation.y = float(self.position[1])
        t.transform.translation.z = 0.0

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def move_robot(self, linear_vel, angular_vel):
        """Send velocity commands to robot"""
        msg = Twist()
        msg.linear.x = linear_vel
        msg.angular.z = angular_vel

        self.cmd_vel_pub.publish(msg)

        # Update internal state (simplified)
        dt = 0.1  # Time step from timer
        self.position[0] += linear_vel * math.cos(self.position[2]) * dt
        self.position[1] += linear_vel * math.sin(self.position[2]) * dt
        self.position[2] += angular_vel * dt

        # Normalize angle
        self.position[2] = math.atan2(
            math.sin(self.position[2]),
            math.cos(self.position[2])
        )

        self.velocity[0] = linear_vel
        self.velocity[1] = angular_vel

def main(args=None):
    rclpy.init(args=args)
    interface = GazeboRobotInterface()

    # Example: Move robot in a square pattern
    def move_square():
        interface.get_logger().info('Moving robot in square pattern')

        # Define square movement: move forward, turn 90 degrees, repeat
        for i in range(4):
            interface.get_logger().info(f'Moving side {i+1} of square')

            # Move forward for 2 seconds
            for _ in range(20):  # 20 iterations * 0.1s = 2 seconds
                interface.move_robot(0.5, 0.0)  # 0.5 m/s linear
                rclpy.spin_once(interface, timeout_sec=0.1)

            # Turn 90 degrees
            interface.get_logger().info('Turning 90 degrees')
            for _ in range(15):  # Turn for 1.5 seconds (adjust as needed)
                interface.move_robot(0.0, math.pi/2/1.5)  # Angular velocity for 90 deg in 1.5s
                rclpy.spin_once(interface, timeout_sec=0.1)

    # Run the square movement
    move_square()

    interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Hardware Integration

### Jetson AI Inference

```python
import jetson.inference
import jetson.utils
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

class JetsonAIProcessor:
    def __init__(self, model_path: str = "ssd-mobilenet-v2", input_size: Tuple[int, int] = (300, 300)):
        """
        Initialize AI processor for Jetson platform

        Args:
            model_path: Path to the detection model
            input_size: Input size for the model
        """
        self.input_size = input_size

        # Initialize Jetson inference
        try:
            self.net = jetson.inference.detectNet(model_path)
        except Exception as e:
            print(f"Error loading detection model: {e}")
            # Fallback to a default model
            self.net = jetson.inference.detectNet("ssd-mobilenet-v2")

        # Initialize camera for Jetson
        self.camera = jetson.utils.gstCamera(*input_size)
        self.display = jetson.utils.glDisplay()

        # PyTorch transforms for additional processing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in the image using Jetson inference

        Args:
            image: Input image (BGR format)

        Returns:
            List of detected objects with bounding boxes and confidence
        """
        # Convert numpy array to CUDA image
        img_cuda = jetson.utils.cudaFromNumpy(image)

        # Perform object detection
        detections = self.net.Detect(img_cuda)

        results = []
        for detection in detections:
            results.append({
                'class_id': int(detection.ClassID),
                'confidence': float(detection.Confidence),
                'left': int(detection.Left),
                'top': int(detection.Top),
                'right': int(detection.Right),
                'bottom': int(detection.Bottom),
                'width': int(detection.Width),
                'height': int(detection.Height),
                'area': int(detection.Area)
            })

        return results

    def preprocess_for_pytorch(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for additional PyTorch processing"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(image_rgb)
        tensor = self.transform(pil_image)

        # Add batch dimension
        return tensor.unsqueeze(0)

    def run_inference_loop(self):
        """Run continuous inference loop"""
        self.camera.Open()

        try:
            while self.display.IsOpen():
                # Capture image from camera
                img, width, height = self.camera.CaptureRGBA()

                # Convert CUDA image to numpy array
                img_np = jetson.utils.cudaToNumpy(img, width, height, 4)  # RGBA

                # Convert RGBA to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGBA2BGR)

                # Perform object detection
                detections = self.detect_objects(img_bgr)

                # Draw bounding boxes on image
                for detection in detections:
                    cv2.rectangle(
                        img_bgr,
                        (detection['left'], detection['top']),
                        (detection['right'], detection['bottom']),
                        (0, 255, 0),
                        2
                    )

                    # Add label
                    label = f"Class {detection['class_id']}: {detection['confidence']:.2f}"
                    cv2.putText(
                        img_bgr,
                        label,
                        (detection['left'], detection['top'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

                # Display the image
                img_cuda = jetson.utils.cudaFromNumpy(img_bgr)
                self.display.Render(img_cuda)

                # Update display
                self.display.SetTitle(f"Object Detection | {self.net.GetNetworkName()}")

        except KeyboardInterrupt:
            print("Inference loop interrupted")
        finally:
            self.camera.Close()

# Example usage for a custom AI model
class CustomJetsonModel:
    def __init__(self, model_path: str):
        """
        Load and run a custom PyTorch model on Jetson

        Args:
            model_path: Path to the PyTorch model file
        """
        # Load model
        self.model = torch.load(model_path)
        self.model.eval()

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Initialize TensorRT optimization (if available)
        try:
            import torch_tensorrt
            self.model = torch_tensorrt.compile(
                self.model,
                inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
                enabled_precisions={torch.float}
            )
            print("Model compiled with TensorRT")
        except ImportError:
            print("TensorRT not available, using standard PyTorch")

    def preprocess_input(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input image for the model"""
        # Resize image
        image_resized = cv2.resize(image, (224, 224))

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        tensor = torch.from_numpy(image_rgb.astype(np.float32)).permute(2, 0, 1) / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # Add batch dimension
        return tensor.unsqueeze(0)

    def predict(self, image: np.ndarray) -> torch.Tensor:
        """Run prediction on the image"""
        input_tensor = self.preprocess_input(image)

        # Move to GPU if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)

        return output

def jetson_example():
    """Example usage of Jetson AI processing"""
    try:
        # Initialize Jetson processor
        processor = JetsonAIProcessor()

        # Example: Load a custom model
        # custom_model = CustomJetsonModel("path/to/your/model.pth")

        print("Starting inference loop...")
        processor.run_inference_loop()

    except Exception as e:
        print(f"Error in Jetson example: {e}")

if __name__ == "__main__":
    jetson_example()
```

## Safety and Monitoring

### Robot Safety Monitor

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from builtin_interfaces.msg import Time
import numpy as np
import threading
import time

class RobotSafetyMonitor(Node):
    def __init__(self):
        super().__init__('robot_safety_monitor')

        # Safety parameters
        self.safety_distance = 0.5  # meters
        self.max_linear_vel = 0.5   # m/s
        self.max_angular_vel = 1.0  # rad/s
        self.emergency_stop = False
        self.safety_lock = threading.Lock()

        # Sensor data storage
        self.laser_data = None
        self.imu_data = None
        self.last_cmd_time = self.get_clock().now()

        # Publishers and subscribers
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel_raw', self.cmd_vel_callback, 10)
        self.safety_pub = self.create_publisher(Bool, '/safety_status', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Sensor subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Timer for safety checks
        self.safety_timer = self.create_timer(0.1, self.safety_check)

        self.get_logger().info('Robot Safety Monitor initialized')

    def cmd_vel_callback(self, msg):
        """Receive raw velocity commands"""
        with self.safety_lock:
            self.raw_cmd = msg
            self.last_cmd_time = self.get_clock().now()

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg

    def safety_check(self):
        """Perform safety checks and publish safe commands"""
        current_time = self.get_clock().now()

        # Check if emergency stop is active
        if self.emergency_stop:
            self.publish_safe_command(0.0, 0.0)
            return

        # Check for sensor timeouts
        if current_time.nanoseconds - self.last_cmd_time.nanoseconds > 1e9:  # 1 second timeout
            self.get_logger().warn('Command timeout - stopping robot')
            self.publish_safe_command(0.0, 0.0)
            return

        # Check laser data for obstacles
        if self.laser_data is not None:
            min_distance = min(self.laser_data.ranges)

            if min_distance < self.safety_distance:
                self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m - emergency stop')
                self.emergency_stop = True
                self.publish_safe_command(0.0, 0.0)
                return

        # Check IMU for abnormal accelerations
        if self.imu_data is not None:
            linear_acc = np.sqrt(
                self.imu_data.linear_acceleration.x**2 +
                self.imu_data.linear_acceleration.y**2 +
                self.imu_data.linear_acceleration.z**2
            )

            # Check for excessive acceleration (indicating collision or instability)
            if linear_acc > 10.0:  # 10 m/s^2 threshold
                self.get_logger().warn(f'Excessive acceleration detected: {linear_acc:.2f} m/s^2')
                self.emergency_stop = True
                self.publish_safe_command(0.0, 0.0)
                return

        # If all checks pass, publish processed command
        if hasattr(self, 'raw_cmd'):
            safe_cmd = self.process_command(self.raw_cmd)
            self.vel_pub.publish(safe_cmd)

    def process_command(self, cmd):
        """Process and limit raw command"""
        safe_cmd = Twist()

        # Limit linear velocity
        safe_cmd.linear.x = max(-self.max_linear_vel,
                               min(self.max_linear_vel, cmd.linear.x))
        safe_cmd.linear.y = max(-self.max_linear_vel,
                               min(self.max_linear_vel, cmd.linear.y))
        safe_cmd.linear.z = max(-self.max_linear_vel,
                               min(self.max_linear_vel, cmd.linear.z))

        # Limit angular velocity
        safe_cmd.angular.x = max(-self.max_angular_vel,
                                min(self.max_angular_vel, cmd.angular.x))
        safe_cmd.angular.y = max(-self.max_angular_vel,
                                min(self.max_angular_vel, cmd.angular.y))
        safe_cmd.angular.z = max(-self.max_angular_vel,
                                min(self.max_angular_vel, cmd.angular.z))

        return safe_cmd

    def publish_safe_command(self, linear_vel, angular_vel):
        """Publish safe stop command"""
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.vel_pub.publish(cmd)

        # Publish safety status
        status_msg = Bool()
        status_msg.data = not self.emergency_stop
        self.safety_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    safety_monitor = RobotSafetyMonitor()

    try:
        rclpy.spin(safety_monitor)
    except KeyboardInterrupt:
        pass
    finally:
        safety_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Utilities and Helper Functions

### Common Robotics Utilities

```python
import numpy as np
import math
from typing import Tuple, List, Optional
import transforms3d as tf3d

class RobotUtils:
    """Common robotics utility functions"""

    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion [x, y, z, w] to Euler angles [roll, pitch, yaw]"""
        return tf3d.euler.quat2euler(q, axes='sxyz')

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles [roll, pitch, yaw] to quaternion [x, y, z, w]"""
        return tf3d.euler.euler2quat(roll, pitch, yaw, axes='sxyz')

    @staticmethod
    def transform_point(point: np.ndarray, translation: np.ndarray,
                       rotation_quat: np.ndarray) -> np.ndarray:
        """Transform a point by translation and rotation"""
        # Convert quaternion to rotation matrix
        rotation_matrix = tf3d.quaternions.quat2mat(rotation_quat)

        # Apply rotation and translation
        transformed = rotation_matrix @ point + translation

        return transformed

    @staticmethod
    def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate 3D distance between two points"""
        return np.linalg.norm(p2 - p1)

    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in radians"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    @staticmethod
    def rotation_matrix_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
        """Convert rotation matrix to axis-angle representation"""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

        if angle < 1e-6:  # Identity rotation
            return np.array([0, 0, 1]), 0.0

        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))

        return axis / np.linalg.norm(axis), angle

class TrajectoryGenerator:
    """Generate smooth trajectories for robot motion"""

    @staticmethod
    def cubic_polynomial_trajectory(start_pos: float, end_pos: float,
                                  duration: float, dt: float = 0.01) -> List[float]:
        """Generate cubic polynomial trajectory"""
        # Cubic polynomial: s(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # Boundary conditions: s(0)=start, s(T)=end, s'(0)=0, s'(T)=0

        T = duration
        a0 = start_pos
        a1 = 0  # Initial velocity = 0
        a2 = 3 * (end_pos - start_pos) / T**2
        a3 = -2 * (end_pos - start_pos) / T**3

        times = np.arange(0, T, dt)
        positions = []

        for t in times:
            pos = a0 + a1*t + a2*t**2 + a3*t**3
            positions.append(pos)

        return positions

    @staticmethod
    def quintic_polynomial_trajectory(start_pos: float, end_pos: float,
                                    start_vel: float = 0.0, end_vel: float = 0.0,
                                    start_acc: float = 0.0, end_acc: float = 0.0,
                                    duration: float, dt: float = 0.01) -> Tuple[List[float], List[float], List[float]]:
        """Generate quintic polynomial trajectory with velocity and acceleration constraints"""
        T = duration

        # Coefficients for quintic polynomial
        a0 = start_pos
        a1 = start_vel
        a2 = start_acc / 2

        a3 = (20*(end_pos - start_pos) - (8*end_vel + 12*start_vel)*T - (3*start_acc - end_acc)*T**2) / (2*T**3)
        a4 = (30*(start_pos - end_pos) + (14*end_vel + 16*start_vel)*T + (3*start_acc - 2*end_acc)*T**2) / (2*T**4)
        a5 = (12*(end_pos - start_pos) - 6*(end_vel + start_vel)*T - (end_acc - start_acc)*T**2) / (2*T**5)

        times = np.arange(0, T, dt)
        positions = []
        velocities = []
        accelerations = []

        for t in times:
            # Position
            pos = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
            positions.append(pos)

            # Velocity
            vel = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
            velocities.append(vel)

            # Acceleration
            acc = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
            accelerations.append(acc)

        return positions, velocities, accelerations

# Example usage
def utilities_example():
    """Example usage of utility functions"""

    # Quaternion-Euler conversions
    quat = RobotUtils.euler_to_quaternion(0.1, 0.2, 0.3)
    euler = RobotUtils.quaternion_to_euler(quat)
    print(f"Quaternion: {quat}")
    print(f"Euler angles: {euler}")

    # Point transformation
    point = np.array([1, 0, 0])
    translation = np.array([0, 0, 1])
    rotation = RobotUtils.euler_to_quaternion(0, 0, math.pi/2)  # 90-degree rotation around Z
    transformed_point = RobotUtils.transform_point(point, translation, rotation)
    print(f"Transformed point: {transformed_point}")

    # Trajectory generation
    positions = TrajectoryGenerator.cubic_polynomial_trajectory(0, 1, 2.0)
    print(f"Cubic trajectory has {len(positions)} points")

    pos, vel, acc = TrajectoryGenerator.quintic_polynomial_trajectory(0, 1, duration=2.0)
    print(f"Quintic trajectory has {len(pos)} points")

if __name__ == "__main__":
    utilities_example()
```

---

Continue with [Simulation Assets Guide](./simulation-assets.md) to explore the creation and utilization of 3D models, environments, and other assets for robotics simulation.