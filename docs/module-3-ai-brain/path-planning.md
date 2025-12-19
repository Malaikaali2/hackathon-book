---
sidebar_position: 5
---

# Path Planning Algorithm Implementation

## Learning Objectives

By the end of this section, you will be able to:

1. Implement classical path planning algorithms (A*, Dijkstra, RRT)
2. Design motion planning systems for robotic navigation
3. Integrate path planning with perception and control systems
4. Optimize planning algorithms for real-time performance
5. Handle dynamic obstacles and replanning scenarios

## Introduction to Path Planning

Path planning is a fundamental capability for autonomous robots, enabling them to navigate from a start position to a goal position while avoiding obstacles and respecting kinematic constraints. In the context of AI-powered robots, path planning bridges the gap between perception (understanding the environment) and control (executing movement).

Path planning algorithms can be categorized into several types:

- **Global planners**: Compute paths using a complete map of the environment
- **Local planners**: Generate short-term paths based on immediate sensor data
- **Sampling-based planners**: Use random sampling to explore configuration space
- **Optimization-based planners**: Formulate path planning as an optimization problem

## Configuration Space and Representation

### Environment Representation

Robots operate in a configuration space (C-space) where each point represents a possible state of the robot. For path planning, we typically represent the environment as:

- **Discrete grid maps**: 2D or 3D grids with occupancy values
- **Topological maps**: Graphs representing connectivity between locations
- **Continuous spaces**: Mathematical representations of free space

### Occupancy Grids

Occupancy grids provide a probabilistic representation of the environment:

```python
import numpy as np
from nav_msgs.msg import OccupancyGrid
import math

class OccupancyGridMap:
    def __init__(self, resolution=0.05, width=100, height=100):
        self.resolution = resolution  # meters per cell
        self.width = width  # number of cells
        self.height = height  # number of cells
        self.origin_x = 0.0
        self.origin_y = 0.0

        # Initialize grid with unknown values (-1)
        self.grid = np.full((height, width), -1, dtype=np.int8)

    def world_to_grid(self, x_world, y_world):
        """Convert world coordinates to grid coordinates"""
        x_grid = int((x_world - self.origin_x) / self.resolution)
        y_grid = int((y_world - self.origin_y) / self.resolution)
        return x_grid, y_grid

    def grid_to_world(self, x_grid, y_grid):
        """Convert grid coordinates to world coordinates"""
        x_world = x_grid * self.resolution + self.origin_x
        y_world = y_grid * self.resolution + self.origin_y
        return x_world, y_world

    def is_occupied(self, x_grid, y_grid):
        """Check if grid cell is occupied"""
        if 0 <= x_grid < self.width and 0 <= y_grid < self.height:
            return self.grid[y_grid, x_grid] > 50  # Occupied if probability > 50%
        return True  # Out of bounds is considered occupied

    def is_free(self, x_grid, y_grid):
        """Check if grid cell is free"""
        if 0 <= x_grid < self.width and 0 <= y_grid < self.height:
            return self.grid[y_grid, x_grid] < 25  # Free if probability < 25%
        return False

    def get_neighbors(self, x, y):
        """Get 8-connected neighbors of a grid cell"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors
```

## Classical Path Planning Algorithms

### A* Algorithm

A* is a popular graph search algorithm that finds optimal paths by combining actual cost from start with heuristic cost to goal:

```python
import heapq
from typing import List, Tuple

class AStarPlanner:
    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid

    def heuristic(self, pos1, pos2):
        """Calculate heuristic distance (Euclidean)"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def plan_path(self, start, goal):
        """Plan path using A* algorithm"""
        # Convert world coordinates to grid coordinates
        start_grid = self.grid.world_to_grid(start[0], start[1])
        goal_grid = self.grid.world_to_grid(goal[0], goal[1])

        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start_grid)]
        closed_set = set()

        # Cost dictionaries
        g_score = {start_grid: 0}
        came_from = {}

        while open_set:
            f_score, current_g, current = heapq.heappop(open_set)

            # Check if we reached the goal
            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            # Skip if already processed
            if current in closed_set:
                continue

            closed_set.add(current)

            # Process neighbors
            for neighbor in self.grid.get_neighbors(current[0], current[1]):
                if self.grid.is_occupied(neighbor[0], neighbor[1]):
                    continue

                # Calculate tentative g_score
                tentative_g = current_g + self.calculate_distance(current, neighbor)

                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue

                # This path to neighbor is better than any previous one
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + self.heuristic(neighbor, goal_grid)

                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None  # No path found

    def calculate_distance(self, pos1, pos2):
        """Calculate distance between adjacent grid cells"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx*dx + dy*dy)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        # Convert grid coordinates back to world coordinates
        world_path = []
        for grid_pos in path:
            world_pos = self.grid.grid_to_world(grid_pos[0], grid_pos[1])
            world_path.append(world_pos)

        return world_path
```

### Dijkstra's Algorithm

Dijkstra's algorithm is similar to A* but uses uniform cost without heuristic:

```python
class DijkstraPlanner:
    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid

    def plan_path(self, start, goal):
        """Plan path using Dijkstra's algorithm"""
        start_grid = self.grid.world_to_grid(start[0], start[1])
        goal_grid = self.grid.world_to_grid(goal[0], goal[1])

        # Priority queue: (g_score, position)
        open_set = [(0, start_grid)]
        closed_set = set()

        # Cost dictionary
        g_score = {start_grid: 0}
        came_from = {}

        while open_set:
            current_g, current = heapq.heappop(open_set)

            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            if current in closed_set:
                continue

            closed_set.add(current)

            for neighbor in self.grid.get_neighbors(current[0], current[1]):
                if self.grid.is_occupied(neighbor[0], neighbor[1]):
                    continue

                tentative_g = current_g + self.calculate_distance(current, neighbor)

                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                heapq.heappush(open_set, (tentative_g, neighbor))

        return None

    def calculate_distance(self, pos1, pos2):
        """Calculate distance between adjacent grid cells"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx*dx + dy*dy)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        # Convert grid coordinates back to world coordinates
        world_path = []
        for grid_pos in path:
            world_pos = self.grid.grid_to_world(grid_pos[0], grid_pos[1])
            world_path.append(world_pos)

        return world_path
```

## Sampling-Based Planning: RRT

Rapidly-exploring Random Trees (RRT) are particularly useful for high-dimensional configuration spaces:

```python
import random

class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRTPlanner:
    def __init__(self, occupancy_grid, step_size=0.5):
        self.grid = occupancy_grid
        self.step_size = step_size

    def plan_path(self, start, goal, max_iterations=1000):
        """Plan path using RRT algorithm"""
        start_grid = self.grid.world_to_grid(start[0], start[1])
        goal_grid = self.grid.world_to_grid(goal[0], goal[1])

        start_node = RRTNode(start_grid[0], start_grid[1])
        tree = [start_node]

        for i in range(max_iterations):
            # Sample random point
            rand_x = random.randint(0, self.grid.width - 1)
            rand_y = random.randint(0, self.grid.height - 1)

            # Find nearest node in tree
            nearest_node = self.find_nearest_node(tree, rand_x, rand_y)

            # Extend towards random point
            new_node = self.extend_towards(nearest_node, rand_x, rand_y)

            if new_node and not self.is_collision(new_node):
                new_node.parent = nearest_node
                tree.append(new_node)

                # Check if we're close to goal
                if self.distance((new_node.x, new_node.y), goal_grid) < 5:  # 5 cells threshold
                    return self.extract_path(new_node, start_node)

        return None

    def find_nearest_node(self, tree, x, y):
        """Find the nearest node in the tree to the given coordinates"""
        nearest = tree[0]
        min_dist = self.distance((nearest.x, nearest.y), (x, y))

        for node in tree:
            dist = self.distance((node.x, node.y), (x, y))
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def extend_towards(self, from_node, to_x, to_y):
        """Extend the tree towards the target point"""
        direction = math.atan2(to_y - from_node.y, to_x - from_node.x)
        new_x = int(from_node.x + self.step_size * math.cos(direction))
        new_y = int(from_node.y + self.step_size * math.sin(direction))

        # Check bounds
        if 0 <= new_x < self.grid.width and 0 <= new_y < self.grid.height:
            return RRTNode(new_x, new_y)
        return None

    def is_collision(self, node):
        """Check if the node is in collision"""
        return self.grid.is_occupied(node.x, node.y)

    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def extract_path(self, goal_node, start_node):
        """Extract path from goal node back to start"""
        path = []
        current = goal_node

        while current != start_node:
            path.append((current.x, current.y))
            current = current.parent

        path.append((start_node.x, start_node.y))
        path.reverse()

        # Convert grid coordinates back to world coordinates
        world_path = []
        for grid_pos in path:
            world_pos = self.grid.grid_to_world(grid_pos[0], grid_pos[1])
            world_path.append(world_pos)

        return world_path
```

## Integration with ROS 2 and Isaac

### Isaac ROS Navigation Integration

Isaac provides optimized navigation capabilities that can be integrated with path planning:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

class IsaacPathPlannerNode(Node):
    def __init__(self):
        super().__init__('isaac_path_planner')

        # Publishers
        self.path_publisher = self.create_publisher(Path, 'planned_path', 10)
        self.visualization_publisher = self.create_publisher(MarkerArray, 'path_visualization', 10)

        # Subscribers
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10
        )

        self.goal_subscription = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_callback,
            10
        )

        self.laser_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )

        # Initialize planners
        self.occupancy_grid = None
        self.a_star_planner = None
        self.current_goal = None

    def map_callback(self, msg):
        """Handle incoming map message"""
        self.occupancy_grid = self.convert_occupancy_grid(msg)
        self.a_star_planner = AStarPlanner(self.occupancy_grid)

    def goal_callback(self, msg):
        """Handle incoming goal pose"""
        if self.a_star_planner:
            self.current_goal = (msg.pose.position.x, msg.pose.position.y)

            # Plan path from current position to goal
            current_pos = self.get_current_position()  # Implementation needed
            path = self.a_star_planner.plan_path(current_pos, self.current_goal)

            if path:
                self.publish_path(path)
                self.visualize_path(path)

    def laser_callback(self, msg):
        """Handle laser scan for local obstacle avoidance"""
        # Update local map with laser scan data
        # This would integrate with local planner for dynamic obstacle avoidance
        pass

    def convert_occupancy_grid(self, ros_grid_msg):
        """Convert ROS OccupancyGrid message to our format"""
        grid = OccupancyGridMap(
            resolution=ros_grid_msg.info.resolution,
            width=ros_grid_msg.info.width,
            height=ros_grid_msg.info.height
        )

        # Convert ROS grid data to our format
        grid.grid = np.array(ros_grid_msg.data).reshape((grid.height, grid.width))
        grid.origin_x = ros_grid_msg.info.origin.position.x
        grid.origin_y = ros_grid_msg.info.origin.position.y

        return grid

    def publish_path(self, path):
        """Publish planned path as ROS Path message"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation
            path_msg.poses.append(pose)

        self.path_publisher.publish(path_msg)

    def visualize_path(self, path):
        """Visualize path in RViz"""
        marker_array = MarkerArray()

        # Create line strip marker for path
        line_marker = Marker()
        line_marker.header.frame_id = 'map'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = 'path'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05  # Line width
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0

        for point in path:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.0
            line_marker.points.append(p)

        marker_array.markers.append(line_marker)
        self.visualization_publisher.publish(marker_array)

    def get_current_position(self):
        """Get current robot position - implementation depends on localization system"""
        # This would typically interface with AMCL or other localization
        # For now, return a placeholder
        return (0.0, 0.0)
```

## Motion Planning for Manipulation

### Configuration Space for Manipulators

Motion planning for robotic arms requires planning in joint space rather than Cartesian space:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class ManipulatorPathPlanner:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.joint_limits = robot_model.get_joint_limits()

    def plan_manipulation_path(self, start_joints, goal_pose, obstacles=None):
        """Plan path for manipulator from start joint configuration to goal pose"""
        # Convert goal pose to joint space using inverse kinematics
        goal_joints = self.inverse_kinematics(goal_pose)

        if goal_joints is None:
            return None  # No valid IK solution

        # Plan path in joint space
        path = self.plan_joint_space_path(start_joints, goal_joints, obstacles)
        return path

    def inverse_kinematics(self, target_pose):
        """Calculate inverse kinematics to find joint angles for target pose"""
        # This would use robot-specific IK solver
        # For demonstration, return a simple approximation
        pass

    def plan_joint_space_path(self, start_joints, goal_joints, obstacles):
        """Plan path in joint space"""
        # Use RRT or other sampling-based method in joint space
        path = []

        # Linear interpolation in joint space as a simple approach
        steps = 50
        for i in range(steps + 1):
            t = i / steps
            joints = []
            for start, goal in zip(start_joints, goal_joints):
                joint_val = start + t * (goal - start)
                joints.append(joint_val)

            # Check collision for this configuration
            if not self.check_collision(joints, obstacles):
                path.append(joints)

        return path

    def check_collision(self, joint_config, obstacles):
        """Check if joint configuration results in collision"""
        # Calculate link positions using forward kinematics
        link_positions = self.forward_kinematics(joint_config)

        # Check collision with obstacles
        for link_pos in link_positions:
            for obstacle in obstacles:
                if self.distance(link_pos, obstacle) < self.robot.link_radius:
                    return True
        return False

    def forward_kinematics(self, joint_config):
        """Calculate forward kinematics to get link positions"""
        # This would use robot-specific FK solver
        pass

    def distance(self, pos1, pos2):
        """Calculate distance between two points"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
```

## Dynamic Path Planning and Replanning

### Handling Moving Obstacles

Real-world environments contain moving obstacles that require replanning:

```python
class DynamicPathPlanner:
    def __init__(self, base_planner):
        self.base_planner = base_planner
        self.current_path = None
        self.path_index = 0
        self.last_replan_time = 0
        self.replan_interval = 1.0  # seconds

    def update_and_follow_path(self, current_pose, dynamic_obstacles):
        """Update path based on dynamic obstacles and follow current path"""
        current_time = time.time()

        # Check if we need to replan
        if self.should_replan(current_pose, dynamic_obstacles, current_time):
            new_path = self.base_planner.plan_path(
                current_pose,
                self.goal,
                dynamic_obstacles
            )

            if new_path:
                self.current_path = new_path
                self.path_index = self.find_closest_path_point(current_pose)
                self.last_replan_time = current_time

        # Follow the current path
        return self.get_next_waypoint()

    def should_replan(self, current_pose, dynamic_obstacles, current_time):
        """Determine if replanning is needed"""
        # Replan if:
        # 1. No path exists
        if self.current_path is None:
            return True

        # 2. Time since last replan exceeds interval
        if current_time - self.last_replan_time > self.replan_interval:
            return True

        # 3. Obstacle collision detected along path
        if self.path_collides_with_obstacles(self.current_path, dynamic_obstacles):
            return True

        # 4. Robot is too far from planned path
        closest_path_point = self.get_closest_path_point(current_pose)
        distance_to_path = self.distance(current_pose, closest_path_point)
        if distance_to_path > 0.5:  # 0.5m threshold
            return True

        return False

    def path_collides_with_obstacles(self, path, obstacles):
        """Check if path collides with any obstacles"""
        for i in range(len(path) - 1):
            for obstacle in obstacles:
                if self.segment_intersects_obstacle(path[i], path[i+1], obstacle):
                    return True
        return False

    def segment_intersects_obstacle(self, start, end, obstacle):
        """Check if line segment intersects with obstacle"""
        # Simplified collision check - in practice would use more sophisticated methods
        segment_length = self.distance(start, end)
        obstacle_radius = obstacle.get('radius', 0.5)  # Default 0.5m radius

        # Check distance from obstacle center to line segment
        dist_to_segment = self.distance_point_to_segment(obstacle['center'], start, end)
        return dist_to_segment < obstacle_radius

    def distance_point_to_segment(self, point, seg_start, seg_end):
        """Calculate distance from point to line segment"""
        # Vector math to calculate distance from point to line segment
        pass
```

## Performance Optimization

### Path Smoothing

Raw planned paths often contain unnecessary waypoints that should be smoothed:

```python
def smooth_path(path, max_iterations=100, tolerance=0.001):
    """Smooth path by removing unnecessary waypoints"""
    if len(path) < 3:
        return path

    smoothed_path = path.copy()

    for _ in range(max_iterations):
        improved = False

        for i in range(1, len(smoothed_path) - 1):
            # Check if we can connect previous and next points directly
            prev_point = smoothed_path[i-1]
            next_point = smoothed_path[i+1]

            # Check if direct connection is collision-free
            if is_line_collision_free(prev_point, next_point, obstacles):
                # Remove the middle point
                smoothed_path.pop(i)
                improved = True
                break  # Restart to check from beginning

        if not improved:
            break

    return smoothed_path

def is_line_collision_free(start, end, obstacles):
    """Check if line segment is collision-free"""
    # Sample points along the line and check for collisions
    num_samples = 10
    for i in range(num_samples + 1):
        t = i / num_samples
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])

        for obstacle in obstacles:
            if distance((x, y), obstacle.center) < obstacle.radius:
                return False
    return True
```

### Multi-Resolution Planning

Use different map resolutions for efficiency:

```python
class MultiResolutionPlanner:
    def __init__(self, fine_grid, coarse_grid):
        self.fine_grid = fine_grid  # High-resolution local planning
        self.coarse_grid = coarse_grid  # Low-resolution global planning

    def plan_path(self, start, goal):
        """Plan path using multi-resolution approach"""
        # Plan coarse path first
        coarse_path = self.plan_coarse_path(start, goal)

        if not coarse_path:
            return None

        # Refine path using fine resolution
        refined_path = []
        for i in range(len(coarse_path) - 1):
            segment_start = coarse_path[i]
            segment_end = coarse_path[i + 1]

            # Plan fine path for this segment
            fine_segment = self.plan_fine_path(segment_start, segment_end)
            if fine_segment:
                refined_path.extend(fine_segment[:-1])  # Exclude last point to avoid duplication
            else:
                # If fine planning fails, try to connect directly
                refined_path.append(segment_start)

        refined_path.append(goal)  # Add final goal
        return refined_path

    def plan_coarse_path(self, start, goal):
        """Plan path on coarse resolution grid"""
        planner = AStarPlanner(self.coarse_grid)
        return planner.plan_path(start, goal)

    def plan_fine_path(self, start, goal):
        """Plan path on fine resolution grid"""
        planner = AStarPlanner(self.fine_grid)
        return planner.plan_path(start, goal)
```

## Summary

Path planning algorithms form the backbone of autonomous navigation systems, enabling robots to move safely and efficiently through their environment. From classical algorithms like A* to sampling-based methods like RRT, each approach has its strengths for different scenarios.

The next section will cover manipulation control systems, which enable robots to interact with objects in their environment.

## References

[All sources will be cited in the References section at the end of the book, following APA format]