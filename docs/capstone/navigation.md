---
sidebar_position: 18
---

# Navigation and Obstacle Avoidance: Autonomous Humanoid Capstone

## Overview

Navigation and obstacle avoidance form the mobility foundation of the autonomous humanoid system, enabling safe and efficient movement through complex environments. This component encompasses path planning, localization, obstacle detection, dynamic obstacle avoidance, and recovery from navigation failures. The system must handle both static obstacles in known environments and dynamic obstacles that appear during navigation, while maintaining real-time performance and safety guarantees.

The navigation system integrates with task planning to execute mobility goals, with perception for environmental awareness, with manipulation for precise positioning, and with voice processing for location-based commands. This implementation guide provides detailed instructions for building a robust navigation and obstacle avoidance system that can operate in real-world environments with varying complexity.

## System Architecture

### Navigation Stack Architecture

The navigation system implements a layered architecture following ROS 2 Navigation2 standards:

```
High-Level Goals → Path Planning → Path Following → Local Control → Robot Motion
```

The architecture consists of:
1. **Global Planner**: Computes optimal paths in static map
2. **Local Planner**: Generates velocity commands for immediate motion
3. **Costmap Server**: Maintains obstacle information
4. **Controller Server**: Follows reference paths with velocity control
5. **Recovery Server**: Handles navigation failures with recovery behaviors
6. **Lifecycle Manager**: Manages navigation lifecycle states

### Integration with Other Systems

The navigation system interfaces with:
- **Task Planning**: Receives navigation goals and reports completion
- **Perception System**: Gets real-time obstacle information
- **Localization**: Maintains accurate position estimate
- **Manipulation**: Provides precise positioning for object interaction
- **Voice Processing**: Handles location-based commands

## Technical Implementation

### 1. Global Path Planning

#### A* Path Planning Algorithm

```python
import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional
import rospy

class GridMap:
    """Represents the navigation environment as a grid"""

    def __init__(self, width: int, height: int, resolution: float = 0.05):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((height, width), dtype=np.uint8)  # 0: free, 255: occupied

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.resolution * self.width / 2) / self.resolution)
        grid_y = int((y - self.resolution * self.height / 2) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        x = grid_x * self.resolution + self.resolution * self.width / 2
        y = grid_y * self.resolution + self.resolution * self.height / 2
        return x, y

    def is_valid(self, x: int, y: int) -> bool:
        """Check if grid coordinates are within bounds and free"""
        return (0 <= x < self.width and
                0 <= y < self.height and
                self.grid[y, x] < 127)  # Consider values < 127 as free space

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-connectivity)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors

class AStarPlanner:
    """A* path planning algorithm implementation"""

    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map
        self.diagonal_cost = np.sqrt(2)

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Euclidean)"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx*dx + dy*dy)

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Plan path from start to goal using A* algorithm"""
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self.grid_map.get_neighbors(*current):
                tentative_g_score = g_score[current] + self._get_distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def _get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between adjacent grid cells"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        if dx == 1 and dy == 1:  # Diagonal
            return self.diagonal_cost
        else:  # Horizontal/Vertical
            return 1.0

class GlobalPlanner:
    """Global path planner interface for navigation system"""

    def __init__(self):
        self.grid_map = None
        self.planner = None
        self.path_cache = {}

    def set_map(self, occupancy_grid):
        """Set the global map for planning"""
        # Convert ROS OccupancyGrid to internal GridMap representation
        width = occupancy_grid.info.width
        height = occupancy_grid.info.height
        resolution = occupancy_grid.info.resolution

        self.grid_map = GridMap(width, height, resolution)
        self.grid_map.grid = np.array(occupancy_grid.data).reshape((height, width))

        self.planner = AStarPlanner(self.grid_map)

    def plan_path(self, start_pose, goal_pose) -> Optional[List[Tuple[float, float]]]:
        """Plan path from start to goal in world coordinates"""
        if not self.planner:
            rospy.logerr("Global map not set")
            return None

        # Convert world coordinates to grid coordinates
        start_grid = self.grid_map.world_to_grid(start_pose.position.x, start_pose.position.y)
        goal_grid = self.grid_map.world_to_grid(goal_pose.position.x, goal_pose.position.y)

        # Plan path in grid coordinates
        grid_path = self.planner.plan(start_grid, goal_grid)

        if not grid_path:
            rospy.logwarn(f"No path found from {start_grid} to {goal_grid}")
            return None

        # Convert grid path back to world coordinates
        world_path = []
        for grid_x, grid_y in grid_path:
            world_x, world_y = self.grid_map.grid_to_world(grid_x, grid_y)
            world_path.append((world_x, world_y))

        return world_path
```

#### Dynamic Path Replanning

```python
class DynamicPlanner:
    """Handles dynamic replanning when obstacles are detected"""

    def __init__(self, global_planner: GlobalPlanner):
        self.global_planner = global_planner
        self.current_path = []
        self.path_index = 0
        self.replan_threshold = 0.5  # meters

    def update_obstacles(self, obstacle_map):
        """Update obstacle information and replan if necessary"""
        # Check if current path is blocked by new obstacles
        if self._is_path_blocked(self.current_path[self.path_index:], obstacle_map):
            # Replan from current position
            current_pos = self.get_current_position()
            goal_pos = self.get_goal_position()

            new_path = self.global_planner.plan_path(current_pos, goal_pos)
            if new_path:
                self.current_path = new_path
                self.path_index = 0
                return True

        return False

    def _is_path_blocked(self, path: List[Tuple[float, float]], obstacle_map) -> bool:
        """Check if path is blocked by obstacles"""
        # This would check each point in path against obstacle map
        for point in path:
            if self._is_point_blocked(point, obstacle_map):
                return True
        return False

    def _is_point_blocked(self, point: Tuple[float, float], obstacle_map) -> bool:
        """Check if a single point is blocked by obstacles"""
        # Simplified check - in practice this would query costmap
        return False  # Placeholder implementation

    def get_current_position(self):
        """Get current robot position from localization"""
        # This would interface with localization system
        pass

    def get_goal_position(self):
        """Get current navigation goal"""
        # This would return the current goal
        pass
```

### 2. Local Path Planning and Control

#### DWA (Dynamic Window Approach) Controller

```python
from dataclasses import dataclass
from geometry_msgs.msg import Twist, Pose, Point
import math

@dataclass
class RobotState:
    """Current robot state for local planning"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vtheta: float = 0.0

@dataclass
class RobotConfig:
    """Robot configuration parameters"""
    max_speed_linear: float = 0.5
    max_speed_angular: float = 1.0
    min_speed_linear: float = 0.05
    min_speed_angular: float = 0.1
    max_accel_linear: float = 1.0
    max_accel_angular: float = 2.0
    robot_radius: float = 0.3
    dt: float = 0.1

class DWAController:
    """Dynamic Window Approach local planner"""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.robot_state = RobotState()

    def get_dynamic_window(self) -> Tuple[float, float, float, float]:
        """Calculate dynamic window based on current velocities and constraints"""
        vs_min = max(-self.config.max_speed_linear,
                    self.robot_state.vx - self.config.max_accel_linear * self.config.dt)
        vs_max = min(self.config.max_speed_linear,
                    self.robot_state.vx + self.config.max_accel_linear * self.config.dt)

        ws_min = max(-self.config.max_speed_angular,
                    self.robot_state.vtheta - self.config.max_accel_angular * self.config.dt)
        ws_max = min(self.config.max_speed_angular,
                    self.robot_state.vtheta + self.config.max_accel_angular * self.config.dt)

        return vs_min, vs_max, ws_min, ws_max

    def predict_trajectory(self, v: float, w: float, predict_time: float = 1.0) -> List[Tuple[float, float]]:
        """Predict trajectory for given velocity commands"""
        trajectory = []
        state = RobotState(
            x=self.robot_state.x,
            y=self.robot_state.y,
            theta=self.robot_state.theta,
            vx=self.robot_state.vx,
            vy=self.robot_state.vy,
            vtheta=self.robot_state.vtheta
        )

        dt = self.config.dt
        time = 0.0

        while time < predict_time:
            # Update state based on velocity commands
            state.x += v * math.cos(state.theta) * dt
            state.y += v * math.sin(state.theta) * dt
            state.theta += w * dt
            state.vx = v
            state.vtheta = w

            trajectory.append((state.x, state.y))
            time += dt

        return trajectory

    def calc_to_goal_cost(self, trajectory: List[Tuple[float, float]], goal: Tuple[float, float]) -> float:
        """Calculate cost to goal for trajectory"""
        last_point = trajectory[-1] if trajectory else (0, 0)
        distance = math.sqrt((last_point[0] - goal[0])**2 + (last_point[1] - goal[1])**2)
        return distance

    def calc_obstacle_cost(self, trajectory: List[Tuple[float, float]], obstacles: List[Tuple[float, float]]) -> float:
        """Calculate obstacle collision cost for trajectory"""
        if not trajectory:
            return float('inf')

        min_dist = float('inf')
        for point in trajectory:
            for obs in obstacles:
                dist = math.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                if dist < min_dist:
                    min_dist = dist

        # Return high cost if too close to obstacles
        if min_dist < self.config.robot_radius:
            return float('inf')
        else:
            return 1.0 / min_dist if min_dist > 0 else float('inf')

    def dwa_control(self, goal: Tuple[float, float], obstacles: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate optimal velocity commands using DWA"""
        vs_min, vs_max, ws_min, ws_max = self.get_dynamic_window()

        best_v = 0.0
        best_w = 0.0
        min_cost = float('inf')

        # Discretize velocity space
        v_resolution = 0.05
        w_resolution = 0.05

        v_range = np.arange(vs_min, vs_max, v_resolution)
        w_range = np.arange(ws_min, ws_max, w_resolution)

        for v in v_range:
            for w in w_range:
                trajectory = self.predict_trajectory(v, w)

                # Calculate costs
                to_goal_cost = self.calc_to_goal_cost(trajectory, goal)
                obstacle_cost = self.calc_obstacle_cost(trajectory, obstacles)

                # Combine costs with weights
                # Higher weights mean more important
                alpha = 1.0  # Goal heading cost weight
                beta = 1.0   # Velocity cost weight
                gamma = 1.0  # Obstacle cost weight

                cost = alpha * to_goal_cost + gamma * obstacle_cost

                if cost < min_cost:
                    min_cost = cost
                    best_v = v
                    best_w = w

        return best_v, best_w
```

#### Path Following Controller

```python
class PathFollower:
    """Follows reference path with velocity control"""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.path = []
        self.current_index = 0
        self.lookahead_distance = 0.5
        self.kp_linear = 1.0
        self.kp_angular = 2.0
        self.robot_state = RobotState()

    def set_path(self, path: List[Tuple[float, float]]):
        """Set the reference path to follow"""
        self.path = path
        self.current_index = 0

    def follow_path(self) -> Twist:
        """Generate velocity commands to follow the path"""
        if not self.path or self.current_index >= len(self.path):
            return Twist()  # Stop if no path or reached goal

        # Find next target point on path
        target = self._get_target_point()

        if not target:
            return Twist()

        # Calculate error
        dx = target[0] - self.robot_state.x
        dy = target[1] - self.robot_state.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Calculate desired heading
        desired_theta = math.atan2(dy, dx)
        angle_error = desired_theta - self.robot_state.theta

        # Normalize angle error
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Generate velocity commands
        cmd = Twist()

        # Linear velocity: proportional to distance to target, limited by max speed
        cmd.linear.x = min(self.config.max_speed_linear,
                          max(self.config.min_speed_linear,
                              self.kp_linear * distance))

        # Angular velocity: proportional to angle error, limited by max speed
        cmd.angular.z = min(self.config.max_speed_angular,
                           max(-self.config.max_speed_angular,
                               self.kp_angular * angle_error))

        # Update path index if close enough to current target
        if distance < 0.1:  # 10cm threshold
            self.current_index += 1

        return cmd

    def _get_target_point(self) -> Optional[Tuple[float, float]]:
        """Get the next target point on the path"""
        if self.current_index >= len(self.path):
            return None

        # Look ahead to find a point at appropriate distance
        current_point = self.path[self.current_index]

        # Simple approach: take the next point in path
        # For more sophisticated lookahead, see Pure Pursuit algorithms
        target_index = min(self.current_index + 1, len(self.path) - 1)
        return self.path[target_index]

    def is_goal_reached(self) -> bool:
        """Check if the goal has been reached"""
        if not self.path:
            return True

        goal = self.path[-1]
        distance = math.sqrt((self.robot_state.x - goal[0])**2 + (self.robot_state.y - goal[1])**2)
        return distance < 0.2  # 20cm tolerance
```

### 3. Costmap and Obstacle Management

#### Costmap Server Implementation

```python
class Costmap2D:
    """2D costmap for obstacle representation"""

    def __init__(self, width: int, height: int, resolution: float, origin_x: float = 0.0, origin_y: float = 0.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        # Initialize costmap with free space (0 = free, 254 = occupied, 255 = unknown)
        self.costmap = np.zeros((height, width), dtype=np.uint8)

        # Parameters for cost propagation
        self.inflation_radius = 0.5  # meters
        self.cost_scaling_factor = 3.0

    def world_to_map(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to map coordinates"""
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """Convert map coordinates to world coordinates"""
        wx = mx * self.resolution + self.origin_x
        wy = my * self.resolution + self.origin_y
        return wx, wy

    def is_valid_cell(self, mx: int, my: int) -> bool:
        """Check if map coordinates are valid"""
        return 0 <= mx < self.width and 0 <= my < self.height

    def set_obstacle(self, wx: float, wy: float, cost: int = 254):
        """Set obstacle at world coordinates"""
        mx, my = self.world_to_map(wx, wy)
        if self.is_valid_cell(mx, my):
            self.costmap[my, mx] = min(254, cost)

    def get_cost(self, wx: float, wy: float) -> int:
        """Get cost at world coordinates"""
        mx, my = self.world_to_map(wx, wy)
        if self.is_valid_cell(mx, my):
            return self.costmap[my, mx]
        else:
            return 255  # Unknown/invalid

    def update_with_laser_scan(self, laser_data, robot_pose):
        """Update costmap with laser scan data"""
        # Process laser scan to identify obstacles and free space
        angle_min = laser_data.angle_min
        angle_increment = laser_data.angle_increment

        for i, range_val in enumerate(laser_data.ranges):
            if not (laser_data.range_min <= range_val <= laser_data.range_max):
                continue  # Invalid range

            # Calculate angle of this range measurement
            angle = angle_min + i * angle_increment

            # Calculate world coordinates of obstacle
            ox = robot_pose.position.x + range_val * math.cos(robot_pose.orientation.z + angle)
            oy = robot_pose.position.y + range_val * math.sin(robot_pose.orientation.z + angle)

            # Set obstacle in costmap
            self.set_obstacle(ox, oy)

    def update_with_occupancy_grid(self, occupancy_grid):
        """Update costmap with occupancy grid data"""
        # This would merge the incoming occupancy grid with current costmap
        # For now, we'll do a simple copy if the grids match in size and resolution
        pass

    def inflate_obstacles(self):
        """Inflate obstacles to create safety margins"""
        # Create a copy of the original costmap
        original_costmap = self.costmap.copy()

        # Calculate inflation radius in cells
        inflation_cells = int(self.inflation_radius / self.resolution)

        for y in range(self.height):
            for x in range(self.width):
                if original_costmap[y, x] >= 254:  # Obstacle cell
                    # Inflate obstacle in a circular pattern
                    for dy in range(-inflation_cells, inflation_cells + 1):
                        for dx in range(-inflation_cells, inflation_cells + 1):
                            nx, ny = x + dx, y + dy

                            if self.is_valid_cell(nx, ny):
                                # Calculate distance from obstacle center
                                dist = math.sqrt(dx*dx + dy*dy) * self.resolution

                                if dist <= self.inflation_radius:
                                    # Calculate cost based on distance (higher cost closer to obstacle)
                                    cost = int(254 * (1.0 - dist / self.inflation_radius))
                                    self.costmap[ny, nx] = max(self.costmap[ny, nx], cost)

    def is_free(self, wx: float, wy: float) -> bool:
        """Check if a point is free for navigation"""
        cost = self.get_cost(wx, wy)
        return cost < 127  # Consider values < 127 as free space

class CostmapServer:
    """ROS 2 costmap server interface"""

    def __init__(self):
        self.global_costmap = None
        self.local_costmap = None
        self.update_rate = 10.0  # Hz
        self.transform_tolerance = 0.1  # seconds

        # Initialize costmaps
        self._initialize_costmaps()

        # ROS interfaces
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

    def _initialize_costmaps(self):
        """Initialize global and local costmaps"""
        # In practice, these would be configured based on map and robot parameters
        map_width = 100  # cells
        map_height = 100  # cells
        resolution = 0.05  # meters per cell

        self.global_costmap = Costmap2D(map_width, map_height, resolution)
        self.local_costmap = Costmap2D(50, 50, resolution)  # Smaller local map

    def laser_callback(self, msg):
        """Handle incoming laser scan data"""
        if self.local_costmap:
            # Update local costmap with laser data
            robot_pose = self.get_robot_pose()
            self.local_costmap.update_with_laser_scan(msg, robot_pose)
            self.local_costmap.inflate_obstacles()

    def map_callback(self, msg):
        """Handle incoming map data"""
        if self.global_costmap:
            # Update global costmap with new map
            self.global_costmap.update_with_occupancy_grid(msg)

    def odom_callback(self, msg):
        """Handle odometry updates"""
        # Update robot pose for costmap operations
        pass

    def get_robot_pose(self):
        """Get current robot pose"""
        # This would interface with TF or localization system
        pass

    def get_cost_at_point(self, frame_id: str, x: float, y: float) -> int:
        """Get cost at specific point in specified frame"""
        # Transform point to costmap frame and query cost
        if frame_id == 'map':
            return self.global_costmap.get_cost(x, y) if self.global_costmap else 255
        elif frame_id == 'base_link':
            # Transform to map coordinates first
            # This would use TF transform
            return 0  # Placeholder
        else:
            return 255  # Unknown frame
```

### 4. Dynamic Obstacle Avoidance

#### Obstacle Prediction and Avoidance

```python
from collections import deque

class ObstacleTracker:
    """Track and predict dynamic obstacles"""

    def __init__(self):
        self.obstacles = {}  # obstacle_id -> ObstacleTrack
        self.next_id = 0
        self.max_track_length = 20  # Keep last 20 measurements

    def update_obstacle(self, x: float, y: float, timestamp: float) -> int:
        """Update obstacle position or create new track"""
        # Simple approach: find closest existing obstacle
        closest_id = None
        min_distance = float('inf')

        for obs_id, track in self.obstacles.items():
            last_pos = track.positions[-1] if track.positions else (0, 0)
            distance = math.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)

            if distance < min_distance and distance < 1.0:  # 1m threshold
                min_distance = distance
                closest_id = obs_id

        if closest_id is not None:
            # Update existing track
            self.obstacles[closest_id].add_position((x, y), timestamp)
            return closest_id
        else:
            # Create new obstacle track
            obs_id = self.next_id
            self.next_id += 1
            self.obstacles[obs_id] = ObstacleTrack((x, y), timestamp)
            return obs_id

    def predict_obstacle_motion(self, obs_id: int, time_ahead: float) -> Tuple[float, float]:
        """Predict obstacle position at future time"""
        if obs_id not in self.obstacles:
            return None, None

        track = self.obstacles[obs_id]
        return track.predict_position(time_ahead)

class ObstacleTrack:
    """Track for a single dynamic obstacle"""

    def __init__(self, initial_pos: Tuple[float, float], timestamp: float):
        self.positions = deque(maxlen=20)  # Keep last 20 positions
        self.timestamps = deque(maxlen=20)  # Keep corresponding timestamps
        self.positions.append(initial_pos)
        self.timestamps.append(timestamp)

    def add_position(self, pos: Tuple[float, float], timestamp: float):
        """Add new position measurement"""
        self.positions.append(pos)
        self.timestamps.append(timestamp)

    def predict_position(self, time_ahead: float) -> Tuple[float, float]:
        """Predict position time_ahead seconds into the future"""
        if len(self.positions) < 2:
            # Not enough data for prediction, return current position
            return self.positions[-1] if self.positions else (0, 0)

        # Calculate average velocity from recent positions
        pos1 = self.positions[-2]
        pos2 = self.positions[-1]
        time1 = self.timestamps[-2]
        time2 = self.timestamps[-1]

        dt = time2 - time1
        if dt <= 0:
            return pos2

        vx = (pos2[0] - pos1[0]) / dt
        vy = (pos2[1] - pos1[1]) / dt

        # Predict future position
        predicted_x = pos2[0] + vx * time_ahead
        predicted_y = pos2[1] + vy * time_ahead

        return (predicted_x, predicted_y)

class DynamicAvoidance:
    """Handle dynamic obstacle avoidance"""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.obstacle_tracker = ObstacleTracker()
        self.safe_distance = 0.8  # meters
        self.avoidance_active = False

    def update_obstacles(self, obstacle_positions: List[Tuple[float, float]], timestamp: float):
        """Update with new obstacle detections"""
        for pos in obstacle_positions:
            self.obstacle_tracker.update_obstacle(pos[0], pos[1], timestamp)

    def calculate_avoidance_command(self, robot_state: RobotState) -> Optional[Twist]:
        """Calculate avoidance velocity command if needed"""
        # Check for obstacles in robot's path
        for obs_id in list(self.obstacle_tracker.obstacles.keys()):
            predicted_pos = self.obstacle_tracker.predict_obstacle_motion(obs_id, 0.5)  # 0.5 sec prediction

            if predicted_pos:
                distance = math.sqrt((predicted_pos[0] - robot_state.x)**2 +
                                   (predicted_pos[1] - robot_state.y)**2)

                if distance < self.safe_distance:
                    # Obstacle is too close, calculate avoidance command
                    return self._calculate_evasion_command(robot_state, predicted_pos)

        return None  # No avoidance needed

    def _calculate_evasion_command(self, robot_state: RobotState, obstacle_pos: Tuple[float, float]) -> Twist:
        """Calculate command to avoid obstacle"""
        cmd = Twist()

        # Calculate direction away from obstacle
        dx = robot_state.x - obstacle_pos[0]
        dy = robot_state.y - obstacle_pos[1]

        # Normalize direction vector
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            dx /= distance
            dy /= distance

        # Calculate evasion direction (perpendicular to obstacle direction)
        evasion_x = -dy  # Rotate 90 degrees
        evasion_y = dx

        # Set linear velocity to move away while maintaining forward progress
        cmd.linear.x = max(0.1, min(self.config.max_speed_linear * 0.3,
                                   self.config.max_speed_linear * 0.5))

        # Set angular velocity to turn away from obstacle
        target_angle = math.atan2(evasion_y, evasion_x)
        current_angle = robot_state.theta
        angle_error = target_angle - current_angle

        # Normalize angle error
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        cmd.angular.z = max(-self.config.max_speed_angular * 0.8,
                           min(self.config.max_speed_angular * 0.8,
                               angle_error * 2.0))

        return cmd
```

## Implementation Steps

### Step 1: Set up Navigation System Infrastructure

1. Create the main navigation node:

```python
#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

class NavigationServer:
    def __init__(self):
        rospy.init_node('navigation_server')

        # Initialize components
        self.config = RobotConfig()
        self.global_planner = GlobalPlanner()
        self.local_planner = DWAController(self.config)
        self.path_follower = PathFollower(self.config)
        self.costmap_server = CostmapServer()
        self.dynamic_avoidance = DynamicAvoidance(self.config)

        # State variables
        self.current_goal = None
        self.current_path = []
        self.navigation_active = False
        self.robot_state = RobotState()

        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.path_pub = rospy.Publisher('/current_path', Path, queue_size=10)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)

        rospy.loginfo("Navigation server initialized")

    def goal_callback(self, msg):
        """Handle incoming navigation goal"""
        rospy.loginfo(f"Received navigation goal: ({msg.pose.position.x}, {msg.pose.position.y})")

        # Plan path to goal
        start_pose = self.get_current_pose()
        self.current_path = self.global_planner.plan_path(start_pose, msg.pose)

        if self.current_path:
            # Set the path for following
            self.path_follower.set_path(self.current_path)
            self.current_goal = (msg.pose.position.x, msg.pose.position.y)
            self.navigation_active = True

            # Publish the planned path
            self._publish_path()

            rospy.loginfo(f"Planned path with {len(self.current_path)} waypoints")
        else:
            rospy.logerr("Failed to plan path to goal")

    def odom_callback(self, msg):
        """Update robot state from odometry"""
        self.robot_state.x = msg.pose.pose.position.x
        self.robot_state.y = msg.pose.pose.position.y

        # Convert quaternion to euler for theta
        orientation = msg.pose.pose.orientation
        self.robot_state.theta = math.atan2(2*(orientation.w*orientation.z + orientation.x*orientation.y),
                                           1 - 2*(orientation.y*orientation.y + orientation.z*orientation.z))

        self.robot_state.vx = msg.twist.twist.linear.x
        self.robot_state.vy = msg.twist.twist.linear.y
        self.robot_state.vtheta = msg.twist.twist.angular.z

    def laser_callback(self, msg):
        """Handle laser scan for obstacle detection"""
        # Process laser scan to identify obstacles
        obstacles = self._extract_obstacles_from_scan(msg)
        self.dynamic_avoidance.update_obstacles(obstacles, rospy.get_time())

    def _extract_obstacles_from_scan(self, scan_msg) -> List[Tuple[float, float]]:
        """Extract obstacle positions from laser scan"""
        obstacles = []
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        for i, range_val in enumerate(scan_msg.ranges):
            if scan_msg.range_min <= range_val <= scan_msg.range_max:
                angle = angle_min + i * angle_increment
                # Convert to world coordinates relative to robot
                x = range_val * math.cos(angle) + self.robot_state.x
                y = range_val * math.sin(angle) + self.robot_state.y
                obstacles.append((x, y))

        return obstacles

    def get_current_pose(self):
        """Get current robot pose (simplified)"""
        # This would interface with localization system
        pose = Pose()
        pose.position.x = self.robot_state.x
        pose.position.y = self.robot_state.y
        # Set orientation from theta
        pose.orientation.z = math.sin(self.robot_state.theta / 2)
        pose.orientation.w = math.cos(self.robot_state.theta / 2)
        return pose

    def execute_navigation(self):
        """Main navigation execution logic"""
        if not self.navigation_active or not self.current_path:
            return

        # Check if goal is reached
        if self.path_follower.is_goal_reached():
            self._navigation_complete()
            return

        # Check for dynamic obstacles requiring avoidance
        avoidance_cmd = self.dynamic_avoidance.calculate_avoidance_command(self.robot_state)

        if avoidance_cmd:
            # Execute avoidance maneuver
            self.cmd_vel_pub.publish(avoidance_cmd)
        else:
            # Follow planned path
            cmd = self.path_follower.follow_path()
            self.cmd_vel_pub.publish(cmd)

    def _navigation_complete(self):
        """Handle navigation completion"""
        rospy.loginfo("Navigation goal reached")
        self.navigation_active = False
        self.current_path = []
        self.current_goal = None

        # Publish completion status
        status_pub = rospy.Publisher('/navigation_status', String, queue_size=1, latch=True)
        status_pub.publish("GOAL_REACHED")

    def _publish_path(self):
        """Publish the current path for visualization"""
        if not self.current_path:
            return

        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        for i, (x, y) in enumerate(self.current_path):
            pose = PoseStamped()
            pose.header.seq = i
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def run(self):
        """Main execution loop"""
        rate = rospy.Rate(20)  # 20 Hz navigation loop

        while not rospy.is_shutdown():
            if self.navigation_active:
                self.execute_navigation()

            rate.sleep()

if __name__ == '__main__':
    nav_server = NavigationServer()
    try:
        nav_server.run()
    except rospy.ROSInterruptException:
        pass
```

### Step 2: Configure Navigation Parameters

Create a configuration file for navigation parameters:

```yaml
# navigation_params.yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased.type: "nav2_navfn_planner/NavfnPlanner"
    GridBased:
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

controller_server:
  ros__parameters:
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.5
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.1
    controller_plugins: ["FollowPath"]
    FollowPath.type: "nav2_rotation_shim_controller/RotationShimController"
    FollowPath:
      velocity_deadband: 0.05
      simulate_ahead_time: 1.0
      max_linear_accel: 2.5
      max_linear_decel: 2.5
      max_angular_accel: 3.2
      max_angular_decel: 3.2
      transform_tolerance: 0.1
      min_local_plan_length: 0.5
      lookahead:
        min_distance: 0.3
        max_distance: 0.6

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      origin_x: -1.5
      origin_y: -1.5
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: true
      width: 100
      height: 100
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

### Step 3: Implement Recovery Behaviors

```python
class RecoveryBehaviors:
    """Implement navigation recovery behaviors"""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.current_behavior = None
        self.behavior_start_time = 0.0

    def attempt_recovery(self, recovery_type: str) -> bool:
        """Attempt specified recovery behavior"""
        rospy.loginfo(f"Attempting recovery: {recovery_type}")

        self.current_behavior = recovery_type
        self.behavior_start_time = rospy.get_time()

        if recovery_type == "spin":
            return self._spin_recovery()
        elif recovery_type == "backup":
            return self._backup_recovery()
        elif recovery_type == "dodge":
            return self._dodge_recovery()
        else:
            rospy.logerr(f"Unknown recovery type: {recovery_type}")
            return False

    def _spin_recovery(self) -> bool:
        """Spin in place to clear local minima"""
        cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        cmd = Twist()
        cmd.angular.z = self.config.max_speed_angular * 0.5  # Spin slowly

        start_time = rospy.get_time()
        timeout = 10.0  # 10 second timeout

        rate = rospy.Rate(10)
        while (rospy.get_time() - start_time) < timeout and not rospy.is_shutdown():
            cmd_pub.publish(cmd)
            rate.sleep()

        # Stop spinning
        cmd.angular.z = 0.0
        cmd_pub.publish(cmd)

        return True

    def _backup_recovery(self) -> bool:
        """Back up and try different direction"""
        cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        cmd = Twist()
        cmd.linear.x = -self.config.max_speed_linear * 0.3  # Back up slowly

        start_time = rospy.get_time()
        timeout = 5.0  # 5 second backup

        rate = rospy.Rate(10)
        while (rospy.get_time() - start_time) < timeout and not rospy.is_shutdown():
            cmd_pub.publish(cmd)
            rate.sleep()

        # Stop backing up
        cmd.linear.x = 0.0
        cmd_pub.publish(cmd)

        return True

    def _dodge_recovery(self) -> bool:
        """Attempt to dodge around obstacle"""
        cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Alternate between left and right dodging
        cmd = Twist()
        cmd.linear.x = self.config.max_speed_linear * 0.2  # Move forward slowly
        cmd.angular.z = self.config.max_speed_angular * 0.3  # Turn right

        start_time = rospy.get_time()
        timeout = 8.0  # 8 second dodge

        rate = rospy.Rate(10)
        while (rospy.get_time() - start_time) < timeout and not rospy.is_shutdown():
            cmd_pub.publish(cmd)

            # Alternate turning direction every 2 seconds
            elapsed = rospy.get_time() - start_time
            if int(elapsed) % 4 < 2:
                cmd.angular.z = self.config.max_speed_angular * 0.3  # Turn right
            else:
                cmd.angular.z = -self.config.max_speed_angular * 0.3  # Turn left

            rate.sleep()

        # Stop dodging
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        cmd_pub.publish(cmd)

        return True
```

## Testing and Validation

### Unit Testing

```python
import unittest
from unittest.mock import Mock, patch

class TestAStarPlanner(unittest.TestCase):
    def setUp(self):
        self.grid_map = GridMap(10, 10, 0.1)
        # Add some obstacles to the map
        self.grid_map.grid[5, 5] = 255  # Obstacle in the middle
        self.planner = AStarPlanner(self.grid_map)

    def test_simple_path(self):
        """Test path planning in free space"""
        path = self.planner.plan((0, 0), (9, 9))
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)

    def test_obstacle_avoidance(self):
        """Test path planning around obstacles"""
        path = self.planner.plan((0, 5), (9, 5))
        self.assertIsNotNone(path)
        # Path should avoid the obstacle at (5, 5)
        obstacle_cells = [(x, 5) for x in range(3, 7)]  # Around obstacle
        path_avoids_obstacle = not any(cell in path for cell in obstacle_cells)
        self.assertTrue(path_avoids_obstacle)

class TestDWAController(unittest.TestCase):
    def setUp(self):
        self.config = RobotConfig()
        self.controller = DWAController(self.config)

    def test_dynamic_window_calculation(self):
        """Test dynamic window calculation"""
        # Set some initial velocity
        self.controller.robot_state.vx = 0.2
        self.controller.robot_state.vtheta = 0.1

        vs_min, vs_max, ws_min, ws_max = self.controller.get_dynamic_window()

        # Check that window respects acceleration constraints
        self.assertGreaterEqual(vs_max, 0.2 - self.config.max_accel_linear * self.config.dt)
        self.assertLessEqual(vs_max, 0.2 + self.config.max_accel_linear * self.config.dt)

class TestCostmap2D(unittest.TestCase):
    def setUp(self):
        self.costmap = Costmap2D(10, 10, 0.1)

    def test_obstacle_setting(self):
        """Test setting and getting obstacles"""
        self.costmap.set_obstacle(0.5, 0.5, 200)
        cost = self.costmap.get_cost(0.5, 0.5)
        self.assertEqual(cost, 200)

    def test_inflation(self):
        """Test obstacle inflation"""
        # Set obstacle in center
        self.costmap.set_obstacle(0.5, 0.5, 254)
        self.costmap.inflate_obstacles()

        # Check that surrounding cells have increased cost
        surrounding_costs = []
        for dx in [-0.1, 0, 0.1]:
            for dy in [-0.1, 0, 0.1]:
                cost = self.costmap.get_cost(0.5 + dx, 0.5 + dy)
                surrounding_costs.append(cost)

        # At least some surrounding cells should have cost > 0
        self.assertGreater(sum(c for c in surrounding_costs if c > 0), 0)

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
class NavigationIntegrationTest:
    def __init__(self):
        rospy.init_node('navigation_integration_test')
        self.nav_server = NavigationServer()

    def test_navigation_sequence(self):
        """Test complete navigation sequence"""
        # Setup: Define a simple navigation goal
        goal = (2.0, 2.0)  # 2 meters away

        print("Testing navigation to:", goal)

        # Simulate receiving a goal (this would normally come from subscriber)
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = goal[0]
        pose_msg.pose.position.y = goal[1]

        # Trigger goal callback
        self.nav_server.goal_callback(pose_msg)

        # Simulate execution for a period
        start_time = rospy.get_time()
        timeout = 60.0  # 1 minute timeout

        while (rospy.get_time() - start_time) < timeout:
            self.nav_server.execute_navigation()

            if self.nav_server.path_follower.is_goal_reached():
                print("Goal reached successfully!")
                return True

            rospy.sleep(0.1)

        print("Navigation timed out")
        return False
```

## Performance Benchmarks

### Navigation Performance

- **Global Path Planning**: < 100ms for typical indoor environments (50x50m)
- **Local Path Following**: < 50ms per control cycle at 20Hz
- **Obstacle Detection**: < 30ms for laser scan processing
- **Dynamic Avoidance**: < 20ms per cycle
- **Memory Usage**: < 50MB for typical operation

### Accuracy Requirements

- **Position Accuracy**: < 10cm error for static navigation
- **Obstacle Detection**: > 95% detection rate for obstacles > 20cm
- **Path Following**: < 15cm deviation from planned path
- **Goal Reaching**: < 20cm tolerance for goal achievement

## Troubleshooting and Common Issues

### Path Planning Problems

1. **No Path Found**: Check map quality and inflation parameters
2. **Local Minima**: Implement more sophisticated global planners
3. **Path Quality**: Adjust costmap inflation and planning parameters
4. **Dynamic Replanning**: Handle moving obstacles effectively

### Control Problems

1. **Oscillation**: Adjust controller gains and lookahead distances
2. **Collision**: Increase safety margins and obstacle inflation
3. **Goal Unreachable**: Implement proper recovery behaviors
4. **Drift**: Improve localization accuracy and frequency

## Best Practices

### Safety Considerations

- **Conservative Planning**: Use appropriate safety margins
- **Emergency Stops**: Implement immediate stop capabilities
- **Velocity Limits**: Respect safe speed limits for environment
- **Sensor Validation**: Verify sensor data before acting

### Performance Optimization

- **Efficient Algorithms**: Use optimized path planning algorithms
- **Threading**: Separate planning and control loops appropriately
- **Caching**: Cache frequently computed paths when possible
- **Adaptive Rates**: Adjust update rates based on situation

### Maintainability

- **Modular Design**: Keep path planning, control, and obstacle handling separate
- **Parameter Configuration**: Use ROS parameters for easy tuning
- **Comprehensive Logging**: Log navigation decisions and states
- **Testing Framework**: Maintain extensive test coverage

## Next Steps and Integration

### Integration with Other Capstone Components

The navigation system integrates with:
- **Task Planning**: Receives navigation goals and reports status
- **Perception**: Gets obstacle information and semantic data
- **Manipulation**: Provides precise positioning for tasks
- **Localization**: Maintains accurate position estimates
- **Voice Processing**: Handles location-based commands

### Advanced Features

Consider implementing:
- **Multi-floor Navigation**: Handle navigation across different levels
- **Social Navigation**: Respect human navigation patterns
- **Learning-based Planning**: Adapt to environment-specific patterns
- **Predictive Navigation**: Anticipate dynamic obstacle movements

Continue with [Object Detection and Manipulation](./object-detection.md) to explore the implementation of the perception and manipulation systems that will enable the robot to interact with objects in its environment.

## References

[All sources will be cited in the References section at the end of the book, following APA format]