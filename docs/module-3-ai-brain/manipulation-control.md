---
sidebar_position: 6
---

# Manipulation Control Systems

## Learning Objectives

By the end of this section, you will be able to:

1. Design control systems for robotic manipulators and end-effectors
2. Implement inverse kinematics for multi-joint robotic arms
3. Create grasping and manipulation strategies for various objects
4. Integrate tactile and force feedback for dexterous manipulation
5. Develop control algorithms that combine vision and touch sensing

## Introduction to Manipulation Control

Robotic manipulation is the capability of a robot to physically interact with objects in its environment. This involves perceiving objects, planning manipulation actions, and executing precise control commands to achieve desired outcomes. Manipulation control systems bridge the gap between perception and action, enabling robots to grasp, move, assemble, and manipulate objects with human-like dexterity.

Manipulation tasks can be categorized into several types:

- **Grasping**: Acquiring and holding objects securely
- **Transport**: Moving objects from one location to another
- **Assembly**: Combining parts to create structures or products
- **Deformable manipulation**: Handling flexible or soft materials
- **Tool use**: Using objects as tools to perform tasks

## Robotic Arm Kinematics

### Forward Kinematics

Forward kinematics calculates the end-effector position and orientation given joint angles. For a robotic arm with n joints:

```
T = T1(θ1) * T2(θ2) * ... * Tn(θn)
```

Where T is the transformation matrix representing the end-effector pose, and Ti(θi) represents the transformation due to joint i.

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class ForwardKinematics:
    def __init__(self, dh_parameters):
        """
        Initialize with Denavit-Hartenberg parameters
        dh_parameters: list of [a, alpha, d, theta_offset] for each joint
        """
        self.dh_params = dh_parameters

    def dh_transform(self, a, alpha, d, theta):
        """Calculate Denavit-Hartenberg transformation matrix"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        transform = np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
        return transform

    def calculate_pose(self, joint_angles):
        """Calculate end-effector pose given joint angles"""
        transform = np.eye(4)  # Identity matrix

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            joint_transform = self.dh_transform(a, alpha, d, theta)
            transform = transform @ joint_transform

        # Extract position and orientation
        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]
        orientation = R.from_matrix(rotation_matrix).as_quat()

        return position, orientation
```

### Inverse Kinematics

Inverse kinematics (IK) solves the more complex problem of finding joint angles to achieve a desired end-effector pose. This is essential for manipulation control.

```python
class InverseKinematics:
    def __init__(self, robot_model):
        self.robot = robot_model

    def analytical_ik(self, target_pose, current_joints):
        """
        Analytical inverse kinematics for simple robot configurations
        For a 6-DOF robot, this would involve geometric solutions
        """
        # This is a simplified example for a 2-DOF planar arm
        target_pos = target_pose[:2]  # x, y position
        current_pos = self.forward_kinematics(current_joints)[:2]

        # Calculate joint angles using geometric relationships
        x, y = target_pos
        l1, l2 = self.robot.link_lengths  # Link lengths

        # Distance from base to target
        r = np.sqrt(x**2 + y**2)

        # Check if target is reachable
        if r > l1 + l2:
            # Target is out of reach, extend towards it
            scale = (l1 + l2) / r
            x *= scale
            y *= scale
            r = l1 + l2
        elif r < abs(l1 - l2):
            # Target is inside workspace, extend towards it
            scale = abs(l1 - l2) / r
            x *= scale
            y *= scale
            r = abs(l1 - l2)

        # Calculate joint angles
        cos_angle2 = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
        angle2 = np.arccos(np.clip(cos_angle2, -1, 1))

        k1 = l1 + l2 * np.cos(angle2)
        k2 = l2 * np.sin(angle2)
        angle1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        return [angle1, angle2]

    def numerical_ik(self, target_pose, current_joints, max_iterations=100, tolerance=1e-4):
        """
        Numerical inverse kinematics using Jacobian transpose method
        """
        joints = np.array(current_joints)

        for i in range(max_iterations):
            # Calculate current end-effector pose
            current_pose = self.forward_kinematics(joints)

            # Calculate error
            error = target_pose - current_pose
            if np.linalg.norm(error) < tolerance:
                return joints

            # Calculate Jacobian
            jacobian = self.calculate_jacobian(joints)

            # Update joint angles using Jacobian transpose
            joints_delta = np.linalg.pinv(jacobian) @ error
            joints += 0.1 * joints_delta  # Learning rate

        return joints

    def calculate_jacobian(self, joints):
        """Calculate geometric Jacobian matrix"""
        # Calculate Jacobian using partial derivatives
        # This is a simplified version - full implementation would be more complex
        pass

    def forward_kinematics(self, joints):
        """Calculate forward kinematics (simplified)"""
        # Implementation would depend on specific robot model
        pass
```

## Grasping Strategies

### Grasp Planning

Grasp planning involves determining where and how to grasp an object for stable manipulation:

```python
class GraspPlanner:
    def __init__(self, robot_gripper):
        self.gripper = robot_gripper
        self.approach_directions = [
            [0, 0, 1],   # From above
            [0, 0, -1],  # From below
            [1, 0, 0],   # From side
            [-1, 0, 0],  # From opposite side
            [0, 1, 0],   # From front
            [0, -1, 0]   # From back
        ]

    def plan_grasp(self, object_mesh, object_pose):
        """Plan stable grasp for given object"""
        # Extract object features from mesh
        contact_points = self.find_contact_points(object_mesh)
        grasp_candidates = []

        for point in contact_points:
            for approach_dir in self.approach_directions:
                grasp = self.evaluate_grasp(point, approach_dir, object_mesh, object_pose)
                if grasp and grasp['quality'] > 0.5:  # Minimum quality threshold
                    grasp_candidates.append(grasp)

        # Sort by quality and return best grasp
        if grasp_candidates:
            best_grasp = max(grasp_candidates, key=lambda g: g['quality'])
            return best_grasp

        return None

    def find_contact_points(self, object_mesh):
        """Find potential contact points on object surface"""
        # This would typically use mesh analysis or point cloud processing
        # For simplicity, return some sample points
        return [
            [0.1, 0.1, 0.1],
            [-0.1, 0.1, 0.1],
            [0.1, -0.1, 0.1],
            [-0.1, -0.1, 0.1]
        ]

    def evaluate_grasp(self, contact_point, approach_dir, object_mesh, object_pose):
        """Evaluate grasp quality at given contact point"""
        # Transform contact point to world coordinates
        world_point = self.transform_point(contact_point, object_pose)

        # Check if approach direction is feasible
        if not self.is_approach_feasible(world_point, approach_dir):
            return None

        # Calculate grasp quality metrics
        quality = self.calculate_grasp_quality(
            world_point, approach_dir, object_mesh, object_pose
        )

        return {
            'position': world_point,
            'approach_direction': approach_dir,
            'grasp_quality': quality,
            'gripper_width': self.calculate_gripper_width(contact_point, object_mesh)
        }

    def calculate_grasp_quality(self, contact_point, approach_dir, object_mesh, object_pose):
        """Calculate grasp quality metric"""
        # Quality factors:
        # - Force closure (ability to resist external forces)
        # - Grasp stability
        # - Accessibility
        # - Object properties (friction, weight)

        # Simplified quality calculation
        quality = 0.7  # Default quality
        return quality

    def transform_point(self, point, pose):
        """Transform point from object frame to world frame"""
        # Apply rotation and translation from pose
        rotation_matrix = R.from_quat(pose.orientation).as_matrix()
        world_point = rotation_matrix @ point + pose.position
        return world_point

    def is_approach_feasible(self, point, approach_dir):
        """Check if approach direction is kinematically feasible"""
        # Check if approach direction would cause self-collision
        # Check if approach direction is within joint limits
        return True  # Simplified for example

    def calculate_gripper_width(self, contact_point, object_mesh):
        """Calculate required gripper width for grasp"""
        # This would analyze the object geometry at the contact point
        return 0.05  # 5cm default
```

### Adaptive Grasping

Adaptive grasping adjusts grip force and strategy based on object properties:

```python
class AdaptiveGraspController:
    def __init__(self, gripper, force_sensor):
        self.gripper = gripper
        self.force_sensor = force_sensor
        self.current_object_properties = None

    def execute_adaptive_grasp(self, grasp_pose, object_properties):
        """Execute grasp with adaptive force control"""
        self.current_object_properties = object_properties

        # Move to pre-grasp position
        pre_grasp_pose = self.calculate_pre_grasp_pose(grasp_pose)
        self.move_to_pose(pre_grasp_pose)

        # Approach object
        self.approach_object(grasp_pose)

        # Close gripper with adaptive force
        grip_force = self.calculate_adaptive_force(object_properties)
        self.close_gripper_with_force(grip_force)

        # Verify grasp success
        if self.verify_grasp_success():
            return True
        else:
            return False

    def calculate_adaptive_force(self, object_properties):
        """Calculate appropriate grip force based on object properties"""
        # Factors affecting grip force:
        # - Object weight
        # - Surface friction
        # - Object fragility
        # - Desired safety margin

        base_force = 5.0  # Base grip force in Newtons

        # Adjust for object weight
        weight_factor = object_properties.get('weight', 1.0) * 9.81  # weight * gravity
        base_force += weight_factor * 0.5  # Add 50% of weight as additional force

        # Adjust for fragility
        fragility_factor = object_properties.get('fragility', 1.0)
        if fragility_factor < 0.5:
            # Fragile object - use minimum force
            base_force = min(base_force, 2.0)
        elif fragility_factor > 1.5:
            # Robust object - can use higher force
            base_force *= 1.5

        # Adjust for surface friction
        friction_factor = object_properties.get('friction', 0.5)
        if friction_factor < 0.3:
            # Low friction - need higher grip force
            base_force *= 1.8
        elif friction_factor > 0.7:
            # High friction - can use lower force
            base_force *= 0.7

        # Ensure within gripper limits
        max_force = self.gripper.max_force
        min_force = self.gripper.min_force
        grip_force = np.clip(base_force, min_force, max_force)

        return grip_force

    def verify_grasp_success(self):
        """Verify that grasp was successful"""
        # Check force sensor readings
        force_readings = self.force_sensor.get_readings()

        # Look for characteristic patterns of successful grasp
        if len(force_readings) > 10:
            recent_forces = force_readings[-10:]
            avg_force = np.mean(recent_forces)

            # Successful grasp typically shows stable force readings
            if avg_force > 0.5 and np.std(recent_forces) < 0.2:
                return True

        return False

    def calculate_pre_grasp_pose(self, grasp_pose):
        """Calculate safe pre-grasp pose"""
        # Move 10cm away from grasp point along approach direction
        approach_dir = grasp_pose['approach_direction']
        pre_grasp_offset = np.array(approach_dir) * 0.1  # 10cm offset
        pre_grasp_pos = np.array(grasp_pose['position']) - pre_grasp_offset

        return {
            'position': pre_grasp_pos,
            'orientation': grasp_pose['orientation']
        }

    def approach_object(self, grasp_pose):
        """Approach object with controlled motion"""
        # Use Cartesian impedance control for safe approach
        self.move_to_pose(grasp_pose, control_mode='impedance')

    def close_gripper_with_force(self, target_force):
        """Close gripper while monitoring applied force"""
        current_force = 0
        while current_force < target_force:
            self.gripper.close_incrementally()
            current_force = self.force_sensor.get_gripper_force()

            if current_force > target_force:
                # Apply force control to maintain target force
                self.gripper.set_force_control(target_force)
                break
```

## Force Control and Compliance

### Impedance Control

Impedance control makes the robot behave like a virtual spring-damper system, providing compliant behavior:

```python
class ImpedanceController:
    def __init__(self, robot, stiffness=1000, damping=20):
        self.robot = robot
        self.stiffness = stiffness  # N/m
        self.damping = damping      # Ns/m
        self.desired_pose = None
        self.mass = 1.0  # Effective mass (kg)

    def set_desired_pose(self, pose):
        """Set desired pose for impedance control"""
        self.desired_pose = pose

    def compute_impedance_force(self, current_pose, current_velocity):
        """Compute impedance force based on position and velocity errors"""
        if self.desired_pose is None:
            return np.zeros(6)  # 6 DOF: 3 for position, 3 for orientation

        # Calculate position and velocity errors
        pos_error = current_pose[:3] - self.desired_pose[:3]
        vel_error = current_velocity[:3]

        # Compute impedance force: F = K * error_pos + D * error_vel
        position_force = -self.stiffness * pos_error
        velocity_force = -self.damping * vel_error

        # Combine forces
        total_force = position_force + velocity_force

        # For orientation, similar approach with rotation errors
        orientation_error = self.calculate_rotation_error(
            current_pose[3:], self.desired_pose[3:]
        )
        orientation_force = -self.stiffness * orientation_error

        return np.concatenate([total_force, orientation_force])

    def calculate_rotation_error(self, current_quat, desired_quat):
        """Calculate rotation error between two quaternions"""
        # Convert quaternions to rotation vectors for error calculation
        current_rot = R.from_quat(current_quat).as_rotvec()
        desired_rot = R.from_quat(desired_quat).as_rotvec()
        return desired_rot - current_rot

    def update_impedance_parameters(self, stiffness=None, damping=None):
        """Update impedance parameters during operation"""
        if stiffness is not None:
            self.stiffness = stiffness
        if damping is not None:
            self.damping = damping
```

### Hybrid Position/Force Control

Hybrid control combines position and force control for complex manipulation tasks:

```python
class HybridPositionForceController:
    def __init__(self, robot, contact_surface_normal=None):
        self.robot = robot
        self.contact_surface_normal = contact_surface_normal or np.array([0, 0, 1])
        self.position_gains = {'P': 100, 'I': 0.1, 'D': 10}
        self.force_gains = {'P': 50, 'I': 0.05, 'D': 5}

    def hybrid_control(self, desired_pos, desired_force, current_pos, current_force):
        """
        Implement hybrid position/force control
        Position control along unconstrained directions
        Force control along constrained directions
        """
        # Define constraint matrix based on contact surface
        constraint_matrix = self.calculate_constraint_matrix()

        # Calculate errors
        pos_error = desired_pos - current_pos
        force_error = desired_force - current_force

        # Apply control in constrained vs unconstrained directions
        control_commands = np.zeros(6)  # 6 DOF

        for i in range(6):
            if constraint_matrix[i] == 0:  # Position controlled direction
                control_commands[i] = (
                    self.position_gains['P'] * pos_error[i] +
                    self.position_gains['I'] * self.integrate_pos_error[i] +
                    self.position_gains['D'] * self.derivative_pos_error[i]
                )
            else:  # Force controlled direction
                control_commands[i] = (
                    self.force_gains['P'] * force_error[i] +
                    self.force_gains['I'] * self.integrate_force_error[i] +
                    self.force_gains['D'] * self.derivative_force_error[i]
                )

        return control_commands

    def calculate_constraint_matrix(self):
        """
        Calculate constraint matrix based on contact geometry
        0 = position controlled, 1 = force controlled
        """
        # For planar contact, typically constrain normal force, allow tangential motion
        constraint_matrix = np.array([0, 0, 1, 0, 0, 0])  # Constrain z-force, allow x,y position
        return constraint_matrix
```

## Isaac ROS Manipulation Integration

### Isaac ROS Manipulation Packages

Isaac provides specialized packages for manipulation tasks:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker

class IsaacManipulationController(Node):
    def __init__(self):
        super().__init__('isaac_manipulation_controller')

        # Publishers for Isaac manipulation
        self.joint_command_publisher = self.create_publisher(
            JointState, 'isaac_manipulator/joint_commands', 10
        )

        self.trajectory_publisher = self.create_publisher(
            JointState, 'isaac_manipulator/trajectory_commands', 10
        )

        self.visualization_publisher = self.create_publisher(
            Marker, 'manipulation_visualization', 10
        )

        # Subscribers
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'isaac_manipulator/joint_states', self.joint_state_callback, 10
        )

        self.ee_pose_subscriber = self.create_subscription(
            Pose, 'isaac_manipulator/ee_pose', self.ee_pose_callback, 10
        )

        # Isaac-specific manipulation services
        self.ik_service_client = self.create_client(
            ComputeIK, 'isaac_compute_ik'
        )

        self.grasp_service_client = self.create_client(
            ExecuteGrasp, 'isaac_execute_grasp'
        )

        # Initialize controllers
        self.current_joint_state = None
        self.current_ee_pose = None
        self.impedance_controller = ImpedanceController(self)

    def joint_state_callback(self, msg):
        """Handle incoming joint state messages"""
        self.current_joint_state = msg

    def ee_pose_callback(self, msg):
        """Handle incoming end-effector pose messages"""
        self.current_ee_pose = msg

    def move_to_pose(self, target_pose, method='ik'):
        """Move manipulator to target pose using inverse kinematics"""
        if method == 'ik':
            # Use Isaac's IK service
            if self.ik_service_client.wait_for_service(timeout_sec=1.0):
                request = ComputeIK.Request()
                request.target_pose = target_pose
                request.current_joints = self.current_joint_state

                future = self.ik_service_client.call_async(request)
                future.add_done_callback(self.ik_response_callback)
            else:
                self.get_logger().error('IK service not available')

    def ik_response_callback(self, future):
        """Handle IK service response"""
        try:
            response = future.result()
            if response.success:
                # Execute joint trajectory
                self.execute_joint_trajectory(response.joint_angles)
            else:
                self.get_logger().error('IK solution not found')
        except Exception as e:
            self.get_logger().error(f'IK service call failed: {e}')

    def execute_grasp(self, grasp_pose, object_properties):
        """Execute grasp using Isaac's manipulation capabilities"""
        if self.grasp_service_client.wait_for_service(timeout_sec=1.0):
            request = ExecuteGrasp.Request()
            request.grasp_pose = grasp_pose
            request.object_properties = self.dict_to_msg(object_properties)

            future = self.grasp_service_client.call_async(request)
            future.add_done_callback(self.grasp_response_callback)
        else:
            self.get_logger().error('Grasp service not available')

    def execute_joint_trajectory(self, joint_angles, duration=5.0):
        """Execute joint trajectory to reach target joint angles"""
        trajectory_msg = JointState()
        trajectory_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        trajectory_msg.position = joint_angles
        trajectory_msg.velocity = [0.0] * len(joint_angles)  # Start/stop with zero velocity
        trajectory_msg.effort = [0.0] * len(joint_angles)

        # Publish trajectory command
        self.trajectory_publisher.publish(trajectory_msg)

    def visualize_grasp_candidate(self, grasp_pose):
        """Visualize grasp candidate in RViz"""
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'grasp_candidates'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Set arrow properties to represent grasp
        marker.pose = grasp_pose
        marker.scale.x = 0.1  # Arrow length
        marker.scale.y = 0.02  # Arrow width
        marker.scale.z = 0.02  # Arrow height
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.visualization_publisher.publish(marker)

    def dict_to_msg(self, properties_dict):
        """Convert dictionary to ROS message (simplified)"""
        # This would convert the dictionary to appropriate ROS message type
        pass
```

## Tactile Sensing Integration

### Tactile Feedback for Dexterity

Tactile sensors provide crucial feedback for dexterous manipulation:

```python
class TactileFeedbackController:
    def __init__(self, tactile_sensors, gripper):
        self.tactile_sensors = tactile_sensors
        self.gripper = gripper
        self.contact_threshold = 0.1  # Threshold for contact detection
        self.slip_detection_threshold = 0.5

    def monitor_tactile_feedback(self):
        """Monitor tactile sensors for contact and slip detection"""
        tactile_data = self.tactile_sensors.get_readings()

        # Detect contact
        contact_detected = self.detect_contact(tactile_data)

        # Detect slip
        slip_detected = self.detect_slip(tactile_data)

        return {
            'contact': contact_detected,
            'slip': slip_detected,
            'pressure_map': tactile_data['pressure'],
            'temperature': tactile_data['temperature']
        }

    def detect_contact(self, tactile_data):
        """Detect if object is in contact with tactile sensors"""
        pressure_values = tactile_data['pressure']
        max_pressure = np.max(pressure_values)
        return max_pressure > self.contact_threshold

    def detect_slip(self, tactile_data):
        """Detect slip based on tactile sensor readings"""
        # Slip detection typically involves:
        # - Sudden changes in pressure distribution
        # - High-frequency vibrations
        # - Asymmetric pressure patterns
        pressure_changes = np.diff(tactile_data['pressure'], axis=0)
        max_change = np.max(np.abs(pressure_changes))
        return max_change > self.slip_detection_threshold

    def adaptive_grasp_control(self, target_force):
        """Adjust grip force based on tactile feedback"""
        feedback = self.monitor_tactile_feedback()

        if feedback['slip']:
            # Increase grip force if slip detected
            current_force = self.gripper.get_current_force()
            new_force = min(current_force * 1.2, self.gripper.max_force)
            self.gripper.set_force(new_force)
        elif feedback['contact']:
            # Maintain stable grip force
            current_force = self.gripper.get_current_force()
            if abs(current_force - target_force) > 0.1:
                self.gripper.set_force(target_force)

    def slip_compensation(self):
        """Compensate for detected slip by adjusting grip strategy"""
        feedback = self.monitor_tactile_feedback()

        if feedback['slip']:
            # Adjust grip strategy
            self.gripper.increase_force_gradually()
            # Consider changing grasp strategy if slip persists
            return True

        return False
```

## Manipulation Task Planning

### Task and Motion Planning (TAMP)

Combining high-level task planning with low-level motion planning:

```python
class TaskAndMotionPlanner:
    def __init__(self, symbolic_planner, motion_planner):
        self.symbolic_planner = symbolic_planner
        self.motion_planner = motion_planner
        self.known_objects = {}
        self.robot_state = {}

    def plan_manipulation_task(self, task_description):
        """
        Plan manipulation task combining symbolic and motion planning
        Example task: "Pick up red block and place it on blue box"
        """
        # Step 1: Symbolic task planning
        symbolic_plan = self.symbolic_planner.plan(task_description)

        # Step 2: Ground symbolic actions to motion plans
        motion_plan = []
        for symbolic_action in symbolic_plan:
            motion_primitive = self.ground_action(symbolic_action)
            if motion_primitive:
                motion_plan.append(motion_primitive)

        return motion_plan

    def ground_action(self, symbolic_action):
        """Ground symbolic action to concrete motion plan"""
        action_type = symbolic_action['type']
        parameters = symbolic_action['parameters']

        if action_type == 'pick':
            return self.plan_pick_action(parameters)
        elif action_type == 'place':
            return self.plan_place_action(parameters)
        elif action_type == 'move_to':
            return self.plan_move_action(parameters)
        else:
            return None

    def plan_pick_action(self, params):
        """Plan pick action with grasp planning and trajectory generation"""
        object_name = params['object']
        object_pose = self.known_objects.get(object_name)

        if not object_pose:
            return None

        # Plan grasp for object
        grasp_planner = GraspPlanner(self.robot_state['gripper'])
        grasp = grasp_planner.plan_grasp(object_pose['mesh'], object_pose['pose'])

        if not grasp:
            return None

        # Plan approach trajectory
        approach_traj = self.motion_planner.plan_trajectory(
            start_pose=self.robot_state['ee_pose'],
            goal_pose=grasp['approach_pose']
        )

        # Plan grasp execution trajectory
        grasp_traj = self.motion_planner.plan_trajectory(
            start_pose=grasp['approach_pose'],
            goal_pose=grasp['grasp_pose']
        )

        return {
            'action': 'pick',
            'object': object_name,
            'grasp': grasp,
            'trajectories': [approach_traj, grasp_traj],
            'gripper_command': 'close'
        }

    def plan_place_action(self, params):
        """Plan place action with placement planning"""
        target_location = params['location']
        object_name = params['object']

        # Plan placement pose
        placement_pose = self.calculate_placement_pose(target_location, object_name)

        # Plan approach trajectory
        approach_traj = self.motion_planner.plan_trajectory(
            start_pose=self.robot_state['ee_pose'],
            goal_pose=placement_pose['approach_pose']
        )

        # Plan placement trajectory
        place_traj = self.motion_planner.plan_trajectory(
            start_pose=placement_pose['approach_pose'],
            goal_pose=placement_pose['place_pose']
        )

        return {
            'action': 'place',
            'object': object_name,
            'placement_pose': placement_pose,
            'trajectories': [approach_traj, place_traj],
            'gripper_command': 'open'
        }

    def calculate_placement_pose(self, target_location, object_name):
        """Calculate safe placement pose"""
        # This would consider object dimensions, target surface stability, etc.
        object_dims = self.known_objects[object_name]['dimensions']
        safe_height = object_dims[2] / 2  # Half object height above surface

        placement_pose = {
            'position': [target_location['x'], target_location['y'], target_location['z'] + safe_height],
            'orientation': [0, 0, 0, 1]  # Default orientation
        }

        # Calculate approach pose (above placement position)
        approach_pose = {
            'position': [target_location['x'], target_location['y'], target_location['z'] + safe_height + 0.1],
            'orientation': [0, 0, 0, 1]
        }

        return {
            'place_pose': placement_pose,
            'approach_pose': approach_pose
        }
```

## Summary

Manipulation control systems enable robots to interact with their environment through precise control of robotic arms and end-effectors. By combining kinematic models, grasp planning, force control, and tactile feedback, robots can perform complex manipulation tasks with dexterity and safety.

The next section will cover GPU optimization techniques, which are essential for accelerating the computation-intensive algorithms used in manipulation control.

## References

[All sources will be cited in the References section at the end of the book, following APA format]