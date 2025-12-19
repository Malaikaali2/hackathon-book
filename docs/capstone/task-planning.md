---
sidebar_position: 17
---

# Task Planning and Execution: Autonomous Humanoid Capstone

## Overview

Task planning and execution represents the cognitive layer of the autonomous humanoid system, bridging high-level goals from voice commands with low-level robot actions. This component decomposes complex tasks into executable sequences, manages dependencies between actions, handles dynamic replanning when obstacles arise, and ensures safe and efficient task completion. The system must handle both simple single-step tasks and complex multi-step operations while maintaining awareness of the robot's capabilities and environmental constraints.

The task planning system integrates seamlessly with voice processing to interpret user commands, with navigation for mobility tasks, with manipulation for object interaction, and with perception for environmental awareness. This implementation guide provides detailed instructions for building a sophisticated task planning and execution framework that can handle the complexity of real-world scenarios.

## System Architecture

### Hierarchical Task Network (HTN) Architecture

The task planning system employs a Hierarchical Task Network (HTN) approach that decomposes high-level goals into sequences of primitive actions:

```
High-Level Goal → Task Decomposition → Action Sequencing → Execution Monitoring → Feedback Integration
```

The architecture consists of:
1. **Goal Parser**: Interprets goals from voice processing system
2. **Task Decomposer**: Breaks complex tasks into subtasks
3. **Action Sequencer**: Orders actions based on dependencies
4. **Executor**: Coordinates with other systems to execute actions
5. **Monitor**: Tracks execution progress and detects failures
6. **Replanner**: Adjusts plans when failures or new information occur

### Integration with Other Systems

The task planning system interfaces with:
- **Voice Processing**: Receives high-level goals and commands
- **Navigation System**: Plans and executes mobility tasks
- **Manipulation System**: Coordinates object interaction tasks
- **Perception System**: Requests environmental information
- **Knowledge Base**: Accesses object properties and location information

## Technical Implementation

### 1. Task Representation and Decomposition

#### Task Structure Definition

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import rospy

class TaskType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMPOSITE = "composite"
    QUERY = "query"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Represents a single task in the planning system"""
    id: str
    type: TaskType
    goal: Dict[str, Any]
    dependencies: List[str]  # Task IDs this task depends on
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_result: Optional[Dict[str, Any]] = None

class TaskDecomposer:
    """Decomposes high-level goals into primitive tasks"""

    def __init__(self):
        self.task_library = self._initialize_task_library()

    def _initialize_task_library(self) -> Dict[str, List[Task]]:
        """Initialize library of known task decompositions"""
        return {
            'go_to_location': [
                Task(
                    id='navigation_task',
                    type=TaskType.NAVIGATION,
                    goal={'target_location': None},
                    dependencies=[]
                )
            ],
            'pick_up_object': [
                Task(
                    id='approach_object',
                    type=TaskType.NAVIGATION,
                    goal={'target_location': None, 'approach_distance': 0.5},
                    dependencies=[]
                ),
                Task(
                    id='detect_object',
                    type=TaskType.PERCEPTION,
                    goal={'object_type': None, 'search_area': None},
                    dependencies=['approach_object']
                ),
                Task(
                    id='grasp_object',
                    type=TaskType.MANIPULATION,
                    goal={'object_id': None, 'grasp_type': 'pinch'},
                    dependencies=['detect_object']
                )
            ],
            'place_object': [
                Task(
                    id='navigate_to_place',
                    type=TaskType.NAVIGATION,
                    goal={'target_location': None},
                    dependencies=[]
                ),
                Task(
                    id='place_object_action',
                    type=TaskType.MANIPULATION,
                    goal={'placement_location': None, 'object_to_place': None},
                    dependencies=['navigate_to_place']
                )
            ]
        }

    def decompose_goal(self, goal: Dict[str, Any]) -> List[Task]:
        """Decompose a high-level goal into primitive tasks"""
        goal_type = goal.get('type')
        goal_params = goal.get('parameters', {})

        if goal_type in self.task_library:
            # Create task copies with specific parameters
            tasks = []
            for template_task in self.task_library[goal_type]:
                task = Task(
                    id=f"{template_task.id}_{rospy.get_time()}",
                    type=template_task.type,
                    goal=self._substitute_parameters(template_task.goal, goal_params),
                    dependencies=template_task.dependencies,
                    priority=goal.get('priority', 0)
                )
                tasks.append(task)

            return tasks
        else:
            # Default decomposition for unknown goals
            return self._default_decomposition(goal)

    def _substitute_parameters(self, template_goal: Dict, params: Dict) -> Dict:
        """Substitute template parameters with actual values"""
        result = template_goal.copy()
        for key, value in params.items():
            if key in result:
                result[key] = value
        return result

    def _default_decomposition(self, goal: Dict[str, Any]) -> List[Task]:
        """Default decomposition for unknown goal types"""
        # For now, return a single composite task
        return [Task(
            id=f"composite_{rospy.get_time()}",
            type=TaskType.COMPOSITE,
            goal=goal,
            dependencies=[],
            priority=goal.get('priority', 0)
        )]
```

#### Task Planning Algorithm

```python
from collections import defaultdict, deque
import heapq

class TaskPlanner:
    """Manages the planning and scheduling of tasks"""

    def __init__(self):
        self.tasks = {}  # task_id -> Task
        self.dependency_graph = defaultdict(list)  # task_id -> [dependent_task_ids]
        self.ready_queue = []  # Priority queue of ready tasks
        self.executor = TaskExecutor()

    def add_tasks(self, tasks: List[Task]):
        """Add tasks to the planning system"""
        for task in tasks:
            self.tasks[task.id] = task

            # Build dependency graph
            for dep_id in task.dependencies:
                self.dependency_graph[dep_id].append(task.id)

        # Add tasks with no dependencies to ready queue
        self._update_ready_queue()

    def _update_ready_queue(self):
        """Update the ready queue based on task dependencies and status"""
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING and self._are_dependencies_met(task_id):
                heapq.heappush(self.ready_queue, (-task.priority, task_id, task))

    def _are_dependencies_met(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied"""
        task = self.tasks[task_id]
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            if self.tasks[dep_id].status != TaskStatus.SUCCESS:
                return False
        return True

    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute based on priority"""
        if self.ready_queue:
            priority, task_id, task = heapq.heappop(self.ready_queue)
            return task
        return None

    def update_task_status(self, task_id: str, status: TaskStatus, result: Optional[Dict] = None):
        """Update the status of a task and propagate changes"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            task.execution_result = result

            # Update ready queue with newly available tasks
            self._update_ready_queue()

            # Check if dependent tasks can now be executed
            for dependent_id in self.dependency_graph[task_id]:
                if self._are_dependencies_met(dependent_id):
                    dependent_task = self.tasks[dependent_id]
                    heapq.heappush(self.ready_queue, (-dependent_task.priority, dependent_id, dependent_task))

    def cancel_task(self, task_id: str):
        """Cancel a pending task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
```

### 2. Action Execution System

#### Task Execution Framework

```python
import threading
import time
from typing import Callable

class TaskExecutor:
    """Executes tasks by interfacing with other robot systems"""

    def __init__(self):
        # Publishers for different system interfaces
        self.navigation_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.manipulation_pub = rospy.Publisher('/manipulation_commands', String, queue_size=10)
        self.perception_pub = rospy.Publisher('/perception_commands', String, queue_size=10)

        # Subscribers for feedback
        self.navigation_sub = rospy.Subscriber('/move_base/status', String, self._navigation_callback)
        self.manipulation_sub = rospy.Subscriber('/manipulation/status', String, self._manipulation_callback)

        self.active_tasks = {}  # task_id -> execution thread
        self.execution_lock = threading.Lock()

    def execute_task(self, task: Task, callback: Callable[[str, TaskStatus, Optional[Dict]], None]):
        """Execute a single task asynchronously"""
        with self.execution_lock:
            thread = threading.Thread(
                target=self._execute_task_thread,
                args=(task, callback)
            )
            thread.daemon = True
            thread.start()
            self.active_tasks[task.id] = thread

        return True

    def _execute_task_thread(self, task: Task, callback: Callable):
        """Execute task in separate thread"""
        try:
            task.start_time = rospy.get_time()

            # Execute based on task type
            if task.type == TaskType.NAVIGATION:
                result = self._execute_navigation_task(task)
            elif task.type == TaskType.MANIPULATION:
                result = self._execute_manipulation_task(task)
            elif task.type == TaskType.PERCEPTION:
                result = self._execute_perception_task(task)
            elif task.type == TaskType.COMPOSITE:
                result = self._execute_composite_task(task)
            else:
                result = {'success': False, 'error': f'Unknown task type: {task.type}'}

            # Determine status based on result
            status = TaskStatus.SUCCESS if result.get('success', False) else TaskStatus.FAILED

            # Update task completion time
            task.end_time = rospy.get_time()
            task.execution_result = result

            # Call completion callback
            callback(task.id, status, result)

        except Exception as e:
            rospy.logerr(f"Error executing task {task.id}: {e}")
            callback(task.id, TaskStatus.FAILED, {'success': False, 'error': str(e)})
        finally:
            with self.execution_lock:
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]

    def _execute_navigation_task(self, task: Task) -> Dict[str, Any]:
        """Execute navigation task"""
        try:
            target_location = task.goal.get('target_location')
            if not target_location:
                return {'success': False, 'error': 'No target location specified'}

            # Convert location name to coordinates
            coordinates = self._get_coordinates_for_location(target_location)
            if not coordinates:
                return {'success': False, 'error': f'Unknown location: {target_location}'}

            # Create and publish navigation goal
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x = coordinates['x']
            goal.pose.position.y = coordinates['y']
            goal.pose.orientation.z = coordinates.get('theta', 0.0)

            self.navigation_pub.publish(goal)

            # Wait for navigation to complete (with timeout)
            start_time = rospy.get_time()
            timeout = task.goal.get('timeout', 60.0)  # Default 60 second timeout

            while not self._is_navigation_complete() and (rospy.get_time() - start_time) < timeout:
                rospy.sleep(0.1)

            if (rospy.get_time() - start_time) >= timeout:
                return {'success': False, 'error': 'Navigation timeout'}

            return {'success': True, 'location': target_location}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_manipulation_task(self, task: Task) -> Dict[str, Any]:
        """Execute manipulation task"""
        try:
            # Create manipulation command based on goal
            command = {
                'action': task.goal.get('action', 'grasp'),
                'object': task.goal.get('object_id'),
                'location': task.goal.get('target_location'),
                'grasp_type': task.goal.get('grasp_type', 'default')
            }

            # Publish manipulation command
            self.manipulation_pub.publish(str(command))

            # Wait for manipulation to complete
            start_time = rospy.get_time()
            timeout = task.goal.get('timeout', 30.0)

            while not self._is_manipulation_complete() and (rospy.get_time() - start_time) < timeout:
                rospy.sleep(0.1)

            if (rospy.get_time() - start_time) >= timeout:
                return {'success': False, 'error': 'Manipulation timeout'}

            return {'success': True, 'action': command['action']}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_perception_task(self, task: Task) -> Dict[str, Any]:
        """Execute perception task"""
        try:
            # Create perception command
            command = {
                'task': task.goal.get('task', 'detect'),
                'object_type': task.goal.get('object_type'),
                'search_area': task.goal.get('search_area')
            }

            # Publish perception command
            self.perception_pub.publish(str(command))

            # Wait for perception results
            start_time = rospy.get_time()
            timeout = task.goal.get('timeout', 10.0)

            results = self._wait_for_perception_results(timeout)

            if not results:
                return {'success': False, 'error': 'Perception timeout or no results'}

            return {'success': True, 'results': results}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_composite_task(self, task: Task) -> Dict[str, Any]:
        """Execute composite task by decomposing and executing subtasks"""
        try:
            # For composite tasks, we might need to decompose further
            # This is a simplified version - in practice, this might trigger
            # a recursive decomposition
            subtasks = self._decompose_composite_task(task)

            for subtask in subtasks:
                result = self._execute_single_task_sync(subtask)
                if not result.get('success', False):
                    return result

            return {'success': True, 'subtasks_completed': len(subtasks)}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _get_coordinates_for_location(self, location_name: str) -> Optional[Dict]:
        """Convert location name to coordinates"""
        # This would typically come from a knowledge base or map
        location_map = {
            'kitchen': {'x': 2.0, 'y': 1.0, 'theta': 0.0},
            'living_room': {'x': -1.0, 'y': 2.0, 'theta': 1.57},
            'bedroom': {'x': 3.0, 'y': -2.0, 'theta': 3.14},
            'office': {'x': -2.0, 'y': -1.0, 'theta': -1.57}
        }

        return location_map.get(location_name.lower().replace(' ', '_'))

    def _is_navigation_complete(self) -> bool:
        """Check if navigation is complete"""
        # This would typically check navigation status feedback
        # For now, we'll simulate completion
        return True  # Simplified for example

    def _is_manipulation_complete(self) -> bool:
        """Check if manipulation is complete"""
        # This would typically check manipulation status feedback
        return True  # Simplified for example

    def _wait_for_perception_results(self, timeout: float) -> Optional[Dict]:
        """Wait for perception results"""
        # This would typically wait for perception system feedback
        return {'objects': [], 'locations': []}  # Simplified for example

    def _decompose_composite_task(self, task: Task) -> List[Task]:
        """Decompose composite task into subtasks"""
        # This would use the TaskDecomposer
        decomposer = TaskDecomposer()
        return decomposer.decompose_goal(task.goal)

    def _execute_single_task_sync(self, task: Task) -> Dict[str, Any]:
        """Execute a single task synchronously (for internal use)"""
        # This is a simplified synchronous execution for subtasks
        if task.type == TaskType.NAVIGATION:
            return self._execute_navigation_task(task)
        elif task.type == TaskType.MANIPULATION:
            return self._execute_manipulation_task(task)
        elif task.type == TaskType.PERCEPTION:
            return self._execute_perception_task(task)
        else:
            return {'success': False, 'error': f'Unsupported task type: {task.type}'}

    def _navigation_callback(self, msg):
        """Handle navigation status updates"""
        pass  # Implementation would update navigation state

    def _manipulation_callback(self, msg):
        """Handle manipulation status updates"""
        pass  # Implementation would update manipulation state
```

### 3. Monitoring and Replanning System

#### Task Monitoring Framework

```python
import time
from datetime import datetime

class TaskMonitor:
    """Monitors task execution and detects failures or anomalies"""

    def __init__(self, planner: TaskPlanner):
        self.planner = planner
        self.task_start_times = {}
        self.task_timeouts = {}
        self.failure_thresholds = {
            'execution_time': 120.0,  # 2 minutes max execution time
            'resource_usage': 0.8,    # 80% max resource usage
        }

    def start_monitoring_task(self, task: Task):
        """Start monitoring a task"""
        self.task_start_times[task.id] = rospy.get_time()

        # Set timeout based on task type and complexity
        if task.type == TaskType.NAVIGATION:
            self.task_timeouts[task.id] = task.goal.get('timeout', 60.0)
        elif task.type == TaskType.MANIPULATION:
            self.task_timeouts[task.id] = task.goal.get('timeout', 30.0)
        else:
            self.task_timeouts[task.id] = task.goal.get('timeout', 20.0)

    def check_task_health(self, task_id: str) -> Dict[str, Any]:
        """Check the health of a running task"""
        if task_id not in self.task_start_times:
            return {'healthy': True, 'status': 'not_monitored'}

        current_time = rospy.get_time()
        start_time = self.task_start_times[task_id]
        elapsed_time = current_time - start_time
        timeout = self.task_timeouts[task_id]

        health_report = {
            'task_id': task_id,
            'elapsed_time': elapsed_time,
            'timeout': timeout,
            'time_remaining': timeout - elapsed_time,
            'healthy': True,
            'issues': []
        }

        # Check for timeout
        if elapsed_time > timeout:
            health_report['healthy'] = False
            health_report['issues'].append(f'Task exceeded timeout: {elapsed_time}s > {timeout}s')

        # Check for resource constraints (simulated)
        # In real implementation, this would check CPU, memory, etc.
        if self._check_resource_usage() > self.failure_thresholds['resource_usage']:
            health_report['healthy'] = False
            health_report['issues'].append('High resource usage detected')

        return health_report

    def _check_resource_usage(self) -> float:
        """Check current resource usage (simulated)"""
        # This would interface with system monitoring tools
        import psutil
        return psutil.cpu_percent() / 100.0

    def detect_failures(self) -> List[str]:
        """Detect failed tasks and return their IDs"""
        failed_tasks = []

        for task_id, task in self.planner.tasks.items():
            if task.status == TaskStatus.RUNNING:
                health = self.check_task_health(task_id)
                if not health['healthy']:
                    failed_tasks.append(task_id)

        return failed_tasks
```

#### Replanning System

```python
class Replanner:
    """Handles dynamic replanning when failures occur or conditions change"""

    def __init__(self, planner: TaskPlanner, monitor: TaskMonitor):
        self.planner = planner
        self.monitor = monitor
        self.failure_recovery_strategies = self._initialize_recovery_strategies()

    def _initialize_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize failure recovery strategies"""
        return {
            'navigation_timeout': self._handle_navigation_timeout,
            'manipulation_failure': self._handle_manipulation_failure,
            'object_not_found': self._handle_object_not_found,
            'obstacle_detected': self._handle_obstacle_detected,
        }

    def handle_task_failure(self, task_id: str, failure_reason: str) -> bool:
        """Handle a failed task and attempt recovery"""
        rospy.logwarn(f"Task {task_id} failed: {failure_reason}")

        # Determine appropriate recovery strategy
        strategy_key = self._classify_failure(failure_reason)

        if strategy_key in self.failure_recovery_strategies:
            recovery_success = self.failure_recovery_strategies[strategy_key](task_id)

            if recovery_success:
                rospy.loginfo(f"Recovery successful for task {task_id}")
                return True
            else:
                rospy.logerr(f"Recovery failed for task {task_id}")
                return False
        else:
            rospy.logerr(f"No recovery strategy for failure: {failure_reason}")
            return False

    def _classify_failure(self, failure_reason: str) -> str:
        """Classify failure reason to appropriate recovery strategy"""
        failure_lower = failure_reason.lower()

        if 'timeout' in failure_lower and 'navigation' in failure_lower:
            return 'navigation_timeout'
        elif 'timeout' in failure_lower and 'manipulation' in failure_lower:
            return 'manipulation_failure'
        elif 'not found' in failure_lower or 'not detected' in failure_lower:
            return 'object_not_found'
        elif 'obstacle' in failure_lower or 'collision' in failure_lower:
            return 'obstacle_detected'
        else:
            return 'general_failure'

    def _handle_navigation_timeout(self, task_id: str) -> bool:
        """Handle navigation timeout by trying alternative routes"""
        try:
            # Get the failed navigation task
            task = self.planner.tasks[task_id]
            target_location = task.goal.get('target_location')

            if not target_location:
                return False

            # Try alternative route (simplified approach)
            alternative_route = self._find_alternative_route(target_location)

            if alternative_route:
                # Create new navigation task with alternative route
                new_task = Task(
                    id=f"{task_id}_retry_{rospy.get_time()}",
                    type=TaskType.NAVIGATION,
                    goal={**task.goal, 'alternative_route': alternative_route},
                    dependencies=task.dependencies,
                    priority=task.priority + 1  # Higher priority for retry
                )

                # Add new task to planner
                self.planner.add_tasks([new_task])

                # Mark original task as failed
                self.planner.update_task_status(task_id, TaskStatus.FAILED,
                                              {'recovery': 'alternative_route_attempted'})

                return True
            else:
                return False

        except Exception as e:
            rospy.logerr(f"Error handling navigation timeout: {e}")
            return False

    def _handle_manipulation_failure(self, task_id: str) -> bool:
        """Handle manipulation failure by trying alternative grasps or approaches"""
        try:
            task = self.planner.tasks[task_id]
            object_id = task.goal.get('object_id')

            if not object_id:
                return False

            # Try alternative grasp strategy
            alternative_grasps = self._get_alternative_grasps(object_id)

            if alternative_grasps:
                # Create new manipulation task with alternative grasp
                new_task = Task(
                    id=f"{task_id}_retry_{rospy.get_time()}",
                    type=TaskType.MANIPULATION,
                    goal={**task.goal, 'grasp_type': alternative_grasps[0]},
                    dependencies=task.dependencies,
                    priority=task.priority + 1
                )

                self.planner.add_tasks([new_task])
                self.planner.update_task_status(task_id, TaskStatus.FAILED,
                                              {'recovery': 'alternative_grasp_attempted'})

                return True
            else:
                return False

        except Exception as e:
            rospy.logerr(f"Error handling manipulation failure: {e}")
            return False

    def _handle_object_not_found(self, task_id: str) -> bool:
        """Handle object not found by expanding search area or querying knowledge base"""
        try:
            task = self.planner.tasks[task_id]
            object_type = task.goal.get('object_type') or task.goal.get('object_id')

            if not object_type:
                return False

            # Query knowledge base for possible locations
            possible_locations = self._query_knowledge_base(object_type)

            if possible_locations:
                # Create perception tasks to search in alternative locations
                search_tasks = []
                for location in possible_locations[:3]:  # Limit to first 3 locations
                    search_task = Task(
                        id=f"search_{object_type}_{location}_{rospy.get_time()}",
                        type=TaskType.PERCEPTION,
                        goal={
                            'task': 'detect',
                            'object_type': object_type,
                            'search_area': location
                        },
                        dependencies=task.dependencies
                    )
                    search_tasks.append(search_task)

                self.planner.add_tasks(search_tasks)
                return True
            else:
                return False

        except Exception as e:
            rospy.logerr(f"Error handling object not found: {e}")
            return False

    def _handle_obstacle_detected(self, task_id: str) -> bool:
        """Handle obstacle detection by replanning around obstacles"""
        try:
            # This would interface with navigation system to replan
            # For now, we'll just mark the task as failed and let higher level
            # decide on next steps
            self.planner.update_task_status(task_id, TaskStatus.FAILED,
                                          {'recovery': 'obstacle_detected_needs_replanning'})
            return False  # Return False to indicate task should be abandoned

        except Exception as e:
            rospy.logerr(f"Error handling obstacle detection: {e}")
            return False

    def _find_alternative_route(self, target_location: str) -> Optional[Dict]:
        """Find alternative route to target location"""
        # This would interface with path planning system
        # For now, return a simple alternative
        return {'type': 'alternative', 'data': 'via_corridor'}

    def _get_alternative_grasps(self, object_id: str) -> List[str]:
        """Get alternative grasp strategies for an object"""
        # This would query object properties and grasp databases
        return ['side_grasp', 'top_grasp', 'pinch_grasp']

    def _query_knowledge_base(self, object_type: str) -> List[str]:
        """Query knowledge base for possible object locations"""
        # This would interface with semantic knowledge base
        knowledge_base = {
            'cup': ['kitchen_counter', 'dining_table', 'office_desk'],
            'book': ['bookshelf', 'office_desk', 'bedside_table'],
            'keys': ['entrance_table', 'kitchen_counter', 'bedroom_dresser']
        }

        return knowledge_base.get(object_type.lower(), [])
```

## Implementation Steps

### Step 1: Set up Task Planning Infrastructure

1. Create the main task planning node:

```python
#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

class TaskPlanningNode:
    def __init__(self):
        rospy.init_node('task_planning_node')

        # Initialize components
        self.decomposer = TaskDecomposer()
        self.planner = TaskPlanner()
        self.monitor = TaskMonitor(self.planner)
        self.replanner = Replanner(self.planner, self.monitor)

        # Publishers and subscribers
        self.goal_subscriber = rospy.Subscriber('/high_level_goals', String, self.goal_callback)
        self.status_publisher = rospy.Publisher('/task_status', String, queue_size=10)

        # Task completion callback
        self.task_callback = self._task_completion_callback

        rospy.loginfo("Task planning system initialized")

    def goal_callback(self, msg):
        """Handle incoming high-level goals"""
        try:
            # Parse goal from message (assuming JSON format)
            import json
            goal = json.loads(msg.data)

            # Decompose goal into tasks
            tasks = self.decomposer.decompose_goal(goal)

            # Add tasks to planner
            self.planner.add_tasks(tasks)

            # Start executing tasks
            self._execute_ready_tasks()

            rospy.loginfo(f"Added {len(tasks)} tasks for goal: {goal}")

        except Exception as e:
            rospy.logerr(f"Error processing goal: {e}")

    def _execute_ready_tasks(self):
        """Execute all ready tasks"""
        while True:
            task = self.planner.get_next_task()
            if task:
                # Monitor the task
                self.monitor.start_monitoring_task(task)

                # Execute the task
                success = self.planner.executor.execute_task(task, self.task_callback)

                if not success:
                    rospy.logerr(f"Failed to start execution of task {task.id}")
                    self.planner.update_task_status(task.id, TaskStatus.FAILED)
            else:
                break  # No more ready tasks

    def _task_completion_callback(self, task_id: str, status: TaskStatus, result: Optional[Dict]):
        """Handle task completion"""
        rospy.loginfo(f"Task {task_id} completed with status: {status}")

        # Update task status in planner
        self.planner.update_task_status(task_id, status, result)

        # Check for failures and handle them
        if status == TaskStatus.FAILED:
            failure_reason = result.get('error', 'Unknown error') if result else 'Unknown error'
            recovery_success = self.replanner.handle_task_failure(task_id, failure_reason)

            if not recovery_success:
                rospy.logerr(f"Could not recover from task failure: {task_id}")

        # Execute any newly ready tasks
        self._execute_ready_tasks()

        # Publish status update
        status_msg = String()
        status_msg.data = f"Task {task_id}: {status.value}"
        self.status_publisher.publish(status_msg)

    def run(self):
        """Main execution loop"""
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            # Check for failed tasks
            failed_tasks = self.monitor.detect_failures()
            for task_id in failed_tasks:
                self.replanner.handle_task_failure(task_id, "timeout_detected")

            rate.sleep()

if __name__ == '__main__':
    node = TaskPlanningNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
```

### Step 2: Implement Knowledge Base Integration

```python
class KnowledgeBaseInterface:
    """Interface to semantic knowledge base"""

    def __init__(self):
        self.object_properties = self._load_object_properties()
        self.location_semantics = self._load_location_semantics()
        self.task_preferences = self._load_task_preferences()

    def _load_object_properties(self) -> Dict:
        """Load object properties from knowledge base"""
        return {
            'cup': {
                'grasp_points': ['handle', 'body'],
                'typical_locations': ['kitchen', 'dining_room', 'office'],
                'manipulation_constraints': {'max_weight': 0.5, 'fragile': True}
            },
            'book': {
                'grasp_points': ['spine', 'cover'],
                'typical_locations': ['bookshelf', 'desk', 'bedside_table'],
                'manipulation_constraints': {'max_weight': 1.0, 'fragile': False}
            }
        }

    def _load_location_semantics(self) -> Dict:
        """Load location semantics and relationships"""
        return {
            'kitchen': {
                'contains': ['cup', 'plate', 'utensil'],
                'accessibility': 'high',
                'navigation_difficulty': 'low'
            },
            'bedroom': {
                'contains': ['book', 'clothes', 'personal_items'],
                'accessibility': 'medium',
                'navigation_difficulty': 'medium'
            }
        }

    def _load_task_preferences(self) -> Dict:
        """Load task execution preferences"""
        return {
            'manipulation': {
                'preferred_grasp': 'pinch',
                'safety_margin': 0.1
            },
            'navigation': {
                'preferred_paths': ['main_corridors'],
                'avoidance_zones': ['construction_areas']
            }
        }

    def get_object_property(self, object_type: str, property_name: str) -> Any:
        """Get a specific property of an object"""
        obj_props = self.object_properties.get(object_type, {})
        return obj_props.get(property_name)

    def get_typical_locations(self, object_type: str) -> List[str]:
        """Get typical locations where an object might be found"""
        return self.get_object_property(object_type, 'typical_locations') or []

    def get_manipulation_constraints(self, object_type: str) -> Dict:
        """Get manipulation constraints for an object"""
        return self.get_object_property(object_type, 'manipulation_constraints') or {}
```

### Step 3: Add Advanced Planning Features

```python
class AdvancedPlanner:
    """Advanced planning features including temporal and resource constraints"""

    def __init__(self, base_planner: TaskPlanner):
        self.base_planner = base_planner
        self.resource_manager = ResourceManager()
        self.temporal_constraints = {}

    def add_temporal_constraint(self, task_id: str, before_task: str, min_duration: float = 0.0):
        """Add temporal constraint: task_id must finish before before_task starts"""
        if before_task not in self.temporal_constraints:
            self.temporal_constraints[before_task] = []
        self.temporal_constraints[before_task].append({
            'task_id': task_id,
            'min_duration': min_duration,
            'constraint_type': 'before'
        })

    def check_resource_availability(self, task: Task) -> bool:
        """Check if required resources are available for task execution"""
        required_resources = self._get_task_resources(task)
        return self.resource_manager.check_availability(required_resources)

    def _get_task_resources(self, task: Task) -> Dict[str, float]:
        """Get resource requirements for a task"""
        resources = {}

        if task.type == TaskType.NAVIGATION:
            resources['navigation_system'] = 1.0
            resources['battery'] = 0.1  # 10% battery usage estimate
        elif task.type == TaskType.MANIPULATION:
            resources['manipulator'] = 1.0
            resources['battery'] = 0.05
        elif task.type == TaskType.PERCEPTION:
            resources['camera'] = 1.0
            resources['computation'] = 0.3

        return resources

class ResourceManager:
    """Manage shared resources across tasks"""

    def __init__(self):
        self.resource_limits = {
            'navigation_system': 1.0,
            'manipulator': 1.0,
            'camera': 1.0,
            'computation': 1.0,
            'battery': 1.0
        }
        self.resource_usage = {key: 0.0 for key in self.resource_limits}
        self.resource_locks = threading.Lock()

    def check_availability(self, resources: Dict[str, float]) -> bool:
        """Check if required resources are available"""
        with self.resource_locks:
            for resource, required in resources.items():
                available = self.resource_limits.get(resource, 0.0) - self.resource_usage.get(resource, 0.0)
                if required > available:
                    return False
            return True

    def allocate_resources(self, task_id: str, resources: Dict[str, float]) -> bool:
        """Allocate resources for a task"""
        if not self.check_availability(resources):
            return False

        with self.resource_locks:
            for resource, amount in resources.items():
                self.resource_usage[resource] = self.resource_usage.get(resource, 0.0) + amount

        return True

    def release_resources(self, task_id: str, resources: Dict[str, float]):
        """Release resources after task completion"""
        with self.resource_locks:
            for resource, amount in resources.items():
                current_usage = self.resource_usage.get(resource, 0.0)
                self.resource_usage[resource] = max(0.0, current_usage - amount)
```

## Testing and Validation

### Unit Testing

```python
import unittest
from unittest.mock import Mock, patch

class TestTaskPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = TaskPlanner()

    def test_task_decomposition(self):
        """Test task decomposition functionality"""
        decomposer = TaskDecomposer()

        goal = {
            'type': 'pick_up_object',
            'parameters': {
                'object_type': 'cup',
                'target_location': 'kitchen'
            }
        }

        tasks = decomposer.decompose_goal(goal)

        # Should create approach, detect, and grasp tasks
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(task.type in [TaskType.NAVIGATION, TaskType.PERCEPTION, TaskType.MANIPULATION]
                          for task in tasks))

    def test_task_dependencies(self):
        """Test task dependency management"""
        task1 = Task(id='task1', type=TaskType.NAVIGATION, goal={}, dependencies=[])
        task2 = Task(id='task2', type=TaskType.PERCEPTION, goal={}, dependencies=['task1'])

        self.planner.add_tasks([task1, task2])

        # Initially, only task1 should be ready
        ready_task = self.planner.get_next_task()
        self.assertEqual(ready_task.id, 'task1')

        # After completing task1, task2 should be ready
        self.planner.update_task_status('task1', TaskStatus.SUCCESS)
        ready_task = self.planner.get_next_task()
        self.assertEqual(ready_task.id, 'task2')

class TestTaskExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = TaskExecutor()

    @patch('rospy.Publisher')
    def test_navigation_execution(self, mock_pub):
        """Test navigation task execution"""
        task = Task(
            id='nav_task',
            type=TaskType.NAVIGATION,
            goal={'target_location': 'kitchen'},
            dependencies=[]
        )

        result = self.executor._execute_navigation_task(task)

        # Should publish navigation goal
        mock_pub.return_value.publish.assert_called_once()
        self.assertTrue(result['success'])

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
class TaskPlanningIntegrationTest:
    def __init__(self):
        rospy.init_node('task_planning_integration_test')
        self.planner = TaskPlanner()
        self.decomposer = TaskDecomposer()

    def test_complete_task_pipeline(self):
        """Test complete task planning and execution pipeline"""
        # Define a complex goal
        goal = {
            'type': 'pick_up_object',
            'parameters': {
                'object_type': 'red_cup',
                'target_location': 'kitchen'
            },
            'priority': 1
        }

        # Decompose goal
        tasks = self.decomposer.decompose_goal(goal)
        print(f"Decomposed into {len(tasks)} tasks")

        # Add tasks to planner
        self.planner.add_tasks(tasks)

        # Execute tasks (this would normally be done asynchronously)
        completed_tasks = 0
        while completed_tasks < len(tasks):
            next_task = self.planner.get_next_task()
            if next_task:
                # Simulate task execution
                print(f"Executing task: {next_task.id} ({next_task.type.value})")
                self.planner.update_task_status(next_task.id, TaskStatus.SUCCESS)
                completed_tasks += 1
            else:
                print("No tasks ready for execution")
                break

        print(f"Completed {completed_tasks} out of {len(tasks)} tasks")
```

## Performance Benchmarks

### Planning Performance

- **Task Decomposition**: < 50ms for simple tasks, < 200ms for complex tasks
- **Plan Generation**: < 100ms for plans with up to 10 tasks
- **Replanning**: < 300ms when failures occur
- **Memory Usage**: < 100MB for typical operation

### Execution Performance

- **Task Switching**: < 10ms between tasks
- **Status Updates**: < 50ms latency for status feedback
- **Resource Management**: < 1ms for resource checks
- **Concurrency**: Support for up to 5 simultaneous tasks

## Troubleshooting and Common Issues

### Planning Problems

1. **Cyclic Dependencies**: Ensure task dependencies don't create cycles
2. **Resource Deadlocks**: Implement resource allocation timeouts
3. **Infinite Loops**: Add iteration limits to planning algorithms
4. **Inconsistent States**: Maintain state consistency across failures

### Execution Problems

1. **Task Synchronization**: Ensure proper coordination between systems
2. **Timeout Management**: Handle timeouts gracefully with recovery
3. **Error Propagation**: Prevent errors from cascading through the system
4. **Resource Conflicts**: Manage shared resources effectively

## Best Practices

### Design Principles

- **Modularity**: Keep planning, execution, and monitoring separate
- **Extensibility**: Design for easy addition of new task types
- **Robustness**: Handle failures gracefully with recovery mechanisms
- **Efficiency**: Optimize for real-time performance requirements

### Safety Considerations

- **Fail-Safe**: Ensure safe robot state when planning fails
- **Validation**: Validate all generated plans before execution
- **Monitoring**: Continuously monitor execution for safety violations
- **Emergency Stop**: Implement emergency task termination

### Maintainability

- **Logging**: Comprehensive logging for debugging and analysis
- **Configuration**: Use parameter server for configurable behaviors
- **Testing**: Maintain high test coverage for all components
- **Documentation**: Document all interfaces and behaviors clearly

## Next Steps and Integration

### Integration with Other Capstone Components

The task planning system integrates with:
- **Voice Processing**: Receives high-level goals and commands
- **Navigation System**: Plans and monitors mobility tasks
- **Manipulation System**: Coordinates object interaction tasks
- **Perception System**: Requests and processes environmental information
- **Failure Handling**: Works with error recovery mechanisms

### Advanced Features

Consider implementing:
- **Learning-Based Planning**: Adapt planning based on execution experience
- **Multi-Robot Coordination**: Coordinate tasks across multiple robots
- **Temporal Planning**: Handle time-dependent task constraints
- **Resource Optimization**: Optimize resource usage across tasks

Continue with [Navigation and Obstacle Avoidance](./navigation.md) to explore the implementation of the navigation system that will execute the mobility tasks planned by this system.

## References

[All sources will be cited in the References section at the end of the book, following APA format]