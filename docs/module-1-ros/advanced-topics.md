---
sidebar_position: 4
---

# Advanced ROS 2 Topics: Services, Actions, and Parameters

## Overview

This section covers advanced ROS 2 communication patterns and system configuration mechanisms. We'll explore services for synchronous request-response communication, actions for long-running tasks with feedback, and the parameter system for runtime configuration.

## Services

Services provide synchronous, request-response communication between nodes. Unlike topics which use a publish-subscribe model, services establish a direct connection between a client and a server for each request.

### Service Definition

Services are defined using `.srv` files that specify both request and response message types. Here's an example service definition:

**File: `example_interfaces/srv/AddTwoInts.srv`**
```
# Request
int64 a
int64 b
---
# Response
int64 sum
```

### Creating a Service Server

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()

    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Service Client

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        self.future = self.cli.call_async(request)
        return self.future

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()

    # Send request
    future = minimal_client.send_request(1, 2)

    try:
        rclpy.spin_until_future_complete(minimal_client, future)
        response = future.result()
        minimal_client.get_logger().info(f'Result: {response.sum}')
    except KeyboardInterrupt:
        pass
    finally:
        minimal_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

Actions are designed for long-running tasks that require feedback and the ability to cancel. They combine features of both topics and services.

### Action Definition

Actions are defined using `.action` files:

**File: `example_interfaces/action/Fibonacci.action`**
```
# Goal
int32 order
---
# Result
int32[] sequence
---
# Feedback
int32[] sequence
```

### Creating an Action Server

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class MinimalActionServer(Node):

    def __init__(self):
        super().__init__('minimal_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info(f'Executing goal: {goal_handle.request.order}')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result

def main(args=None):
    rclpy.init(args=args)
    minimal_action_server = MinimalActionServer()

    try:
        rclpy.spin(minimal_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating an Action Client

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class MinimalActionClient(Node):

    def __init__(self):
        super().__init__('minimal_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

def main(args=None):
    rclpy.init(args=args)
    minimal_action_client = MinimalActionClient()

    minimal_action_client.send_goal(10)

    try:
        rclpy.spin(minimal_action_client)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_action_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parameter System

The parameter system allows nodes to be configured at runtime. Parameters can be declared, set, and modified dynamically.

### Declaring and Using Parameters

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):

    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('my_int_param', 42)
        self.declare_parameter('my_string_param', 'default_value')
        self.declare_parameter('my_double_param', 3.14)

        # Get parameter values
        my_int = self.get_parameter('my_int_param').value
        my_string = self.get_parameter('my_string_param').value
        my_double = self.get_parameter('my_double_param').value

        self.get_logger().info(f'Int: {my_int}, String: {my_string}, Double: {my_double}')

        # Set a callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    parameter_node = ParameterNode()

    try:
        rclpy.spin(parameter_node)
    except KeyboardInterrupt:
        pass
    finally:
        parameter_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Setting Parameters at Launch

Parameters can be set in launch files:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='parameter_node',
            name='parameter_node',
            parameters=[
                {'my_int_param': 100},
                {'my_string_param': 'configured_value'},
                {'my_double_param': 2.71},
            ]
        )
    ])
```

## Quality of Service (QoS) Settings

QoS settings allow fine-tuning of communication behavior to match application requirements:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# Create a QoS profile
qos_profile = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # Keep messages for late-joining subscribers
    history=QoSHistoryPolicy.KEEP_LAST,  # Keep only the last N messages
    reliability=QoSReliabilityPolicy.RELIABLE  # Ensure all messages are delivered
)

# Use the QoS profile when creating a publisher
publisher = node.create_publisher(String, 'topic', qos_profile)
```

## Lifecycle Nodes

Lifecycle nodes provide a more controlled way to manage node states (unconfigured, inactive, active, finalized):

```python
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn

class LifecycleNodeExample(LifecycleNode):

    def __init__(self):
        super().__init__('lifecycle_node')

    def on_configure(self, state):
        self.get_logger().info(f'Configuring node from state {state}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info(f'Activating node from state {state}')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info(f'Deactivating node from state {state}')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info(f'Cleaning up node from state {state}')
        return TransitionCallbackReturn.SUCCESS
```

## Best Practices

1. **Use services** for operations that should return a result immediately
2. **Use actions** for long-running operations that need feedback or cancellation
3. **Use topics** for continuous data streams and event notifications
4. **Parameter validation**: Always validate parameter values before using them
5. **QoS matching**: Ensure publishers and subscribers have compatible QoS settings
6. **Resource cleanup**: Always clean up resources in node destruction

## References

[All sources will be cited in the References section at the end of the book, following APA format]