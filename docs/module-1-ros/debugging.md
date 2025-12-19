---
sidebar_position: 7
---

# Verification and Debugging in ROS 2

## Overview

Effective verification and debugging are crucial for developing robust robotic systems. This section covers essential tools, techniques, and best practices for debugging ROS 2 applications and verifying system behavior.

## Debugging Tools and Techniques

### 1. ROS 2 Command Line Tools

#### Node Inspection
```bash
# List all active nodes
ros2 node list

# Get information about a specific node
ros2 node info <node_name>

# Find nodes that match a pattern
ros2 node list | grep <pattern>
```

#### Topic Inspection
```bash
# List all topics
ros2 topic list

# Get information about a specific topic
ros2 topic info <topic_name>

# Echo messages on a topic
ros2 topic echo <topic_name> <msg_type>

# Echo with filtering (first 10 messages)
ros2 topic echo <topic_name> <msg_type> --field <field_name> --limit 10

# Publish a message to a topic
ros2 topic pub <topic_name> <msg_type> '{field1: value1, field2: value2}'
```

#### Service and Action Inspection
```bash
# List all services
ros2 service list

# Call a service
ros2 service call <service_name> <service_type> '{request_field: value}'

# List all actions
ros2 action list

# Send a goal to an action
ros2 action send_goal <action_name> <action_type> '{goal_field: value}'
```

### 2. Logging and Diagnostics

#### ROS 2 Logging System
```python
import rclpy
from rclpy.node import Node

class DebuggingNode(Node):

    def __init__(self):
        super().__init__('debugging_node')

        # Different log levels
        self.get_logger().debug('Debug message')
        self.get_logger().info('Info message')
        self.get_logger().warn('Warning message')
        self.get_logger().error('Error message')
        self.get_logger().fatal('Fatal message')

        # Logging with parameters
        value = 42
        self.get_logger().info(f'Current value: {value}')

        # Conditional logging
        if value > 40:
            self.get_logger().warn(f'Value {value} is higher than threshold')

def main(args=None):
    rclpy.init(args=args)
    node = DebuggingNode()

    # Set log level programmatically
    node.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Launch File Logging Configuration
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='debugging_node',
            name='debugging_node',
            parameters=[
                {'use_sim_time': False},
            ],
            # Configure logging
            arguments=['--ros-args', '--log-level', 'debug'],
            # Remap topics for debugging
            remappings=[
                ('/original_topic', '/debug_topic'),
            ]
        )
    ])
```

### 3. Interactive Debugging

#### Using rqt Tools
```bash
# Launch the rqt GUI
rqt

# Launch specific rqt plugins
rqt_graph                    # Visualize node connections
rqt_console                  # View log messages
rqt_plot                     # Plot numerical values
rqt_topic                    # Monitor topics
rqt_service_caller           # Call services interactively
rqt_publisher              # Publish messages interactively
```

#### Creating Custom Debug Nodes
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class DebugBridgeNode(Node):
    """Bridge node to convert and debug different message types"""

    def __init__(self):
        super().__init__('debug_bridge_node')

        # Subscribers for different sensor types
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10)

        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)

        # Publishers for debugging
        self.debug_pub = self.create_publisher(
            String, 'debug_info', 10)

        # Timer for periodic status reports
        self.timer = self.create_timer(1.0, self.status_callback)

        self.lidar_data = None
        self.cmd_vel_data = None

    def lidar_callback(self, msg):
        """Process and debug lidar data"""
        self.lidar_data = msg
        self.get_logger().debug(f'Lidar ranges: {len(msg.ranges)} points')

        # Check for potential issues
        if any(r < 0.1 for r in msg.ranges if r > 0):
            self.get_logger().warn('Lidar detected very close obstacle')

        # Publish debug info
        debug_msg = String()
        debug_msg.data = f'Lidar: {len(msg.ranges)} ranges, min: {min(msg.ranges) if msg.ranges else "N/A"}'
        self.debug_pub.publish(debug_msg)

    def cmd_vel_callback(self, msg):
        """Process and debug velocity commands"""
        self.cmd_vel_data = msg
        self.get_logger().debug(f'Velocity command: {msg.linear.x}, {msg.angular.z}')

        # Check for unusual commands
        if abs(msg.linear.x) > 1.0 or abs(msg.angular.z) > 1.0:
            self.get_logger().warn('Unusually high velocity command')

    def status_callback(self):
        """Periodic status report"""
        if self.lidar_data and self.cmd_vel_data:
            status_msg = String()
            status_msg.data = f'Status: Lidar OK, CmdVel: ({self.cmd_vel_data.linear.x:.2f}, {self.cmd_vel_data.angular.z:.2f})'
            self.debug_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DebugBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Verification Techniques

### 1. Unit Testing

#### Basic Unit Tests
```python
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from my_package.my_node import MyNode

class TestMyNode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = MyNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()

    def test_node_initialization(self):
        """Test that the node initializes correctly"""
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.get_name(), 'my_node')

    def test_parameter_declaration(self):
        """Test that parameters are properly declared"""
        self.assertTrue(self.node.has_parameter('my_parameter'))
        self.assertEqual(self.node.get_parameter('my_parameter').value, 'default_value')

if __name__ == '__main__':
    unittest.main()
```

#### Integration Testing with Mock Data
```python
import unittest
import rclpy
from std_msgs.msg import String
from my_package.processing_node import ProcessingNode

class TestProcessingNode(unittest.TestCase):

    def setUp(self):
        rclpy.init()
        self.node = ProcessingNode()
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_message_processing(self):
        """Test message processing with mock data"""
        # Create a publisher to send test data
        pub = self.node.create_publisher(String, 'input_topic', 10)

        # Wait for publisher to be ready
        self.executor.spin_once(timeout_sec=1.0)

        # Send test message
        test_msg = String()
        test_msg.data = 'test_data'
        pub.publish(test_msg)

        # Process for a short time to allow message handling
        self.executor.spin_once(timeout_sec=0.1)

        # Add assertions based on expected behavior
        # This would depend on your specific node implementation

if __name__ == '__main__':
    unittest.main()
```

### 2. System Verification

#### Creating Verification Nodes
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from builtin_interfaces.msg import Time

class SystemVerifier(Node):
    """Node to verify system behavior and performance"""

    def __init__(self):
        super().__init__('system_verifier')

        # Publishers for verification results
        self.passed_pub = self.create_publisher(Bool, 'verification_passed', 10)
        self.report_pub = self.create_publisher(String, 'verification_report', 10)

        # Timer for periodic verification
        self.timer = self.create_timer(5.0, self.verify_system)

        # Storage for verification results
        self.verification_results = {}
        self.start_time = self.get_clock().now()

    def verify_system(self):
        """Perform system verification checks"""
        results = {
            'node_connections': self.check_node_connections(),
            'topic_quality': self.check_topic_quality(),
            'performance_metrics': self.check_performance(),
            'data_integrity': self.check_data_integrity()
        }

        # Aggregate results
        all_passed = all(results.values())

        # Publish results
        passed_msg = Bool()
        passed_msg.data = all_passed
        self.passed_pub.publish(passed_msg)

        report_msg = String()
        report_msg.data = f"Verification at {self.get_clock().now().nanoseconds}: {results}"
        self.report_pub.publish(report_msg)

        self.get_logger().info(f'System verification: {all_passed}')

        if not all_passed:
            self.get_logger().error(f'Verification failed: {results}')

    def check_node_connections(self):
        """Check if expected nodes are connected"""
        # Implementation depends on your system architecture
        # Example: check if navigation, perception, and control nodes are active
        nodes = self.get_node_names()
        expected_nodes = ['navigation_node', 'perception_node', 'controller_node']

        return all(node in nodes for node in expected_nodes)

    def check_topic_quality(self):
        """Check topic message rates and quality"""
        # Implementation would check message rates, timestamps, etc.
        # This is a simplified example
        return True

    def check_performance(self):
        """Check system performance metrics"""
        # Implementation would check CPU, memory, etc.
        return True

    def check_data_integrity(self):
        """Check data integrity and validity"""
        # Implementation would validate message contents
        return True

def main(args=None):
    rclpy.init(args=args)
    verifier = SystemVerifier()

    try:
        rclpy.spin(verifier)
    except KeyboardInterrupt:
        pass
    finally:
        verifier.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Debugging Techniques

### 1. Performance Profiling

#### CPU and Memory Monitoring
```python
import rclpy
from rclpy.node import Node
import psutil
import os
from std_msgs.msg import String

class PerformanceMonitor(Node):
    """Monitor system performance for debugging"""

    def __init__(self):
        super().__init__('performance_monitor')

        self.perf_pub = self.create_publisher(String, 'performance_stats', 10)
        self.timer = self.create_timer(1.0, self.monitor_performance)

        # Get process ID for this node
        self.process = psutil.Process(os.getpid())

    def monitor_performance(self):
        """Monitor and report performance metrics"""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent()

            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB

            # System-wide metrics
            system_cpu = psutil.cpu_percent()
            system_memory = psutil.virtual_memory().percent

            perf_msg = String()
            perf_msg.data = (
                f'Process: CPU={cpu_percent:.1f}%, Memory={memory_mb:.1f}MB | '
                f'System: CPU={system_cpu:.1f}%, Memory={system_memory:.1f}%'
            )

            self.perf_pub.publish(perf_msg)

            # Log warnings for performance issues
            if cpu_percent > 80:
                self.get_logger().warn(f'High CPU usage: {cpu_percent}%')
            if memory_mb > 500:  # 500MB threshold
                self.get_logger().warn(f'High memory usage: {memory_mb:.1f}MB')

        except Exception as e:
            self.get_logger().error(f'Performance monitoring error: {e}')

def main(args=None):
    rclpy.init(args=args)
    monitor = PerformanceMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Message Filtering and Analysis

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import numpy as np

class MessageAnalyzer(Node):
    """Analyze message patterns for debugging"""

    def __init__(self):
        super().__init__('message_analyzer')

        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

        self.message_stats = {
            'count': 0,
            'timestamps': [],
            'rates': [],
            'data_stats': []
        }

    def scan_callback(self, msg):
        """Analyze laser scan messages"""
        self.message_stats['count'] += 1

        # Track timing
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.message_stats['timestamps'].append(current_time)

        # Calculate message rate
        if len(self.message_stats['timestamps']) > 1:
            time_diff = np.diff(self.message_stats['timestamps'])
            if len(time_diff) > 0:
                rate = 1.0 / np.mean(time_diff)
                self.message_stats['rates'].append(rate)

                if rate < 5.0:  # If rate drops below 5Hz
                    self.get_logger().warn(f'Low message rate: {rate:.2f} Hz')

        # Analyze data patterns
        valid_ranges = [r for r in msg.ranges if r > 0 and not np.isnan(r)]
        if valid_ranges:
            self.message_stats['data_stats'].append({
                'min_range': min(valid_ranges) if valid_ranges else None,
                'max_range': max(valid_ranges) if valid_ranges else None,
                'mean_range': np.mean(valid_ranges) if valid_ranges else None,
                'num_valid': len(valid_ranges)
            })

        # Check for data anomalies
        if len(valid_ranges) < 10:  # Too few valid readings
            self.get_logger().warn(f'Very few valid ranges: {len(valid_ranges)}')

        # Check for sensor errors
        if not valid_ranges and len(msg.ranges) > 0:
            self.get_logger().error('All ranges invalid - sensor error?')

def main(args=None):
    rclpy.init(args=args)
    analyzer = MessageAnalyzer()

    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Verification and Debugging

### 1. Systematic Approach
- Start with high-level system behavior before diving into details
- Use logging levels appropriately (DEBUG, INFO, WARN, ERROR)
- Implement early failure detection
- Create diagnostic nodes for complex systems

### 2. Documentation and Traceability
- Document debugging procedures and common issues
- Use meaningful node and topic names
- Maintain version control for configurations
- Create runbooks for common debugging scenarios

### 3. Automated Testing
- Implement unit tests for individual components
- Create integration tests for subsystems
- Set up continuous integration for regression testing
- Use simulation for testing before real robot deployment

### 4. Performance Monitoring
- Monitor CPU, memory, and network usage
- Track message rates and latencies
- Set up alerts for performance degradation
- Profile code for bottlenecks

## Troubleshooting Common Issues

### 1. Communication Issues
- Check topic connections: `ros2 topic info <topic>`
- Verify QoS settings match between publishers and subscribers
- Check network configuration for multi-machine setups
- Use `ros2 doctor` for system diagnostics

### 2. Timing Issues
- Check message timestamps for delays
- Verify system clocks are synchronized
- Monitor message processing rates
- Check for blocking operations in callbacks

### 3. Resource Issues
- Monitor CPU and memory usage
- Check for memory leaks in long-running nodes
- Verify proper resource cleanup
- Monitor disk space for logging

## References

[All sources will be cited in the References section at the end of the book, following APA format]