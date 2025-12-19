---
sidebar_position: 3
---

# Hands-on Lab: Publisher/Subscriber Nodes

## Objective

Create and deploy a basic ROS 2 package with multiple nodes communicating over topics (Quigley et al., 2009). This lab will help you understand the fundamental publish-subscribe communication pattern in ROS 2.

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Basic Python or C++ programming knowledge
- Understanding of ROS 2 basic concepts

## Lab Setup

### Creating a New Package

First, create a new ROS 2 package for our lab:

```bash
# Create the package
ros2 pkg create --build-type ament_python publisher_subscriber_lab

# Navigate to the package directory
cd publisher_subscriber_lab
```

### Package Structure

Your package structure should look like this:

```
publisher_subscriber_lab/
├── publisher_subscriber_lab/
│   ├── __init__.py
│   ├── publisher_node.py
│   └── subscriber_node.py
├── test/
├── package.xml
├── setup.cfg
├── setup.py
└── README.md
```

## Step 1: Creating the Publisher Node

Create the publisher node that will generate and publish messages:

**File: `publisher_subscriber_lab/publisher_node.py`**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random
import time

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        # Create a publisher with String message type on 'topic' topic
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # Set timer period to 0.5 seconds (2 Hz)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter for message numbering
        self.i = 0

        # Log that the publisher is starting
        self.get_logger().info('Publisher node initialized')

    def timer_callback(self):
        # Create a String message
        msg = String()
        msg.data = f'Hello World: {self.i}'

        # Publish the message
        self.publisher_.publish(msg)

        # Log the published message
        self.get_logger().info(f'Publishing: "{msg.data}"')

        # Increment the counter
        self.i += 1

def main(args=None):
    # Initialize the ROS client library
    rclpy.init(args=args)

    # Create the publisher node
    minimal_publisher = MinimalPublisher()

    try:
        # Spin the node to process callbacks
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        minimal_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 2: Creating the Subscriber Node

Create the subscriber node that will receive and process messages:

**File: `publisher_subscriber_lab/subscriber_node.py`**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')

        # Create a subscription to 'topic' with String message type
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)  # QoS history depth

        # Prevent unused variable warning
        self.subscription  # prevent unused variable warning

        # Log that the subscriber is starting
        self.get_logger().info('Subscriber node initialized')

    def listener_callback(self, msg):
        # Log the received message
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    # Initialize the ROS client library
    rclpy.init(args=args)

    # Create the subscriber node
    minimal_subscriber = MinimalSubscriber()

    try:
        # Spin the node to process callbacks
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        minimal_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 3: Configuring the Package

Update the `setup.py` file to make the nodes executable:

**File: `setup.py`**

```python
from setuptools import find_packages, setup

package_name = 'publisher_subscriber_lab'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A simple publisher-subscriber lab for ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher = publisher_subscriber_lab.publisher_node:main',
            'subscriber = publisher_subscriber_lab.subscriber_node:main',
        ],
    },
)
```

## Step 4: Building the Package

Build your package:

```bash
cd ~/ros2_ws  # or your ROS workspace
colcon build --packages-select publisher_subscriber_lab
source install/setup.bash
```

## Step 5: Running the Nodes

Open two terminal windows and source your ROS workspace in both:

**Terminal 1 (Publisher):**
```bash
source ~/ros2_ws/install/setup.bash
ros2 run publisher_subscriber_lab publisher
```

**Terminal 2 (Subscriber):**
```bash
source ~/ros2_ws/install/setup.bash
ros2 run publisher_subscriber_lab subscriber
```

You should see the publisher sending messages and the subscriber receiving them:

```
# Publisher output:
[INFO] [1680000000.123456789] [minimal_publisher]: Publishing: "Hello World: 0"
[INFO] [1680000000.623456789] [minimal_publisher]: Publishing: "Hello World: 1"
...

# Subscriber output:
[INFO] [1680000000.123456789] [minimal_subscriber]: I heard: "Hello World: 0"
[INFO] [1680000000.623456789] [minimal_subscriber]: I heard: "Hello World: 1"
...
```

## Step 6: Advanced Exercise - Custom Message

Create a custom message to practice more complex communication:

1. Create a directory for custom messages:
```bash
mkdir msg
```

2. Create a custom message definition in `msg/Num.msg`:
```
int64 num
string description
float64 value
```

3. Update `package.xml` to include message generation dependencies:
```xml
<depend>builtin_interfaces</depend>
<depend>std_msgs</depend>
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

4. Update `setup.py` to include message generation:
```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'publisher_subscriber_lab'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include custom message files
        (os.path.join('share', package_name, 'msg'), glob('msg/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A simple publisher-subscriber lab for ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher = publisher_subscriber_lab.publisher_node:main',
            'subscriber = publisher_subscriber_lab.subscriber_node:main',
        ],
    },
)
```

## Verification

To verify your implementation:

1. Check that nodes are running:
```bash
ros2 node list
```

2. Check the topic connection:
```bash
ros2 topic list
ros2 topic info /topic
```

3. Echo the topic to see messages:
```bash
ros2 topic echo /topic std_msgs/msg/String
```

## Expected Outcome

After completing this lab, you should be able to:
- Create a ROS 2 package with publisher and subscriber nodes
- Implement publish-subscribe communication pattern
- Run multiple nodes and observe their interaction
- Understand the basic structure of ROS 2 nodes

## References

[All sources will be cited in the References section at the end of the book, following APA format]

(Quigley et al., 2009) Quigley, M., Conley, K., Gerkey, B. P., Faust, J., Foote, T., Leibs, J., ... & Ng, A. Y. (2009). ROS: an open-source Robot Operating System. ICRA Workshop on Open Source Software, 3(3.2), 5.