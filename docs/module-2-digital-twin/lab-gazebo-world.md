---
sidebar_position: 7
---

# Hands-on Lab: Gazebo World Building

## Objective

Create a custom simulation environment in Gazebo with multiple objects, obstacles, and features that can be used for robot navigation and testing. This lab will provide practical experience in building realistic simulation environments that can be used for robotics development.

## Prerequisites

Before starting this lab, ensure you have:
- Gazebo Classic or Gazebo Garden installed
- Basic understanding of SDF (Simulation Description Format)
- Knowledge of XML structure and syntax
- Understanding of coordinate systems in robotics
- Basic command line skills

## Lab Setup

### 1. Creating the Project Structure

First, create the directory structure for your custom world:

```bash
# Create project directory
mkdir -p ~/gazebo_worlds/my_custom_world/models
mkdir -p ~/gazebo_worlds/my_custom_world/worlds

# Set GAZEBO_MODEL_PATH environment variable
echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/gazebo_worlds/my_custom_world/models' >> ~/.bashrc
source ~/.bashrc
```

### 2. Basic World Template

Create a basic world file that we'll build upon:

**File: `~/gazebo_worlds/my_custom_world/worlds/basic_lab.world`**

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_lab">
    <!-- Physics engine configuration -->
    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Add your custom models here -->

  </world>
</sdf>
```

## Step 1: Creating Basic Models

### 1.1 Creating a Simple Table Model

Create a simple table model that will be used in our environment:

**File: `~/gazebo_worlds/my_custom_world/models/table/model.sdf`**

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="table">
    <static>true</static>
    <link name="table_surface">
      <pose>0 0 0.4 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>1.0</iyy>
          <iyz>0.0</iyz>
          <izz>1.0</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>1.0 0.6 0.05</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>1.0 0.6 0.05</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.6 0.2 1</ambient>
          <diffuse>0.8 0.6 0.2 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <!-- Table legs -->
    <link name="leg_1">
      <pose>-0.4 -0.25 0.2 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.02</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.02</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>
    </link>

    <link name="leg_2">
      <pose>-0.4 0.25 0.2 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.02</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.02</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>
    </link>

    <link name="leg_3">
      <pose>0.4 -0.25 0.2 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.02</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.02</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>
    </link>

    <link name="leg_4">
      <pose>0.4 0.25 0.2 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.02</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.02</radius>
            <length>0.3</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>
    </link>

    <!-- Connect legs to table surface -->
    <joint name="leg1_to_surface" type="fixed">
      <parent>table_surface</parent>
      <child>leg_1</child>
    </joint>
    <joint name="leg2_to_surface" type="fixed">
      <parent>table_surface</parent>
      <child>leg_2</child>
    </joint>
    <joint name="leg3_to_surface" type="fixed">
      <parent>table_surface</parent>
      <child>leg_3</child>
    </joint>
    <joint name="leg4_to_surface" type="fixed">
      <parent>table_surface</parent>
      <child>leg_4</child>
    </joint>
  </model>
</sdf>
```

### 1.2 Creating the Model Configuration

**File: `~/gazebo_worlds/my_custom_world/models/table/model.config`**

```xml
<?xml version="1.0"?>
<model>
  <name>Table</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A simple table model for simulation environments.</description>
</model>
```

## Step 2: Creating Obstacle Models

### 2.1 Creating a Wall Segment Model

**File: `~/gazebo_worlds/my_custom_world/models/wall_segment/model.sdf`**

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="wall_segment">
    <static>true</static>
    <link name="wall_link">
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>1.0</iyy>
          <iyz>0.0</iyz>
          <izz>1.0</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>2.0 0.1 1.0</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>2.0 0.1 1.0</size>
          </box>
        </geometry>
        <material>
          <ambient>0.7 0.7 0.7 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
```

**File: `~/gazebo_worlds/my_custom_world/models/wall_segment/model.config`**

```xml
<?xml version="1.0"?>
<model>
  <name>Wall Segment</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A wall segment for building room layouts.</description>
</model>
```

### 2.2 Creating a Cylindrical Obstacle

**File: `~/gazebo_worlds/my_custom_world/models/cylinder_obstacle/model.sdf`**

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="cylinder_obstacle">
    <static>true</static>
    <link name="cylinder_link">
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.5</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.5</iyy>
          <iyz>0.0</iyz>
          <izz>0.5</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.2</radius>
            <length>0.8</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.2</radius>
            <length>0.8</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.8 0.3 0.3 1</ambient>
          <diffuse>0.8 0.3 0.3 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
```

**File: `~/gazebo_worlds/my_custom_world/models/cylinder_obstacle/model.config`**

```xml
<?xml version="1.0"?>
<model>
  <name>Cylinder Obstacle</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A cylindrical obstacle for navigation testing.</description>
</model>
```

## Step 3: Building the Complete World

### 3.1 Creating the Main World File

Now create a comprehensive world file that includes all the models we've created:

**File: `~/gazebo_worlds/my_custom_world/worlds/navigation_lab.world`**

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="navigation_lab">
    <!-- Physics engine configuration -->
    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Room walls -->
    <model name="room_wall_north">
      <pose>0 5 0.5 0 0 0</pose>
      <include>
        <uri>model://wall_segment</uri>
      </include>
    </model>

    <model name="room_wall_south">
      <pose>0 -5 0.5 0 0 3.14159</pose>
      <include>
        <uri>model://wall_segment</uri>
      </include>
    </model>

    <model name="room_wall_east">
      <pose>5 0 0.5 0 0 1.5708</pose>
      <include>
        <uri>model://wall_segment</uri>
      </include>
    </model>

    <model name="room_wall_west">
      <pose>-5 0 0.5 0 0 -1.5708</pose>
      <include>
        <uri>model://wall_segment</uri>
      </include>
    </model>

    <!-- Interior obstacles -->
    <model name="center_table">
      <pose>0 0 0 0 0 0</pose>
      <include>
        <uri>model://table</uri>
      </include>
    </model>

    <model name="obstacle_1">
      <pose>2 2 0.4 0 0 0</pose>
      <include>
        <uri>model://cylinder_obstacle</uri>
      </include>
    </model>

    <model name="obstacle_2">
      <pose>-2 -2 0.4 0 0 0</pose>
      <include>
        <uri>model://cylinder_obstacle</uri>
      </include>
    </model>

    <model name="obstacle_3">
      <pose>3 -1 0.4 0 0 0</pose>
      <include>
        <uri>model://cylinder_obstacle</uri>
      </include>
    </model>

    <!-- Goal marker -->
    <model name="goal_marker">
      <static>true</static>
      <link name="marker_link">
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.1</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 1 0 0.7</ambient>
            <diffuse>0 1 0 0.7</diffuse>
          </material>
        </visual>
      </link>
      <pose>4 4 0.05 0 0 0</pose>
    </model>

    <!-- Starting position marker -->
    <model name="start_marker">
      <static>true</static>
      <link name="marker_link">
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.1</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>1 0 0 0.7</ambient>
            <diffuse>1 0 0 0.7</diffuse>
          </material>
        </visual>
      </link>
      <pose>-4 -4 0.05 0 0 0</pose>
    </model>

    <!-- Add a simple robot for testing -->
    <model name="turtlebot3_burger">
      <include>
        <uri>model://turtlebot3_burger</uri>
      </include>
      <pose>-4 -4 0.0 0 0 0</pose>
    </model>

  </world>
</sdf>
```

## Step 4: Testing Your World

### 4.1 Running the Simulation

Now let's test our custom world:

```bash
# Launch Gazebo with your custom world
gzserver ~/gazebo_worlds/my_custom_world/worlds/navigation_lab.world &
gzclient &
```

Or if using Gazebo Classic:
```bash
gazebo ~/gazebo_worlds/my_custom_world/worlds/navigation_lab.world
```

### 4.2 Verifying the Environment

Check that:
- All walls are positioned correctly to form a room
- Obstacles are placed as intended
- The table is centered in the room
- The start and goal markers are visible
- The robot spawns at the correct location

## Step 5: Advanced Features

### 5.1 Adding Sensors to the World

You can also add world-level sensors for monitoring the environment:

```xml
<!-- Add this to your world file to include an overhead camera -->
<model name="overhead_camera">
  <static>true</static>
  <link name="sensor_link">
    <pose>0 0 10 0 1.57 0</pose>
    <visual name="visual">
      <geometry>
        <sphere><radius>0.01</radius></sphere>
      </geometry>
      <material>
        <ambient>0 1 0 1</ambient>
      </material>
    </visual>
  </link>
  <sensor name="overhead_camera_sensor" type="camera">
    <pose>0 0 0 0 0 0</pose>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <always_on>1</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>
  </sensor>
</model>
```

### 5.2 Adding Weather Effects

For more advanced simulation, you can add weather effects:

```xml
<!-- Add this to your world file -->
<scene>
  <ambient>0.4 0.4 0.4 1</ambient>
  <background>0.7 0.7 0.7 1</background>
  <shadows>true</shadows>
</scene>

<!-- Add fog effect -->
<atmosphere type="adiabatic">
  <temperature>288.15</temperature>
  <pressure>101325</pressure>
  <density>1.225</density>
</atmosphere>
```

## Step 6: World Customization

### 6.1 Parameterizing Your World

You can make your world more flexible by using SDF parameters:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="parameterized_lab">
    <!-- Parameters can be passed when launching -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Room dimensions as parameters -->
    <model name="room_walls">
      <static>true</static>
      <!-- Walls would be generated based on parameters -->
    </model>

  </world>
</sdf>
```

### 6.2 Creating Multiple Room Configurations

You can create different configurations of the same environment:

1. **Empty Room**: For basic navigation testing
2. **Sparse Obstacles**: Few obstacles for path planning
3. **Dense Obstacles**: Many obstacles for advanced navigation
4. **Dynamic Environment**: Moving obstacles (requires plugins)

## Troubleshooting Common Issues

### 1. Models Not Appearing
- **Check GAZEBO_MODEL_PATH**: Ensure it includes your models directory
- **Verify model.config**: Make sure the file exists and is properly formatted
- **Check file permissions**: Ensure Gazebo can read the model files

### 2. Physics Issues
- **Objects falling through surfaces**: Check collision geometries and poses
- **Unstable simulations**: Adjust physics parameters in the world file
- **Performance issues**: Simplify collision geometries

### 3. Lighting and Rendering
- **Dark environment**: Verify sun inclusion and scene settings
- **Poor visibility**: Adjust camera parameters and lighting

## Testing and Validation

### 1. Visual Inspection
- Verify all objects appear as expected
- Check that objects are positioned correctly
- Ensure there are no overlapping or floating objects

### 2. Physics Validation
- Test with a robot to ensure navigation is possible
- Check that obstacles provide proper collision
- Verify that static objects remain stationary

### 3. Performance Testing
- Monitor simulation frame rate
- Test with multiple robots if applicable
- Ensure the environment runs in real-time

## Extending Your World

### 1. Adding Interactive Elements
- Moving doors or gates
- Buttons that trigger events
- Transport mechanisms

### 2. Sensor Testing Areas
- Dedicated areas for LIDAR testing
- Visual marker zones
- Electromagnetic interference zones

### 3. Benchmark Scenarios
- Standard navigation challenges
- Mapping test areas
- Manipulation task zones

## Best Practices

### 1. Modular Design
- Create reusable components (walls, furniture, obstacles)
- Use consistent naming conventions
- Organize models in logical categories

### 2. Performance Optimization
- Use simple collision geometries when possible
- Limit the number of complex models
- Consider Level of Detail (LOD) for distant objects

### 3. Documentation
- Comment your SDF files clearly
- Document the purpose of each element
- Keep a changelog for world modifications

## Conclusion

In this lab, you've learned to:
- Create custom models for Gazebo environments
- Build complex simulation worlds with multiple objects
- Position objects accurately using coordinate systems
- Test and validate your simulation environments
- Apply best practices for world design

This foundation allows you to create increasingly complex and realistic simulation environments for robotics research and development. Continue experimenting with different objects, layouts, and configurations to build your skills in simulation environment design.

## References

[All sources will be cited in the References section at the end of the book, following APA format]