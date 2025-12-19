---
sidebar_position: 101
---

# Terminology Consistency Guide

## Overview

This document serves as a comprehensive guide to ensure consistent terminology usage across all modules of the Physical AI and Humanoid Robotics book. Consistent terminology is critical for academic clarity and helps students build a coherent understanding of concepts across different modules.

## Standardized Terminology

### Core Concepts

| Term | Definition | Preferred Usage | Variants to Avoid |
|------|------------|-----------------|-------------------|
| Robot Operating System 2 | Middleware framework for robotics communication | ROS 2 | ROS2, Robot Operating System 2, ros2 |
| Physical AI | AI systems that interact with the physical world | Physical AI | Physical Artificial Intelligence, embodied AI |
| Humanoid Robot | Robot with human-like form and capabilities | Humanoid Robot | Human-like robot, anthropomorphic robot |
| Vision-Language-Action | System connecting visual perception, language, and physical action | Vision-Language-Action (VLA) | Vision-Language-Action system, VLA system |
| Digital Twin | Virtual replica of a physical system | Digital Twin | Virtual model, simulation model |
| Isaac Sim | NVIDIA's robotics simulation platform | Isaac Sim | NVIDIA Isaac Sim, Isaac Simulator |
| Gazebo | Open-source robotics simulation environment | Gazebo | Gazebo Simulator, Gazebo Simulation |
| Perception Pipeline | System for processing sensor data to understand environment | Perception Pipeline | Sensory processing, perception system |
| Task Planning | Process of decomposing high-level goals into executable actions | Task Planning | Action planning, motion planning (when referring to specific movements) |
| Sim-to-Real Transfer | Process of transferring behaviors from simulation to real robots | Sim-to-Real | Simulation to reality, sim-to-real transfer |

### Technical Terms

| Term | Definition | Preferred Usage | Variants to Avoid |
|------|------------|-----------------|-------------------|
| Node | Individual process in ROS 2 communication system | Node | ros node, ros2 node, ROS node |
| Topic | Named bus over which nodes exchange messages | Topic | ROS topic, ros2 topic |
| Service | Synchronous request/response communication pattern | Service | ROS service, ros2 service |
| Action | Asynchronous goal-oriented communication pattern | Action | ROS action, ros2 action |
| Transform | Spatial relationship between coordinate frames | Transform | TF, transformation |
| Coordinate Frame | Reference system for spatial relationships | Coordinate Frame | frame, coordinate system |
| End Effector | Terminal device on a robot manipulator | End Effector | gripper, tool, effector |
| Degrees of Freedom | Independent parameters defining system configuration | Degrees of Freedom (DOF) | DOF only, degrees-of-freedom |
| Inverse Kinematics | Mathematics of determining joint angles for desired end pose | Inverse Kinematics (IK) | IK only, inverse kinematics problem |
| Forward Kinematics | Mathematics of determining end pose from joint angles | Forward Kinematics (FK) | FK only, forward kinematics problem |
| Point Cloud | Set of data points in 3D space | Point Cloud | 3D point cloud, point cloud data |
| Occupancy Grid | 2D representation of environment occupancy probabilities | Occupancy Grid | occupancy map, 2D map |
| Simultaneous Localization and Mapping | Process of building map while localizing | Simultaneous Localization and Mapping (SLAM) | SLAM only, mapping and localization |

### AI and Machine Learning Terms

| Term | Definition | Preferred Usage | Variants to Avoid |
|------|------------|-----------------|-------------------|
| Neural Network | Computing system inspired by biological neural networks | Neural Network | artificial neural network, ANN |
| Deep Learning | Subset of ML using multiple neural network layers | Deep Learning | deep neural networks, DNN |
| Convolutional Neural Network | Neural network for processing grid-like data | Convolutional Neural Network (CNN) | CNN only, convolutional network |
| Recurrent Neural Network | Neural network with connections forming directed cycles | Recurrent Neural Network (RNN) | RNN only, recurrent network |
| Reinforcement Learning | Learning through interaction and reward signals | Reinforcement Learning (RL) | RL only, reinforcement learning |
| Computer Vision | Field of computer science dealing with visual understanding | Computer Vision | machine vision, computer imaging |
| Natural Language Processing | Field of AI dealing with human language understanding | Natural Language Processing (NLP) | NLP only, computational linguistics |
| Transformer Model | Neural network architecture using attention mechanisms | Transformer Model | transformer, attention model |
| Multimodal Learning | Learning from multiple types of input data | Multimodal Learning | cross-modal learning, multi-modal |
| Embedding | Dense vector representation of categorical data | Embedding | embedding vector, encoded representation |

### Robotics-Specific Terms

| Term | Definition | Preferred Usage | Variants to Avoid |
|------|------------|-----------------|-------------------|
| Manipulation | Process of controlling objects in the environment | Manipulation | object manipulation, manipulator control |
| Navigation | Process of moving through environment to goal location | Navigation | robot navigation, path following |
| Localization | Process of determining robot's position in environment | Localization | robot localization, pose estimation |
| Path Planning | Process of determining route to goal location | Path Planning | path planning algorithm, route planning |
| Motion Planning | Process of determining collision-free movement | Motion Planning | trajectory planning, motion generation |
| Grasping | Process of securely holding an object | Grasping | grasp planning, gripping |
| Manipulator | Robot arm designed for object manipulation | Manipulator | robotic arm, robot manipulator |
| Mobile Base | Platform providing locomotion capability | Mobile Base | robot base, mobile platform |
| Sensor Fusion | Process of combining data from multiple sensors | Sensor Fusion | data fusion, sensor integration |
| Control Loop | Continuous cycle of sensing, planning, and acting | Control Loop | control cycle, feedback loop |

## Consistency Guidelines

### Capitalization and Formatting

- Module titles: "Module X: Title" (capitalized, with colon)
- Technical terms: Use consistent capitalization (e.g., "ROS 2" not "Ros2")
- Acronyms: Spell out full term first, then use acronym (e.g., "Simultaneous Localization and Mapping (SLAM)")
- Code elements: Use backticks for technical terms (`node`, `topic`, `service`)

### Usage Examples

#### Correct Usage
- "We will use ROS 2 as the communication framework for our robot."
- "The Vision-Language-Action (VLA) system processes voice commands."
- "The perception pipeline analyzes sensor data to detect objects."
- "Our task planning module breaks down high-level goals."

#### Incorrect Usage to Avoid
- "We will use ros2 as the communication framework" (inconsistent capitalization)
- "The VLA system processes voice commands" (should spell out first use)
- "The sensory processing system analyzes data" (inconsistent term)
- "Our action planning module breaks down goals" (wrong technical term)

## Cross-Module Consistency Checks

### Module 1 (ROS 2) Consistency
- Use "ROS 2" consistently, not "ROS" alone
- Refer to "nodes," "topics," "services," and "actions" with proper definitions
- Use "communication architecture" consistently for ROS 2's role

### Module 2 (Digital Twin) Consistency
- Use "Digital Twin" consistently, not "simulation model"
- Refer to "simulation environment" for Gazebo, "virtual environment" for Unity
- Use "physics simulation" and "sensor simulation" consistently

### Module 3 (AI-Robot Brain) Consistency
- Use "Isaac Sim" and "Isaac ROS" with proper capitalization
- Use "perception pipeline" consistently for sensor processing
- Use "neural network inference" for AI model execution

### Module 4 (Vision-Language-Action) Consistency
- Use "Vision-Language-Action" or "VLA" consistently
- Use "multimodal" for systems processing multiple input types
- Use "language understanding" for NLP components

## Review Checklist

When reviewing content, check for:

- [ ] Consistent use of preferred terminology
- [ ] Proper introduction of acronyms (full term first)
- [ ] Consistent capitalization of technical terms
- [ ] Consistent formatting of code elements
- [ ] Consistent module and section references
- [ ] Consistent use of technical definitions
- [ ] Consistent naming of system components
- [ ] Consistent use of brand/trademark names

## Updates and Maintenance

This terminology guide should be updated when:
- New technical terms are introduced in course content
- Industry standard terminology changes
- Student feedback indicates confusion about specific terms
- New modules introduce conflicting terminology

Last updated: December 2025