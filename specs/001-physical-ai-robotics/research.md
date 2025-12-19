# Research Summary: Physical AI & Humanoid Robotics Book

## Decision: Simulation Tools Selection
**Rationale**: After evaluating the requirements, Gazebo is selected as the primary simulation tool due to its widespread adoption in robotics research, comprehensive physics engine, and strong ROS 2 integration. Unity will be used as a secondary option for advanced visualization and gaming-style interfaces.

**Alternatives considered**:
- NVIDIA Isaac Sim: More specialized for NVIDIA platforms but less general-purpose
- Webots: Good alternative but less ROS 2 integration
- MuJoCo: Commercial and expensive for educational use

## Decision: Hardware Assumptions
**Rationale**: On-premise hardware is assumed as the primary deployment model to ensure reproducibility and allow hands-on experience. Cloud alternatives will be documented as secondary options for users with resource constraints.

**Alternatives considered**:
- Cloud-only: Less hands-on experience, dependency on internet connectivity
- Hybrid: Complex to manage, decided to focus on primary on-premise with cloud as fallback

## Decision: Depth vs Breadth Tradeoffs
**Rationale**: Each module will provide sufficient depth to enable practical implementation while maintaining breadth across all required topics. Module 1 (ROS 2) will have the most depth as it's foundational for all other modules.

**Alternatives considered**:
- More depth in fewer modules: Would miss important topics
- Equal depth across all modules: Would compromise foundational understanding

## Decision: Robot Platform Assumptions
**Rationale**: The book will focus on general humanoid robotics concepts that can be applied to various platforms, with specific examples using TurtleBot3 as a proxy robot due to its widespread availability and ROS 2 support. More advanced humanoid platforms will be discussed conceptually.

**Alternatives considered**:
- Specific vendor robots: Would limit applicability
- Custom-built robots: Too complex for educational purposes