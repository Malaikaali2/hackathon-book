---
sidebar_position: 3
---

# Instruction Following and Task Planning

## Learning Objectives

By the end of this section, you will be able to:

1. Design systems that parse and interpret natural language instructions for robots
2. Implement hierarchical task planning frameworks for complex robotic behaviors
3. Create semantic parsing mechanisms that convert language to executable actions
4. Develop context-aware instruction interpretation that considers environmental constraints
5. Build robust error handling and recovery mechanisms for failed instruction execution

## Introduction to Instruction Following

Instruction following represents one of the most challenging aspects of Vision-Language-Action (VLA) systems. Unlike traditional programming approaches where robot behaviors are explicitly coded, instruction following systems must interpret natural language commands and translate them into appropriate sequences of robotic actions.

The complexity of instruction following stems from several factors:

- **Ambiguity**: Natural language is inherently ambiguous, with multiple possible interpretations
- **Context Dependence**: Instructions often depend on environmental context and previous interactions
- **Spatial Reasoning**: Many robot instructions involve spatial relationships and navigation
- **Temporal Dependencies**: Complex instructions require sequences of actions with temporal relationships
- **Physical Constraints**: Robot capabilities and environmental constraints limit feasible actions

## Hierarchical Task Planning Architecture

### Three-Level Planning Hierarchy

Effective instruction following systems typically employ a hierarchical planning architecture with three levels:

1. **Task Level**: High-level goal decomposition and sequencing
2. **Action Level**: Mid-level primitive selection and parameterization
3. **Motion Level**: Low-level trajectory generation and control

```python
class HierarchicalInstructionFollower:
    def __init__(self):
        self.task_planner = TaskPlanner()
        self.action_planner = ActionPlanner()
        self.motion_planner = MotionPlanner()

    def follow_instruction(self, instruction, context):
        """Follow a natural language instruction using hierarchical planning"""
        # Parse high-level task from instruction
        high_level_task = self.parse_instruction(instruction, context)

        # Decompose task hierarchically
        task_plan = self.task_planner.decompose(high_level_task, context)

        # Execute plan at different levels
        for subtask in task_plan:
            action_sequence = self.action_planner.plan(subtask, context)
            for action in action_sequence:
                motion_commands = self.motion_planner.generate(action, context)
                self.execute_motion(motion_commands)

    def parse_instruction(self, instruction, context):
        """Parse natural language instruction into structured task representation"""
        # This would use NLP techniques to extract task structure
        pass
```

### Task Decomposition Framework

Task decomposition breaks complex instructions into manageable subtasks:

```python
class TaskPlanner:
    def __init__(self):
        self.task_library = self.load_task_library()
        self.decomposition_rules = self.load_decomposition_rules()

    def decompose(self, high_level_task, context):
        """Decompose high-level task into executable subtasks"""
        if high_level_task in self.task_library:
            return self.task_library[high_level_task]

        # Apply decomposition rules
        subtasks = self.apply_decomposition_rules(high_level_task, context)
        return subtasks

    def load_task_library(self):
        """Load predefined task decompositions"""
        return {
            'clean_table': [
                'identify_objects_on_table',
                'grasp_object',
                'move_to_trash',
                'place_object',
                'repeat_until_clean'
            ],
            'set_table': [
                'identify_dining_area',
                'fetch_plate',
                'move_to_table',
                'place_plate',
                'fetch_cup',
                'move_to_table',
                'place_cup'
            ]
        }

    def apply_decomposition_rules(self, task, context):
        """Apply learned decomposition rules to novel tasks"""
        # Example: "bring X from Y to Z" -> navigate to Y, grasp X, navigate to Z, place X
        if self.matches_pattern(task, 'bring_X_from_Y_to_Z'):
            obj = self.extract_object(task)
            source = self.extract_location(task, 'source')
            target = self.extract_location(task, 'target')

            return [
                NavigateToLocation(source),
                GraspObject(obj),
                NavigateToLocation(target),
                PlaceObject(obj)
            ]

        return [task]  # No decomposition possible

    def matches_pattern(self, task, pattern):
        """Check if task matches a known pattern"""
        # Implementation would use NLP pattern matching
        pass

    def extract_object(self, task):
        """Extract object reference from task"""
        # Implementation would use NLP to identify objects
        pass

    def extract_location(self, task, location_type):
        """Extract location reference from task"""
        # Implementation would identify source/target locations
        pass
```

## Semantic Parsing and Natural Language Understanding

### Grammar-Based Parsing

Grammar-based parsing uses formal rules to convert natural language to structured representations:

```python
import nltk
from nltk import CFG
from nltk.parse import ChartParser

class SemanticParser:
    def __init__(self):
        self.grammar = self.create_robot_grammar()
        self.parser = ChartParser(self.grammar)

    def create_robot_grammar(self):
        """Create grammar for robot instruction language"""
        # Define grammar rules for robot instructions
        grammar_rules = """
            S -> COMMAND
            COMMAND -> ACTION OBJECT LOCATION
                   | ACTION OBJECT
                   | NAVIGATE LOCATION
                   | ACTION NAVIGATE LOCATION
            ACTION -> 'pick' | 'grasp' | 'place' | 'move' | 'go' | 'navigate' | 'bring' | 'fetch'
            OBJECT -> 'ball' | 'cup' | 'book' | 'box' | 'red_ball' | 'blue_cup'
            LOCATION -> 'kitchen' | 'living_room' | 'table' | 'shelf' | 'counter' | 'toilet' | 'bedroom'
                     | 'left' | 'right' | 'front' | 'back' | 'near_ball' | 'by_cup'
        """
        return CFG.fromstring(grammar_rules)

    def parse(self, instruction):
        """Parse natural language instruction using grammar"""
        tokens = instruction.lower().split()

        try:
            # Parse the instruction
            parses = list(self.parser.parse(tokens))
            if parses:
                return self.convert_parse_to_action(parses[0])
        except:
            # Handle parsing failure
            return self.handle_parsing_failure(instruction)

        return None

    def convert_parse_to_action(self, parse_tree):
        """Convert parse tree to executable action representation"""
        # Convert parse tree to structured action
        action = {}

        for subtree in parse_tree.subtrees():
            if subtree.label() == 'ACTION':
                action['action_type'] = str(subtree[0])
            elif subtree.label() == 'OBJECT':
                action['object'] = str(subtree[0])
            elif subtree.label() == 'LOCATION':
                action['location'] = str(subtree[0])

        return action

    def handle_parsing_failure(self, instruction):
        """Handle cases where grammar-based parsing fails"""
        # Use alternative parsing methods (e.g., neural parsing)
        return self.neural_parse(instruction)

    def neural_parse(self, instruction):
        """Use neural network for parsing (fallback method)"""
        # Implementation would use a trained neural parser
        pass
```

### Neural Semantic Parsing

Neural approaches can handle more complex and ambiguous language:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralSemanticParser(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.action_classifier = nn.Linear(hidden_dim, output_dim)
        self.location_classifier = nn.Linear(hidden_dim, output_dim)
        self.object_classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, tokens):
        """Parse tokens into structured representation"""
        embedded = self.embedding(tokens)  # [B, T, embed_dim]
        lstm_out, (hidden, _) = self.lstm(embedded)  # [B, T, hidden_dim]

        # Use final hidden state for classification
        final_hidden = hidden[-1]  # [B, hidden_dim]

        action_probs = F.softmax(self.action_classifier(final_hidden), dim=-1)
        location_probs = F.softmax(self.location_classifier(final_hidden), dim=-1)
        object_probs = F.softmax(self.object_classifier(final_hidden), dim=-1)

        return {
            'action': action_probs,
            'location': location_probs,
            'object': object_probs
        }

class InstructionUnderstandingSystem:
    def __init__(self):
        self.neural_parser = NeuralSemanticParser(
            vocab_size=10000, embed_dim=256, hidden_dim=512, output_dim=100
        )
        self.grammar_parser = SemanticParser()

    def understand_instruction(self, instruction, context):
        """Understand natural language instruction using multiple approaches"""
        # Try grammar-based parsing first (for structured commands)
        structured_action = self.grammar_parser.parse(instruction)

        if structured_action:
            return self.refine_with_context(structured_action, context)

        # Fall back to neural parsing for complex/ambiguous commands
        tokens = self.tokenize(instruction)
        neural_output = self.neural_parser(tokens)

        return self.convert_neural_output_to_action(neural_output, context)

    def refine_with_context(self, action, context):
        """Refine parsed action based on environmental context"""
        # Resolve ambiguities using context
        if 'location' in action and action['location'] == 'it':
            # Resolve pronoun based on context
            action['location'] = self.resolve_pronoun('it', context)

        return action

    def resolve_pronoun(self, pronoun, context):
        """Resolve pronouns based on context"""
        # Implementation would use context to resolve pronouns
        # e.g., "it" might refer to the last mentioned object
        pass
```

## Context-Aware Instruction Interpretation

### Environmental Context Modeling

Context-aware systems consider environmental information when interpreting instructions:

```python
class ContextAwareInterpreter:
    def __init__(self):
        self.object_detector = ObjectDetectionSystem()
        self.spatial_reasoner = SpatialReasoner()
        self.context_memory = ContextMemory()

    def interpret_instruction(self, instruction, current_context):
        """Interpret instruction considering environmental context"""
        # Get current environmental state
        environment_state = self.get_environment_state(current_context)

        # Parse instruction with context
        parsed_instruction = self.parse_with_context(instruction, environment_state)

        # Resolve ambiguities using context
        resolved_instruction = self.resolve_ambiguities(
            parsed_instruction, environment_state
        )

        return resolved_instruction

    def get_environment_state(self, context):
        """Get current environmental state"""
        state = {
            'visible_objects': self.object_detector.detect(context['image']),
            'robot_pose': context['robot_pose'],
            'navigation_map': context['map'],
            'recent_interactions': self.context_memory.get_recent()
        }
        return state

    def parse_with_context(self, instruction, environment_state):
        """Parse instruction using environmental context"""
        # Example: "the red ball" -> specific red ball in view
        parsed = self.semantic_parser.parse(instruction)

        if 'object' in parsed and 'the' in instruction:
            # Resolve definite reference using visual context
            specific_object = self.resolve_object_reference(
                parsed['object'], environment_state
            )
            parsed['specific_object'] = specific_object

        return parsed

    def resolve_ambiguities(self, parsed_instruction, environment_state):
        """Resolve ambiguities using environmental context"""
        # Resolve spatial prepositions
        if 'location' in parsed_instruction:
            resolved_location = self.spatial_reasoner.resolve_location(
                parsed_instruction['location'], environment_state
            )
            parsed_instruction['resolved_location'] = resolved_location

        # Resolve object references
        if 'specific_object' in parsed_instruction:
            # Ensure the referenced object is accessible
            if not self.is_object_accessible(
                parsed_instruction['specific_object'], environment_state
            ):
                raise ValueError("Referenced object is not accessible")

        return parsed_instruction

    def resolve_object_reference(self, object_type, environment_state):
        """Resolve object reference to specific instance"""
        visible_objects = environment_state['visible_objects']

        # Filter by type
        matching_objects = [
            obj for obj in visible_objects
            if obj['type'] == object_type
        ]

        if len(matching_objects) == 1:
            return matching_objects[0]
        elif len(matching_objects) > 1:
            # Need additional disambiguation
            return self.disambiguate_objects(matching_objects, environment_state)
        else:
            # Object not visible, might be in memory
            return self.search_memory(object_type)

    def disambiguate_objects(self, objects, environment_state):
        """Disambiguate between multiple objects of the same type"""
        # Use spatial relationships, colors, sizes, etc. for disambiguation
        # This would implement more sophisticated disambiguation logic
        pass

    def is_object_accessible(self, object_info, environment_state):
        """Check if object is accessible to robot"""
        # Check if object is within reach, not obstructed, etc.
        robot_pose = environment_state['robot_pose']
        object_pose = object_info['pose']

        distance = self.calculate_distance(robot_pose, object_pose)
        return distance < self.robot_reach_distance

    def calculate_distance(self, pose1, pose2):
        """Calculate distance between two poses"""
        # Implementation would calculate 3D distance
        pass
```

### Temporal Context and History

Maintaining interaction history helps with reference resolution and task continuity:

```python
class ContextMemory:
    def __init__(self, max_history=50):
        self.max_history = max_history
        self.interaction_history = []
        self.object_references = {}  # Track object mentions
        self.location_references = {}  # Track location mentions

    def add_interaction(self, instruction, action, result):
        """Add interaction to history"""
        interaction = {
            'timestamp': time.time(),
            'instruction': instruction,
            'action': action,
            'result': result,
            'objects_mentioned': self.extract_objects(instruction)
        }

        self.interaction_history.append(interaction)

        # Maintain history size
        if len(self.interaction_history) > self.max_history:
            self.interaction_history.pop(0)

    def get_recent(self, n=5):
        """Get n most recent interactions"""
        return self.interaction_history[-n:]

    def resolve_reference(self, reference, current_context):
        """Resolve linguistic reference using context and history"""
        if reference in ['it', 'this', 'that']:
            # Resolve pronoun to most recently mentioned object
            recent_objects = self.get_recent_objects()
            if recent_objects:
                return recent_objects[-1]

        elif reference in ['there', 'here']:
            # Resolve spatial reference
            if reference == 'here':
                return current_context['robot_pose']
            elif reference == 'there':
                # 'There' often refers to location mentioned in same instruction
                # or location pointed to by demonstrative
                pass

        return None

    def get_recent_objects(self):
        """Get recently mentioned objects"""
        recent_objects = []
        for interaction in reversed(self.interaction_history[-10:]):
            if 'objects_mentioned' in interaction:
                recent_objects.extend(interaction['objects_mentioned'])
        return recent_objects

    def extract_objects(self, instruction):
        """Extract object references from instruction"""
        # Implementation would use NLP to identify objects
        pass

    def update_object_location(self, object_id, new_location):
        """Update object location in memory"""
        if object_id in self.object_references:
            self.object_references[object_id]['location'] = new_location
            self.object_references[object_id]['last_seen'] = time.time()
```

## Instruction-to-Action Mapping

### Action Space Mapping

Converting parsed instructions to robot actions requires careful mapping:

```python
class InstructionToActionMapper:
    def __init__(self):
        self.action_space = self.define_action_space()
        self.instruction_templates = self.load_instruction_templates()

    def define_action_space(self):
        """Define the robot's action space"""
        return {
            'navigation': {
                'move_forward': {'params': ['distance']},
                'turn_left': {'params': ['angle']},
                'turn_right': {'params': ['angle']},
                'navigate_to': {'params': ['location', 'orientation']}
            },
            'manipulation': {
                'grasp': {'params': ['object', 'grasp_type']},
                'place': {'params': ['location', 'orientation']},
                'pick_and_place': {'params': ['source', 'target']},
                'push': {'params': ['object', 'direction', 'force']},
                'pull': {'params': ['object', 'direction', 'force']}
            },
            'perception': {
                'look_at': {'params': ['location']},
                'find_object': {'params': ['object_type']},
                'scan_area': {'params': ['location', 'radius']}
            }
        }

    def map_instruction_to_action(self, parsed_instruction, context):
        """Map parsed instruction to executable action"""
        instruction_type = self.classify_instruction(parsed_instruction)

        if instruction_type == 'navigation':
            return self.map_navigation_instruction(parsed_instruction, context)
        elif instruction_type == 'manipulation':
            return self.map_manipulation_instruction(parsed_instruction, context)
        elif instruction_type == 'perception':
            return self.map_perception_instruction(parsed_instruction, context)
        else:
            return self.handle_unknown_instruction(parsed_instruction)

    def map_navigation_instruction(self, parsed_instruction, context):
        """Map navigation-related instructions to actions"""
        action_type = parsed_instruction.get('action_type', 'navigate_to')

        if action_type in ['go', 'move', 'navigate', 'walk']:
            target_location = parsed_instruction.get('location')
            if target_location:
                # Resolve location to specific coordinates
                resolved_location = self.resolve_location(target_location, context)
                return {
                    'action': 'navigate_to',
                    'parameters': {
                        'target_pose': resolved_location,
                        'avoid_obstacles': True
                    }
                }

        elif action_type in ['forward', 'backward', 'left', 'right']:
            # Relative movement
            distance = parsed_instruction.get('distance', 1.0)
            return {
                'action': f'move_{action_type}',
                'parameters': {'distance': distance}
            }

    def map_manipulation_instruction(self, parsed_instruction, context):
        """Map manipulation-related instructions to actions"""
        action_type = parsed_instruction.get('action_type', 'grasp')

        if action_type in ['grasp', 'pick', 'grab', 'take']:
            object_to_grasp = parsed_instruction.get('object')
            if object_to_grasp:
                return {
                    'action': 'grasp',
                    'parameters': {
                        'object': object_to_grasp,
                        'grasp_type': 'precision'
                    }
                }

        elif action_type in ['place', 'put', 'set', 'drop']:
            target_location = parsed_instruction.get('location', 'default')
            return {
                'action': 'place',
                'parameters': {
                    'location': target_location,
                    'orientation': 'default'
                }
            }

    def resolve_location(self, location_desc, context):
        """Resolve location description to coordinates"""
        # This would use spatial reasoning and mapping
        if location_desc in context['known_locations']:
            return context['known_locations'][location_desc]
        else:
            # Use spatial reasoning to find location
            return self.spatial_reasoner.find_location(location_desc, context)

    def classify_instruction(self, parsed_instruction):
        """Classify instruction type"""
        action = parsed_instruction.get('action_type', '').lower()

        navigation_keywords = ['go', 'move', 'navigate', 'walk', 'drive', 'travel']
        manipulation_keywords = ['grasp', 'pick', 'place', 'put', 'take', 'grab']
        perception_keywords = ['look', 'find', 'see', 'scan', 'search']

        if any(keyword in action for keyword in navigation_keywords):
            return 'navigation'
        elif any(keyword in action for keyword in manipulation_keywords):
            return 'manipulation'
        elif any(keyword in action for keyword in perception_keywords):
            return 'perception'
        else:
            return 'unknown'
```

## Error Handling and Recovery

### Robust Execution Framework

Handling errors gracefully is crucial for reliable instruction following:

```python
class RobustInstructionExecutor:
    def __init__(self):
        self.error_handlers = self.initialize_error_handlers()
        self.recovery_strategies = self.load_recovery_strategies()

    def execute_with_error_handling(self, instruction_plan, context):
        """Execute instruction plan with error handling"""
        for i, action in enumerate(instruction_plan):
            try:
                result = self.execute_action(action, context)

                if not result['success']:
                    # Handle failure
                    recovery_plan = self.generate_recovery_plan(
                        action, result['error'], context
                    )
                    if recovery_plan:
                        self.execute_recovery_plan(recovery_plan, context)
                    else:
                        # Cannot recover, escalate
                        raise InstructionExecutionError(
                            f"Cannot execute action: {action}, error: {result['error']}"
                        )

            except Exception as e:
                # Handle unexpected errors
                recovery_plan = self.handle_unexpected_error(e, context)
                if recovery_plan:
                    self.execute_recovery_plan(recovery_plan, context)
                else:
                    raise

    def generate_recovery_plan(self, failed_action, error, context):
        """Generate recovery plan for failed action"""
        error_type = self.classify_error(error)

        if error_type == 'object_not_found':
            # Object not found - try alternative object or ask for clarification
            return self.recovery_object_not_found(failed_action, context)

        elif error_type == 'navigation_failure':
            # Navigation failed - try alternative path or approach
            return self.recovery_navigation_failure(failed_action, context)

        elif error_type == 'grasp_failure':
            # Grasp failed - try different grasp or approach
            return self.recovery_grasp_failure(failed_action, context)

        elif error_type == 'collision_detected':
            # Collision detected - replan or ask for help
            return self.recovery_collision_detected(failed_action, context)

        return None

    def recovery_object_not_found(self, action, context):
        """Recovery plan when object is not found"""
        object_type = action['parameters'].get('object')

        # Strategy 1: Search in alternative locations
        alternative_locations = self.get_alternative_locations(object_type, context)
        if alternative_locations:
            search_actions = []
            for location in alternative_locations:
                search_actions.append({
                    'action': 'navigate_to',
                    'parameters': {'target_pose': location}
                })
                search_actions.append({
                    'action': 'find_object',
                    'parameters': {'object_type': object_type}
                })

            # Add original action if object found
            search_actions.append(action)
            return search_actions

        # Strategy 2: Ask for clarification
        return [{
            'action': 'request_clarification',
            'parameters': {
                'message': f"I couldn't find the {object_type}. Can you help me locate it?",
                'options': ['show_location', 'different_object', 'cancel_task']
            }
        }]

    def recovery_navigation_failure(self, action, context):
        """Recovery plan when navigation fails"""
        target_pose = action['parameters']['target_pose']

        # Strategy 1: Try alternative path
        alternative_paths = self.find_alternative_paths(target_pose, context)
        if alternative_paths:
            return [{
                'action': 'navigate_to',
                'parameters': {'target_pose': alternative_paths[0]}
            }]

        # Strategy 2: Approach more carefully
        return [{
            'action': 'navigate_to',
            'parameters': {
                'target_pose': target_pose,
                'speed': 'slow',
                'safety_margin': 'high'
            }
        }]

    def recovery_grasp_failure(self, action, context):
        """Recovery plan when grasp fails"""
        object_info = action['parameters']['object']

        # Strategy 1: Try different grasp approach
        grasp_strategies = [
            'top_grasp', 'side_grasp', 'pinch_grasp', 'power_grasp'
        ]

        recovery_actions = []
        for strategy in grasp_strategies[1:]:  # Skip original strategy
            recovery_actions.append({
                'action': 'grasp',
                'parameters': {
                    'object': object_info,
                    'grasp_type': strategy
                }
            })

        return recovery_actions

    def classify_error(self, error):
        """Classify error type"""
        error_msg = str(error).lower()

        if 'not found' in error_msg or 'missing' in error_msg:
            return 'object_not_found'
        elif 'navigation' in error_msg or 'path' in error_msg or 'obstacle' in error_msg:
            return 'navigation_failure'
        elif 'grasp' in error_msg or 'gripper' in error_msg or 'slip' in error_msg:
            return 'grasp_failure'
        elif 'collision' in error_msg or 'crash' in error_msg:
            return 'collision_detected'
        else:
            return 'unknown_error'

    def handle_unexpected_error(self, error, context):
        """Handle unexpected errors"""
        error_msg = f"Unexpected error occurred: {error}"

        # Log error for debugging
        self.log_error(error, context)

        # Ask for human assistance
        return [{
            'action': 'request_assistance',
            'parameters': {
                'message': error_msg,
                'context': context
            }
        }]

    def log_error(self, error, context):
        """Log error for debugging and improvement"""
        error_log = {
            'timestamp': time.time(),
            'error': str(error),
            'context': context,
            'stack_trace': traceback.format_exc()
        }

        # Save to persistent storage for analysis
        self.save_error_log(error_log)
```

## Isaac Integration for Instruction Following

### ROS 2 Interface for Instruction Following

Integrating instruction following with ROS 2 and Isaac systems:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Image, CameraInfo
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class IsaacInstructionFollowerNode(Node):
    def __init__(self):
        super().__init__('isaac_instruction_follower')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_publisher = self.create_publisher(String, 'instruction_status', 10)

        # Subscribers
        self.instruction_subscriber = self.create_subscription(
            String, 'natural_language_instruction', self.instruction_callback, 10
        )

        # Action server for complex instructions
        self.instruction_server = ActionServer(
            self,
            ExecuteInstruction,
            'execute_instruction',
            self.execute_instruction_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Initialize instruction following system
        self.instruction_system = HierarchicalInstructionFollower()
        self.context_provider = ContextProvider()

        self.get_logger().info('Isaac Instruction Follower initialized')

    def instruction_callback(self, msg):
        """Handle incoming natural language instruction"""
        try:
            self.get_logger().info(f'Received instruction: {msg.data}')

            # Publish status
            status_msg = String()
            status_msg.data = f'Processing: {msg.data}'
            self.status_publisher.publish(status_msg)

            # Get current context
            context = self.context_provider.get_current_context()

            # Follow instruction
            result = self.instruction_system.follow_instruction(msg.data, context)

            # Publish completion status
            status_msg.data = f'Completed: {msg.data}'
            self.status_publisher.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error following instruction: {e}')
            status_msg = String()
            status_msg.data = f'Error: {str(e)}'
            self.status_publisher.publish(status_msg)

    def goal_callback(self, goal_request):
        """Handle instruction execution goal"""
        self.get_logger().info(f'Received instruction goal: {goal_request.instruction}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle instruction execution cancellation"""
        self.get_logger().info('Instruction execution cancelled')
        return CancelResponse.ACCEPT

    def execute_instruction_callback(self, goal_handle):
        """Execute instruction as action"""
        feedback_msg = ExecuteInstruction.Feedback()
        result_msg = ExecuteInstruction.Result()

        try:
            instruction = goal_handle.request.instruction
            self.get_logger().info(f'Executing instruction: {instruction}')

            # Get context
            context = self.context_provider.get_current_context()

            # Execute with progress feedback
            instruction_plan = self.instruction_system.task_planner.decompose(
                instruction, context
            )

            total_steps = len(instruction_plan)
            completed_steps = 0

            for i, subtask in enumerate(instruction_plan):
                # Update feedback
                feedback_msg.current_task = str(subtask)
                feedback_msg.progress = (i + 1) / total_steps * 100.0
                goal_handle.publish_feedback(feedback_msg)

                # Execute subtask
                action_sequence = self.instruction_system.action_planner.plan(
                    subtask, context
                )

                for action in action_sequence:
                    motion_commands = self.instruction_system.motion_planner.generate(
                        action, context
                    )
                    self.execute_motion(motion_commands)

                completed_steps += 1

            # Complete successfully
            result_msg.success = True
            result_msg.message = f'Successfully executed: {instruction}'
            goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f'Instruction execution failed: {e}')
            result_msg.success = False
            result_msg.message = f'Failed: {str(e)}'
            goal_handle.abort()

        return result_msg

    def execute_motion(self, motion_commands):
        """Execute motion commands"""
        # Publish velocity commands or joint trajectories
        # This would interface with robot controllers
        pass

class ContextProvider:
    def __init__(self):
        # This would interface with perception and mapping systems
        pass

    def get_current_context(self):
        """Get current environmental context"""
        # This would gather information from various sensors and systems
        context = {
            'robot_pose': self.get_robot_pose(),
            'map': self.get_current_map(),
            'objects': self.get_detected_objects(),
            'time': time.time()
        }
        return context

    def get_robot_pose(self):
        """Get current robot pose"""
        # Implementation would use localization system
        pass

    def get_current_map(self):
        """Get current navigation map"""
        # Implementation would use mapping system
        pass

    def get_detected_objects(self):
        """Get currently detected objects"""
        # Implementation would use perception system
        pass

# Action definition (would be in .action file)
class ExecuteInstruction:
    def __init__(self):
        self.instruction = ""
        self.timeout = 0.0

    class Feedback:
        def __init__(self):
            self.current_task = ""
            self.progress = 0.0

    class Result:
        def __init__(self):
            self.success = False
            self.message = ""
```

## Evaluation and Validation

### Instruction Following Benchmarks

Evaluating instruction following systems requires comprehensive benchmarks:

```python
class InstructionFollowingEvaluator:
    def __init__(self, instruction_follower):
        self.instruction_follower = instruction_follower
        self.metrics = []

    def evaluate_on_dataset(self, dataset):
        """Evaluate instruction follower on dataset"""
        results = {
            'success_rate': 0.0,
            'average_time': 0.0,
            'error_types': {},
            'context_usage': 0.0
        }

        total_tasks = len(dataset)
        successful_tasks = 0
        total_time = 0.0

        for instruction, expected_result, context in dataset:
            start_time = time.time()

            try:
                actual_result = self.instruction_follower.follow_instruction(
                    instruction, context
                )

                if self.evaluate_result(actual_result, expected_result):
                    successful_tasks += 1
                    execution_time = time.time() - start_time
                    total_time += execution_time
                else:
                    self.record_error('incorrect_result', instruction, context)

            except Exception as e:
                self.record_error('execution_error', instruction, str(e))

        results['success_rate'] = successful_tasks / total_tasks if total_tasks > 0 else 0
        results['average_time'] = total_time / successful_tasks if successful_tasks > 0 else 0

        return results

    def evaluate_result(self, actual, expected):
        """Evaluate if actual result matches expected result"""
        # This would implement domain-specific evaluation
        # For navigation: check if robot reached correct location
        # For manipulation: check if object was moved correctly
        # For complex tasks: check if overall goal was achieved
        pass

    def evaluate_context_usage(self, instruction, context):
        """Evaluate if system properly uses contextual information"""
        # Test if system can resolve ambiguities using context
        ambiguous_instruction = "Grasp it"

        # System should use context to determine what "it" refers to
        result = self.instruction_follower.follow_instruction(ambiguous_instruction, context)

        # Check if the correct object was grasped based on context
        correct_resolution = self.check_context_resolution(result, context)
        return correct_resolution

    def test_robustness(self, instruction_follower):
        """Test robustness to various challenges"""
        robustness_tests = {
            'noise_resilience': self.test_noise_resilience,
            'ambiguity_resolution': self.test_ambiguity_resolution,
            'error_recovery': self.test_error_recovery,
            'temporal_coherence': self.test_temporal_coherence
        }

        results = {}
        for test_name, test_func in robustness_tests.items():
            results[test_name] = test_func(instruction_follower)

        return results

    def test_ambiguity_resolution(self, instruction_follower):
        """Test ability to resolve ambiguous instructions"""
        ambiguous_cases = [
            ("Bring the cup", {"cups": [{"id": 1, "color": "red"}, {"id": 2, "color": "blue"}]}),
            ("Go there", {"pointed_location": [1.0, 2.0, 0.0]}),
            ("Do it again", {"previous_action": "pick_object"})
        ]

        resolved_correctly = 0
        for instruction, context in ambiguous_cases:
            try:
                result = instruction_follower.follow_instruction(instruction, context)
                if self.check_ambiguity_resolution(result, instruction, context):
                    resolved_correctly += 1
            except:
                pass  # Count as failure

        return resolved_correctly / len(ambiguous_cases)

# Example usage
def main():
    # Initialize instruction follower
    instruction_follower = HierarchicalInstructionFollower()

    # Create evaluator
    evaluator = InstructionFollowingEvaluator(instruction_follower)

    # Run evaluation
    results = evaluator.evaluate_on_dataset(test_dataset)

    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Average Time: {results['average_time']:.2f}s")

if __name__ == "__main__":
    main()
```

## Summary

Instruction following and task planning form the cognitive core of Vision-Language-Action systems, enabling robots to interpret natural language commands and execute them as sequences of physical actions. Through hierarchical planning, semantic parsing, context awareness, and robust error handling, these systems bridge the gap between human communication and robotic execution.

The next section will explore embodied language models, which connect abstract language concepts to concrete physical experiences and robotic capabilities.

## References

[All sources will be cited in the References section at the end of the book, following APA format]