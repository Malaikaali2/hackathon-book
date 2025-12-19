---
sidebar_position: 7
---

# Natural Language to Robot Action Mapping

## Learning Objectives

By the end of this section, you will be able to:

1. Design systems that translate natural language commands into executable robot actions
2. Implement semantic parsing techniques that extract action-relevant information from language
3. Create robust mapping frameworks that handle ambiguous and complex language
4. Develop context-aware translation mechanisms that consider environmental constraints
5. Build validation and verification systems for language-to-action translation
6. Create fallback strategies for handling unmappable language commands

## Introduction to Language-to-Action Mapping

Natural language to robot action mapping is the critical process that transforms human linguistic expressions into executable robotic behaviors. This transformation involves multiple levels of interpretation, from understanding the literal meaning of words to inferring the intended action and grounding it in the robot's physical capabilities and environmental context.

The complexity of this mapping stems from the fundamental differences between natural language and robotic action:

- **Ambiguity**: Natural language is inherently ambiguous, while robot actions require precise specifications
- **Context Dependency**: Language meaning often depends on context, which must be properly interpreted
- **Physical Constraints**: Robot capabilities and environmental constraints limit feasible actions
- **Temporal Aspects**: Language may describe sequences, durations, and temporal relationships
- **Spatial Reasoning**: Many commands involve spatial relationships and navigation

### The Translation Pipeline

The language-to-action mapping typically follows a pipeline approach:

```
Natural Language → Semantic Parsing → Action Extraction → Parameter Grounding → Execution Planning → Robot Control
```

Each stage refines the representation and adds more concrete details until the action can be executed by the robot's control systems.

## Semantic Parsing for Action Extraction

### Grammar-Based Semantic Parsing

Grammar-based approaches use formal rules to extract action-relevant information:

```python
import nltk
from nltk import CFG
from nltk.parse import ChartParser
import re
from typing import Dict, List, Tuple

class GrammarBasedSemanticParser:
    def __init__(self):
        self.grammar = self.create_robot_command_grammar()
        self.parser = ChartParser(self.grammar)
        self.action_templates = self.load_action_templates()

    def create_robot_command_grammar(self):
        """Create grammar for robot command language"""
        grammar_str = """
            S -> COMMAND | SUBORDINATE_CLAUSE S
            COMMAND -> ACTION OBJECT LOCATION
                   | ACTION LOCATION
                   | ACTION OBJECT
                   | ACTION
                   | NAVIGATE LOCATION
                   | GRASP OBJECT
                   | PLACE LOCATION
            SUBORDINATE_CLAUSE -> 'then' | 'after' | 'and' | 'before'
            ACTION -> 'go' | 'move' | 'navigate' | 'walk' | 'drive' | 'travel'
                    | 'grasp' | 'pick' | 'grab' | 'take' | 'lift'
                    | 'place' | 'put' | 'set' | 'drop'
                    | 'look' | 'find' | 'search' | 'locate'
                    | 'stop' | 'wait' | 'help'
            OBJECT -> 'ball' | 'cup' | 'box' | 'book' | 'red_ball' | 'blue_cup'
                    | 'the_ball' | 'a_cup' | 'it' | 'that' | 'this'
            LOCATION -> 'kitchen' | 'living_room' | 'bedroom' | 'office'
                      | 'table' | 'shelf' | 'counter' | 'couch'
                      | 'left' | 'right' | 'front' | 'back'
                      | 'near_ball' | 'by_cup'
        """
        return CFG.fromstring(grammar_str)

    def load_action_templates(self):
        """Load predefined action templates"""
        return {
            'navigate_to': {
                'pattern': r'(?:go to|move to|navigate to|walk to|drive to)\s+(?:the\s+)?(\w+)',
                'action_type': 'navigation',
                'parameters': ['target_location']
            },
            'grasp_object': {
                'pattern': r'(?:grasp|pick up|take|grab)\s+(?:the\s+)?(\w+)',
                'action_type': 'manipulation',
                'parameters': ['object_to_grasp']
            },
            'place_object': {
                'pattern': r'(?:place|put|set)\s+(?:the\s+)?(\w+)\s+(?:on|at|in)\s+(?:the\s+)?(\w+)',
                'action_type': 'manipulation',
                'parameters': ['object_to_place', 'target_location']
            }
        }

    def parse_command(self, command: str) -> Dict:
        """Parse command using grammar-based approach"""
        tokens = command.lower().split()

        try:
            # Parse using CFG
            parses = list(self.parser.parse(tokens))
            if parses:
                return self.extract_action_from_parse(parses[0])
        except:
            # If grammar parsing fails, use regex patterns
            return self.regex_parse_command(command)

        return {'action_type': 'unknown', 'parameters': {}}

    def extract_action_from_parse(self, parse_tree):
        """Extract action from parse tree"""
        action_info = {'action_type': None, 'parameters': {}}

        for subtree in parse_tree.subtrees():
            if subtree.label() == 'ACTION':
                action_info['action_type'] = str(subtree[0])
            elif subtree.label() == 'OBJECT':
                action_info['parameters']['object'] = str(subtree[0])
            elif subtree.label() == 'LOCATION':
                action_info['parameters']['location'] = str(subtree[0])

        # Map to specific action type
        action_info['action_type'] = self.map_to_specific_action(action_info['action_type'])
        return action_info

    def regex_parse_command(self, command: str) -> Dict:
        """Parse command using regex patterns as fallback"""
        command_lower = command.lower()

        for template_name, template in self.action_templates.items():
            match = re.search(template['pattern'], command_lower)
            if match:
                params = {}
                for i, param_name in enumerate(template['parameters']):
                    if i < len(match.groups()):
                        params[param_name] = match.group(i + 1)

                return {
                    'action_type': template['action_type'],
                    'template': template_name,
                    'parameters': params
                }

        return {'action_type': 'unknown', 'parameters': {}}

    def map_to_specific_action(self, generic_action: str) -> str:
        """Map generic action to specific robot action"""
        action_mapping = {
            'go': 'navigate_to',
            'move': 'navigate_to',
            'navigate': 'navigate_to',
            'walk': 'navigate_to',
            'grasp': 'grasp_object',
            'pick': 'grasp_object',
            'grab': 'grasp_object',
            'place': 'place_object',
            'put': 'place_object',
            'set': 'place_object'
        }

        return action_mapping.get(generic_action, generic_action)
```

### Neural Semantic Parsing

Neural approaches can handle more complex and ambiguous language:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

class NeuralSemanticParser(nn.Module):
    def __init__(self, vocab_size=30522, hidden_dim=768, action_dim=100):
        super().__init__()

        # Language model for encoding
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.language_model = AutoModel.from_pretrained('bert-base-uncased')

        # Action classification head
        self.action_classifier = nn.Linear(hidden_dim, action_dim)

        # Parameter extraction heads
        self.object_extractor = nn.Linear(hidden_dim, hidden_dim)
        self.location_extractor = nn.Linear(hidden_dim, hidden_dim)
        self.quantity_extractor = nn.Linear(hidden_dim, hidden_dim)

        # Action embedding space
        self.action_embeddings = nn.Embedding(action_dim, hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_text):
        """Parse input text into action representation"""
        # Tokenize and encode
        encoded = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        # Get language model output
        outputs = self.language_model(**encoded)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_dim]

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classify action
        action_logits = self.action_classifier(pooled_output)
        action_probs = F.softmax(action_logits, dim=-1)

        # Extract parameters
        object_features = self.object_extractor(pooled_output)
        location_features = self.location_extractor(pooled_output)
        quantity_features = self.quantity_extractor(pooled_output)

        return {
            'action_probs': action_probs,
            'object_features': object_features,
            'location_features': location_features,
            'quantity_features': quantity_features,
            'pooled_features': pooled_output
        }

    def extract_action_structure(self, input_text):
        """Extract structured action from text"""
        outputs = self.forward(input_text)

        # Get predicted action
        predicted_action_id = torch.argmax(outputs['action_probs'], dim=-1).item()

        # Get action name (would map to actual action names)
        action_name = f"action_{predicted_action_id}"

        # Extract parameters using attention over tokens
        encoded = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        token_embeddings = self.language_model(**encoded).last_hidden_state
        attention_weights = torch.softmax(
            torch.matmul(token_embeddings, outputs['object_features'].unsqueeze(-1)).squeeze(-1),
            dim=-1
        )

        # Extract relevant tokens based on attention
        relevant_tokens = torch.argmax(attention_weights, dim=-1)
        extracted_params = self.extract_parameters_from_tokens(
            input_text, relevant_tokens
        )

        return {
            'action_type': action_name,
            'parameters': extracted_params,
            'confidence': torch.max(outputs['action_probs']).item()
        }

    def extract_parameters_from_tokens(self, text, attention_indices):
        """Extract parameters by analyzing attended tokens"""
        tokens = self.tokenizer.tokenize(text)

        # Simple extraction based on attention weights
        # In practice, use more sophisticated parameter extraction
        params = {}

        # Look for object-related tokens
        object_keywords = ['ball', 'cup', 'box', 'book', 'table', 'chair']
        for token in tokens:
            if token in object_keywords:
                params['object'] = token
                break

        # Look for location-related tokens
        location_keywords = ['kitchen', 'bedroom', 'office', 'table', 'shelf']
        for token in tokens:
            if token in location_keywords:
                params['location'] = token
                break

        return params

class HybridSemanticParser:
    def __init__(self):
        self.grammar_parser = GrammarBasedSemanticParser()
        self.neural_parser = NeuralSemanticParser()

    def parse_command(self, command: str, context: Dict = None) -> Dict:
        """Parse command using hybrid approach"""
        # Try grammar-based parsing first (for structured commands)
        grammar_result = self.grammar_parser.parse_command(command)

        if grammar_result['action_type'] != 'unknown':
            return self.refine_with_context(grammar_result, context)

        # Fall back to neural parsing for complex/ambiguous commands
        neural_result = self.neural_parser.extract_action_structure(command)

        # Convert neural result to expected format
        refined_result = {
            'action_type': neural_result['action_type'],
            'parameters': neural_result['parameters'],
            'confidence': neural_result.get('confidence', 0.5)
        }

        return self.refine_with_context(refined_result, context)

    def refine_with_context(self, parsed_result: Dict, context: Dict) -> Dict:
        """Refine parsed result using contextual information"""
        if not context:
            return parsed_result

        # Resolve pronouns using context
        if 'object' in parsed_result['parameters']:
            resolved_object = self.resolve_pronoun(
                parsed_result['parameters']['object'], context
            )
            if resolved_object:
                parsed_result['parameters']['object'] = resolved_object

        # Ground locations in context
        if 'location' in parsed_result['parameters']:
            resolved_location = self.ground_location(
                parsed_result['parameters']['location'], context
            )
            if resolved_location:
                parsed_result['parameters']['resolved_location'] = resolved_location

        # Validate against robot capabilities
        if 'action_type' in parsed_result:
            if not self.is_action_feasible(parsed_result['action_type'], context):
                parsed_result['action_type'] = 'request_clarification'

        return parsed_result

    def resolve_pronoun(self, pronoun: str, context: Dict) -> str:
        """Resolve pronouns using context"""
        pronoun_lower = pronoun.lower()

        if pronoun_lower in ['it', 'that', 'this']:
            # Get most recently mentioned object
            recent_objects = context.get('recent_objects', [])
            if recent_objects:
                return recent_objects[-1]

        return pronoun  # Return as-is if cannot resolve

    def ground_location(self, location: str, context: Dict) -> Dict:
        """Ground location reference in context"""
        named_locations = context.get('named_locations', {})

        for name, pose in named_locations.items():
            if location.lower() in name.lower():
                return {
                    'name': name,
                    'pose': pose,
                    'grounded': True
                }

        return {'name': location, 'grounded': False}

    def is_action_feasible(self, action_type: str, context: Dict) -> bool:
        """Check if action is feasible given robot capabilities"""
        robot_capabilities = context.get('robot_capabilities', [])

        action_requirements = {
            'navigation': ['navigation'],
            'manipulation': ['manipulation'],
            'perception': ['perception']
        }

        required_capability = action_requirements.get(action_type, [])
        return any(cap in robot_capabilities for cap in required_capability)
```

## Context-Aware Action Mapping

### Environmental Context Integration

Effective language-to-action mapping must consider environmental context:

```python
class ContextAwareActionMapper:
    def __init__(self):
        self.spatial_reasoner = SpatialReasoner()
        self.object_resolver = ObjectResolver()
        self.constraint_checker = ConstraintChecker()

    def map_with_context(self, parsed_command: Dict, environment_context: Dict) -> Dict:
        """Map command to action considering environmental context"""
        action_mapping = {
            'action_type': parsed_command['action_type'],
            'original_parameters': parsed_command['parameters'],
            'contextual_parameters': {},
            'feasibility': True,
            'confidence': parsed_command.get('confidence', 0.8)
        }

        # Ground spatial references
        if 'location' in parsed_command['parameters']:
            grounded_location = self.spatial_reasoner.ground_location(
                parsed_command['parameters']['location'], environment_context
            )
            action_mapping['contextual_parameters']['target_pose'] = grounded_location

        # Resolve object references
        if 'object' in parsed_command['parameters']:
            resolved_object = self.object_resolver.resolve_object(
                parsed_command['parameters']['object'], environment_context
            )
            action_mapping['contextual_parameters']['target_object'] = resolved_object

        # Check constraints
        action_mapping['feasibility'] = self.constraint_checker.check_constraints(
            action_mapping, environment_context
        )

        # Adjust parameters based on context
        action_mapping = self.adjust_parameters_for_context(
            action_mapping, environment_context
        )

        return action_mapping

    def adjust_parameters_for_context(self, action_mapping: Dict, context: Dict) -> Dict:
        """Adjust action parameters based on context"""
        action_type = action_mapping['action_type']
        params = action_mapping['contextual_parameters']

        if action_type == 'navigate_to':
            # Adjust navigation parameters based on environment
            target_pose = params.get('target_pose')
            if target_pose:
                # Consider obstacles, doorways, etc.
                adjusted_params = self.adjust_navigation_params(target_pose, context)
                params.update(adjusted_params)

        elif action_type == 'grasp_object':
            # Adjust grasp parameters based on object properties
            target_object = params.get('target_object')
            if target_object:
                adjusted_params = self.adjust_grasp_params(target_object, context)
                params.update(adjusted_params)

        return action_mapping

    def adjust_navigation_params(self, target_pose: Dict, context: Dict) -> Dict:
        """Adjust navigation parameters based on environment"""
        adjusted = {}

        # Consider doorways and narrow passages
        if context.get('map'):
            path = self.find_path_to_target(target_pose, context['map'])
            adjusted['preferred_path'] = path

        # Consider safety constraints
        safety_margin = context.get('safety_settings', {}).get('margin', 0.1)
        adjusted['safety_margin'] = safety_margin

        # Consider speed constraints in crowded areas
        if self.is_area_crowded(target_pose, context):
            adjusted['max_speed'] = 0.2  # Slow down in crowded areas
        else:
            adjusted['max_speed'] = 0.5  # Normal speed

        return adjusted

    def adjust_grasp_params(self, target_object: Dict, context: Dict) -> Dict:
        """Adjust grasp parameters based on object properties"""
        adjusted = {}

        # Choose appropriate grasp based on object properties
        obj_shape = target_object.get('shape', 'unknown')
        obj_size = target_object.get('dimensions', {'width': 0.1, 'height': 0.1, 'depth': 0.1})
        obj_weight = target_object.get('weight', 0.1)

        grasp_type = self.select_grasp_type(obj_shape, obj_size, obj_weight)
        adjusted['grasp_type'] = grasp_type

        # Adjust gripper width
        max_dimension = max(obj_size['width'], obj_size['depth'])
        adjusted['gripper_width'] = max_dimension * 1.2  # Add safety margin

        # Consider approach direction based on object orientation
        obj_orientation = target_object.get('orientation', [0, 0, 0, 1])
        approach_dir = self.compute_approach_direction(obj_orientation, obj_shape)
        adjusted['approach_direction'] = approach_dir

        return adjusted

    def select_grasp_type(self, shape: str, dimensions: Dict, weight: float) -> str:
        """Select appropriate grasp type based on object properties"""
        if shape == 'cylindrical':
            return 'side_grasp'
        elif dimensions['height'] < dimensions['width'] and dimensions['height'] < dimensions['depth']:
            return 'top_grasp'
        elif weight > 0.5:  # Heavy object
            return 'power_grasp'
        else:
            return 'precision_grasp'

    def compute_approach_direction(self, orientation: List[float], shape: str) -> List[float]:
        """Compute optimal approach direction based on object orientation"""
        # Default approach direction (from above for top grasp)
        if shape == 'cylindrical':
            return [0, 0, -1]  # Approach from above
        else:
            return [0, -1, 0]  # Approach from front

    def is_area_crowded(self, target_pose: Dict, context: Dict) -> bool:
        """Check if target area is crowded"""
        # Implementation would check for nearby people/objects
        # using occupancy grid or detection results
        return False

    def find_path_to_target(self, target_pose: Dict, map_data: Dict) -> List[Dict]:
        """Find path to target considering the map"""
        # Implementation would use path planning algorithm
        return [{'x': 0, 'y': 0}, target_pose]

class SpatialReasoner:
    def __init__(self):
        self.spatial_relations = {
            'relative': ['left', 'right', 'front', 'back', 'near', 'far'],
            'absolute': ['kitchen', 'living_room', 'bedroom', 'office']
        }

    def ground_location(self, location_ref: str, context: Dict) -> Dict:
        """Ground location reference in environmental context"""
        location_lower = location_ref.lower()

        # Check for absolute locations
        absolute_locations = context.get('named_locations', {})
        for name, pose in absolute_locations.items():
            if location_lower in name.lower():
                return {
                    'type': 'absolute',
                    'name': name,
                    'pose': pose,
                    'resolved': True
                }

        # Check for relative locations
        if any(rel in location_lower for rel in self.spatial_relations['relative']):
            return self.resolve_relative_location(location_ref, context)

        # Default: return as reference
        return {
            'type': 'reference',
            'name': location_ref,
            'resolved': False
        }

    def resolve_relative_location(self, location_ref: str, context: Dict) -> Dict:
        """Resolve relative location reference"""
        # Parse relative location (e.g., "to the left of the table")
        words = location_ref.split()

        # Find spatial relation and reference object
        spatial_rel = None
        ref_object = None

        for i, word in enumerate(words):
            if word in self.spatial_relations['relative']:
                spatial_rel = word
                # Look for object after the relation
                if i + 2 < len(words):
                    ref_object = ' '.join(words[i+2:])
                break

        if spatial_rel and ref_object:
            # Find reference object in context
            ref_obj = self.find_object_in_context(ref_object, context)
            if ref_obj and 'pose' in ref_obj:
                # Compute relative position
                relative_pose = self.compute_relative_position(
                    ref_obj['pose'], spatial_rel, context
                )

                return {
                    'type': 'relative',
                    'spatial_relation': spatial_rel,
                    'reference_object': ref_obj,
                    'pose': relative_pose,
                    'resolved': True
                }

        return {
            'type': 'relative',
            'name': location_ref,
            'resolved': False
        }

    def find_object_in_context(self, object_ref: str, context: Dict) -> Dict:
        """Find object in context"""
        detected_objects = context.get('objects', [])

        for obj in detected_objects:
            obj_name = obj.get('type', '').lower()
            if object_ref.lower() in obj_name:
                return obj

        return None

    def compute_relative_position(self, ref_pose: List[float], relation: str, context: Dict) -> List[float]:
        """Compute position relative to reference object"""
        ref_x, ref_y, ref_z = ref_pose[:3]

        # Define relative offsets
        offsets = {
            'left': (-0.5, 0, 0),
            'right': (0.5, 0, 0),
            'front': (0, 0.5, 0),
            'back': (0, -0.5, 0),
            'near': (0, 0, 0),  # Same position
            'far': (1, 1, 0)   # Further away
        }

        offset = offsets.get(relation, (0, 0, 0))
        return [ref_x + offset[0], ref_y + offset[1], ref_z + offset[2]]

class ObjectResolver:
    def __init__(self):
        self.object_keywords = {
            'graspable': ['ball', 'cup', 'box', 'book', 'bottle'],
            'furniture': ['table', 'chair', 'shelf', 'couch', 'counter'],
            'locations': ['kitchen', 'bedroom', 'office', 'living_room']
        }

    def resolve_object(self, object_ref: str, context: Dict) -> Dict:
        """Resolve object reference in context"""
        object_lower = object_ref.lower()

        # Check for pronouns
        if object_lower in ['it', 'that', 'this']:
            return self.resolve_pronoun(object_lower, context)

        # Find in detected objects
        detected_objects = context.get('objects', [])
        for obj in detected_objects:
            obj_type = obj.get('type', '').lower()
            obj_color = obj.get('color', '').lower()

            # Check if reference matches object
            if (object_lower == obj_type or
                object_lower == f"{obj_color}_{obj_type}" or
                obj_type in object_lower):
                return obj

        # If not found, return reference with confidence
        return {
            'type': object_ref,
            'found': False,
            'confidence': 0.1
        }

    def resolve_pronoun(self, pronoun: str, context: Dict) -> Dict:
        """Resolve pronoun using context"""
        if pronoun in ['it', 'that', 'this']:
            # Get most recently mentioned object
            recent_objects = context.get('recent_objects', [])
            if recent_objects:
                return recent_objects[-1]

        return {'type': pronoun, 'found': False, 'confidence': 0.0}
```

## Action Execution Planning

### Converting Mapped Actions to Executable Plans

Once actions are mapped from language, they need to be converted to executable plans:

```python
class ActionExecutionPlanner:
    def __init__(self):
        self.action_library = self.load_action_library()
        self.motion_planner = MotionPlanner()
        self.skill_executor = SkillExecutor()

    def load_action_library(self):
        """Load action templates with execution specifications"""
        return {
            'navigate_to': {
                'prerequisites': ['navigation_enabled', 'map_available'],
                'execution_steps': [
                    'plan_path',
                    'execute_navigation',
                    'verify_arrival'
                ],
                'parameters': ['target_pose', 'avoid_obstacles', 'max_speed'],
                'timeout': 30.0
            },
            'grasp_object': {
                'prerequisites': ['manipulation_enabled', 'object_reachable'],
                'execution_steps': [
                    'compute_approach_pose',
                    'execute_approach',
                    'execute_grasp',
                    'verify_grasp'
                ],
                'parameters': ['object_pose', 'grasp_type', 'gripper_width'],
                'timeout': 15.0
            },
            'place_object': {
                'prerequisites': ['object_grasped'],
                'execution_steps': [
                    'compute_place_pose',
                    'execute_place',
                    'verify_placement',
                    'release_object'
                ],
                'parameters': ['target_pose', 'orientation'],
                'timeout': 10.0
            },
            'find_object': {
                'prerequisites': ['perception_enabled'],
                'execution_steps': [
                    'scan_area',
                    'detect_objects',
                    'identify_target',
                    'update_context'
                ],
                'parameters': ['object_type', 'search_area'],
                'timeout': 20.0
            }
        }

    def plan_execution(self, action_mapping: Dict) -> List[Dict]:
        """Plan execution steps for mapped action"""
        action_type = action_mapping['action_type']

        if action_type not in self.action_library:
            raise ValueError(f"Unknown action type: {action_type}")

        action_spec = self.action_library[action_type]
        contextual_params = action_mapping.get('contextual_parameters', {})
        original_params = action_mapping.get('original_parameters', {})

        # Combine parameters
        all_params = {**original_params, **contextual_params}

        # Validate prerequisites
        if not self.check_prerequisites(action_type, all_params):
            raise ValueError(f"Prerequisites not met for action: {action_type}")

        # Create execution plan
        execution_plan = []
        for step in action_spec['execution_steps']:
            step_plan = self.create_step_plan(step, all_params, action_type)
            execution_plan.append(step_plan)

        return execution_plan

    def check_prerequisites(self, action_type: str, parameters: Dict) -> bool:
        """Check if action prerequisites are met"""
        action_spec = self.action_library.get(action_type, {})
        prerequisites = action_spec.get('prerequisites', [])

        # Check each prerequisite
        for prereq in prerequisites:
            if not self.evaluate_prerequisite(prereq, parameters):
                return False

        return True

    def evaluate_prerequisite(self, prerequisite: str, parameters: Dict) -> bool:
        """Evaluate specific prerequisite"""
        # This would interface with system state
        # For now, return True for all prerequisites
        return True

    def create_step_plan(self, step: str, parameters: Dict, action_type: str) -> Dict:
        """Create plan for specific execution step"""
        step_plan = {
            'step': step,
            'action_type': action_type,
            'parameters': parameters,
            'required_capabilities': self.get_required_capabilities(step, action_type),
            'timeout': self.get_step_timeout(step, action_type)
        }

        # Add step-specific logic
        if step == 'plan_path':
            step_plan['command'] = 'plan_navigation_path'
            step_plan['args'] = {
                'start_pose': parameters.get('current_pose'),
                'target_pose': parameters.get('target_pose'),
                'avoid_obstacles': parameters.get('avoid_obstacles', True)
            }
        elif step == 'execute_navigation':
            step_plan['command'] = 'execute_navigation'
            step_plan['args'] = {
                'target_pose': parameters.get('target_pose'),
                'max_speed': parameters.get('max_speed', 0.5)
            }
        elif step == 'compute_approach_pose':
            step_plan['command'] = 'compute_grasp_approach'
            step_plan['args'] = {
                'object_pose': parameters.get('object_pose'),
                'approach_direction': parameters.get('approach_direction', [0, 0, 1])
            }

        return step_plan

    def get_required_capabilities(self, step: str, action_type: str) -> List[str]:
        """Get required capabilities for execution step"""
        capability_map = {
            'navigate_to': {
                'plan_path': ['navigation', 'path_planning'],
                'execute_navigation': ['navigation', 'motion_control'],
                'verify_arrival': ['localization']
            },
            'grasp_object': {
                'compute_approach_pose': ['manipulation', 'kinematics'],
                'execute_approach': ['manipulation', 'motion_control'],
                'execute_grasp': ['manipulation', 'gripper_control'],
                'verify_grasp': ['force_sensing', 'gripper_feedback']
            }
        }

        action_caps = capability_map.get(action_type, {})
        return action_caps.get(step, [])

    def get_step_timeout(self, step: str, action_type: str) -> float:
        """Get timeout for execution step"""
        timeout_map = {
            'navigate_to': {
                'plan_path': 5.0,
                'execute_navigation': 30.0,
                'verify_arrival': 2.0
            },
            'grasp_object': {
                'compute_approach_pose': 1.0,
                'execute_approach': 10.0,
                'execute_grasp': 5.0,
                'verify_grasp': 2.0
            }
        }

        action_timeouts = timeout_map.get(action_type, {})
        return action_timeouts.get(step, 5.0)

class MotionPlanner:
    def __init__(self):
        self.path_planner = PathPlanner()
        self.trajectory_generator = TrajectoryGenerator()

    def plan_navigation_path(self, start_pose: List[float], target_pose: List[float],
                           map_data: Dict, avoid_obstacles: bool = True) -> List[List[float]]:
        """Plan navigation path from start to target"""
        # This would implement actual path planning
        # For now, return a simple straight-line path
        path = self.interpolate_path(start_pose, target_pose)
        return path

    def plan_manipulation_trajectory(self, start_pose: List[float], target_pose: List[float],
                                   constraints: Dict = None) -> List[Dict]:
        """Plan manipulation trajectory"""
        # This would implement Cartesian or joint-space trajectory planning
        trajectory = self.generate_trajectory(start_pose, target_pose, constraints)
        return trajectory

    def interpolate_path(self, start: List[float], end: List[float],
                        num_points: int = 10) -> List[List[float]]:
        """Interpolate path between start and end poses"""
        path = []
        for i in range(num_points + 1):
            t = i / num_points
            point = [
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                start[2] + t * (end[2] - start[2])
            ]
            path.append(point)
        return path

    def generate_trajectory(self, start_pose: List[float], target_pose: List[float],
                          constraints: Dict) -> List[Dict]:
        """Generate trajectory with constraints"""
        # Implementation would generate smooth trajectory
        # considering velocity, acceleration, and joint limits
        pass

class SkillExecutor:
    def __init__(self):
        self.skills = self.load_skills()

    def load_skills(self):
        """Load available robot skills"""
        return {
            'navigate': NavigateSkill(),
            'grasp': GraspSkill(),
            'place': PlaceSkill(),
            'look_at': LookAtSkill(),
            'find_object': FindObjectSkill()
        }

    def execute_skill(self, skill_name: str, parameters: Dict) -> Dict:
        """Execute robot skill with given parameters"""
        if skill_name not in self.skills:
            return {
                'success': False,
                'error': f'Skill {skill_name} not available',
                'result': None
            }

        skill = self.skills[skill_name]
        return skill.execute(parameters)

class BaseSkill:
    def __init__(self, name: str):
        self.name = name

    def execute(self, parameters: Dict) -> Dict:
        """Execute the skill with given parameters"""
        raise NotImplementedError

    def validate_parameters(self, parameters: Dict) -> bool:
        """Validate skill parameters"""
        return True

class NavigateSkill(BaseSkill):
    def __init__(self):
        super().__init__("navigate")

    def execute(self, parameters: Dict) -> Dict:
        """Execute navigation skill"""
        try:
            target_pose = parameters['target_pose']
            max_speed = parameters.get('max_speed', 0.5)
            avoid_obstacles = parameters.get('avoid_obstacles', True)

            # Execute navigation
            result = self.perform_navigation(target_pose, max_speed, avoid_obstacles)

            return {
                'success': result['reached_target'],
                'final_pose': result.get('final_pose'),
                'execution_time': result.get('time', 0.0),
                'path_length': result.get('path_length', 0.0),
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'result': None
            }

    def perform_navigation(self, target_pose: List[float], max_speed: float,
                         avoid_obstacles: bool) -> Dict:
        """Perform actual navigation"""
        # Implementation would interface with navigation stack
        # For now, simulate navigation
        import time
        start_time = time.time()

        # Simulate navigation
        time.sleep(0.1)  # Simulate execution time

        return {
            'reached_target': True,
            'final_pose': target_pose,
            'time': time.time() - start_time,
            'path_length': 1.0  # Simulated path length
        }
```

## Ambiguity Resolution and Fallback Strategies

### Handling Ambiguous Language

Language-to-action mapping must handle ambiguous commands gracefully:

```python
class AmbiguityResolver:
    def __init__(self):
        self.ambiguity_patterns = self.load_ambiguity_patterns()
        self.disambiguation_strategies = self.load_disambiguation_strategies()

    def load_ambiguity_patterns(self):
        """Load patterns that indicate ambiguity"""
        return {
            'object_ambiguity': [
                r'the \w+',  # "the ball" when multiple balls exist
                r'it',       # Pronoun without clear reference
                r'that'      # Demonstrative without clear reference
            ],
            'spatial_ambiguity': [
                r'over there',    # Vague location
                r'somewhere',     # Indefinite location
                r'around here'    # General area reference
            ],
            'action_ambiguity': [
                r'do something',  # Vague action request
                r'move it',      # Action without clear target
                r'handle that'   # Generic handling instruction
            ]
        }

    def load_disambiguation_strategies(self):
        """Load strategies for resolving ambiguities"""
        return {
            'object_ambiguity': [
                'ask_for_clarification',
                'use_most_recent_object',
                'use_closest_object',
                'use_largest_object'
            ],
            'spatial_ambiguity': [
                'ask_for_clarification',
                'use_robot_location',
                'use_pointing_gesture',
                'use_visual_attention'
            ],
            'action_ambiguity': [
                'ask_for_clarification',
                'use_default_action',
                'request_specific_action'
            ]
        }

    def detect_ambiguity(self, command: str, context: Dict) -> Dict:
        """Detect ambiguities in command"""
        ambiguity_info = {
            'types': [],
            'detected': False,
            'strategies': []
        }

        command_lower = command.lower()

        # Check for object ambiguity
        if self.check_pattern_match(command_lower, self.ambiguity_patterns['object_ambiguity']):
            ambiguity_info['types'].append('object_ambiguity')
            ambiguity_info['detected'] = True
            ambiguity_info['strategies'].extend(
                self.disambiguation_strategies['object_ambiguity']
            )

        # Check for spatial ambiguity
        if self.check_pattern_match(command_lower, self.ambiguity_patterns['spatial_ambiguity']):
            ambiguity_info['types'].append('spatial_ambiguity')
            ambiguity_info['detected'] = True
            ambiguity_info['strategies'].extend(
                self.disambiguation_strategies['spatial_ambiguity']
            )

        # Check for action ambiguity
        if self.check_pattern_match(command_lower, self.ambiguity_patterns['action_ambiguity']):
            ambiguity_info['types'].append('action_ambiguity')
            ambiguity_info['detected'] = True
            ambiguity_info['strategies'].extend(
                self.disambiguation_strategies['action_ambiguity']
            )

        return ambiguity_info

    def check_pattern_match(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given patterns"""
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def resolve_ambiguity(self, command: str, ambiguity_info: Dict, context: Dict) -> Dict:
        """Resolve detected ambiguities"""
        if not ambiguity_info['detected']:
            return {'command': command, 'resolved': True, 'clarification_needed': False}

        # For each ambiguity type, apply resolution strategy
        resolved_command = command
        clarification_needed = False

        for ambiguity_type in ambiguity_info['types']:
            if ambiguity_type == 'object_ambiguity':
                resolution = self.resolve_object_ambiguity(command, context)
                if resolution['clarification_needed']:
                    clarification_needed = True
                else:
                    resolved_command = resolution['command']
            elif ambiguity_type == 'spatial_ambiguity':
                resolution = self.resolve_spatial_ambiguity(command, context)
                if resolution['clarification_needed']:
                    clarification_needed = True
                else:
                    resolved_command = resolution['command']
            elif ambiguity_type == 'action_ambiguity':
                resolution = self.resolve_action_ambiguity(command, context)
                if resolution['clarification_needed']:
                    clarification_needed = True
                else:
                    resolved_command = resolution['command']

        return {
            'command': resolved_command,
            'resolved': True,
            'clarification_needed': clarification_needed,
            'original_ambiguities': ambiguity_info['types']
        }

    def resolve_object_ambiguity(self, command: str, context: Dict) -> Dict:
        """Resolve object reference ambiguity"""
        detected_objects = context.get('objects', [])

        if len(detected_objects) == 0:
            # No objects detected, need clarification
            return {
                'command': command,
                'clarification_needed': True,
                'request': 'I cannot see any objects. Could you please point to or describe the object you mean?'
            }

        elif len(detected_objects) == 1:
            # Only one object, assume this is the intended one
            target_object = detected_objects[0]
            resolved_command = self.substitute_object_reference(command, target_object)
            return {
                'command': resolved_command,
                'clarification_needed': False
            }

        else:
            # Multiple objects, need to disambiguate
            # Look for additional context clues in the command
            specific_clues = self.find_specific_clues(command, detected_objects)

            if specific_clues:
                # Found specific clues, use them to identify object
                target_object = specific_clues[0]  # Use first match
                resolved_command = self.substitute_object_reference(command, target_object)
                return {
                    'command': resolved_command,
                    'clarification_needed': False
                }
            else:
                # Cannot disambiguate automatically, ask for clarification
                object_names = [obj.get('type', 'object') for obj in detected_objects[:3]]
                object_list = ', '.join(object_names)
                return {
                    'command': command,
                    'clarification_needed': True,
                    'request': f'I see multiple objects: {object_list}. Could you specify which one you mean?'
                }

    def find_specific_clues(self, command: str, objects: List[Dict]) -> List[Dict]:
        """Find objects that match specific clues in the command"""
        command_lower = command.lower()
        matching_objects = []

        for obj in objects:
            obj_type = obj.get('type', '').lower()
            obj_color = obj.get('color', '').lower()
            obj_size = obj.get('size', '').lower()
            obj_location = obj.get('location', '').lower()

            # Check if object properties match command clues
            if (obj_type in command_lower or
                obj_color in command_lower or
                obj_size in command_lower or
                obj_location in command_lower):
                matching_objects.append(obj)

        return matching_objects

    def substitute_object_reference(self, command: str, target_object: Dict) -> str:
        """Substitute ambiguous object reference with specific object"""
        # This is a simplified substitution
        # In practice, use more sophisticated NLP
        obj_type = target_object.get('type', 'object')

        # Replace ambiguous references with specific object
        command = re.sub(r'\bthe \w+\b', f"the {obj_type}", command)
        command = re.sub(r'\bit\b|\bthat\b|\bthis\b', f"the {obj_type}", command)

        return command

    def resolve_spatial_ambiguity(self, command: str, context: Dict) -> Dict:
        """Resolve spatial reference ambiguity"""
        # Check for vague spatial references
        if 'over there' in command.lower() or 'somewhere' in command.lower():
            # Use pointing gesture or visual attention if available
            pointing_direction = context.get('pointing_direction')
            if pointing_direction:
                # Convert pointing to specific location
                pointed_location = self.direction_to_location(pointing_direction, context)
                resolved_command = self.substitute_spatial_reference(command, pointed_location)
                return {
                    'command': resolved_command,
                    'clarification_needed': False
                }
            else:
                # Ask for clarification
                return {
                    'command': command,
                    'clarification_needed': True,
                    'request': 'Could you please point to or be more specific about where you mean?'
                }

        return {
            'command': command,
            'clarification_needed': False
        }

    def direction_to_location(self, direction: List[float], context: Dict) -> str:
        """Convert pointing direction to location name"""
        # This would use spatial reasoning to identify location
        # For now, return a generic location
        return "the area you're pointing to"

    def substitute_spatial_reference(self, command: str, location: str) -> str:
        """Substitute spatial reference with specific location"""
        command = re.sub(r'over there|somewhere|around here', location, command, flags=re.IGNORECASE)
        return command

    def resolve_action_ambiguity(self, command: str, context: Dict) -> Dict:
        """Resolve action ambiguity"""
        # For vague action requests, ask for clarification
        return {
            'command': command,
            'clarification_needed': True,
            'request': f"I'm not sure what you mean by '{command}'. Could you please be more specific about what you'd like me to do?"
        }

class FallbackManager:
    def __init__(self):
        self.fallback_strategies = [
            'ask_for_clarification',
            'request_repetition',
            'suggest_alternative',
            'execute_safe_default',
            'handoff_to_human'
        ]

    def handle_unmappable_command(self, command: str, context: Dict) -> Dict:
        """Handle commands that cannot be mapped to actions"""
        # Determine appropriate fallback strategy
        strategy = self.select_fallback_strategy(command, context)

        if strategy == 'ask_for_clarification':
            return self.ask_for_clarification(command, context)
        elif strategy == 'request_repetition':
            return self.request_repetition(command)
        elif strategy == 'suggest_alternative':
            return self.suggest_alternative_actions(command, context)
        elif strategy == 'execute_safe_default':
            return self.execute_safe_default_action(command, context)
        elif strategy == 'handoff_to_human':
            return self.handoff_to_human(command, context)

    def select_fallback_strategy(self, command: str, context: Dict) -> str:
        """Select appropriate fallback strategy"""
        command_lower = command.lower()

        # Check for specific conditions that suggest certain strategies
        if any(word in command_lower for word in ['help', 'please', 'can you']):
            return 'ask_for_clarification'
        elif any(word in command_lower for word in ['stop', 'wait', 'pause']):
            return 'execute_safe_default'
        elif len(command.split()) < 3:
            return 'request_repetition'
        else:
            return 'ask_for_clarification'

    def ask_for_clarification(self, command: str, context: Dict) -> Dict:
        """Ask user for clarification"""
        return {
            'action_type': 'request_clarification',
            'message': f"I didn't understand '{command}'. Could you please rephrase or be more specific?",
            'options': self.generate_possible_interpretations(command, context),
            'strategy': 'ask_for_clarification'
        }

    def generate_possible_interpretations(self, command: str, context: Dict) -> List[str]:
        """Generate possible interpretations of ambiguous command"""
        # This would use NLP to generate possible interpretations
        # For now, return some generic options
        return [
            f"Did you mean: go somewhere?",
            f"Did you mean: find something?",
            f"Did you mean: do a specific action?"
        ]

    def request_repetition(self, command: str) -> Dict:
        """Request user to repeat command"""
        return {
            'action_type': 'request_repetition',
            'message': f"I couldn't process '{command}'. Could you please repeat that?",
            'strategy': 'request_repetition'
        }

    def suggest_alternative_actions(self, command: str, context: Dict) -> Dict:
        """Suggest alternative actions based on partial understanding"""
        available_actions = context.get('available_actions', [
            'navigate to location',
            'grasp object',
            'place object',
            'find object',
            'stop movement'
        ])

        return {
            'action_type': 'suggest_alternative',
            'message': f"I'm not sure how to '{command}'. Here are things I can do: {', '.join(available_actions[:3])}",
            'suggestions': available_actions,
            'strategy': 'suggest_alternative'
        }

    def execute_safe_default_action(self, command: str, context: Dict) -> Dict:
        """Execute a safe default action"""
        return {
            'action_type': 'safe_default',
            'executed_action': 'stop_movement',
            'message': f"Pausing as I'm not sure about '{command}'",
            'strategy': 'execute_safe_default'
        }

    def handoff_to_human(self, command: str, context: Dict) -> Dict:
        """Hand off to human operator"""
        return {
            'action_type': 'handoff',
            'message': f"I cannot perform '{command}'. Connecting you with a human operator.",
            'strategy': 'handoff_to_human'
        }
```

## Isaac Integration for Language-to-Action Mapping

### ROS 2 Interface for Action Mapping

Integrating language-to-action mapping with ROS 2 and Isaac systems:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class IsaacLanguageActionMapperNode(Node):
    def __init__(self):
        super().__init__('isaac_language_action_mapper')

        # Publishers
        self.action_plan_publisher = self.create_publisher(String, 'action_plan', 10)
        self.status_publisher = self.create_publisher(String, 'action_mapping_status', 10)

        # Subscribers
        self.language_command_subscriber = self.create_subscription(
            String, 'natural_language_command', self.language_command_callback, 10
        )

        # Action server for complex command mapping
        self.mapping_server = ActionServer(
            self,
            MapLanguageToAction,
            'map_language_to_action',
            self.map_language_to_action_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Initialize components
        self.hybrid_parser = HybridSemanticParser()
        self.context_aware_mapper = ContextAwareActionMapper()
        self.action_planner = ActionExecutionPlanner()
        self.ambiguity_resolver = AmbiguityResolver()
        self.fallback_manager = FallbackManager()

        self.get_logger().info('Isaac Language-to-Action Mapper Node initialized')

    def language_command_callback(self, msg):
        """Handle incoming natural language commands"""
        try:
            command = msg.data
            self.get_logger().info(f'Received command: {command}')

            # Get current context
            context = self.get_current_context()

            # Detect and resolve ambiguities
            ambiguity_info = self.ambiguity_resolver.detect_ambiguity(command, context)

            if ambiguity_info['detected']:
                resolution = self.ambiguity_resolver.resolve_ambiguity(
                    command, ambiguity_info, context
                )

                if resolution['clarification_needed']:
                    # Publish clarification request
                    clarification_msg = String()
                    clarification_msg.data = resolution['request']
                    self.status_publisher.publish(clarification_msg)
                    return

                command = resolution['command']

            # Parse command
            parsed_command = self.hybrid_parser.parse_command(command, context)

            # Map to contextual action
            action_mapping = self.context_aware_mapper.map_with_context(
                parsed_command, context
            )

            # Check feasibility
            if not action_mapping['feasibility']:
                fallback_result = self.fallback_manager.handle_unmappable_command(
                    command, context
                )
                self.handle_fallback_result(fallback_result)
                return

            # Plan execution
            execution_plan = self.action_planner.plan_execution(action_mapping)

            # Publish action plan
            plan_msg = String()
            plan_msg.data = str(execution_plan)
            self.action_plan_publisher.publish(plan_msg)

            self.get_logger().info(f'Mapped command to action plan: {execution_plan}')

        except Exception as e:
            self.get_logger().error(f'Error mapping command: {e}')
            # Handle with fallback
            fallback_result = self.fallback_manager.handle_unmappable_command(
                msg.data, self.get_current_context()
            )
            self.handle_fallback_result(fallback_result)

    def goal_callback(self, goal_request):
        """Handle mapping goal"""
        self.get_logger().info(f'Received mapping goal: {goal_request.command}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle mapping cancellation"""
        self.get_logger().info('Mapping cancelled')
        return CancelResponse.ACCEPT

    def map_language_to_action_callback(self, goal_handle):
        """Map language to action as action server"""
        feedback_msg = MapLanguageToAction.Feedback()
        result_msg = MapLanguageToAction.Result()

        try:
            command = goal_handle.request.command
            self.get_logger().info(f'Mapping command: {command}')

            # Get context
            context = self.get_current_context()

            # Update feedback
            feedback_msg.status = 'Parsing command'
            goal_handle.publish_feedback(feedback_msg)

            # Parse command
            parsed_command = self.hybrid_parser.parse_command(command, context)

            # Update feedback
            feedback_msg.status = 'Resolving ambiguities'
            goal_handle.publish_feedback(feedback_msg)

            # Detect and resolve ambiguities
            ambiguity_info = self.ambiguity_resolver.detect_ambiguity(command, context)
            if ambiguity_info['detected']:
                resolution = self.ambiguity_resolver.resolve_ambiguity(
                    command, ambiguity_info, context
                )

                if resolution['clarification_needed']:
                    result_msg.success = False
                    result_msg.message = resolution['request']
                    result_msg.requires_clarification = True
                    goal_handle.succeed()
                    return result_msg

                # Use resolved command
                resolved_command = resolution['command']
                parsed_command = self.hybrid_parser.parse_command(resolved_command, context)

            # Update feedback
            feedback_msg.status = 'Mapping to action'
            goal_handle.publish_feedback(feedback_msg)

            # Map to action
            action_mapping = self.context_aware_mapper.map_with_context(
                parsed_command, context
            )

            # Check feasibility
            if not action_mapping['feasibility']:
                fallback_result = self.fallback_manager.handle_unmappable_command(
                    command, context
                )
                result_msg.success = False
                result_msg.message = fallback_result['message']
                goal_handle.abort()
                return result_msg

            # Plan execution
            execution_plan = self.action_planner.plan_execution(action_mapping)

            # Update feedback
            feedback_msg.status = 'Planning complete'
            goal_handle.publish_feedback(feedback_msg)

            # Set result
            result_msg.success = True
            result_msg.message = f'Successfully mapped command to action plan'
            result_msg.action_plan = str(execution_plan)
            result_msg.action_mapping = str(action_mapping)

            goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f'Mapping failed: {e}')
            result_msg.success = False
            result_msg.message = f'Mapping failed: {str(e)}'
            goal_handle.abort()

        return result_msg

    def get_current_context(self):
        """Get current system context"""
        # This would integrate with perception, mapping, and other systems
        return {
            'objects': self.get_detected_objects(),
            'robot_pose': self.get_robot_pose(),
            'named_locations': self.get_known_locations(),
            'robot_capabilities': self.get_robot_capabilities(),
            'available_actions': self.get_available_actions(),
            'recent_objects': self.get_recent_objects()
        }

    def get_detected_objects(self):
        """Get currently detected objects"""
        # Implementation would interface with perception system
        return []

    def get_robot_pose(self):
        """Get current robot pose"""
        # Implementation would interface with localization system
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def get_known_locations(self):
        """Get known named locations"""
        # Implementation would interface with mapping system
        return {
            'kitchen': [5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            'living_room': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        }

    def get_robot_capabilities(self):
        """Get robot capabilities"""
        return ['navigation', 'manipulation', 'perception']

    def get_available_actions(self):
        """Get available robot actions"""
        return ['navigate_to', 'grasp_object', 'place_object', 'find_object']

    def get_recent_objects(self):
        """Get recently referenced objects"""
        return []

    def handle_fallback_result(self, fallback_result):
        """Handle fallback result"""
        if 'message' in fallback_result:
            status_msg = String()
            status_msg.data = fallback_result['message']
            self.status_publisher.publish(status_msg)

class MapLanguageToAction:
    def __init__(self):
        self.command = ""

    class Feedback:
        def __init__(self):
            self.status = ""

    class Result:
        def __init__(self):
            self.success = False
            self.message = ""
            self.action_plan = ""
            self.action_mapping = ""
            self.requires_clarification = False

# Action execution orchestrator
class ActionOrchestrator:
    def __init__(self, node):
        self.node = node
        self.action_queue = []
        self.is_executing = False

    def queue_action_plan(self, action_plan: List[Dict], priority: int = 0):
        """Queue action plan for execution"""
        plan_item = {
            'plan': action_plan,
            'priority': priority,
            'timestamp': time.time(),
            'status': 'queued'
        }

        # Insert based on priority
        self.action_queue.append(plan_item)
        self.action_queue.sort(key=lambda x: (-x['priority'], x['timestamp']))

        # Start execution if not already running
        if not self.is_executing:
            self.execute_next_plan()

    def execute_next_plan(self):
        """Execute the next action plan in the queue"""
        if self.action_queue:
            self.is_executing = True
            next_plan = self.action_queue.pop(0)
            next_plan['status'] = 'executing'

            # Execute plan asynchronously
            self.execute_action_plan(next_plan['plan'])

    def execute_action_plan(self, plan: List[Dict]):
        """Execute action plan"""
        for step in plan:
            try:
                result = self.execute_step(step)
                if not result['success']:
                    self.node.get_logger().error(f'Step failed: {step}')
                    break
            except Exception as e:
                self.node.get_logger().error(f'Step execution error: {e}')
                break

        self.is_executing = False
        if self.action_queue:
            self.execute_next_plan()

    def execute_step(self, step: Dict):
        """Execute individual step"""
        # This would interface with appropriate execution systems
        # For now, simulate execution
        import time
        time.sleep(0.1)  # Simulate execution time

        return {'success': True, 'result': 'completed'}
```

## Evaluation and Validation

### Language-to-Action Mapping Evaluation

Evaluating language-to-action mapping systems requires comprehensive benchmarks:

```python
class LanguageActionMappingEvaluator:
    def __init__(self, mapper_system):
        self.mapper_system = mapper_system
        self.results_log = []

    def evaluate_mapping_accuracy(self, test_cases):
        """Evaluate accuracy of language-to-action mapping"""
        correct_mappings = 0
        total_cases = len(test_cases)

        for test_case in test_cases:
            command = test_case['command']
            expected_action = test_case['expected_action']
            context = test_case.get('context', {})

            try:
                # Map command to action
                parsed = self.mapper_system.hybrid_parser.parse_command(command, context)
                mapped = self.mapper_system.context_aware_mapper.map_with_context(parsed, context)

                # Check if mapping is correct
                if self.actions_match(mapped, expected_action):
                    correct_mappings += 1
            except Exception as e:
                print(f"Mapping error for '{command}': {e}")

        accuracy = correct_mappings / total_cases if total_cases > 0 else 0.0
        return accuracy

    def evaluate_ambiguity_resolution(self, ambiguous_cases):
        """Evaluate ambiguity resolution effectiveness"""
        correctly_resolved = 0
        total_ambiguous = len(ambiguous_cases)

        for case in ambiguous_cases:
            command = case['command']
            expected_resolution = case['expected_resolution']
            context = case.get('context', {})

            try:
                ambiguity_info = self.mapper_system.ambiguity_resolver.detect_ambiguity(
                    command, context
                )

                if ambiguity_info['detected']:
                    resolution = self.mapper_system.ambiguity_resolver.resolve_ambiguity(
                        command, ambiguity_info, context
                    )

                    if self.resolutions_match(resolution, expected_resolution):
                        correctly_resolved += 1
            except Exception:
                pass

        resolution_rate = correctly_resolved / total_ambiguous if total_ambiguous > 0 else 0.0
        return resolution_rate

    def evaluate_fallback_effectiveness(self, unmappable_cases):
        """Evaluate effectiveness of fallback strategies"""
        effective_fallbacks = 0
        total_unmappable = len(unmappable_cases)

        for case in unmappable_cases:
            command = case['command']
            context = case.get('context', {})

            try:
                fallback_result = self.mapper_system.fallback_manager.handle_unmappable_command(
                    command, context
                )

                # Check if fallback was appropriate and helpful
                if self.is_appropriate_fallback(fallback_result, case):
                    effective_fallbacks += 1
            except Exception:
                pass

        effectiveness = effective_fallbacks / total_unmappable if total_unmappable > 0 else 0.0
        return effectiveness

    def evaluate_execution_success(self, mapped_actions):
        """Evaluate success of executing mapped actions"""
        successful_executions = 0
        total_actions = len(mapped_actions)

        for action in mapped_actions:
            try:
                # Plan execution
                execution_plan = self.mapper_system.action_planner.plan_execution(action)

                # Simulate execution (in real system, would execute on robot)
                execution_result = self.simulate_execution(execution_plan)

                if execution_result['success']:
                    successful_executions += 1
            except Exception:
                pass

        success_rate = successful_executions / total_actions if total_actions > 0 else 0.0
        return success_rate

    def run_comprehensive_evaluation(self, dataset):
        """Run comprehensive evaluation of mapping system"""
        results = {}

        # Mapping accuracy
        mapping_tests = [case for case in dataset if 'mapping' in case.get('type', '')]
        results['mapping_accuracy'] = self.evaluate_mapping_accuracy(mapping_tests)

        # Ambiguity resolution
        ambiguity_tests = [case for case in dataset if 'ambiguity' in case.get('type', '')]
        results['ambiguity_resolution_rate'] = self.evaluate_ambiguity_resolution(ambiguity_tests)

        # Fallback effectiveness
        fallback_tests = [case for case in dataset if 'fallback' in case.get('type', '')]
        results['fallback_effectiveness'] = self.evaluate_fallback_effectiveness(fallback_tests)

        # Execution success
        execution_tests = [case for case in dataset if 'execution' in case.get('type', '')]
        results['execution_success_rate'] = self.evaluate_execution_success(execution_tests)

        # Overall system response time
        results['average_response_time'] = self.measure_response_time(dataset)

        return results

    def actions_match(self, mapped_action, expected_action):
        """Check if mapped action matches expected action"""
        # Compare key aspects of the action
        if mapped_action['action_type'] != expected_action.get('action_type'):
            return False

        # Check important parameters
        expected_params = expected_action.get('parameters', {})
        mapped_params = mapped_action.get('contextual_parameters', {})

        for param, expected_value in expected_params.items():
            if param not in mapped_params:
                return False
            if mapped_params[param] != expected_value:
                return False

        return True

    def resolutions_match(self, resolution, expected_resolution):
        """Check if ambiguity resolution matches expected resolution"""
        # Compare resolution results
        return resolution.get('command') == expected_resolution.get('command')

    def is_appropriate_fallback(self, fallback_result, test_case):
        """Check if fallback was appropriate for the case"""
        # This would evaluate if the chosen fallback strategy was appropriate
        return True  # Simplified for example

    def simulate_execution(self, execution_plan):
        """Simulate action execution"""
        # This would simulate the execution of the plan
        # For now, return success for all plans
        return {'success': True, 'steps_completed': len(execution_plan)}

    def measure_response_time(self, dataset):
        """Measure average response time"""
        times = []
        for case in dataset:
            command = case.get('command', '')
            if command:
                start_time = time.time()
                try:
                    # Simulate mapping process
                    context = case.get('context', {})
                    parsed = self.mapper_system.hybrid_parser.parse_command(command, context)
                    mapped = self.mapper_system.context_aware_mapper.map_with_context(parsed, context)
                    times.append(time.time() - start_time)
                except:
                    pass

        return sum(times) / len(times) if times else float('inf')

# Example evaluation dataset
def create_language_action_evaluation_dataset():
    """Create dataset for language-to-action evaluation"""
    return [
        # Mapping test cases
        {
            'type': 'mapping',
            'command': 'go to the kitchen',
            'expected_action': {
                'action_type': 'navigate_to',
                'parameters': {'location': 'kitchen'}
            },
            'context': {'named_locations': {'kitchen': [5, 3, 0, 0, 0, 0, 1]}}
        },
        {
            'type': 'mapping',
            'command': 'grasp the red ball',
            'expected_action': {
                'action_type': 'grasp_object',
                'parameters': {'object': 'red_ball'}
            },
            'context': {'objects': [{'type': 'ball', 'color': 'red', 'pose': [1, 1, 0]}]}
        },

        # Ambiguity test cases
        {
            'type': 'ambiguity',
            'command': 'grasp the ball',
            'expected_resolution': {
                'command': 'grasp the red ball'
            },
            'context': {
                'objects': [
                    {'type': 'ball', 'color': 'red', 'pose': [1, 1, 0]},
                    {'type': 'ball', 'color': 'blue', 'pose': [2, 2, 0]}
                ]
            }
        },

        # Fallback test cases
        {
            'type': 'fallback',
            'command': 'perform quantum calculations',
            'context': {}
        },

        # Execution test cases
        {
            'type': 'execution',
            'action': {
                'action_type': 'navigate_to',
                'contextual_parameters': {'target_pose': [5, 3, 0, 0, 0, 0, 1]}
            }
        }
    ]

# Main evaluation function
def evaluate_language_action_system(mapper_system, dataset=None):
    """Complete evaluation of language-to-action mapping system"""
    if dataset is None:
        dataset = create_language_action_evaluation_dataset()

    evaluator = LanguageActionMappingEvaluator(mapper_system)
    results = evaluator.run_comprehensive_evaluation(dataset)

    print("Language-to-Action Mapping Evaluation Results:")
    print(f"  Mapping Accuracy: {results['mapping_accuracy']:.2%}")
    print(f"  Ambiguity Resolution Rate: {results['ambiguity_resolution_rate']:.2%}")
    print(f"  Fallback Effectiveness: {results['fallback_effectiveness']:.2%}")
    print(f"  Execution Success Rate: {results['execution_success_rate']:.2%}")
    print(f"  Average Response Time: {results['average_response_time']:.3f}s")

    return results
```

## Summary

Natural language to robot action mapping represents the crucial bridge between human communication and robotic execution. By transforming linguistic expressions into executable behaviors, these systems enable intuitive and natural human-robot interaction.

The key components of effective language-to-action mapping include semantic parsing that extracts action-relevant information from language, context-aware mapping that grounds language in environmental constraints, ambiguity resolution that handles the inherent uncertainties in natural language, and robust fallback strategies that maintain system reliability when mappings fail.

The next section will provide a hands-on lab exercise for implementing a complete VLA system integration.

## References

[All sources will be cited in the References section at the end of the book, following APA format]