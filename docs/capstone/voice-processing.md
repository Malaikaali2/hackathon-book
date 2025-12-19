---
sidebar_position: 16
---

# Voice Command Processing Implementation: Autonomous Humanoid Capstone

## Overview

Voice command processing forms the primary interaction interface for the autonomous humanoid system, enabling natural language communication between users and the robot. This component is responsible for converting spoken commands into structured actions that the robot can understand and execute. The implementation integrates speech recognition, natural language processing, and intent classification to create an intuitive human-robot interaction system.

The voice processing pipeline must handle various command types, from simple navigation requests ("Go to the kitchen") to complex manipulation tasks ("Pick up the red cup and place it on the table"). This implementation guide provides detailed instructions for building a robust voice command processing system that meets the performance and accuracy requirements of the capstone project.

## System Architecture

### Voice Processing Pipeline

The voice command processing system follows a multi-stage pipeline architecture:

```
Audio Input → Speech Recognition → Natural Language Processing → Intent Classification → Action Planning → Command Execution
```

Each stage performs specific processing tasks:
1. **Audio Processing**: Capture and preprocess audio input for recognition
2. **Speech Recognition**: Convert speech to text using ASR (Automatic Speech Recognition)
3. **Natural Language Understanding**: Parse text for intent and parameters
4. **Context Processing**: Apply contextual knowledge to disambiguate commands
5. **Action Mapping**: Map understood commands to robot action primitives

### Component Integration

The voice processing system integrates with other capstone components:
- **ROS 2 Interface**: Publishes commands to action servers and navigation system
- **Perception System**: Requests object detection and localization for manipulation tasks
- **Task Planning**: Provides high-level goals to the planning system
- **Navigation System**: Issues navigation goals for location-based commands
- **Manipulation System**: Sends manipulation parameters for object interaction

## Technical Implementation

### 1. Speech Recognition Integration

#### Speech-to-Text Setup

The speech recognition component uses either cloud-based services (Google Speech-to-Text, Azure Cognitive Services) or on-premise solutions (Vosk, SpeechRecognition with CMU Sphinx):

```python
import speech_recognition as sr
import rospy
from std_msgs.msg import String

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_publisher = rospy.Publisher('/voice_commands', String, queue_size=10)

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def listen_for_command(self):
        """Listen for voice command and return recognized text"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5.0)

            # Use Google Speech Recognition (requires internet)
            command_text = self.recognizer.recognize_google(audio)
            return command_text
        except sr.WaitTimeoutError:
            rospy.loginfo("No speech detected within timeout")
            return None
        except sr.UnknownValueError:
            rospy.loginfo("Could not understand audio")
            return None
        except sr.RequestError as e:
            rospy.logerr(f"Speech recognition error: {e}")
            return None
```

#### Performance Optimization

For real-time performance, implement continuous listening with wake word detection:

```python
class WakeWordDetector:
    def __init__(self, wake_word="robot"):
        self.wake_word = wake_word.lower()
        self.is_listening = False

    def detect_wake_word(self, audio_text):
        """Detect wake word to activate full processing"""
        if self.wake_word in audio_text.lower():
            return True
        return False

class ContinuousVoiceProcessor:
    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.wake_detector = WakeWordDetector()
        self.nlp_processor = NaturalLanguageProcessor()

    def continuous_processing_loop(self):
        """Main processing loop for voice commands"""
        while not rospy.is_shutdown():
            # Listen for wake word
            if not self.wake_detector.is_listening:
                audio_text = self.speech_recognizer.listen_for_command()
                if audio_text and self.wake_detector.detect_wake_word(audio_text):
                    self.wake_detector.is_listening = True
                    rospy.loginfo("Wake word detected - ready for command")
            else:
                # Process full command
                command_text = self.speech_recognizer.listen_for_command()
                if command_text:
                    self.process_command(command_text)
                    self.wake_detector.is_listening = False
```

### 2. Natural Language Processing

#### Command Parsing

The NLP component parses recognized text to extract intent and parameters:

```python
import spacy
import re
from typing import Dict, List, Tuple

class NaturalLanguageProcessor:
    def __init__(self):
        # Load spaCy model for English
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            rospy.logerr("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            raise

    def parse_command(self, text: str) -> Dict:
        """Parse command text and extract structured information"""
        doc = self.nlp(text)

        # Extract intent based on action verbs
        intent = self.extract_intent(doc)

        # Extract objects and locations
        entities = self.extract_entities(doc)

        # Extract quantities and modifiers
        modifiers = self.extract_modifiers(doc)

        return {
            'intent': intent,
            'entities': entities,
            'modifiers': modifiers,
            'original_text': text
        }

    def extract_intent(self, doc) -> str:
        """Extract the primary intent from the command"""
        # Define common action patterns
        navigation_patterns = ['go', 'move', 'navigate', 'walk', 'drive', 'travel', 'go to', 'move to']
        manipulation_patterns = ['pick', 'grab', 'take', 'lift', 'place', 'put', 'set', 'move', 'bring', 'get']
        query_patterns = ['where', 'what', 'find', 'locate', 'show', 'tell']

        # Check for action verbs
        for token in doc:
            if token.lemma_ in navigation_patterns:
                return 'NAVIGATE'
            elif token.lemma_ in manipulation_patterns:
                return 'MANIPULATE'
            elif token.lemma_ in query_patterns:
                return 'QUERY'

        # Default to navigation if no clear intent
        return 'NAVIGATE'

    def extract_entities(self, doc) -> Dict:
        """Extract named entities (objects, locations, people)"""
        entities = {
            'objects': [],
            'locations': [],
            'people': []
        }

        # Look for objects based on dependencies and POS tags
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ in ['dobj', 'pobj', 'attr']:
                # Check if it's a location
                location_keywords = ['kitchen', 'living room', 'bedroom', 'office', 'table', 'couch', 'door', 'window']
                if any(keyword in token.text.lower() for keyword in location_keywords):
                    entities['locations'].append(token.text)
                else:
                    entities['objects'].append(token.text)

        # Extract colors and other descriptors
        for token in doc:
            if token.pos_ == 'ADJ':
                # Look for color adjectives near objects
                for child in token.children:
                    if child.pos_ in ['NOUN', 'PROPN']:
                        # This adjective likely describes the noun
                        entities['objects'].append(f"{token.text} {child.text}")

        return entities
```

#### Context Management

Implement context awareness to handle pronouns and references:

```python
class ContextManager:
    def __init__(self):
        self.current_context = {}
        self.conversation_history = []

    def update_context(self, command_result):
        """Update context based on command execution results"""
        self.current_context.update({
            'last_action': command_result.get('action'),
            'last_location': command_result.get('location'),
            'last_object': command_result.get('object'),
            'timestamp': rospy.get_time()
        })

    def resolve_pronouns(self, text: str) -> str:
        """Resolve pronouns like 'it', 'there', 'here' based on context"""
        resolved_text = text

        # Replace 'it' with last mentioned object
        if 'it' in text.lower() and self.current_context.get('last_object'):
            resolved_text = resolved_text.replace('it', self.current_context['last_object'])

        # Replace 'there' with last location
        if 'there' in text.lower() and self.current_context.get('last_location'):
            resolved_text = resolved_text.replace('there', self.current_context['last_location'])

        return resolved_text
```

### 3. Intent Classification and Action Mapping

#### Intent Classification System

```python
class IntentClassifier:
    def __init__(self):
        # Define command patterns with regex
        self.patterns = {
            'NAVIGATE': [
                r'go to (the )?(?P<location>\w+)',
                r'move to (the )?(?P<location>\w+)',
                r'go to the (?P<location>\w+)',
                r'go to (?P<location>\w+)',
                r'navigate to (the )?(?P<location>\w+)'
            ],
            'MANIPULATE_PICK': [
                r'pick up (the )?(?P<object>\w+)',
                r'grab (the )?(?P<object>\w+)',
                r'take (the )?(?P<object>\w+)',
                r'get (the )?(?P<object>\w+)'
            ],
            'MANIPULATE_PLACE': [
                r'place (the )?(?P<object>\w+) (on|at) (the )?(?P<location>\w+)',
                r'put (the )?(?P<object>\w+) (on|at) (the )?(?P<location>\w+)',
                r'set (the )?(?P<object>\w+) (on|at) (the )?(?P<location>\w+)'
            ],
            'QUERY': [
                r'where is (the )?(?P<object>\w+)',
                r'find (the )?(?P<object>\w+)',
                r'locate (the )?(?P<object>\w+)'
            ]
        }

    def classify_intent(self, parsed_command: Dict) -> Dict:
        """Classify intent and extract parameters"""
        original_text = parsed_command['original_text'].lower()

        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, original_text)
                if match:
                    return {
                        'intent': intent,
                        'parameters': match.groupdict(),
                        'confidence': 0.9  # High confidence for pattern matches
                    }

        # Fallback to NLP-based classification
        return self.fallback_classification(parsed_command)

    def fallback_classification(self, parsed_command: Dict) -> Dict:
        """Fallback classification using NLP analysis"""
        intent = parsed_command['intent']
        entities = parsed_command['entities']

        # Determine specific manipulation intent
        if intent == 'MANIPULATE':
            # Look for placement verbs vs pickup verbs
            original_text = parsed_command['original_text'].lower()
            if any(verb in original_text for verb in ['place', 'put', 'set', 'on', 'at']):
                intent = 'MANIPULATE_PLACE'
            else:
                intent = 'MANIPULATE_PICK'

        # Extract parameters from entities
        parameters = {}
        if entities['objects']:
            parameters['object'] = entities['objects'][0]
        if entities['locations']:
            parameters['location'] = entities['locations'][0]

        return {
            'intent': intent,
            'parameters': parameters,
            'confidence': 0.7  # Lower confidence for fallback
        }
```

#### Action Mapping System

```python
class ActionMapper:
    def __init__(self):
        self.navigation_publisher = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=10)
        self.manipulation_publisher = rospy.Publisher('/manipulation_commands', String, queue_size=10)
        self.query_publisher = rospy.Publisher('/query_commands', String, queue_size=10)

    def map_to_action(self, classified_intent: Dict):
        """Map classified intent to robot action"""
        intent = classified_intent['intent']
        parameters = classified_intent['parameters']

        if intent == 'NAVIGATE':
            self.execute_navigation(parameters)
        elif intent in ['MANIPULATE_PICK', 'MANIPULATE_PLACE']:
            self.execute_manipulation(intent, parameters)
        elif intent == 'QUERY':
            self.execute_query(parameters)

    def execute_navigation(self, parameters: Dict):
        """Execute navigation command"""
        location = parameters.get('location')
        if not location:
            rospy.logerr("No location specified for navigation command")
            return

        # Convert location name to coordinates (from map)
        location_coords = self.get_coordinates_for_location(location)

        if location_coords:
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose = location_coords

            # Publish navigation goal
            self.navigation_publisher.publish(goal)
        else:
            rospy.logerr(f"Unknown location: {location}")

    def execute_manipulation(self, intent: str, parameters: Dict):
        """Execute manipulation command"""
        obj = parameters.get('object')
        location = parameters.get('location')

        if not obj:
            rospy.logerr("No object specified for manipulation command")
            return

        # Create manipulation command
        command = {
            'action': 'pick' if 'PICK' in intent else 'place',
            'object': obj,
            'target_location': location if location else None
        }

        # Publish manipulation command
        self.manipulation_publisher.publish(str(command))

    def get_coordinates_for_location(self, location_name: str) -> Pose:
        """Convert location name to map coordinates"""
        # This would typically come from a saved map of location coordinates
        location_map = {
            'kitchen': Pose(position=Point(2.0, 1.0, 0.0), orientation=Quaternion(0, 0, 0, 1)),
            'living room': Pose(position=Point(-1.0, 2.0, 0.0), orientation=Quaternion(0, 0, 0, 1)),
            'bedroom': Pose(position=Point(3.0, -2.0, 0.0), orientation=Quaternion(0, 0, 0, 1)),
            'office': Pose(position=Point(-2.0, -1.0, 0.0), orientation=Quaternion(0, 0, 0, 1))
        }

        return location_map.get(location_name.lower())
```

## Implementation Steps

### Step 1: Set up Speech Recognition Environment

1. Install required dependencies:
```bash
pip install speechrecognition pyaudio spacy
python -m spacy download en_core_web_sm
```

2. Test microphone access:
```python
import speech_recognition as sr
r = sr.Recognizer()
mic = sr.Microphone()

print("Available microphones:")
for device_index in range(mic.get_device_count()):
    device_info = mic.get_device_info(device_index)
    print(f"Device {device_index}: {device_info['name']}")
```

### Step 2: Implement Core Processing Classes

Create the main voice processing node that integrates all components:

```python
#!/usr/bin/env python3

import rospy
import speech_recognition as sr
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from move_base_msgs.msg import MoveBaseGoal, MoveBaseActionGoal

class VoiceCommandProcessor:
    def __init__(self):
        rospy.init_node('voice_command_processor')

        # Initialize components
        self.speech_recognizer = SpeechRecognizer()
        self.nlp_processor = NaturalLanguageProcessor()
        self.intent_classifier = IntentClassifier()
        self.action_mapper = ActionMapper()
        self.context_manager = ContextManager()

        # Publishers and subscribers
        self.status_publisher = rospy.Publisher('/voice_status', String, queue_size=10)
        self.command_subscriber = rospy.Subscriber('/voice_commands', String, self.command_callback)

        rospy.loginfo("Voice command processor initialized")

    def command_callback(self, msg):
        """Process incoming voice command"""
        try:
            # Parse the command
            parsed = self.nlp_processor.parse_command(msg.data)

            # Classify intent
            classified = self.intent_classifier.classify_intent(parsed)

            # Check confidence threshold
            if classified['confidence'] < 0.5:
                rospy.logwarn(f"Low confidence command: {msg.data} (confidence: {classified['confidence']})")
                return

            # Map to action
            self.action_mapper.map_to_action(classified)

            # Update context
            self.context_manager.update_context({
                'command': msg.data,
                'intent': classified['intent'],
                'parameters': classified['parameters']
            })

            # Publish status
            status_msg = String()
            status_msg.data = f"Executed command: {msg.data}"
            self.status_publisher.publish(status_msg)

        except Exception as e:
            rospy.logerr(f"Error processing command: {e}")

    def start_listening(self):
        """Start continuous voice command processing"""
        rospy.loginfo("Starting voice command processing...")
        rospy.spin()

if __name__ == '__main__':
    processor = VoiceCommandProcessor()
    processor.start_listening()
```

### Step 3: Performance Optimization

1. Implement audio preprocessing for noise reduction:
```python
import numpy as np
from scipy import signal

class AudioPreprocessor:
    def __init__(self):
        # Design a simple low-pass filter to reduce background noise
        self.b, self.a = signal.butter(4, 0.2, 'low')

    def preprocess_audio(self, audio_data):
        """Apply noise reduction to audio data"""
        filtered = signal.filtfilt(self.b, self.a, audio_data)
        return filtered
```

2. Add caching for frequently recognized phrases:
```python
from functools import lru_cache

class CachedNLPProcessor(NaturalLanguageProcessor):
    @lru_cache(maxsize=100)
    def parse_command_cached(self, text: str):
        """Cached version of command parsing for frequently used commands"""
        return self.parse_command(text)
```

### Step 4: Error Handling and Robustness

Implement comprehensive error handling:

```python
class RobustVoiceProcessor:
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 5.0

    def robust_recognition(self):
        """Attempt recognition with retries and fallbacks"""
        for attempt in range(self.max_retries):
            try:
                result = self.speech_recognizer.listen_for_command()
                if result:
                    return result
            except Exception as e:
                rospy.logwarn(f"Recognition attempt {attempt + 1} failed: {e}")
                rospy.sleep(self.retry_delay)

        # If all attempts fail, return None or trigger fallback
        return None

    def handle_recognition_error(self, error):
        """Handle different types of recognition errors"""
        if "connection" in str(error).lower():
            rospy.logerr("Connection error - falling back to offline recognition")
            # Implement offline recognition
        elif "timeout" in str(error).lower():
            rospy.loginfo("Recognition timeout - continuing to listen")
        else:
            rospy.logerr(f"Recognition error: {error}")
```

## Testing and Validation

### Unit Testing

Create comprehensive unit tests for each component:

```python
import unittest
from unittest.mock import Mock, patch

class TestVoiceCommandProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = VoiceCommandProcessor()

    def test_navigation_intent_parsing(self):
        """Test parsing of navigation commands"""
        test_commands = [
            "Go to the kitchen",
            "Move to the living room",
            "Navigate to the office"
        ]

        for command in test_commands:
            parsed = self.processor.nlp_processor.parse_command(command)
            classified = self.processor.intent_classifier.classify_intent(parsed)

            self.assertEqual(classified['intent'], 'NAVIGATE')
            self.assertGreater(classified['confidence'], 0.5)

    def test_manipulation_intent_parsing(self):
        """Test parsing of manipulation commands"""
        test_commands = [
            "Pick up the red cup",
            "Grab the blue box",
            "Take the green ball"
        ]

        for command in test_commands:
            parsed = self.processor.nlp_processor.parse_command(command)
            classified = self.processor.intent_classifier.classify_intent(parsed)

            self.assertIn('MANIPULATE', classified['intent'])
            self.assertGreater(classified['confidence'], 0.5)

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

Test the complete voice processing pipeline:

```python
class VoiceProcessingIntegrationTest:
    def __init__(self):
        rospy.init_node('voice_processing_test')
        self.processor = VoiceCommandProcessor()

    def test_complete_pipeline(self):
        """Test complete voice processing pipeline"""
        test_commands = [
            ("Go to the kitchen", 'NAVIGATE'),
            ("Pick up the red cup", 'MANIPULATE_PICK'),
            ("Put the cup on the table", 'MANIPULATE_PLACE')
        ]

        for command, expected_intent in test_commands:
            # Simulate voice input
            self.processor.command_callback(String(data=command))

            # Verify intent classification and action mapping
            # (This would require monitoring published messages)
            print(f"Processed: {command} -> Expected: {expected_intent}")
```

## Performance Benchmarks

### Response Time Requirements

- **Audio Processing**: < 100ms
- **Speech Recognition**: < 500ms
- **NLP Processing**: < 200ms
- **Total Response Time**: < 3 seconds

### Accuracy Requirements

- **Speech Recognition**: >90% accuracy in quiet environments
- **Intent Classification**: >85% accuracy for common commands
- **Entity Extraction**: >80% accuracy for objects and locations

### Resource Usage

- **CPU Usage**: < 20% during continuous processing
- **Memory Usage**: < 500MB
- **Network Usage**: < 100KB/s for cloud-based recognition

## Troubleshooting and Common Issues

### Audio Quality Issues

1. **Background Noise**: Use noise suppression techniques or better microphones
2. **Audio Clipping**: Adjust microphone sensitivity and input levels
3. **Recognition Failures**: Implement fallback recognition methods

### Recognition Problems

1. **Poor Accuracy**: Train custom language models for domain-specific vocabulary
2. **Latency Issues**: Optimize for local processing when possible
3. **Wake Word Detection**: Fine-tune sensitivity to reduce false positives

### Integration Issues

1. **Message Format Mismatches**: Ensure consistent message formats between components
2. **Timing Issues**: Implement proper synchronization between processing stages
3. **Context Loss**: Maintain context across system restarts

## Best Practices

### Security Considerations

- **Privacy**: Do not store voice recordings without consent
- **Authentication**: Implement voice-based authentication for sensitive commands
- **Data Protection**: Encrypt voice data during transmission

### Accessibility

- **Multiple Languages**: Support for different languages and accents
- **Alternative Input**: Provide text-based command alternatives
- **Feedback**: Clear audio/visual feedback for command recognition

### Maintainability

- **Modular Design**: Keep components loosely coupled
- **Configuration**: Use parameter server for configurable settings
- **Logging**: Comprehensive logging for debugging and monitoring

## Next Steps and Integration

### Integration with Other Capstone Components

The voice command processing system integrates with:
- **Task Planning**: Provides high-level goals to the planning system
- **Navigation**: Issues navigation goals based on location commands
- **Manipulation**: Sends object and action parameters to manipulation system
- **Perception**: Requests object detection for unknown objects

### Advanced Features

Consider implementing:
- **Multi-turn Conversations**: Support for follow-up questions and clarifications
- **Emotion Recognition**: Detect user emotional state from voice
- **Adaptive Learning**: Learn user preferences and speech patterns over time

Continue with [Task Planning and Execution](./task-planning.md) to explore the implementation of the task planning and execution engine that will coordinate the actions initiated by voice commands.

## References

[All sources will be cited in the References section at the end of the book, following APA format]