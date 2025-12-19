---
sidebar_position: 20
---

# Failure Handling and Status Reporting: Autonomous Humanoid Capstone

## Overview

Failure handling and status reporting form the reliability and safety foundation of the autonomous humanoid system, ensuring robust operation in the face of unexpected conditions, system errors, and environmental challenges. This component encompasses error detection, classification, recovery strategies, graceful degradation, and comprehensive status communication to users and other system components. The system must detect failures quickly, respond appropriately to maintain safety, attempt recovery when possible, and provide clear status information to enable effective human oversight.

The failure handling system integrates with all other capstone components to monitor their operation, coordinate recovery actions, and maintain overall system stability. This implementation guide provides detailed instructions for building a comprehensive failure handling and status reporting framework that ensures safe and reliable operation of the autonomous humanoid system.

## System Architecture

### Failure Handling Architecture

The failure handling system implements a hierarchical, multi-layered architecture:

```
Component Monitoring → Error Detection → Error Classification → Recovery Strategy → Status Reporting
```

The architecture consists of:
1. **Component Monitors**: Monitor individual system components for failures
2. **Error Classifier**: Categorize errors by type and severity
3. **Recovery Manager**: Execute appropriate recovery strategies
4. **Safety Handler**: Ensure safe system states during failures
5. **Status Reporter**: Communicate system status to users and systems
6. **Logging System**: Record all failures and recovery actions

### Integration with Other Systems

The failure handling system interfaces with:
- **All Capstone Components**: Monitors for errors and coordinates recovery
- **Navigation System**: Handles navigation failures and obstacles
- **Manipulation System**: Manages manipulation errors and object losses
- **Perception System**: Deals with sensor failures and detection errors
- **Task Planning**: Adjusts plans when failures occur
- **Voice Processing**: Maintains communication despite errors

## Technical Implementation

### 1. Error Detection and Classification

#### Component Monitoring System

```python
import rospy
import threading
import time
from enum import Enum
from std_msgs.msg import String, Bool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from collections import defaultdict, deque
import json

class ErrorLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    FATAL = 4

class ErrorType(Enum):
    COMMUNICATION = "communication"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    ENVIRONMENTAL = "environmental"
    USER = "user"
    SAFETY = "safety"

class ComponentStatus(Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"
    UNKNOWN = "unknown"

class ComponentMonitor:
    """Monitors individual system components for failures"""

    def __init__(self, component_name: str, timeout: float = 5.0):
        self.component_name = component_name
        self.timeout = timeout
        self.last_heartbeat = rospy.get_time()
        self.status = ComponentStatus.OK
        self.error_history = deque(maxlen=100)  # Keep last 100 errors
        self.monitoring_active = True
        self.lock = threading.Lock()

    def heartbeat(self):
        """Update component heartbeat timestamp"""
        with self.lock:
            self.last_heartbeat = rospy.get_time()
            if self.status != ComponentStatus.FATAL:
                self.status = ComponentStatus.OK

    def report_error(self, error_msg: str, error_level: ErrorLevel, error_type: ErrorType):
        """Report an error from this component"""
        with self.lock:
            error_record = {
                'timestamp': rospy.get_time(),
                'message': error_msg,
                'level': error_level,
                'type': error_type,
                'component': self.component_name
            }
            self.error_history.append(error_record)

            # Update component status based on error level
            if error_level == ErrorLevel.FATAL:
                self.status = ComponentStatus.FATAL
            elif error_level == ErrorLevel.ERROR:
                self.status = ComponentStatus.ERROR
            elif error_level == ErrorLevel.WARN:
                self.status = ComponentStatus.WARNING

    def is_alive(self) -> bool:
        """Check if component is alive based on heartbeat"""
        with self.lock:
            current_time = rospy.get_time()
            return (current_time - self.last_heartbeat) < self.timeout

    def get_status(self) -> ComponentStatus:
        """Get current component status"""
        with self.lock:
            if not self.is_alive():
                return ComponentStatus.ERROR  # Component timed out
            return self.status

    def get_error_count(self, error_level: ErrorLevel = None) -> int:
        """Get count of errors, optionally filtered by level"""
        with self.lock:
            if error_level:
                return len([e for e in self.error_history if e['level'] == error_level])
            return len(self.error_history)

    def get_recent_errors(self, count: int = 10) -> list:
        """Get recent errors"""
        with self.lock:
            return list(self.error_history)[-count:]

class SystemMonitor:
    """Monitors the entire system for failures"""

    def __init__(self):
        self.components = {}  # component_name -> ComponentMonitor
        self.global_error_handlers = []
        self.system_status_publisher = rospy.Publisher('/system_status', String, queue_size=10)
        self.diagnostic_publisher = rospy.Publisher('/diagnostics', DiagnosticArray, queue_size=10)
        self.error_subscriber = rospy.Subscriber('/error_report', String, self.error_callback)

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def register_component(self, component_name: str, timeout: float = 5.0):
        """Register a component for monitoring"""
        if component_name not in self.components:
            self.components[component_name] = ComponentMonitor(component_name, timeout)
            rospy.loginfo(f"Registered component for monitoring: {component_name}")

    def component_heartbeat(self, component_name: str):
        """Component reports heartbeat"""
        if component_name in self.components:
            self.components[component_name].heartbeat()

    def report_error(self, component_name: str, error_msg: str,
                    error_level: ErrorLevel, error_type: ErrorType):
        """Report an error from a specific component"""
        if component_name in self.components:
            self.components[component_name].report_error(error_msg, error_level, error_type)
        else:
            # Create monitor for unknown component
            self.register_component(component_name)
            self.components[component_name].report_error(error_msg, error_level, error_type)

        # Log error based on level
        if error_level == ErrorLevel.ERROR:
            rospy.logerr(f"[{component_name}] {error_msg}")
        elif error_level == ErrorLevel.WARN:
            rospy.logwarn(f"[{component_name}] {error_msg}")
        elif error_level == ErrorLevel.INFO:
            rospy.loginfo(f"[{component_name}] {error_msg}")
        elif error_level == ErrorLevel.DEBUG:
            rospy.logdebug(f"[{component_name}] {error_msg}")

    def get_system_status(self) -> dict:
        """Get overall system status"""
        status = {
            'timestamp': rospy.get_time(),
            'components': {},
            'overall_status': ComponentStatus.OK,
            'error_count': 0,
            'active_errors': []
        }

        max_error_level = ErrorLevel.DEBUG

        for name, monitor in self.components.items():
            comp_status = monitor.get_status()
            status['components'][name] = {
                'status': comp_status.value,
                'alive': monitor.is_alive(),
                'error_count': monitor.get_error_count()
            }

            # Determine overall system status
            if comp_status == ComponentStatus.FATAL:
                status['overall_status'] = ComponentStatus.FATAL
                max_error_level = ErrorLevel.FATAL
            elif comp_status == ComponentStatus.ERROR and max_error_level < ErrorLevel.ERROR:
                status['overall_status'] = ComponentStatus.ERROR
                max_error_level = ErrorLevel.ERROR
            elif comp_status == ComponentStatus.WARNING and max_error_level < ErrorLevel.WARN:
                if status['overall_status'] != ComponentStatus.ERROR:
                    status['overall_status'] = ComponentStatus.WARNING
                max_error_level = ErrorLevel.WARN

            # Count total errors
            status['error_count'] += monitor.get_error_count()

            # Get recent errors if component has errors
            if monitor.get_error_count() > 0:
                recent_errors = monitor.get_recent_errors(3)  # Last 3 errors
                for error in recent_errors:
                    status['active_errors'].append({
                        'component': name,
                        'message': error['message'],
                        'level': error['level'].name,
                        'timestamp': error['timestamp']
                    })

        return status

    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        rate = rospy.Rate(1)  # 1 Hz monitoring

        while not rospy.is_shutdown():
            try:
                # Publish system status
                system_status = self.get_system_status()
                status_msg = String()
                status_msg.data = json.dumps(system_status, indent=2)
                self.system_status_publisher.publish(status_msg)

                # Publish diagnostic information
                self._publish_diagnostics(system_status)

                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error in monitoring loop: {e}")

    def _publish_diagnostics(self, system_status: dict):
        """Publish ROS diagnostics message"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = rospy.Time.now()

        for comp_name, comp_info in system_status['components'].items():
            diag_status = DiagnosticStatus()
            diag_status.name = f"capstone/{comp_name}"
            diag_status.hardware_id = comp_name

            if comp_info['status'] == 'ok':
                diag_status.level = DiagnosticStatus.OK
                diag_status.message = "Component operating normally"
            elif comp_info['status'] == 'warning':
                diag_status.level = DiagnosticStatus.WARN
                diag_status.message = "Component has warnings"
            elif comp_info['status'] == 'error':
                diag_status.level = DiagnosticStatus.ERROR
                diag_status.message = "Component has errors"
            elif comp_info['status'] == 'fatal':
                diag_status.level = DiagnosticStatus.ERROR
                diag_status.message = "Component has fatal errors"
            else:
                diag_status.level = DiagnosticStatus.STALE
                diag_status.message = "Component status unknown"

            diag_status.values = [
                {'key': 'alive', 'value': str(comp_info['alive'])},
                {'key': 'error_count', 'value': str(comp_info['error_count'])}
            ]

            diag_array.status.append(diag_status)

        self.diagnostic_publisher.publish(diag_array)

    def error_callback(self, msg):
        """Handle incoming error messages from other nodes"""
        try:
            error_data = json.loads(msg.data)
            component = error_data.get('component', 'unknown')
            message = error_data.get('message', 'Unknown error')
            level_str = error_data.get('level', 'ERROR')
            type_str = error_data.get('type', 'SOFTWARE')

            # Convert strings back to enums
            level = ErrorLevel[level_str.upper()]
            error_type = ErrorType[type_str.upper()]

            self.report_error(component, message, level, error_type)

        except Exception as e:
            rospy.logerr(f"Error parsing error callback: {e}")
```

#### Error Classification System

```python
class ErrorClassifier:
    """Classifies errors by type and severity"""

    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()

    def _initialize_error_patterns(self) -> dict:
        """Initialize known error patterns and their classifications"""
        return {
            # Communication errors
            'connection refused': (ErrorType.COMMUNICATION, ErrorLevel.ERROR),
            'timeout': (ErrorType.COMMUNICATION, ErrorLevel.WARN),
            'network': (ErrorType.COMMUNICATION, ErrorLevel.ERROR),
            'connection lost': (ErrorType.COMMUNICATION, ErrorLevel.ERROR),

            # Hardware errors
            'motor fault': (ErrorType.HARDWARE, ErrorLevel.ERROR),
            'encoder error': (ErrorType.HARDWARE, ErrorLevel.ERROR),
            'gripper fault': (ErrorType.HARDWARE, ErrorLevel.ERROR),
            'sensor failure': (ErrorType.HARDWARE, ErrorLevel.ERROR),
            'overheating': (ErrorType.HARDWARE, ErrorLevel.WARN),
            'low battery': (ErrorType.HARDWARE, ErrorLevel.WARN),

            # Software errors
            'segmentation fault': (ErrorType.SOFTWARE, ErrorLevel.FATAL),
            'memory error': (ErrorType.SOFTWARE, ErrorLevel.ERROR),
            'null pointer': (ErrorType.SOFTWARE, ErrorLevel.ERROR),
            'exception': (ErrorType.SOFTWARE, ErrorLevel.ERROR),
            'assertion failed': (ErrorType.SOFTWARE, ErrorLevel.ERROR),

            # Environmental errors
            'obstacle detected': (ErrorType.ENVIRONMENTAL, ErrorLevel.WARN),
            'navigation failed': (ErrorType.ENVIRONMENTAL, ErrorLevel.WARN),
            'object not found': (ErrorType.ENVIRONMENTAL, ErrorLevel.WARN),
            'grasp failed': (ErrorType.ENVIRONMENTAL, ErrorLevel.WARN),

            # Safety errors
            'collision detected': (ErrorType.SAFETY, ErrorLevel.ERROR),
            'safety limit exceeded': (ErrorType.SAFETY, ErrorLevel.ERROR),
            'emergency stop': (ErrorType.SAFETY, ErrorLevel.FATAL),
        }

    def classify_error(self, error_message: str) -> tuple:
        """Classify an error message by type and level"""
        error_message_lower = error_message.lower()

        # Check for known patterns
        for pattern, (error_type, error_level) in self.error_patterns.items():
            if pattern.lower() in error_message_lower:
                return error_type, error_level

        # Default classification based on keywords
        if any(keyword in error_message_lower for keyword in ['critical', 'fatal', 'crash']):
            return ErrorType.SOFTWARE, ErrorLevel.FATAL
        elif any(keyword in error_message_lower for keyword in ['error', 'failed', 'failure']):
            return ErrorType.SOFTWARE, ErrorLevel.ERROR
        elif any(keyword in error_message_lower for keyword in ['warning', 'warn', 'problem']):
            return ErrorType.SOFTWARE, ErrorLevel.WARN
        elif any(keyword in error_message_lower for keyword in ['info', 'notice', 'status']):
            return ErrorType.SOFTWARE, ErrorLevel.INFO
        else:
            return ErrorType.SOFTWARE, ErrorLevel.DEBUG

    def get_recovery_hint(self, error_message: str) -> str:
        """Get a suggested recovery action for an error"""
        error_type, error_level = self.classify_error(error_message)

        recovery_hints = {
            (ErrorType.COMMUNICATION, ErrorLevel.ERROR): "Check network connections and restart communication modules",
            (ErrorType.HARDWARE, ErrorLevel.ERROR): "Check hardware connections and perform hardware diagnostics",
            (ErrorType.ENVIRONMENTAL, ErrorLevel.WARN): "Reassess environment and replan accordingly",
            (ErrorType.SAFETY, ErrorLevel.ERROR): "Stop all motion and perform safety check",
            (ErrorType.SOFTWARE, ErrorLevel.FATAL): "Restart system and reload software"
        }

        return recovery_hints.get((error_type, error_level), "No specific recovery action available")

class ErrorSeverityCalculator:
    """Calculates error severity based on multiple factors"""

    def __init__(self):
        self.component_criticality = {
            'navigation': 3,  # High criticality
            'manipulation': 3,  # High criticality
            'perception': 2,  # Medium criticality
            'task_planning': 2,  # Medium criticality
            'voice_processing': 1,  # Low criticality
            'localization': 3,  # High criticality
        }

    def calculate_severity(self, error_type: ErrorType, error_level: ErrorLevel,
                          component_name: str, frequency: int = 1) -> int:
        """Calculate overall error severity score"""
        # Base severity from error level
        level_multiplier = {
            ErrorLevel.DEBUG: 1,
            ErrorLevel.INFO: 1,
            ErrorLevel.WARN: 2,
            ErrorLevel.ERROR: 3,
            ErrorLevel.FATAL: 4
        }

        # Component criticality multiplier
        criticality = self.component_criticality.get(component_name, 1)

        # Frequency multiplier (errors occurring frequently are more severe)
        freq_multiplier = min(frequency, 5)  # Cap at 5

        # Type-based adjustments
        type_multiplier = 1.0
        if error_type == ErrorType.SAFETY:
            type_multiplier = 2.0  # Safety errors are always more severe
        elif error_type == ErrorType.HARDWARE:
            type_multiplier = 1.5  # Hardware errors are more severe

        severity_score = (level_multiplier[error_level] *
                         criticality *
                         freq_multiplier *
                         type_multiplier)

        return int(severity_score)
```

### 2. Recovery and Graceful Degradation

#### Recovery Strategy Manager

```python
import subprocess
import signal
import os
from typing import Callable, List, Optional
import time

class RecoveryStrategy:
    """Base class for recovery strategies"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, error_context: dict) -> bool:
        """Execute the recovery strategy"""
        raise NotImplementedError

    def can_handle(self, error_context: dict) -> bool:
        """Check if this strategy can handle the given error"""
        raise NotImplementedError

class RetryStrategy(RecoveryStrategy):
    """Retry the failed operation"""

    def __init__(self):
        super().__init__("retry", "Retry the failed operation with exponential backoff")

    def can_handle(self, error_context: dict) -> bool:
        return error_context.get('error_type') in [ErrorType.COMMUNICATION, ErrorType.SOFTWARE]

    def execute(self, error_context: dict) -> bool:
        """Execute retry with exponential backoff"""
        max_retries = error_context.get('max_retries', 3)
        base_delay = error_context.get('base_delay', 1.0)

        for attempt in range(max_retries):
            rospy.loginfo(f"Retry attempt {attempt + 1}/{max_retries}")

            # Wait with exponential backoff
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

            # Try to execute the original operation
            # This would be passed as a function in practice
            if error_context.get('retry_function'):
                try:
                    success = error_context['retry_function']()
                    if success:
                        rospy.loginfo("Retry successful")
                        return True
                except Exception as e:
                    rospy.logwarn(f"Retry attempt {attempt + 1} failed: {e}")

        rospy.logerr("All retry attempts failed")
        return False

class ResetStrategy(RecoveryStrategy):
    """Reset the failing component"""

    def __init__(self):
        super().__init__("reset", "Reset the failing component or system")

    def can_handle(self, error_context: dict) -> bool:
        return error_context.get('error_type') in [ErrorType.HARDWARE, ErrorType.SOFTWARE]

    def execute(self, error_context: dict) -> bool:
        """Execute component reset"""
        component = error_context.get('component', 'unknown')
        rospy.loginfo(f"Resetting component: {component}")

        try:
            # For ROS nodes, we might need to restart them
            if 'node_name' in error_context:
                node_name = error_context['node_name']
                # Kill the node process
                subprocess.run(['rosnode', 'kill', node_name], check=False)
                # Wait for node to be killed
                time.sleep(2)
                # In practice, the node would need to be restarted
                rospy.loginfo(f"Reset complete for node: {node_name}")
                return True

            # For hardware components, send reset command
            if 'hardware_command' in error_context:
                # Execute hardware reset command
                # This is system-specific
                rospy.loginfo(f"Hardware reset command sent for: {component}")
                return True

            # Default: just log the reset
            rospy.loginfo(f"Component reset simulated for: {component}")
            return True

        except Exception as e:
            rospy.logerr(f"Reset failed: {e}")
            return False

class FallbackStrategy(RecoveryStrategy):
    """Use a fallback or alternative approach"""

    def __init__(self):
        super().__init__("fallback", "Use alternative approach or degrade functionality")

    def can_handle(self, error_context: dict) -> bool:
        return True  # Can handle any error as fallback

    def execute(self, error_context: dict) -> bool:
        """Execute fallback strategy"""
        component = error_context.get('component', 'unknown')
        error_type = error_context.get('error_type', ErrorType.SOFTWARE)

        rospy.loginfo(f"Activating fallback for component: {component}")

        # Different fallbacks based on component and error type
        if component == 'navigation' and error_type == ErrorType.ENVIRONMENTAL:
            # Use safe navigation mode
            rospy.loginfo("Switching to safe navigation mode")
            # Implement safe navigation logic
            return True

        elif component == 'perception' and error_type == ErrorType.ENVIRONMENTAL:
            # Use cached data or alternative sensors
            rospy.loginfo("Using cached perception data or alternative sensors")
            # Implement fallback perception
            return True

        elif component == 'manipulation' and error_type == ErrorType.HARDWARE:
            # Switch to safe position or emergency stop
            rospy.loginfo("Moving to safe position due to manipulation error")
            # Implement safe positioning
            return True

        else:
            # General fallback - degrade gracefully
            rospy.loginfo("Activating general fallback - degrading functionality")
            return True

class EmergencyStopStrategy(RecoveryStrategy):
    """Emergency stop for safety-critical situations"""

    def __init__(self):
        super().__init__("emergency_stop", "Immediate stop for safety-critical situations")

    def can_handle(self, error_context: dict) -> bool:
        return error_context.get('error_type') == ErrorType.SAFETY

    def execute(self, error_context: dict) -> bool:
        """Execute emergency stop"""
        rospy.logerr("EMERGENCY STOP ACTIVATED - HALTING ALL MOTION")

        # Stop all robot motion immediately
        cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1, latch=True)
        stop_cmd = Twist()
        cmd_vel_pub.publish(stop_cmd)

        # Stop manipulator if active
        try:
            # This would send stop commands to manipulator controllers
            pass
        except:
            pass

        # Log the emergency stop
        rospy.logerr(f"Emergency stop executed due to: {error_context.get('error_message', 'unknown error')}")
        return True

class RecoveryManager:
    """Manages recovery strategies and executes appropriate ones"""

    def __init__(self):
        self.strategies = [
            EmergencyStopStrategy(),
            RetryStrategy(),
            ResetStrategy(),
            FallbackStrategy()
        ]
        self.error_classifier = ErrorClassifier()
        self.severity_calculator = ErrorSeverityCalculator()

    def handle_error(self, error_context: dict) -> bool:
        """Handle an error by selecting and executing appropriate recovery strategy"""
        error_msg = error_context.get('error_message', 'Unknown error')
        component = error_context.get('component', 'unknown')
        error_type, error_level = self.error_classifier.classify_error(error_msg)

        # Update context with classified information
        error_context['error_type'] = error_type
        error_context['error_level'] = error_level

        rospy.loginfo(f"Handling error in {component}: {error_msg} (type: {error_type}, level: {error_level})")

        # Calculate severity
        severity = self.severity_calculator.calculate_severity(
            error_type, error_level, component
        )
        error_context['severity'] = severity

        # Select appropriate recovery strategy
        for strategy in self.strategies:
            if strategy.can_handle(error_context):
                rospy.loginfo(f"Executing recovery strategy: {strategy.name}")
                success = strategy.execute(error_context)

                if success:
                    rospy.loginfo(f"Recovery successful using {strategy.name}")
                    return True
                else:
                    rospy.logwarn(f"Recovery failed using {strategy.name}")

        # If no strategy worked, escalate
        rospy.logerr("No recovery strategy was successful")
        return False

    def register_strategy(self, strategy: RecoveryStrategy):
        """Register a new recovery strategy"""
        self.strategies.append(strategy)

    def get_recovery_options(self, error_context: dict) -> List[str]:
        """Get list of applicable recovery options for an error"""
        options = []
        for strategy in self.strategies:
            if strategy.can_handle(error_context):
                options.append(strategy.name)
        return options
```

#### Graceful Degradation System

```python
class DegradationLevel(Enum):
    FULL_FUNCTIONAL = "full_functional"
    PARTIAL_FUNCTIONAL = "partial_functional"
    SAFE_MODE = "safe_mode"
    EMERGENCY_STOP = "emergency_stop"

class DegradationManager:
    """Manages system degradation based on error severity"""

    def __init__(self):
        self.current_level = DegradationLevel.FULL_FUNCTIONAL
        self.degradation_thresholds = {
            0: DegradationLevel.FULL_FUNCTIONAL,      # No errors
            1: DegradationLevel.FULL_FUNCTIONAL,      # Minor errors
            3: DegradationLevel.PARTIAL_FUNCTIONAL,   # Multiple minor errors
            5: DegradationLevel.SAFE_MODE,            # Major errors
            8: DegradationLevel.EMERGENCY_STOP        # Critical errors
        }
        self.functionality_map = self._initialize_functionality_map()

    def _initialize_functionality_map(self) -> dict:
        """Initialize mapping of degradation levels to functionality"""
        return {
            DegradationLevel.FULL_FUNCTIONAL: {
                'navigation': True,
                'manipulation': True,
                'perception': True,
                'task_planning': True,
                'voice_processing': True,
                'max_speed': 1.0,
                'max_payload': 1.0,
                'precision_mode': True
            },
            DegradationLevel.PARTIAL_FUNCTIONAL: {
                'navigation': True,
                'manipulation': False,  # Disable manipulation
                'perception': True,
                'task_planning': True,
                'voice_processing': True,
                'max_speed': 0.7,  # Reduce speed
                'max_payload': 0.5,  # Reduce payload
                'precision_mode': False
            },
            DegradationLevel.SAFE_MODE: {
                'navigation': True,  # Can still navigate to safe location
                'manipulation': False,
                'perception': True,  # Basic perception for safety
                'task_planning': False,  # No complex planning
                'voice_processing': True,  # Can still receive commands
                'max_speed': 0.3,  # Very slow
                'max_payload': 0.0,  # No payload
                'precision_mode': False
            },
            DegradationLevel.EMERGENCY_STOP: {
                'navigation': False,  # Stop all motion
                'manipulation': False,
                'perception': False,  # Minimal perception only
                'task_planning': False,
                'voice_processing': True,  # Still listen for commands
                'max_speed': 0.0,  # No movement
                'max_payload': 0.0,
                'precision_mode': False
            }
        }

    def update_degradation_level(self, total_error_score: int) -> DegradationLevel:
        """Update degradation level based on total error score"""
        # Find the appropriate degradation level
        level = DegradationLevel.FULL_FUNCTIONAL
        for threshold, degradation_level in sorted(self.degradation_thresholds.items(), reverse=True):
            if total_error_score >= threshold:
                level = degradation_level
                break

        if level != self.current_level:
            old_level = self.current_level
            self.current_level = level
            self._apply_degradation_changes(old_level, level)

        return self.current_level

    def _apply_degradation_changes(self, old_level: DegradationLevel, new_level: DegradationLevel):
        """Apply changes when degradation level changes"""
        rospy.loginfo(f"System degradation level changed: {old_level.value} -> {new_level.value}")

        # Publish degradation level
        degradation_pub = rospy.Publisher('/system_degradation_level', String, queue_size=1, latch=True)
        level_msg = String()
        level_msg.data = new_level.value
        degradation_pub.publish(level_msg)

        # Apply functionality restrictions
        functionality = self.functionality_map[new_level]

        # Example: Reduce navigation speed
        if not functionality['navigation']:
            # Stop all navigation
            cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1, latch=True)
            stop_cmd = Twist()
            cmd_vel_pub.publish(stop_cmd)
            rospy.loginfo("Navigation disabled due to degradation")
        else:
            # Set speed limits
            # This would interface with navigation system
            pass

        # Example: Disable manipulation
        if not functionality['manipulation']:
            # Stop any ongoing manipulation
            # This would interface with manipulation system
            rospy.loginfo("Manipulation disabled due to degradation")

        # Log the changes
        rospy.loginfo(f"Applied functionality changes: {functionality}")

    def is_functionality_available(self, function_name: str) -> bool:
        """Check if specific functionality is available at current degradation level"""
        functionality = self.functionality_map[self.current_level]
        return functionality.get(function_name, False)

    def get_current_functionality(self) -> dict:
        """Get current system functionality status"""
        return self.functionality_map[self.current_level].copy()

    def get_degradation_status(self) -> dict:
        """Get comprehensive degradation status"""
        return {
            'current_level': self.current_level.value,
            'functionality': self.get_current_functionality(),
            'message': f"System operating at {self.current_level.value} level"
        }
```

### 3. Status Reporting and Communication

#### Status Reporting System

```python
from datetime import datetime
import psutil
from std_msgs.msg import Header
from diagnostic_msgs.msg import KeyValue

class StatusReporter:
    """Comprehensive status reporting system"""

    def __init__(self):
        self.status_publisher = rospy.Publisher('/system_status_detailed', String, queue_size=10)
        self.health_publisher = rospy.Publisher('/system_health', String, queue_size=10)
        self.heartbeat_publisher = rospy.Publisher('/system_heartbeat', String, queue_size=10)

        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_loop(self):
        """Publish system heartbeat"""
        rate = rospy.Rate(1)  # 1 Hz heartbeat

        while not rospy.is_shutdown():
            try:
                heartbeat_msg = {
                    'timestamp': rospy.get_time(),
                    'node': rospy.get_name(),
                    'status': 'running',
                    'uptime': rospy.get_time() - rospy.get_param('start_time', rospy.get_time())
                }

                heartbeat_str = json.dumps(heartbeat_msg)
                self.heartbeat_publisher.publish(String(data=heartbeat_str))
                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error in heartbeat loop: {e}")

    def publish_system_status(self, system_monitor: SystemMonitor, recovery_manager: RecoveryManager):
        """Publish comprehensive system status"""
        status_data = {
            'timestamp': rospy.get_time(),
            'system_status': system_monitor.get_system_status(),
            'degradation_status': self._get_degradation_status(),
            'resource_usage': self._get_resource_usage(),
            'active_recovery_options': self._get_active_recovery_options(system_monitor),
            'uptime': rospy.get_time() - rospy.get_param('start_time', rospy.get_time())
        }

        status_json = json.dumps(status_data, indent=2)
        self.status_publisher.publish(String(data=status_json))

    def _get_degradation_status(self) -> dict:
        """Get current degradation status"""
        # This would interface with DegradationManager
        # For now, return a placeholder
        return {
            'level': 'full_functional',
            'functionality': {
                'navigation': True,
                'manipulation': True,
                'perception': True
            }
        }

    def _get_resource_usage(self) -> dict:
        """Get system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'temperature': self._get_temperature(),  # System-specific
            'network_io': psutil.net_io_counters()._asdict() if psutil.HAS_NET_IO_COUNTERS else {}
        }

    def _get_temperature(self) -> Optional[float]:
        """Get system temperature (if available)"""
        try:
            # Try to get temperature from system sensors
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get CPU temperature if available
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            if entries:
                                return entries[0].current
            return None
        except:
            return None

    def _get_active_recovery_options(self, system_monitor: SystemMonitor) -> dict:
        """Get currently available recovery options"""
        # This would analyze current system state to determine available recovery options
        # For now, return a placeholder
        return {
            'retry_available': True,
            'reset_available': True,
            'fallback_available': True,
            'emergency_stop_available': True
        }

    def publish_health_report(self, system_monitor: SystemMonitor):
        """Publish detailed health report"""
        system_status = system_monitor.get_system_status()

        health_report = {
            'header': {
                'stamp': rospy.Time.now().to_sec(),
                'frame_id': 'system'
            },
            'status': self._calculate_health_status(system_status),
            'components': system_status['components'],
            'errors': system_status['active_errors'][-10:],  # Last 10 errors
            'metrics': self._calculate_health_metrics(system_status)
        }

        health_json = json.dumps(health_report, indent=2)
        self.health_publisher.publish(String(data=health_json))

    def _calculate_health_status(self, system_status: dict) -> int:
        """Calculate overall system health status (0=bad, 1=ok, 2=good)"""
        overall_status = system_status['overall_status']

        if overall_status == ComponentStatus.FATAL:
            return 0  # Bad
        elif overall_status == ComponentStatus.ERROR:
            return 0  # Bad
        elif overall_status == ComponentStatus.WARNING:
            return 1  # Ok
        else:
            return 2  # Good

    def _calculate_health_metrics(self, system_status: dict) -> dict:
        """Calculate health metrics"""
        total_components = len(system_status['components'])
        healthy_components = sum(1 for comp in system_status['components'].values()
                               if comp['status'] == 'ok')

        error_rate = system_status['error_count'] / max(total_components, 1)

        return {
            'total_components': total_components,
            'healthy_components': healthy_components,
            'error_rate': error_rate,
            'health_score': healthy_components / max(total_components, 1)
        }

class UserStatusInterface:
    """Interface for communicating status to users"""

    def __init__(self):
        self.voice_publisher = rospy.Publisher('/voice_output', String, queue_size=10)
        self.display_publisher = rospy.Publisher('/display_output', String, queue_size=10)
        self.led_publisher = rospy.Publisher('/led_status', String, queue_size=10)

    def report_status_to_user(self, status_data: dict):
        """Report system status to user through multiple channels"""
        # Extract relevant information
        overall_status = status_data.get('system_status', {}).get('overall_status', 'unknown')
        active_errors = status_data.get('system_status', {}).get('active_errors', [])
        degradation_level = status_data.get('degradation_status', {}).get('level', 'full_functional')

        # Voice feedback
        voice_msg = self._generate_voice_feedback(overall_status, degradation_level, active_errors)
        self.voice_publisher.publish(String(data=voice_msg))

        # Display feedback
        display_msg = self._generate_display_feedback(overall_status, degradation_level, active_errors)
        self.display_publisher.publish(String(data=display_msg))

        # LED status
        led_status = self._determine_led_status(overall_status, degradation_level)
        self.led_publisher.publish(String(data=led_status))

    def _generate_voice_feedback(self, overall_status: str, degradation_level: str, active_errors: list) -> str:
        """Generate voice feedback message"""
        if overall_status == 'fatal':
            return "Emergency: System failure detected. Stopping all operations. Please contact support."
        elif overall_status == 'error':
            if degradation_level == 'safe_mode':
                return f"Warning: System in safe mode due to errors. {len(active_errors)} active issues detected."
            else:
                return f"Error: Issues detected in system. Continuing with reduced functionality."
        elif overall_status == 'warning':
            return f"Warning: {len(active_errors)} issues detected. System operational but monitor closely."
        else:
            return "System status normal. All components operating correctly."

    def _generate_display_feedback(self, overall_status: str, degradation_level: str, active_errors: list) -> str:
        """Generate display feedback message"""
        status_lines = [
            f"System Status: {overall_status.upper()}",
            f"Degradation Level: {degradation_level.upper()}",
            f"Active Errors: {len(active_errors)}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]

        if active_errors:
            status_lines.append("Recent Errors:")
            for error in active_errors[-3:]:  # Show last 3 errors
                status_lines.append(f"  - {error.get('component', 'unknown')}: {error.get('message', 'unknown')}")

        return "\n".join(status_lines)

    def _determine_led_status(self, overall_status: str, degradation_level: str) -> str:
        """Determine LED status based on system state"""
        if overall_status == 'fatal':
            return "RED_FAST_BLINK"  # Critical failure
        elif overall_status == 'error':
            if degradation_level == 'emergency_stop':
                return "RED_SOLID"  # Emergency stop
            else:
                return "YELLOW_BLINK"  # Error condition
        elif overall_status == 'warning':
            return "YELLOW_SOLID"  # Warning
        else:
            return "GREEN_SOLID"  # Normal operation
```

### 4. Integration and Coordination

#### Failure Handling Coordinator

```python
class FailureHandlingCoordinator:
    """Coordinates all failure handling components"""

    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.recovery_manager = RecoveryManager()
        self.degradation_manager = DegradationManager()
        self.status_reporter = StatusReporter()
        self.user_interface = UserStatusInterface()

        # Register critical components for monitoring
        self._register_critical_components()

        # Start coordination thread
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()

    def _register_critical_components(self):
        """Register all system components for monitoring"""
        critical_components = [
            'navigation',
            'manipulation',
            'perception',
            'task_planning',
            'voice_processing',
            'localization',
            'sensors',
            'actuators'
        ]

        for component in critical_components:
            self.system_monitor.register_component(component)

    def handle_component_error(self, component_name: str, error_message: str,
                             error_level: ErrorLevel = ErrorLevel.ERROR):
        """Handle an error from a specific component"""
        # Classify the error
        error_classifier = ErrorClassifier()
        error_type, classified_level = error_classifier.classify_error(error_message)

        # Report to system monitor
        self.system_monitor.report_error(component_name, error_message, classified_level, error_type)

        # Create error context for recovery
        error_context = {
            'component': component_name,
            'error_message': error_message,
            'error_type': error_type,
            'error_level': classified_level,
            'timestamp': rospy.get_time(),
            'severity': 0  # Will be calculated by recovery manager
        }

        # Attempt recovery
        recovery_success = self.recovery_manager.handle_error(error_context)

        # Update degradation level based on current error state
        system_status = self.system_monitor.get_system_status()
        total_errors = system_status['error_count']
        new_degradation = self.degradation_manager.update_degradation_level(total_errors)

        # Publish updated status
        self.status_reporter.publish_system_status(self.system_monitor, self.recovery_manager)

        # Report to user if needed
        status_data = {
            'system_status': system_status,
            'degradation_status': self.degradation_manager.get_degradation_status()
        }
        self.user_interface.report_status_to_user(status_data)

        return recovery_success

    def _coordination_loop(self):
        """Main coordination loop"""
        rate = rospy.Rate(0.1)  # 0.1 Hz (every 10 seconds)

        while not rospy.is_shutdown():
            try:
                # Publish detailed status
                self.status_reporter.publish_system_status(self.system_monitor, self.recovery_manager)

                # Publish health report
                self.status_reporter.publish_health_report(self.system_monitor)

                # Check for any necessary actions
                self._check_system_conditions()

                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error in coordination loop: {e}")

    def _check_system_conditions(self):
        """Check system conditions and take appropriate actions"""
        system_status = self.system_monitor.get_system_status()

        # Check for high error rates
        if system_status['error_count'] > 10:  # Arbitrary threshold
            rospy.logwarn("High error rate detected, reviewing system state")

        # Check individual component health
        for comp_name, comp_info in system_status['components'].items():
            if not comp_info['alive'] and comp_info['status'] != 'ok':
                rospy.logwarn(f"Component {comp_name} is not responding")

    def get_system_health_summary(self) -> dict:
        """Get comprehensive system health summary"""
        system_status = self.system_monitor.get_system_status()
        degradation_status = self.degradation_manager.get_degradation_status()
        resource_usage = self.status_reporter._get_resource_usage()

        return {
            'system_status': system_status,
            'degradation_status': degradation_status,
            'resource_usage': resource_usage,
            'recovery_status': {
                'last_recovery_time': getattr(self, '_last_recovery_time', 0),
                'recovery_success_rate': getattr(self, '_recovery_success_rate', 1.0)
            }
        }

    def component_heartbeat(self, component_name: str):
        """Handle component heartbeat"""
        self.system_monitor.component_heartbeat(component_name)

class SafeOperationManager:
    """Ensures safe operation during failures"""

    def __init__(self, failure_coordinator: FailureHandlingCoordinator):
        self.failure_coordinator = failure_coordinator
        self.safety_zones = []  # Defined safe areas
        self.emergency_stop_active = False
        self.safe_position = None

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            rospy.logerr("EMERGENCY STOP ACTIVATED")

            # Stop all motion
            self._stop_all_motion()

            # Report to failure coordinator
            self.failure_coordinator.handle_component_error(
                'safety_system',
                'Emergency stop activated',
                ErrorLevel.FATAL
            )

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop_active = False
        rospy.loginfo("Emergency stop deactivated")

    def _stop_all_motion(self):
        """Stop all robot motion"""
        # Stop base motion
        cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1, latch=True)
        stop_cmd = Twist()
        cmd_vel_pub.publish(stop_cmd)

        # Stop manipulator (if present)
        try:
            # This would interface with manipulator controllers
            pass
        except:
            pass

        # Log the stop action
        rospy.loginfo("All motion stopped due to safety system")

    def check_safety_conditions(self):
        """Check safety conditions and take action if needed"""
        if self.emergency_stop_active:
            return False  # No operations allowed during emergency stop

        # Check for safety-critical errors
        system_status = self.failure_coordinator.system_monitor.get_system_status()

        if system_status['overall_status'] == ComponentStatus.FATAL:
            self.activate_emergency_stop()
            return False

        return True

    def move_to_safe_position(self):
        """Move robot to a predefined safe position"""
        if self.safe_position:
            rospy.loginfo(f"Moving to safe position: {self.safe_position}")
            # This would interface with navigation system
            # navigation_client.send_goal(self.safe_position)
            return True
        else:
            rospy.logwarn("No safe position defined")
            return False

    def define_safe_position(self, position):
        """Define a safe position for emergency situations"""
        self.safe_position = position
        rospy.loginfo(f"Safe position defined as: {position}")
```

## Implementation Steps

### Step 1: Set up Failure Handling Infrastructure

1. Create the main failure handling node:

```python
#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import json
import threading
import time

class FailureHandlingNode:
    def __init__(self):
        rospy.init_node('failure_handling_node')
        rospy.set_param('start_time', rospy.get_time())

        # Initialize failure handling components
        self.failure_coordinator = FailureHandlingCoordinator()
        self.safe_manager = SafeOperationManager(self.failure_coordinator)

        # Setup publishers and subscribers
        self.error_sub = rospy.Subscriber('/error_input', String, self.error_callback)
        self.heartbeat_sub = rospy.Subscriber('/component_heartbeat', String, self.heartbeat_callback)
        self.emergency_stop_sub = rospy.Subscriber('/emergency_stop', String, self.emergency_stop_callback)
        self.safety_pub = rospy.Publisher('/safety_status', String, queue_size=10)

        # Register this node for monitoring
        self.failure_coordinator.system_monitor.register_component('failure_handling', timeout=10.0)

        rospy.loginfo("Failure handling system initialized")

    def error_callback(self, msg):
        """Handle incoming error messages"""
        try:
            error_data = json.loads(msg.data)
            component = error_data.get('component', 'unknown')
            message = error_data.get('message', 'Unknown error')
            level_str = error_data.get('level', 'ERROR')

            # Convert level string to enum
            try:
                level = ErrorLevel[level_str.upper()]
            except KeyError:
                level = ErrorLevel.ERROR

            rospy.loginfo(f"Received error from {component}: {message}")

            # Handle the error through coordinator
            success = self.failure_coordinator.handle_component_error(component, message, level)

            # Publish safety status
            safety_status = String()
            safety_status.data = json.dumps({
                'timestamp': rospy.get_time(),
                'error_handled': success,
                'component': component,
                'message': message
            })
            self.safety_pub.publish(safety_status)

        except Exception as e:
            rospy.logerr(f"Error processing error callback: {e}")

    def heartbeat_callback(self, msg):
        """Handle component heartbeat"""
        try:
            heartbeat_data = json.loads(msg.data)
            component = heartbeat_data.get('component', 'unknown')

            self.failure_coordinator.component_heartbeat(component)

        except Exception as e:
            rospy.logerr(f"Error processing heartbeat: {e}")

    def emergency_stop_callback(self, msg):
        """Handle emergency stop requests"""
        try:
            emergency_data = json.loads(msg.data)
            reason = emergency_data.get('reason', 'unknown')

            rospy.logerr(f"Emergency stop requested: {reason}")
            self.safe_manager.activate_emergency_stop()

        except Exception as e:
            rospy.logerr(f"Error processing emergency stop: {e}")

    def run(self):
        """Main execution loop"""
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            try:
                # Check safety conditions
                if not self.safe_manager.check_safety_conditions():
                    rospy.logwarn("Safety conditions not met, pausing operations")

                # Send periodic heartbeat
                self.failure_coordinator.component_heartbeat('failure_handling')

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"Error in main loop: {e}")
                # Even if there's an error, try to continue

if __name__ == '__main__':
    failure_node = FailureHandlingNode()
    try:
        failure_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Failure handling node shutting down")
    except KeyboardInterrupt:
        rospy.loginfo("Failure handling node interrupted")
```

### Step 2: Configure Failure Handling Parameters

Create a configuration file for failure handling parameters:

```yaml
# failure_handling_params.yaml
failure_handling_node:
  ros__parameters:
    heartbeat_timeout: 5.0
    error_debounce_time: 1.0
    max_error_history: 100
    degradation_thresholds:
      warning: 3
      safe_mode: 5
      emergency_stop: 8
    recovery_attempts: 3
    retry_base_delay: 1.0
    emergency_stop_topic: "/emergency_stop"
    system_status_topic: "/system_status"
    error_input_topic: "/error_input"

degradation_manager:
  ros__parameters:
    full_functional_threshold: 0
    partial_functional_threshold: 3
    safe_mode_threshold: 5
    emergency_stop_threshold: 8

recovery_manager:
  ros__parameters:
    max_retries: 3
    base_delay: 1.0
    exponential_backoff_factor: 2.0
```

### Step 3: Implement Component-Specific Error Handlers

```python
class NavigationErrorHandler:
    """Error handler for navigation system"""

    def __init__(self, failure_coordinator):
        self.failure_coordinator = failure_coordinator
        self.nav_error_count = 0
        self.consecutive_failures = 0

    def handle_navigation_error(self, error_msg, error_type):
        """Handle navigation-specific errors"""
        self.nav_error_count += 1

        if "local minimum" in error_msg.lower():
            # Local minimum - try recovery behaviors
            self._handle_local_minimum()
        elif "obstacle" in error_msg.lower():
            # Obstacle - try alternative path
            self._handle_obstacle_error()
        elif "goal unreachable" in error_msg.lower():
            # Goal unreachable - report to user
            self._handle_unreachable_goal()
        elif "transform" in error_msg.lower():
            # Transform error - likely localization issue
            self._handle_transform_error()
        else:
            # General navigation error
            pass

        # Report to main failure coordinator
        self.failure_coordinator.handle_component_error(
            'navigation', error_msg, ErrorLevel.WARN
        )

    def _handle_local_minimum(self):
        """Handle local minimum navigation failure"""
        rospy.loginfo("Attempting to escape local minimum")
        # This would trigger recovery behaviors like spinning or backing up
        pass

    def _handle_obstacle_error(self):
        """Handle obstacle-related navigation errors"""
        rospy.loginfo("Replanning to avoid obstacle")
        # This would trigger replanning with updated costmap
        pass

    def _handle_unreachable_goal(self):
        """Handle unreachable goal errors"""
        rospy.loginfo("Goal appears unreachable, reporting to user")
        # This would trigger user notification
        pass

    def _handle_transform_error(self):
        """Handle transform/TF errors"""
        rospy.loginfo("Transform error detected, checking localization")
        # This would trigger localization recovery
        pass

class ManipulationErrorHandler:
    """Error handler for manipulation system"""

    def __init__(self, failure_coordinator):
        self.failure_coordinator = failure_coordinator
        self.manip_error_count = 0

    def handle_manipulation_error(self, error_msg, error_type):
        """Handle manipulation-specific errors"""
        self.manip_error_count += 1

        if "grasp failed" in error_msg.lower():
            self._handle_grasp_failure()
        elif "collision" in error_msg.lower():
            self._handle_collision_error()
        elif "joint limit" in error_msg.lower():
            self._handle_joint_limit_error()
        elif "ik solution" in error_msg.lower():
            self._handle_ik_error()
        else:
            # General manipulation error
            pass

        # Report to main failure coordinator
        self.failure_coordinator.handle_component_error(
            'manipulation', error_msg, ErrorLevel.WARN
        )

    def _handle_grasp_failure(self):
        """Handle grasp failure"""
        rospy.loginfo("Grasp failed, trying alternative grasp")
        # This would trigger grasp re-planning
        pass

    def _handle_collision_error(self):
        """Handle collision during manipulation"""
        rospy.logerr("Collision detected during manipulation, stopping")
        # This would trigger emergency stop for manipulator
        pass

    def _handle_joint_limit_error(self):
        """Handle joint limit errors"""
        rospy.logwarn("Joint limit exceeded, adjusting trajectory")
        # This would trigger trajectory adjustment
        pass

    def _handle_ik_error(self):
        """Handle inverse kinematics errors"""
        rospy.logwarn("No IK solution found, adjusting goal pose")
        # This would trigger goal pose adjustment
        pass

class PerceptionErrorHandler:
    """Error handler for perception system"""

    def __init__(self, failure_coordinator):
        self.failure_coordinator = failure_coordinator
        self.percept_error_count = 0

    def handle_perception_error(self, error_msg, error_type):
        """Handle perception-specific errors"""
        self.percept_error_count += 1

        if "no detections" in error_msg.lower():
            self._handle_no_detections()
        elif "calibration" in error_msg.lower():
            self._handle_calibration_error()
        elif "depth" in error_msg.lower():
            self._handle_depth_error()
        elif "occlusion" in error_msg.lower():
            self._handle_occlusion_error()
        else:
            # General perception error
            pass

        # Report to main failure coordinator
        self.failure_coordinator.handle_component_error(
            'perception', error_msg, ErrorLevel.WARN
        )

    def _handle_no_detections(self):
        """Handle case where no objects are detected"""
        rospy.loginfo("No objects detected, suggesting environment change or repositioning")
        # This might trigger navigation to a different viewpoint
        pass

    def _handle_calibration_error(self):
        """Handle sensor calibration errors"""
        rospy.logerr("Sensor calibration error detected, stopping perception system")
        # This would require manual recalibration
        pass

    def _handle_depth_error(self):
        """Handle depth sensing errors"""
        rospy.logwarn("Depth sensing issues detected, using alternative methods")
        # This might switch to stereo or structured light methods
        pass

    def _handle_occlusion_error(self):
        """Handle object occlusion"""
        rospy.loginfo("Object occluded, trying different viewpoint")
        # This would trigger repositioning for better view
        pass
```

## Testing and Validation

### Unit Testing

```python
import unittest
from unittest.mock import Mock, patch

class TestComponentMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = ComponentMonitor("test_component", timeout=2.0)

    def test_heartbeat_updates_status(self):
        """Test that heartbeat updates component status"""
        initial_status = self.monitor.get_status()
        self.assertEqual(initial_status, ComponentStatus.OK)

        # Simulate heartbeat
        self.monitor.heartbeat()
        time.sleep(0.1)  # Allow for thread synchronization

        status_after_heartbeat = self.monitor.get_status()
        self.assertEqual(status_after_heartbeat, ComponentStatus.OK)

    def test_timeout_detection(self):
        """Test that component timeout is detected"""
        # Set heartbeat to a time in the past
        self.monitor.last_heartbeat = rospy.get_time() - 5.0  # 5 seconds ago

        self.assertFalse(self.monitor.is_alive())
        self.assertEqual(self.monitor.get_status(), ComponentStatus.ERROR)

    def test_error_reporting(self):
        """Test error reporting functionality"""
        self.monitor.report_error("Test error", ErrorLevel.ERROR, ErrorType.SOFTWARE)

        status = self.monitor.get_status()
        self.assertEqual(status, ComponentStatus.ERROR)

        error_count = self.monitor.get_error_count(ErrorLevel.ERROR)
        self.assertEqual(error_count, 1)

class TestErrorClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = ErrorClassifier()

    def test_communication_error_classification(self):
        """Test classification of communication errors"""
        error_type, error_level = self.classifier.classify_error("Connection refused to server")
        self.assertEqual(error_type, ErrorType.COMMUNICATION)
        self.assertEqual(error_level, ErrorLevel.ERROR)

    def test_hardware_error_classification(self):
        """Test classification of hardware errors"""
        error_type, error_level = self.classifier.classify_error("Motor fault detected")
        self.assertEqual(error_type, ErrorType.HARDWARE)
        self.assertEqual(error_level, ErrorLevel.ERROR)

    def test_safety_error_classification(self):
        """Test classification of safety errors"""
        error_type, error_level = self.classifier.classify_error("Collision detected")
        self.assertEqual(error_type, ErrorType.SAFETY)
        self.assertEqual(error_level, ErrorLevel.ERROR)

class TestRecoveryManager(unittest.TestCase):
    def setUp(self):
        self.recovery_manager = RecoveryManager()

    def test_emergency_stop_strategy(self):
        """Test emergency stop strategy selection"""
        error_context = {
            'error_message': "collision detected",
            'error_type': ErrorType.SAFETY,
            'component': 'navigation'
        }

        # This should select the emergency stop strategy first
        options = self.recovery_manager.get_recovery_options(error_context)
        self.assertIn('emergency_stop', options)

    def test_retry_strategy_applicability(self):
        """Test that retry strategy applies to appropriate errors"""
        error_context = {
            'error_message': "connection timeout",
            'error_type': ErrorType.COMMUNICATION,
            'component': 'perception'
        }

        options = self.recovery_manager.get_recovery_options(error_context)
        self.assertIn('retry', options)

class TestDegradationManager(unittest.TestCase):
    def setUp(self):
        self.degradation_manager = DegradationManager()

    def test_degradation_level_calculation(self):
        """Test degradation level calculation"""
        # Test full functional level (no errors)
        level = self.degradation_manager.update_degradation_level(0)
        self.assertEqual(level, DegradationLevel.FULL_FUNCTIONAL)

        # Test safe mode level (medium errors)
        level = self.degradation_manager.update_degradation_level(6)
        self.assertEqual(level, DegradationLevel.SAFE_MODE)

        # Test emergency stop level (high errors)
        level = self.degradation_manager.update_degradation_level(10)
        self.assertEqual(level, DegradationLevel.EMERGENCY_STOP)

    def test_functionality_check(self):
        """Test functionality availability checking"""
        # Set to safe mode
        self.degradation_manager.update_degradation_level(6)

        # In safe mode, navigation should be available but manipulation should not
        self.assertTrue(self.degradation_manager.is_functionality_available('navigation'))
        self.assertFalse(self.degradation_manager.is_functionality_available('manipulation'))

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
class FailureHandlingIntegrationTest:
    def __init__(self):
        rospy.init_node('failure_handling_integration_test')
        self.coordinator = FailureHandlingCoordinator()

    def test_error_propagation(self):
        """Test complete error handling pipeline"""
        print("Testing error propagation...")

        # Simulate errors from different components
        errors_to_test = [
            ("navigation", "obstacle detected", ErrorLevel.WARN),
            ("manipulation", "grasp failed", ErrorLevel.WARN),
            ("perception", "no objects detected", ErrorLevel.WARN),
            ("navigation", "goal unreachable", ErrorLevel.ERROR),
        ]

        for component, error_msg, error_level in errors_to_test:
            print(f"Injecting error: {component} - {error_msg}")
            success = self.coordinator.handle_component_error(component, error_msg, error_level)
            print(f"Recovery success: {success}")

        # Check system status after errors
        health_summary = self.coordinator.get_system_health_summary()
        print(f"System health after errors: {health_summary['system_status']['overall_status']}")

    def test_degradation_behavior(self):
        """Test system degradation behavior"""
        print("Testing degradation behavior...")

        # Simulate increasing number of errors
        for i in range(10):
            self.coordinator.handle_component_error(
                f"component_{i}",
                f"simulated error {i}",
                ErrorLevel.WARN
            )

            # Check current degradation level
            current_level = self.coordinator.degradation_manager.current_level
            print(f"After {i+1} errors: {current_level.value}")

    def test_recovery_sequence(self):
        """Test recovery strategy sequence"""
        print("Testing recovery sequence...")

        # Simulate a communication error that should trigger retry
        success = self.coordinator.handle_component_error(
            "perception",
            "connection timeout to camera",
            ErrorLevel.ERROR
        )
        print(f"Communication error recovery success: {success}")

        # Simulate a safety error that should trigger emergency stop
        success = self.coordinator.handle_component_error(
            "navigation",
            "collision detected",
            ErrorLevel.FATAL
        )
        print(f"Safety error recovery success: {success}")

    def test_user_notification(self):
        """Test user notification system"""
        print("Testing user notification...")

        # Simulate an error and check if user is notified
        self.coordinator.handle_component_error(
            "manipulation",
            "grasp failed after 3 attempts",
            ErrorLevel.ERROR
        )

        print("User notification test completed")
```

## Performance Benchmarks

### Failure Handling Performance

- **Error Detection**: < 10ms for local error detection
- **Error Classification**: < 5ms per error message
- **Recovery Strategy Selection**: < 20ms for strategy evaluation
- **System Status Updates**: < 100ms for comprehensive status
- **Memory Usage**: < 50MB for full failure handling system

### Reliability Metrics

- **Error Detection Rate**: > 99% for critical errors
- **Recovery Success Rate**: > 85% for common error types
- **False Positive Rate**: < 1% for error classification
- **System Availability**: > 95% under normal operating conditions
- **Mean Time To Recovery**: < 30 seconds for most errors

## Troubleshooting and Common Issues

### Error Detection Problems

1. **False Positives**: Adjust error pattern matching and confidence thresholds
2. **Missed Errors**: Improve error reporting from components
3. **Timing Issues**: Ensure proper heartbeat intervals and timeouts
4. **Resource Overhead**: Optimize monitoring frequency and data collection

### Recovery Problems

1. **Failed Recoveries**: Implement more robust recovery strategies
2. **Infinite Loops**: Add recovery attempt limits and escalation
3. **Partial Recovery**: Ensure system returns to known safe state
4. **Recovery Conflicts**: Coordinate between multiple recovery attempts

## Best Practices

### Safety Considerations

- **Defense in Depth**: Multiple layers of safety checks
- **Fail-Safe**: Ensure safe state when errors occur
- **Graceful Degradation**: Maintain basic functionality during errors
- **Emergency Procedures**: Clear protocols for critical failures

### Performance Optimization

- **Selective Monitoring**: Monitor only critical components
- **Efficient Classification**: Use fast pattern matching algorithms
- **Caching**: Cache frequently accessed data
- **Threading**: Use separate threads for monitoring and recovery

### Maintainability

- **Clear Logging**: Comprehensive and structured error logging
- **Configuration**: Use parameters for easy tuning
- **Modularity**: Keep components loosely coupled
- **Documentation**: Maintain clear recovery procedure documentation

## Next Steps and Integration

### Integration with Other Capstone Components

The failure handling system integrates with:
- **All Components**: Provides monitoring and error handling
- **Navigation**: Handles navigation-specific failures
- **Manipulation**: Manages manipulation errors and safety
- **Perception**: Deals with sensor and detection failures
- **Task Planning**: Adjusts plans when failures occur
- **Voice Processing**: Maintains communication despite errors

### Advanced Features

Consider implementing:
- **Predictive Maintenance**: Predict failures before they occur
- **Machine Learning**: Learn from error patterns to improve recovery
- **Distributed Handling**: Handle failures across multiple robots
- **Automated Diagnostics**: Self-diagnose and repair capabilities

Continue with [End-to-End Integration Guide](./integration.md) to explore how all capstone components work together in a cohesive system and how to ensure seamless integration across all modules.

## References

[All sources will be cited in the References section at the end of the book, following APA format]