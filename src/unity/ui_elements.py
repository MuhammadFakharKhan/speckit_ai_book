"""
Human-Robot Interaction UI Elements for Unity

Defines the structure and functionality of UI elements for human-robot interaction
in Unity, including control panels, visualization tools, and interaction interfaces.
"""

from enum import Enum
from typing import Dict, List, Callable, Any, Optional
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod


class UIElementType(Enum):
    """Types of UI elements for human-robot interaction"""
    CONTROL_PANEL = "control_panel"
    JOINT_SLIDER = "joint_slider"
    TELEOPERATION = "teleoperation"
    VISUALIZATION = "visualization"
    DEBUG_PANEL = "debug_panel"
    ROBOT_STATUS = "robot_status"
    SCENE_CONTROL = "scene_control"


@dataclass
class UIElement:
    """Base UI element with common properties"""
    name: str
    element_type: UIElementType
    position: tuple  # (x, y, z) in UI space
    size: tuple  # (width, height)
    visible: bool = True
    enabled: bool = True
    parent: Optional[str] = None
    callbacks: Dict[str, List[Callable]] = None

    def __post_init__(self):
        if self.callbacks is None:
            self.callbacks = {}


@dataclass
class JointSliderElement(UIElement):
    """UI element for controlling individual joints"""
    joint_name: str = ""
    min_value: float = -3.14
    max_value: float = 3.14
    current_value: float = 0.0
    step_size: float = 0.01
    unit: str = "rad"

    def update_joint_position(self, new_position: float):
        """Update the joint position with validation"""
        self.current_value = max(self.min_value, min(self.max_value, new_position))
        self._trigger_callback("value_changed", self.current_value)

    def _trigger_callback(self, event: str, data: Any = None):
        """Trigger registered callbacks for an event"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                callback(self.name, data)


@dataclass
class TeleoperationElement(UIElement):
    """UI element for teleoperation controls"""
    control_type: str = "joystick"  # joystick, keyboard, gamepad
    axes_mapping: Dict[str, str] = None  # Maps UI axes to robot commands
    sensitivity: float = 1.0
    deadzone: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.axes_mapping is None:
            self.axes_mapping = {
                "horizontal": "linear_x",
                "vertical": "linear_y",
                "rotation": "angular_z"
            }


@dataclass
class VisualizationElement(UIElement):
    """UI element for robot visualization controls"""
    view_mode: str = "perspective"  # perspective, orthographic, top, side, front
    show_trajectories: bool = True
    show_colliders: bool = False
    show_joints: bool = True
    transparency: float = 1.0  # 0.0 = transparent, 1.0 = opaque
    trail_length: int = 100  # Number of previous positions to show


@dataclass
class RobotStatusElement(UIElement):
    """UI element for displaying robot status"""
    status_fields: List[str] = None  # e.g., ["battery", "connection", "mode", "position"]
    refresh_rate: float = 1.0  # Hz

    def __post_init__(self):
        super().__post_init__()
        if self.status_fields is None:
            self.status_fields = ["battery", "connection", "mode", "position"]


@dataclass
class SceneControlElement(UIElement):
    """UI element for controlling the simulation scene"""
    simulation_speed: float = 1.0
    gravity_enabled: bool = True
    collision_detection: bool = True
    time_scale: float = 1.0
    pause_simulation: bool = False


class UIElementManager:
    """
    Manager for creating, organizing, and controlling UI elements
    """

    def __init__(self):
        self.elements: Dict[str, UIElement] = {}
        self.element_groups: Dict[str, List[str]] = {}  # Group elements by function
        self.event_handlers: Dict[str, List[Callable]] = {}

    def create_element(self, element_type: UIElementType, name: str, **kwargs) -> UIElement:
        """Create a new UI element based on type"""
        position = kwargs.get('position', (0, 0, 0))
        size = kwargs.get('size', (100, 30))
        visible = kwargs.get('visible', True)
        enabled = kwargs.get('enabled', True)

        if element_type == UIElementType.JOINT_SLIDER:
            element = JointSliderElement(
                name=name,
                element_type=element_type,
                position=position,
                size=size,
                visible=visible,
                enabled=enabled,
                joint_name=kwargs.get('joint_name', ''),
                min_value=kwargs.get('min_value', -3.14),
                max_value=kwargs.get('max_value', 3.14),
                current_value=kwargs.get('current_value', 0.0),
                step_size=kwargs.get('step_size', 0.01),
                unit=kwargs.get('unit', 'rad')
            )
        elif element_type == UIElementType.TELEOPERATION:
            element = TeleoperationElement(
                name=name,
                element_type=element_type,
                position=position,
                size=size,
                visible=visible,
                enabled=enabled,
                control_type=kwargs.get('control_type', 'joystick'),
                sensitivity=kwargs.get('sensitivity', 1.0),
                deadzone=kwargs.get('deadzone', 0.1)
            )
        elif element_type == UIElementType.VISUALIZATION:
            element = VisualizationElement(
                name=name,
                element_type=element_type,
                position=position,
                size=size,
                visible=visible,
                enabled=enabled,
                view_mode=kwargs.get('view_mode', 'perspective'),
                show_trajectories=kwargs.get('show_trajectories', True),
                show_colliders=kwargs.get('show_colliders', False),
                show_joints=kwargs.get('show_joints', True),
                transparency=kwargs.get('transparency', 1.0),
                trail_length=kwargs.get('trail_length', 100)
            )
        elif element_type == UIElementType.ROBOT_STATUS:
            element = RobotStatusElement(
                name=name,
                element_type=element_type,
                position=position,
                size=size,
                visible=visible,
                enabled=enabled,
                status_fields=kwargs.get('status_fields', ["battery", "connection", "mode"]),
                refresh_rate=kwargs.get('refresh_rate', 1.0)
            )
        elif element_type == UIElementType.SCENE_CONTROL:
            element = SceneControlElement(
                name=name,
                element_type=element_type,
                position=position,
                size=size,
                visible=visible,
                enabled=enabled,
                simulation_speed=kwargs.get('simulation_speed', 1.0),
                gravity_enabled=kwargs.get('gravity_enabled', True),
                collision_detection=kwargs.get('collision_detection', True),
                time_scale=kwargs.get('time_scale', 1.0),
                pause_simulation=kwargs.get('pause_simulation', False)
            )
        else:
            # Default UIElement for other types
            element = UIElement(
                name=name,
                element_type=element_type,
                position=position,
                size=size,
                visible=visible,
                enabled=enabled
            )

        self.elements[name] = element
        return element

    def add_element_to_group(self, group_name: str, element_name: str):
        """Add an element to a functional group"""
        if group_name not in self.element_groups:
            self.element_groups[group_name] = []
        if element_name not in self.element_groups[group_name]:
            self.element_groups[group_name].append(element_name)

    def get_elements_by_group(self, group_name: str) -> List[UIElement]:
        """Get all elements in a specific group"""
        if group_name not in self.element_groups:
            return []
        return [self.elements[name] for name in self.element_groups[group_name] if name in self.elements]

    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler for UI events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def trigger_event(self, event_type: str, data: Any = None):
        """Trigger an event and notify all registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                handler(data)

    def update_element(self, name: str, **kwargs) -> bool:
        """Update properties of an existing element"""
        if name not in self.elements:
            return False

        element = self.elements[name]
        for key, value in kwargs.items():
            if hasattr(element, key):
                setattr(element, key, value)

        # Trigger update event
        self.trigger_event("element_updated", {"name": name, "properties": kwargs})
        return True

    def get_element(self, name: str) -> Optional[UIElement]:
        """Get an element by name"""
        return self.elements.get(name)

    def get_all_elements(self) -> Dict[str, UIElement]:
        """Get all UI elements"""
        return self.elements.copy()

    def serialize_layout(self) -> str:
        """Serialize the UI layout to JSON"""
        layout_data = {
            "elements": {},
            "groups": self.element_groups,
            "events": list(self.event_handlers.keys())
        }

        for name, element in self.elements.items():
            # Convert element to dictionary, handling special types
            element_dict = {
                "type": element.element_type.value,
                "position": element.position,
                "size": element.size,
                "visible": element.visible,
                "enabled": element.enabled,
                "parent": element.parent
            }

            # Add type-specific properties
            if isinstance(element, JointSliderElement):
                element_dict.update({
                    "joint_name": element.joint_name,
                    "min_value": element.min_value,
                    "max_value": element.max_value,
                    "current_value": element.current_value,
                    "step_size": element.step_size,
                    "unit": element.unit
                })
            elif isinstance(element, TeleoperationElement):
                element_dict.update({
                    "control_type": element.control_type,
                    "axes_mapping": element.axes_mapping,
                    "sensitivity": element.sensitivity,
                    "deadzone": element.deadzone
                })
            elif isinstance(element, VisualizationElement):
                element_dict.update({
                    "view_mode": element.view_mode,
                    "show_trajectories": element.show_trajectories,
                    "show_colliders": element.show_colliders,
                    "show_joints": element.show_joints,
                    "transparency": element.transparency,
                    "trail_length": element.trail_length
                })
            elif isinstance(element, RobotStatusElement):
                element_dict.update({
                    "status_fields": element.status_fields,
                    "refresh_rate": element.refresh_rate
                })
            elif isinstance(element, SceneControlElement):
                element_dict.update({
                    "simulation_speed": element.simulation_speed,
                    "gravity_enabled": element.gravity_enabled,
                    "collision_detection": element.collision_detection,
                    "time_scale": element.time_scale,
                    "pause_simulation": element.pause_simulation
                })

            layout_data["elements"][name] = element_dict

        return json.dumps(layout_data, indent=2)

    def load_layout(self, layout_json: str):
        """Load UI layout from JSON"""
        layout_data = json.loads(layout_json)

        # Clear existing elements
        self.elements.clear()
        self.element_groups = layout_data.get("groups", {})
        self.event_handlers = {event: [] for event in layout_data.get("events", [])}

        # Recreate elements
        for name, element_data in layout_data["elements"].items():
            element_type = UIElementType(element_data["type"])
            common_props = {
                "position": tuple(element_data["position"]),
                "size": tuple(element_data["size"]),
                "visible": element_data["visible"],
                "enabled": element_data["enabled"],
                "parent": element_data.get("parent")
            }

            element = self.create_element(element_type, name, **common_props, **element_data)
            self.elements[name] = element


class InteractionUIManager:
    """
    Main manager for human-robot interaction UI
    """

    def __init__(self):
        self.ui_manager = UIElementManager()
        self.robot_joint_names: List[str] = []
        self.active_controls: Dict[str, Any] = {}

    def setup_default_interaction_ui(self, joint_names: List[str] = None):
        """Set up default interaction UI elements"""
        if joint_names:
            self.robot_joint_names = joint_names

        # Create main control panel
        control_panel = self.ui_manager.create_element(
            UIElementType.CONTROL_PANEL,
            "main_control_panel",
            position=(10, 10, 0),
            size=(300, 600),
            visible=True
        )

        # Create joint sliders for each joint
        if self.robot_joint_names:
            for i, joint_name in enumerate(self.robot_joint_names):
                slider = self.ui_manager.create_element(
                    UIElementType.JOINT_SLIDER,
                    f"slider_{joint_name}",
                    position=(20, 50 + i * 40, 0),
                    size=(260, 30),
                    joint_name=joint_name,
                    min_value=-2.0,
                    max_value=2.0,
                    current_value=0.0
                )
                self.ui_manager.add_element_to_group("joint_controls", f"slider_{joint_name}")

        # Create teleoperation controls
        teleop = self.ui_manager.create_element(
            UIElementType.TELEOPERATION,
            "teleop_joystick",
            position=(320, 10, 0),
            size=(200, 200),
            control_type="joystick"
        )
        self.ui_manager.add_element_to_group("teleoperation", "teleop_joystick")

        # Create visualization controls
        viz = self.ui_manager.create_element(
            UIElementType.VISUALIZATION,
            "visualization_panel",
            position=(320, 220, 0),
            size=(200, 150),
            view_mode="perspective"
        )
        self.ui_manager.add_element_to_group("visualization", "visualization_panel")

        # Create robot status display
        status = self.ui_manager.create_element(
            UIElementType.ROBOT_STATUS,
            "robot_status",
            position=(320, 380, 0),
            size=(200, 100)
        )
        self.ui_manager.add_element_to_group("status", "robot_status")

        # Create scene controls
        scene = self.ui_manager.create_element(
            UIElementType.SCENE_CONTROL,
            "scene_controls",
            position=(530, 10, 0),
            size=(200, 150)
        )
        self.ui_manager.add_element_to_group("scene", "scene_controls")

        print(f"Created {len(self.ui_manager.get_all_elements())} UI elements for interaction")

    def update_joint_slider(self, joint_name: str, position: float):
        """Update a joint slider with new position"""
        element_name = f"slider_{joint_name}"
        if element_name in self.ui_manager.elements:
            element = self.ui_manager.elements[element_name]
            if isinstance(element, JointSliderElement):
                element.update_joint_position(position)

    def get_ui_layout_json(self) -> str:
        """Get the UI layout as JSON"""
        return self.ui_manager.serialize_layout()

    def register_control_callback(self, control_name: str, callback: Callable):
        """Register a callback for a specific control"""
        self.active_controls[control_name] = callback

    def handle_control_input(self, control_name: str, value: Any):
        """Handle input from a control element"""
        if control_name in self.active_controls:
            self.active_controls[control_name](value)


# Example usage
if __name__ == "__main__":
    # Create interaction UI manager
    ui_manager = InteractionUIManager()

    # Set up with example joint names
    joint_names = ["hip_joint", "knee_joint", "ankle_joint", "shoulder_joint", "elbow_joint", "wrist_joint"]
    ui_manager.setup_default_interaction_ui(joint_names)

    # Print the UI layout
    print("UI Layout:")
    print(ui_manager.get_ui_layout_json())

    # Example of updating a joint slider
    ui_manager.update_joint_slider("hip_joint", 1.57)
    print("\nUpdated hip joint slider to 1.57 radians")