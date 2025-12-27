"""
Robot State Synchronizer for Gazebo-Unity Integration

Manages synchronization of robot states between Gazebo simulation and Unity visualization.
Handles timing, data transformation, and state consistency between the two systems.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import math


class SyncMode(Enum):
    """Synchronization mode for Gazebo-Unity communication"""
    REAL_TIME = "real_time"  # Synchronize in real-time
    FIXED_RATE = "fixed_rate"  # Synchronize at fixed intervals
    INTERPOLATED = "interpolated"  # Interpolate between states


@dataclass
class JointState:
    """Represents the state of a robot joint"""
    name: str
    position: float
    velocity: float
    effort: float
    timestamp: float


@dataclass
class Transform:
    """Represents a 3D transformation (position + rotation)"""
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # Quaternion (x, y, z, w)
    timestamp: float


@dataclass
class RobotState:
    """Complete robot state representation"""
    joint_states: Dict[str, JointState]
    transforms: Dict[str, Transform]
    velocities: Dict[str, Tuple[float, float, float]]  # Linear and angular velocities
    timestamp: float
    simulation_time: float


class GazeboUnityStateSynchronizer:
    """
    Synchronizes robot states between Gazebo simulation and Unity visualization
    """

    def __init__(self, sync_mode: SyncMode = SyncMode.REAL_TIME, update_rate: float = 30.0):
        self.sync_mode = sync_mode
        self.update_rate = update_rate  # Hz
        self.update_interval = 1.0 / update_rate if update_rate > 0 else 0.01

        # State storage
        self.gazebo_state: Optional[RobotState] = None
        self.unity_state: Optional[RobotState] = None
        self.state_history: List[RobotState] = []

        # Threading and synchronization
        self.lock = threading.RLock()
        self.running = False
        self.sync_thread: Optional[threading.Thread] = None

        # Timing
        self.last_sync_time = 0.0
        self.simulation_speed = 1.0  # 1.0 = real-time, 0.5 = half speed, etc.

        # Interpolation settings
        self.interpolation_enabled = sync_mode == SyncMode.INTERPOLATED
        self.max_history_size = 100

        # Callbacks
        self.state_update_callbacks = []

    def start_synchronization(self):
        """Start the synchronization process"""
        if self.running:
            return

        self.running = True
        self.sync_thread = threading.Thread(target=self._synchronization_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()

        print(f"Started Gazebo-Unity synchronization at {self.update_rate} Hz")

    def stop_synchronization(self):
        """Stop the synchronization process"""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
        print("Stopped Gazebo-Unity synchronization")

    def _synchronization_loop(self):
        """Main synchronization loop running in separate thread"""
        while self.running:
            start_time = time.time()

            # Perform synchronization based on mode
            if self.sync_mode == SyncMode.REAL_TIME:
                self._sync_real_time()
            elif self.sync_mode == SyncMode.FIXED_RATE:
                self._sync_fixed_rate()
            elif self.sync_mode == SyncMode.INTERPOLATED:
                self._sync_interpolated()

            # Maintain update rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.update_interval - elapsed)
            time.sleep(sleep_time)

    def _sync_real_time(self):
        """Synchronize states in real-time"""
        with self.lock:
            if self.gazebo_state:
                # Directly pass gazebo state to unity
                self.unity_state = self._transform_state_for_unity(self.gazebo_state)
                self._trigger_callbacks(self.unity_state)

    def _sync_fixed_rate(self):
        """Synchronize states at fixed intervals"""
        current_time = time.time()
        if current_time - self.last_sync_time >= self.update_interval:
            with self.lock:
                if self.gazebo_state:
                    self.unity_state = self._transform_state_for_unity(self.gazebo_state)
                    self._trigger_callbacks(self.unity_state)
                    self.last_sync_time = current_time

    def _sync_interpolated(self):
        """Synchronize states with interpolation between samples"""
        with self.lock:
            if len(self.state_history) >= 2 and self.gazebo_state:
                # Interpolate between the last two states
                interpolated_state = self._interpolate_states(
                    self.state_history[-2],
                    self.state_history[-1],
                    self.gazebo_state.timestamp
                )
                self.unity_state = self._transform_state_for_unity(interpolated_state)
                self._trigger_callbacks(self.unity_state)

    def _interpolate_states(self, state1: RobotState, state2: RobotState, target_time: float) -> RobotState:
        """Interpolate between two robot states"""
        # Calculate interpolation factor (0.0 to 1.0)
        time_diff = state2.timestamp - state1.timestamp
        if time_diff <= 0:
            return state1

        factor = min(1.0, max(0.0, (target_time - state1.timestamp) / time_diff))

        # Interpolate joint positions
        interpolated_joints = {}
        for joint_name in state1.joint_states:
            if joint_name in state2.joint_states:
                joint1 = state1.joint_states[joint_name]
                joint2 = state2.joint_states[joint_name]

                # Linear interpolation for position
                pos = joint1.position + factor * (joint2.position - joint1.position)
                vel = joint1.velocity + factor * (joint2.velocity - joint1.velocity)
                eff = joint1.effort + factor * (joint2.effort - joint1.effort)

                interpolated_joints[joint_name] = JointState(
                    name=joint_name,
                    position=pos,
                    velocity=vel,
                    effort=eff,
                    timestamp=target_time
                )

        # Interpolate transforms using spherical linear interpolation (slerp) for rotations
        interpolated_transforms = {}
        for frame_name in state1.transforms:
            if frame_name in state2.transforms:
                transform1 = state1.transforms[frame_name]
                transform2 = state2.transforms[frame_name]

                # Linear interpolation for position
                pos_x = transform1.position[0] + factor * (transform2.position[0] - transform1.position[0])
                pos_y = transform1.position[1] + factor * (transform2.position[1] - transform1.position[1])
                pos_z = transform1.position[2] + factor * (transform2.position[2] - transform1.position[2])

                # Spherical linear interpolation for rotation (slerp)
                rot = self._slerp(transform1.rotation, transform2.rotation, factor)

                interpolated_transforms[frame_name] = Transform(
                    position=(pos_x, pos_y, pos_z),
                    rotation=rot,
                    timestamp=target_time
                )

        return RobotState(
            joint_states=interpolated_joints,
            transforms=interpolated_transforms,
            velocities=self._interpolate_velocities(state1.velocities, state2.velocities, factor),
            timestamp=target_time,
            simulation_time=state1.simulation_time + factor * (state2.simulation_time - state1.simulation_time)
        )

    def _slerp(self, quat1: Tuple[float, float, float, float], quat2: Tuple[float, float, float, float], factor: float) -> Tuple[float, float, float, float]:
        """Spherical linear interpolation between two quaternions"""
        # Convert to local variables for readability
        x1, y1, z1, w1 = quat1
        x2, y2, z2, w2 = quat2

        # Calculate dot product
        dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2

        # If dot product is negative, negate one quaternion to take shorter path
        if dot < 0.0:
            x2, y2, z2, w2 = -x2, -y2, -z2, -w2
            dot = -dot

        # Perform linear interpolation if quaternions are very close
        if dot > 0.9995:
            x = x1 + factor * (x2 - x1)
            y = y1 + factor * (y2 - y1)
            z = z1 + factor * (z2 - z1)
            w = w1 + factor * (w2 - w1)
        else:
            # Calculate angle between quaternions
            theta_0 = math.acos(abs(dot))
            sin_theta_0 = math.sin(theta_0)
            theta = theta_0 * factor
            sin_theta = math.sin(theta)

            # Interpolate
            s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0

            x = s0 * x1 + s1 * x2
            y = s0 * y1 + s1 * y2
            z = s0 * z1 + s1 * z2
            w = s0 * w1 + s1 * w2

        # Normalize result
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm > 0:
            return (x / norm, y / norm, z / norm, w / norm)

        return (0, 0, 0, 1)

    def _interpolate_velocities(self, vel1: Dict[str, Tuple[float, float, float]],
                              vel2: Dict[str, Tuple[float, float, float]],
                              factor: float) -> Dict[str, Tuple[float, float, float]]:
        """Interpolate between velocity dictionaries"""
        interpolated = {}
        for name in vel1:
            if name in vel2:
                v1 = vel1[name]
                v2 = vel2[name]
                interpolated[name] = tuple(
                    v1[i] + factor * (v2[i] - v1[i]) for i in range(3)
                )
        return interpolated

    def _transform_state_for_unity(self, state: RobotState) -> RobotState:
        """Transform robot state for Unity coordinate system"""
        # Unity uses a left-handed coordinate system (Z forward, Y up)
        # While ROS/Gazebo uses right-handed (X forward, Z up)
        # Transform accordingly

        transformed_joints = {}
        for name, joint in state.joint_states.items():
            transformed_joints[name] = JointState(
                name=joint.name,
                position=joint.position,
                velocity=joint.velocity,
                effort=joint.effort,
                timestamp=joint.timestamp
            )

        transformed_transforms = {}
        for name, transform in state.transforms.items():
            # Transform from Gazebo (right-handed) to Unity (left-handed)
            # ROS: X forward, Y left, Z up
            # Unity: X right, Y up, Z forward
            unity_pos = (
                transform.position[1],  # Y -> X
                transform.position[2],  # Z -> Y
                transform.position[0]   # X -> Z
            )

            # For rotation, we need to transform the quaternion
            # This is a simplified transformation - in practice, you'd need to apply
            # the full coordinate system transformation
            unity_rot = transform.rotation  # Placeholder - would need proper transformation

            transformed_transforms[name] = Transform(
                position=unity_pos,
                rotation=unity_rot,
                timestamp=transform.timestamp
            )

        return RobotState(
            joint_states=transformed_joints,
            transforms=transformed_transforms,
            velocities=state.velocities,
            timestamp=state.timestamp,
            simulation_time=state.simulation_time
        )

    def update_gazebo_state(self, state: RobotState):
        """Update the Gazebo state, which will be synchronized to Unity"""
        with self.lock:
            self.gazebo_state = state

            # Add to history for interpolation
            if self.interpolation_enabled:
                self.state_history.append(state)
                if len(self.state_history) > self.max_history_size:
                    self.state_history.pop(0)

    def get_unity_state(self) -> Optional[RobotState]:
        """Get the current Unity state"""
        with self.lock:
            return self.unity_state

    def register_state_callback(self, callback: callable):
        """Register a callback to be called when state is updated"""
        self.state_update_callbacks.append(callback)

    def _trigger_callbacks(self, state: RobotState):
        """Trigger all registered callbacks with the new state"""
        for callback in self.state_update_callbacks:
            try:
                callback(state)
            except Exception as e:
                print(f"Error in state callback: {e}")

    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status information"""
        with self.lock:
            return {
                'running': self.running,
                'sync_mode': self.sync_mode.value,
                'update_rate': self.update_rate,
                'gazebo_state_available': self.gazebo_state is not None,
                'unity_state_available': self.unity_state is not None,
                'state_history_size': len(self.state_history),
                'last_sync_time': self.last_sync_time
            }

    def set_simulation_speed(self, speed: float):
        """Set the simulation speed relative to real-time"""
        self.simulation_speed = max(0.1, min(10.0, speed))  # Clamp between 0.1x and 10x


class StateSynchronizerManager:
    """
    Manager for multiple state synchronizers
    """
    def __init__(self):
        self.synchronizers: Dict[str, GazeboUnityStateSynchronizer] = {}

    def create_synchronizer(self, robot_name: str, sync_mode: SyncMode = SyncMode.REAL_TIME, update_rate: float = 30.0) -> GazeboUnityStateSynchronizer:
        """Create a new state synchronizer for a robot"""
        synchronizer = GazeboUnityStateSynchronizer(sync_mode, update_rate)
        self.synchronizers[robot_name] = synchronizer
        return synchronizer

    def get_synchronizer(self, robot_name: str) -> Optional[GazeboUnityStateSynchronizer]:
        """Get a state synchronizer by robot name"""
        return self.synchronizers.get(robot_name)

    def start_all_synchronizers(self):
        """Start all synchronizers"""
        for synchronizer in self.synchronizers.values():
            synchronizer.start_synchronization()

    def stop_all_synchronizers(self):
        """Stop all synchronizers"""
        for synchronizer in self.synchronizers.values():
            synchronizer.stop_synchronization()

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all synchronizers"""
        statuses = {}
        for name, synchronizer in self.synchronizers.items():
            statuses[name] = synchronizer.get_sync_status()
        return statuses


# Example usage
if __name__ == "__main__":
    # Create synchronizer manager
    manager = StateSynchronizerManager()

    # Create synchronizer for humanoid robot
    humanoid_sync = manager.create_synchronizer("humanoid", SyncMode.INTERPOLATED, 60.0)

    # Register a callback for state updates
    def on_state_update(state: RobotState):
        print(f"State updated at {state.timestamp}, joints: {len(state.joint_states)}")

    humanoid_sync.register_state_callback(on_state_update)

    # Start synchronization
    manager.start_all_synchronizers()

    # Simulate receiving state from Gazebo
    import random

    for i in range(100):
        # Create a mock robot state
        joint_states = {}
        for joint_name in ['hip_joint', 'knee_joint', 'ankle_joint', 'shoulder_joint', 'elbow_joint', 'wrist_joint']:
            joint_states[joint_name] = JointState(
                name=joint_name,
                position=random.uniform(-1.0, 1.0),
                velocity=random.uniform(-0.1, 0.1),
                effort=random.uniform(-10.0, 10.0),
                timestamp=time.time()
            )

        transforms = {
            'base_link': Transform(
                position=(i * 0.01, 0.0, 0.5),
                rotation=(0.0, 0.0, 0.0, 1.0),
                timestamp=time.time()
            )
        }

        mock_state = RobotState(
            joint_states=joint_states,
            transforms=transforms,
            velocities={'base_link': (0.1, 0.0, 0.0)},
            timestamp=time.time(),
            simulation_time=i * 0.01
        )

        # Update the synchronizer with the mock state
        humanoid_sync.update_gazebo_state(mock_state)

        time.sleep(0.05)

    # Stop synchronization
    manager.stop_all_synchronizers()