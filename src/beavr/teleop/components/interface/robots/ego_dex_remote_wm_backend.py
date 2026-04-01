from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
try:
    import zmq
except ImportError:  # pragma: no cover - hand visualization is optional
    zmq = None

from beavr.teleop.configs.constants import robots

from .ego_dex_wm_client import WMClient

logger = logging.getLogger(__name__)

_FLIP_Z_4X4 = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)
_LEFT_STEP_LOCAL_POST_Z_FLIP_4X4 = np.array(
    [
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_RIGHT_STEP_LOCAL_POST_Z_FLIP_4X4 = np.array(
    [
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_STEP_LOCAL_POST_Z_FLIP_BY_SIDE = {
    robots.LEFT: _LEFT_STEP_LOCAL_POST_Z_FLIP_4X4,
    robots.RIGHT: _RIGHT_STEP_LOCAL_POST_Z_FLIP_4X4,
}
_DROP_DISTANCE_JOINTS = frozenset({"palm"})
DEFAULT_STEP_DISTANCE_THRESHOLD = 0.0018351837001815872
DEFAULT_STEP_HAND_SCALING_ENABLE = True
DEFAULT_STEP_HAND_SCALE_MIN = 0.0
DEFAULT_STEP_HAND_SCALE_MAX = 10.0
HAND_VISUALIZATION_DISPLAY_OFFSET_Y_M = 0.10
HAND_VISUALIZATION_LIVE_SIDE_COLORS = {
    robots.LEFT: (1.0, 0.22, 0.22, 0.98),
    robots.RIGHT: (1.0, 0.22, 0.22, 0.98),
}
HAND_VISUALIZATION_REFERENCE_SIDE_COLORS = {
    robots.LEFT: (1.0, 0.86, 0.18, 0.98),
    robots.RIGHT: (0.24, 0.56, 1.0, 0.98),
}
_MIN_VALID_BONE_LENGTH = 1e-6
_CONTROL_PARENT_BY_JOINT = {
    "thumb_metacarpal": "wrist",
    "thumb_proximal": "thumb_metacarpal",
    "thumb_distal": "thumb_proximal",
    "thumb_tip": "thumb_distal",
    "index_metacarpal": "wrist",
    "index_proximal": "index_metacarpal",
    "index_intermediate": "index_proximal",
    "index_distal": "index_intermediate",
    "index_tip": "index_distal",
    "middle_metacarpal": "wrist",
    "middle_proximal": "middle_metacarpal",
    "middle_intermediate": "middle_proximal",
    "middle_distal": "middle_intermediate",
    "middle_tip": "middle_distal",
    "ring_metacarpal": "wrist",
    "ring_proximal": "ring_metacarpal",
    "ring_intermediate": "ring_proximal",
    "ring_distal": "ring_intermediate",
    "ring_tip": "ring_distal",
    "little_metacarpal": "wrist",
    "little_proximal": "little_metacarpal",
    "little_intermediate": "little_proximal",
    "little_distal": "little_intermediate",
    "little_tip": "little_distal",
}
_CONTROL_JOINT_ORDER = ("wrist",) + tuple(_CONTROL_PARENT_BY_JOINT.keys())
_CONTROL_EDGES = tuple((parent, child) for child, parent in _CONTROL_PARENT_BY_JOINT.items())


def _flip_keypoints_z_array(keypoints_xyz: np.ndarray) -> np.ndarray:
    flipped = np.asarray(keypoints_xyz, dtype=np.float32).copy()
    if flipped.ndim == 2 and flipped.shape[1] == 3:
        flipped[:, 2] *= -1.0
    return flipped


def _flip_transform_z_array(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float32)
    if matrix.shape != (4, 4):
        return matrix
    return _FLIP_Z_4X4 @ matrix @ _FLIP_Z_4X4


def _apply_step_local_post_z_flip_transform_array(side: str, transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float32)
    if matrix.shape != (4, 4):
        return matrix
    local_post_transform = _STEP_LOCAL_POST_Z_FLIP_BY_SIDE.get(side)
    if local_post_transform is None:
        return matrix
    return matrix @ local_post_transform


def _ordered_joint_names(
    hand_payload: Mapping[str, Any],
    count: int,
    *,
    fallback_names: Optional[Sequence[Any]] = None,
) -> list[str]:
    joint_order = hand_payload.get("joint_order")
    if isinstance(joint_order, Sequence) and not isinstance(joint_order, (str, bytes)):
        return [str(joint_name) for joint_name in joint_order[:count]]
    if fallback_names is not None:
        return [str(joint_name) for joint_name in list(fallback_names)[:count]]
    return [str(index) for index in range(count)]


def _extract_hand_points_for_display(
    hand_payload: Mapping[str, Any],
    *,
    payload_space: str,
) -> list[tuple[str, np.ndarray]]:
    joint_transforms_world = hand_payload.get("joint_transforms_world")
    if isinstance(joint_transforms_world, Mapping):
        joint_names = _ordered_joint_names(
            hand_payload,
            len(joint_transforms_world),
            fallback_names=joint_transforms_world.keys(),
        )
        points: list[tuple[str, np.ndarray]] = []
        for joint_name in joint_names:
            raw_transform = joint_transforms_world.get(joint_name)
            if raw_transform is None:
                continue
            transform = np.asarray(raw_transform, dtype=np.float32)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                continue
            if payload_space == "step":
                transform = _flip_transform_z_array(transform)
            points.append((joint_name, transform[:3, 3].astype(np.float32)))
        if points:
            return points

    keypoints_xyz = hand_payload.get("keypoints_xyz")
    if keypoints_xyz is None:
        return []
    keypoints = np.asarray(keypoints_xyz, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        return []
    if payload_space == "step":
        keypoints = _flip_keypoints_z_array(keypoints)
    joint_names = _ordered_joint_names(hand_payload, keypoints.shape[0])
    return [
        (joint_name, point.astype(np.float32))
        for joint_name, point in zip(joint_names, keypoints, strict=False)
        if np.isfinite(point).all()
    ]


def _extract_step_space_positions(
    hand_payload: Mapping[str, Any],
    *,
    drop_palm: bool,
) -> dict[str, np.ndarray]:
    joint_transforms_world = hand_payload.get("joint_transforms_world")
    if isinstance(joint_transforms_world, Mapping):
        joint_names = _ordered_joint_names(
            hand_payload,
            len(joint_transforms_world),
            fallback_names=joint_transforms_world.keys(),
        )
        positions: dict[str, np.ndarray] = {}
        for joint_name in joint_names:
            if drop_palm and joint_name in _DROP_DISTANCE_JOINTS:
                continue
            raw_transform = joint_transforms_world.get(joint_name)
            if raw_transform is None:
                continue
            transform = np.asarray(raw_transform, dtype=np.float32)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                continue
            positions[joint_name] = transform[:3, 3].astype(np.float32)
        if positions:
            return positions

    keypoints_xyz = hand_payload.get("keypoints_xyz")
    if keypoints_xyz is None:
        return {}
    keypoints = np.asarray(keypoints_xyz, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        return {}
    joint_names = _ordered_joint_names(hand_payload, keypoints.shape[0])
    return {
        joint_name: point.astype(np.float32)
        for joint_name, point in zip(joint_names, keypoints, strict=False)
        if np.isfinite(point).all() and not (drop_palm and joint_name in _DROP_DISTANCE_JOINTS)
    }


def _copy_position_dict(positions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        str(joint_name): np.asarray(position, dtype=np.float32).copy()
        for joint_name, position in positions.items()
        if np.asarray(position, dtype=np.float32).shape == (3,)
        and np.isfinite(np.asarray(position, dtype=np.float32)).all()
    }


def _set_hand_payload_positions(
    hand_payload: Mapping[str, Any],
    positions_by_joint: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(hand_payload))
    positions = _copy_position_dict(positions_by_joint)

    joint_transforms_world = payload.get("joint_transforms_world")
    if isinstance(joint_transforms_world, Mapping):
        updated_transforms: dict[str, Any] = {}
        for joint_name, raw_transform in joint_transforms_world.items():
            transform = np.asarray(raw_transform, dtype=np.float32)
            if transform.shape == (4, 4) and np.isfinite(transform).all():
                updated_transform = transform.copy()
                if joint_name in positions:
                    updated_transform[:3, 3] = positions[joint_name]
                updated_transforms[str(joint_name)] = updated_transform.tolist()
            else:
                updated_transforms[str(joint_name)] = raw_transform
        wrist_transform = updated_transforms.get("wrist")
        if wrist_transform is not None:
            updated_transforms["palm"] = copy.deepcopy(wrist_transform)
        payload["joint_transforms_world"] = updated_transforms

    keypoints_xyz = payload.get("keypoints_xyz")
    if keypoints_xyz is not None:
        keypoints = np.asarray(keypoints_xyz, dtype=np.float32)
        if keypoints.ndim == 2 and keypoints.shape[1] == 3:
            updated_keypoints = keypoints.copy()
            joint_names = _ordered_joint_names(payload, updated_keypoints.shape[0])
            if "wrist" in positions:
                positions = dict(positions)
                positions["palm"] = positions["wrist"].copy()
            for index, joint_name in enumerate(joint_names):
                position = positions.get(joint_name)
                if position is None:
                    continue
                updated_keypoints[index] = position
            payload["keypoints_xyz"] = updated_keypoints.tolist()

    return payload


def _normalize_step_transport_hand_payload(hand_payload: Mapping[str, Any]) -> dict[str, Any]:
    positions = _extract_step_space_positions(hand_payload, drop_palm=False)
    wrist_position = positions.get("wrist")
    if wrist_position is None:
        return copy.deepcopy(dict(hand_payload))
    positions["palm"] = wrist_position.copy()
    return _set_hand_payload_positions(hand_payload, positions)


def _extract_control_step_positions(hand_payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    positions = _extract_step_space_positions(hand_payload, drop_palm=False)
    return {
        joint_name: positions[joint_name]
        for joint_name in _CONTROL_JOINT_ORDER
        if joint_name in positions
    }


def _make_hand_visualization_point(
    position: np.ndarray,
    color: tuple[float, float, float, float],
    *,
    point_size: float,
) -> dict[str, Any]:
    display_position = np.asarray(position, dtype=np.float32).copy()
    display_position[1] += HAND_VISUALIZATION_DISPLAY_OFFSET_Y_M
    return {
        "position": {
            "x": float(display_position[0]),
            "y": float(display_position[1]),
            "z": float(display_position[2]),
        },
        "color": {
            "r": float(color[0]),
            "g": float(color[1]),
            "b": float(color[2]),
            "a": float(color[3]),
        },
        "size": float(point_size),
    }


class TeleopHandVisualizationPublisher:
    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = 15102,
        fps: float = 15.0,
        point_size: float = 0.015,
    ):
        self._bind_host = str(bind_host)
        self._port = int(port)
        self._min_publish_period_s = 1.0 / max(float(fps), 1.0)
        self._point_size = max(float(point_size), 1e-4)
        self._sequence = 0
        self._last_publish_wall_time_s = 0.0
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._lock = threading.Lock()

    @property
    def bind_address(self) -> str:
        return f"tcp://{self._bind_host}:{self._port}"

    def start(self) -> None:
        if self._socket is not None:
            return
        if zmq is None:
            raise RuntimeError("pyzmq is required for teleop hand visualization publishing.")
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 8)
        self._socket.bind(self.bind_address)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def publish(
        self,
        *,
        live_step_hands: Mapping[str, Optional[Mapping[str, Any]]],
        reference_step_hands: Mapping[str, Optional[Mapping[str, Any]]],
        show_reference: bool,
    ) -> None:
        if self._socket is None:
            return
        now_s = time.time()
        if now_s - self._last_publish_wall_time_s < self._min_publish_period_s:
            return

        points: list[dict[str, Any]] = []
        for side in (robots.LEFT, robots.RIGHT):
            live_hand = live_step_hands.get(side)
            if isinstance(live_hand, Mapping):
                for joint_name, position in _extract_hand_points_for_display(
                    live_hand,
                    payload_space="step",
                ):
                    points.append(
                        _make_hand_visualization_point(
                            position,
                            HAND_VISUALIZATION_LIVE_SIDE_COLORS[side],
                            point_size=self._point_size if joint_name not in {"wrist", "palm"} else self._point_size * 1.2,
                        )
                    )

        if show_reference:
            for side in (robots.LEFT, robots.RIGHT):
                reference_hand = reference_step_hands.get(side)
                if not isinstance(reference_hand, Mapping):
                    continue
                for joint_name, position in _extract_hand_points_for_display(
                    reference_hand,
                    payload_space="step",
                ):
                    points.append(
                        _make_hand_visualization_point(
                            position,
                            HAND_VISUALIZATION_REFERENCE_SIDE_COLORS[side],
                            point_size=self._point_size if joint_name not in {"wrist", "palm"} else self._point_size * 1.2,
                        )
                    )

        if not points:
            return

        payload = {
            "frame": "tracking_space",
            "sequence": self._sequence,
            "points": points,
        }
        with self._lock:
            self._socket.send_string(json.dumps(payload, separators=(",", ":")))
            self._sequence += 1
            self._last_publish_wall_time_s = now_s


@dataclass(frozen=True)
class RobotActionCommand:
    hand_side: str
    timestamp_s: Optional[float]
    joint_positions_rad: np.ndarray


class RemoteEgoDexWMBackend:
    """Teleop-side backend that mediates between EgoDexRobot and WMClient."""

    def __init__(
        self,
        wm_client: WMClient,
        dof: int = 16,
        heartbeat_hz: float = 2.0,
        bootstrap_reset: bool = True,
        reset_frame_dir: Optional[str] = None,
        step_distance_threshold: float = DEFAULT_STEP_DISTANCE_THRESHOLD,
        step_hand_scaling_enable: bool = DEFAULT_STEP_HAND_SCALING_ENABLE,
        step_hand_scale_min: float = DEFAULT_STEP_HAND_SCALE_MIN,
        step_hand_scale_max: float = DEFAULT_STEP_HAND_SCALE_MAX,
        hand_visualization_bind_host: Optional[str] = None,
        hand_visualization_port: Optional[int] = None,
        hand_visualization_fps: float = 15.0,
        hand_visualization_point_size: float = 0.015,
    ):
        self._wm_client = wm_client
        self._dof = int(dof)
        self._default_joint_state = np.zeros(self._dof, dtype=np.float32)
        self._step_period_s = 1.0 / max(float(heartbeat_hz), 1.0)
        self._latest_obs_frame: Optional[np.ndarray] = None
        self._latest_state: Mapping[str, Any] = {}
        self._last_event_record: Optional[dict] = None
        self._last_obs_wall_time_s = 0.0
        self._last_request_wall_time_s = 0.0
        self._last_action_by_side: dict[str, dict[str, Any]] = {}
        self._latest_input_by_side: dict[str, dict[str, Any]] = {
            robots.LEFT: {},
            robots.RIGHT: {},
        }
        self._step_distance_threshold = max(float(step_distance_threshold), 0.0)
        self._step_hand_scaling_enable = bool(step_hand_scaling_enable)
        self._step_hand_scale_min = float(step_hand_scale_min)
        self._step_hand_scale_max = float(step_hand_scale_max)
        if self._step_hand_scale_max < self._step_hand_scale_min:
            self._step_hand_scale_min, self._step_hand_scale_max = (
                self._step_hand_scale_max,
                self._step_hand_scale_min,
            )
        self._accepted_step_hands: dict[str, Optional[dict[str, Any]]] = {
            robots.LEFT: None,
            robots.RIGHT: None,
        }
        self._reset_reference_step_hands: dict[str, Optional[dict[str, Any]]] = {
            robots.LEFT: None,
            robots.RIGHT: None,
        }
        self._step_hand_bone_scale_by_side: dict[str, dict[tuple[str, str], float]] = {
            robots.LEFT: {},
            robots.RIGHT: {},
        }
        self._step_hand_scaling_calibrated_by_side: dict[str, bool] = {
            robots.LEFT: False,
            robots.RIGHT: False,
        }
        self._pending_reset_step_snapshot: Optional[dict[str, Any]] = None
        self._last_gate_value: Optional[float] = None
        self._last_blocked: bool = False
        self._reset_frame_dir = str(reset_frame_dir) if reset_frame_dir else None
        if self._reset_frame_dir is not None:
            os.makedirs(self._reset_frame_dir, exist_ok=True)
        self._hand_visualization_publisher: Optional[TeleopHandVisualizationPublisher] = None
        if hand_visualization_port is not None:
            self._hand_visualization_publisher = TeleopHandVisualizationPublisher(
                bind_host=hand_visualization_bind_host or "0.0.0.0",
                port=hand_visualization_port,
                fps=hand_visualization_fps,
                point_size=hand_visualization_point_size,
            )
            self._hand_visualization_publisher.start()

        if bootstrap_reset:
            try:
                self.reset(new=True)
            except Exception:
                logger.exception("Initial WM reset failed")

    @staticmethod
    def _decode_jpg(jpg_bytes: bytes) -> Optional[np.ndarray]:
        if not jpg_bytes:
            return None
        encoded = np.frombuffer(jpg_bytes, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _extract_joint_state(self, side: str) -> Optional[np.ndarray]:
        if not self._latest_state:
            return None

        for key in (side, f"{side}_joint_state", f"{side}_joint_states"):
            value = self._latest_state.get(key)
            if value is not None:
                return np.asarray(value, dtype=np.float32).reshape(-1)

        value = self._latest_state.get("joint_state", self._latest_state.get("joint_states"))
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _request_step(self, payload: Mapping[str, Any], event_name: str) -> None:
        obs_jpg = self._wm_client.wm_step(payload)
        frame = self._decode_jpg(obs_jpg)
        if frame is None:
            logger.warning("WM response could not be decoded as a JPG frame")
            return

        now_s = time.time()
        self._latest_obs_frame = frame
        self._last_obs_wall_time_s = now_s
        self._latest_state = dict(self._latest_state)

        hands = payload.get("hands") if isinstance(payload, Mapping) else None
        if isinstance(hands, Mapping):
            for side, hand_payload in hands.items():
                if not isinstance(hand_payload, Mapping):
                    continue
                joint_positions = hand_payload.get("joint_positions_rad")
                if joint_positions is not None:
                    self._latest_state[f"{side}_joint_state"] = joint_positions
            hands_with_keypoints = sorted(
                side
                for side, hand_payload in hands.items()
                if isinstance(hand_payload, Mapping) and hand_payload.get("keypoints_xyz") is not None
            )
            hands_with_transforms = sorted(
                side
                for side, hand_payload in hands.items()
                if isinstance(hand_payload, Mapping)
                and hand_payload.get("joint_transforms_world") is not None
            )
            hands_with_actions = sorted(
                side
                for side, hand_payload in hands.items()
                if isinstance(hand_payload, Mapping) and hand_payload.get("joint_positions_rad") is not None
            )
        else:
            hands_with_keypoints = []
            hands_with_transforms = []
            hands_with_actions = []

        self._last_event_record = {
            "event": event_name,
            "received_at_s": now_s,
            "timestamp_s": payload.get("timestamp_s"),
            "wm_step_schema": "two_hand_snapshot",
            "hands_with_keypoints": hands_with_keypoints,
            "hands_with_joint_transforms_world": hands_with_transforms,
            "hands_with_actions": hands_with_actions,
            "world_frame": payload.get("world_frame"),
            "action_dof_by_side": {
                side: len(hand_payload.get("joint_positions_rad"))
                for side, hand_payload in (hands.items() if isinstance(hands, Mapping) else [])
                if isinstance(hand_payload, Mapping)
                and isinstance(hand_payload.get("joint_positions_rad"), list)
            },
        }

    @staticmethod
    def _copy_joint_transforms_world(joint_transforms_world: Any) -> Optional[dict[str, Any]]:
        if joint_transforms_world is None:
            return None
        if not isinstance(joint_transforms_world, Mapping):
            return None
        return {
            str(joint_name): transform
            for joint_name, transform in joint_transforms_world.items()
        }

    @staticmethod
    def _flip_keypoints_z(keypoints_xyz: Any) -> Any:
        if keypoints_xyz is None:
            return None
        keypoints = np.asarray(keypoints_xyz, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] != 3:
            return keypoints_xyz
        return _flip_keypoints_z_array(keypoints).tolist()

    @staticmethod
    def _flip_transform_z(transform: Any) -> Any:
        matrix = np.asarray(transform, dtype=np.float32)
        if matrix.shape != (4, 4):
            return transform
        return _flip_transform_z_array(matrix).tolist()

    def _flip_joint_transforms_world_z(self, joint_transforms_world: Any) -> Any:
        if joint_transforms_world is None or not isinstance(joint_transforms_world, Mapping):
            return joint_transforms_world
        return {
            str(joint_name): self._flip_transform_z(transform)
            for joint_name, transform in joint_transforms_world.items()
        }

    def _apply_step_local_post_z_flip_joint_transforms_world(
        self,
        side: str,
        joint_transforms_world: Any,
    ) -> Any:
        if joint_transforms_world is None or not isinstance(joint_transforms_world, Mapping):
            return joint_transforms_world
        return {
            str(joint_name): _apply_step_local_post_z_flip_transform_array(side, transform).tolist()
            for joint_name, transform in joint_transforms_world.items()
        }

    def _normalize_step_hand_payload(self, hand_payload: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
        if not isinstance(hand_payload, Mapping):
            return None
        return _normalize_step_transport_hand_payload(hand_payload)

    def _normalize_step_hands(
        self,
        hands: Mapping[str, Optional[Mapping[str, Any]]],
    ) -> dict[str, Optional[dict[str, Any]]]:
        return {
            side: self._normalize_step_hand_payload(hands.get(side))
            for side in (robots.LEFT, robots.RIGHT)
        }

    def _calibrate_step_hand_scaling_for_side(
        self,
        side: str,
        *,
        live_hand: Mapping[str, Any],
        reset_hand: Mapping[str, Any],
    ) -> bool:
        live_positions = _extract_control_step_positions(live_hand)
        reset_positions = _extract_control_step_positions(reset_hand)
        if "wrist" not in live_positions or "wrist" not in reset_positions:
            return False

        bone_scales: dict[tuple[str, str], float] = {}
        for parent_joint, child_joint in _CONTROL_EDGES:
            live_parent = live_positions.get(parent_joint)
            live_child = live_positions.get(child_joint)
            reset_parent = reset_positions.get(parent_joint)
            reset_child = reset_positions.get(child_joint)
            if (
                live_parent is None
                or live_child is None
                or reset_parent is None
                or reset_child is None
            ):
                continue
            live_length = float(np.linalg.norm(live_child - live_parent))
            reset_length = float(np.linalg.norm(reset_child - reset_parent))
            if not np.isfinite(live_length) or live_length <= _MIN_VALID_BONE_LENGTH:
                continue
            if not np.isfinite(reset_length):
                continue
            scale_value = float(np.clip(
                reset_length / live_length,
                self._step_hand_scale_min,
                self._step_hand_scale_max,
            ))
            bone_scales[(parent_joint, child_joint)] = scale_value

        self._step_hand_bone_scale_by_side[side] = bone_scales
        self._step_hand_scaling_calibrated_by_side[side] = bool(bone_scales)
        return bool(bone_scales)

    def _maybe_calibrate_step_hand_scaling(
        self,
        live_step_hands: Mapping[str, Optional[Mapping[str, Any]]],
    ) -> None:
        if not self._step_hand_scaling_enable:
            return
        for side in (robots.LEFT, robots.RIGHT):
            if self._step_hand_scaling_calibrated_by_side.get(side):
                continue
            live_hand = live_step_hands.get(side)
            reset_hand = self._reset_reference_step_hands.get(side)
            if not isinstance(live_hand, Mapping) or not isinstance(reset_hand, Mapping):
                continue
            self._calibrate_step_hand_scaling_for_side(
                side,
                live_hand=live_hand,
                reset_hand=reset_hand,
            )

    def _apply_step_hand_scaling_to_hand_payload(
        self,
        side: str,
        hand_payload: Optional[Mapping[str, Any]],
    ) -> Optional[dict[str, Any]]:
        normalized_hand = self._normalize_step_hand_payload(hand_payload)
        if normalized_hand is None:
            return None
        if not self._step_hand_scaling_enable:
            return normalized_hand

        current_positions = _extract_control_step_positions(normalized_hand)
        wrist_position = current_positions.get("wrist")
        if wrist_position is None:
            return normalized_hand

        scaled_positions: dict[str, np.ndarray] = {
            "wrist": wrist_position.copy(),
            "palm": wrist_position.copy(),
        }
        side_scales = self._step_hand_bone_scale_by_side.get(side, {})
        for parent_joint, child_joint in _CONTROL_EDGES:
            parent_scaled = scaled_positions.get(parent_joint)
            parent_raw = current_positions.get(parent_joint)
            child_raw = current_positions.get(child_joint)
            if parent_scaled is None or parent_raw is None or child_raw is None:
                continue
            scale_value = float(side_scales.get((parent_joint, child_joint), 1.0))
            if not np.isfinite(scale_value):
                scale_value = 1.0
            scaled_positions[child_joint] = parent_scaled + scale_value * (child_raw - parent_raw)

        return _set_hand_payload_positions(normalized_hand, scaled_positions)

    def _build_hand_payload(self, side: str) -> Optional[dict[str, Any]]:
        input_payload = self._latest_input_by_side.get(side, {})
        action_payload = self._last_action_by_side.get(side)
        if not input_payload and action_payload is None:
            return None

        source = input_payload.get("source")
        if source is None:
            source = "action_only" if action_payload is not None else "none"

        return {
            "source": source,
            "timestamp_s": input_payload.get("timestamp_s"),
            "keypoints_xyz": self._flip_keypoints_z(input_payload.get("keypoints_xyz")),
            "is_relative": bool(input_payload.get("is_relative", False)),
            "world_frame": input_payload.get("world_frame"),
            "joint_order": input_payload.get("joint_order"),
            "joint_transforms_world": self._flip_joint_transforms_world_z(
                input_payload.get("joint_transforms_world")
            ),
            "joint_positions_rad": (
                action_payload.get("joint_positions_rad") if action_payload is not None else None
            ),
            "action_timestamp_s": action_payload.get("timestamp_s") if action_payload is not None else None,
        }

    def _build_current_input_step_hand_payload(self, side: str) -> Optional[dict[str, Any]]:
        input_payload = self._latest_input_by_side.get(side, {})
        if not input_payload:
            return None
        step_joint_transforms_world = self._flip_joint_transforms_world_z(
            input_payload.get("joint_transforms_world")
        )
        step_joint_transforms_world = self._apply_step_local_post_z_flip_joint_transforms_world(
            side,
            step_joint_transforms_world,
        )
        return {
            "source": input_payload.get("source"),
            "timestamp_s": input_payload.get("timestamp_s"),
            "keypoints_xyz": self._flip_keypoints_z(input_payload.get("keypoints_xyz")),
            "is_relative": bool(input_payload.get("is_relative", False)),
            "world_frame": input_payload.get("world_frame"),
            "joint_order": input_payload.get("joint_order"),
            "joint_transforms_world": step_joint_transforms_world,
        }

    def _build_current_input_step_hands(self) -> dict[str, Optional[dict[str, Any]]]:
        return {
            robots.LEFT: self._build_current_input_step_hand_payload(robots.LEFT),
            robots.RIGHT: self._build_current_input_step_hand_payload(robots.RIGHT),
        }

    def _build_scaled_current_input_step_hands(self) -> dict[str, Optional[dict[str, Any]]]:
        normalized_hands = self._normalize_step_hands(self._build_current_input_step_hands())
        self._maybe_calibrate_step_hand_scaling(normalized_hands)
        return {
            side: self._apply_step_hand_scaling_to_hand_payload(side, normalized_hands.get(side))
            for side in (robots.LEFT, robots.RIGHT)
        }

    def _copy_step_hands(self, hands: Mapping[str, Any]) -> dict[str, Optional[dict[str, Any]]]:
        return {
            side: copy.deepcopy(hands.get(side)) if isinstance(hands.get(side), Mapping) else None
            for side in (robots.LEFT, robots.RIGHT)
        }

    @staticmethod
    def _snapshot_payload_from_hands(hands: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": "two_hand_snapshot",
            "sent_at_s": time.time(),
            "timestamp_s": time.time(),
            "hands": copy.deepcopy(dict(hands)),
        }
        timestamps = [
            hand_payload.get("timestamp_s")
            for hand_payload in hands.values()
            if isinstance(hand_payload, Mapping) and hand_payload.get("timestamp_s") is not None
        ]
        if timestamps:
            payload["timestamp_s"] = max(timestamps)
        world_frames = {
            hand_payload.get("world_frame")
            for hand_payload in hands.values()
            if isinstance(hand_payload, Mapping) and hand_payload.get("world_frame")
        }
        if len(world_frames) == 1:
            payload["world_frame"] = next(iter(world_frames))
        return payload

    def _attach_latest_actions(self, snapshot_payload: Mapping[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(dict(snapshot_payload))
        hands = payload.get("hands")
        if not isinstance(hands, Mapping):
            return payload
        updated_hands: dict[str, Any] = {}
        for side in (robots.LEFT, robots.RIGHT):
            hand_payload = hands.get(side)
            if not isinstance(hand_payload, Mapping):
                updated_hands[side] = hand_payload
                continue
            updated_hand = dict(hand_payload)
            action_payload = self._last_action_by_side.get(side)
            if action_payload is not None:
                updated_hand["joint_positions_rad"] = action_payload.get("joint_positions_rad")
                updated_hand["action_timestamp_s"] = action_payload.get("timestamp_s")
            updated_hands[side] = updated_hand
        payload["hands"] = updated_hands
        payload["sent_at_s"] = time.time()
        return payload

    def _seed_from_reset_state(self, state: Mapping[str, Any]) -> None:
        hands = state.get("hands")
        self._step_hand_bone_scale_by_side = {
            robots.LEFT: {},
            robots.RIGHT: {},
        }
        self._step_hand_scaling_calibrated_by_side = {
            robots.LEFT: False,
            robots.RIGHT: False,
        }
        if not isinstance(hands, Mapping):
            self._reset_reference_step_hands = {robots.LEFT: None, robots.RIGHT: None}
            self._accepted_step_hands = {robots.LEFT: None, robots.RIGHT: None}
            self._pending_reset_step_snapshot = None
            return
        normalized_hands = self._normalize_step_hands(
            {
                side: hands.get(side) if isinstance(hands.get(side), Mapping) else None
                for side in (robots.LEFT, robots.RIGHT)
            }
        )
        copied_hands = self._copy_step_hands(normalized_hands)
        self._reset_reference_step_hands = self._copy_step_hands(copied_hands)
        self._accepted_step_hands = copied_hands
        self._pending_reset_step_snapshot = self._snapshot_payload_from_hands(copied_hands)

    def _compute_step_distance_gate(
        self,
        current_step_hands: Mapping[str, Optional[Mapping[str, Any]]],
        reference_step_hands: Mapping[str, Optional[Mapping[str, Any]]],
    ) -> Optional[float]:
        total_squared_error = 0.0
        for side in (robots.LEFT, robots.RIGHT):
            current_hand = current_step_hands.get(side)
            reference_hand = reference_step_hands.get(side)
            if not isinstance(current_hand, Mapping) or not isinstance(reference_hand, Mapping):
                return None
            current_positions = _extract_step_space_positions(current_hand, drop_palm=True)
            reference_positions = _extract_step_space_positions(reference_hand, drop_palm=True)
            common_joints = sorted(set(current_positions) & set(reference_positions))
            if not common_joints:
                return None
            squared_errors = [
                float(np.dot(
                    current_positions[joint_name] - reference_positions[joint_name],
                    current_positions[joint_name] - reference_positions[joint_name],
                ))
                for joint_name in common_joints
            ]
            total_squared_error += float(sum(squared_errors))
        return float(total_squared_error)

    def _publish_hand_visualization(
        self,
        *,
        live_step_hands: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
        blocked: bool,
    ) -> None:
        if self._hand_visualization_publisher is None:
            return
        if live_step_hands is None:
            live_step_hands = self._build_scaled_current_input_step_hands()
        self._hand_visualization_publisher.publish(
            live_step_hands=live_step_hands,
            reference_step_hands=self._accepted_step_hands,
            show_reference=bool(blocked),
        )

    def _build_snapshot_payload(
        self,
        hands: Optional[Mapping[str, Optional[Mapping[str, Any]]]] = None,
    ) -> Optional[dict[str, Any]]:
        if hands is None:
            hands = self._build_scaled_current_input_step_hands()
        if not any(isinstance(hands.get(side), Mapping) for side in (robots.LEFT, robots.RIGHT)):
            return None

        input_timestamps = [
            hand_payload.get("timestamp_s")
            for hand_payload in hands.values()
            if isinstance(hand_payload, Mapping) and hand_payload.get("timestamp_s") is not None
        ]
        world_frames = {
            hand_payload.get("world_frame")
            for hand_payload in hands.values()
            if isinstance(hand_payload, Mapping) and hand_payload.get("world_frame")
        }

        payload: dict[str, Any] = {
            "source": "two_hand_snapshot",
            "sent_at_s": time.time(),
            "timestamp_s": max(input_timestamps) if input_timestamps else time.time(),
            "hands": hands,
        }
        if len(world_frames) == 1:
            payload["world_frame"] = next(iter(world_frames))
        return self._attach_latest_actions(payload)

    def _write_reset_frame(self, obs_jpg: bytes, *, new: bool) -> Optional[str]:
        if not obs_jpg or self._reset_frame_dir is None:
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            self._reset_frame_dir,
            f"ego_dex_wm_reset_new_{str(bool(new)).lower()}_{timestamp}.jpg",
        )
        with open(path, "wb") as file_obj:
            file_obj.write(obs_jpg)
        return path

    def reset(self, new: bool = True) -> Tuple[Optional[np.ndarray], Mapping[str, Any]]:
        obs_jpg, state = self._wm_client.reset(new=new)
        self._latest_state = state
        self._latest_obs_frame = self._decode_jpg(obs_jpg)
        self._last_obs_wall_time_s = time.time()
        self._seed_from_reset_state(state)
        current_step_hands = self._build_scaled_current_input_step_hands()
        self._last_gate_value = None
        self._last_blocked = False
        reset_frame_path = self._write_reset_frame(obs_jpg, new=bool(new))
        self._last_event_record = {
            "event": "remote_wm_reset",
            "received_at_s": self._last_obs_wall_time_s,
            "new": bool(new),
            "state_keys": sorted(state.keys()),
            "reset_frame_path": reset_frame_path,
            "seeded_step_pose": self._pending_reset_step_snapshot is not None,
        }
        self._publish_hand_visualization(live_step_hands=current_step_hands, blocked=False)
        return self._latest_obs_frame, self._latest_state

    def send_action(self, action: RobotActionCommand) -> None:
        joint_positions = np.asarray(action.joint_positions_rad, dtype=np.float32).reshape(-1).tolist()
        self._last_action_by_side[action.hand_side] = {
            "hand_side": action.hand_side,
            "timestamp_s": action.timestamp_s,
            "joint_positions_rad": joint_positions,
        }
        self._latest_state = dict(self._latest_state)
        self._latest_state[f"{action.hand_side}_joint_state"] = joint_positions

    def on_keypoints(
        self,
        side: str,
        keypoints_xyz: np.ndarray,
        timestamp_s: float,
        *,
        is_relative: bool,
        world_frame: Optional[str],
        joint_order: Optional[Any],
        joint_transforms_world: Optional[Any],
    ) -> None:
        keypoints = np.asarray(keypoints_xyz, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] != 3:
            return
        self._latest_input_by_side[side] = {
            "source": "xr_hand_joint_poses" if joint_transforms_world is not None else "vr_keypoints",
            "timestamp_s": float(timestamp_s),
            "keypoints_xyz": keypoints.tolist(),
            "is_relative": bool(is_relative),
            "world_frame": str(world_frame) if world_frame else None,
            "joint_order": list(joint_order) if joint_order is not None else None,
            "joint_transforms_world": self._copy_joint_transforms_world(joint_transforms_world),
        }

    def step(self) -> None:
        current_step_hands = self._build_scaled_current_input_step_hands()
        payload = self._build_snapshot_payload(current_step_hands)

        gate_value: Optional[float] = None
        blocked = False
        if any(self._accepted_step_hands.get(side) is not None for side in (robots.LEFT, robots.RIGHT)):
            gate_value = self._compute_step_distance_gate(current_step_hands, self._accepted_step_hands)
            blocked = gate_value is None or gate_value > self._step_distance_threshold
        self._last_gate_value = gate_value
        self._last_blocked = blocked
        self._publish_hand_visualization(live_step_hands=current_step_hands, blocked=blocked)

        if payload is None or blocked:
            return
        now_s = time.time()
        if now_s - self._last_request_wall_time_s < self._step_period_s:
            return
        self._last_request_wall_time_s = now_s

        used_reset_seed_pose = False
        request_payload = payload
        if self._pending_reset_step_snapshot is not None:
            request_payload = self._attach_latest_actions(self._pending_reset_step_snapshot)
            used_reset_seed_pose = True

        try:
            self._request_step(
                request_payload,
                event_name="remote_wm_step_reset_seed" if used_reset_seed_pose else "remote_wm_step",
            )
            if used_reset_seed_pose:
                self._pending_reset_step_snapshot = None
            else:
                self._accepted_step_hands = self._copy_step_hands(request_payload.get("hands", {}))
            if self._last_event_record is not None:
                self._last_event_record["step_distance_threshold"] = self._step_distance_threshold
                self._last_event_record["gate_value"] = gate_value
                self._last_event_record["used_reset_seed_pose"] = used_reset_seed_pose
        except Exception:
            logger.exception("Remote WM step failed")

    def get_joint_state(self, side: str) -> np.ndarray:
        joint_state = self._extract_joint_state(side)
        if joint_state is None:
            return self._default_joint_state
        if joint_state.size == self._dof:
            return joint_state
        if joint_state.size > self._dof:
            return joint_state[: self._dof]
        padded = np.zeros(self._dof, dtype=np.float32)
        padded[: joint_state.size] = joint_state
        return padded

    def get_camera_frame(self) -> Optional[np.ndarray]:
        return self._latest_obs_frame

    def set_step_distance_threshold(self, value: float) -> float:
        self._step_distance_threshold = max(float(value), 0.0)
        return self._step_distance_threshold

    def get_gate_status(self) -> dict[str, Any]:
        has_reference_pose = any(
            self._accepted_step_hands.get(side) is not None for side in (robots.LEFT, robots.RIGHT)
        )
        reference_source: Optional[str] = None
        if has_reference_pose:
            reference_source = "reset" if self._pending_reset_step_snapshot is not None else "last_step"
        scaling_bones_by_side = {
            side: len(self._step_hand_bone_scale_by_side.get(side, {}))
            for side in (robots.LEFT, robots.RIGHT)
        }
        return {
            "gate_metric": "sum_squared_error",
            "gate_error": self._last_gate_value,
            "step_distance_threshold": self._step_distance_threshold,
            "blocked": bool(self._last_blocked),
            "has_reference_pose": has_reference_pose,
            "reference_source": reference_source,
            "step_hand_scaling_enabled": bool(self._step_hand_scaling_enable),
            "step_hand_scaling_calibrated_by_side": dict(self._step_hand_scaling_calibrated_by_side),
            "step_hand_scaling_bones_by_side": scaling_bones_by_side,
            "step_hand_scale_min": self._step_hand_scale_min,
            "step_hand_scale_max": self._step_hand_scale_max,
        }

    def pop_pose_record(self) -> Optional[dict]:
        record = self._last_event_record
        self._last_event_record = None
        return record
