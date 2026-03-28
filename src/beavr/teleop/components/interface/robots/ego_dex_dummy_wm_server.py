from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
try:
    import zmq
except ImportError:  # pragma: no cover - optional until feather debug is enabled
    zmq = None

from beavr.teleop.configs.constants import robots

HAND_JOINT_ORDER: Tuple[str, ...] = (
    "wrist",
    "palm",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
)

HAND_EDGES: Tuple[Tuple[str, str], ...] = (
    ("wrist", "palm"),
    ("palm", "thumb_metacarpal"),
    ("thumb_metacarpal", "thumb_proximal"),
    ("thumb_proximal", "thumb_distal"),
    ("thumb_distal", "thumb_tip"),
    ("palm", "index_metacarpal"),
    ("index_metacarpal", "index_proximal"),
    ("index_proximal", "index_intermediate"),
    ("index_intermediate", "index_distal"),
    ("index_distal", "index_tip"),
    ("palm", "middle_metacarpal"),
    ("middle_metacarpal", "middle_proximal"),
    ("middle_proximal", "middle_intermediate"),
    ("middle_intermediate", "middle_distal"),
    ("middle_distal", "middle_tip"),
    ("palm", "ring_metacarpal"),
    ("ring_metacarpal", "ring_proximal"),
    ("ring_proximal", "ring_intermediate"),
    ("ring_intermediate", "ring_distal"),
    ("ring_distal", "ring_tip"),
    ("palm", "little_metacarpal"),
    ("little_metacarpal", "little_proximal"),
    ("little_proximal", "little_intermediate"),
    ("little_intermediate", "little_distal"),
    ("little_distal", "little_tip"),
)

AXIS_COLORS = (
    (70, 70, 255),
    (80, 220, 80),
    (255, 140, 70),
)

SIDE_COLORS = {
    robots.LEFT: {
        "edge": (255, 180, 80),
        "joint": (255, 220, 120),
    },
    robots.RIGHT: {
        "edge": (80, 180, 255),
        "joint": (120, 220, 255),
    },
}

FEATHER_SIDE_COLORS = {
    robots.LEFT: (1.0, 0.74, 0.28, 0.95),
    robots.RIGHT: (0.34, 0.78, 1.0, 0.95),
}
FEATHER_LIVE_SIDE_COLORS = {
    robots.LEFT: (1.0, 0.22, 0.22, 0.98),
    robots.RIGHT: (1.0, 0.22, 0.22, 0.98),
}
FEATHER_VERTICAL_OFFSET_Y_M = 0.10

_FLIP_Z_4X4 = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)


class _MinimalPoseRenderer:
    def __init__(self, image_width: int, image_height: int):
        self._image_width = int(image_width)
        self._image_height = int(image_height)
        self._projection_scale = float(min(self._image_width, self._image_height)) * 0.9
        self._projection_depth_offset = 1.1
        self._y_center = 0.95
        self._axis_scale = 0.018

    def _project_point(self, xyz: np.ndarray) -> Optional[Tuple[int, int]]:
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            return None
        z = float(xyz[2] + self._projection_depth_offset)
        if z <= 1e-3:
            return None
        x = float(xyz[0])
        y = float(xyz[1] - self._y_center)
        u = int(self._image_width * 0.5 + self._projection_scale * x / z)
        v = int(self._image_height * 0.82 - self._projection_scale * y / z)
        if -50 <= u <= self._image_width + 50 and -50 <= v <= self._image_height + 50:
            return u, v
        return None

    @staticmethod
    def _make_background(image_width: int, image_height: int) -> np.ndarray:
        frame = np.full((image_height, image_width, 3), 18, dtype=np.uint8)
        frame[:, :, 1] = 30
        frame[:, :, 2] = 24
        for y in range(0, image_height, 72):
            cv2.line(frame, (0, y), (image_width, y), (28, 40, 34), 1, lineType=cv2.LINE_AA)
        for x in range(0, image_width, 72):
            cv2.line(frame, (x, 0), (x, image_height), (28, 40, 34), 1, lineType=cv2.LINE_AA)
        return frame

    @staticmethod
    def _joint_positions_from_keypoints(keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            joint_name: keypoints[index].astype(np.float32)
            for index, joint_name in enumerate(HAND_JOINT_ORDER)
            if index < keypoints.shape[0]
        }

    @staticmethod
    def _joint_positions_from_transforms(
        joint_transforms_world: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        positions: Dict[str, np.ndarray] = {}
        for joint_name, transform in joint_transforms_world.items():
            if transform.shape != (4, 4):
                continue
            if _is_placeholder_transform(transform):
                continue
            positions[joint_name] = transform[:3, 3].astype(np.float32)
        return positions

    def _draw_hand(
        self,
        frame: np.ndarray,
        side: str,
        joint_positions: Mapping[str, np.ndarray],
        joint_transforms_world: Optional[Mapping[str, np.ndarray]],
    ) -> None:
        palette = SIDE_COLORS[side]
        projected: Dict[str, Optional[Tuple[int, int]]] = {}
        for joint_name in HAND_JOINT_ORDER:
            xyz = joint_positions.get(joint_name)
            projected[joint_name] = self._project_point(xyz) if xyz is not None else None

        for joint_a, joint_b in HAND_EDGES:
            point_a = projected.get(joint_a)
            point_b = projected.get(joint_b)
            if point_a is None or point_b is None:
                continue
            cv2.line(frame, point_a, point_b, palette["edge"], 2, lineType=cv2.LINE_AA)

        for joint_name in HAND_JOINT_ORDER:
            point = projected.get(joint_name)
            if point is None:
                continue
            radius = 4 if joint_name in {"wrist", "palm"} else 3
            cv2.circle(frame, point, radius, palette["joint"], -1, lineType=cv2.LINE_AA)

        if joint_transforms_world is None:
            return

        for joint_name in HAND_JOINT_ORDER:
            transform = joint_transforms_world.get(joint_name)
            if transform is None or transform.shape != (4, 4) or _is_placeholder_transform(transform):
                continue
            origin = transform[:3, 3].astype(np.float32)
            rotation = transform[:3, :3].astype(np.float32)
            origin_uv = self._project_point(origin)
            if origin_uv is None:
                continue
            for axis_index, color in enumerate(AXIS_COLORS):
                endpoint = origin + rotation[:, axis_index] * self._axis_scale
                endpoint_uv = self._project_point(endpoint)
                if endpoint_uv is None:
                    continue
                cv2.line(frame, origin_uv, endpoint_uv, color, 1, lineType=cv2.LINE_AA)

    def render(
        self,
        joint_transforms_world_by_side: Mapping[str, Optional[Mapping[str, np.ndarray]]],
        keypoints_by_side: Mapping[str, Optional[np.ndarray]],
        last_pose_timestamps: Mapping[str, Optional[float]],
        step_count: int,
    ) -> np.ndarray:
        frame = self._make_background(self._image_width, self._image_height)

        if not any(timestamp is not None for timestamp in last_pose_timestamps.values()):
            cv2.putText(
                frame,
                "Waiting for XR hand poses...",
                (32, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (220, 240, 220),
                2,
                lineType=cv2.LINE_AA,
            )
            return frame

        for side in (robots.LEFT, robots.RIGHT):
            transforms = joint_transforms_world_by_side.get(side)
            if transforms:
                joint_positions = self._joint_positions_from_transforms(transforms)
            else:
                keypoints = keypoints_by_side.get(side)
                joint_positions = (
                    self._joint_positions_from_keypoints(keypoints)
                    if isinstance(keypoints, np.ndarray)
                    else {}
                )
            if joint_positions:
                self._draw_hand(frame, side, joint_positions, transforms)

        now_s = time.time()
        pulse = 0.5 + 0.5 * math.sin(now_s * 2.5)
        cv2.putText(
            frame,
            "Dummy WM XR hand poses",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (230, 250, 235),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"step={step_count}",
            (18, self._image_height - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (190, 220, 200),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"t={now_s:.2f}",
            (18, self._image_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (190, 220, 200),
            2,
            lineType=cv2.LINE_AA,
        )
        radius = int(10 + 10 * pulse)
        cv2.circle(
            frame,
            (self._image_width - 34, 34),
            radius,
            (80, int(120 + 100 * pulse), 255),
            -1,
            lineType=cv2.LINE_AA,
        )
        return frame


def _is_placeholder_transform(transform: np.ndarray) -> bool:
    return bool(
        np.allclose(transform[:3, 3], 0.0, atol=1e-6)
        and np.allclose(transform[:3, :3], np.eye(3, dtype=np.float32), atol=1e-6)
    )


def _flip_transform_z(transform: np.ndarray) -> np.ndarray:
    return _FLIP_Z_4X4 @ transform @ _FLIP_Z_4X4


def _make_feather_point(
    position: np.ndarray,
    color: Tuple[float, float, float, float],
    size: float,
) -> dict[str, Any]:
    shifted_position = np.asarray(position, dtype=np.float32).copy()
    shifted_position[1] += FEATHER_VERTICAL_OFFSET_Y_M
    return {
        "position": {
            "x": float(shifted_position[0]),
            "y": float(shifted_position[1]),
            "z": float(shifted_position[2]),
        },
        "color": {
            "r": float(color[0]),
            "g": float(color[1]),
            "b": float(color[2]),
            "a": float(color[3]),
        },
        "size": float(size),
    }


def _hand_payload_to_feather_points(
    *,
    side: str,
    hand_payload: Mapping[str, Any],
    point_size: float,
    color_by_side: Mapping[str, Tuple[float, float, float, float]] = FEATHER_SIDE_COLORS,
) -> list[dict[str, Any]]:
    joint_transforms_world = hand_payload.get("joint_transforms_world")
    if isinstance(joint_transforms_world, Mapping):
        joint_order = hand_payload.get("joint_order")
        order = (
            tuple(str(joint_name) for joint_name in joint_order)
            if isinstance(joint_order, Sequence)
            else HAND_JOINT_ORDER
        )
        points = []
        color = color_by_side[side]
        for joint_name in order:
            raw_transform = joint_transforms_world.get(joint_name)
            if raw_transform is None:
                continue
            transform = np.asarray(raw_transform, dtype=np.float32)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                continue
            if _is_placeholder_transform(transform):
                continue
            tracking_space_transform = _flip_transform_z(transform)
            position = tracking_space_transform[:3, 3].astype(np.float32)
            points.append(
                _make_feather_point(
                    position=position,
                    color=color,
                    size=point_size if joint_name not in {"wrist", "palm"} else point_size * 1.2,
                )
            )
        if points:
            return points

    keypoints_xyz = hand_payload.get("keypoints_xyz")
    if keypoints_xyz is None:
        return []
    keypoints = np.asarray(keypoints_xyz, dtype=np.float32)
    if keypoints.ndim == 2 and keypoints.shape[1] == 3:
        tracking_space_keypoints = keypoints.copy()
        tracking_space_keypoints[:, 2] *= -1.0
        color = color_by_side[side]
        return [
            _make_feather_point(position=point, color=color, size=point_size)
            for point in tracking_space_keypoints
            if np.isfinite(point).all()
        ]

    return []


def _payload_to_feather_points(
    payload: Mapping[str, Any],
    *,
    point_size: float,
    color_by_side: Mapping[str, Tuple[float, float, float, float]] = FEATHER_SIDE_COLORS,
) -> list[dict[str, Any]]:
    hands = payload.get("hands")
    if not isinstance(hands, Mapping):
        return []

    points: list[dict[str, Any]] = []
    for side in (robots.LEFT, robots.RIGHT):
        hand_payload = hands.get(side)
        if not isinstance(hand_payload, Mapping):
            continue
        points.extend(
                _hand_payload_to_feather_points(
                    side=side,
                    hand_payload=hand_payload,
                    point_size=point_size,
                    color_by_side=color_by_side,
                )
            )
    return points


class FeatherTrajectoryPublisher:
    """Publish a combined feather overlay from looped trajectory frames and live `/step` payloads."""

    def __init__(
        self,
        trajectory_jsonl_path: Optional[str] = None,
        bind_host: str = "0.0.0.0",
        port: int = 15102,
        fps: float = 15.0,
        point_size: float = 0.015,
        step_payload_enabled: bool = False,
        live_stale_timeout_s: float = 0.35,
    ):
        self._trajectory_jsonl_path = trajectory_jsonl_path
        self._bind_host = str(bind_host)
        self._port = int(port)
        self._fps = max(float(fps), 0.1)
        self._point_size = max(float(point_size), 1e-4)
        self._step_payload_enabled = bool(step_payload_enabled)
        self._live_stale_timeout_s = max(float(live_stale_timeout_s), 0.0)
        self._sequence = 0
        self._frames = (
            self._load_frames(trajectory_jsonl_path)
            if trajectory_jsonl_path is not None
            else []
        )
        self._latest_step_points: list[dict[str, Any]] = []
        self._latest_step_update_s: Optional[float] = None
        self._state_lock = threading.Lock()
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def bind_address(self) -> str:
        return f"tcp://{self._bind_host}:{self._port}"

    @property
    def source_path(self) -> Optional[str]:
        return self._trajectory_jsonl_path

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def trajectory_enabled(self) -> bool:
        return self._trajectory_jsonl_path is not None

    @property
    def step_payload_enabled(self) -> bool:
        return self._step_payload_enabled

    def start(self) -> None:
        if self._thread is not None:
            return
        if zmq is None:
            raise RuntimeError(
                "pyzmq is required for feather debug streaming. Install the teleop dependencies first."
            )

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 8)
        self._socket.bind(self.bind_address)

        self._thread = threading.Thread(
            target=self._publish_loop,
            name="FeatherTrajectoryPublisher",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def publish_step_payload(self, payload: Mapping[str, Any]) -> None:
        if not self._step_payload_enabled:
            return
        points = _payload_to_feather_points(
            payload,
            point_size=self._point_size,
            color_by_side=FEATHER_LIVE_SIDE_COLORS,
        )
        with self._state_lock:
            self._latest_step_points = points
            self._latest_step_update_s = time.monotonic() if points else None


    def _publish_loop(self) -> None:
        if self._socket is None:
            return

        # Allow subscribers time to connect before the first payload.
        time.sleep(0.25)
        period_s = 1.0 / self._fps
        frame_index = 0

        while not self._stop_event.is_set():
            tick_start_s = time.perf_counter()
            now_s = time.monotonic()
            points: list[dict[str, Any]] = []

            if self._frames:
                points.extend(self._frames[frame_index]["points"])
                frame_index = (frame_index + 1) % len(self._frames)

            if self._step_payload_enabled:
                with self._state_lock:
                    latest_step_points = list(self._latest_step_points)
                    latest_step_update_s = self._latest_step_update_s
                if (
                    latest_step_update_s is not None
                    and now_s - latest_step_update_s <= self._live_stale_timeout_s
                ):
                    points.extend(latest_step_points)

            if points:
                payload = {
                    "frame": "tracking_space",
                    "sequence": self._sequence,
                    "points": points,
                }
                self._socket.send_string(json.dumps(payload, separators=(",", ":")))
                self._sequence += 1

            sleep_s = period_s - (time.perf_counter() - tick_start_s)
            if sleep_s > 0.0:
                self._stop_event.wait(timeout=sleep_s)

    def _load_frames(self, trajectory_jsonl_path: str) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        with open(trajectory_jsonl_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                points = self._record_to_points(record)
                if not points:
                    continue
                frames.append(
                    {
                        "frame": "tracking_space",
                        "points": points,
                    }
                )

        if not frames:
            raise ValueError(
                f"No feather points could be extracted from trajectory file: {trajectory_jsonl_path}"
            )
        return frames

    def _record_to_points(self, record: Mapping[str, Any]) -> list[dict[str, Any]]:
        hands = record.get("hands")
        if not isinstance(hands, Mapping):
            return []

        points: list[dict[str, Any]] = []
        for side in (robots.LEFT, robots.RIGHT):
            hand_payload = hands.get(side)
            if not isinstance(hand_payload, Mapping):
                continue
            points.extend(
                _hand_payload_to_feather_points(
                    side=side,
                    hand_payload=hand_payload,
                    point_size=self._point_size,
                    color_by_side=FEATHER_SIDE_COLORS,
                )
            )

        return points


class DummyWMServer:
    """Testing WM service that renders world-frame XR hand poses."""

    def __init__(
        self,
        dof: int = 16,
        width: int = 960,
        height: int = 720,
        payload_log_dir: str = "logs/dummy_wm_server",
        feather_publisher: Optional[FeatherTrajectoryPublisher] = None,
        reset_pose_jsonl_path: Optional[str] = None,
    ):
        self._dof = int(dof)
        self._width = int(width)
        self._height = int(height)
        self._renderer = _MinimalPoseRenderer(image_width=self._width, image_height=self._height)
        self._feather_publisher = feather_publisher
        self._right_joint_state = np.zeros(self._dof, dtype=np.float32)
        self._left_joint_state = np.zeros(self._dof, dtype=np.float32)
        self._latest_keypoints: Dict[str, Optional[np.ndarray]] = {robots.LEFT: None, robots.RIGHT: None}
        self._latest_joint_transforms_world: Dict[str, Optional[Dict[str, np.ndarray]]] = {
            robots.LEFT: None,
            robots.RIGHT: None,
        }
        self._last_pose_timestamps: Dict[str, Optional[float]] = {robots.LEFT: None, robots.RIGHT: None}
        self._latest_source: Dict[str, str] = {robots.LEFT: "none", robots.RIGHT: "none"}
        self._world_frame_by_side: Dict[str, Optional[str]] = {robots.LEFT: None, robots.RIGHT: None}
        self._last_reset_s = time.time()
        self._step_count = 0
        self._reset_pose_jsonl_path = str(reset_pose_jsonl_path) if reset_pose_jsonl_path else None
        self._reset_step_snapshots = (
            self._load_reset_step_snapshots(self._reset_pose_jsonl_path)
            if self._reset_pose_jsonl_path is not None
            else []
        )
        os.makedirs(payload_log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._payload_log_path = os.path.join(payload_log_dir, f"dummy_wm_payloads_{timestamp}.jsonl")
        self._payload_log_file = open(self._payload_log_path, "a", buffering=1)
        self._payload_log_lock = threading.Lock()

    @property
    def payload_log_path(self) -> str:
        return self._payload_log_path

    def log_request(self, endpoint: str, payload: Mapping[str, Any], client_address: str) -> None:
        record = {
            "event": "dummy_wm_request",
            "received_at_s": time.time(),
            "endpoint": endpoint,
            "client_address": client_address,
            "payload": dict(payload),
        }
        with self._payload_log_lock:
            self._payload_log_file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._feather_publisher is not None:
            self._feather_publisher.close()
        with self._payload_log_lock:
            self._payload_log_file.close()

    def _sanitize_action(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        if array.size == self._dof:
            return array
        if array.size > self._dof:
            return array[: self._dof]
        padded = np.zeros(self._dof, dtype=np.float32)
        padded[: array.size] = array
        return padded

    @staticmethod
    def _sanitize_keypoints(values: Any) -> Optional[np.ndarray]:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 2 and array.shape == (robots.OCULUS_NUM_KEYPOINTS, 3):
            return array
        if array.size == robots.OCULUS_NUM_KEYPOINTS * 3:
            return array.reshape(robots.OCULUS_NUM_KEYPOINTS, 3)
        return None

    @staticmethod
    def _sanitize_transform_matrix(value: Any) -> Optional[np.ndarray]:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (4, 4):
            return None
        if not np.isfinite(array).all():
            return None
        array[3] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return array

    def _sanitize_joint_transforms_world(
        self,
        raw_transforms: Any,
        joint_order: Optional[Sequence[Any]],
    ) -> Optional[Dict[str, np.ndarray]]:
        if not isinstance(raw_transforms, Mapping):
            return None
        if joint_order is None:
            order = HAND_JOINT_ORDER
        else:
            order = tuple(str(joint_name) for joint_name in joint_order)

        transforms: Dict[str, np.ndarray] = {}
        for joint_name in order:
            transform = self._sanitize_transform_matrix(raw_transforms.get(joint_name))
            if transform is not None:
                transforms[str(joint_name)] = transform
        return transforms or None

    def _decode_payload(
        self, payload: Mapping[str, Any]
    ) -> Tuple[
        str,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[Dict[str, np.ndarray]],
        Optional[str],
        Optional[float],
    ]:
        side = str(payload.get("hand_side", robots.RIGHT)).lower()
        if side not in {robots.LEFT, robots.RIGHT}:
            side = robots.RIGHT

        action = None
        if "joint_positions_rad" in payload:
            action = self._sanitize_action(payload.get("joint_positions_rad"))

        keypoints = self._sanitize_keypoints(payload.get("keypoints_xyz"))
        joint_transforms_world = self._sanitize_joint_transforms_world(
            payload.get("joint_transforms_world"),
            payload.get("joint_order"),
        )
        world_frame = payload.get("world_frame")
        if not isinstance(world_frame, str):
            world_frame = None
        timestamp_s = payload.get("timestamp_s")
        timestamp = float(timestamp_s) if isinstance(timestamp_s, (int, float)) else None
        return side, action, keypoints, joint_transforms_world, world_frame, timestamp

    def _apply_hand_update(
        self,
        side: str,
        action: Optional[np.ndarray],
        keypoints: Optional[np.ndarray],
        joint_transforms_world: Optional[Dict[str, np.ndarray]],
        world_frame: Optional[str],
        timestamp_s: Optional[float],
        *,
        clear_transforms: bool,
    ) -> None:
        if action is not None:
            if side == robots.LEFT:
                self._left_joint_state = action
            else:
                self._right_joint_state = action
        if keypoints is not None:
            self._latest_keypoints[side] = keypoints
        if clear_transforms:
            self._latest_joint_transforms_world[side] = None
            self._world_frame_by_side[side] = None
            if keypoints is not None:
                self._latest_source[side] = "keypoints"
        elif joint_transforms_world is not None:
            self._latest_joint_transforms_world[side] = joint_transforms_world
            self._latest_source[side] = "joint_transforms_world"
        elif keypoints is not None:
            self._latest_source[side] = "keypoints"
        if world_frame is not None:
            self._world_frame_by_side[side] = world_frame
        if keypoints is not None or joint_transforms_world is not None:
            self._last_pose_timestamps[side] = timestamp_s if timestamp_s is not None else time.time()

    def _apply_snapshot_payload(self, payload: Mapping[str, Any]) -> None:
        hands = payload.get("hands")
        if not isinstance(hands, Mapping):
            return

        default_world_frame = payload.get("world_frame")
        if not isinstance(default_world_frame, str):
            default_world_frame = None
        default_timestamp_s = payload.get("timestamp_s")
        if not isinstance(default_timestamp_s, (int, float)):
            default_timestamp_s = None

        for side in (robots.LEFT, robots.RIGHT):
            hand_payload = hands.get(side)
            if hand_payload is None:
                continue
            if not isinstance(hand_payload, Mapping):
                continue
            decoded_payload = dict(hand_payload)
            decoded_payload["hand_side"] = side
            _, action, keypoints, joint_transforms_world, world_frame, timestamp_s = self._decode_payload(
                decoded_payload
            )
            clear_transforms = (
                "joint_transforms_world" in decoded_payload and joint_transforms_world is None
            )
            self._apply_hand_update(
                side,
                action,
                keypoints,
                joint_transforms_world,
                world_frame if world_frame is not None else default_world_frame,
                timestamp_s if timestamp_s is not None else default_timestamp_s,
                clear_transforms=clear_transforms,
            )

    def _state_dict(self) -> Dict[str, Any]:
        return {
            "left_joint_state": self._left_joint_state.tolist(),
            "right_joint_state": self._right_joint_state.tolist(),
            "left_world_frame": self._world_frame_by_side[robots.LEFT],
            "right_world_frame": self._world_frame_by_side[robots.RIGHT],
            "last_pose_timestamps": self._last_pose_timestamps,
            "last_reset_s": self._last_reset_s,
        }

    def _draw_status_overlay(self, frame: np.ndarray) -> None:
        lines = [
            (
                f"R source={self._latest_source[robots.RIGHT]} "
                f"frame={self._world_frame_by_side[robots.RIGHT]} "
                f"ts={self._last_pose_timestamps[robots.RIGHT]}"
            ),
            (
                f"L source={self._latest_source[robots.LEFT]} "
                f"frame={self._world_frame_by_side[robots.LEFT]} "
                f"ts={self._last_pose_timestamps[robots.LEFT]}"
            ),
            "axes: x=red y=green z=blue",
        ]
        y = self._height - 94
        for line in lines:
            cv2.putText(
                frame,
                line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 232, 220),
                1,
                lineType=cv2.LINE_AA,
            )
            y += 22

    @staticmethod
    def _flip_keypoints_z(keypoints_xyz: Any) -> Any:
        if keypoints_xyz is None:
            return None
        keypoints = np.asarray(keypoints_xyz, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] != 3:
            return keypoints_xyz
        flipped = keypoints.copy()
        flipped[:, 2] *= -1.0
        return flipped.tolist()

    def _record_to_reset_step_snapshot(self, record: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        hands = record.get("hands")
        if not isinstance(hands, Mapping):
            return None

        snapshot_hands: dict[str, Any] = {}
        for side in (robots.LEFT, robots.RIGHT):
            hand_payload = hands.get(side)
            if not isinstance(hand_payload, Mapping):
                continue
            snapshot_hand: dict[str, Any] = {
                "source": hand_payload.get("source", "jsonl_reset_seed"),
                "timestamp_s": hand_payload.get("timestamp_s", record.get("timestamp_s")),
                "is_relative": bool(hand_payload.get("is_relative", False)),
                "world_frame": hand_payload.get("world_frame", record.get("world_frame")),
                "joint_order": hand_payload.get("joint_order"),
                "keypoints_xyz": hand_payload.get("keypoints_xyz"),
                "joint_positions_rad": hand_payload.get("joint_positions_rad"),
                "action_timestamp_s": hand_payload.get("action_timestamp_s"),
            }
            joint_transforms_world = hand_payload.get("joint_transforms_world")
            if isinstance(joint_transforms_world, Mapping):
                snapshot_hand["joint_transforms_world"] = {
                    str(joint_name): np.asarray(transform, dtype=np.float32).tolist()
                    for joint_name, transform in joint_transforms_world.items()
                    if np.asarray(transform, dtype=np.float32).shape == (4, 4)
                }
            snapshot_hands[side] = snapshot_hand

        if not snapshot_hands:
            return None

        snapshot = {
            "source": "reset_seed_jsonl",
            "timestamp_s": record.get("timestamp_s", time.time()),
            "hands": snapshot_hands,
        }
        world_frame = record.get("world_frame")
        if isinstance(world_frame, str):
            snapshot["world_frame"] = world_frame
        return snapshot

    def _load_reset_step_snapshots(self, trajectory_jsonl_path: str) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        with open(trajectory_jsonl_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                snapshot = self._record_to_reset_step_snapshot(record)
                if snapshot is not None:
                    snapshots.append(snapshot)
        if not snapshots:
            raise ValueError(f"No usable reset poses found in JSONL: {trajectory_jsonl_path}")
        return snapshots

    def _sample_reset_step_snapshot(self) -> Optional[dict[str, Any]]:
        if not self._reset_step_snapshots:
            return None
        return copy.deepcopy(random.choice(self._reset_step_snapshots))

    def _render(self) -> np.ndarray:
        frame = self._renderer.render(
            joint_transforms_world_by_side=self._latest_joint_transforms_world,
            keypoints_by_side=self._latest_keypoints,
            last_pose_timestamps=self._last_pose_timestamps,
            step_count=self._step_count,
        )
        self._draw_status_overlay(frame)
        return frame

    @staticmethod
    def _encode_jpg(frame: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("Failed to encode dummy WM frame")
        return np.asarray(buffer).tobytes()

    def wm_step(self, payload: Mapping[str, Any]) -> bytes:
        if self._feather_publisher is not None:
            self._feather_publisher.publish_step_payload(payload)
        if isinstance(payload.get("hands"), Mapping):
            self._apply_snapshot_payload(payload)
        else:
            side, action, keypoints, joint_transforms_world, world_frame, timestamp_s = self._decode_payload(
                payload
            )
            self._apply_hand_update(
                side,
                action,
                keypoints,
                joint_transforms_world,
                world_frame,
                timestamp_s,
                clear_transforms=("joint_transforms_world" in payload and joint_transforms_world is None),
            )
        self._step_count += 1
        return self._encode_jpg(self._render())

    def reset(self, new: bool = True) -> Tuple[bytes, Mapping[str, Any]]:
        if new:
            self._right_joint_state = np.zeros(self._dof, dtype=np.float32)
            self._left_joint_state = np.zeros(self._dof, dtype=np.float32)
            self._latest_keypoints = {robots.LEFT: None, robots.RIGHT: None}
            self._latest_joint_transforms_world = {robots.LEFT: None, robots.RIGHT: None}
            self._last_pose_timestamps = {robots.LEFT: None, robots.RIGHT: None}
            self._latest_source = {robots.LEFT: "none", robots.RIGHT: "none"}
            self._world_frame_by_side = {robots.LEFT: None, robots.RIGHT: None}
            self._step_count = 0
        reset_snapshot = self._sample_reset_step_snapshot()
        if reset_snapshot is not None:
            self._apply_snapshot_payload(reset_snapshot)
        self._last_reset_s = time.time()
        state = self._state_dict()
        if reset_snapshot is not None:
            state["hands"] = copy.deepcopy(reset_snapshot.get("hands", {}))
            if "timestamp_s" in reset_snapshot:
                state["timestamp_s"] = reset_snapshot.get("timestamp_s")
            if "world_frame" in reset_snapshot:
                state["world_frame"] = reset_snapshot.get("world_frame")
        return self._encode_jpg(self._render()), state

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "step_count": self._step_count,
            "last_reset_s": self._last_reset_s,
            "payload_log_path": self._payload_log_path,
            "has_left_keypoints": self._latest_keypoints[robots.LEFT] is not None,
            "has_right_keypoints": self._latest_keypoints[robots.RIGHT] is not None,
            "has_left_joint_transforms_world": self._latest_joint_transforms_world[robots.LEFT]
            is not None,
            "has_right_joint_transforms_world": self._latest_joint_transforms_world[robots.RIGHT]
            is not None,
            "feather_debug_enabled": (
                self._feather_publisher.trajectory_enabled if self._feather_publisher is not None else False
            ),
            "feather_debug_bind_address": (
                self._feather_publisher.bind_address if self._feather_publisher is not None else None
            ),
            "feather_debug_source_path": (
                self._feather_publisher.source_path if self._feather_publisher is not None else None
            ),
            "feather_debug_fps": self._feather_publisher.fps if self._feather_publisher is not None else None,
            "feather_debug_frame_count": (
                self._feather_publisher.frame_count if self._feather_publisher is not None else 0
            ),
            "feather_step_payload_enabled": (
                self._feather_publisher.step_payload_enabled if self._feather_publisher is not None else False
            ),
            "feather_step_payload_bind_address": (
                self._feather_publisher.bind_address
                if self._feather_publisher is not None and self._feather_publisher.step_payload_enabled
                else None
            ),
        }


class _DummyWMHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, server_address, request_handler_class, wm_server: DummyWMServer):
        super().__init__(server_address, request_handler_class)
        self.wm_server = wm_server

    def handle_error(self, request, client_address) -> None:
        exc_type, exc_value, _ = sys.exc_info()
        if exc_type is OSError and isinstance(exc_value, OSError) and exc_value.errno == 9:
            return
        super().handle_error(request, client_address)


class _RequestHandler(BaseHTTPRequestHandler):
    server: _DummyWMHTTPServer

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> Mapping[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        if not raw_body:
            return {}
        decoded = json.loads(raw_body.decode("utf-8"))
        if not isinstance(decoded, Mapping):
            raise ValueError("Request body must be a JSON object")
        return decoded

    def _write_json(self, payload: Mapping[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_jpg(self, jpg_bytes: bytes, state: Optional[Mapping[str, Any]] = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg_bytes)))
        if state is not None:
            self.send_header("X-WM-State", json.dumps(state))
        self.end_headers()
        self.wfile.write(jpg_bytes)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(self.server.wm_server.health())
            return
        self._write_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
        except Exception as exc:
            self._write_json({"error": str(exc)}, status=400)
            return

        self.server.wm_server.log_request(
            endpoint=self.path,
            payload=payload,
            client_address=str(self.client_address[0]),
        )

        if self.path in {"/step", "/wm_step"}:
            try:
                jpg_bytes = self.server.wm_server.wm_step(payload)
            except Exception as exc:
                self._write_json({"error": str(exc)}, status=500)
                return
            self._write_jpg(jpg_bytes)
            return

        if self.path == "/reset":
            try:
                jpg_bytes, state = self.server.wm_server.reset(bool(payload.get("new", True)))
            except Exception as exc:
                self._write_json({"error": str(exc)}, status=500)
                return
            self._write_jpg(jpg_bytes, state=state)
            return

        self._write_json({"error": "not found"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy EgoDex WM server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--dof", type=int, default=16)
    parser.add_argument("--payload-log-dir", default="logs/dummy_wm_server")
    parser.add_argument(
        "--reset-pose-jsonl",
        default=None,
        help="Optional JSONL trajectory used to seed `/reset` with a random step-space hand pose.",
    )
    parser.add_argument(
        "--feather-trajectory-jsonl",
        default=None,
        help="Optional JSONL trajectory to stream into the Unity feather debug overlay.",
    )
    parser.add_argument(
        "--feather-bind-host",
        default="0.0.0.0",
        help="Interface to bind the feather PUB socket to. Default: 0.0.0.0",
    )
    parser.add_argument("--feather-port", type=int, default=15102)
    parser.add_argument("--feather-fps", type=float, default=15.0)
    parser.add_argument("--feather-point-size", type=float, default=0.015)
    parser.add_argument(
        "--feather-from-step-payload",
        action="store_true",
        help="Publish the exact `/step` hand payload back to the Unity feather overlay.",
    )
    args = parser.parse_args()

    feather_publisher: Optional[FeatherTrajectoryPublisher] = None
    if args.feather_trajectory_jsonl or args.feather_from_step_payload:
        feather_publisher = FeatherTrajectoryPublisher(
            trajectory_jsonl_path=args.feather_trajectory_jsonl,
            bind_host=args.feather_bind_host,
            port=args.feather_port,
            fps=args.feather_fps,
            point_size=args.feather_point_size,
            step_payload_enabled=args.feather_from_step_payload,
        )
        feather_publisher.start()

    wm_server = DummyWMServer(
        dof=args.dof,
        width=args.width,
        height=args.height,
        payload_log_dir=args.payload_log_dir,
        feather_publisher=feather_publisher,
        reset_pose_jsonl_path=args.reset_pose_jsonl,
    )
    http_server = _DummyWMHTTPServer((args.host, args.port), _RequestHandler, wm_server)
    print(f"Dummy WM server listening on http://{args.host}:{args.port}")
    print(f"Payload log: {wm_server.payload_log_path}")
    if feather_publisher is not None:
        if feather_publisher.trajectory_enabled:
            print(
                "Feather debug stream enabled: "
                f"{feather_publisher.bind_address} "
                f"(source={feather_publisher.source_path}, frames={feather_publisher.frame_count}, fps={feather_publisher.fps:.2f})"
            )
        if feather_publisher.step_payload_enabled:
            print(
                "Feather step-payload echo enabled: "
                f"{feather_publisher.bind_address} "
                f"(fps={feather_publisher.fps:.2f})"
            )
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http_server.shutdown()
        http_server.server_close()
        wm_server.close()


if __name__ == "__main__":
    main()
