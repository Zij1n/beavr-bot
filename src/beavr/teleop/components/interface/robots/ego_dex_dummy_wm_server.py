from __future__ import annotations

import argparse
import json
import math
import time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

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


class DummyWMServer:
    """Testing WM service that renders world-frame XR hand poses."""

    def __init__(self, dof: int = 16, width: int = 960, height: int = 720):
        self._dof = int(dof)
        self._width = int(width)
        self._height = int(height)
        self._renderer = _MinimalPoseRenderer(image_width=self._width, image_height=self._height)
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
        side, action, keypoints, joint_transforms_world, world_frame, timestamp_s = self._decode_payload(
            payload
        )
        if action is not None:
            if side == robots.LEFT:
                self._left_joint_state = action
            else:
                self._right_joint_state = action
        if keypoints is not None:
            self._latest_keypoints[side] = keypoints
        if "joint_transforms_world" in payload and joint_transforms_world is None:
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
        self._last_reset_s = time.time()
        return self._encode_jpg(self._render()), self._state_dict()

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "step_count": self._step_count,
            "last_reset_s": self._last_reset_s,
            "has_left_keypoints": self._latest_keypoints[robots.LEFT] is not None,
            "has_right_keypoints": self._latest_keypoints[robots.RIGHT] is not None,
            "has_left_joint_transforms_world": self._latest_joint_transforms_world[robots.LEFT]
            is not None,
            "has_right_joint_transforms_world": self._latest_joint_transforms_world[robots.RIGHT]
            is not None,
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

        if self.path == "/wm_step":
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
    args = parser.parse_args()

    wm_server = DummyWMServer(dof=args.dof, width=args.width, height=args.height)
    http_server = _DummyWMHTTPServer((args.host, args.port), _RequestHandler, wm_server)
    print(f"Dummy WM server listening on http://{args.host}:{args.port}")
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http_server.shutdown()
        http_server.server_close()


if __name__ == "__main__":
    main()
