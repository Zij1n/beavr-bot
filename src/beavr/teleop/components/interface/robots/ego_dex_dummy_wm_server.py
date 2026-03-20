from __future__ import annotations

import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from beavr.teleop.configs.constants import robots

LEFT_JOINTS = [
    "leftShoulder",
    "leftArm",
    "leftForearm",
    "leftHand",
    "leftThumbKnuckle",
    "leftThumbIntermediateBase",
    "leftThumbIntermediateTip",
    "leftThumbTip",
    "leftIndexFingerMetacarpal",
    "leftIndexFingerKnuckle",
    "leftIndexFingerIntermediateBase",
    "leftIndexFingerIntermediateTip",
    "leftIndexFingerTip",
    "leftMiddleFingerMetacarpal",
    "leftMiddleFingerKnuckle",
    "leftMiddleFingerIntermediateBase",
    "leftMiddleFingerIntermediateTip",
    "leftMiddleFingerTip",
    "leftRingFingerMetacarpal",
    "leftRingFingerKnuckle",
    "leftRingFingerIntermediateBase",
    "leftRingFingerIntermediateTip",
    "leftRingFingerTip",
    "leftLittleFingerMetacarpal",
    "leftLittleFingerKnuckle",
    "leftLittleFingerIntermediateBase",
    "leftLittleFingerIntermediateTip",
    "leftLittleFingerTip",
]

RIGHT_JOINTS = [
    "rightShoulder",
    "rightArm",
    "rightForearm",
    "rightHand",
    "rightThumbKnuckle",
    "rightThumbIntermediateBase",
    "rightThumbIntermediateTip",
    "rightThumbTip",
    "rightIndexFingerMetacarpal",
    "rightIndexFingerKnuckle",
    "rightIndexFingerIntermediateBase",
    "rightIndexFingerIntermediateTip",
    "rightIndexFingerTip",
    "rightMiddleFingerMetacarpal",
    "rightMiddleFingerKnuckle",
    "rightMiddleFingerIntermediateBase",
    "rightMiddleFingerIntermediateTip",
    "rightMiddleFingerTip",
    "rightRingFingerMetacarpal",
    "rightRingFingerKnuckle",
    "rightRingFingerIntermediateBase",
    "rightRingFingerIntermediateTip",
    "rightRingFingerTip",
    "rightLittleFingerMetacarpal",
    "rightLittleFingerKnuckle",
    "rightLittleFingerIntermediateBase",
    "rightLittleFingerIntermediateTip",
    "rightLittleFingerTip",
]

EGO_DEX_UPPER_LIMB_JOINTS = LEFT_JOINTS + RIGHT_JOINTS

HAND_KEYPOINT_INDEX = {
    "Hand": 0,
    "ThumbKnuckle": 2,
    "ThumbIntermediateBase": 3,
    "ThumbIntermediateTip": 4,
    "ThumbTip": 5,
    "IndexFingerMetacarpal": 6,
    "IndexFingerKnuckle": 7,
    "IndexFingerIntermediateBase": 8,
    "IndexFingerIntermediateTip": 9,
    "IndexFingerTip": 10,
    "MiddleFingerMetacarpal": 11,
    "MiddleFingerKnuckle": 12,
    "MiddleFingerIntermediateBase": 13,
    "MiddleFingerIntermediateTip": 14,
    "MiddleFingerTip": 15,
    "RingFingerMetacarpal": 16,
    "RingFingerKnuckle": 17,
    "RingFingerIntermediateBase": 18,
    "RingFingerIntermediateTip": 19,
    "RingFingerTip": 20,
    "LittleFingerMetacarpal": 21,
    "LittleFingerKnuckle": 22,
    "LittleFingerIntermediateBase": 23,
    "LittleFingerIntermediateTip": 24,
    "LittleFingerTip": 25,
}


def _build_edges(prefix: str) -> list[Tuple[str, str]]:
    return [
        (f"{prefix}Shoulder", f"{prefix}Arm"),
        (f"{prefix}Arm", f"{prefix}Forearm"),
        (f"{prefix}Forearm", f"{prefix}Hand"),
        (f"{prefix}Hand", f"{prefix}ThumbKnuckle"),
        (f"{prefix}ThumbKnuckle", f"{prefix}ThumbIntermediateBase"),
        (f"{prefix}ThumbIntermediateBase", f"{prefix}ThumbIntermediateTip"),
        (f"{prefix}ThumbIntermediateTip", f"{prefix}ThumbTip"),
        (f"{prefix}Hand", f"{prefix}IndexFingerMetacarpal"),
        (f"{prefix}IndexFingerMetacarpal", f"{prefix}IndexFingerKnuckle"),
        (f"{prefix}IndexFingerKnuckle", f"{prefix}IndexFingerIntermediateBase"),
        (f"{prefix}IndexFingerIntermediateBase", f"{prefix}IndexFingerIntermediateTip"),
        (f"{prefix}IndexFingerIntermediateTip", f"{prefix}IndexFingerTip"),
        (f"{prefix}Hand", f"{prefix}MiddleFingerMetacarpal"),
        (f"{prefix}MiddleFingerMetacarpal", f"{prefix}MiddleFingerKnuckle"),
        (f"{prefix}MiddleFingerKnuckle", f"{prefix}MiddleFingerIntermediateBase"),
        (f"{prefix}MiddleFingerIntermediateBase", f"{prefix}MiddleFingerIntermediateTip"),
        (f"{prefix}MiddleFingerIntermediateTip", f"{prefix}MiddleFingerTip"),
        (f"{prefix}Hand", f"{prefix}RingFingerMetacarpal"),
        (f"{prefix}RingFingerMetacarpal", f"{prefix}RingFingerKnuckle"),
        (f"{prefix}RingFingerKnuckle", f"{prefix}RingFingerIntermediateBase"),
        (f"{prefix}RingFingerIntermediateBase", f"{prefix}RingFingerIntermediateTip"),
        (f"{prefix}RingFingerIntermediateTip", f"{prefix}RingFingerTip"),
        (f"{prefix}Hand", f"{prefix}LittleFingerMetacarpal"),
        (f"{prefix}LittleFingerMetacarpal", f"{prefix}LittleFingerKnuckle"),
        (f"{prefix}LittleFingerKnuckle", f"{prefix}LittleFingerIntermediateBase"),
        (f"{prefix}LittleFingerIntermediateBase", f"{prefix}LittleFingerIntermediateTip"),
        (f"{prefix}LittleFingerIntermediateTip", f"{prefix}LittleFingerTip"),
    ]


SKELETON_EDGES = _build_edges("left") + _build_edges("right")


class _MinimalSkeletonRenderer:
    def __init__(self, image_width: int, image_height: int):
        self._image_width = int(image_width)
        self._image_height = int(image_height)
        self._projection_scale = float(min(self._image_width, self._image_height)) * 0.9
        self._projection_depth_offset = 1.1
        self._y_center = 0.95

    def _project_point(self, xyz: np.ndarray) -> Optional[Tuple[int, int]]:
        if not np.all(np.isfinite(xyz)):
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

    def render(
        self,
        joint_xyz: Dict[str, np.ndarray],
        last_pose_timestamps: Dict[str, Optional[float]],
        step_count: int,
    ) -> np.ndarray:
        frame = np.full((self._image_height, self._image_width, 3), 18, dtype=np.uint8)
        frame[:, :, 1] = 30
        frame[:, :, 2] = 24

        for y in range(0, self._image_height, 72):
            cv2.line(frame, (0, y), (self._image_width, y), (28, 40, 34), 1, lineType=cv2.LINE_AA)
        for x in range(0, self._image_width, 72):
            cv2.line(frame, (x, 0), (x, self._image_height), (28, 40, 34), 1, lineType=cv2.LINE_AA)

        if not any(timestamp is not None for timestamp in last_pose_timestamps.values()):
            cv2.putText(
                frame,
                "Waiting for VR keypoints...",
                (32, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (220, 240, 220),
                2,
                lineType=cv2.LINE_AA,
            )
            return frame

        projected: Dict[str, Optional[Tuple[int, int]]] = {}
        for joint_name in EGO_DEX_UPPER_LIMB_JOINTS:
            projected[joint_name] = self._project_point(joint_xyz[joint_name])

        for joint_a, joint_b in SKELETON_EDGES:
            point_a = projected.get(joint_a)
            point_b = projected.get(joint_b)
            if point_a is None or point_b is None:
                continue
            color = (255, 180, 80) if joint_a.startswith("left") else (80, 180, 255)
            cv2.line(frame, point_a, point_b, color, 2, lineType=cv2.LINE_AA)

        for joint_name in EGO_DEX_UPPER_LIMB_JOINTS:
            point = projected.get(joint_name)
            if point is None:
                continue
            color = (255, 210, 110) if joint_name.startswith("left") else (110, 210, 255)
            radius = 4 if joint_name.endswith("Hand") else 3
            cv2.circle(frame, point, radius, color, -1, lineType=cv2.LINE_AA)

        now_s = time.time()
        pulse = 0.5 + 0.5 * math.sin(now_s * 2.5)
        cv2.putText(
            frame,
            "Dummy WM oracle skeleton",
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


class DummyWMServer:
    """Testing WM service that renders an oracle-style skeleton from raw keypoints."""

    def __init__(self, dof: int = 16, width: int = 960, height: int = 720):
        self._dof = int(dof)
        self._width = int(width)
        self._height = int(height)
        self._renderer = _MinimalSkeletonRenderer(image_width=self._width, image_height=self._height)
        self._right_joint_state = np.zeros(self._dof, dtype=np.float32)
        self._left_joint_state = np.zeros(self._dof, dtype=np.float32)
        self._latest_keypoints: Dict[str, Optional[np.ndarray]] = {robots.LEFT: None, robots.RIGHT: None}
        self._last_pose_timestamps: Dict[str, Optional[float]] = {robots.LEFT: None, robots.RIGHT: None}
        self._latest_source: Dict[str, str] = {robots.LEFT: "none", robots.RIGHT: "none"}
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

    def _decode_payload(self, payload: Mapping[str, Any]) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        side = str(payload.get("hand_side", robots.RIGHT)).lower()
        if side not in {robots.LEFT, robots.RIGHT}:
            side = robots.RIGHT

        action = None
        if "joint_positions_rad" in payload:
            action = self._sanitize_action(payload.get("joint_positions_rad"))

        keypoints = self._sanitize_keypoints(payload.get("keypoints_xyz"))
        timestamp_s = payload.get("timestamp_s")
        timestamp = float(timestamp_s) if isinstance(timestamp_s, (int, float)) else None
        return side, action, keypoints, timestamp

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return (vector / norm).astype(np.float32)

    def _estimate_arm_chain(self, side: str, keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        hand = keypoints[HAND_KEYPOINT_INDEX["Hand"]]
        palm_center = np.mean(
            keypoints[
                [
                    HAND_KEYPOINT_INDEX["IndexFingerMetacarpal"],
                    HAND_KEYPOINT_INDEX["MiddleFingerMetacarpal"],
                    HAND_KEYPOINT_INDEX["RingFingerMetacarpal"],
                    HAND_KEYPOINT_INDEX["LittleFingerMetacarpal"],
                ]
            ],
            axis=0,
        )
        forearm_dir = self._normalize(hand - palm_center)
        side_dir = (
            np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            if side == robots.LEFT
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        forearm = hand + forearm_dir * 0.11
        arm = forearm + forearm_dir * 0.14 + side_dir * 0.03
        shoulder = arm + side_dir * 0.09 + np.array([0.0, 0.08, -0.02], dtype=np.float32)
        prefix = "left" if side == robots.LEFT else "right"
        return {
            f"{prefix}Shoulder": shoulder,
            f"{prefix}Arm": arm,
            f"{prefix}Forearm": forearm,
            f"{prefix}Hand": hand,
        }

    def _estimate_side_joint_xyz(self, side: str, keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        prefix = "left" if side == robots.LEFT else "right"
        joint_xyz = self._estimate_arm_chain(side, keypoints)
        for suffix, keypoint_index in HAND_KEYPOINT_INDEX.items():
            joint_xyz[f"{prefix}{suffix}"] = keypoints[keypoint_index].astype(np.float32)
        return joint_xyz

    def _build_joint_xyz(self) -> Dict[str, np.ndarray]:
        joint_xyz = {
            name: np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            for name in EGO_DEX_UPPER_LIMB_JOINTS
        }
        for side in (robots.LEFT, robots.RIGHT):
            keypoints = self._latest_keypoints[side]
            if keypoints is None:
                continue
            joint_xyz.update(self._estimate_side_joint_xyz(side, keypoints))
        return joint_xyz

    def _state_dict(self) -> Dict[str, Any]:
        return {
            "left_joint_state": self._left_joint_state.tolist(),
            "right_joint_state": self._right_joint_state.tolist(),
            "last_pose_timestamps": self._last_pose_timestamps,
            "last_reset_s": self._last_reset_s,
        }

    def _draw_status_overlay(self, frame: np.ndarray) -> None:
        lines = [
            f"R source={self._latest_source[robots.RIGHT]} ts={self._last_pose_timestamps[robots.RIGHT]}",
            f"L source={self._latest_source[robots.LEFT]} ts={self._last_pose_timestamps[robots.LEFT]}",
        ]
        y = self._height - 72
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
            joint_xyz=self._build_joint_xyz(),
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
        side, action, keypoints, timestamp_s = self._decode_payload(payload)
        if action is not None:
            if side == robots.LEFT:
                self._left_joint_state = action
            else:
                self._right_joint_state = action
        if keypoints is not None:
            self._latest_keypoints[side] = keypoints
            self._last_pose_timestamps[side] = timestamp_s if timestamp_s is not None else time.time()
            self._latest_source[side] = "keypoints"
        self._step_count += 1
        return self._encode_jpg(self._render())

    def reset(self, new: bool = True) -> Tuple[bytes, Mapping[str, Any]]:
        if new:
            self._right_joint_state = np.zeros(self._dof, dtype=np.float32)
            self._left_joint_state = np.zeros(self._dof, dtype=np.float32)
            self._latest_keypoints = {robots.LEFT: None, robots.RIGHT: None}
            self._last_pose_timestamps = {robots.LEFT: None, robots.RIGHT: None}
            self._latest_source = {robots.LEFT: "none", robots.RIGHT: "none"}
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
        }


class _DummyWMHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler_class, wm_server: DummyWMServer):
        super().__init__(server_address, request_handler_class)
        self.wm_server = wm_server


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
    http_server.serve_forever()


if __name__ == "__main__":
    main()
