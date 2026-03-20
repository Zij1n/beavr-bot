import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from beavr.teleop.common.network.publisher import (
    ZMQCompressedImageTransmitter,
    ZMQPublisherManager,
)
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import InputFrame
from beavr.teleop.components.interface.robots.ego_dex_remote_wm_backend import RobotActionCommand
from beavr.teleop.components.operator.operator_types import JointTarget
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class ActionDecoder:
    """Convert action payloads into typed 16-DoF joint commands with dedupe."""

    def __init__(self, enabled_sides: Sequence[str]):
        self._last_action_keys: Dict[str, Optional[Tuple[Any, ...]]] = dict.fromkeys(enabled_sides, None)

    def decode_unique(
        self, subscriber_side: str, msg: Any
    ) -> Optional[Tuple[Optional[RobotActionCommand], dict]]:
        hand_side = subscriber_side
        timestamp_s = None
        positions = None

        if isinstance(msg, JointTarget):
            hand_side = msg.hand_side
            timestamp_s = msg.timestamp_s
            positions = np.asarray(msg.joint_positions_rad, dtype=np.float32)
        elif isinstance(msg, dict) and "joint_positions_rad" in msg:
            hand_side = msg.get("hand_side", subscriber_side)
            timestamp_s = msg.get("timestamp_s")
            positions = np.asarray(msg["joint_positions_rad"], dtype=np.float32)
        else:
            try:
                positions = np.asarray(msg, dtype=np.float32)
            except Exception:
                return None

        positions = positions.reshape(-1).astype(np.float32)
        finite_mask = np.isfinite(positions)
        nan_count = int((~finite_mask).sum())
        sanitized_positions = [
            float(value) if bool(is_finite) else None
            for value, is_finite in zip(positions.tolist(), finite_mask.tolist(), strict=False)
        ]

        dedupe_key = (timestamp_s, tuple(sanitized_positions))
        if self._last_action_keys.get(subscriber_side) == dedupe_key:
            return None
        self._last_action_keys[subscriber_side] = dedupe_key

        if hand_side not in (robots.LEFT, robots.RIGHT):
            hand_side = subscriber_side

        record = {
            "event": "action",
            "received_at_s": time.time(),
            "timestamp_s": timestamp_s,
            "hand_side": hand_side,
            "joint_positions_rad": sanitized_positions,
            "is_valid": nan_count == 0,
            "nan_count": nan_count,
        }

        action_command: Optional[RobotActionCommand] = None
        if nan_count == 0:
            action_command = RobotActionCommand(
                hand_side=hand_side,
                timestamp_s=timestamp_s,
                joint_positions_rad=positions,
            )
        return action_command, record


class EgoDexSessionLogger:
    def __init__(self, log_dir: str, log_prefix: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(log_dir, f"{log_prefix}_{timestamp}.jsonl")
        self._log_file = open(self._log_path, "a", buffering=1)
        logger.info("EgoDex logger writing to: %s", self._log_path)
        self.log(
            {
                "event": "session_start",
                "received_at_s": time.time(),
                "design": "ego_dex_remote_wm",
                "observation_driver": "raw_vr_keypoints",
                "observation_transport": "wm_client_jpg",
                "action_schema": "joint_positions_rad[16]",
                "keypoint_schema": f"{robots.OCULUS_NUM_KEYPOINTS}x3",
                "pose_schema": f"{robots.OCULUS_NUM_KEYPOINTS}x4x4 world-frame hand poses when detector mode=absolute",
            }
        )

    def log(self, record: dict) -> None:
        self._log_file.write(json.dumps(record) + "\n")


class EgoDexRobot(Component):
    """Teleop main loop that forwards inputs to the remote WM backend."""

    def __init__(
        self,
        host: str,
        keypoint_stream_port: int,
        right_action_subscribe_port: int,
        left_action_subscribe_port: int,
        right_joint_state_publish_port: int,
        left_joint_state_publish_port: int,
        camera_publish_port: int,
        backend: Any,
        enable_right: bool = True,
        enable_left: bool = True,
        dof: int = 16,
        fps: float = 30.0,
        image_width: int = 960,
        image_height: int = 720,
        log_dir: str = "logs",
        log_prefix: str = "ego_dex",
    ):
        self.notify_component_start("ego_dex robot")

        self._host = host
        self._backend = backend
        self._fps = max(float(fps), 1.0)
        self._image_width = int(image_width)
        self._image_height = int(image_height)
        self._dof = int(dof)
        self._default_joint_state = np.zeros(self._dof, dtype=np.float32)
        self._warned_invalid_backend_frame = False
        self._warned_missing_backend_frame = False

        self._publisher = ZMQPublisherManager.get_instance()
        self._camera_tx = ZMQCompressedImageTransmitter(host=host, port=camera_publish_port)

        self._enabled_sides: list[str] = []
        self._action_subscribers: Dict[str, ZMQSubscriber] = {}
        self._keypoint_subscribers: Dict[str, ZMQSubscriber] = {}
        self._joint_state_ports: Dict[str, int] = {}
        self._last_keypoint_keys: Dict[str, Optional[Tuple[Any, ...]]] = {}

        if enable_right:
            self._enabled_sides.append(robots.RIGHT)
            self._action_subscribers[robots.RIGHT] = ZMQSubscriber(
                host=host,
                port=right_action_subscribe_port,
                topic="joint_angles",
            )
            self._keypoint_subscribers[robots.RIGHT] = ZMQSubscriber(
                host=host,
                port=keypoint_stream_port,
                topic=robots.RIGHT,
                message_type=InputFrame,
            )
            self._joint_state_ports[robots.RIGHT] = right_joint_state_publish_port
            self._last_keypoint_keys[robots.RIGHT] = None

        if enable_left:
            self._enabled_sides.append(robots.LEFT)
            self._action_subscribers[robots.LEFT] = ZMQSubscriber(
                host=host,
                port=left_action_subscribe_port,
                topic="joint_angles",
            )
            self._keypoint_subscribers[robots.LEFT] = ZMQSubscriber(
                host=host,
                port=keypoint_stream_port,
                topic=robots.LEFT,
                message_type=InputFrame,
            )
            self._joint_state_ports[robots.LEFT] = left_joint_state_publish_port
            self._last_keypoint_keys[robots.LEFT] = None

        self._action_decoder = ActionDecoder(self._enabled_sides)
        self._logger = EgoDexSessionLogger(log_dir=log_dir, log_prefix=log_prefix)

    def _sanitize_joint_state(self, state: Optional[np.ndarray]) -> np.ndarray:
        if state is None:
            return self._default_joint_state
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        state_array = np.nan_to_num(state_array, nan=0.0, posinf=0.0, neginf=0.0)
        if state_array.size == self._dof:
            return state_array
        if state_array.size > self._dof:
            return state_array[: self._dof]
        padded = np.zeros(self._dof, dtype=np.float32)
        padded[: state_array.size] = state_array
        return padded

    @staticmethod
    def _reshape_keypoints(raw_keypoints: Any) -> Optional[np.ndarray]:
        keypoint_array = np.asarray(raw_keypoints, dtype=np.float32)
        expected_values = robots.OCULUS_NUM_KEYPOINTS * 3
        if keypoint_array.size != expected_values:
            return None
        return keypoint_array.reshape(robots.OCULUS_NUM_KEYPOINTS, 3)

    def _build_placeholder_frame(self) -> np.ndarray:
        frame = np.full((self._image_height, self._image_width, 3), 24, dtype=np.uint8)
        cv2.putText(
            frame,
            "No backend observation frame",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (225, 235, 225),
            2,
            lineType=cv2.LINE_AA,
        )
        return frame

    def _poll_actions(self) -> None:
        for side, subscriber in self._action_subscribers.items():
            msg = subscriber.recv_keypoints()
            if msg is None:
                continue

            decoded = self._action_decoder.decode_unique(side, msg)
            if decoded is None:
                continue

            action_command, record = decoded
            if action_command is not None:
                try:
                    self._backend.send_action(action_command)
                except Exception:
                    logger.exception("Failed to send action to backend for side=%s", side)
            self._logger.log(record)

    def _poll_keypoints(self) -> None:
        for side, subscriber in self._keypoint_subscribers.items():
            msg = subscriber.recv_keypoints()
            if not isinstance(msg, InputFrame):
                continue

            dedupe_key = (msg.timestamp_s, bool(msg.is_relative), msg.world_frame)
            if self._last_keypoint_keys.get(side) == dedupe_key:
                continue
            self._last_keypoint_keys[side] = dedupe_key

            keypoints = self._reshape_keypoints(msg.keypoints)
            if keypoints is None:
                continue

            try:
                self._backend.on_keypoints(
                    side,
                    keypoints,
                    msg.timestamp_s,
                    is_relative=bool(msg.is_relative),
                    world_frame=msg.world_frame,
                    joint_order=msg.joint_order,
                    joint_transforms_world=msg.joint_transforms_world,
                )
            except Exception:
                logger.exception("Backend on_keypoints failed for side=%s", side)

        try:
            pose_record = self._backend.pop_pose_record()
        except Exception:
            logger.exception("Backend pose logging hook failed")
            pose_record = None

        if pose_record is not None:
            self._logger.log(pose_record)

    def _publish_joint_states(self) -> None:
        for side, port in self._joint_state_ports.items():
            try:
                joint_state = self._backend.get_joint_state(side)
            except Exception:
                logger.exception("Failed to get backend joint state for side=%s", side)
                joint_state = None

            self._publisher.publish(
                host=self._host,
                port=port,
                topic="joint_angles",
                data=self._sanitize_joint_state(joint_state),
            )

    @staticmethod
    def _coerce_frame(frame: Any) -> Optional[np.ndarray]:
        frame_array = np.asarray(frame)
        if frame_array.ndim not in (2, 3):
            return None
        if frame_array.dtype != np.uint8:
            frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
        if frame_array.ndim == 2:
            return cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR)
        if frame_array.shape[2] == 1:
            return cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR)
        if frame_array.shape[2] == 3:
            return frame_array
        if frame_array.shape[2] == 4:
            return cv2.cvtColor(frame_array, cv2.COLOR_BGRA2BGR)
        return None

    def _publish_observation(self) -> None:
        try:
            backend_frame = self._backend.get_camera_frame()
        except Exception:
            logger.exception("Failed to fetch camera frame from backend")
            backend_frame = None

        if backend_frame is None:
            if not self._warned_missing_backend_frame:
                logger.warning("Backend returned no camera frame; publishing placeholder.")
                self._warned_missing_backend_frame = True
            frame = self._build_placeholder_frame()
        else:
            frame = self._coerce_frame(backend_frame)
            if frame is None:
                if not self._warned_invalid_backend_frame:
                    logger.warning("Backend frame shape is invalid; publishing placeholder.")
                    self._warned_invalid_backend_frame = True
                frame = self._build_placeholder_frame()

        self._camera_tx.send_image(frame)

    def step(self) -> None:
        self._poll_actions()
        self._poll_keypoints()
        try:
            self._backend.step()
        except Exception:
            logger.exception("Backend step failed")
        self._publish_joint_states()
        self._publish_observation()

    def stream(self):
        period_s = 1.0 / self._fps
        while True:
            tick_start = time.perf_counter()
            self.step()
            elapsed_s = time.perf_counter() - tick_start
            if elapsed_s < period_s:
                time.sleep(period_s - elapsed_s)
