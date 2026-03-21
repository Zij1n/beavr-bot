from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import cv2
import numpy as np

from beavr.teleop.configs.constants import robots

from .ego_dex_wm_client import WMClient

logger = logging.getLogger(__name__)


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
        heartbeat_hz: float = 15.0,
        bootstrap_reset: bool = True,
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
            "keypoints_xyz": input_payload.get("keypoints_xyz"),
            "is_relative": bool(input_payload.get("is_relative", False)),
            "world_frame": input_payload.get("world_frame"),
            "joint_order": input_payload.get("joint_order"),
            "joint_transforms_world": input_payload.get("joint_transforms_world"),
            "joint_positions_rad": (
                action_payload.get("joint_positions_rad") if action_payload is not None else None
            ),
            "action_timestamp_s": action_payload.get("timestamp_s") if action_payload is not None else None,
        }

    def _build_snapshot_payload(self) -> Optional[dict[str, Any]]:
        hands = {
            robots.LEFT: self._build_hand_payload(robots.LEFT),
            robots.RIGHT: self._build_hand_payload(robots.RIGHT),
        }
        if hands[robots.LEFT] is None and hands[robots.RIGHT] is None:
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
        return payload

    def reset(self, new: bool = True) -> Tuple[Optional[np.ndarray], Mapping[str, Any]]:
        obs_jpg, state = self._wm_client.reset(new=new)
        self._latest_state = state
        self._latest_obs_frame = self._decode_jpg(obs_jpg)
        self._last_obs_wall_time_s = time.time()
        self._last_event_record = {
            "event": "remote_wm_reset",
            "received_at_s": self._last_obs_wall_time_s,
            "new": bool(new),
            "state_keys": sorted(state.keys()),
        }
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
        payload = self._build_snapshot_payload()
        if payload is None:
            return
        now_s = time.time()
        if now_s - self._last_request_wall_time_s < self._step_period_s:
            return
        self._last_request_wall_time_s = now_s
        try:
            self._request_step(payload, event_name="remote_wm_step")
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

    def pop_pose_record(self) -> Optional[dict]:
        record = self._last_event_record
        self._last_event_record = None
        return record
