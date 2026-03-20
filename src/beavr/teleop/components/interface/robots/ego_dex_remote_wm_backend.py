from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import cv2
import numpy as np

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
        heartbeat_hz: float = 10.0,
        bootstrap_reset: bool = True,
    ):
        self._wm_client = wm_client
        self._dof = int(dof)
        self._default_joint_state = np.zeros(self._dof, dtype=np.float32)
        self._heartbeat_period_s = 1.0 / max(float(heartbeat_hz), 1.0)
        self._latest_obs_frame: Optional[np.ndarray] = None
        self._latest_state: Mapping[str, Any] = {}
        self._last_event_record: Optional[dict] = None
        self._last_obs_wall_time_s = 0.0
        self._last_payload: Optional[dict[str, Any]] = None
        self._last_action_by_side: dict[str, dict[str, Any]] = {}

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

        hand_side = str(payload.get("hand_side", "right"))
        joint_positions = payload.get("joint_positions_rad")
        if joint_positions is not None:
            self._latest_state[f"{hand_side}_joint_state"] = joint_positions

        self._last_event_record = {
            "event": event_name,
            "received_at_s": now_s,
            "hand_side": hand_side,
            "timestamp_s": payload.get("timestamp_s"),
            "has_keypoints": "keypoints_xyz" in payload,
            "has_joint_transforms_world": "joint_transforms_world" in payload,
            "world_frame": payload.get("world_frame"),
            "is_relative": payload.get("is_relative"),
            "action_dof": len(joint_positions) if isinstance(joint_positions, list) else None,
        }

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

        payload: dict[str, Any] = {
            "source": "xr_hand_joint_poses" if joint_transforms_world is not None else "vr_keypoints",
            "hand_side": side,
            "timestamp_s": float(timestamp_s),
            "keypoints_xyz": keypoints.tolist(),
            "is_relative": bool(is_relative),
        }
        if world_frame:
            payload["world_frame"] = str(world_frame)
        if joint_order is not None:
            payload["joint_order"] = list(joint_order)
        payload["joint_transforms_world"] = (
            dict(joint_transforms_world) if joint_transforms_world is not None else None
        )
        latest_action = self._last_action_by_side.get(side)
        if latest_action is not None:
            payload["joint_positions_rad"] = latest_action["joint_positions_rad"]

        self._last_payload = payload
        self._request_step(payload, event_name="remote_wm_keypoints")

    def step(self) -> None:
        if self._last_payload is None:
            return
        if time.time() - self._last_obs_wall_time_s < self._heartbeat_period_s:
            return
        try:
            self._request_step(self._last_payload, event_name="remote_wm_heartbeat")
        except Exception:
            logger.exception("Remote WM heartbeat failed")

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
