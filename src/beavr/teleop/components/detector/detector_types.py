from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence, Tuple

"""
Detector → Operator contracts: minimal fields required by operators.

Notes:
- Units: positions in meters, orientations as XYZW unit quaternions,
  velocities in m/s and rad/s, time is seconds (monotonic).
- "hand_side" distinguishes left/right streams when bimanual.
"""


HandSide = Literal["left", "right"]


@dataclass(frozen=True)
class ButtonEvent:
    """Discrete button input from VR device.

    - name: logical button identifier (e.g., "trigger", "menu").
    - value: pressed/released boolean or small integer payload.
    """

    timestamp_s: float
    hand_side: HandSide
    name: str
    value: int


@dataclass(frozen=True)
class SessionCommand:
    """Session-level commands emitted by the detector.

    Common commands: "pause", "resume", "reset", "home".
    """

    timestamp_s: float
    command: Literal["pause", "resume", "reset", "home"]


@dataclass(frozen=True)
class InputFrame:
    """Raw per-frame inputs necessary for retargeting.

    - hand_side: stream source side.
    - keypoints: flattened (N,3) landmarks as a sequence of triples.
    - is_relative: whether keypoints are relative to wrist (1) or absolute (0),
      matching current detector output semantics.
    - frame_vectors: optional 3 orthonormal vectors (x,y,z) in absolute mode.
    - world_frame: name of the fixed frame for absolute joint poses, if available.
    - joint_order: ordered joint names matching detector payload indices.
    - joint_transforms_world: optional 4x4 world-frame pose for each tracked joint.
    """

    timestamp_s: float
    hand_side: HandSide
    keypoints: Sequence[Tuple[float, float, float]]
    is_relative: bool
    frame_vectors: Optional[
        Tuple[
            Tuple[float, float, float],
            Tuple[float, float, float],
            Tuple[float, float, float],
        ]
    ] = None
    world_frame: Optional[str] = None
    joint_order: Optional[Sequence[str]] = None
    joint_transforms_world: Optional[Mapping[str, Tuple[Tuple[float, ...], ...]]] = None


__all__ = [
    "HandSide",
    "ButtonEvent",
    "SessionCommand",
    "InputFrame",
]
