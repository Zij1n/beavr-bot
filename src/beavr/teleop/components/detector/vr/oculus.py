import logging
import time
from typing import Optional, Sequence, Tuple, Union

import zmq

from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.utils import create_pull_socket
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import (
    ButtonEvent,
    InputFrame,
    SessionCommand,
)
from beavr.teleop.configs.constants import network, robots

logger = logging.getLogger(__name__)


UNITY_XR_WORLD_FRAME = "unity_xr_world"
OCULUS_JOINT_ORDER: Tuple[str, ...] = (
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


class OculusVRHandDetector(Component):
    """
    Unified OculusVRHandDetector that can handle left, right, or bimanual hand detection.

    This class dynamically configures itself based on the provided hand configuration,
    eliminating the need for separate single-hand and bimanual detector classes.
    """

    def __init__(
        self,
        host: str,
        oculus_pub_port: int,
        button_port: int,
        teleop_reset_port: int,
        hand_config: Union[str, str] = robots.RIGHT,
        right_hand_port: Optional[int] = None,
        left_hand_port: Optional[int] = None,
    ):
        """
        Initialize the unified OculusVRHandDetector component.

        Args:
            host: The host address of the Oculus VR headset.
            oculus_pub_port: The port number for publishing keypoint data.
            button_port: The port number for button events.
            teleop_reset_port: The port number for teleop reset commands.
            hand_config: Configuration mode - 'left', 'right', or 'bimanual'
            right_hand_port: Port for right hand data (required for right/bimanual)
            left_hand_port: Port for left hand data (required for left/bimanual)
        """
        self.notify_component_start(robots.VR_DETECTOR)

        self.host = host
        self.oculus_pub_port = oculus_pub_port
        self.button_port = button_port
        self.teleop_reset_port = teleop_reset_port
        self.hand_config = hand_config

        # Validate and set hand ports based on configuration
        self._configure_hand_ports(right_hand_port, left_hand_port)

        # Initialize sockets based on configuration
        self._initialize_sockets()

        # Initialize publisher and timing
        self.publisher_manager = ZMQPublisherManager.get_instance()
        self.timer = FrequencyTimer(robots.VR_FREQ)
        self.last_received = dict.fromkeys(self.sockets, 0)

    def _configure_hand_ports(self, right_hand_port: Optional[int], left_hand_port: Optional[int]):
        """Configure hand ports based on the hand configuration."""
        self.hand_ports = {}

        if self.hand_config in [robots.RIGHT, robots.BIMANUAL]:
            if right_hand_port is None:
                right_hand_port = network.RIGHT_HAND_PORT
            self.hand_ports[robots.RIGHT] = right_hand_port

        if self.hand_config in [robots.LEFT, robots.BIMANUAL]:
            if left_hand_port is None:
                left_hand_port = network.LEFT_HAND_PORT
            self.hand_ports[robots.LEFT] = left_hand_port

    def _initialize_sockets(self):
        """Initialize sockets based on hand configuration."""
        self.sockets = {}

        # Create hand-specific keypoint sockets
        for hand_side, port in self.hand_ports.items():
            socket_key = f"{robots.KEYPOINTS}_{hand_side}"
            self.sockets[socket_key] = create_pull_socket(self.host, port)

        # Shared sockets for button and pause (only one instance needed)
        self.sockets[robots.BUTTON] = create_pull_socket(self.host, self.button_port)
        self.sockets[robots.PAUSE] = create_pull_socket(self.host, self.teleop_reset_port)

    @staticmethod
    def _quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float):
        quat = [float(qx), float(qy), float(qz), float(qw)]
        norm = sum(value * value for value in quat) ** 0.5
        if norm < 1e-8:
            return (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )

        x, y, z, w = (value / norm for value in quat)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        )

    def _pose_to_transform(
        self, pose_values: Sequence[float]
    ) -> Tuple[Tuple[float, float, float, float], ...]:
        px, py, pz, qx, qy, qz, qw = (float(value) for value in pose_values)
        rot = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
        return (
            (rot[0][0], rot[0][1], rot[0][2], px),
            (rot[1][0], rot[1][1], rot[1][2], py),
            (rot[2][0], rot[2][1], rot[2][2], pz),
            (0.0, 0.0, 0.0, 1.0),
        )

    @staticmethod
    def _split_message(data_str: str) -> tuple[str, str]:
        mode, separator, body = data_str.partition(":")
        if not separator:
            raise ValueError("Hand payload is missing mode separator ':'.")
        body = body.strip()
        if body.endswith(":"):
            body = body[:-1]
        return mode.strip().lower(), body

    def _parse_legacy_input_frame(self, data_str: str, hand_side: str) -> InputFrame:
        mode, body = self._split_message(data_str)
        values = []

        for coord in body.split("|"):
            coord = coord.strip()
            if not coord:
                continue
            parts = coord.split(",")
            if len(parts) < 3:
                raise ValueError(f"Legacy joint entry must have at least 3 floats, got: {coord!r}")
            values.extend(float(val) for val in parts[:3])

        return InputFrame(
            timestamp_s=time.time(),
            hand_side=hand_side,
            keypoints=values,
            is_relative=mode != robots.ABSOLUTE,
            frame_vectors=None,
        )

    def _parse_pose_input_frame(self, data_str: str, hand_side: str) -> InputFrame:
        _, body = self._split_message(data_str)
        joint_entries = [entry.strip() for entry in body.split("|") if entry.strip()]
        if len(joint_entries) != len(OCULUS_JOINT_ORDER):
            raise ValueError(
                f"Expected {len(OCULUS_JOINT_ORDER)} joint entries, got {len(joint_entries)}."
            )

        keypoints = []
        joint_transforms_world = {}
        for joint_name, joint_entry in zip(OCULUS_JOINT_ORDER, joint_entries, strict=True):
            values = [float(value) for value in joint_entry.split(",")]
            if len(values) != 7:
                raise ValueError(f"Pose joint entry must have 7 floats, got: {joint_entry!r}")
            keypoints.extend(values[:3])
            joint_transforms_world[joint_name] = self._pose_to_transform(values)

        return InputFrame(
            timestamp_s=time.time(),
            hand_side=hand_side,
            keypoints=keypoints,
            is_relative=False,
            frame_vectors=None,
            world_frame=UNITY_XR_WORLD_FRAME,
            joint_order=OCULUS_JOINT_ORDER,
            joint_transforms_world=joint_transforms_world,
        )

    def _parse_input_frame(self, data: bytes, hand_side: str) -> InputFrame:
        data_str = data.decode().strip()
        _, body = self._split_message(data_str)
        first_joint = next((entry.strip() for entry in body.split("|") if entry.strip()), "")
        field_count = len(first_joint.split(",")) if first_joint else 0

        if field_count == 7:
            return self._parse_pose_input_frame(data_str, hand_side=hand_side)
        if field_count == 3:
            return self._parse_legacy_input_frame(data_str, hand_side=hand_side)
        raise ValueError(
            "Unsupported hand payload: "
            f"expected 3 or 7 floats per joint entry, got {field_count}. "
            f"preview={data_str[:160]!r}"
        )

    @staticmethod
    def _is_ignorable_hand_message(data: bytes) -> bool:
        data_str = data.decode(errors="replace").strip()
        return data_str.startswith("DIAGNOSTIC_TEST_")

    def _receive_data(self, socket_name):
        """Receive data from a socket."""
        try:
            data = self.sockets[socket_name].recv(zmq.NOBLOCK)
            self.last_received[socket_name] = time.time()
            return data
        except zmq.Again:
            return None

    def stream(self):
        """Main streaming loop for unified VR hand detection."""
        logger.info(f"Starting VR hand detection with configuration: {self.hand_config}")

        while True:
            self.timer.start_loop()

            # Process keypoint data for all configured hands
            for hand_side in self.hand_ports:
                socket_key = f"{robots.KEYPOINTS}_{hand_side}"
                keypoint_data = self._receive_data(socket_key)

                if keypoint_data is not None:
                    if self._is_ignorable_hand_message(keypoint_data):
                        continue
                    try:
                        input_frame = self._parse_input_frame(keypoint_data, hand_side=hand_side)
                    except ValueError as exc:
                        logger.warning(
                            "Skipping malformed VR hand payload for side=%s: %s",
                            hand_side,
                            exc,
                        )
                        continue
                    except Exception:
                        logger.exception("Failed to parse VR hand payload for side=%s", hand_side)
                        continue

                    # TODO: We really only need to publish ONCE!
                    # We can store all information in a single schema table

                    self.publisher_manager.publish(
                        host=self.host,
                        port=self.oculus_pub_port,
                        topic=hand_side,
                        data=input_frame,
                    )

            # Process and publish button state (shared across hands)
            if button_data := self._receive_data(robots.BUTTON):
                # For button events, use the first configured hand side as the source
                # or 'right' as default for bimanual setups
                hand_side = (
                    robots.RIGHT if robots.RIGHT in self.hand_ports else list(self.hand_ports.keys())[0]
                )

                self.publisher_manager.publish(
                    host=self.host,
                    port=self.oculus_pub_port,
                    topic=robots.BUTTON,
                    data=ButtonEvent(
                        timestamp_s=time.time(),
                        hand_side=hand_side,
                        name=robots.BUTTON,
                        value=robots.ARM_LOW_RESOLUTION
                        if button_data == b"Low"
                        else robots.ARM_HIGH_RESOLUTION,
                    ),
                )

            # Process and publish pause state (shared across hands)
            if pause_data := self._receive_data(robots.PAUSE):
                self.publisher_manager.publish(
                    host=self.host,
                    port=self.oculus_pub_port,
                    topic=robots.PAUSE,
                    data=SessionCommand(
                        timestamp_s=time.time(),
                        command="resume" if pause_data == b"Low" else "pause",
                    ),
                )

            self.timer.end_loop()

        # TODO: We need better cleanup than this
        # Cleanup sockets on exit
        for name, socket in self.sockets.items():
            socket.close()
            logger.info(f"Closed {name} socket")
        logger.info("Stopped VR hand detection process.")
