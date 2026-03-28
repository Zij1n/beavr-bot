"""EgoDex teleop config wired to the remote WM backend path."""

import os
from dataclasses import dataclass, field

from beavr.teleop.common.configs.loader import Laterality, log_laterality_configuration
from beavr.teleop.components.detector.vr.keypoint_transform import TransformHandPositionCoords
from beavr.teleop.components.detector.vr.oculus import OculusVRHandDetector
from beavr.teleop.components.interface.robots.ego_dex_remote_wm_backend import (
    RemoteEgoDexWMBackend,
)
from beavr.teleop.components.interface.robots.ego_dex_robot import EgoDexRobot
from beavr.teleop.components.interface.robots.ego_dex_wm_client import WMClient
from beavr.teleop.components.operator.robots.leap_operator import LeapHandOperator
from beavr.teleop.configs.constants import ports, robots
from beavr.teleop.configs.robots import TeleopRobotConfig

EGO_DEX_HOST = "127.0.0.1"


@dataclass
class OculusVRHandDetectorCfg:
    host: str = EGO_DEX_HOST
    oculus_pub_port: int = ports.KEYPOINT_STREAM_PORT
    button_port: int = ports.RESOLUTION_BUTTON_PORT
    teleop_reset_port: int = ports.TELEOP_RESET_PORT
    hand_config: str = robots.BIMANUAL
    right_hand_port: int = ports.RIGHT_HAND_OCULUS_RECEIVER_PORT
    left_hand_port: int = ports.LEFT_HAND_OCULUS_RECEIVER_PORT

    def build(self):
        return OculusVRHandDetector(
            host=self.host,
            oculus_pub_port=self.oculus_pub_port,
            button_port=self.button_port,
            teleop_reset_port=self.teleop_reset_port,
            hand_config=self.hand_config,
            right_hand_port=self.right_hand_port,
            left_hand_port=self.left_hand_port,
        )


@dataclass
class TransformHandPositionCoordsCfg:
    host: str = EGO_DEX_HOST
    keypoint_sub_port: int = ports.KEYPOINT_STREAM_PORT
    keypoint_transform_pub_port: int = ports.KEYPOINT_TRANSFORM_PORT
    hand_side: str = robots.RIGHT
    moving_average_limit: int = 1

    def build(self):
        return TransformHandPositionCoords(
            host=self.host,
            keypoint_sub_port=self.keypoint_sub_port,
            keypoint_transform_pub_port=self.keypoint_transform_pub_port,
            hand_side=self.hand_side,
            moving_average_limit=self.moving_average_limit,
        )


@dataclass
class LeapHandOperatorCfg:
    host: str = EGO_DEX_HOST
    transformed_keypoints_port: int = ports.KEYPOINT_TRANSFORM_PORT
    joint_angle_subscribe_port: int = ports.JOINT_PUBLISHER_PORT
    joint_angle_publish_port: int = ports.CARTESIAN_COMMAND_PUBLISHER_PORT
    reset_publish_port: int = ports.TELEOP_RESET_PUBLISH_PORT
    hand_side: str = robots.RIGHT
    logging_config: dict = field(default_factory=lambda: {"enabled": False})
    finger_configs: dict = field(
        default_factory=lambda: {
            "freeze_index": False,
            "freeze_middle": False,
            "freeze_ring": False,
            "freeze_thumb": False,
            "no_index": False,
            "no_middle": False,
            "no_ring": False,
            "no_thumb": False,
            "three_dim": True,
        }
    )

    def build(self):
        return LeapHandOperator(
            host=self.host,
            transformed_keypoints_port=self.transformed_keypoints_port,
            joint_angle_subscribe_port=self.joint_angle_subscribe_port,
            joint_angle_publish_port=self.joint_angle_publish_port,
            reset_publish_port=self.reset_publish_port,
            finger_configs=self.finger_configs,
            logging_config=self.logging_config,
            hand_side=self.hand_side,
        )


@dataclass
class EgoDexRobotCfg:
    host: str = EGO_DEX_HOST
    keypoint_stream_port: int = ports.KEYPOINT_STREAM_PORT
    right_action_subscribe_port: int = ports.CARTESIAN_COMMAND_PUBLISHER_PORT
    left_action_subscribe_port: int = ports.CARTESIAN_COMMAND_PUBLISHER_PORT_LEFT
    right_joint_state_publish_port: int = ports.JOINT_PUBLISHER_PORT
    left_joint_state_publish_port: int = ports.JOINT_PUBLISHER_PORT_LEFT
    camera_publish_port: int = ports.CAM_PORT_OFFSET + ports.VIZ_PORT_OFFSET
    enable_right: bool = True
    enable_left: bool = True
    dof: int = 16
    fps: float = 30.0
    image_width: int = 960
    image_height: int = 720
    log_dir: str = "logs"
    log_prefix: str = "ego_dex"
    backend_cli_host: str = "127.0.0.1"
    backend_cli_command_port: int = ports.EGO_DEX_BACKEND_CLI_COMMAND_PORT
    backend_cli_status_port: int = ports.EGO_DEX_BACKEND_CLI_STATUS_PORT
    wm_scheme: str = "http"
    wm_hostname: str = "127.0.0.1"
    wm_port: int = 18080
    wm_timeout_s: float = 3.0
    wm_heartbeat_hz: float = 2.0
    step_distance_threshold: float = 0.22577618051049414
    hand_visualization_bind_host: str = "0.0.0.0"
    hand_visualization_port: int = 15102
    hand_visualization_fps: float = 15.0
    hand_visualization_point_size: float = 0.015

    def build(self):
        wm_scheme = os.getenv("EGO_DEX_WM_SCHEME", self.wm_scheme)
        wm_host = os.getenv("EGO_DEX_WM_HOST", self.wm_hostname)
        wm_port = int(os.getenv("EGO_DEX_WM_PORT", str(self.wm_port)))
        wm_timeout_s = float(os.getenv("EGO_DEX_WM_TIMEOUT_S", str(self.wm_timeout_s)))
        wm_heartbeat_hz = float(os.getenv("EGO_DEX_WM_HEARTBEAT_HZ", str(self.wm_heartbeat_hz)))
        backend_cli_host = os.getenv("EGO_DEX_BACKEND_CLI_HOST", self.backend_cli_host)
        backend_cli_command_port = int(
            os.getenv("EGO_DEX_BACKEND_CLI_COMMAND_PORT", str(self.backend_cli_command_port))
        )
        backend_cli_status_port = int(
            os.getenv("EGO_DEX_BACKEND_CLI_STATUS_PORT", str(self.backend_cli_status_port))
        )
        step_distance_threshold = float(
            os.getenv("EGO_DEX_STEP_DISTANCE_THRESHOLD", str(self.step_distance_threshold))
        )
        hand_visualization_enabled = os.getenv(
            "EGO_DEX_VISUALIZE_HANDS_ENABLE",
            os.getenv("EGO_DEX_FEATHER_ENABLE", "0"),
        ).strip().lower() in {"1", "true", "yes", "on"}
        hand_visualization_bind_host = os.getenv(
            "EGO_DEX_VISUALIZE_HANDS_BIND_HOST",
            os.getenv("EGO_DEX_FEATHER_BIND_HOST", self.hand_visualization_bind_host),
        )
        hand_visualization_port = int(
            os.getenv(
                "EGO_DEX_VISUALIZE_HANDS_PORT",
                os.getenv("EGO_DEX_FEATHER_PORT", str(self.hand_visualization_port)),
            )
        )
        hand_visualization_fps = float(
            os.getenv(
                "EGO_DEX_VISUALIZE_HANDS_FPS",
                os.getenv("EGO_DEX_FEATHER_FPS", str(self.hand_visualization_fps)),
            )
        )
        hand_visualization_point_size = float(
            os.getenv(
                "EGO_DEX_VISUALIZE_HANDS_POINT_SIZE",
                os.getenv("EGO_DEX_FEATHER_POINT_SIZE", str(self.hand_visualization_point_size)),
            )
        )

        backend = RemoteEgoDexWMBackend(
            wm_client=WMClient(hostname=wm_host, port=wm_port, timeout_s=wm_timeout_s, scheme=wm_scheme),
            dof=self.dof,
            heartbeat_hz=wm_heartbeat_hz,
            reset_frame_dir=self.log_dir,
            step_distance_threshold=step_distance_threshold,
            hand_visualization_bind_host=(
                hand_visualization_bind_host if hand_visualization_enabled else None
            ),
            hand_visualization_port=hand_visualization_port if hand_visualization_enabled else None,
            hand_visualization_fps=hand_visualization_fps,
            hand_visualization_point_size=hand_visualization_point_size,
        )

        return EgoDexRobot(
            host=self.host,
            keypoint_stream_port=self.keypoint_stream_port,
            right_action_subscribe_port=self.right_action_subscribe_port,
            left_action_subscribe_port=self.left_action_subscribe_port,
            right_joint_state_publish_port=self.right_joint_state_publish_port,
            left_joint_state_publish_port=self.left_joint_state_publish_port,
            camera_publish_port=self.camera_publish_port,
            backend=backend,
            enable_right=self.enable_right,
            enable_left=self.enable_left,
            dof=self.dof,
            fps=self.fps,
            image_width=self.image_width,
            image_height=self.image_height,
            log_dir=self.log_dir,
            log_prefix=self.log_prefix,
            backend_cli_host=backend_cli_host,
            backend_cli_command_port=backend_cli_command_port,
            backend_cli_status_port=backend_cli_status_port,
        )


@dataclass
@TeleopRobotConfig.register_subclass("ego_dex")
class EgoDexTeleopConfig:
    robot_name: str = "ego_dex"
    laterality: Laterality = Laterality.BIMANUAL
    detector: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    visualizers: list = field(default_factory=list)
    operators: list = field(default_factory=list)
    robots: list = field(default_factory=list)

    def __post_init__(self):
        self.laterality = Laterality.BIMANUAL
        log_laterality_configuration(self.laterality, self.robot_name)
        self._configure()

    def _configure(self):
        self.detector = [
            OculusVRHandDetectorCfg(
                host=EGO_DEX_HOST,
                hand_config=robots.BIMANUAL,
                right_hand_port=ports.RIGHT_HAND_OCULUS_RECEIVER_PORT,
                left_hand_port=ports.LEFT_HAND_OCULUS_RECEIVER_PORT,
            )
        ]

        self.transforms = [
            TransformHandPositionCoordsCfg(
                hand_side=robots.RIGHT,
                host=EGO_DEX_HOST,
                keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                keypoint_transform_pub_port=ports.KEYPOINT_TRANSFORM_PORT,
                moving_average_limit=1,
            ),
            TransformHandPositionCoordsCfg(
                hand_side=robots.LEFT,
                host=EGO_DEX_HOST,
                keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                keypoint_transform_pub_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                moving_average_limit=1,
            ),
        ]

        self.visualizers = []

        self.operators = [
            LeapHandOperatorCfg(
                host=EGO_DEX_HOST,
                transformed_keypoints_port=ports.KEYPOINT_TRANSFORM_PORT,
                joint_angle_subscribe_port=ports.JOINT_PUBLISHER_PORT,
                joint_angle_publish_port=ports.CARTESIAN_COMMAND_PUBLISHER_PORT,
                reset_publish_port=ports.TELEOP_RESET_PUBLISH_PORT,
                hand_side=robots.RIGHT,
                logging_config={"enabled": False},
            ),
            LeapHandOperatorCfg(
                host=EGO_DEX_HOST,
                transformed_keypoints_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                joint_angle_subscribe_port=ports.JOINT_PUBLISHER_PORT_LEFT,
                joint_angle_publish_port=ports.CARTESIAN_COMMAND_PUBLISHER_PORT_LEFT,
                reset_publish_port=ports.TELEOP_RESET_PUBLISH_PORT,
                hand_side=robots.LEFT,
                logging_config={"enabled": False},
            ),
        ]

        self.robots = [
            EgoDexRobotCfg(
                host=EGO_DEX_HOST,
                keypoint_stream_port=ports.KEYPOINT_STREAM_PORT,
                right_action_subscribe_port=ports.CARTESIAN_COMMAND_PUBLISHER_PORT,
                left_action_subscribe_port=ports.CARTESIAN_COMMAND_PUBLISHER_PORT_LEFT,
                right_joint_state_publish_port=ports.JOINT_PUBLISHER_PORT,
                left_joint_state_publish_port=ports.JOINT_PUBLISHER_PORT_LEFT,
                camera_publish_port=ports.CAM_PORT_OFFSET + ports.VIZ_PORT_OFFSET,
                enable_right=True,
                enable_left=True,
                dof=16,
                fps=30.0,
                image_width=960,
                image_height=720,
                log_dir="logs",
                log_prefix="ego_dex",
            )
        ]
