import numpy as np

from beavr.teleop.components.interface.robots.ego_dex_remote_wm_backend import (
    RemoteEgoDexWMBackend,
    _FLIP_Z_4X4,
    _LEFT_STEP_LOCAL_POST_Z_FLIP_4X4,
    _RIGHT_STEP_LOCAL_POST_Z_FLIP_4X4,
)
from beavr.teleop.configs.constants import robots


def _make_transform(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def test_step_payload_applies_side_specific_local_post_rotation_after_z_flip():
    backend = RemoteEgoDexWMBackend(wm_client=object(), bootstrap_reset=False)
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    transform = _make_transform(rotation, (0.12, -0.34, 0.56))
    keypoints = [[0.12, -0.34, 0.56]]

    for side in (robots.LEFT, robots.RIGHT):
        backend._latest_input_by_side[side] = {
            "source": "xr_hand_joint_poses",
            "timestamp_s": 1.23,
            "keypoints_xyz": keypoints,
            "is_relative": False,
            "world_frame": "unity_xr_world",
            "joint_order": ["wrist"],
            "joint_transforms_world": {"wrist": transform.tolist()},
        }

    left_payload = backend._build_current_input_step_hand_payload(robots.LEFT)
    right_payload = backend._build_current_input_step_hand_payload(robots.RIGHT)

    left_transform = np.asarray(left_payload["joint_transforms_world"]["wrist"], dtype=np.float32)
    right_transform = np.asarray(right_payload["joint_transforms_world"]["wrist"], dtype=np.float32)

    expected_base = _FLIP_Z_4X4 @ transform @ _FLIP_Z_4X4
    expected_left = expected_base @ _LEFT_STEP_LOCAL_POST_Z_FLIP_4X4
    expected_right = expected_base @ _RIGHT_STEP_LOCAL_POST_Z_FLIP_4X4

    np.testing.assert_allclose(left_transform, expected_left, atol=1e-6)
    np.testing.assert_allclose(right_transform, expected_right, atol=1e-6)
    np.testing.assert_allclose(left_transform[:3, 3], expected_base[:3, 3], atol=1e-6)
    np.testing.assert_allclose(right_transform[:3, 3], expected_base[:3, 3], atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(left_payload["keypoints_xyz"], dtype=np.float32),
        np.array([[0.12, -0.34, -0.56]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(right_payload["keypoints_xyz"], dtype=np.float32),
        np.array([[0.12, -0.34, -0.56]], dtype=np.float32),
        atol=1e-6,
    )
