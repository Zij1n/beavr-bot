# EgoDex WM Payload Specification

## Purpose

This document describes the JSON payload sent from the EgoDex teleoperation stack to a world-model service through `WMClient`.

It is written for a backend developer with zero context of this repository.

It explains:

- how the WM service is called
- what JSON fields are sent on each update
- which fields are authoritative
- which fields are auxiliary
- what the WM service must return
- what assumptions the teleop side makes about state, timing, and idempotency

This document describes the **WM-facing contract**.
It does **not** describe the raw Quest / Unity socket format.
The raw headset payload is parsed inside the teleop process first, then converted into the JSON described here.

## System Overview

At runtime there are four relevant pieces:

1. VR / XR hand tracking app
2. EgoDex teleop process
3. `WMClient`
4. WM service

The WM service does not talk directly to the headset app.
It only receives HTTP JSON requests from `WMClient`.

The teleop side maintains the transport and sends a `wm_step` request whenever a fresh hand update arrives, plus periodic heartbeat replays of the latest payload to keep the observation stream alive.

## Transport Contract

The teleop side calls two HTTP endpoints:

- `POST /wm_step`
- `POST /reset`

Implementation references:

- [ego_dex_wm_client.py](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_wm_client.py)
- [ego_dex_remote_wm_backend.py](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_remote_wm_backend.py)

Request content type:

- `Content-Type: application/json`

Preferred response content type:

- `image/jpeg`

## High-Level Semantics

Each `wm_step` request is a **single-hand trigger update**.

That means:

- `hand_side` tells you which hand caused this update
- the payload may include the latest action for that same hand
- the WM service is expected to maintain world state across calls
- the WM service may render both hands even though only one hand triggered the request

The teleop side does not send a full scene snapshot on every call.
The WM service should be stateful.

## Primary Input Modes

There are two practical payload shapes.

### Primary mode: XR hand joint poses

This is the main path for the real WM backend.

In this mode, the request contains:

- `joint_transforms_world`
- `joint_order`
- `world_frame`
- `keypoints_xyz`
- optional `joint_positions_rad`

This mode is identified by:

- `source = "xr_hand_joint_poses"`

### Fallback mode: keypoints only

This is a degraded fallback for older detector payloads.

In this mode, the request contains:

- `keypoints_xyz`
- optional `joint_positions_rad`
- no usable `joint_transforms_world`

This mode is identified by:

- `source = "vr_keypoints"`

A real world-model backend should treat `xr_hand_joint_poses` as the authoritative path.

## `POST /wm_step`

### Request Schema

Current live request schema:

```json
{
  "source": "xr_hand_joint_poses",
  "hand_side": "right",
  "timestamp_s": 1774005000.123,
  "keypoints_xyz": [[0.10, 1.20, 0.30], [0.11, 1.21, 0.31]],
  "is_relative": false,
  "world_frame": "unity_xr_world",
  "joint_order": [
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
    "little_tip"
  ],
  "joint_transforms_world": {
    "wrist": [[1, 0, 0, 0.10], [0, 1, 0, 1.20], [0, 0, 1, 0.30], [0, 0, 0, 1]],
    "palm": [[1, 0, 0, 0.11], [0, 1, 0, 1.21], [0, 0, 1, 0.31], [0, 0, 0, 1]]
  },
  "joint_positions_rad": [
    -0.23, 0.14, 0.00, 0.35,
    -0.09, 0.28, 0.01, 0.32,
    0.02, 0.31, 0.02, 0.35,
    0.10, 0.61, 1.24, 0.02
  ]
}
```

Notes:

- The example truncates arrays for readability.
- In real traffic, `keypoints_xyz` has `26 x 3` values.
- In real traffic, `joint_transforms_world` has 26 entries, one per joint.

## Field-by-Field Definitions

### `source`

Type:

- string

Allowed values currently used:

- `"xr_hand_joint_poses"`
- `"vr_keypoints"`

Meaning:

- identifies whether the payload includes full joint poses or only keypoints

Backend guidance:

- prefer `joint_transforms_world` when `source == "xr_hand_joint_poses"`
- use `keypoints_xyz` only as fallback or debugging support

### `hand_side`

Type:

- string

Allowed values:

- `"left"`
- `"right"`

Meaning:

- identifies which hand triggered this update

Important:

- this does not mean the other hand is invalid
- the WM service should usually keep the latest state for both hands internally

### `timestamp_s`

Type:

- float

Meaning:

- timestamp attached on the teleop side when the hand update was parsed

Important:

- heartbeats may resend the latest payload with the same timestamp
- the WM service should tolerate repeated payloads
- do not assume strict monotonic increase on every HTTP call

### `keypoints_xyz`

Type:

- nested float array
- shape: `26 x 3`

Meaning:

- per-joint tracked 3D positions
- included for compatibility, fallback rendering, debugging, and sanity checking

Coordinate frame:

- same frame as `joint_transforms_world` when pose data is present
- currently labeled by `world_frame = "unity_xr_world"`

Backend guidance:

- do not treat this as the main pose representation when `joint_transforms_world` is present
- it is useful for consistency checks and fallback logic

### `is_relative`

Type:

- boolean

Current meaning:

- for legacy keypoint-only packets, this reflects the old detector mode semantics
- for current 7-float XR pose packets, the teleop side now normalizes them into world-frame joint poses and sends `false`

Backend guidance:

- for the real WM backend, this field is informational only
- do not gate pose handling on this flag when `joint_transforms_world` is present

### `world_frame`

Type:

- string or absent

Current value for XR pose packets:

- `"unity_xr_world"`

Meaning:

- identifies the fixed coordinate frame of the joint poses and positions

Important:

- every transform in `joint_transforms_world` is expressed in this frame
- this is an absolute frame, not wrist-relative and not parent-relative

### `joint_order`

Type:

- list of 26 strings

Meaning:

- canonical joint ordering matching the tracked hand skeleton
- useful if your backend wants deterministic array packing instead of a dictionary

Current order:

1. `wrist`
2. `palm`
3. `thumb_metacarpal`
4. `thumb_proximal`
5. `thumb_distal`
6. `thumb_tip`
7. `index_metacarpal`
8. `index_proximal`
9. `index_intermediate`
10. `index_distal`
11. `index_tip`
12. `middle_metacarpal`
13. `middle_proximal`
14. `middle_intermediate`
15. `middle_distal`
16. `middle_tip`
17. `ring_metacarpal`
18. `ring_proximal`
19. `ring_intermediate`
20. `ring_distal`
21. `ring_tip`
22. `little_metacarpal`
23. `little_proximal`
24. `little_intermediate`
25. `little_distal`
26. `little_tip`

### `joint_transforms_world`

Type:

- mapping from joint name to `4 x 4` float matrix

Shape:

- 26 entries when present
- each matrix has shape `4 x 4`

Matrix layout:

```text
H_world_joint =
[ R00 R01 R02 tx ]
[ R10 R11 R12 ty ]
[ R20 R21 R22 tz ]
[  0   0   0   1 ]
```

Where:

- `R` is the `3 x 3` joint orientation
- `t = [tx, ty, tz]` is the joint position
- both are expressed in the fixed `world_frame`

Critical semantics:

- this is the authoritative pose representation
- these transforms are absolute poses in world frame
- they are not parent-relative transforms
- they are not wrist-relative transforms
- they are not normalized hand-local transforms

Backend guidance:

- use this field as the main input for the real world-model backend
- if you need parent-relative transforms, compute them yourself:
  - `H_parent_joint = inv(H_world_parent) @ H_world_joint`

### `joint_positions_rad`

Type:

- optional list of 16 floats

Meaning:

- retargeted finger action generated by the teleop-side Leap hand operator
- this is auxiliary action/state information
- it is not the primary full-hand pose representation

Grouping:

- indices `0:4` = index finger
- indices `4:8` = middle finger
- indices `8:12` = ring finger
- indices `12:16` = thumb

Not included here:

- wrist translation
- wrist world orientation as a command
- forearm pose
- shoulder pose

Backend guidance:

- treat this as optional side information
- do not reconstruct the full hand or arm from this vector if `joint_transforms_world` is available

## What The WM Service Should Do

For a real world-model backend, the intended use is:

1. keep the latest state for both hands
2. on each `wm_step`, update only the side that triggered the request
3. use `joint_transforms_world` as the authoritative pose input
4. optionally use `joint_positions_rad` as an auxiliary action/control signal
5. render or generate the current observation image
6. return a JPG image

The WM service should be robust to:

- repeated payloads
- one-sided updates
- missing `joint_positions_rad`
- fallback keypoint-only packets

## Heartbeat Behavior

The teleop backend may resend the most recent payload periodically even if no fresh hand update arrives.

Why:

- keeps the observation stream alive
- avoids frozen image transport when upstream pauses briefly

Implications for the WM service:

- repeated `wm_step` calls may contain identical payloads
- duplicate processing must be safe
- rendering the same frame again is acceptable

## `POST /reset`

### Request

The teleop side sends:

```json
{
  "new": true
}
```

Meaning:

- `true`: reset the WM service state for a fresh episode
- `false`: soft reset behavior is backend-defined, but the teleop client supports it

### Response

Preferred response:

- raw JPG bytes
- `Content-Type: image/jpeg`

Optional state return:

- header `X-WM-State` containing a JSON object

Allowed alternative:

- JSON response containing an image plus state

The teleop client accepts these JSON image keys:

- `obs_jpg`
- `observation_jpg`
- `observation`
- `obs`
- `jpg`
- `image`

The image may be:

- raw bytes in an HTTP binary response
- base64 string in JSON
- byte array encoded as a JSON integer list

## `POST /wm_step` Response

Preferred response:

- raw JPG bytes
- `Content-Type: image/jpeg`

The teleop side decodes that JPG and publishes it as the observation stream.

If you return JSON instead, include the observation image under one of the accepted keys listed above.

## State Returned By `/reset`

If your backend wants to return state metadata, use a JSON object.

The teleop side stores it as an opaque mapping.

Recognized practical patterns include:

```json
{
  "left_joint_state": [...],
  "right_joint_state": [...],
  "episode_id": "...",
  "seed": 123
}
```

The teleop side may read joint-state-like keys if present, but otherwise treats the mapping as backend-defined state.

## Robustness Requirements

A production WM backend should handle these cases:

1. `joint_transforms_world` present, `joint_positions_rad` absent
2. only one hand updating for several frames
3. repeated identical timestamps
4. repeated identical payloads
5. temporary fallback to `vr_keypoints`
6. malformed or missing optional fields

Recommended behavior:

- validate required fields
- ignore unknown extra fields
- tolerate optional fields being absent
- keep internal per-hand state across requests

## Recommended Backend Input Priority

If multiple pose-related fields are present, use this priority:

1. `joint_transforms_world`
2. `keypoints_xyz`
3. `joint_positions_rad`

That ordering matches the intended meaning of the data.

## Minimal Correct Backend Assumption

If you want the simplest correct assumption set for implementation, use this:

- every `wm_step` request is JSON
- `joint_transforms_world` is the main signal when present
- transforms are world-frame absolute `4 x 4` poses
- the service should keep state across calls
- the service returns a JPG observation every step

## Reference Files

- [ego_dex_wm_client.py](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_wm_client.py)
- [ego_dex_remote_wm_backend.py](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_remote_wm_backend.py)
- [oculus.py](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/detector/vr/oculus.py)
