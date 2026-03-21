# EgoDex WM Payload Specification

## Purpose

This document defines the HTTP request/response contract between the EgoDex teleoperation stack and a WM service.

It is written for a WM backend developer with zero context of this repository.

It describes:

- which endpoint is called
- how often it is called
- the exact JSON shape sent to the WM service
- the shape of every array in that JSON
- which fields are authoritative
- what the WM service must return

This document describes the **WM-facing payload**, not the raw headset socket payload.

## Runtime Model

The teleop process keeps the latest:

- left-hand pose data
- right-hand pose data
- left-hand action
- right-hand action

Then it calls `step` at a fixed rate of **2 Hz**.

Each `step` call sends a **two-hand snapshot payload** containing the latest known state for both hands.

This is the canonical WM contract.

## Transport Contract

The teleop side calls:

- `POST /step`
- `POST /reset`

Reference files:

- [ego_dex_wm_client.py](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_wm_client.py)
- [ego_dex_remote_wm_backend.py](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_remote_wm_backend.py)

Request content type:

- `Content-Type: application/json`

Preferred response content type:

- `image/jpeg`

## `POST /step`

### Request Frequency

The teleop side sends one snapshot every `1/2` second by default.

Important:

- `step` is no longer a one-hand trigger event
- `step` is a periodic two-hand snapshot
- one or both hands may still contain `null` fields if data has not been observed yet

### Canonical Request Schema

The live payload is best understood as JSONC-style pseudocode:

```jsonc
{
  "source": "two_hand_snapshot",                  // string
  "sent_at_s": "<float seconds>",                 // scalar float, when teleop sent this HTTP request
  "timestamp_s": "<float seconds>",               // scalar float, max hand timestamp currently cached
  "world_frame": "unity_xr_world",                // string when known, may be omitted
  "hands": {                                       // object with exactly 2 keys
    "left": "<HandPayload | null>",
    "right": "<HandPayload | null>"
  }
}
```

Where `HandPayload` has this shape:

```jsonc
{
  "source": "xr_hand_joint_poses",               // string: "xr_hand_joint_poses" | "vr_keypoints" | "action_only" | "none"
  "timestamp_s": "<float seconds | null>",
  "keypoints_xyz": "<float[26][3] | null>",
  "is_relative": false,                            // bool
  "world_frame": "unity_xr_world",               // string | null
  "joint_order": "<string[26] | null>",
  "joint_transforms_world": "<JointTransformMap | null>",
  "joint_positions_rad": "<float[16] | null>",
  "action_timestamp_s": "<float seconds | null>"
}
```

Where `JointTransformMap` has this shape:

```jsonc
{
  "wrist": "<float[4][4]>",
  "palm": "<float[4][4]>",
  "thumb_metacarpal": "<float[4][4]>",
  "thumb_proximal": "<float[4][4]>",
  "thumb_distal": "<float[4][4]>",
  "thumb_tip": "<float[4][4]>",
  "index_metacarpal": "<float[4][4]>",
  "index_proximal": "<float[4][4]>",
  "index_intermediate": "<float[4][4]>",
  "index_distal": "<float[4][4]>",
  "index_tip": "<float[4][4]>",
  "middle_metacarpal": "<float[4][4]>",
  "middle_proximal": "<float[4][4]>",
  "middle_intermediate": "<float[4][4]>",
  "middle_distal": "<float[4][4]>",
  "middle_tip": "<float[4][4]>",
  "ring_metacarpal": "<float[4][4]>",
  "ring_proximal": "<float[4][4]>",
  "ring_intermediate": "<float[4][4]>",
  "ring_distal": "<float[4][4]>",
  "ring_tip": "<float[4][4]>",
  "little_metacarpal": "<float[4][4]>",
  "little_proximal": "<float[4][4]>",
  "little_intermediate": "<float[4][4]>",
  "little_distal": "<float[4][4]>",
  "little_tip": "<float[4][4]>"
}
```

### Array Shapes Summary

- top-level `hands`: 2 keys exactly, `left` and `right`
- `keypoints_xyz`: `float[26][3]`
- `joint_order`: `string[26]`
- `joint_transforms_world[joint_name]`: `float[4][4]`
- `joint_transforms_world`: 26 named matrices total
- `joint_positions_rad`: `float[16]` when present

### Joint Order

The 26 tracked joints are always ordered as:

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

## Meaning of the Main Fields

### `hands.left` and `hands.right`

These contain the latest cached state for each side.

Possible cases:

- full XR pose data present
- only keypoints present
- only action present
- no data yet, in which case the side may be `null`

### `joint_transforms_world`

This is the primary pose input for the real WM backend.

Each value is a `4x4` homogeneous transform:

```text
H_world_joint =
[ R00 R01 R02 tx ]
[ R10 R11 R12 ty ]
[ R20 R21 R22 tz ]
[  0   0   0   1 ]
```

Meaning:

- `R` is the `3x3` joint orientation
- `[tx, ty, tz]` is the joint position
- both are expressed in the fixed `world_frame`

Critical semantics:

- these are **world-frame absolute poses**
- they are not parent-relative
- they are not wrist-relative
- they are not normalized hand-local poses

If you need parent-relative transforms, compute them yourself:

```text
H_parent_joint = inv(H_world_parent) @ H_world_joint
```

### `keypoints_xyz`

This is a `26 x 3` landmark array.

It is included for:

- compatibility
- fallback rendering
- debugging
- sanity checking

If `joint_transforms_world` is present, use that as the primary signal.

### `joint_positions_rad`

This is optional auxiliary hand action/state.

Shape:

- `float[16]`

Meaning:

- a 16-DoF retargeted finger action vector
- not a full hand pose representation
- not required for rendering if `joint_transforms_world` is present

Finger grouping:

- `0:4` index
- `4:8` middle
- `8:12` ring
- `12:16` thumb

### `is_relative`

This field is informational.

For the current XR pose path, the teleop side converts incoming pose packets into world-frame hand poses before forwarding to WM.

Practical guidance:

- if `joint_transforms_world` is present, do not gate pose handling on `is_relative`

## Recommended Backend Behavior

A real WM backend should:

1. read both hands from every `step` request
2. use `joint_transforms_world` as the authoritative pose input when present
3. treat `keypoints_xyz` as fallback/debug data
4. treat `joint_positions_rad` as optional auxiliary state
5. maintain its own world state across requests
6. return one observation image per request

## `POST /reset`

### Request

The teleop side sends:

```json
{
  "new": true
}
```

### Response

Preferred response:

- raw JPG bytes
- `Content-Type: image/jpeg`

Optional state return:

- header `X-WM-State` containing a JSON object

Allowed JSON image keys if returning JSON instead of raw bytes:

- `obs_jpg`
- `observation_jpg`
- `observation`
- `obs`
- `jpg`
- `image`

The teleop side also accepts a base64 string response for the image.

## `POST /step` Response

Preferred response:

- raw JPG bytes
- `Content-Type: image/jpeg`

Also accepted:

- JSON object containing an image under one of the accepted keys above
- JSON string that is directly a base64-encoded image
- plain-text base64 image body

The teleop side decodes the image and publishes it as the observation stream.

## Robustness Requirements

A production WM backend should tolerate:

- repeated snapshots
- repeated timestamps
- one hand updating while the other stays unchanged
- one side being `null`
- missing `joint_positions_rad`
- fallback packets that do not include `joint_transforms_world`

## Backward Compatibility Note

The dummy WM server currently still accepts the older single-hand payload for debugging, but the canonical contract for new backend development is the two-hand snapshot schema described above.

## Reference Files

- [ego_dex_wm_client.py](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_wm_client.py)
- [ego_dex_remote_wm_backend.py](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_remote_wm_backend.py)
- [ego_dex_dummy_wm_server.py](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_dummy_wm_server.py)
