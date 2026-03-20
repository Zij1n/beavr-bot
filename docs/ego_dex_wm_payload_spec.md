# EgoDex WM Payload Specification

## Purpose

This document specifies the request and response contract used between the EgoDex teleoperation stack and a world-model service accessed through `WMClient`.

This document is written for the WM server developer.
It assumes no prior knowledge of this repository.

It explains:

- what request payload is sent to the WM server
- what response the WM server must return
- what the `joint_positions_rad` action means
- what the `keypoints_xyz` landmarks mean

## Important Summary

The current live WM request payload contains:

- a hand side
- a timestamp
- raw 3D hand landmarks (`keypoints_xyz`)
- optional 16-DoF retargeted finger action (`joint_positions_rad`)

## Transport Contract

The teleop side uses an HTTP client with two endpoints:

- `POST /wm_step`
- `POST /reset`

Reference implementation:
- [`ego_dex_wm_client.py`](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_wm_client.py)
- [`ego_dex_remote_wm_backend.py`](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/components/interface/robots/ego_dex_remote_wm_backend.py)

## `POST /wm_step`

### Request Content Type

The client sends:

- `Content-Type: application/json`

### Current Request Schema

The current live request schema is:

```json
{
  "source": "vr_keypoints",
  "hand_side": "right",
  "timestamp_s": 1773974331.22,
  "keypoints_xyz": [
    [0.012, 0.842, 0.031],
    [0.018, 0.846, 0.028],
    [0.021, 0.851, 0.024]
  ],
  "joint_positions_rad": [
    -0.23, 0.14, 0.00, 0.35,
    -0.09, 0.28, 0.01, 0.32,
    0.02, 0.31, 0.02, 0.35,
    0.10, 0.61, 1.24, 0.02
  ]
}
```

### Field Definitions

#### `source`

Type:
- string

Current expected value:
- `"vr_keypoints"`

Meaning:
- indicates that this payload was triggered by a fresh raw VR landmark update

#### `hand_side`

Type:
- string

Allowed values:
- `"left"`
- `"right"`

Meaning:
- identifies which side produced the trigger update
- the WM server may still maintain internal state for both hands and render both together

#### `timestamp_s`

Type:
- float

Meaning:
- timestamp from the teleop-side VR stream
- primarily useful for synchronization, logging, and deduplication

Important note:
- the backend may resend the latest payload as a heartbeat to keep the observation stream alive
- heartbeat resends may reuse an older timestamp
- the WM server should tolerate repeated payloads and treat them idempotently if needed

#### `keypoints_xyz`

Type:
- nested list of floats
- shape: `26 x 3`

Meaning:
- raw 3D hand landmarks from the Quest hand-tracking path
- this is the primary observation-driving input
- if the WM server needs to visualize the hand and arm correctly, this is the field it should use

Units:
- treated as metric 3D positions in the VR detector frame

#### `joint_positions_rad`

Type:
- list of 16 floats
- optional

Meaning:
- retargeted finger joint command produced by the teleop-side Leap operator
- useful as side information or joint-state bookkeeping
- not suitable for reconstructing full arm or wrist pose

Important note:
- the WM server should not use this vector as the primary source for hand/arm visualization
- it may be absent
- it does not contain sufficient information for full upper-limb pose

## `joint_positions_rad` Action Semantics

The 16 values are grouped by finger.

Current grouping is:

- `0:4` index finger
- `4:8` middle finger
- `8:12` ring finger
- `12:16` thumb

There is no explicit field for:

- wrist translation
- elbow pose
- shoulder pose
- whole-arm transform

This action vector is produced from a transformed hand-local retargeting pipeline.
That means global hand translation and most whole-arm pose information have already been removed before these 16 values are produced.

For that reason:

- `joint_positions_rad` is useful for finger articulation
- `joint_positions_rad` is not sufficient for upper-limb pose reconstruction

## Why `keypoints_xyz` Is The Primary Input

The raw landmark field preserves the spatial geometry of the tracked hand.
From the landmark cloud, the WM server can estimate:

- wrist position
- approximate forearm direction
- finger chain geometry

This is enough to build an oracle-style skeleton renderer.

By contrast, the 16-DoF action vector only tells you how the retargeter wants the fingers to articulate.
It does not tell you where the hand is in space.

## Landmark Layout For `keypoints_xyz`

The current landmark indexing follows a 26-point hand layout.
Named indices are:

- `0`: wrist / hand root
- `1`: palm
- `2`: thumb metacarpal / knuckle base
- `3`: thumb proximal
- `4`: thumb distal
- `5`: thumb tip
- `6`: index metacarpal
- `7`: index proximal / knuckle
- `8`: index intermediate
- `9`: index distal
- `10`: index tip
- `11`: middle metacarpal
- `12`: middle proximal / knuckle
- `13`: middle intermediate
- `14`: middle distal
- `15`: middle tip
- `16`: ring metacarpal
- `17`: ring proximal / knuckle
- `18`: ring intermediate
- `19`: ring distal
- `20`: ring tip
- `21`: little metacarpal
- `22`: little proximal / knuckle
- `23`: little intermediate
- `24`: little distal
- `25`: little tip

Reference constants:
- [`robots.py`](/scratch/zh2025/ICL_video_gen/data/beavr-bot/src/beavr/teleop/configs/constants/robots.py)

## WM Response Contract

### `POST /wm_step` Response

Preferred response:

- raw JPG bytes
- `Content-Type: image/jpeg`

Allowed alternative:

- JSON object containing one of:
  - `obs_jpg`
  - `observation_jpg`
  - `observation`
  - `obs`
  - `jpg`
  - `image`

If a JSON response is used, the image may be:

- base64-encoded string
- raw byte array serialized as a JSON list of integers

### `POST /reset` Request

The client sends:

```json
{
  "new": true
}
```

### `POST /reset` Response

Preferred response:

- raw JPG bytes
- `Content-Type: image/jpeg`
- optional header `X-WM-State` containing a JSON-encoded state mapping

Allowed alternative:

- JSON object with image plus state

Recognized state keys on the teleop side include:

- `left_joint_state`
- `right_joint_state`
- `joint_state`
- `joint_states`

If no state is returned, the teleop backend will still function, but it will rely on cached action information for joint-state publishing.

## Canonical Example For Current WM Developers

If you are implementing the current WM server today, the practical reading is:

- `keypoints_xyz` = main input for visualization and geometry
- `joint_positions_rad` = optional hand-action metadata
- return a JPG image for every `wm_step`

## Future Protocol Extension

If a future revision adds explicit joint poses, a good extension would be:

```json
{
  "source": "vr_keypoints",
  "hand_side": "right",
  "timestamp_s": 1773974331.22,
  "keypoints_xyz": [[...]],
  "joint_positions_rad": [...],
  "joint_transforms": {
    "wrist": [[...], [...], [...], [...]],
    "index_knuckle": [[...], [...], [...], [...]]
  }
}
```
