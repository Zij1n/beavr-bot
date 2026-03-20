# EgoDex XR Hand Pose Plan

## Verified Sender Contract

Source provided by user and cross-checked against the Unity sender:
- `GestureDetectorXR.cs`
- hand channels: `RightHand`, `LeftHand`

The hand payload is **not JSON**.

Each hand message is a plain string with this format:

```text
<mode>:<joint0>|<joint1>|<joint2>|...|<joint25>:
```

Where:

- `<mode>` is either `relative` or `absolute`
- each `<jointN>` is:

```text
px,py,pz,qx,qy,qz,qw
```

So the exact payload shape per hand is:

- `26` joints
- `7` floats per joint
- `182` floats total

## Joint Order

The sender uses this exact 26-joint order:

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

## Per-Joint Meaning

Each joint contributes:

```text
px,py,pz,qx,qy,qz,qw
```

Map this as:

- translation: `(px, py, pz)`
- orientation quaternion: `(qx, qy, qz, qw)`

This is the real XR joint orientation coming from the app.

## Missing Joint Placeholder

If a joint is not tracked, the sender inserts:

```text
0,0,0,0,0,0,1
```

So there is no explicit tracked flag on the wire.

## World Frame Interpretation

The sender currently forwards the XR / Unity pose directly.

Important verified observation:
- `ToWorldPosition(...)` returns the input position unchanged in the Unity sender.

So for planning purposes, the pose should be treated as already expressed in the sender's XR / Unity world frame.

The Python side should preserve that frame consistently and label it explicitly when forwarding to WM.

## What Python Should Do

The Python side does not need to estimate orientation from landmarks anymore.

Instead, it should:

1. parse the plain string payload
2. recover `26` joint positions and `26` joint quaternions
3. convert each `(position, quaternion)` pair into a world-frame `4x4`
4. forward those `26` transforms to the WM backend / WM server

So the authoritative WM payload should be:

```json
{
  "source": "xr_hand_joint_poses",
  "world_frame": "unity_xr_world",
  "hand_side": "right",
  "timestamp_s": 123.4,
  "joint_order": ["wrist", "palm", "..."],
  "joint_transforms_world": {
    "wrist": [[...], [...], [...], [...]],
    "palm": [[...], [...], [...], [...]],
    "...": [[...], [...], [...], [...]]
  },
  "joint_positions_rad": [ ... optional Leap action ... ]
}
```

## Transform Shape

Each `joint_transforms_world[joint_name]` should be:

```text
H_world_joint =
[ R00 R01 R02 tx ]
[ R10 R11 R12 ty ]
[ R20 R21 R22 tz ]
[  0   0   0   1 ]
```

Where:

- `R` comes from the quaternion `(qx, qy, qz, qw)`
- `[tx, ty, tz]` comes from `(px, py, pz)`

These transforms must be:

- absolute world-frame poses
- not wrist-relative
- not parent-relative
- not hand-local normalized poses

## Parser Plan

The detector parser should do this:

1. split once on the first `:`
2. read the left side as `mode`
3. remove the final trailing `:` from the body
4. split the body by `|`
5. expect exactly `26` joint records
6. split each joint record by `,`
7. expect exactly `7` floats per joint
8. map them to `(px, py, pz, qx, qy, qz, qw)`
9. convert quaternion to rotation matrix
10. build `4x4` transform per joint

## Files To Change

1. `src/beavr/teleop/components/detector/detector_types.py`
- extend `InputFrame` to carry:
  - `world_frame`
  - `joint_order`
  - `joint_transforms_world`

2. `src/beavr/teleop/components/detector/vr/oculus.py`
- replace the old `x,y,z`-only parser with the real `7-float` parser
- convert quaternions to `4x4`
- keep joint translations for existing keypoint consumers

3. `src/beavr/teleop/components/interface/robots/ego_dex_robot.py`
- forward `joint_transforms_world` from the detector message to the backend

4. `src/beavr/teleop/components/interface/robots/ego_dex_remote_wm_backend.py`
- use `joint_transforms_world` as the primary WM payload
- keep `joint_positions_rad` optional and auxiliary

5. `src/beavr/teleop/components/interface/robots/ego_dex_dummy_wm_server.py`
- consume `joint_transforms_world`
- visualize joint positions from translation
- visualize per-joint orientation axes from rotation matrices

## Reverted Assumption

A previous speculative assumption that the app would send JSON was wrong.

That assumption should not be used.

The current verified sender contract is the plain string format described above.

## Validation

After implementation, verify:

1. each parsed hand message contains exactly `26` joints
2. each joint contains exactly `7` floats
3. each quaternion is mapped into a valid `3x3` rotation block
4. each WM payload contains `26 x 4 x 4` pose data per hand
5. dummy WM server shows correct per-joint orientation axes
