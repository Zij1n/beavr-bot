# EgoDex Hand Scaling Plan

## Goal

Scale the live tracked hand that is sent via `/step` so it matches the hand returned by `/reset`.

This is not display-only:

- the scaled hand must be the one used for gate comparison
- the scaled hand must be the one sent to `/step`
- the red live-hand visualization must be derived from the exact same scaled pose that is sent via `/step`

The dummy server reset image already renders the sampled JSONL hand pose and joint orientations. No new dummy-server rendering work is needed for this plan.

## Source Of Truth And Mapping Constraints

The JSONL `/reset` examples are the primary source of truth for the reset-side hand representation.

The legacy mapping note is useful, but should be treated as advisory if it disagrees with the actual JSONL examples or runtime payloads.

Important constraints to bake into the implementation:

- live VR `palm` does not have an independent EgoDex meaning for control
- JSONL `/reset` uses Quest-style joint names, but `palm` and `wrist` are overlapped
- JSONL `/reset` contains a distinct `thumb_metacarpal` transform, so it should be treated as a real candidate control joint unless downstream mapping code proves otherwise
- the internal EgoDex mapping may fan one source joint into multiple internal targets, but `/step` transport still carries a single Quest-style `thumb_tip`

Implication:

- scaling, calibration, and gating should operate on an effective EgoDex control skeleton, not on the raw Quest joint tree
- `palm` should be collapsed to `wrist` as the first normalization step
- after that normalization, `palm` should not influence calibration or gate error
- the outgoing `/step` payload should preserve that normalized convention, so transport `palm == wrist`
- `thumb_metacarpal` should stay in the first-pass control skeleton because the reset data treats it as a distinct joint
- if later code inspection proves the internal EgoDex action mapping ignores `thumb_metacarpal`, that can be changed in a follow-up, but the plan should not assume that up front

## Chosen Approach

Use per-bone automatic calibration at `/reset`, then keep those scale factors fixed until the next `/reset`.

Implementation shape:

1. Convert the live VR hand into step space first.
2. Immediately normalize `palm := wrist` in both live and reset transport payloads.
3. Normalize both the live hand and the `/reset` hand into the same effective control skeleton.
4. Estimate one scale factor per bone from their length ratio.
5. Clamp those factors to a safe range.
6. For every later live pose:
   - reconstruct scaled joint positions recursively along the control tree
   - keep each joint rotation unchanged
   - replace only the translation part of each used `4x4` transform
7. Expand the scaled control skeleton back into the full Quest-style `/step` transport payload.
8. Use that exact same expanded scaled pose for gating, `/step`, and the red live-hand visualization.

This keeps the articulated structure closer to the `/reset` hand than a single wrist-relative global scale.

## Why This Instead Of Global Wrist Scaling

Global wrist-relative scaling is simpler, but it assumes every part of the hand should grow by the same factor from the wrist.

That is weak for this use case:

- palm width and finger lengths often mismatch differently
- thumb proportions usually differ from the other fingers
- matching the `/reset` hand is a skeleton-shape problem, not just an overall-size problem
- `palm` is not an independently meaningful calibration target here because reset collapses it onto `wrist`

Per-bone scaling directly targets the mismatch that actually matters to EgoDex.

## Effective Control Skeleton

Define a reduced control skeleton containing only the source joints that meaningfully drive EgoDex behavior.

Per hand:

- `wrist`
- `thumb_metacarpal -> thumb_proximal -> thumb_distal -> thumb_tip`
- `index_metacarpal -> index_proximal -> index_intermediate -> index_distal -> index_tip`
- `middle_metacarpal -> middle_proximal -> middle_intermediate -> middle_distal -> middle_tip`
- `ring_metacarpal -> ring_proximal -> ring_intermediate -> ring_distal -> ring_tip`
- `little_metacarpal -> little_proximal -> little_intermediate -> little_distal -> little_tip`

Tree structure:

- `wrist -> thumb_metacarpal -> thumb_proximal -> thumb_distal -> thumb_tip`
- `wrist -> index_metacarpal -> index_proximal -> index_intermediate -> index_distal -> index_tip`
- `wrist -> middle_metacarpal -> middle_proximal -> middle_intermediate -> middle_distal -> middle_tip`
- `wrist -> ring_metacarpal -> ring_proximal -> ring_intermediate -> ring_distal -> ring_tip`
- `wrist -> little_metacarpal -> little_proximal -> little_intermediate -> little_distal -> little_tip`

Explicit exclusions from the control skeleton:

- `palm`

## Transport Skeleton Rules

The WM transport payload still uses full Quest-style `joint_transforms_world`.

First-pass expansion rules from the scaled control skeleton back to transport:

- `wrist`: scaled root transform
- `palm`: copy the scaled `wrist` transform exactly
- all control-skeleton joints: use the scaled translation with the original live rotation

This keeps the outgoing payload compatible while matching the reset/jsonl convention where `palm == wrist`.

## Data Model

Store scaling state inside the teleop WM backend, not in the dummy server.

Per side:

- `reset_reference_transport_hands_step`: `/reset` hand pose in step space, full Quest transport form
- `reset_reference_control_hands_step`: `/reset` hand pose normalized into the control skeleton
- `latest_live_transport_hands_step`: most recent live hand pose in step space, full transport form
- `latest_live_control_hands_step`: most recent live hand pose normalized into the control skeleton
- `bone_scale_by_side`: mapping from control-tree edge to scale factor
- `hand_scaling_enabled_by_side`
- `hand_scaling_source`: `identity`, `calibrated_from_reset`, or `partial_calibration`

The backend should treat left and right hands independently.

## Normalization

Before calibration or gate comparison, normalize both reset and live hand payloads into the same control skeleton.

Normalization rules:

- start from `joint_transforms_world`
- first rewrite `palm` to be an exact copy of `wrist`
- extract only control-skeleton joints
- treat `wrist` as the only hand root
- after normalization, do not create or use a separate `wrist -> palm` bone

## Calibration On `/reset`

Calibration should happen when the backend receives the `/reset` state.

Inputs:

- reset pose from `/reset` in step space, normalized into the control skeleton
- most recent live VR hand pose in step space, also normalized into the control skeleton

Per control-tree bone:

1. extract parent and child positions from the reset hand
2. extract parent and child positions from the live hand
3. compute lengths:
   - `L_reset = ||p_reset_child - p_reset_parent||`
   - `L_live = ||p_live_child - p_live_parent||`
4. if `L_live` is too small or data is missing, fall back to `1.0`
5. otherwise set:
   - `scale_raw = L_reset / L_live`
6. clamp:
   - `scale = clamp(scale_raw, min_scale, max_scale)`

Recommended default clamp:

- `min_scale = 0.0`
- `max_scale = 10.0`

## Per-Bone Scale Policy

There is no regularization layer in the first implementation.

Each control-tree edge gets its own scale directly from the reset/live bone-length ratio:

- `scale(parent, child) = clamp(L_reset / L_live, min_scale, max_scale)`

The scale used at runtime for a bone is exactly that per-bone value.

Reason:

- the goal is to match the reset skeleton as directly as possible
- adding median blending or finger-level smoothing would inject assumptions that are not clearly justified
- a plain per-bone policy is easier to reason about and easier to debug

If a bone estimate is unusable, only that bone falls back to `1.0`.

## Runtime Scaling

For each incoming live hand pose:

1. convert to step space
2. normalize into the control skeleton
3. recursively reconstruct scaled positions from the control root:
   - `p'_root = p_root`
   - for each edge `parent -> child`:
     - `v = p_child - p_parent`
     - `p'_child = p'_parent + scale(parent, child) * v`
4. keep each control-joint rotation block unchanged
5. write scaled translations back into control-joint transforms
6. expand the scaled control skeleton back into the full transport payload

This preserves live articulation and orientation while resizing only the joints that actually matter for EgoDex.

## Integration Points

Apply scaling in the teleop backend after axis flip and before any downstream use.

Order of operations:

1. receive live VR hand pose
2. flip into step space
3. normalize `palm := wrist`
4. normalize into the control skeleton
5. apply per-bone scaling on the control skeleton
6. expand back into transport form
7. use the expanded scaled transport pose for gate comparison
8. use the expanded scaled transport pose for `/step`
9. publish the red live-hand visualization from that same expanded scaled transport pose
10. only use a different visualization payload for the blocked reference overlay

Important:

- do not compare the raw live pose against the accepted step pose
- do not compare using the pre-normalized live `palm`
- do not publish raw tracking input as the red live-hand visualization
- the red visualization payload must be byte-for-byte derived from the same scaled transport-hand structure used for `/step`

The backend should have a single canonical scaled transport pose used by all three paths.

## Gate Error Rules

The gate error should be computed on the effective control skeleton only.

That means:

- `palm` is normalized to `wrist` before the comparison path
- compare only the joints that actually participate in the control skeleton

This avoids polluting the threshold with joints that are duplicated by convention (`palm`).

## Reset Timing Edge Cases

Case 1: `/reset` arrives and live hand is available

- calibrate immediately

Case 2: `/reset` arrives but no live hand has been received yet

- store the reset reference control skeleton
- defer calibration until the first usable live hand arrives
- once calibration completes, keep the factors fixed until the next `/reset`

Case 3: only one side has usable data

- calibrate that side only
- leave the other side at identity scaling

## Fallback Rules

Use identity scaling when:

- reset hand missing for that side
- live hand missing for that side
- a needed control bone is missing
- bone length is near zero
- scale estimate is NaN or non-finite

The system should degrade safely to "no scaling" rather than emitting a corrupted `/step` pose.

## Configuration

Add backend config and launcher args for:

- `step_hand_scaling_enable`
- `step_hand_scale_min`
- `step_hand_scale_max`

Optional later additions:

- debug dump of calibrated scales
- per-side enable/disable
- explicit recalibration command from `backend_cli.py`

## CLI / Debug Status

Expose live status in `backend_cli.py`:

- whether scaling is enabled
- whether calibration has completed for each side
- number of calibrated control bones per side
- compact min/median/max scale summary per side
- residual error on the control skeleton

This is useful when debugging why a hand is not matching the `/reset` pose.

## Validation Plan

1. Mapping checks

- verify from sample JSONL that reset `palm` and `wrist` overlap
- verify live transport is normalized so `palm == wrist` before calibration and gating
- verify outgoing scaled transport sets `palm == wrist`
- verify `thumb_metacarpal` is present and distinct in the sample JSONL
- verify `thumb_metacarpal` participates in calibration and gate error in the first pass

2. Unit-level checks

- if live hand and reset hand are identical on the control skeleton, all scales should be `~1.0`
- if a synthetic hand has every control bone `1.2x` shorter than reset, calibrated scales should be `~1.2`
- missing control bones should fall back cleanly

3. Runtime checks

- after calibration, compare scaled live control-bone lengths against reset control-bone lengths
- report residual error per side

4. Visual checks

- red visualization should match the pose actually sent via `/step`
- `/reset` reference and scaled live hand should align much more closely than raw live hand

5. Safety checks

- threshold gating must use the scaled control-informed transport pose
- no NaNs or invalid transforms should be sent to `/step`

## Non-Goals For The First Pass

- changing joint rotations
- estimating new scales every frame
- fitting a nonlinear hand shape model
- changing dummy-server rendering behavior

## First Implementation Slice

To keep risk controlled, implement in this order:

1. add control-skeleton parent map helpers
2. add transport normalization helpers with early `palm := wrist`
3. add normalization helpers from transport payload to control skeleton
4. add per-bone calibration storage in backend state
5. calibrate from `/reset` plus latest live hand on the control skeleton
6. apply recursive scaling to live control-skeleton transforms
7. expand scaled control skeleton back into transport form with `palm := wrist`
8. route scaled transport pose into gate comparison and `/step`
9. route the same scaled transport pose into the red live-hand visualization path
10. keep blocked reference visualization separate
11. expose scaling status in backend CLI
12. add config / launcher args
13. add residual-error debug logging on the control skeleton

This yields a working end-to-end version before any further refinement.
