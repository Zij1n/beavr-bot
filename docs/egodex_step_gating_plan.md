# EgoDex Step Gating Plan

## Goal

Gate `/step` on the teleop side using the current live VR hand pose versus the last pose that was actually sent to `/step`, while keeping the VR-app overlay useful for alignment.

## Rules

- Current VR hand input arrives in tracking-space coordinates.
- Any live VR hand pose sent via `/step` must be flipped into step-space first.
- Poses returned from `/reset` or loaded from the JSONL reset seed are already treated as step-space poses.
- The JSONL reset seed must be passed through without another axis flip.
- The `palm` joint is dropped only for distance calculation.
- Display-only hand-visualization points are always shifted upward by `+0.10 m`.
- That display offset must never affect `/step`, `/reset`, or threshold checks.

## Gating Behavior

- Keep an accepted pose on the teleop side.
- Seed the accepted pose from `/reset`.
- While the current VR hand is farther than the threshold from the accepted pose:
  - do not call `/step`
  - show the current live hand normally
  - show the accepted/reference pose only as an overlay
- Once the current VR hand is within threshold:
  - allow `/step`
  - for the first successful `/step` after reset, send the reset-seed pose once
  - after that, update the accepted pose only when a `/step` call succeeds

## VR-App Overlay

- Live hand joints are always shown.
- Reference overlay appears only while blocked by threshold.
- Reference colors:
  - left: yellow
  - right: blue
- The looped JSONL trajectory overlay is removed from the active path.

## Dummy Server

- The dummy server should not own the `/step` gating logic.
- The only new server-side behavior is meaningful `/reset` seeding:
  - load a random frame from the JSONL file
  - apply it as the reset pose
  - return that pose in reset state so teleop can seed its accepted pose

## Config / Launch

- Threshold is editable from CLI args via the launcher, then forwarded into teleop config/env.
- Reset JSONL path is also configurable from the launcher and passed only to the dummy server.
- Hand-visualization publisher binding is handled on the teleop side.

## Default Threshold

Using the current JSONL file, in XYZ position space, dropping only `palm`, and using:

- gate value: sum of all compared joint distances across both hands

the adjacent-frame average gate value is:

`0.11288809025524707`

So the default threshold will be:

`0.22577618051049414`
