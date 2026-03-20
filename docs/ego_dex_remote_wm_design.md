# EgoDex Remote WM Design

## 1. Purpose

This document explains the reduced EgoDex remote world-model design from first principles.
It is written for someone with zero context about this repo.

The goal is to make the `ego_dex` teleop path work with a remote world-model service that returns an observation image, while keeping the implementation small and easy to review.

The reduced design has four logical runtime pieces:

1. config plus teleop main loop
2. remote WM backend
3. WMClient
4. testing WMServer with oracle plotting

There is one repo-specific constraint:

- `teleop.py` does not instantiate the robot directly.
- It loads `beavr.teleop.configs.robots.ego_dex_config` through the shared config loader.
- Because of that, the "config plus main loop" logical piece lives in two files on disk:
  - `src/beavr/teleop/configs/robots/ego_dex_config.py`
  - `src/beavr/teleop/components/interface/robots/ego_dex_robot.py`

That is the only reason this design uses five code files instead of four.

## 2. Problem Being Solved

We want the EgoDex teleop path to do three things:

1. receive raw Meta Quest hand keypoints from the detector path
2. send those keypoints to a world-model service
3. publish the returned image as the observation seen by the VR client

There is also an existing Leap retargeting path in the repo. That path converts normalized hand keypoints into a 16-DoF finger command. That action is still useful for bookkeeping and for compatibility with the rest of the teleop stack, but it must not be the main rendering input for the world-model visualization.

Why not?

Because the Leap action does not contain full spatial arm pose.
It contains finger configuration only.
If the visualization is driven from Leap action alone, the image cannot correctly reflect whole-hand translation, wrist position, or arm placement.
That is exactly why the earlier incorrect render only showed small finger motion.

The render must be driven by raw VR keypoints.

## 3. High-Level Architecture

The runtime topology is:

1. `OculusVRHandDetector` publishes raw Quest keypoints
2. `EgoDexRobot` subscribes to those raw keypoints
3. `EgoDexRobot` forwards them to `RemoteEgoDexWMBackend`
4. `RemoteEgoDexWMBackend` packages a WM payload
5. `WMClient` sends the payload to a remote WM service
6. the WM service returns JPG bytes
7. `RemoteEgoDexWMBackend` decodes the JPG into an image frame
8. `EgoDexRobot` publishes that frame as the observation stream

A parallel action path also exists:

1. Quest keypoints are transformed into a canonical hand-local frame
2. `LeapHandOperator` solves for a 16-DoF finger action
3. `EgoDexRobot` receives that action on the existing operator topic
4. `EgoDexRobot` forwards it to `RemoteEgoDexWMBackend`
5. the backend caches it and may attach it to the WM payload as optional metadata

Important:

- raw keypoints drive observation generation
- Leap action does not drive observation generation
- Leap action is optional side information only

## 4. Files In Scope

The reduced implementation should contain only these runtime files:

1. `src/beavr/teleop/configs/robots/ego_dex_config.py`
2. `src/beavr/teleop/components/interface/robots/ego_dex_robot.py`
3. `src/beavr/teleop/components/interface/robots/ego_dex_remote_wm_backend.py`
4. `src/beavr/teleop/components/interface/robots/ego_dex_wm_client.py`
5. `src/beavr/teleop/components/interface/robots/ego_dex_dummy_wm_server.py`

And one design document:

6. `docs/ego_dex_remote_wm_design.md`

Everything else from the earlier backend package is unnecessary for this reduced path and should be removed.

## 5. Responsibilities Per File

### 5.1 `ego_dex_config.py`

This file exists because the shared loader expects a robot config module.
It is the bridge from `teleop.py --robot_name=ego_dex` into the new runtime.

Responsibilities:

- register the `ego_dex` robot name
- construct the existing detector, transform, and operator components already used by the repo
- construct the `WMClient`
- construct the `RemoteEgoDexWMBackend`
- construct the `EgoDexRobot`
- wire ports, host, image size, and timing configuration

This file should not contain:

- rendering logic
- HTTP request logic
- world-model state logic
- backend mode branching for unrelated renderers

### 5.2 `ego_dex_robot.py`

This is the teleop main-loop component.
It sits inside the normal teleop runtime and only orchestrates data movement.

Responsibilities:

- subscribe to raw VR keypoints for left and right hands
- subscribe to operator action messages for left and right hands
- sanitize and deduplicate action messages
- forward actions to the backend
- forward raw keypoints to the backend
- ask the backend for joint state to publish back onto the existing `joint_angles` topics
- ask the backend for the current observation frame
- publish that observation frame as the VR-visible image stream
- keep a simple session log for debugging

This file must not contain:

- HTTP transport code
- skeleton plotting code
- oracle render code
- remote service protocol details beyond the backend method calls

### 5.3 `ego_dex_remote_wm_backend.py`

This is the teleop-side adapter between the teleop loop and the external world-model service.

Responsibilities:

- define the action object passed from `EgoDexRobot`
- cache the latest action per side
- build a WM payload whenever fresh raw keypoints arrive
- call `WMClient.wm_step(payload)`
- decode returned JPG bytes into an OpenCV frame
- cache the latest observation frame
- cache the latest state mapping returned by `reset`
- expose `get_camera_frame()` to the robot loop
- expose `get_joint_state(side)` to the robot loop
- optionally perform heartbeat resends of the latest WM payload to keep the image stream alive

This file must not contain:

- HTTP server code
- oracle skeleton rendering code
- detector/operator construction
- teleop config registration

### 5.4 `ego_dex_wm_client.py`

This file defines the transport contract to the world-model service.
It should be the only place that knows how to communicate with the remote service.

Responsibilities:

- send `wm_step` requests
- send `reset` requests
- parse supported response forms
- return raw JPG bytes and a JSON-like state mapping

Required public interface:

- `WMClient(hostname, port, timeout_s)`
- `WMClient.wm_step(payload) -> bytes`
- `WMClient.reset(new) -> tuple[bytes, Mapping]`

This file is the intended extension point for wiring to a real backend.
If you later want a custom RPC bridge, simulator bridge, robot middleware client, or learned-model wrapper, this is the file you replace or extend.

### 5.5 `ego_dex_dummy_wm_server.py`

This is the local test service.
It behaves like a minimal world-model server so the rest of the pipeline can be tested without a real backend.

Responsibilities:

- serve `POST /wm_step`
- serve `POST /reset`
- serve `GET /health`
- accept the same payload schema that the real backend will receive
- render an oracle-style skeleton image from raw Quest keypoints
- return JPG observation bytes

This file must be self-contained.
It should not reuse the earlier local renderer classes from the removed backend package.
That keeps review simple.

## 6. Why Raw Keypoints Must Drive Rendering

This is the most important design rule.

The Meta Quest detector provides raw hand landmarks in space.
Those landmarks preserve where the hand is, how it is oriented, and how the fingers are arranged.
From that information, a renderer can estimate a full upper-limb chain for visualization.

The Leap action path does not preserve that information.
Before Leap IK, the transform path does the following:

1. subtract wrist position
2. rotate the hand into a canonical local frame
3. solve finger action in that normalized frame

That means the final action loses most of the global hand pose information that a skeleton render needs.

Therefore:

- using raw keypoints for rendering is correct
- using Leap action for rendering is incorrect for this task

The backend may still forward `joint_positions_rad` as optional metadata because:

- it is useful for bookkeeping
- it allows the backend to publish joint state even if the WM service itself does not compute any state
- it keeps the teleop-side joint-state topics alive

But `joint_positions_rad` is not the primary visual input.

## 7. Public Interfaces

### 7.1 Action object from teleop loop to backend

The robot loop passes a structured action object to the backend.

Fields:

- `hand_side`: `"left"` or `"right"`
- `timestamp_s`: float or `None`
- `joint_positions_rad`: 1D float array with length 16

The 16 action values come from the existing Leap operator.
Current grouping is:

- `0:4` index finger
- `4:8` middle finger
- `8:12` ring finger
- `12:16` thumb

There is no wrist, forearm, elbow, or shoulder in this action.

### 7.2 Backend methods used by `EgoDexRobot`

`EgoDexRobot` depends on these backend methods:

- `send_action(action)`
- `on_keypoints(side, keypoints_xyz, timestamp_s)`
- `step()`
- `reset(new=True)`
- `get_joint_state(side)`
- `get_camera_frame()`
- `pop_pose_record()`

Expected behavior:

- `send_action` updates cached action state
- `on_keypoints` is the main observation-driving method
- `step` is only for background maintenance such as heartbeat
- `get_joint_state` returns a 16-element vector for publishing
- `get_camera_frame` returns the latest observation as a BGR OpenCV frame
- `pop_pose_record` returns one log record or `None`

### 7.3 WMClient contract

`WMClient` is intentionally small.

#### `wm_step(payload) -> bytes`

Input:

- JSON-serializable mapping

Output:

- JPG bytes

#### `reset(new) -> (bytes, Mapping)`

Input:

- boolean `new`

Output:

- observation JPG bytes
- state mapping

The default HTTP implementation may support either raw-image responses or JSON-wrapped image responses, but the semantic contract stays the same.

### 7.4 WMServer API contract

#### `POST /wm_step`

Request body:

- JSON object described in Section 8

Response:

- preferred: raw JPEG bytes with `Content-Type: image/jpeg`
- optional alternative: JSON object containing image bytes encoded as base64 or byte list

#### `POST /reset`

Request body:

```json
{"new": true}
```

Response:

- preferred: raw JPEG bytes with optional state in header
- optional alternative: JSON object containing image plus state

#### `GET /health`

Response:

- JSON object with basic liveness information

## 8. WM Payload Schema

The canonical payload sent from the backend to the WM service is:

```json
{
  "source": "vr_keypoints",
  "hand_side": "right",
  "timestamp_s": 1773974331.22,
  "keypoints_xyz": [[0.01, 0.84, 0.03], [0.02, 0.85, 0.02]],
  "joint_positions_rad": [-0.23, 0.14, 0.00, 0.35]
}
```

Field meanings:

- `source`
  - string describing where the payload came from
  - for this implementation the expected value is `"vr_keypoints"`

- `hand_side`
  - `"left"` or `"right"`
  - identifies which side produced the trigger update

- `timestamp_s`
  - source timestamp from the VR stream
  - float seconds

- `keypoints_xyz`
  - raw Quest keypoints
  - shape `26 x 3`
  - units are whatever the detector currently emits for the Quest stream
  - this is the render-driving input

- `joint_positions_rad`
  - optional 16-value action vector
  - carried as side information only
  - not required for rendering

Notes:

- the backend sends a payload when a new raw keypoint frame arrives
- the service should be robust to only one side updating at a time
- the service may keep internal state for both left and right sides and render both together

## 9. Keypoint Layout Used By The Dummy Server

The test server uses the 26-point Quest layout already present in the repo.
Named indices used by the renderer are:

- `0`: hand root
- `2`: thumb knuckle
- `3`: thumb intermediate base
- `4`: thumb intermediate tip
- `5`: thumb tip
- `6`: index metacarpal
- `7`: index knuckle
- `8`: index intermediate base
- `9`: index intermediate tip
- `10`: index tip
- `11`: middle metacarpal
- `12`: middle knuckle
- `13`: middle intermediate base
- `14`: middle intermediate tip
- `15`: middle tip
- `16`: ring metacarpal
- `17`: ring knuckle
- `18`: ring intermediate base
- `19`: ring intermediate tip
- `20`: ring tip
- `21`: little metacarpal
- `22`: little knuckle
- `23`: little intermediate base
- `24`: little intermediate tip
- `25`: little tip

Index `1` is not used by the dummy renderer.

## 10. Oracle Plotting Strategy In The Dummy Server

The testing WMServer is not a learned world model.
It is just a deterministic visualization service.

Its job is to make the remote WM path observable during development.

The renderer performs these steps:

1. receive raw keypoints for one side
2. store them as the latest state for that side
3. estimate a simple arm chain from the hand geometry
4. combine estimated arm joints with the hand/finger keypoints
5. project 3D points into a fixed camera view
6. draw a left and right skeleton on a canvas
7. return the resulting image as JPG bytes

### 10.1 Arm estimation

The raw keypoints do not directly provide shoulder and elbow in this path, so the dummy renderer estimates them.

One simple estimation scheme is:

1. let `hand = keypoints[0]`
2. compute `palm_center` from the four finger metacarpals
3. let `forearm_dir = normalize(hand - palm_center)`
4. offset backward from the hand along `forearm_dir`
5. add a side-dependent lateral shift to build a plausible arm and shoulder chain

This is only for visualization.
It is not a physical robot model.
But it is enough to make the plotted arm move in the same general way as the earlier local oracle render.

### 10.2 Rendering style

The dummy renderer should be minimal and self-contained.
It should:

- use a fixed background
- draw left and right skeleton edges
- draw per-joint points
- show a small status overlay
- show a live step counter or timestamp so a frozen image is obvious

It should not depend on the removed local oracle renderer classes.

## 11. Detailed Runtime Sequence

### 11.1 Startup sequence

1. user runs `teleop.py --robot_name=ego_dex --laterality=bimanual`
2. shared loader imports `ego_dex_config.py`
3. config builds detector, transforms, operators, WMClient, backend, and `EgoDexRobot`
4. backend optionally calls `WMClient.reset(new=True)`
5. teleop loop starts running `EgoDexRobot.stream()`

### 11.2 Action update sequence

1. operator publishes `joint_angles`
2. `EgoDexRobot` receives the message
3. `ActionDecoder` converts it to a structured action object
4. `EgoDexRobot` calls `backend.send_action(action)`
5. backend caches latest action for that side
6. backend may expose that cached action as joint state immediately

Important:

- this sequence alone does not trigger a new observation request

### 11.3 Keypoint update sequence

1. detector publishes raw Quest keypoints
2. `EgoDexRobot` receives the keypoint frame
3. robot loop reshapes the payload into `26 x 3`
4. `EgoDexRobot` calls `backend.on_keypoints(side, keypoints_xyz, timestamp_s)`
5. backend builds WM payload using raw keypoints and optional cached action
6. backend calls `WMClient.wm_step(payload)`
7. world-model service returns JPG bytes
8. backend decodes JPG to frame and caches it
9. robot loop later publishes the cached frame to the VR image topic

### 11.4 Publish sequence

Each stream tick the robot loop:

1. publishes joint state from `backend.get_joint_state(side)`
2. publishes observation frame from `backend.get_camera_frame()`

If no valid frame exists yet, it publishes a placeholder image instead.

## 12. Failure Handling

The reduced design should degrade predictably.

### 12.1 No WM response

If the client cannot reach the WM service:

- backend logs the failure
- previous observation frame remains cached if available
- otherwise the robot loop publishes a placeholder image

### 12.2 Invalid keypoint payload

If keypoints are malformed:

- robot loop drops the keypoint frame
- no WM request is sent for that frame

### 12.3 Invalid JPG response

If the WM service returns bytes that cannot be decoded as an image:

- backend logs a warning
- previous good frame remains cached if available
- otherwise placeholder is shown

### 12.4 No action yet

This is acceptable.
The backend should still send keypoints and render observations.
Action is optional metadata.

## 13. State Ownership

This design keeps state ownership simple.

### 13.1 `EgoDexRobot` owns

- ZMQ subscribers
- ZMQ publishers
- tick timing
- placeholder frame generation
- session log file

### 13.2 `RemoteEgoDexWMBackend` owns

- latest action per side
- latest WM payload
- latest decoded observation frame
- latest joint-state mapping
- last backend event record
- heartbeat timing

### 13.3 `WMClient` owns

- transport details
- request formatting
- response parsing
- timeouts

### 13.4 `DummyWMServer` owns

- latest raw keypoints per side
- dummy state for reset/health
- skeleton rendering logic
- HTTP serving logic

## 14. Minimal Run Commands

From repo root:

```bash
conda activate beavr_teleop
cd /scratch/zh2025/ICL_video_gen/data/beavr-bot
```

Start the test WM service:

```bash
PYTHONPATH=src python -m beavr.teleop.components.interface.robots.ego_dex_dummy_wm_server --host 127.0.0.1 --port 18080
```

Start teleop using the remote WM path:

```bash
PYTHONPATH=src \
EGO_DEX_WM_HOST=127.0.0.1 \
EGO_DEX_WM_PORT=18080 \
python teleop.py --robot_name=ego_dex --laterality=bimanual
```

## 15. Extension Path For A Real Backend

When replacing the dummy WMServer with a real service, the preferred path is:

1. keep `ego_dex_robot.py` unchanged
2. keep `ego_dex_remote_wm_backend.py` unchanged
3. replace or extend `WMClient` in `ego_dex_wm_client.py`
4. make the real service honor the same `wm_step` and `reset` contract

That keeps the teleop-side integration stable while letting the backend transport or service implementation evolve independently.

## 16. Summary

This design is intentionally narrow.

- `ego_dex_robot.py` orchestrates teleop I/O
- `ego_dex_remote_wm_backend.py` owns remote WM state and requests
- `ego_dex_wm_client.py` owns transport
- `ego_dex_dummy_wm_server.py` provides a test oracle renderer
- raw keypoints are the only correct render-driving input for this path

That separation is the smallest design that preserves the correct visualization behavior and still leaves a clean extension point for a real backend.
