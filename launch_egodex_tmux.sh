#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="egodex_remote_wm"
CONDA_ENV="beavr_teleop"
WM_SCHEME="http"
HOST="127.0.0.1"
PORT="18080"
RESET_POSE_JSONL=""
VISUALIZE_HANDS_ENABLE=1
VISUALIZE_HANDS_BIND_HOST="0.0.0.0"
VISUALIZE_HANDS_PORT="15102"
VISUALIZE_HANDS_FPS="15"
STEP_DISTANCE_THRESHOLD=""
STEP_HAND_SCALING_ENABLE=""
STEP_HAND_SCALE_MIN=""
STEP_HAND_SCALE_MAX=""
WM_HEARTBEAT_HZ=""
LOCAL_WM_ENABLE=0
ATTACH=0
RUN=1
FRESH=1

usage() {
  cat <<'USAGE'
Usage: ./launch_egodex_tmux.sh [options]

Creates a two-pane tmux session for the EgoDex remote WM demo.

Default behavior:
- kills any existing tmux session `egodex_remote_wm`
- launches teleop immediately in a detached tmux session
- connects to an external WM by default
- enables teleop hand visualization by default
- in the common case, you only need to pass `--host` and `--port`

Options:
  --fresh              Kill any existing session with the same name first. Default.
  --run                Start both commands immediately instead of prefilling. Default.
  --detached           Do not attach after creating the session. Default.
  --attach             Attach after creating the session.
  --prefill            Prefill commands instead of starting them immediately.
  --session NAME       Override tmux session name.
  --env NAME           Override conda environment name.
  --wm-scheme SCHEME   WM URL scheme for teleop. Default: http
  --host HOST          WM host passed to both panes. Default: 127.0.0.1
  --port PORT          WM port passed to both panes. Default: 18080
  --external-wm        Do not launch the local dummy WM server pane. Default.
  --local-wm           Launch the local dummy WM server pane.
  --reset-pose-jsonl PATH
                      Seed dummy-server `/reset` from a random JSONL frame.
                      This also enables teleop hand visualization.
  --visualize-hands   Enable teleop hand visualization without a reset JSONL. Default.
  --no-visualize-hands
                      Disable teleop hand visualization.
  --visualize-hands-bind-host HOST
                      Teleop hand-visualization PUB bind host. Default: 0.0.0.0
  --visualize-hands-port PORT
                      Teleop hand-visualization PUB port. Default: 15102
  --visualize-hands-fps FPS
                      Teleop hand-visualization stream rate. Default: 15
  --step-distance-threshold VALUE
                      Override the teleop `/step` gating threshold.
  --disable-step-hand-scaling
                      Disable teleop per-bone hand scaling before `/step`.
  --step-hand-scale-min VALUE
                      Minimum per-bone hand scale clamp. Default: config/env
  --step-hand-scale-max VALUE
                      Maximum per-bone hand scale clamp. Default: config/env
  --wm-heartbeat-hz HZ
                      WM request/update rate for teleop. Defaults to config/env.
  -h, --help           Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      FRESH=1
      shift
      ;;
    --run)
      RUN=1
      shift
      ;;
    --detached)
      ATTACH=0
      shift
      ;;
    --attach)
      ATTACH=1
      shift
      ;;
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --wm-scheme)
      WM_SCHEME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --external-wm)
      LOCAL_WM_ENABLE=0
      shift
      ;;
    --local-wm)
      LOCAL_WM_ENABLE=1
      shift
      ;;
    --reset-pose-jsonl)
      RESET_POSE_JSONL="$2"
      VISUALIZE_HANDS_ENABLE=1
      LOCAL_WM_ENABLE=1
      shift 2
      ;;
    --visualize-hands)
      VISUALIZE_HANDS_ENABLE=1
      shift
      ;;
    --no-visualize-hands)
      VISUALIZE_HANDS_ENABLE=0
      shift
      ;;
    --visualize-hands-bind-host)
      VISUALIZE_HANDS_BIND_HOST="$2"
      shift 2
      ;;
    --visualize-hands-port)
      VISUALIZE_HANDS_PORT="$2"
      shift 2
      ;;
    --visualize-hands-fps)
      VISUALIZE_HANDS_FPS="$2"
      shift 2
      ;;
    --feather-jsonl)
      RESET_POSE_JSONL="$2"
      VISUALIZE_HANDS_ENABLE=1
      LOCAL_WM_ENABLE=1
      shift 2
      ;;
    --feather-enable)
      VISUALIZE_HANDS_ENABLE=1
      shift
      ;;
    --feather-bind-host)
      VISUALIZE_HANDS_BIND_HOST="$2"
      shift 2
      ;;
    --feather-port)
      VISUALIZE_HANDS_PORT="$2"
      shift 2
      ;;
    --feather-fps)
      VISUALIZE_HANDS_FPS="$2"
      shift 2
      ;;
    --prefill)
      RUN=0
      shift
      ;;
    --step-distance-threshold)
      STEP_DISTANCE_THRESHOLD="$2"
      shift 2
      ;;
    --disable-step-hand-scaling)
      STEP_HAND_SCALING_ENABLE="0"
      shift
      ;;
    --step-hand-scale-min)
      STEP_HAND_SCALE_MIN="$2"
      shift 2
      ;;
    --step-hand-scale-max)
      STEP_HAND_SCALE_MAX="$2"
      shift 2
      ;;
    --wm-heartbeat-hz)
      WM_HEARTBEAT_HZ="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not installed or not on PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Could not find conda activation script at: $CONDA_SH" >&2
  exit 1
fi

WM_CMD="source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cd \"$REPO_ROOT\" && PYTHONPATH=src python -m beavr.teleop.components.interface.robots.ego_dex_dummy_wm_server --host $HOST --port $PORT"
if [[ -n "$RESET_POSE_JSONL" ]]; then
  WM_CMD+=" --reset-pose-jsonl \"$RESET_POSE_JSONL\""
fi

TELEOP_ENV="PYTHONPATH=src EGO_DEX_WM_SCHEME=$WM_SCHEME EGO_DEX_WM_HOST=$HOST EGO_DEX_WM_PORT=$PORT"
if [[ -n "$WM_HEARTBEAT_HZ" ]]; then
  TELEOP_ENV+=" EGO_DEX_WM_HEARTBEAT_HZ=$WM_HEARTBEAT_HZ"
fi
if [[ "$VISUALIZE_HANDS_ENABLE" -eq 1 ]]; then
  TELEOP_ENV+=" EGO_DEX_VISUALIZE_HANDS_ENABLE=1"
  TELEOP_ENV+=" EGO_DEX_VISUALIZE_HANDS_BIND_HOST=$VISUALIZE_HANDS_BIND_HOST"
  TELEOP_ENV+=" EGO_DEX_VISUALIZE_HANDS_PORT=$VISUALIZE_HANDS_PORT"
  TELEOP_ENV+=" EGO_DEX_VISUALIZE_HANDS_FPS=$VISUALIZE_HANDS_FPS"
fi
if [[ -n "$STEP_DISTANCE_THRESHOLD" ]]; then
  TELEOP_ENV+=" EGO_DEX_STEP_DISTANCE_THRESHOLD=$STEP_DISTANCE_THRESHOLD"
fi
if [[ -n "$STEP_HAND_SCALING_ENABLE" ]]; then
  TELEOP_ENV+=" EGO_DEX_STEP_HAND_SCALING_ENABLE=$STEP_HAND_SCALING_ENABLE"
fi
if [[ -n "$STEP_HAND_SCALE_MIN" ]]; then
  TELEOP_ENV+=" EGO_DEX_STEP_HAND_SCALE_MIN=$STEP_HAND_SCALE_MIN"
fi
if [[ -n "$STEP_HAND_SCALE_MAX" ]]; then
  TELEOP_ENV+=" EGO_DEX_STEP_HAND_SCALE_MAX=$STEP_HAND_SCALE_MAX"
fi
TELEOP_CMD="source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cd \"$REPO_ROOT\" && $TELEOP_ENV python teleop.py --robot_name=ego_dex --laterality=bimanual"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  if [[ "$FRESH" -eq 1 ]]; then
    tmux kill-session -t "$SESSION_NAME"
  else
    if [[ "$ATTACH" -eq 1 ]]; then
      exec tmux attach-session -t "$SESSION_NAME"
    fi
    echo "Session already exists: $SESSION_NAME"
    exit 0
  fi
fi

tmux new-session -d -s "$SESSION_NAME" -n egodex -c "$REPO_ROOT"
PRIMARY_PANE="$(tmux list-panes -t "${SESSION_NAME}:egodex" -F '#{pane_id}' | head -n 1)"
TELEOP_PANE="$PRIMARY_PANE"
if [[ "$LOCAL_WM_ENABLE" -eq 1 ]]; then
  TELEOP_PANE="$(tmux split-window -h -t "$PRIMARY_PANE" -c "$REPO_ROOT" -P -F '#{pane_id}')"
  tmux select-layout -t "${SESSION_NAME}:egodex" even-horizontal >/dev/null

  tmux select-pane -t "$PRIMARY_PANE"
  tmux send-keys -t "$PRIMARY_PANE" C-c
  sleep 0.05
  tmux send-keys -t "$PRIMARY_PANE" -l "$WM_CMD"
fi

tmux select-pane -t "$TELEOP_PANE"
tmux send-keys -t "$TELEOP_PANE" C-c
sleep 0.05
tmux send-keys -t "$TELEOP_PANE" -l "$TELEOP_CMD"

if [[ "$RUN" -eq 1 ]]; then
  if [[ "$LOCAL_WM_ENABLE" -eq 1 ]]; then
    tmux send-keys -t "$PRIMARY_PANE" Enter
    sleep 0.2
  fi
  tmux send-keys -t "$TELEOP_PANE" Enter
fi

tmux select-pane -t "$PRIMARY_PANE"

if [[ "$ATTACH" -eq 1 ]]; then
  exec tmux attach-session -t "$SESSION_NAME"
fi

echo "Created tmux session: $SESSION_NAME"
if [[ "$LOCAL_WM_ENABLE" -eq 1 ]]; then
  echo "Left pane: dummy WM server"
  echo "Right pane: teleop"
else
  echo "Pane: teleop"
fi
if [[ "$RUN" -eq 0 ]]; then
  if [[ "$LOCAL_WM_ENABLE" -eq 1 ]]; then
    echo "Commands are prefilled. Press Enter in each pane to start."
  else
    echo "Command is prefilled. Press Enter in the pane to start."
  fi
fi
