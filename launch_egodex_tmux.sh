#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="egodex_remote_wm"
CONDA_ENV="beavr_teleop"
HOST="127.0.0.1"
PORT="18080"
FEATHER_JSONL=""
FEATHER_BIND_HOST="0.0.0.0"
FEATHER_PORT="15102"
FEATHER_FPS="15"
FEATHER_FROM_STEP_PAYLOAD=0
WM_HEARTBEAT_HZ=""
ATTACH=1
RUN=0
FRESH=0

usage() {
  cat <<'USAGE'
Usage: ./launch_egodex_tmux.sh [options]

Creates a two-pane tmux session for the EgoDex remote WM demo.

Default behavior:
- creates/reattaches tmux session `egodex_remote_wm`
- left pane: dummy WM server command prefilled
- right pane: teleop command prefilled
- attaches automatically
- you only need to press Enter in each pane

Options:
  --fresh              Kill any existing session with the same name first.
  --run                Start both commands immediately instead of prefilling.
  --detached           Do not attach after creating the session.
  --session NAME       Override tmux session name.
  --env NAME           Override conda environment name.
  --host HOST          WM host passed to both panes. Default: 127.0.0.1
  --port PORT          WM port passed to both panes. Default: 18080
  --feather-jsonl PATH Enable the feather debug point stream from a trajectory JSONL.
  --feather-bind-host HOST
                      Feather PUB bind host. Default: 0.0.0.0
  --feather-port PORT Feather PUB port. Default: 15102
  --feather-fps FPS   Feather stream rate. Default: 15
  --feather-from-step-payload
                      Echo the exact `/step` hand payload back to the feather port.
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
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV="$2"
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
    --feather-jsonl)
      FEATHER_JSONL="$2"
      shift 2
      ;;
    --feather-bind-host)
      FEATHER_BIND_HOST="$2"
      shift 2
      ;;
    --feather-port)
      FEATHER_PORT="$2"
      shift 2
      ;;
    --feather-fps)
      FEATHER_FPS="$2"
      shift 2
      ;;
    --feather-from-step-payload)
      FEATHER_FROM_STEP_PAYLOAD=1
      shift
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
TELEOP_CMD="source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cd \"$REPO_ROOT\" && PYTHONPATH=src EGO_DEX_WM_HOST=$HOST EGO_DEX_WM_PORT=$PORT python teleop.py --robot_name=ego_dex --laterality=bimanual"

if [[ -n "$FEATHER_JSONL" || "$FEATHER_FROM_STEP_PAYLOAD" -eq 1 ]]; then
  WM_CMD+=" --feather-bind-host $FEATHER_BIND_HOST"
  WM_CMD+=" --feather-port $FEATHER_PORT"
  WM_CMD+=" --feather-fps $FEATHER_FPS"
fi

if [[ -n "$FEATHER_JSONL" ]]; then
  WM_CMD+=" --feather-trajectory-jsonl \"$FEATHER_JSONL\""
fi

if [[ "$FEATHER_FROM_STEP_PAYLOAD" -eq 1 ]]; then
  WM_CMD+=" --feather-from-step-payload"
fi

if [[ -n "$WM_HEARTBEAT_HZ" ]]; then
  TELEOP_CMD="source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cd \"$REPO_ROOT\" && PYTHONPATH=src EGO_DEX_WM_HOST=$HOST EGO_DEX_WM_PORT=$PORT EGO_DEX_WM_HEARTBEAT_HZ=$WM_HEARTBEAT_HZ python teleop.py --robot_name=ego_dex --laterality=bimanual"
fi

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
LEFT_PANE="$(tmux list-panes -t "${SESSION_NAME}:egodex" -F '#{pane_id}' | head -n 1)"
RIGHT_PANE="$(tmux split-window -h -t "$LEFT_PANE" -c "$REPO_ROOT" -P -F '#{pane_id}')"
tmux select-layout -t "${SESSION_NAME}:egodex" even-horizontal >/dev/null

tmux select-pane -t "$LEFT_PANE"
tmux send-keys -t "$LEFT_PANE" C-c
sleep 0.05
tmux send-keys -t "$LEFT_PANE" -l "$WM_CMD"


tmux select-pane -t "$RIGHT_PANE"
tmux send-keys -t "$RIGHT_PANE" C-c
sleep 0.05
tmux send-keys -t "$RIGHT_PANE" -l "$TELEOP_CMD"

if [[ "$RUN" -eq 1 ]]; then
  tmux send-keys -t "$LEFT_PANE" Enter
  sleep 0.2
  tmux send-keys -t "$RIGHT_PANE" Enter
fi

tmux select-pane -t "$LEFT_PANE"

if [[ "$ATTACH" -eq 1 ]]; then
  exec tmux attach-session -t "$SESSION_NAME"
fi

echo "Created tmux session: $SESSION_NAME"
echo "Left pane: dummy WM server"
echo "Right pane: teleop"
if [[ "$RUN" -eq 0 ]]; then
  echo "Commands are prefilled. Press Enter in each pane to start."
fi
