#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from typing import Any, Optional

import zmq

from beavr.teleop.configs.constants import ports


def _format_float(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def _format_status(status: dict[str, Any]) -> str:
    return (
        "[status] "
        f"gate_error={_format_float(status.get('gate_error'))} "
        f"threshold={_format_float(status.get('step_distance_threshold'))} "
        f"blocked={status.get('blocked')} "
        f"reference={status.get('reference_source')} "
        f"ok={status.get('ok')}"
    )


class StatusPrinter(threading.Thread):
    def __init__(self, socket: zmq.Socket, print_period_s: float = 0.5):
        super().__init__(daemon=True)
        self._socket = socket
        self._print_period_s = max(float(print_period_s), 0.1)
        self._stop_event = threading.Event()
        self._latest_status: Optional[dict[str, Any]] = None
        self._latest_lock = threading.Lock()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        last_print_time_s = 0.0
        while not self._stop_event.is_set():
            events = dict(poller.poll(timeout=100))
            if self._socket not in events:
                continue
            try:
                status = self._socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            except Exception:
                continue
            if not isinstance(status, dict):
                continue
            with self._latest_lock:
                self._latest_status = status
            now_s = time.time()
            if now_s - last_print_time_s >= self._print_period_s:
                print(_format_status(status), flush=True)
                last_print_time_s = now_s


def _send_command(socket: zmq.Socket, command: str, value: Optional[float] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"command": command}
    if value is not None:
        payload["value"] = float(value)
    socket.send_json(payload)
    response = socket.recv_json()
    if not isinstance(response, dict):
        return {"ok": False, "error": "invalid response"}
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive EgoDex backend shell")
    parser.add_argument("--host", default=os.getenv("EGO_DEX_BACKEND_CLI_HOST", "127.0.0.1"))
    parser.add_argument(
        "--command-port",
        type=int,
        default=int(os.getenv("EGO_DEX_BACKEND_CLI_COMMAND_PORT", str(ports.EGO_DEX_BACKEND_CLI_COMMAND_PORT))),
    )
    parser.add_argument(
        "--status-port",
        type=int,
        default=int(os.getenv("EGO_DEX_BACKEND_CLI_STATUS_PORT", str(ports.EGO_DEX_BACKEND_CLI_STATUS_PORT))),
    )
    args = parser.parse_args()

    context = zmq.Context()
    command_socket = context.socket(zmq.REQ)
    command_socket.setsockopt(zmq.LINGER, 0)
    command_socket.setsockopt(zmq.RCVTIMEO, 3000)
    command_socket.setsockopt(zmq.SNDTIMEO, 3000)
    command_socket.connect(f"tcp://{args.host}:{args.command_port}")

    status_socket = context.socket(zmq.SUB)
    status_socket.setsockopt(zmq.LINGER, 0)
    status_socket.setsockopt(zmq.CONFLATE, 1)
    status_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    status_socket.connect(f"tcp://{args.host}:{args.status_port}")

    status_printer = StatusPrinter(status_socket)
    status_printer.start()

    print("backend_cli shell")
    print("commands: reset, status, thr <value>, help, exit")
    try:
        while True:
            try:
                raw_command = input("backend> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                break

            if not raw_command:
                continue
            command_parts = raw_command.split()
            command = command_parts[0].lower()
            if command in {"exit", "quit"}:
                break
            if command == "help":
                print("commands: reset, status, thr <value>, help, exit")
                continue
            try:
                if command == "thr":
                    if len(command_parts) != 2:
                        print("usage: thr <value>")
                        continue
                    try:
                        threshold_value = float(command_parts[1])
                    except ValueError:
                        print(f"invalid threshold value: {command_parts[1]}")
                        continue
                    response = _send_command(command_socket, command, threshold_value)
                elif command in {"reset", "status"}:
                    response = _send_command(command_socket, command)
                else:
                    print(f"unsupported command: {raw_command}")
                    continue
            except Exception as exc:
                print(f"[error] failed to send command: {exc}")
                continue
            print(json.dumps(response, sort_keys=True))
    finally:
        status_printer.stop()
        status_printer.join(timeout=1.0)
        status_socket.close(linger=0)
        command_socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
