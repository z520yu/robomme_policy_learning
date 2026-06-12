#!/usr/bin/env python3
"""Chunked RoboMME eval launcher.

Runs the official examples/robomme/eval.py in short episode chunks so the
simulator process is restarted periodically. This avoids long-lived Vulkan
renderer failures while preserving the official rollout logic.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


TASKS_DEFAULT = "PickXtimes,VideoUnmask,VideoRepick,MoveCube"

MODELS = {
    "pi05_baseline": {
        "label": "pi05_baseline_4tasks50ep_gpu2",
        "config": "pi05_baseline",
        "checkpoint": "runs/ckpts/pi05_baseline/pi05_baseline/79999",
        "history_flag": "--args.no-use-history",
        "port": 28411,
    },
    "perceptual-framesamp-modul": {
        "label": "perceptual-framesamp-modul_4tasks50ep_gpu2",
        "config": "mme_vla_suite",
        "checkpoint": "runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999",
        "history_flag": "--args.use-history",
        "port": 28412,
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def bool_progress(progress: dict[str, dict[str, Any]], tasks: list[str], episodes: int) -> dict[str, dict[str, bool]]:
    actual: dict[str, dict[str, bool]] = {}
    for task in tasks:
        task_progress = progress.get(task, {})
        actual[task] = {}
        for ep in range(episodes):
            value = task_progress.get(str(ep))
            if isinstance(value, bool):
                actual[task][str(ep)] = value
    return actual


def write_chunk_progress(
    progress_path: Path,
    actual: dict[str, dict[str, bool]],
    tasks: list[str],
    current_task: str,
    pending_eps: list[int],
    episodes: int,
) -> None:
    # Do not write empty tasks into progress.json. The official eval script
    # computes rates for every key in log_dict and crashes on empty tasks.
    chunk_progress: dict[str, dict[str, Any]] = {
        task: dict(values) for task, values in actual.items() if values
    }
    pending = {str(ep) for ep in pending_eps}
    task_progress = chunk_progress.setdefault(current_task, {})
    for ep in range(episodes):
        key = str(ep)
        if key in pending:
            task_progress.pop(key, None)
        elif key not in task_progress:
            # Temporary skip marker. It is removed again after the chunk.
            task_progress[key] = False
    write_json(progress_path, chunk_progress)


def clean_progress(progress_path: Path, actual: dict[str, dict[str, bool]]) -> None:
    compact_actual = {task: dict(values) for task, values in actual.items() if values}
    write_json(progress_path, compact_actual)
    log_path = progress_path.with_name("log.json")
    if log_path.exists():
        log_path.unlink()


def write_final_outputs(base_dir: Path, actual: dict[str, dict[str, bool]], tasks: list[str], episodes: int) -> None:
    write_json(base_dir / "progress.json", actual)
    success_rate: dict[str, float] = {}
    for task in tasks:
        values = [actual.get(task, {}).get(str(ep), False) for ep in range(episodes)]
        success_rate[task] = sum(values) / len(values)
    total_success_rate = sum(success_rate.values()) / len(success_rate) if success_rate else 0.0
    write_json(base_dir / "log.json", {"success_rate": success_rate, "total_success_rate": total_success_rate})
    detail: dict[str, Any] = {}
    total_success = 0
    total_count = 0
    for task in tasks:
        values = [actual.get(task, {}).get(str(ep), False) for ep in range(episodes)]
        success = sum(values)
        total_success += success
        total_count += len(values)
        detail[task] = {
            "success": success,
            "evaluated": len(values),
            "rate": success / len(values),
            "values": values,
        }
    detail["__overall__"] = {
        "success": total_success,
        "evaluated": total_count,
        "rate": total_success / total_count if total_count else 0.0,
    }
    write_json(base_dir / "chunked_summary.json", detail)


def wait_for_server(proc: subprocess.Popen[Any], log_path: Path, timeout_s: int) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-8000:] if log_path.exists() else ""
            raise RuntimeError(f"server exited with {proc.returncode}\n{tail}")
        if log_path.exists() and "server listening" in log_path.read_text(errors="replace"):
            return
        time.sleep(2)
    tail = log_path.read_text(errors="replace")[-8000:] if log_path.exists() else ""
    raise TimeoutError(f"server did not become ready in {timeout_s}s\n{tail}")


def terminate(proc: subprocess.Popen[Any], grace_s: int = 10) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=grace_s)


def run_client(
    *,
    sim_python: Path,
    env: dict[str, str],
    log_path: Path,
    port: int,
    task: str,
    save_dir: str,
    policy_name: str,
    seed: int,
    ckpt_id: int,
    history_flag: str,
    timeout_s: int,
) -> int:
    cmd = [
        str(sim_python),
        "-u",
        "examples/robomme/eval.py",
        "--args.host=127.0.0.1",
        f"--args.port={port}",
        "--args.obs-horizon=16",
        "--args.max-steps=1300",
        f"--args.save-dir={save_dir}",
        history_flag,
        f"--args.policy-name={policy_name}",
        f"--args.model-seed={seed}",
        f"--args.model-ckpt-id={ckpt_id}",
        f"--args.only-tasks={task}",
    ]
    with log_path.open("ab") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return 124


def run_model(args: argparse.Namespace, model_key: str) -> None:
    spec = MODELS[model_key]
    tasks = args.tasks.split(",")
    policy_name = args.policy_name_override or f"{spec['label']}_{args.run_id}"
    base_dir = Path(args.save_dir) / policy_name / f"ckpt{args.ckpt_id}" / f"seed{args.seed}"
    progress_path = base_dir / "progress.json"
    base_dir.mkdir(parents=True, exist_ok=True)

    existing = read_json(progress_path, {})
    actual = bool_progress(existing, tasks, args.episodes)
    clean_progress(progress_path, actual)

    policy_python = Path(os.environ["ROBOMME_POLICY_VENV"]) / "bin/python"
    sim_python = Path(os.environ["ROBOMME_SIM_VENV"]) / "bin/python"
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    server_log = log_root / f"{policy_name}_server.log"
    client_log = log_root / f"{policy_name}_client.log"
    server_env = os.environ.copy()
    server_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(args.xla_mem_fraction),
        }
    )
    client_env = os.environ.copy()
    client_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "SAPIEN_RENDER_DEVICE": "cuda",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    server_cmd = [
        str(policy_python),
        "scripts/serve_policy.py",
        f"--seed={args.seed}",
        f"--port={spec['port']}",
        "policy:checkpoint",
        f"--policy.dir={spec['checkpoint']}",
        f"--policy.config={spec['config']}",
    ]

    log(f"===== START_MODEL {policy_name} =====")
    with server_log.open("ab") as server_file:
        server_proc = subprocess.Popen(server_cmd, stdout=server_file, stderr=subprocess.STDOUT, env=server_env)

    try:
        wait_for_server(server_proc, server_log, args.server_timeout_s)
        log(f"SERVER_READY {policy_name} port={spec['port']} pid={server_proc.pid}")

        for task in tasks:
            for start in range(0, args.episodes, args.chunk_size):
                end = min(args.episodes, start + args.chunk_size)
                chunk_eps = list(range(start, end))
                pending = [ep for ep in chunk_eps if str(ep) not in actual.get(task, {})]
                if not pending:
                    log(f"SKIP_DONE {policy_name} {task} {start}:{end}")
                    continue

                for attempt in range(1, args.retries + 1):
                    log(f"RUN_CHUNK {policy_name} {task} eps={pending} attempt={attempt}/{args.retries}")
                    write_chunk_progress(progress_path, actual, tasks, task, pending, args.episodes)
                    log_json = base_dir / "log.json"
                    if log_json.exists():
                        log_json.unlink()
                    status = run_client(
                        sim_python=sim_python,
                        env=client_env,
                        log_path=client_log,
                        port=spec["port"],
                        task=task,
                        save_dir=args.save_dir,
                        policy_name=policy_name,
                        seed=args.seed,
                        ckpt_id=args.ckpt_id,
                        history_flag=spec["history_flag"],
                        timeout_s=args.client_timeout_s,
                    )
                    latest = read_json(progress_path, {})
                    latest_task = latest.get(task, {})
                    actual.setdefault(task, {})
                    for ep in pending:
                        value = latest_task.get(str(ep))
                        if isinstance(value, bool):
                            actual[task][str(ep)] = value
                    clean_progress(progress_path, actual)

                    still_pending = [ep for ep in pending if str(ep) not in actual.get(task, {})]
                    if not still_pending:
                        successes = sum(actual[task][str(ep)] for ep in pending)
                        log(f"CHUNK_DONE {policy_name} {task} {start}:{end} status={status} success={successes}/{len(pending)}")
                        break

                    log(f"CHUNK_INCOMPLETE {policy_name} {task} status={status} remaining={still_pending}")
                    pending = still_pending
                    if attempt == args.retries:
                        raise RuntimeError(f"chunk failed after retries: {policy_name} {task} {still_pending}")
                    time.sleep(args.retry_sleep_s)

        write_final_outputs(base_dir, actual, tasks, args.episodes)
        log(f"MODEL_DONE {policy_name}")
        log(json.dumps(read_json(base_dir / "chunked_summary.json", {}), indent=2))
    finally:
        terminate(server_proc)
        log(f"SERVER_STOPPED {policy_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="pi05_baseline,perceptual-framesamp-modul")
    parser.add_argument("--tasks", default=TASKS_DEFAULT)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep-s", type=int, default=10)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ckpt-id", type=int, default=79999)
    parser.add_argument("--save-dir", default="runs/evaluation_4task_full")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--policy-name-override", default="")
    parser.add_argument("--server-timeout-s", type=int, default=900)
    parser.add_argument("--client-timeout-s", type=int, default=7200)
    parser.add_argument("--xla-mem-fraction", type=float, default=0.82)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    for model_key in args.models.split(","):
        model_key = model_key.strip()
        if not model_key:
            continue
        if model_key not in MODELS:
            raise KeyError(f"unknown model: {model_key}")
        run_model(args, model_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
