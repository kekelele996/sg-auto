#!/usr/bin/env python3
"""Open a queued platform task in the ChatGPT desktop app and track it."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from api.common import queue_prompt_sha256
from queue_log import LogWriter, RolloutLogFollower, discover_rollout, log_id_from_result_file, public_log_path, task_marker

DEFAULT_PUSH_HELPERS = (
    Path.home() / "no_cloud" / "common" / "solo2-monitor" / "CodexQueuePush.app" / "Contents" / "MacOS" / "CodexQueuePush",
    Path.home() / "no_cloud" / "common" / "solo2-migration-20260915" / "payload" / "program" / "solo2-monitor" / "CodexQueuePush.app" / "Contents" / "MacOS" / "CodexQueuePush",
)
DEFAULT_WAIT_TIMEOUT = 12 * 60 * 60
DEFAULT_STARTUP_TIMEOUT = 5 * 60
DEFAULT_TERMINAL_STABILITY_SECONDS = 6.0
TERMINAL_FAILURE_STATES = {"blocked", "failed", "error"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_push_helper(value: Path | None = None) -> Path:
    candidates = [value.expanduser()] if value else []
    candidates.extend(DEFAULT_PUSH_HELPERS)
    discovered = shutil.which("CodexQueuePush")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise FileNotFoundError("找不到可执行的 CodexQueuePush 辅助程序")


def write_result(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path, default: dict | None = None) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default or {}
    return value if isinstance(value, dict) else (default or {})


def build_prompt(*, task_name: str, task_root: Path, workdir: Path, workdir_prompt: str) -> str:
    """The queue preamble for one task.

    The working directory is no longer spelled out: it always follows the
    selected ChatGPT folder, so the only thing the executor needs from us is the
    reserved task root it must use.
    """
    return (
        "本次监控队列已分配唯一任务名。\n"
        f"- {task_marker(task_name)}\n"
        f"- 监控台已预留空任务根目录：`{task_root}`\n"
        "- 该任务根目录是唯一允许使用的位置，禁止在 sologsb-harness 或其他目录创建任务。\n"
        "- 初始化时必须显式传入 `--task-root` 指向上面的绝对路径，不得只依赖当前目录。\n\n"
        f"{workdir_prompt.strip()}\n"
    )


def build_deep_link(*, workdir: Path, prompt: str, folder_id: str = "") -> str:
    """Build the ChatGPT deep link.

    ``projectId`` is not one of the five parameters that trigger the app's
    ``WD()`` route handler, but ``WD()`` reads and forwards it, so passing the
    raw ``projects.id`` from the app's own database makes the new thread land in
    the selected folder.  ``path`` is still sent so the working directory is
    correct even when the id is not honoured.
    """
    params: dict[str, str] = {
        "path": str(workdir),
        "mode": "work",
        "prompt": prompt,
    }
    if folder_id:
        params["projectId"] = folder_id
    return "codex://threads/new?" + urllib.parse.urlencode(params)


def _last_log_line() -> str:
    log_path = Path.home() / "Library" / "Logs" / "CodexQueuePush.log"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return lines[-1] if lines else ""


def _write_running_result(
    result_file: Path,
    *,
    stage: str,
    workdir: Path,
    task_root: Path,
    args: argparse.Namespace,
    push_helper: Path,
    trigger_prompt_file: Path,
    deep_link: str,
    state_status: str = "",
    prompt_sha256: str = "",
) -> None:
    write_result(result_file, {
        "status": "running",
        "stage": stage,
        "workdir": str(workdir),
        "taskRoot": str(task_root),
        "taskName": args.task_name,
        "projectCode": args.project_code,
        "taskType": args.task_type,
        "difficulty": args.difficulty,
        "side": args.side,
        "pushHelper": str(push_helper),
        "triggerPromptPath": str(trigger_prompt_file),
        "deepLink": deep_link,
        "stateStatus": state_status,
        "promptSha256": prompt_sha256,
        "folderId": str(getattr(args, "folder_id", "") or ""),
        "folderPath": str(getattr(args, "folder_path", "") or ""),
        "variantId": str(getattr(args, "variant_id", "") or ""),
        # Quota bookkeeping the monitor reads back.  The monitor pre-deducts a
        # platform task before launching the worker; the executor reports the
        # task it actually used so the two can be reconciled.
        "platformTaskId": str(getattr(args, "platform_task_id", "") or ""),
        "platformTaskNo": str(getattr(args, "platform_task_no", "") or ""),
        "quotaReportedAt": utc_now(),
    })


def wait_for_task(
    result_file: Path,
    *,
    workdir: Path,
    task_root: Path,
    args: argparse.Namespace,
    push_helper: Path,
    trigger_prompt_file: Path,
    deep_link: str,
    timeout: float,
    startup_timeout: float,
    submitted_at: float,
    writer: LogWriter,
    follower: RolloutLogFollower | None = None,
    prompt_sha256: str = "",
    terminal_stability_seconds: float = DEFAULT_TERMINAL_STABILITY_SECONDS,
) -> int:
    deadline = time.monotonic() + timeout
    last_status = ""
    terminal_fingerprint = ""
    terminal_since = 0.0
    while time.monotonic() < deadline:
        if follower is not None:
            follower.pump()
        state_path = task_root / "monitor" / "state.json"
        init_failure_path = task_root / "monitor" / "init-failure.json"
        init_failure = read_json(init_failure_path, {})
        if init_failure:
            error = str(init_failure.get("error") or init_failure.get("reason") or "任务初始化失败")
            writer.emit(f"[失败] 任务初始化失败：{error}")
            write_result(result_file, {
                "status": "failed",
                "stage": "task-init",
                "workdir": str(workdir),
                "taskRoot": str(task_root),
                "error": error,
                "triggerPromptPath": str(trigger_prompt_file),
                "deepLink": deep_link,
            })
            return 2
        state = read_json(state_path, {})
        status = str(state.get("status") or "").strip()
        if (
            startup_timeout > 0
            and not task_root.exists()
            and time.monotonic() - submitted_at >= startup_timeout
        ):
            error = f"桌面任务启动超时：{startup_timeout:g} 秒内未创建任务目录"
            writer.emit(f"[失败] {error}")
            write_result(result_file, {
                "status": "failed",
                "stage": "desktop-start-timeout",
                "workdir": str(workdir),
                "taskRoot": str(task_root),
                "stateStatus": last_status,
                "error": error,
                "triggerPromptPath": str(trigger_prompt_file),
                "deepLink": deep_link,
            })
            return 2
        if status and status != last_status:
            writer.emit(f"[状态] {status}")
            _write_running_result(
                result_file,
                stage="desktop-task-running",
                workdir=workdir,
                task_root=task_root,
                args=args,
                push_helper=push_helper,
                trigger_prompt_file=trigger_prompt_file,
                deep_link=deep_link,
                state_status=status,
                prompt_sha256=prompt_sha256,
            )
            last_status = status
        terminal = status == "complete" or status in TERMINAL_FAILURE_STATES
        if terminal:
            fingerprint = status
            now = time.monotonic()
            if fingerprint != terminal_fingerprint:
                terminal_fingerprint = fingerprint
                terminal_since = now
            if terminal_stability_seconds <= 0 or now - terminal_since >= terminal_stability_seconds:
                if status == "complete":
                    writer.emit("[完成] 任务状态已连续稳定为 complete")
                    write_result(result_file, {
                        "status": "finished",
                        "stage": "task-complete",
                        "workdir": str(workdir),
                        "taskRoot": str(task_root),
                        "stateStatus": status,
                        "triggerPromptPath": str(trigger_prompt_file),
                        "deepLink": deep_link,
                    })
                    return 0
                writer.emit(f"[失败] 任务状态已连续稳定为 {status}")
                write_result(result_file, {
                    "status": "failed",
                    "stage": "task-state",
                    "workdir": str(workdir),
                    "taskRoot": str(task_root),
                    "stateStatus": status,
                    "error": f"ChatGPT 任务状态为 {status}",
                    "triggerPromptPath": str(trigger_prompt_file),
                    "deepLink": deep_link,
                })
                return 2
        else:
            terminal_fingerprint = ""
            terminal_since = 0.0
        time.sleep(2)
    write_result(result_file, {
        "status": "failed",
        "stage": "wait-timeout",
        "workdir": str(workdir),
        "taskRoot": str(task_root),
        "stateStatus": last_status,
        "error": f"等待 ChatGPT 任务完成超时（{timeout:g} 秒）",
        "triggerPromptPath": str(trigger_prompt_file),
        "deepLink": deep_link,
    })
    writer.emit(f"[失败] 等待任务完成超时（{timeout:g} 秒）")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-script", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--push-helper", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--project-code", required=True)
    parser.add_argument("--task-type", default="0-1代码生成")
    parser.add_argument("--difficulty", default="困难")
    parser.add_argument("--side", default="both", choices=["A", "B", "both"])
    parser.add_argument("--wait-timeout", type=float, default=DEFAULT_WAIT_TIMEOUT)
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT)
    parser.add_argument("--terminal-stability-seconds", type=float, default=DEFAULT_TERMINAL_STABILITY_SECONDS)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--trigger-prompt-file", type=Path, required=True)
    # ChatGPT app folder the new thread should be created in.
    parser.add_argument("--folder-id", default="")
    parser.add_argument("--folder-path", default="")
    # Platform task the monitor pre-deducted for this queue item.
    parser.add_argument("--platform-task-id", default="")
    parser.add_argument("--platform-task-no", default="")
    parser.add_argument("--variant-id", default="")
    args = parser.parse_args()

    workdir = args.workdir.expanduser().resolve()
    result_file = args.result_file.expanduser().resolve()
    trigger_prompt_file = args.trigger_prompt_file.expanduser().resolve()
    task_root = workdir / args.task_name
    log_path = args.log_file.expanduser().resolve() if args.log_file else public_log_path(log_id_from_result_file(result_file))
    writer = LogWriter(log_path)
    writer.emit(f"[启动] 队列任务 {args.task_name} / {args.project_code}")
    writer.emit(f"[环境] 工作目录：{workdir}")
    if not workdir.is_dir():
        writer.emit(f"[失败] 工作目录不存在：{workdir}")
        write_result(result_file, {
            "status": "failed",
            "stage": "validate-workdir",
            "workdir": str(workdir),
            "error": f"工作目录不存在: {workdir}",
        })
        return 2
    if not trigger_prompt_file.is_file():
        writer.emit(f"[失败] 触发 Prompt 不存在：{trigger_prompt_file}")
        write_result(result_file, {
            "status": "failed",
            "stage": "validate-prompt",
            "workdir": str(workdir),
            "error": f"触发 Prompt 不存在: {trigger_prompt_file}",
        })
        return 2
    trigger_prompt = trigger_prompt_file.read_text(encoding="utf-8").strip()
    if not trigger_prompt:
        writer.emit("[失败] 触发 Prompt 为空")
        write_result(result_file, {
            "status": "failed",
            "stage": "validate-prompt",
            "workdir": str(workdir),
            "error": "触发 Prompt 为空",
        })
        return 2
    try:
        push_helper = resolve_push_helper(args.push_helper)
    except FileNotFoundError as exc:
        writer.emit(f"[失败] {exc}")
        write_result(result_file, {
            "status": "failed",
            "stage": "resolve-push-helper",
            "workdir": str(workdir),
            "taskRoot": str(task_root),
            "error": str(exc),
        })
        print(str(exc), file=sys.stderr, flush=True)
        return 2

    prompt = build_prompt(
        task_name=args.task_name,
        task_root=task_root,
        workdir=workdir,
        workdir_prompt=trigger_prompt,
    )
    prompt_sha256 = queue_prompt_sha256(prompt)
    deep_link = build_deep_link(
        workdir=workdir,
        prompt=prompt,
        folder_id=str(getattr(args, "folder_id", "") or ""),
    )
    writer.emit("[桌面] 正在打开 ChatGPT 并提交任务")
    if args.folder_id:
        writer.emit(f"[文件夹] 目标 projectId={args.folder_id} path={args.folder_path or '-'}")
    if args.platform_task_id:
        writer.emit(f"[配额] 监控台已预扣除 platformTaskId={args.platform_task_id} taskNo={args.platform_task_no or '-'}")
    _write_running_result(
        result_file,
        stage="desktop-submitting",
        workdir=workdir,
        task_root=task_root,
        args=args,
        push_helper=push_helper,
        trigger_prompt_file=trigger_prompt_file,
        deep_link=deep_link,
        prompt_sha256=prompt_sha256,
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        subprocess.run(
            ["open", "-a", "ChatGPT", str(workdir)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        time.sleep(0.5)
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        completed = subprocess.run(
            [str(push_helper), deep_link],
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=40,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        writer.emit(f"[失败] 启动 ChatGPT 桌面会话失败：{exc}")
        write_result(result_file, {
            "status": "failed",
            "stage": "desktop-submit",
            "workdir": str(workdir),
            "taskRoot": str(task_root),
            "error": str(exc),
            "pushHelper": str(push_helper),
            "deepLink": deep_link,
        })
        print(f"启动 ChatGPT 桌面会话失败: {exc}", file=sys.stderr, flush=True)
        return 2

    if completed.returncode != 0:
        error = _last_log_line() or (completed.stderr or completed.stdout).strip()
        if completed.returncode == 3:
            error = "CodexQueuePush 缺少辅助功能权限，请在系统设置中授权"
        writer.emit(f"[失败] {error or f'CodexQueuePush 退出码 {completed.returncode}'}")
        write_result(result_file, {
            "status": "failed",
            "stage": "desktop-submit",
            "workdir": str(workdir),
            "taskRoot": str(task_root),
            "pushHelper": str(push_helper),
            "deepLink": deep_link,
            "exitCode": completed.returncode,
            "error": error or f"CodexQueuePush 退出码 {completed.returncode}",
        })
        return 2

    if args.wait_timeout <= 0:
        writer.emit("[桌面] 任务已提交；本次不等待执行结果")
        write_result(result_file, {
            "status": "finished",
            "stage": "desktop-submitted",
            "workdir": str(workdir),
            "taskRoot": str(task_root),
            "pushHelper": str(push_helper),
            "triggerPromptPath": str(trigger_prompt_file),
            "deepLink": deep_link,
        })
        return 0

    _write_running_result(
        result_file,
        stage="desktop-submitted",
        workdir=workdir,
        task_root=task_root,
        args=args,
        push_helper=push_helper,
        trigger_prompt_file=trigger_prompt_file,
        deep_link=deep_link,
        prompt_sha256=prompt_sha256,
    )
    writer.emit("[桌面] 任务已提交，正在定位 ChatGPT 执行轨迹")
    submitted_at = time.monotonic()
    follower: RolloutLogFollower | None = None
    initial_state = read_json(task_root / "monitor" / "state.json", {})
    initial_status = str(initial_state.get("status") or "")
    initial_failure = read_json(task_root / "monitor" / "init-failure.json", {})
    if initial_status in {"complete", *TERMINAL_FAILURE_STATES} or initial_failure:
        writer.emit("[日志] 任务已有终态记录，跳过轨迹定位")
    else:
        rollout = discover_rollout(args.task_name, timeout=30)
        if rollout is None:
            writer.emit("[警告] 未定位到 ChatGPT rollout，将继续按任务状态跟踪")
        else:
            writer.emit(f"[日志] 已连接 ChatGPT 轨迹：{rollout}")
            follower = RolloutLogFollower(rollout, writer, args.task_name)
            follower.pump()
    return wait_for_task(
        result_file,
        workdir=workdir,
        task_root=task_root,
        args=args,
        push_helper=push_helper,
        trigger_prompt_file=trigger_prompt_file,
        deep_link=deep_link,
        timeout=args.wait_timeout,
        startup_timeout=args.startup_timeout,
        submitted_at=submitted_at,
        writer=writer,
        follower=follower,
        prompt_sha256=prompt_sha256,
        terminal_stability_seconds=args.terminal_stability_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
