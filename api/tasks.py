"""Task discovery, snapshotting and trace parsing.

Two snapshot shapes come out of this module:

* :meth:`TaskScanner.cards` — the slim list payload the card grid renders.  It
  carries only card-sized fields so a 67-task snapshot stays in the low tens of
  kilobytes instead of the megabyte the old full snapshot cost.
* :meth:`TaskScanner.detail` — the heavy per-task payload (sides, candidates,
  workflow, prompt) fetched on demand when a card is opened.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .common import (
    CANDIDATE_ID_RE,
    EVENT_LIMIT,
    RUNNABLE_TASK_STATUSES,
    SIDE_DONE_STATUSES,
    SIDE_FAILED_STATUSES,
    SIDES,
    TERMINAL_TASK_STATUSES,
    FileCache,
    ProcessTable,
    age_seconds,
    iso_from_timestamp,
    parse_time,
    read_json,
    redact_text,
    safe_slug,
    short_hash,
    tail_lines,
)

TRACE_CACHE_LIMIT = 400
PRUNE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode", "source", "workspace"}


# --------------------------------------------------------------------------- #
# trace parsing
# --------------------------------------------------------------------------- #
def _new_trace_stats() -> dict[str, Any]:
    return {
        "sessionId": "",
        "model": "",
        "harnessVersion": "",
        "eventCount": 0,
        "assistantTurns": 0,
        "toolCalls": 0,
        "toolResults": 0,
        "thinkingTokens": 0,
        "apiRetries": 0,
        "compactions": 0,
        "commands": 0,
        "filesTouched": [],
        "lastPhase": "starting",
        "lastEventType": "",
        "lastSummary": "等待首个执行事件",
        "lastText": "",
        "lastTool": "",
        "lastResult": "",
        "lastError": "",
        "result": {},
        "todos": [],
        "todoUpdatedAt": "",
        "recent": [],
        "recentLimit": EVENT_LIMIT,
        "seq": 0,
    }


def _push_event(stats: dict[str, Any], kind: str, title: str, detail: str = "") -> None:
    stats["seq"] += 1
    stats["recent"].append({
        "seq": stats["seq"],
        "kind": kind,
        "title": title,
        "detail": redact_text(detail, 500),
        "at": str(stats.get("currentEventAt") or ""),
    })
    recent_limit = int(stats.get("recentLimit") or EVENT_LIMIT)
    if len(stats["recent"]) > recent_limit:
        del stats["recent"][: len(stats["recent"]) - recent_limit]


def _touch_file(stats: dict[str, Any], path: str) -> None:
    value = str(path or "").strip()
    if not value:
        return
    touched = stats["filesTouched"]
    if value not in touched:
        touched.append(value)
    if len(touched) > 80:
        del touched[: len(touched) - 80]


def _tool_detail(block: dict[str, Any]) -> str:
    name = str(block.get("name") or "Tool")
    payload = block.get("input") if isinstance(block.get("input"), dict) else {}
    if name == "Bash":
        return redact_text(payload.get("command") or "", 420)
    if name in {"Read", "Write", "Edit"}:
        return redact_text(payload.get("file_path") or payload.get("path") or "", 240)
    if name in {"Glob", "Grep"}:
        return redact_text(
            " ".join(str(payload.get(key) or "") for key in ("pattern", "path") if payload.get(key)),
            240,
        )
    try:
        return redact_text(json.dumps(payload, ensure_ascii=False), 360)
    except (TypeError, ValueError):
        return ""


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return redact_text(content, 420)
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict):
                pieces.append(str(item.get("text") or item.get("content") or ""))
            else:
                pieces.append(str(item))
        return redact_text(" ".join(piece for piece in pieces if piece), 420)
    return redact_text(content, 420)


def _content_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict)]


def process_trace_event(stats: dict[str, Any], event: dict[str, Any]) -> None:
    stats["eventCount"] += 1
    event_type = str(event.get("type") or "")
    stats["lastEventType"] = event_type
    stats["currentEventAt"] = str(event.get("timestamp") or "")
    if not stats.get("sessionId"):
        stats["sessionId"] = str(event.get("sessionId") or event.get("session_id") or "")

    if event_type == "system":
        subtype = str(event.get("subtype") or "")
        if subtype == "init":
            stats["sessionId"] = str(event.get("session_id") or event.get("sessionId") or stats["sessionId"])
            stats["model"] = str(event.get("model") or stats["model"])
            stats["harnessVersion"] = str(event.get("claude_code_version") or stats["harnessVersion"])
            stats["lastPhase"] = "thinking"
            stats["lastSummary"] = "会话已建立，等待模型开始执行"
            _push_event(stats, "system", "会话已建立", stats["model"])
        elif subtype == "thinking_tokens":
            delta = event.get("estimated_tokens_delta")
            total = event.get("estimated_tokens")
            if isinstance(total, (int, float)):
                stats["thinkingTokens"] = int(total)
            elif isinstance(delta, (int, float)):
                stats["thinkingTokens"] += int(delta)
            stats["lastPhase"] = "thinking"
            stats["lastSummary"] = "模型正在思考"
        elif subtype in {"api_retry", "api_error"}:
            stats["apiRetries"] += 1
            attempt = event.get("attempt")
            maximum = event.get("max_retries")
            status = event.get("error_status") or event.get("error") or subtype
            message = f"接口重试 {attempt or '-'} / {maximum or '-'}，{status}"
            stats["lastPhase"] = "retrying"
            stats["lastSummary"] = message
            stats["lastError"] = message
            _push_event(stats, "warning", "接口重试", message)
        elif subtype in {"compact_boundary", "compact"}:
            stats["compactions"] += 1
            stats["lastSummary"] = "上下文压缩边界"
            _push_event(stats, "system", "上下文压缩", f"第 {stats['compactions']} 次")
        return

    if event_type == "assistant":
        stats["assistantTurns"] += 1
        for block in _content_blocks(event):
            block_type = str(block.get("type") or "")
            if block_type == "thinking":
                stats["lastPhase"] = "thinking"
                stats["lastSummary"] = "模型正在思考"
                _push_event(stats, "thinking", "思考")
            elif block_type == "text":
                text = redact_text(block.get("text") or "", 520)
                if text:
                    stats["lastPhase"] = "responding"
                    stats["lastText"] = text
                    stats["lastSummary"] = text
                    _push_event(stats, "assistant", "模型输出", text)
            elif block_type == "tool_use":
                name = str(block.get("name") or "Tool")
                detail = _tool_detail(block)
                stats["toolCalls"] += 1
                stats["lastTool"] = name
                stats["lastPhase"] = f"tool:{name}"
                stats["lastSummary"] = f"调用 {name}" + (f"：{detail}" if detail else "")
                if name == "Bash":
                    stats["commands"] += 1
                if name in {"Write", "Edit"}:
                    payload = block.get("input") if isinstance(block.get("input"), dict) else {}
                    _touch_file(stats, str(payload.get("file_path") or payload.get("path") or ""))
                if name == "TodoWrite":
                    payload = block.get("input") if isinstance(block.get("input"), dict) else {}
                    todos = []
                    for item in payload.get("todos") or []:
                        if not isinstance(item, dict):
                            continue
                        content = redact_text(item.get("content") or item.get("activeForm") or "", 500)
                        if not content:
                            continue
                        status = str(item.get("status") or "pending")
                        if status not in {"pending", "in_progress", "completed"}:
                            status = "pending"
                        todos.append({
                            "content": content,
                            "activeForm": redact_text(item.get("activeForm") or "", 500),
                            "status": status,
                        })
                    stats["todos"] = todos
                    stats["todoUpdatedAt"] = str(stats.get("currentEventAt") or "")
                    completed = sum(1 for item in todos if item.get("status") == "completed")
                    detail = f"{completed}/{len(todos)} 完成" if todos else "清空待办"
                _push_event(stats, "todo" if name == "TodoWrite" else "tool", f"调用 {name}", detail)
        return

    if event_type == "user":
        for block in _content_blocks(event):
            block_type = str(block.get("type") or "")
            if block_type == "text":
                text = redact_text(block.get("text") or "", 420)
                if text:
                    stats["lastPhase"] = "prompt"
                    stats["lastSummary"] = text
                    _push_event(stats, "user", "真人输入", text)
                continue
            if block_type != "tool_result":
                continue
            stats["toolResults"] += 1
            text = _result_text(block.get("content"))
            stats["lastResult"] = text
            stats["lastPhase"] = "tool_result"
            stats["lastSummary"] = text or "工具已返回"
            if block.get("is_error"):
                stats["lastError"] = text or "工具执行失败"
                _push_event(stats, "error", "工具失败", stats["lastError"])
            else:
                _push_event(stats, "result", "工具返回", text)
        return

    if event_type == "queue-operation":
        operation = str(event.get("operation") or "queue")
        detail = redact_text(event.get("content") or event.get("message") or operation, 420)
        stats["lastPhase"] = "queued"
        stats["lastSummary"] = f"队列操作 {operation}"
        _push_event(stats, "system", f"队列 {operation}", detail)
        return

    if event_type == "result":
        stats["result"] = {
            key: event.get(key)
            for key in ("subtype", "stop_reason", "num_turns", "duration_ms", "total_cost_usd", "is_error")
            if event.get(key) is not None
        }
        stop_reason = str(event.get("stop_reason") or "")
        if event.get("is_error"):
            stats["lastPhase"] = "error"
            stats["lastError"] = redact_text(event.get("error") or event.get("result") or "回合异常结束", 420)
            _push_event(stats, "error", "回合异常结束", stats["lastError"])
        else:
            stats["lastPhase"] = "done"
            stats["lastSummary"] = f"回合结束 stop_reason={stop_reason or '-'}"
            _push_event(stats, "done", "回合结束", stats["lastSummary"])


def read_trace_events(path: Path | None, *, limit: int = 200, max_bytes: int = 4 * 1024 * 1024) -> list[dict[str, Any]]:
    """Structured events from the tail of a trajectory, oldest first.

    ``read_trace_stats`` parses up to 32 MB, which is far too much for a view that
    is refreshed while you watch.  This reads only the last ``max_bytes`` and
    returns the bounded event window, so a 10 MB trajectory costs ~10 ms.
    """
    if path is None or not path.is_file():
        return []
    stats = _new_trace_stats()
    stats["recentLimit"] = max(1, min(int(limit), 2000))
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            stream.seek(max(0, size - max_bytes))
            for raw in stream.read(max_bytes).splitlines():
                if not raw.strip():
                    continue
                try:
                    event = json.loads(raw.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                if isinstance(event, dict):
                    process_trace_event(stats, event)
    except OSError:
        return []
    return list(stats.get("recent") or [])


class TraceCache:
    """Incrementally parse stream-json logs and retain bounded recent events.

    Entries are evicted oldest-first once ``max_entries`` is reached; the old
    cache never evicted anything, so a long-lived monitor kept every candidate
    directory's state resident forever.
    """

    def __init__(self, max_bytes: int = 16 * 1024 * 1024, *, lightweight: bool = False, max_entries: int = TRACE_CACHE_LIMIT):
        self.max_bytes = max_bytes
        self.lightweight = lightweight
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._entries: dict[str, dict[str, Any]] = {}

    def _evict_locked(self) -> None:
        while len(self._entries) > self.max_entries:
            oldest = next(iter(self._entries))
            self._entries.pop(oldest, None)

    def stats(self, path: Path | None) -> dict[str, Any]:
        if path is None:
            return _new_trace_stats()
        if self.lightweight:
            return _new_trace_stats()
        key = str(path)
        try:
            stat = path.stat()
        except OSError:
            return _new_trace_stats()
        if stat.st_size > self.max_bytes:
            identity = (stat.st_ino, stat.st_dev)
            with self._lock:
                entry = self._entries.get(key)
                if (
                    entry
                    and entry.get("tail") is True
                    and entry.get("identity") == identity
                    and entry.get("size") == stat.st_size
                    and entry.get("mtime") == stat.st_mtime
                ):
                    self._entries[key] = self._entries.pop(key)
                    return copy.deepcopy(entry["stats"])
            stats = self._tail_stats(path, stat.st_mtime)
            with self._lock:
                self._entries[key] = {
                    "identity": identity,
                    "offset": stat.st_size,
                    "partial": b"",
                    "stats": stats,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "tail": True,
                }
                self._evict_locked()
            return copy.deepcopy(stats)

        with self._lock:
            entry = self._entries.get(key)
            identity = (stat.st_ino, stat.st_dev)
            if not entry or entry.get("identity") != identity or stat.st_size < entry.get("offset", 0):
                entry = {
                    "identity": identity,
                    "offset": 0,
                    "partial": b"",
                    "stats": _new_trace_stats(),
                }
                self._entries[key] = entry
            offset = int(entry.get("offset") or 0)
            if stat.st_size > offset:
                try:
                    with path.open("rb") as stream:
                        stream.seek(offset)
                        chunk = stream.read(stat.st_size - offset)
                except OSError:
                    chunk = b""
                entry["offset"] = offset + len(chunk)
                combined = bytes(entry.get("partial") or b"") + chunk
                lines = combined.split(b"\n")
                entry["partial"] = lines.pop() if lines else b""
                for raw in lines:
                    if not raw.strip():
                        continue
                    try:
                        event = json.loads(raw.decode("utf-8", errors="replace"))
                    except ValueError:
                        continue
                    if isinstance(event, dict):
                        process_trace_event(entry["stats"], event)
            entry["mtime"] = stat.st_mtime
            entry["size"] = stat.st_size
            self._entries[key] = self._entries.pop(key)
            self._evict_locked()
            return copy.deepcopy(entry["stats"])

    def _tail_stats(self, path: Path, mtime: float) -> dict[str, Any]:
        stats = _new_trace_stats()
        try:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - self.max_bytes))
                lines = stream.read(self.max_bytes).splitlines()
        except OSError:
            return stats
        for raw in lines:
            try:
                event = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if isinstance(event, dict):
                process_trace_event(stats, event)
        return stats


def read_trace_stats(path: Path | None, *, event_limit: int = 400, max_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    """Parse one trace file on demand and retain a larger event window for detail views."""
    stats = _new_trace_stats()
    stats["recentLimit"] = max(1, min(int(event_limit), 2000))
    if path is None or not path.is_file():
        return stats
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size > max_bytes:
                stream.seek(size - max_bytes)
            for raw in stream.read(max_bytes).splitlines():
                try:
                    event = json.loads(raw.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                if isinstance(event, dict):
                    process_trace_event(stats, event)
    except OSError:
        return stats
    return stats


class DockerCache:
    """``docker ps`` snapshot with a short TTL.

    A failing ``docker`` call is not cached: the old cache stored the error for
    the whole TTL, which stalled the queue for two seconds per transient failure.
    """

    def __init__(self, ttl: float = 2.0):
        self.ttl = max(0.0, float(ttl))
        self._lock = threading.Lock()
        self._at = 0.0
        self._data: dict[str, Any] = {"items": [], "byName": {}, "error": ""}

    @staticmethod
    def _docker_json(value: Any) -> dict[str, Any]:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _ps(*scope: str) -> subprocess.CompletedProcess | str:
        """Run ``docker ps``; return the failure text instead of raising."""
        try:
            return subprocess.run(
                ["docker", "ps", *scope, "--no-trunc", "--format", "{{json .}}"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Docker 不可用：{exc}"

    def _fetch(self) -> dict[str, Any]:
        result = self._ps("-a")
        if isinstance(result, str):
            return {"items": [], "byName": {}, "error": result}
        if result.returncode != 0:
            # A single corrupted container fails the whole ``docker ps -a`` run
            # ("rw layer snapshot not found for container ...").  That reads as
            # "Docker unavailable" and blocks every launch even though the
            # daemon and the running containers are fine.  The scheduler only
            # accounts running containers, so fall back to the running-only
            # listing; only when that fails too is Docker really unusable.
            fallback = self._ps()
            if isinstance(fallback, str) or fallback.returncode != 0:
                return {"items": [], "byName": {}, "error": (result.stderr or "docker ps 失败").strip()}
            result = fallback
        items: list[dict[str, Any]] = []
        by_name: dict[str, dict[str, Any]] = {}
        for line in result.stdout.splitlines():
            parsed = self._docker_json(line)
            if not parsed:
                continue
            item = {
                "id": str(parsed.get("ID") or ""),
                "name": str(parsed.get("Names") or ""),
                "state": str(parsed.get("State") or "").lower(),
                "status": str(parsed.get("Status") or ""),
                "createdAt": str(parsed.get("CreatedAt") or ""),
                "image": str(parsed.get("Image") or ""),
                "labels": str(parsed.get("Labels") or ""),
            }
            items.append(item)
            if item["name"]:
                by_name[item["name"]] = item
        return {"items": items, "byName": by_name, "error": ""}

    def get(self, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            cached = self._data
            if not force and cached.get("fetchedOk") and now - self._at < self.ttl:
                return copy.deepcopy(cached)
        data = self._fetch()
        if data.get("error"):
            # Keep the previous good snapshot visible but report the error so the
            # scheduler refuses to start work it cannot account for.
            with self._lock:
                previous = self._data
                failed = {
                    "items": previous.get("items") or [],
                    "byName": previous.get("byName") or {},
                    "error": str(data.get("error") or ""),
                    "stale": True,
                }
                self._at = now
                self._data = failed
                return copy.deepcopy(failed)
        with self._lock:
            self._at = now
            self._data = {**data, "fetchedOk": True, "stale": False}
            return copy.deepcopy(self._data)


# --------------------------------------------------------------------------- #
# task discovery
# --------------------------------------------------------------------------- #
def _is_task_state(state: dict[str, Any]) -> bool:
    if not isinstance(state, dict):
        return False
    if not str(state.get("taskName") or "").strip():
        return False
    if isinstance(state.get("sides"), dict):
        return True
    return bool(state.get("initialSnapshot") or state.get("promptPath"))


def _git_head(repo: Path) -> str:
    git_dir = repo / ".git"
    if not git_dir.is_dir():
        return ""
    head = git_dir / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if value.startswith("ref: "):
        ref = value[5:].strip()
        try:
            return (git_dir / ref).read_text(encoding="utf-8").strip()[:12]
        except OSError:
            try:
                for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == ref:
                        return parts[0][:12]
            except OSError:
                pass
            return ""
    return value[:12]


def _is_candidate_id(value: Any) -> bool:
    return bool(CANDIDATE_ID_RE.fullmatch(str(value or "")))


def _runtime_root(task_root: Path, identifier: str) -> Path:
    if _is_candidate_id(identifier):
        return task_root / "monitor" / "runtime" / "candidates" / identifier
    return task_root / "monitor" / "runtime" / identifier.lower()


def _rejected_root(task_root: Path, identifier: str) -> Path:
    if _is_candidate_id(identifier):
        return task_root / "workspace" / "轨迹文件" / "candidates" / identifier / "rejected"
    return task_root / "workspace" / "轨迹文件" / identifier.lower() / "rejected"


def _attempt_dirs(task_root: Path, side: str) -> list[Path]:
    root = _runtime_root(task_root, side)
    if not root.is_dir():
        return []

    def attempt_no(item: Path) -> int:
        match = re.search(r"(\d+)$", item.name)
        return int(match.group(1)) if match else 0

    return sorted((item for item in root.glob("attempt-*") if item.is_dir()), key=attempt_no)


def _trace_session_id(path: Path | None, trace_cache: TraceCache) -> str:
    if path is None:
        return ""
    stats = trace_cache.stats(path)
    sid = str(stats.get("sessionId") or "").strip()
    if sid:
        return sid
    if path.name.endswith(".jsonl") and path.stem != "stdout":
        candidate = path.stem.strip()
        if re.fullmatch(r"[0-9a-fA-F-]{32,40}", candidate):
            return candidate
    return ""


def _container_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value or "")


def _container_for_side(
    task_name: str,
    side: str,
    state_side: dict[str, Any],
    attempt: dict[str, Any] | None,
    docker: dict[str, Any],
) -> dict[str, Any]:
    by_name = docker.get("byName") or {}
    names: list[str] = []
    if isinstance(attempt, dict):
        container = attempt.get("container")
        if isinstance(container, dict) and container.get("name"):
            names.append(str(container["name"]))
        if attempt.get("container") and isinstance(attempt.get("container"), str):
            names.append(str(attempt["container"]))
    container = state_side.get("container")
    if isinstance(container, dict) and container.get("name"):
        names.append(str(container["name"]))
    elif container:
        names.append(str(container))
    for name in names:
        if name in by_name:
            return copy.deepcopy(by_name[name])
    prefix = f"sologsb-{safe_slug(task_name)}-{side.lower()}-"
    candidates = [
        item for item in (docker.get("items") or [])
        if str(item.get("name") or "").startswith(prefix)
    ]
    if candidates:
        candidates.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
        return copy.deepcopy(candidates[0])
    return {}


def container_group_name(container_name: Any) -> str:
    """Strip the ``sologsb-`` prefix and any ``-candidate-N`` suffix."""
    text = str(container_name or "").strip()
    prefix = "sologsb-"
    if not text.startswith(prefix):
        return ""
    remainder = text[len(prefix):]
    marker = "-candidate-"
    if marker in remainder:
        return remainder.split(marker, 1)[0]
    return remainder


def task_container_names(task_name: str, docker: dict[str, Any]) -> list[str]:
    prefix = f"sologsb-{safe_slug(task_name)}-"
    return [
        str(item.get("name") or "")
        for item in (docker.get("items") or [])
        if str(item.get("state") or "").lower() == "running"
        and str(item.get("name") or "").startswith(prefix)
    ]


def discover_task_roots(
    roots: Iterable[str | Path],
    max_depth: int = 5,
    file_cache: FileCache | None = None,
) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    cache = file_cache or FileCache()
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            continue
        for current, dirnames, _filenames in os.walk(root):
            path = Path(current)
            try:
                relative_depth = len(path.relative_to(root).parts)
            except ValueError:
                relative_depth = 99
            state = cache.json(path / "monitor" / "state.json", {})
            if _is_task_state(state if isinstance(state, dict) else {}):
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(path.resolve())
                dirnames[:] = []
                continue
            if relative_depth >= max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [name for name in dirnames if name not in PRUNE_DIRS and not name.startswith(".")]
    return sorted(found, key=lambda item: item.name.lower())


# --------------------------------------------------------------------------- #
# side / candidate snapshots
# --------------------------------------------------------------------------- #
def _phase_for_side(state_side: dict[str, Any], trace: dict[str, Any]) -> tuple[str, str]:
    raw = str(state_side.get("status") or "")
    if raw == "staged":
        return "staged", "结构校验通过，等待 A/B 双侧审核发布"
    if raw == "clean":
        return "clean", "已发布"
    if raw == "blocked":
        return "blocked", "连续失败，已阻断"
    if raw == "attempt_invalid":
        return "failed", "本轮无效，需要重跑"
    if raw == "invalidated":
        return "failed", "现场已作废，需要重跑"
    if raw == "cancelled":
        return "cancelled", "已有两个候选先完成，此候选已主动停止"
    if raw == "running":
        phase = str(trace.get("lastPhase") or "starting")
        mapping = {
            "starting": "starting",
            "thinking": "thinking",
            "responding": "responding",
            "retrying": "retrying",
            "tool_result": "tool_result",
            "done": "finishing",
            "error": "error",
        }
        if phase.startswith("tool:"):
            return "tool", phase.split(":", 1)[1]
        return mapping.get(phase, "running"), phase
    return "idle", raw or "等待启动"


def _progress_stages(trace: dict[str, Any], state_side: dict[str, Any], phase: str) -> list[dict[str, str]]:
    raw = str(state_side.get("status") or "")
    has_start = bool(trace.get("sessionId") or trace.get("eventCount") or raw in {"running", "staged", "clean"})
    has_thinking = bool(
        trace.get("assistantTurns")
        or trace.get("thinkingTokens")
        or trace.get("toolCalls")
        or raw in {"staged", "clean"}
    )
    has_tools = bool(trace.get("toolCalls") or raw in {"staged", "clean"})
    has_change = bool(trace.get("filesTouched") or trace.get("commands") or raw in {"staged", "clean"})
    has_finish = bool(trace.get("result") or raw in {"staged", "clean"})
    flags = [has_start, has_thinking, has_tools, has_change, has_finish]
    labels = ["启动", "思考", "工具", "改动", "收尾"]
    current = 0
    for index, flag in enumerate(flags):
        if flag:
            current = index
    if phase in {"thinking", "retrying"} and has_start:
        current = 1
    elif phase in {"tool", "tool_result"} and has_tools:
        current = 2
    elif phase in {"finishing", "staged", "clean"}:
        current = 4
    items: list[dict[str, str]] = []
    for index, (label, flag) in enumerate(zip(labels, flags)):
        if flag:
            item_state = "done"
        elif index == current:
            item_state = "current"
        else:
            item_state = "pending"
        items.append({"label": label, "state": item_state})
    return items


def _attempt_history(
    task_root: Path,
    side: str,
    state_side: dict[str, Any],
    trace_cache: TraceCache,
    docker: dict[str, Any],
    *,
    include_events: bool = False,
    event_limit: int = 400,
    file_cache: FileCache | None = None,
    process_table: ProcessTable | None = None,
) -> list[dict[str, Any]]:
    """De-duplicated history of the current runtime and rejected attempts.

    Attempt numbers restart after a side is rerun, so SessionID is the primary
    identity; runtime and rejected copies of the same session are merged.
    """
    cache = file_cache or FileCache()
    records: list[dict[str, Any]] = []
    by_session: dict[str, dict[str, Any]] = {}
    current_attempt = int(state_side.get("attempt") or 0)
    current_status = str(state_side.get("status") or "")
    result_record = cache.json(_runtime_root(task_root, side) / "result.json", {})
    if not isinstance(result_record, dict):
        result_record = {}

    def add_record(record: dict[str, Any]) -> dict[str, Any]:
        sid = str(record.get("sessionId") or "")
        if sid and sid in by_session:
            existing = by_session[sid]
            existing["finishedAt"] = existing.get("finishedAt") or record.get("finishedAt") or ""
            existing["error"] = existing.get("error") or record.get("error") or ""
            existing["rejectionReason"] = existing.get("rejectionReason") or record.get("rejectionReason") or ""
            if existing.get("status") == "running" and record.get("source") == "rejected":
                existing["status"] = "attempt_invalid"
            sources = existing.setdefault("sources", [existing.get("source", "")])
            if record.get("source") and record.get("source") not in sources:
                sources.append(record.get("source"))
            return existing
        record.setdefault("sources", [record.get("source", "")])
        records.append(record)
        if sid:
            by_session[sid] = record
        return record

    for attempt_dir in _attempt_dirs(task_root, side):
        number = int(re.search(r"(\d+)$", attempt_dir.name).group(1)) if re.search(r"(\d+)$", attempt_dir.name) else 0
        meta = cache.json(attempt_dir / "attempt.json", {})
        if not isinstance(meta, dict):
            meta = {}
        stdout_path = attempt_dir / "stdout.jsonl"
        trace_path = stdout_path if stdout_path.is_file() else None
        stats = read_trace_stats(trace_path, event_limit=event_limit) if include_events else trace_cache.stats(trace_path)
        sid = str(meta.get("sessionId") or stats.get("sessionId") or "")
        is_current = bool(number == current_attempt or (sid and sid == str(state_side.get("sessionId") or "")))
        status = current_status if is_current and current_status else "archived"
        error = str(state_side.get("error") or "") if is_current else ""
        started_at = str(meta.get("startedAt") or state_side.get("startedAt") or "") if is_current else str(meta.get("startedAt") or "")
        finished_at = ""
        if is_current and current_status in SIDE_DONE_STATUSES:
            finished_at = str(state_side.get("stagedAt") or state_side.get("publishedAt") or result_record.get("stagedAt") or "")
        record = {
            "key": f"session:{sid}" if sid else f"runtime:{number}:{attempt_dir.stat().st_mtime_ns}",
            "attempt": number,
            "source": "runtime",
            "sources": ["runtime"],
            "isCurrent": is_current,
            "status": status,
            "sessionId": sid,
            "startedAt": started_at,
            "finishedAt": finished_at,
            "durationSeconds": age_seconds(started_at),
            "model": str(meta.get("model") or stats.get("model") or ""),
            "harnessVersion": str(meta.get("harnessVersion") or stats.get("harnessVersion") or ""),
            "contextWindow": meta.get("declaredContextWindow"),
            "container": _container_name(meta.get("container")),
            "error": error,
            "rejectionReason": "",
            "tracePath": str(trace_path or ""),
            "stdoutPath": str(stdout_path) if stdout_path.is_file() else "",
            "stderrPath": str(attempt_dir / "stderr.log") if (attempt_dir / "stderr.log").is_file() else "",
            "rejectedPath": "",
            "validation": meta.get("validation") if isinstance(meta.get("validation"), dict) else {},
            "trace": {
                key: stats.get(key)
                for key in (
                    "eventCount", "assistantTurns", "toolCalls", "toolResults", "thinkingTokens",
                    "apiRetries", "compactions", "commands", "filesTouched", "lastPhase",
                    "lastSummary", "lastText", "lastTool", "lastResult", "lastError", "result",
                    "todos", "todoUpdatedAt",
                )
            },
        }
        if include_events:
            record["events"] = list(stats.get("recent") or [])
        add_record(record)

    rejected_root = _rejected_root(task_root, side)
    if rejected_root.is_dir():
        for rejected_dir in sorted(
            (item for item in rejected_root.glob("attempt-*") if item.is_dir()),
            key=lambda item: int(re.search(r"(\d+)$", item.name).group(1)) if re.search(r"(\d+)$", item.name) else 0,
        ):
            number = int(re.search(r"(\d+)$", rejected_dir.name).group(1)) if re.search(r"(\d+)$", rejected_dir.name) else 0
            rejection = cache.json(rejected_dir / "rejection.json", {})
            if not isinstance(rejection, dict):
                rejection = {}
            candidates = sorted(
                (item for item in rejected_dir.glob("*.jsonl") if item.is_file() and item.name != "stdout.jsonl"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            stdout_path = rejected_dir / "stdout.jsonl"
            trace_path = candidates[0] if candidates else (stdout_path if stdout_path.is_file() else None)
            stats = read_trace_stats(trace_path, event_limit=event_limit) if include_events else trace_cache.stats(trace_path)
            sid = _trace_session_id(trace_path, trace_cache) or str(stats.get("sessionId") or "")
            rejected_at = str(rejection.get("recordedAt") or "")
            reason = str(rejection.get("reason") or "历史尝试未通过单轮校验")
            record = {
                "key": f"rejected:{number}:{sid or rejected_dir.stat().st_mtime_ns}",
                "attempt": number,
                "source": "rejected",
                "sources": ["rejected"],
                "isCurrent": False,
                "status": "attempt_invalid",
                "sessionId": sid,
                "startedAt": "",
                "finishedAt": rejected_at,
                "durationSeconds": None,
                "model": str(stats.get("model") or ""),
                "harnessVersion": str(stats.get("harnessVersion") or ""),
                "contextWindow": None,
                "container": "",
                "error": reason,
                "rejectionReason": reason,
                "tracePath": str(trace_path or ""),
                "stdoutPath": str(stdout_path) if stdout_path.is_file() else "",
                "stderrPath": str(rejected_dir / "stderr.log") if (rejected_dir / "stderr.log").is_file() else "",
                "rejectedPath": str(rejected_dir),
                "validation": {},
                "trace": {
                    key: stats.get(key)
                    for key in (
                        "eventCount", "assistantTurns", "toolCalls", "toolResults", "thinkingTokens",
                        "apiRetries", "compactions", "commands", "filesTouched", "lastPhase",
                        "lastSummary", "lastText", "lastTool", "lastResult", "lastError", "result",
                    )
                },
            }
            if include_events:
                record["events"] = list(stats.get("recent") or [])
            add_record(record)

    def sort_key(item: dict[str, Any]) -> tuple[float, int, str]:
        timestamp = parse_time(item.get("startedAt")) or parse_time(item.get("finishedAt"))
        return (
            timestamp.timestamp() if timestamp else 0.0,
            int(item.get("attempt") or 0),
            str(item.get("key") or ""),
        )

    records.sort(key=sort_key)
    return records


def _side_snapshot(
    task_root: Path,
    task_name: str,
    state: dict[str, Any],
    side: str,
    trace_cache: TraceCache,
    docker: dict[str, Any],
    stale_seconds: float,
    now: float,
    *,
    state_side_override: dict[str, Any] | None = None,
    file_cache: FileCache | None = None,
    process_table: ProcessTable | None = None,
) -> dict[str, Any]:
    from .common import runner_pid_alive

    cache = file_cache or FileCache()
    state_side = copy.deepcopy(
        state_side_override
        if state_side_override is not None
        else (state.get("sides") or {}).get(side) or {}
    )
    candidate_id = str(state_side.get("candidateId") or "")
    runtime_key = candidate_id if _is_candidate_id(candidate_id) else side
    attempts = _attempt_dirs(task_root, runtime_key)
    attempt_dir = attempts[-1] if attempts else None
    attempt = cache.json(attempt_dir / "attempt.json", {}) if attempt_dir else {}
    if not isinstance(attempt, dict):
        attempt = {}
    stdout_path = attempt_dir / "stdout.jsonl" if attempt_dir else None
    trace = trace_cache.stats(stdout_path)
    try:
        trace_mtime = stdout_path.stat().st_mtime if stdout_path and stdout_path.exists() else 0.0
    except OSError:
        trace_mtime = 0.0
    container = _container_for_side(task_name, runtime_key, state_side, attempt, docker)
    runner_live = runner_pid_alive(state_side, task_root, side, process_table)
    container_running = str(container.get("state") or "") == "running"
    active = bool(runner_live or container_running)
    raw_status = str(state_side.get("status") or "")
    silent_seconds = max(0.0, now - trace_mtime) if trace_mtime else None
    if raw_status == "running" and not active and (silent_seconds is None or silent_seconds >= stale_seconds):
        is_stale = True
    else:
        is_stale = False
    phase, phase_detail = _phase_for_side(state_side, trace)
    if is_stale:
        phase, phase_detail = "stale", "执行器与容器均不存活，现场已静默"
    if raw_status not in {"running", "staged", "clean", "blocked", "attempt_invalid", "invalidated"}:
        if not attempt and not container and not trace.get("eventCount"):
            phase, phase_detail = "idle", "尚未启动"
    ever_started = bool(attempt or state_side or container or trace.get("eventCount"))
    can_resume = bool(
        side.upper() in SIDES
        and state.get("status") not in TERMINAL_TASK_STATUSES
        and raw_status not in SIDE_DONE_STATUSES
        and state.get("status") in RUNNABLE_TASK_STATUSES
    )
    started_at = str(attempt.get("startedAt") or state_side.get("startedAt") or "")
    elapsed = age_seconds(started_at, now=now)
    last_activity_at = iso_from_timestamp(trace_mtime) if trace_mtime else ""
    last_error = str(trace.get("lastError") or state_side.get("error") or "")
    if active and not last_error:
        health = "running"
    elif raw_status in SIDE_DONE_STATUSES:
        health = raw_status
    elif is_stale:
        health = "stale"
    elif raw_status in SIDE_FAILED_STATUSES:
        health = "failed"
    elif raw_status == "cancelled":
        health = "idle"
    elif raw_status == "running":
        health = "starting"
    else:
        health = "idle"
    container_snapshot = {
        "id": container.get("id", ""),
        "name": container.get("name", ""),
        "state": container.get("state", "missing"),
        "status": container.get("status", "not found"),
        "image": container.get("image", ""),
        "createdAt": container.get("createdAt", ""),
        "running": container_running,
    }
    if candidate_id:
        workspace_path = state_side.get("workspacePath") or (task_root / "source" / "candidates" / candidate_id)
    else:
        workspace_path = task_root / "source" / side.lower()
    return {
        "side": side,
        "status": raw_status or "idle",
        "phase": phase,
        "phaseDetail": phase_detail,
        "health": health,
        "stale": is_stale,
        "active": active,
        "runnerAlive": runner_live,
        "runnerPid": state_side.get("runPid"),
        "attempt": state_side.get("attempt") or attempt.get("attempt") or len(attempts),
        "attemptCount": len(attempts),
        "sessionId": str(attempt.get("sessionId") or trace.get("sessionId") or state_side.get("sessionId") or ""),
        "model": str(attempt.get("model") or trace.get("model") or ""),
        "harnessVersion": str(attempt.get("harnessVersion") or trace.get("harnessVersion") or ""),
        "startedAt": started_at,
        "elapsedSeconds": elapsed,
        "lastActivityAt": last_activity_at,
        "silentSeconds": silent_seconds,
        "everStarted": ever_started,
        "canResume": can_resume,
        "needsResume": bool(is_stale or raw_status in SIDE_FAILED_STATUSES),
        "container": container_snapshot,
        "trace": {
            key: trace.get(key)
            for key in (
                "eventCount", "assistantTurns", "toolCalls", "toolResults", "thinkingTokens",
                "apiRetries", "compactions", "commands", "filesTouched", "lastPhase",
                "lastSummary", "lastText", "lastTool", "lastResult", "lastError", "result",
                "sessionId", "todos", "todoUpdatedAt",
            )
        },
        "stages": _progress_stages(trace, state_side, phase),
        "events": list((trace.get("recent") or [])[-24:]),
        "history": [
            {
                key: item.get(key)
                for key in (
                    "key", "attempt", "source", "sources", "isCurrent", "status",
                    "sessionId", "startedAt", "finishedAt", "durationSeconds",
                    "container", "error", "rejectionReason", "tracePath", "stdoutPath", "stderrPath",
                )
            }
            | {
                "trace": {
                    key: (item.get("trace") or {}).get(key)
                    for key in ("eventCount", "toolCalls", "apiRetries", "lastSummary", "lastError", "result")
                }
            }
            for item in _attempt_history(
                task_root, side, state_side, trace_cache, docker,
                file_cache=file_cache, process_table=process_table,
            )
        ],
        "workspace": str(workspace_path),
        "candidateId": candidate_id,
        "mappedSide": str(state_side.get("mappedSide") or ""),
        "completionOrder": state_side.get("completionOrder"),
        "attemptDir": str(attempt_dir or ""),
        "stdoutPath": str(stdout_path or ""),
    }


def _candidate_snapshot(
    task_root: Path,
    task_name: str,
    state: dict[str, Any],
    candidate: str,
    trace_cache: TraceCache,
    docker: dict[str, Any],
    stale_seconds: float,
    now: float,
    *,
    file_cache: FileCache | None = None,
    process_table: ProcessTable | None = None,
) -> dict[str, Any]:
    record = copy.deepcopy((state.get("candidates") or {}).get(candidate) or {})
    record["candidateId"] = candidate
    snapshot = _side_snapshot(
        task_root,
        task_name,
        state,
        candidate,
        trace_cache,
        docker,
        stale_seconds,
        now,
        state_side_override=record,
        file_cache=file_cache,
        process_table=process_table,
    )
    snapshot["candidateId"] = candidate
    snapshot["mappedSide"] = str(record.get("mappedSide") or "")
    snapshot["completionOrder"] = record.get("completionOrder")
    return snapshot


def _review_completed(path: Path, file_cache: FileCache | None = None) -> bool:
    review = (file_cache or FileCache()).json(path, {})
    if not isinstance(review, dict):
        return False
    return bool(
        review.get("completed") is True
        and review.get("interrupted") is False
        and isinstance(review.get("unfinished"), list)
        and not review.get("unfinished")
    )


def _workflow_steps(task_root: Path, state: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    sides = task.get("sides") or {}
    statuses = {side: str((sides.get(side) or {}).get("status") or "idle") for side in SIDES}
    candidate_items = task.get("candidates") or []
    candidate_statuses = {
        str(item.get("candidateId") or ""): str(item.get("status") or "idle")
        for item in candidate_items
    }
    mapping = task.get("candidateMapping") if isinstance(task.get("candidateMapping"), dict) else {}
    done_statuses = SIDE_DONE_STATUSES
    both_run_done = all(statuses[side] in done_statuses for side in SIDES)
    candidate_race_done = bool(
        mapping.get("A", {}).get("candidateId")
        and mapping.get("B", {}).get("candidateId")
        and both_run_done
    )
    candidate_race_failed = any(status in {"blocked", "attempt_invalid"} for status in candidate_statuses.values())

    semantic_a = task_root / "monitor" / "semantic" / "a.review.json"
    semantic_b = task_root / "monitor" / "semantic" / "b.review.json"
    semantic_done = _review_completed(semantic_a) and _review_completed(semantic_b)

    publish_done = bool(
        str(state.get("status") or "") in {"ab_clean", "verified", "gsb_ready", "recorded", "complete"}
        and all((sides.get(side) or {}).get("artifactSnapshot") for side in SIDES)
    )
    evidence_path = task_root / "monitor" / "evidence.json"
    audit_path = task_root / "monitor" / "audit.json"
    audit_done = evidence_path.is_file() and audit_path.is_file()
    excel_path = task_root / "workspace" / "评审文件" / "交付表.xlsx"
    guide_path = task_root / "workspace" / "评审文件" / "GSB提交字段说明.md"
    gsb_done = excel_path.is_file() and guide_path.is_file()
    videos = {
        side: sorted((task_root / "workspace" / "视频信息" / side.lower() / "视频").glob("*.mp4"))
        for side in SIDES
    }
    recording_done = all(videos[side] for side in SIDES)
    complete_done = str(state.get("status") or "") == "complete"

    steps: list[dict[str, Any]] = []

    def add(key: str, label: str, done: bool, detail: str, *, blocked: bool = False, current: bool = False) -> None:
        steps.append({
            "key": key,
            "label": label,
            "done": bool(done),
            "blocked": bool(blocked),
            "currentHint": bool(current),
            "detail": detail,
        })

    source_ok = (task_root / "source" / "origin").is_dir() and bool(state.get("source") or state.get("sourcePath"))
    prompt_check = read_json(task_root / "monitor" / "prompt" / "prompt-check-r01.json", {})
    prompt_ok = bool(state.get("promptSha256")) and (not isinstance(prompt_check, dict) or prompt_check.get("ok") is not False)
    add("setup", "接入源码与提示词", source_ok and prompt_ok, "源码和唯一提示词已接入" if source_ok and prompt_ok else "等待源码或提示词校验")
    github_ok = bool(state.get("repoUrl") and state.get("initialSnapshot") and candidate_race_done)
    candidate_detail = "；".join(
        f"{item.get('candidateId')}={item.get('status')}"
        + (f"→{item.get('mappedSide')}" if item.get("mappedSide") else "")
        for item in candidate_items
    ) or "等待启动候选竞速"
    add(
        "candidates",
        "候选并行竞速并映射 A/B",
        candidate_race_done,
        candidate_detail,
        blocked=candidate_race_failed and not candidate_race_done,
        current=not candidate_race_done,
    )
    add(
        "github",
        "上传源码并初始化 GitHub main/A/B",
        github_ok,
        "候选映射后已创建公开仓库和三支" if github_ok else "等待前两名候选完成后建库",
        current=candidate_race_done and not github_ok,
    )
    add("review", "结构与语义完成审核", both_run_done and semantic_done, "A/B 审核文件均完成" if both_run_done and semantic_done else ("等待两侧结构校验与语义审核" if both_run_done else "等待候选映射与 A/B 审核"), current=both_run_done and not semantic_done)
    add("publish", "原子发布 A/B 产物", publish_done, "A/B 产物 commit 已发布" if publish_done else "等待双侧审核通过后发布", current=semantic_done and not publish_done)
    add("audit", "真实构建与运行验证", audit_done, "evidence.json / audit.json 已生成" if audit_done else "等待 verification-plan 与 audit", current=publish_done and not audit_done)
    add("gsb", "生成 GSB 交付表", gsb_done, "21 字段 Excel 与字段说明已生成" if gsb_done else "等待 audit 后生成 GSB 文案与交付表", current=audit_done and not gsb_done)
    add("record", "A/B 真实录屏", recording_done, f"A={len(videos['A'])} 个，B={len(videos['B'])} 个视频" if not recording_done else "A/B 视频均已生成", current=gsb_done and not recording_done)
    add("complete", "最终 status 复核", complete_done, "任务已标记 complete" if complete_done else "等待全部交付物由 status 复核", current=recording_done and not complete_done)

    first_open = next((index for index, step in enumerate(steps) if not step["done"]), None)
    for index, step in enumerate(steps):
        if step["done"]:
            step["state"] = "done"
        elif step["blocked"]:
            step["state"] = "blocked"
        elif index == first_open or step["currentHint"]:
            step["state"] = "current"
        else:
            step["state"] = "pending"
        step.pop("currentHint", None)
    done_count = sum(1 for step in steps if step["done"])
    return {
        "done": done_count,
        "total": len(steps),
        "percent": round(done_count / len(steps) * 100) if steps else 0,
        "steps": steps,
    }


STATUS_LABELS = {
    "prepared": "已接入",
    "prompt_ready": "提示词就绪",
    "repo_ready": "仓库就绪",
    "running": "执行中",
    "candidates_running": "候选竞速中",
    "candidates_ready": "候选竞速结束",
    "a_staged": "A 已校验",
    "b_staged": "B 已校验",
    "semantic_review_required": "待语义审核",
    "ab_clean": "A/B 已发布",
    "verified": "已验证",
    "gsb_ready": "GSB 就绪",
    "recorded": "已录屏",
    "complete": "已完成",
    "attempt_invalid": "本轮无效",
    "blocked": "已阻断",
    "staged": "已校验",
    "clean": "已发布",
    "invalidated": "已作废",
    "cancelled": "未进前二",
    "idle": "未启动",
}

# Coarse buckets used by the status filter row.
STATUS_BUCKETS = {
    "waiting": "等待中",
    "running": "运行中",
    "attention": "需处理",
    "finished": "已完成",
    "failed": "失败",
}


def _bucket_for_task(card: dict[str, Any]) -> str:
    if card.get("needsAttention"):
        return "attention"
    status = str(card.get("status") or "").lower()
    if status in TERMINAL_TASK_STATUSES or status == "complete":
        return "finished"
    if status in {"blocked", "failed", "error", "attempt_invalid", "invalidated"}:
        return "failed"
    if status in {"", "idle", "prepared"} and not card.get("active"):
        return "waiting"
    return "running"


class TaskScanner:
    """Builds slim cards and full task details from the task tree."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        trace_cache: TraceCache | None = None,
        docker_cache: DockerCache | None = None,
        file_cache: FileCache | None = None,
        process_table: ProcessTable | None = None,
    ):
        self.config = config
        monitor_cfg = config.get("monitor") or {}
        self.trace_cache = trace_cache or TraceCache(
            int(monitor_cfg.get("traceMaxBytes") or 16 * 1024 * 1024),
            lightweight=not bool(monitor_cfg.get("parseTraceOnSnapshot", False)),
        )
        self.docker_cache = docker_cache or DockerCache(float(monitor_cfg.get("dockerCacheSeconds") or 2.0))
        self.file_cache = file_cache or FileCache()
        self.process_table = process_table or ProcessTable(float(monitor_cfg.get("dockerCacheSeconds") or 2.0))
        self._stale_seconds = float((monitor_cfg.get("autoResume") or {}).get("staleSeconds") or 420)
        self._session_max_idle = float(monitor_cfg.get("sessionMaxIdleSeconds") or 6 * 3600)

    # -- internals -------------------------------------------------------- #
    @staticmethod
    def _task_name(task_root: Path, state: dict[str, Any]) -> str:
        """The canonical task name.

        The directory name wins over ``state.taskName``.  Containers are named
        from ``safe_slug(task_root.name)`` and the queue reserves directories
        under the same name, so the directory is the identity everything else is
        keyed on.  ``state.taskName`` is written by the skill and can hold a
        placeholder (a real task was observed with the literal value ``"task"``),
        which made the container count silently come out as zero.
        """
        recorded = str(state.get("taskName") or "").strip()
        directory = task_root.name
        if recorded and recorded == directory:
            return directory
        return directory

    def _state(self, task_root: Path) -> dict[str, Any]:
        state = self.file_cache.json(task_root / "monitor" / "state.json", {})
        return state if isinstance(state, dict) else {}

    def _selection(self, task_root: Path) -> dict[str, Any]:
        platform = self.file_cache.json(task_root / "monitor" / "platform-selection.json", {})
        selection = platform.get("selection") if isinstance(platform, dict) else {}
        return selection if isinstance(selection, dict) else {}

    def _prompt_text(self, task_root: Path, prompt_path: str) -> str:
        path = Path(prompt_path) if prompt_path else None
        if path is None or not path.is_file():
            return ""
        return self.file_cache.text(path, limit=20000)

    def _candidate_ids(self, state: dict[str, Any]) -> list[str]:
        configured = [str(item) for item in (state.get("candidateIds") or []) if _is_candidate_id(item)]
        state_candidates = state.get("candidates") if isinstance(state.get("candidates"), dict) else {}
        merged = list(dict.fromkeys(configured + [str(item) for item in state_candidates if _is_candidate_id(item)]))
        merged.sort(key=lambda item: int(item.split("-")[-1]))
        return merged

    # -- public API ------------------------------------------------------- #
    def discover(self, active_roots: list[Path]) -> list[Path]:
        return discover_task_roots(active_roots, file_cache=self.file_cache)

    def cards(self, active_roots: list[Path], dismissed: set[str] | None = None, now: float | None = None) -> list[dict[str, Any]]:
        now = now if now is not None else time.time()
        docker = self.docker_cache.get()
        cards: list[dict[str, Any]] = []
        for task_root in self.discover(active_roots):
            card = self.card(task_root, docker=docker, now=now)
            if dismissed and card["id"] in dismissed:
                continue
            cards.append(card)
        cards.sort(key=self._sort_key, reverse=True)
        return cards

    def card(self, task_root: Path, *, docker: dict[str, Any] | None = None, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        docker = docker if docker is not None else self.docker_cache.get()
        state = self._state(task_root)
        selection = self._selection(task_root)
        name = self._task_name(task_root, state)
        sides = state.get("sides") or {}
        candidates = state.get("candidates") or {}

        badges: list[dict[str, Any]] = []
        silent: list[float] = []
        active = False
        needs_attention = False
        candidate_ids = self._candidate_ids(state)
        if candidate_ids:
            # A/B are markers on candidates, not separate things to show.  With
            # two candidates they simply *are* A and B, so the badge carries the
            # mapped letter; above two the number identifies the candidate and
            # only the survivors also show a letter.
            for candidate_id in candidate_ids:
                record = candidates.get(candidate_id) or {}
                status = str(record.get("status") or "idle")
                mapped = str(record.get("mappedSide") or "")
                if status == "running":
                    active = True
                if status in SIDE_FAILED_STATUSES:
                    needs_attention = True
                label = mapped or ("" if status == "cancelled" else candidate_id.split("-")[-1])
                badges.append({
                    "key": candidate_id,
                    "label": label,
                    "state": (
                        "running" if status == "running"
                        else "done" if status in SIDE_DONE_STATUSES
                        else "failed" if status in SIDE_FAILED_STATUSES
                        else "cancelled" if status == "cancelled"
                        else "idle"
                    ),
                    "title": f"{candidate_id}{mapped and f' → {mapped}' or ''} · {STATUS_LABELS.get(status, status)}",
                })
        else:
            # Pre-race or legacy tasks: fall back to the two logical sides.
            for side in SIDES:
                record = sides.get(side) or {}
                status = str(record.get("status") or "idle")
                if status == "running":
                    active = True
                if status in SIDE_FAILED_STATUSES:
                    needs_attention = True
                badges.append({
                    "key": side,
                    "label": side,
                    "state": (
                        "running" if status == "running"
                        else "done" if status in SIDE_DONE_STATUSES
                        else "failed" if status in SIDE_FAILED_STATUSES
                        else "idle"
                    ),
                    "title": f"{side} · {STATUS_LABELS.get(status, status)}",
                })

        active_containers = len(task_container_names(name, docker))
        status = str(state.get("status") or "unknown")
        card = {
            "id": short_hash(str(task_root)),
            "name": name,
            "status": status,
            "statusLabel": STATUS_LABELS.get(status, status),
            "phase": self._phase_label(state, candidate_ids),
            "projectCode": str(selection.get("projectCode") or ""),
            "projectName": str(selection.get("projectName") or ""),
            "taskNo": str(selection.get("taskNo") or ""),
            "taskType": str(state.get("taskType") or selection.get("taskType") or ""),
            "difficulty": str(state.get("difficulty") or ""),
            "createdAt": str(state.get("createdAt") or ""),
            "updatedAt": str(state.get("updatedAt") or ""),
            "repoName": str(state.get("repoName") or ""),
            "active": active,
            "needsAttention": needs_attention,
            "activeContainers": active_containers,
            "badges": badges,
            "candidateCount": len(candidate_ids),
            "mappedSides": {
                side: str((sides.get(side) or {}).get("candidateId") or "")
                for side in SIDES
                if (sides.get(side) or {}).get("candidateId")
            },
        }
        card["bucket"] = _bucket_for_task(card)
        card["silentSeconds"] = min(silent) if silent else None
        return card

    def _phase_label(self, state: dict[str, Any], candidate_ids: list[str]) -> str:
        status = str(state.get("status") or "")
        if status in STATUS_LABELS:
            return STATUS_LABELS[status]
        sides = state.get("sides") or {}
        running = [side for side in SIDES if str((sides.get(side) or {}).get("status") or "") == "running"]
        if running:
            return f"{'/'.join(running)} 执行中"
        if candidate_ids:
            return "候选竞速"
        return "未启动"

    def _sort_key(self, card: dict[str, Any]) -> float:
        values = [card.get("updatedAt"), card.get("createdAt")]
        stamps = [parse_time(value).timestamp() for value in values if parse_time(value)]
        return max(stamps) if stamps else 0.0

    def detail(self, task_id: str, active_roots: list[Path], dismissed: set[str] | None = None) -> dict[str, Any] | None:
        for task_root in self.discover(active_roots):
            if short_hash(str(task_root)) != task_id:
                continue
            if dismissed and task_id in dismissed:
                return None
            return self.detail_for_root(task_root)
        return None

    def detail_for_root(self, task_root: Path) -> dict[str, Any]:
        now = time.time()
        docker = self.docker_cache.get()
        state = self._state(task_root)
        selection = self._selection(task_root)
        name = self._task_name(task_root, state)
        card = self.card(task_root, docker=docker, now=now)

        sides: dict[str, Any] = {}
        for side in SIDES:
            sides[side] = _side_snapshot(
                task_root, name, state, side,
                self.trace_cache, docker, self._stale_seconds, now,
                file_cache=self.file_cache, process_table=self.process_table,
            )
        candidates = [
            _candidate_snapshot(
                task_root, name, state, candidate_id,
                self.trace_cache, docker, self._stale_seconds, now,
                file_cache=self.file_cache, process_table=self.process_table,
            )
            for candidate_id in self._candidate_ids(state)
        ]
        local_head: dict[str, str] = {}
        for side in SIDES:
            workspace = str((state.get("sides") or {}).get(side, {}).get("workspacePath") or "")
            local_head[side] = _git_head(Path(workspace)) if workspace else _git_head(task_root / "source" / side.lower())

        # One entry per candidate, with its A/B marker folded in.  A and B are
        # labels on a candidate rather than separate entities, so they must not
        # be rendered as their own panel — with two candidates they *are* A and
        # B, and only above two does the race decide which two survive.
        units: list[dict[str, Any]] = []
        if candidates:
            for candidate in candidates:
                mapped = str(candidate.get("mappedSide") or "")
                units.append({
                    **candidate,
                    "key": str(candidate.get("candidateId") or ""),
                    # Unmapped candidates show their number; once the race maps
                    # them the letter replaces it, exactly like the card badge.
                    "label": mapped or str(candidate.get("candidateId") or "").split("-")[-1],
                    "kind": "candidate",
                })
        else:
            # Pre-race or legacy tasks with no candidates.  Only show a side that
            # has actually done something — an idle placeholder card is noise.
            def started(record: dict[str, Any]) -> bool:
                status = str(record.get("status") or "")
                container = record.get("container") or {}
                return bool(
                    (status and status != "idle")
                    or container.get("name")
                    or record.get("sessionId")
                    or int(record.get("attemptCount") or 0)
                )

            chosen = [side for side in SIDES if started(sides.get(side) or {})]
            for side in (chosen or list(SIDES)):
                record = sides.get(side) or {}
                units.append({
                    **record,
                    "key": side,
                    "label": side,
                    "kind": "side",
                    "mappedSide": side,
                })

        detail = {
            **card,
            "taskRoot": str(task_root),
            "repoUrl": str(state.get("repoUrl") or ""),
            "artifacts": {
                "semanticA": (task_root / "monitor" / "semantic" / "a.review.json").is_file(),
                "semanticB": (task_root / "monitor" / "semantic" / "b.review.json").is_file(),
                "evidence": (task_root / "monitor" / "evidence.json").is_file(),
                "audit": (task_root / "monitor" / "audit.json").is_file(),
                "gsb": (task_root / "workspace" / "评审文件" / "交付表.xlsx").is_file(),
            },
            "promptPath": str(state.get("promptPath") or ""),
            "promptSha256": str(state.get("promptSha256") or ""),
            "promptText": self._prompt_text(task_root, str(state.get("promptPath") or "")),
            "initialSnapshot": str(state.get("initialSnapshot") or ""),
            "variantName": str(selection.get("variantName") or ""),
            "sides": sides,
            "candidates": candidates,
            "units": units,
            "candidateMapping": copy.deepcopy(state.get("candidateMapping") or {}),
            "localHead": local_head,
            "workflow": _workflow_steps(task_root, state, {
                "sides": sides,
                "candidates": candidates,
                "candidateMapping": state.get("candidateMapping") or {},
            }),
            "appSessions": [],
            "queue": {},
        }
        detail["needsAttention"] = any(
            item.get("needsResume") or item.get("apiRetries")
            for item in [*sides.values(), *candidates]
        )
        return detail

    def summary(self, cards: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {key: 0 for key in STATUS_BUCKETS}
        for card in cards:
            counts[card.get("bucket") or "waiting"] += 1
        return {
            "tasks": len(cards),
            "active": counts["running"],
            "attention": counts["attention"],
            "finished": counts["finished"],
            "failed": counts["failed"],
            "waiting": counts["waiting"],
            "containers": sum(int(card.get("activeContainers") or 0) for card in cards),
        }

    def stdout_path(self, task_id: str, side: str, active_roots: list[Path]) -> Path | None:
        """Locate the newest stdout.jsonl for a side or candidate of a task."""
        task = self.detail(task_id, active_roots)
        if not task:
            return None
        key = str(side or "").strip()
        if key.upper() in SIDES:
            record = (task.get("sides") or {}).get(key.upper()) or {}
        else:
            record = next(
                (item for item in (task.get("candidates") or []) if str(item.get("candidateId") or "").lower() == key.lower()),
                {},
            )
        value = str(record.get("stdoutPath") or "")
        path = Path(value) if value else None
        return path if path is not None and path.is_file() else None

    def format_stdout_lines(self, path: Path, lines: int) -> list[str]:
        """Render the tail of a stream-json trajectory as readable log lines."""
        result: list[str] = []
        for raw in tail_lines(path, max(1, min(int(lines), 2000))):
            try:
                event = json.loads(raw)
            except ValueError:
                if raw.strip():
                    result.append(redact_text(raw, 1500))
                continue
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "")
            if event_type == "assistant":
                details = []
                for block in _content_blocks(event):
                    block_type = str(block.get("type") or "")
                    if block_type == "text":
                        details.append(redact_text(block.get("text") or "", 700))
                    elif block_type == "tool_use":
                        details.append(f"tool={block.get('name')} {_tool_detail(block)}")
                    elif block_type == "thinking":
                        details.append("thinking")
                result.append("assistant | " + " | ".join(item for item in details if item))
            elif event_type == "user":
                texts = [
                    redact_text(block.get("content") or "", 700)
                    for block in _content_blocks(event)
                    if block.get("type") == "tool_result"
                ]
                if texts:
                    result.append("tool_result | " + " | ".join(texts))
            else:
                result.append(redact_text(raw, 1200))
        return result
