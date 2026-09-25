"""How many times each project has been used, counted from the QC platform (SOLO2).

Solo Manager's quota ``used`` no longer tracks real use, so the count comes
from our own GSB submissions: every submission whose repo name carries the
project code (``kekelele996/gb-532-qq2k`` -> ``gb-532``) is one use of that
project for its ``question_type``.  Discarded submissions do not count.

The limit (``automation.projectUsage.limit``, default 10) applies to the
project's total by default, or to each task type with ``scope: perType``.
Items still pending or running in our queue count as uses too, since they
will be submitted.  A refresh thread keeps the counts current; callers only
read the cache, so nothing on the request path waits on SOLO2.

A failed fetch is retried once right away.  When both attempts fail the
model behind the containers is often down too, so the first failed refresh
of a streak asks the LLM guard for one probe (``on_failed``): the guard then
pauses and resumes the queue by its own settings.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .common import keychain_read, utc_now

SKILL_CONFIG_PATH = Path("~/.codex/sologsb/config.json").expanduser()
DEFAULT_API_BASE = "https://solo2.jzxhnh.com/api/v1"
PAGE_SIZE = 100
MAX_PAGES = 100
NOT_COUNTED_STAGES = {"DISCARDED"}
SCOPES = {"total", "perType"}
FETCH_ATTEMPTS = 2
REPO_CODE = re.compile(r"^([a-z]+)-?(\d+)(?:-|$)", re.IGNORECASE)

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "limit": 10,
    "scope": "total",
    "refreshSeconds": 120,
}


def project_usage_settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = (config.get("automation") or {}).get("projectUsage") or {}
    try:
        limit = int(cfg.get("limit", DEFAULTS["limit"]))
    except (TypeError, ValueError):
        limit = DEFAULTS["limit"]
    try:
        refresh = int(cfg.get("refreshSeconds", DEFAULTS["refreshSeconds"]))
    except (TypeError, ValueError):
        refresh = DEFAULTS["refreshSeconds"]
    scope = str(cfg.get("scope") or DEFAULTS["scope"])
    return {
        "enabled": bool(cfg.get("enabled", DEFAULTS["enabled"])),
        "limit": max(1, min(1000, limit)),
        "scope": scope if scope in SCOPES else DEFAULTS["scope"],
        "refreshSeconds": max(30, min(3600, refresh)),
    }


def repo_project_code(repo_id: Any) -> str:
    """``owner/cy402-hearing-x`` -> ``cy-402``; ``""`` when the name has no code."""
    name = str(repo_id or "").rsplit("/", 1)[-1]
    match = REPO_CODE.match(name)
    return f"{match.group(1).lower()}-{int(match.group(2))}" if match else ""


def count_submissions(items: list[dict[str, Any]]) -> dict[str, Any]:
    codes: dict[str, dict[str, Any]] = {}
    unparsed: list[str] = []
    counted = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("stage") or item.get("status") or "") in NOT_COUNTED_STAGES:
            continue
        code = repo_project_code(item.get("repo_id"))
        if not code:
            unparsed.append(str(item.get("repo_id") or item.get("id") or ""))
            continue
        task_type = str(item.get("question_type") or "未知")
        entry = codes.setdefault(code, {"total": 0, "byType": {}})
        entry["total"] += 1
        entry["byType"][task_type] = entry["byType"].get(task_type, 0) + 1
        counted += 1
    return {"codes": codes, "counted": counted, "unparsed": unparsed}


def _solo2_credentials(config: dict[str, Any], skill_config_path: Path = SKILL_CONFIG_PATH) -> list[tuple[str, str, str]]:
    """(source, cookie, csrf) candidates: the skill's config first, then the keychain."""
    found: list[tuple[str, str, str]] = []
    try:
        data = json.loads(skill_config_path.read_text(encoding="utf-8"))
        solo2 = data.get("solo2") if isinstance(data, dict) else None
    except (OSError, ValueError):
        solo2 = None
    if isinstance(solo2, dict) and str(solo2.get("cookie") or "").strip():
        found.append(("skill-config", str(solo2["cookie"]).strip(), str(solo2.get("csrf") or "").strip()))
    cookie = keychain_read("solo2-jzxhnh-cookie")
    if cookie:
        found.append(("keychain", cookie, keychain_read("solo2-jzxhnh-csrf")))
    return found


def fetch_submissions(config: dict[str, Any], timeout: float = 20) -> dict[str, Any]:
    """Every submission of the logged-in SOLO2 user, all pages."""
    base = str((config.get("solo2") or {}).get("apiBaseUrl") or DEFAULT_API_BASE).rstrip("/")
    credentials = _solo2_credentials(config)
    if not credentials:
        raise RuntimeError("读不到 SOLO2 登录态（技能配置 solo2.cookie / 钥匙串 solo2-jzxhnh-cookie）")
    errors: list[str] = []
    for source, cookie, csrf in credentials:
        headers = {"Cookie": cookie, "x-csrf-token": csrf, "Accept": "application/json"}
        items: list[dict[str, Any]] = []
        try:
            page = 1
            while page <= MAX_PAGES:
                request = urllib.request.Request(
                    f"{base}/gsb/submissions?page={page}&page_size={PAGE_SIZE}", headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                batch = payload.get("items") if isinstance(payload, dict) else None
                if not isinstance(batch, list):
                    raise ValueError("返回格式不对：缺少 items")
                items.extend(item for item in batch if isinstance(item, dict))
                meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                if not batch or page >= int(meta.get("total_pages") or 1):
                    break
                page += 1
        except urllib.error.HTTPError as exc:
            errors.append(f"{source}: HTTP {exc.code}")
            continue
        except (OSError, ValueError) as exc:
            errors.append(f"{source}: {exc}")
            continue
        return {"items": items, "source": source}
    raise RuntimeError("拉取质检平台提交失败：" + "；".join(errors))


class ProjectUsage:
    """Cached per-project use counts plus the enqueue gate built on them."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        emit: Callable[..., None] | None = None,
        fetch: Callable[[dict[str, Any]], dict[str, Any]] = fetch_submissions,
        clock: Callable[[], float] = time.time,
        retry_delay: float = 3.0,
    ):
        self.config = config
        self._emit = emit or (lambda *args, **kwargs: None)
        self._fetch = fetch
        self._clock = clock
        self._retry_delay = retry_delay
        # Called with the error when a refresh first fails (after the retry).
        self.on_failed: Callable[[str], Any] | None = None
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.health: Any = None
        # Codes currently pending or running in our queue -> {taskType: n}.
        self.inflight: Callable[[], dict[str, dict[str, int]]] = lambda: {}
        self._codes: dict[str, dict[str, Any]] = {}
        self._fetched_at = 0.0
        self._status: dict[str, Any] = {"fetchedAt": "", "error": "", "counted": 0, "unparsed": [], "source": ""}

    def settings(self) -> dict[str, Any]:
        return project_usage_settings(self.config)

    # -- refresh ------------------------------------------------------------ #
    def refresh(self) -> dict[str, Any]:
        with self._refresh_lock:
            fetched = None
            error = ""
            for attempt in range(FETCH_ATTEMPTS):
                if attempt and self._stop.wait(self._retry_delay):
                    break
                try:
                    fetched = self._fetch(self.config)
                    break
                except Exception as exc:
                    error = str(exc)
            if fetched is None:
                with self._lock:
                    first = not self._status.get("error")
                    self._status = {**self._status, "error": error[:300], "failedAt": utc_now()}
                    self._fetched_at = self._clock()
                if first:
                    self._emit("project_usage.failed", level="warning",
                               detail=f"质检平台使用次数连续 {FETCH_ATTEMPTS} 次拉取失败，沿用上次数据：{error}")
                    if self.on_failed is not None:
                        try:
                            self.on_failed(error)
                        except Exception as exc:  # the probe request must not break the refresh
                            self._emit("project_usage.on_failed_error", level="error", detail=str(exc))
                return self.status()
            counted = count_submissions(fetched.get("items") or [])
            with self._lock:
                recovered = bool(self._status.get("error"))
                self._codes = counted["codes"]
                self._fetched_at = self._clock()
                self._status = {
                    "fetchedAt": utc_now(),
                    "error": "",
                    "counted": counted["counted"],
                    "submissions": len(fetched.get("items") or []),
                    "unparsed": counted["unparsed"][:20],
                    "source": str(fetched.get("source") or ""),
                }
            if recovered:
                self._emit("project_usage.recovered", detail="质检平台使用次数恢复拉取")
            return self.status()

    def due(self) -> bool:
        return self.settings()["enabled"] and self._clock() - self._fetched_at >= self.settings()["refreshSeconds"]

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._run, name="project-usage", daemon=True)
        thread.start()
        self._thread = thread
        return thread

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while True:
            if self.health is not None:
                self.health.begin("project-usage")
            error = ""
            try:
                if self.due():
                    self.refresh()
            except Exception as exc:
                error = str(exc)
            finally:
                if self.health is not None:
                    self.health.end("project-usage", error)
            if self._stop.wait(5):
                return

    # -- reading ------------------------------------------------------------- #
    def status(self) -> dict[str, Any]:
        with self._lock:
            status = copy.deepcopy(self._status)
            loaded = bool(self._codes) or bool(status.get("fetchedAt"))
            by_type: dict[str, int] = {}
            for entry in self._codes.values():
                for task_type, n in entry["byType"].items():
                    by_type[task_type] = by_type.get(task_type, 0) + n
        return {**self.settings(), **status, "loaded": loaded, "byType": by_type}

    def usage(self, code: str, task_type: str = "") -> dict[str, Any]:
        """Uses of one project: submitted (QC) + queued, against the limit."""
        key = str(code or "").strip().casefold()
        settings = self.settings()
        with self._lock:
            entry = copy.deepcopy(self._codes.get(key) or {"total": 0, "byType": {}})
            loaded = bool(self._status.get("fetchedAt"))
        try:
            queued_types = dict(self.inflight().get(key) or {})
        except Exception:
            queued_types = {}
        if settings["scope"] == "perType" and task_type:
            used = int(entry["byType"].get(task_type, 0))
            queued = int(queued_types.get(task_type, 0))
        else:
            used = int(entry["total"])
            queued = sum(int(value) for value in queued_types.values())
        limit = settings["limit"]
        return {
            "code": key,
            "used": used,
            "queued": queued,
            "limit": limit,
            "remaining": max(0, limit - used - queued),
            "total": int(entry["total"]),
            "byType": entry["byType"],
            "queuedByType": queued_types,
            "scope": settings["scope"],
            "taskType": task_type,
            "loaded": loaded,
        }

    def all_usage(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            codes = list(self._codes)
        return {code: self.usage(code) for code in codes}

    def blocked_reason(self, code: str, task_type: str = "", *, include_queued: bool = True) -> str:
        """Why ``code`` may not take another run, or ``""`` when it may.

        Without a first successful fetch nothing is blocked: an expired SOLO2
        login must not stop the whole queue (the page shows the error).
        """
        settings = self.settings()
        if not settings["enabled"]:
            return ""
        usage = self.usage(code, task_type)
        if not usage["loaded"]:
            return ""
        taken = usage["used"] + (usage["queued"] if include_queued else 0)
        if taken < usage["limit"]:
            return ""
        scope = f"{task_type} " if settings["scope"] == "perType" and task_type else ""
        queued = f"，队列中 {usage['queued']}" if include_queued and usage["queued"] else ""
        return f"{scope}质检已用 {usage['used']}{queued}，达到上限 {usage['limit']}"
