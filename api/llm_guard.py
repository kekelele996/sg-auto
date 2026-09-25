"""Pause the queue while the LLM the containers call is down, resume when it is back.

The candidates inside the containers talk to ``anthropicBaseUrl``; when that
endpoint drops, every task launched meanwhile burns quota and fails.  The
guard probes the endpoint with a one-token request.  After ``failThreshold``
failures in a row it pauses the queue — after the first failure it retries
within ``retrySeconds`` instead of waiting a full interval — and marks the pause as its own
(``automation.llmGuard.pausedByGuard``), then probes every ``pausedProbeSeconds``
and resumes only a pause it made itself: a manual pause is never lifted.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .common import keychain_read, utc_now

SKILL_CONFIG_PATH = Path("~/.codex/sologsb/config.json").expanduser()
KEY_KEYCHAIN_SERVICE = "benzhi-claude-code-gaobo-pi-a453493f"
DEFAULT_MODEL = "auto_model/urm"

PROBE_RANGE = (60, 86400)
PAUSED_PROBE_RANGE = (60, 86400)
RETRY_RANGE = (10, 3600)

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # Every probe is a real request on the key and takes one of its
    # concurrent slots, so the healthy-state probe runs only every 30 minutes.
    "probeSeconds": 1800,
    "pausedProbeSeconds": 300,
    # A failure below the threshold is confirmed or cleared quickly instead of
    # waiting a whole probeSeconds, so an outage pauses within a minute.
    "retrySeconds": 30,
    "failThreshold": 2,
    "timeoutSeconds": 30,
}


def llm_guard_settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = (config.get("automation") or {}).get("llmGuard") or {}

    def number(key: str, low: int, high: int) -> int:
        try:
            value = int(cfg.get(key, DEFAULTS[key]))
        except (TypeError, ValueError):
            value = DEFAULTS[key]
        return max(low, min(high, value))

    return {
        "enabled": bool(cfg.get("enabled", DEFAULTS["enabled"])),
        "probeSeconds": number("probeSeconds", PROBE_RANGE[0], PROBE_RANGE[1]),
        "pausedProbeSeconds": number("pausedProbeSeconds", PAUSED_PROBE_RANGE[0], PAUSED_PROBE_RANGE[1]),
        "retrySeconds": number("retrySeconds", RETRY_RANGE[0], RETRY_RANGE[1]),
        "failThreshold": number("failThreshold", 1, 20),
        "timeoutSeconds": number("timeoutSeconds", 5, 120),
        "pausedByGuard": bool(cfg.get("pausedByGuard")),
    }


def _skill_claude_config(path: Path = SKILL_CONFIG_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    claude = data.get("claude") if isinstance(data, dict) else None
    return claude if isinstance(claude, dict) else {}


def resolve_endpoint(config: dict[str, Any], skill_config_path: Path = SKILL_CONFIG_PATH) -> dict[str, str]:
    """The base URL, key and model the containers are started with."""
    claude = _skill_claude_config(skill_config_path)
    base_url = (
        str((config.get("automation") or {}).get("anthropicBaseUrl") or "")
        or os.environ.get("SOLOSB_ANTHROPIC_BASE_URL", "")
        or str(claude.get("baseUrl") or "")
    ).rstrip("/")
    key = (
        os.environ.get("SOLOSB_CLAUDE_KEY", "")
        or str(claude.get("apiKey") or "")
        or keychain_read(KEY_KEYCHAIN_SERVICE)
    ).strip()
    model = os.environ.get("SOLOSB_MODEL", "") or str(claude.get("model") or "") or DEFAULT_MODEL
    return {"baseUrl": base_url, "key": key, "model": model}


PROBE_PROMPT = "只回复两个字母：OK"
PROBE_MAX_TOKENS = 64


def _reply_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def probe_llm(endpoint: dict[str, str], timeout: float = 30) -> dict[str, Any]:
    """Ask the model for a short reply; only a non-empty text answer counts.

    An HTTP 200 alone is not enough: a gateway can answer with an error body
    or an empty message while the model behind it is down.  Never returns the
    key.
    """
    started = time.time()
    result: dict[str, Any] = {
        "ok": False,
        "at": utc_now(),
        "baseUrl": endpoint.get("baseUrl", ""),
        "model": endpoint.get("model", ""),
        "status": None,
        "latencyMs": None,
        "reply": "",
        "replyModel": "",
        "stopReason": "",
        "outputTokens": None,
        "error": "",
    }
    if not endpoint.get("baseUrl"):
        result["error"] = "未配置 anthropicBaseUrl"
        return result
    if not endpoint.get("key"):
        result["error"] = "未找到 Claude Key（SOLOSB_CLAUDE_KEY / 技能配置 / 钥匙串）"
        return result
    body = json.dumps({
        "model": endpoint["model"],
        "max_tokens": PROBE_MAX_TOKENS,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint['baseUrl']}/v1/messages",
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": endpoint["key"],
            "authorization": f"Bearer {endpoint['key']}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result["status"] = response.status
            raw = response.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw or "{}")
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            result["error"] = f"返回的不是 JSON：{raw[:120]}"
        elif payload.get("type") == "error":
            error = payload.get("error") or {}
            result["error"] = str(error.get("message") if isinstance(error, dict) else error)[:300]
        else:
            text = _reply_text(payload)
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            result["reply"] = text[:200]
            result["replyModel"] = str(payload.get("model") or "")
            result["stopReason"] = str(payload.get("stop_reason") or "")
            result["outputTokens"] = usage.get("output_tokens")
            if text:
                result["ok"] = True
            else:
                result["error"] = f"模型没有返回文本（stop_reason={result['stopReason'] or '空'}）"
    except urllib.error.HTTPError as exc:
        result["status"] = exc.code
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:
            detail = ""
        result["error"] = f"HTTP {exc.code} {detail}".strip()[:300]
    except (OSError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:300]
    result["latencyMs"] = int((time.time() - started) * 1000)
    key = endpoint.get("key") or ""
    if key and key in result["error"]:
        result["error"] = result["error"].replace(key, "***")
    return result


def _iso(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


class LlmGuard:
    """Background probe loop; pauses and resumes ``automation.paused``."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        persist: Callable[[], None],
        emit: Callable[..., None],
        probe: Callable[[dict[str, str], float], dict[str, Any]] = probe_llm,
        resolve: Callable[[dict[str, Any]], dict[str, str]] = resolve_endpoint,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config
        self._persist = persist
        self._emit = emit
        self._probe = probe
        self._resolve = resolve
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.health: Any = None
        # Called with the outage's start (epoch seconds) after an auto resume,
        # so the service can clean up what failed while the model was down.
        self.on_resumed: Callable[[float], Any] | None = None
        self.last_ok_at = 0.0
        self.outage_since = 0.0
        self.failures = 0
        self.last_probe: dict[str, Any] = {}
        self.last_probe_at = 0.0
        self.last_change = ""

    # -- config ------------------------------------------------------------ #
    def _cfg(self) -> dict[str, Any]:
        return self.config.setdefault("automation", {}).setdefault("llmGuard", {})

    def settings(self) -> dict[str, Any]:
        return llm_guard_settings(self.config)

    def _interval(self, settings: dict[str, Any]) -> int:
        if settings["pausedByGuard"]:
            return settings["pausedProbeSeconds"]
        if 0 < self.failures < settings["failThreshold"]:
            return min(settings["retrySeconds"], settings["probeSeconds"])
        return settings["probeSeconds"]

    def status(self) -> dict[str, Any]:
        settings = self.settings()
        interval = self._interval(settings)
        next_at = self.last_probe_at + interval if self.last_probe_at else 0.0
        return {
            **settings,
            "failures": self.failures,
            "lastProbe": dict(self.last_probe),
            "lastChange": self.last_change,
            # An absolute time, not a countdown: the snapshot hub pushes the
            # page whenever the queue payload changes, and a ticking number
            # made it re-render every build.
            "nextProbeAt": _iso(next_at) if settings["enabled"] and next_at else "",
        }

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            cfg = self._cfg()
            cfg["enabled"] = bool(enabled)
            self.failures = 0
            # Turning the guard off hands its pause back to the operator.
            if not enabled:
                cfg["pausedByGuard"] = False
                cfg.pop("outageSince", None)
            self.last_probe_at = 0.0
            self._persist()

    def set_intervals(self, probe_seconds: Any = None, paused_probe_seconds: Any = None) -> dict[str, Any]:
        """Change how often the key is probed; ``None`` keeps a value as it is."""
        wanted: dict[str, int] = {}
        for key, value, (low, high) in (
            ("probeSeconds", probe_seconds, PROBE_RANGE),
            ("pausedProbeSeconds", paused_probe_seconds, PAUSED_PROBE_RANGE),
        ):
            if value is None:
                continue
            try:
                number = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} 必须是整数秒") from None
            if not low <= number <= high:
                raise ValueError(f"{key} 必须在 {low} 到 {high} 秒之间")
            wanted[key] = number
        with self._lock:
            self._cfg().update(wanted)
            self._persist()
        return self.settings()

    def note_manual_pause(self) -> None:
        """Any manual start/pause takes ownership of ``paused`` away from the guard."""
        with self._lock:
            if self._cfg().get("pausedByGuard"):
                self._cfg()["pausedByGuard"] = False
                self._cfg().pop("outageSince", None)
            self.failures = 0

    # -- probing ------------------------------------------------------------ #
    def test(self) -> dict[str, Any]:
        """A one-off probe for the page; also feeds the guard's state."""
        settings = self.settings()
        result = self._probe(self._resolve(self.config), settings["timeoutSeconds"])
        self._apply(result, settings)
        return result

    def due(self) -> bool:
        settings = self.settings()
        if not settings["enabled"]:
            return False
        return self._clock() - self.last_probe_at >= self._interval(settings)

    def step(self) -> dict[str, Any] | None:
        """Probe if due; pause or resume the queue on a state change."""
        if not self.due():
            return None
        settings = self.settings()
        automation = self.config.get("automation") or {}
        # Nothing is launching and the pause is not ours: no need to spend a call.
        if bool(automation.get("paused", True)) and not settings["pausedByGuard"]:
            self.last_probe_at = self._clock()
            return None
        result = self._probe(self._resolve(self.config), settings["timeoutSeconds"])
        self._apply(result, settings)
        return result

    def _apply(self, result: dict[str, Any], settings: dict[str, Any]) -> None:
        resumed_since = self._apply_locked(result, settings)
        if resumed_since is not None and self.on_resumed is not None:
            try:
                self.on_resumed(resumed_since)
            except Exception as exc:  # recovery must never break the probe loop
                self._emit("llm_guard.recover_failed", level="error", detail=str(exc))

    def _apply_locked(self, result: dict[str, Any], settings: dict[str, Any]) -> float | None:
        with self._lock:
            self.last_probe = dict(result)
            self.last_probe_at = self._clock()
            automation = self.config.setdefault("automation", {})
            cfg = self._cfg()
            if result.get("ok"):
                self.failures = 0
                self.last_ok_at = self.last_probe_at
                if settings["enabled"] and cfg.get("pausedByGuard"):
                    # Persisted with the pause so a restart mid-outage keeps it.
                    since = self.outage_since or float(cfg.pop("outageSince", 0) or 0)
                    cfg.pop("outageSince", None)
                    cfg["pausedByGuard"] = False
                    automation["paused"] = False
                    self.last_change = f"{utc_now()} 大模型恢复，队列自动启动"
                    self._persist()
                    self._emit("llm_guard.resumed", detail=f"大模型可用（{result.get('latencyMs')}ms），队列自动启动")
                    # The model went down some time after the last good probe;
                    # without one (e.g. after a restart) take one full interval.
                    self.outage_since = 0.0
                    return since or (self.last_probe_at - settings["probeSeconds"])
                return None
            self.failures += 1
            if self.failures == 1:
                self.outage_since = self.last_ok_at or (self.last_probe_at - settings["probeSeconds"])
            self._emit("llm_guard.probe_failed", level="warning",
                       detail=f"第 {self.failures} 次失败：{result.get('error')}")
            if (
                settings["enabled"]
                and self.failures >= settings["failThreshold"]
                and not bool(automation.get("paused", True))
            ):
                automation["paused"] = True
                cfg["pausedByGuard"] = True
                cfg["outageSince"] = round(self.outage_since, 3)
                self.last_change = f"{utc_now()} 大模型不可用，队列自动暂停"
                self._persist()
                self._emit("llm_guard.paused", level="warning",
                           detail=f"大模型连续 {self.failures} 次不可用，队列自动暂停，"
                                  f"每 {settings['pausedProbeSeconds']} 秒复测：{result.get('error')}")
            return None

    # -- thread ------------------------------------------------------------- #
    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._run, name="llm-guard", daemon=True)
        thread.start()
        self._thread = thread
        return thread

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(5):
            if self.health is not None:
                self.health.begin("llm-guard")
            error = ""
            try:
                self.step()
            except Exception as exc:  # the loop must survive a bad probe
                error = str(exc)
                self._emit("loop.llm-guard.failed", level="error", detail=error)
            finally:
                if self.health is not None:
                    self.health.end("llm-guard", error)
