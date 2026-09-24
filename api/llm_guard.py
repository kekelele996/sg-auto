"""Pause the queue while the LLM the containers call is down, resume when it is back.

The candidates inside the containers talk to ``anthropicBaseUrl``; when that
endpoint drops, every task launched meanwhile burns quota and fails.  The
guard probes the endpoint with a one-token request.  After ``failThreshold``
failures in a row it pauses the queue and marks the pause as its own
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

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "probeSeconds": 60,
    "pausedProbeSeconds": 300,
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
        "probeSeconds": number("probeSeconds", 15, 3600),
        "pausedProbeSeconds": number("pausedProbeSeconds", 60, 3600),
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


def probe_llm(endpoint: dict[str, str], timeout: float = 30) -> dict[str, Any]:
    """One ``/v1/messages`` call with ``max_tokens=1``; never returns the key."""
    started = time.time()
    result: dict[str, Any] = {
        "ok": False,
        "at": utc_now(),
        "baseUrl": endpoint.get("baseUrl", ""),
        "model": endpoint.get("model", ""),
        "status": None,
        "latencyMs": None,
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
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
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
            payload = json.loads(response.read().decode("utf-8") or "{}")
        if isinstance(payload, dict) and payload.get("type") == "error":
            error = payload.get("error") or {}
            result["error"] = str(error.get("message") if isinstance(error, dict) else error)[:300]
        else:
            result["ok"] = True
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
        self.failures = 0
        self.last_probe: dict[str, Any] = {}
        self.last_probe_at = 0.0
        self.last_change = ""

    # -- config ------------------------------------------------------------ #
    def _cfg(self) -> dict[str, Any]:
        return self.config.setdefault("automation", {}).setdefault("llmGuard", {})

    def settings(self) -> dict[str, Any]:
        return llm_guard_settings(self.config)

    def status(self) -> dict[str, Any]:
        settings = self.settings()
        interval = settings["pausedProbeSeconds"] if settings["pausedByGuard"] else settings["probeSeconds"]
        next_at = self.last_probe_at + interval if self.last_probe_at else 0.0
        return {
            **settings,
            "failures": self.failures,
            "lastProbe": dict(self.last_probe),
            "lastChange": self.last_change,
            "nextProbeInSeconds": max(0, int(next_at - self._clock())) if settings["enabled"] and next_at else None,
        }

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            cfg = self._cfg()
            cfg["enabled"] = bool(enabled)
            self.failures = 0
            # Turning the guard off hands its pause back to the operator.
            if not enabled:
                cfg["pausedByGuard"] = False
            self.last_probe_at = 0.0
            self._persist()

    def note_manual_pause(self) -> None:
        """Any manual start/pause takes ownership of ``paused`` away from the guard."""
        with self._lock:
            if self._cfg().get("pausedByGuard"):
                self._cfg()["pausedByGuard"] = False
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
        interval = settings["pausedProbeSeconds"] if settings["pausedByGuard"] else settings["probeSeconds"]
        return self._clock() - self.last_probe_at >= interval

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
        with self._lock:
            self.last_probe = dict(result)
            self.last_probe_at = self._clock()
            automation = self.config.setdefault("automation", {})
            cfg = self._cfg()
            if result.get("ok"):
                self.failures = 0
                if settings["enabled"] and cfg.get("pausedByGuard"):
                    cfg["pausedByGuard"] = False
                    automation["paused"] = False
                    self.last_change = f"{utc_now()} 大模型恢复，队列自动启动"
                    self._persist()
                    self._emit("llm_guard.resumed", detail=f"大模型可用（{result.get('latencyMs')}ms），队列自动启动")
                return
            self.failures += 1
            self._emit("llm_guard.probe_failed", level="warning",
                       detail=f"第 {self.failures} 次失败：{result.get('error')}")
            if (
                settings["enabled"]
                and self.failures >= settings["failThreshold"]
                and not bool(automation.get("paused", True))
            ):
                automation["paused"] = True
                cfg["pausedByGuard"] = True
                self.last_change = f"{utc_now()} 大模型不可用，队列自动暂停"
                self._persist()
                self._emit("llm_guard.paused", level="warning",
                           detail=f"大模型连续 {self.failures} 次不可用，队列自动暂停，"
                                  f"每 {settings['pausedProbeSeconds']} 秒复测：{result.get('error')}")

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
