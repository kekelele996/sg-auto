"""Hold new launches while Docker cannot start a container in reasonable time.

``docker ps`` answering is not enough: under load OrbStack still lists
containers but takes minutes to create one, the skill's ``docker run`` times
out (180 s) and every task launched meanwhile burns quota on
``attempt_invalid`` candidates.  The probe starts the skill's own image with a
bind mount and ``/bin/true`` — the same path a candidate takes, minus the
setup — in a background thread, so the scheduler lock never waits on Docker.

``FAIL_THRESHOLD`` slow or failed probes in a row mark Docker unhealthy and the
queue logs an outage; the same number of fast probes clear it.  Launches
already wait after one failed probe (the gate is in QueueManager.tick).  The
gate never touches ``automation.paused``: nothing has to be resumed by hand
and it cannot fight the LLM guard's pause.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .common import iso_from_timestamp

FALLBACK_IMAGE = "adminfather/benzhi-claude-code2:20260919"
PROBE_LABEL = "sologsb.probe=true"
FAIL_THRESHOLD = 2

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # A bare container slower than this is a failed probe; a candidate's real
    # setup takes longer and the skill gives up at 180 s.
    "slowSeconds": 60,
    "intervalSeconds": 300,
    # While unhealthy, probe more often so launches resume soon after recovery.
    "unhealthyIntervalSeconds": 60,
}


def docker_probe_settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = (config.get("automation") or {}).get("dockerProbe") or {}

    def number(key: str, low: int, high: int) -> int:
        try:
            value = int(cfg.get(key, DEFAULTS[key]))
        except (TypeError, ValueError):
            value = DEFAULTS[key]
        return max(low, min(high, value))

    return {
        "enabled": bool(cfg.get("enabled", DEFAULTS["enabled"])),
        "slowSeconds": number("slowSeconds", 10, 170),
        "intervalSeconds": number("intervalSeconds", 60, 3600),
        "unhealthyIntervalSeconds": number("unhealthyIntervalSeconds", 20, 3600),
        "image": str(cfg.get("image") or ""),
    }


def resolve_image(config: dict[str, Any]) -> str:
    """The image the skill starts candidates from."""
    configured = str(((config.get("automation") or {}).get("dockerProbe") or {}).get("image") or "")
    if configured:
        return configured
    if os.environ.get("SOLOSB_DOCKER_IMAGE"):
        return os.environ["SOLOSB_DOCKER_IMAGE"]
    script = str(config.get("skillScript") or "")
    if script:
        runner = Path(script).expanduser().parent / "side_runner.py"
        try:
            text = runner.read_text(encoding="utf-8")
        except OSError:
            text = ""
        match = re.search(r'DEFAULT_IMAGE\s*=\s*os\.environ\.get\(\s*"SOLOSB_DOCKER_IMAGE"\s*,\s*"([^"]+)"', text)
        if match:
            return match.group(1)
    return FALLBACK_IMAGE


def _remove_probes(names: list[str] | None = None) -> None:
    """Best effort: a Docker slow enough to fail the probe may be slow to rm too."""
    try:
        if names is None:
            proc = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={PROBE_LABEL}"],
                                  capture_output=True, text=True, timeout=30, check=False)
            names = proc.stdout.split()
        if names:
            subprocess.run(["docker", "rm", "-f", *names], capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_probe(image: str, timeout: float) -> dict[str, Any]:
    """Start and remove one throwaway container; report how long it took."""
    # A probe that timed out earlier may have left its container behind.
    _remove_probes()
    name = f"sologsb-probe-{os.getpid()}-{int(time.time())}"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sologsb-probe-") as mount:
        command = [
            "docker", "run", "--rm", "--pull=never", "--name", name, "--label", PROBE_LABEL,
            "--mount", f"type=bind,src={mount},dst=/probe", "--entrypoint", "/bin/true", image,
        ]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            _remove_probes([name])
            return {"ok": False, "seconds": round(time.monotonic() - started, 1),
                    "error": f"{timeout:g} 秒内未能启动容器"}
        except OSError as exc:
            return {"ok": False, "seconds": round(time.monotonic() - started, 1), "error": str(exc)}
    seconds = round(time.monotonic() - started, 1)
    if proc.returncode != 0:
        error = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"ok": False, "seconds": seconds,
                "error": (error[-1] if error else f"docker run 退出码 {proc.returncode}")[:300]}
    return {"ok": True, "seconds": seconds, "error": ""}


class DockerProbe:
    """Probes Docker off-thread; ``healthy`` is what the launch gate reads."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        emit: Callable[..., None] | None = None,
        probe: Callable[[str, float], dict[str, Any]] = run_probe,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config
        self._emit = emit
        self._probe = probe
        self._clock = clock
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.healthy = True
        self.failures = 0
        self.successes = 0
        self.last_probe: dict[str, Any] = {}
        self.last_probe_at = 0.0
        self.last_result_at = 0.0
        self.unhealthy_since = 0.0

    def settings(self) -> dict[str, Any]:
        return docker_probe_settings(self.config)

    def blocking(self) -> bool:
        """True while launches must wait; also schedules the next probe."""
        settings = self.settings()
        if not settings["enabled"]:
            return False
        self.maybe_probe(settings)
        # A single failed probe already holds: the next one, due within
        # unhealthyIntervalSeconds, confirms the outage or clears it.
        return not self.healthy or not self._fresh(settings) or not self.last_probe.get("ok")

    def _fresh(self, settings: dict[str, Any]) -> bool:
        # After a long idle spell the first launch waits for one new probe.
        return bool(self.last_result_at) and self._clock() - self.last_result_at < 2 * settings["intervalSeconds"]

    def maybe_probe(self, settings: dict[str, Any] | None = None) -> bool:
        settings = settings or self.settings()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            interval = settings["intervalSeconds"] if self.healthy else settings["unhealthyIntervalSeconds"]
            # A probe that just failed below the threshold is confirmed quickly.
            if self.failures:
                interval = min(interval, settings["unhealthyIntervalSeconds"])
            # Counted from the last result: a probe that took minutes must not
            # be followed at once by another on an already saturated Docker.
            last = max(self.last_probe_at, self.last_result_at)
            if last and self._clock() - last < interval:
                return False
            self.last_probe_at = self._clock()
            thread = threading.Thread(target=self._run, args=(settings,), name="docker-probe", daemon=True)
            self._thread = thread
        thread.start()
        return True

    def _run(self, settings: dict[str, Any]) -> None:
        slow = settings["slowSeconds"]
        try:
            # Let a slow start finish (up to the skill's own limit) so the page
            # shows the real duration instead of just "timed out".
            result = self._probe(settings["image"] or resolve_image(self.config), float(min(180, slow * 2)))
        except Exception as exc:  # the probe must never kill its thread silently
            result = {"ok": False, "seconds": 0.0, "error": str(exc)}
        if result.get("ok") and float(result.get("seconds") or 0) > slow:
            result = {**result, "ok": False, "error": f"启动空容器用了 {result['seconds']:g} 秒，超过 {slow} 秒"}
        self.record(result)

    def record(self, result: dict[str, Any]) -> None:
        now = self._clock()
        with self._lock:
            self.last_probe = {**result, "at": iso_from_timestamp(now)}
            self.last_result_at = now
            if result.get("ok"):
                self.failures = 0
                self.successes += 1
                recovered = not self.healthy and self.successes >= FAIL_THRESHOLD
                failed = False
                if recovered:
                    self.healthy = True
                    self.unhealthy_since = 0.0
            else:
                self.successes = 0
                self.failures += 1
                failed = self.healthy and self.failures >= FAIL_THRESHOLD
                recovered = False
                if failed:
                    self.healthy = False
                    self.unhealthy_since = now
        if failed:
            self._log("warning", "queue.docker_slow",
                      detail=f"Docker 连续 {FAIL_THRESHOLD} 次启动容器过慢/失败（{result.get('error')}），暂停启动新任务")
        elif recovered:
            self._log("info", "queue.docker_probe_recovered",
                      detail=f"Docker 启动空容器 {result.get('seconds')} 秒，恢复启动新任务")

    def hold_reason(self) -> str:
        if not self.last_probe or (self.healthy and self.last_probe.get("ok")):
            return "正在检测 Docker 能否正常启动容器，完成后启动"
        error = str(self.last_probe.get("error") or "")
        return f"Docker 创建容器过慢/失败（{error}）：暂停启动新任务，恢复后自动继续"

    def status(self) -> dict[str, Any]:
        settings = self.settings()
        return {
            **settings,
            "healthy": self.healthy,
            "failures": self.failures,
            "lastProbe": dict(self.last_probe),
            "unhealthySince": iso_from_timestamp(self.unhealthy_since) if self.unhealthy_since else "",
            "probing": bool(self._thread is not None and self._thread.is_alive()),
        }

    def _log(self, level: str, event: str, **fields: Any) -> None:
        if self._emit is None:
            return
        try:
            self._emit(event, level=level, **fields)
        except Exception:
            pass
