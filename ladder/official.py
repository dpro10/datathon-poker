"""Official matches without web servers.

PipeAgentHandle is the official judge's HttpAgentHandle with one change: requests travel over
a pipe to a sandboxed bot process (ladder/agent_host.py) instead of HTTP. Everything else, the
time bank, the 1 ms grace, the 1 s limit, error counting, forfeits, batching of finished hands,
is the judge's own code, and matches are played and scored with its play_half and
combine_halves. Each half runs in brand-new bot processes, like the tournament's pods.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / "official"
HOST = ROOT / "ladder" / "agent_host.py"
if str(OFFICIAL) not in sys.path:
    sys.path.insert(0, str(OFFICIAL))

import evaluator as official_evaluator  # noqa: E402
import judge_engine as je  # noqa: E402
from game import AgentConfig  # noqa: E402

from arena import sandbox  # noqa: E402

official_evaluator.warmup()  # build the judge's table once, before any threads start
# The tournament talks HTTP, which costs a bot about 0.55 ms of the free 1 ms per request.
# Pipes are nearly free, so shrink the grace to charge thinking time the way the tournament does.
je.NETWORK_GRACE_SECONDS = 0.00045


class PipeAgentHandle(je.HttpAgentHandle):
    def __init__(self, bot_path: str, name: str, config: AgentConfig, log_path: Path | None = None,
                 sandboxed: bool = True):
        # The judge's handle fields, without its URL parsing.
        self.base_url, self.host, self.port = f"ladder:{name}", "pipe", 0
        self.config, self.token = config, None
        self.stats = je.AgentStats()
        self._conn, self._info, self._pending_results = None, {}, []
        self.name, self.fatal, self._seq, self._dead = name, None, 0, False
        self.workdir = os.path.realpath(tempfile.mkdtemp(prefix="pokerbot_"))
        bot_path = str(Path(bot_path).resolve())
        if sandboxed:
            cmd = sandbox.wrap(str(HOST), [bot_path, "--limits"], str(Path(bot_path).parent), self.workdir,
                               extra_reads=(OFFICIAL, ROOT / "ladder"))
            env = {"PATH": "/usr/bin:/bin", "HOME": self.workdir, "TMPDIR": self.workdir, "LANG": "en_US.UTF-8",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        else:
            cmd = [sys.executable, "-B", str(HOST), bot_path]
            env = dict(os.environ)
        self.log = open(log_path, "a", encoding="utf-8") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     cwd=self.workdir, env=env, text=True, encoding="utf-8", bufsize=1)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        try:
            for line in self.proc.stdout:
                self.lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self.lines.put(None)

    def _close(self) -> None:
        pass  # a pipe has no connection to reset

    def _request(self, method, path, body, timeout):
        if self._dead:
            raise ConnectionError(f"bot is not running ({self.fatal or 'process exited'})")
        self._seq += 1
        try:
            self.proc.stdin.write(json.dumps({"id": self._seq, "method": method, "path": path, "body": body}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._dead = True
            raise ConnectionError("bot process exited") from exc
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("no reply in time")
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty:
                raise TimeoutError("no reply in time") from None
            if line is None:
                self._dead = True
                raise ConnectionError(f"bot process exited ({self.fatal or 'crashed'})")
            if len(line) > je.MAX_RESPONSE_BYTES:
                raise ValueError("reply too large")
            try:
                reply = json.loads(line)
            except ValueError:
                continue
            if not isinstance(reply, dict):
                continue
            if "fatal" in reply:
                self._dead = True
                self.fatal = reply["fatal"]
                raise ConnectionError(f"bot failed to start: {self.fatal}")
            if reply.get("id") != self._seq:
                continue  # late answer to a request that already timed out
            return int(reply.get("status", 500)), reply.get("data")

    def shutdown(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        if self.log is not subprocess.DEVNULL:
            self.log.close()
        shutil.rmtree(self.workdir, ignore_errors=True)


def play_half(a: tuple[str, str], b: tuple[str, str], seed: str, half: int, deals: int,
              out_dir: Path | None = None, sandboxed: bool = True) -> "je.HalfResult":
    """a, b: (bot_path, name). Fresh bot processes for this half only."""
    config = AgentConfig(num_hands=deals)
    handles = []
    try:
        for path, name in (a, b):
            log = out_dir / f"{name}_half{half}.log" if out_dir else None
            handles.append(PipeAgentHandle(path, name, config, log, sandboxed))
        for h in handles:
            h.wait_healthy(je.STARTUP_TIMEOUT_SECONDS)
        return je.play_half(handles[0], handles[1], seed, half, deals)
    finally:
        for h in handles:
            h.shutdown()


def play_match(a: tuple[str, str], b: tuple[str, str], deals: int = 1000, seed: str | None = None,
               out_dir: Path | None = None, sandboxed: bool = True) -> "je.MatchResult":
    """Both halves at once (separate processes), then the official verdict."""
    seed = seed or je.new_match_seed()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        h1, h2 = pool.map(lambda h: play_half(a, b, seed, h, deals, out_dir, sandboxed), (1, 2))
    return je.combine_halves(h1, h2)


def summarize(m: "je.MatchResult") -> dict:
    stats = {k: [s.to_dict() for s in v] for k, v in m.agent_stats.items()}
    return {"verdict": m.verdict, "points": m.points, "mean_bb": m.mean_bb, "ci": list(m.ci),
            "stderr_bb": m.stderr_bb, "reason": m.reason, "deals": m.num_deals, "seed": m.match_seed,
            "stats": stats, "seconds": round(m.elapsed_seconds, 1)}
