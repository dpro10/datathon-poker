"""How the arena talks to bots: a local process per bot, or a bot served over HTTP by someone else."""
from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class BotProcess:
    """Runs a bot file in its own Python process with a fresh empty working folder.

    sandboxed=True (for code we did not write): macOS sandbox, no network, no files outside its
    folder, no environment variables from this machine, CPU and file-size limits.
    """

    def __init__(self, path: str, log_path: Path | None, sandboxed: bool = False):
        self.fatal: str | None = None
        self.dead = False
        self.seq = 0
        self.workdir = os.path.realpath(tempfile.mkdtemp(prefix="pokerbot_"))
        if sandboxed:
            from . import sandbox

            cmd = sandbox.command(path, self.workdir) + ["--limits"]
            env = {"PATH": "/usr/bin:/bin", "HOME": self.workdir, "TMPDIR": self.workdir,
                   "LANG": "en_US.UTF-8", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        else:
            cmd = [sys.executable, "-m", "arena.runner", path]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.log = open(log_path, "a", encoding="utf-8") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            cwd=self.workdir, env=env, text=True, encoding="utf-8", bufsize=1,
        )
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

    def request(self, msg: dict, timeout: float):
        """Returns (reply, status) where status is ok, error, timeout or crashed."""
        if self.dead:
            return None, "crashed"
        self.seq += 1
        msg = dict(msg, id=self.seq)
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.dead = True
            return None, "crashed"
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None, "timeout"
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty:
                return None, "timeout"
            if line is None:
                self.dead = True
                return None, "crashed"
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(reply, dict):
                continue
            if "fatal" in reply:
                self.dead = True
                self.fatal = reply["fatal"]
                return reply, "crashed"
            if reply.get("id") != self.seq:
                continue  # late answer to a request that already timed out
            if "error" in reply:
                return reply, "error"
            return reply, "ok"

    def kill(self, reason: str) -> None:
        self.dead = True
        self.fatal = reason
        try:
            self.proc.kill()
        except OSError:
            pass

    def close(self) -> None:
        if not self.dead:
            self.request({"type": "match_end"}, 2.0)
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        if self.log is not subprocess.DEVNULL:
            self.log.close()
        shutil.rmtree(self.workdir, ignore_errors=True)


class BotHTTP:
    """A bot someone is running with `python3 -m arena serve` (their code stays on their laptop)."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.fatal: str | None = None
        self.dead = False
        self.seq = 0

    def request(self, msg: dict, timeout: float):
        if self.dead:
            return None, "crashed"
        self.seq += 1
        body = json.dumps(dict(msg, id=self.seq)).encode()
        req = urllib.request.Request(self.url + "/message", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                reply = json.loads(r.read())
        except (socket.timeout, TimeoutError):
            return None, "timeout"
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                return None, "timeout"
            self.dead = True
            self.fatal = f"cannot reach {self.url}: {e.reason}"
            return None, "crashed"
        except (OSError, ValueError) as e:
            return {"error": f"{type(e).__name__}: {e}"}, "error"
        if "error" in reply:
            return reply, "error"
        return reply, "ok"

    def kill(self, reason: str) -> None:
        self.dead = True
        self.fatal = reason

    def close(self) -> None:
        if not self.dead:
            self.request({"type": "match_end"}, 2.0)


def fetch_http_name(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/", timeout=5) as r:
            return json.loads(r.read()).get("name")
    except Exception:
        return None


def open_bot(kind: str, target: str, log_path: Path | None, sandboxed: bool = False):
    if kind == "http":
        return BotHTTP(target)
    return BotProcess(target, log_path, sandboxed)
