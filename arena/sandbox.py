"""macOS sandbox for bots we did not write (used by the ladder).

A sandboxed bot can read Python, numpy, pokerlib/arena and its own folder, and can write only to
its private scratch folder. It cannot use the network, start other programs, or read anything
else on the machine (home folder, keys, other bots).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def available() -> bool:
    return sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()


def _q(path: str) -> str:
    return '"' + os.path.realpath(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _python_read_paths() -> list[str]:
    paths = {sys.base_prefix, sys.prefix, os.path.dirname(os.path.realpath(sys.executable))}
    for p in sys.path:
        if p and os.path.isdir(p) and "site-packages" in p and not p.startswith(str(Path.home())):
            paths.add(p)
    if sys.platform == "darwin" and Path("/opt/homebrew").exists():
        # Homebrew Python links against libraries under these folders; config and data folders stay hidden.
        paths |= {"/opt/homebrew/Cellar", "/opt/homebrew/opt", "/opt/homebrew/lib"}
    return sorted(paths)


def profile(bot_dir: str, workdir: str, extra_reads=()) -> str:
    reads = _python_read_paths() + [
        "/System", "/usr/lib", "/usr/share", "/private/var/db/dyld", "/private/var/db/timezone",
        "/dev", str(ROOT / "arena"), str(ROOT / "pokerlib"), bot_dir, workdir, *map(str, extra_reads),
    ]
    read_rules = " ".join(f"(subpath {_q(p)})" for p in reads if os.path.exists(p))
    exec_rules = " ".join(f"(subpath {_q(p)})" for p in _python_read_paths() if os.path.exists(p))
    return f"""(version 1)
(deny default)
(allow process-exec {exec_rules} (literal {_q(sys.executable)}))
(allow file-read-metadata)
(allow file-read* {read_rules} (literal "/") (literal {_q(str(ROOT))}) (literal "/private/etc/localtime"))
(allow file-write* (subpath {_q(workdir)}) (literal "/dev/null"))
(allow sysctl-read)
(allow signal (target self))
(deny network*)
"""


def wrap(script: str, args: list, bot_dir: str, workdir: str, extra_reads=()) -> list[str]:
    """Command that runs `python -I -B script *args` inside the sandbox."""
    return ["/usr/bin/sandbox-exec", "-p", profile(bot_dir, workdir, extra_reads),
            sys.executable, "-I", "-B", script, *map(str, args)]


def command(bot_path: str, workdir: str) -> list[str]:
    """Command that runs the bot host for bot_path inside the sandbox."""
    bot_dir = str(Path(bot_path).resolve().parent)
    runner = str(ROOT / "arena" / "runner.py")
    return ["/usr/bin/sandbox-exec", "-p", profile(bot_dir, workdir),
            sys.executable, "-I", "-B", runner, str(Path(bot_path).resolve())]
