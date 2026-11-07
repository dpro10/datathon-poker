"""python3 -m ladder <serve | stop | submit | list | retire | restore>"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

from .core import DATA_DIR, Ladder

CONFIG = DATA_DIR / "config.json"
PIDFILE = DATA_DIR / "server.pid"
TUNNEL_PIDFILE = DATA_DIR / "tunnel.pid"
URLFILE = DATA_DIR / "public_url.txt"
KEYFILE = DATA_DIR / "my_keys.json"
WORDS = ("ace", "king", "queen", "river", "turn", "flop", "chip", "stack", "pot", "bluff",
         "suited", "nuts", "maroon", "spade", "button", "blind", "check", "raise", "deuce", "kicker")


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    os.chmod(path, 0o600)


def _start_tunnel(port: int):
    proc = subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    def pump():
        # Publish the link only once Cloudflare reports the tunnel connected. Looking the name up
        # earlier makes computers cache "no such address" for many minutes.
        url, published = None, False
        for line in proc.stderr:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m and not url:
                url = m.group(0)
            if " ERR " in line:
                print(f"tunnel: {line.strip()}", flush=True)
            if url and not published and "Registered tunnel connection" in line:
                published = True
                URLFILE.write_text(url + "\n")
                print(f"Public link: {url}", flush=True)

    threading.Thread(target=pump, daemon=True).start()
    return proc


def cmd_serve(args) -> int:
    cfg = _load(CONFIG)
    if args.invite:
        cfg["invite"] = args.invite
    if not cfg.get("invite"):
        cfg["invite"] = f"{secrets.choice(WORDS)}-{secrets.choice(WORDS)}-{secrets.randbelow(9000) + 1000}"
    _save(CONFIG, cfg)
    ladder = Ladder(invite=cfg["invite"], deals=args.deals, workers=args.workers)
    if not ladder.sandboxed:
        print("WARNING: macOS sandbox not available; uploaded bots would run unrestricted. Refusing to start.")
        return 1
    ladder.ensure_house_bots()
    from .web import make_server

    server = make_server(ladder, "127.0.0.1", args.port)
    ladder.start()
    PIDFILE.write_text(str(os.getpid()))
    if args.tunnel:
        URLFILE.unlink(missing_ok=True)
    helpers = []
    if sys.platform == "darwin":
        helpers.append(subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]))
    if args.tunnel:
        helpers.append(_start_tunnel(args.port))
    print(f"Ladder running at http://127.0.0.1:{args.port}   invite code: {cfg['invite']}", flush=True)
    print(f"Official rules: {args.deals} deals per half, both halves in fresh sandboxed processes, "
          f"{args.workers} matches at a time", flush=True)

    def on_term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ladder.shutdown(wait=5)
        for p in helpers:
            p.terminate()
        PIDFILE.unlink(missing_ok=True)
        if args.tunnel:
            URLFILE.unlink(missing_ok=True)
    return 0


def cmd_tunnel(args) -> int:
    """Keep the public link in its own process, so restarting the ladder keeps the same link."""
    URLFILE.unlink(missing_ok=True)
    TUNNEL_PIDFILE.write_text(str(os.getpid()))
    proc = _start_tunnel(args.port)
    signal.signal(signal.SIGTERM, lambda *_: proc.terminate())
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0
    finally:
        URLFILE.unlink(missing_ok=True)
        TUNNEL_PIDFILE.unlink(missing_ok=True)


def _kill(pidfile: Path, what: str) -> bool:
    try:
        pid = int(pidfile.read_text())
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped the {what} (pid {pid}).")
        return True
    except (OSError, ValueError):
        print(f"The {what} isn't running.")
        return False


def cmd_stop(args) -> int:
    ok = _kill(PIDFILE, "ladder")
    if args.all:
        ok = _kill(TUNNEL_PIDFILE, "public link") and ok
    return 0 if ok else 1


def _git_name() -> str:
    try:
        out = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip().split()[0] if out.stdout.strip() else os.environ.get("USER", "me")
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("USER", "me")


def cmd_submit(args) -> int:
    server = (args.server or "http://127.0.0.1:8787").rstrip("/")
    invite = args.invite or _load(CONFIG).get("invite", "")
    path = Path(args.file)
    name = args.name or path.stem
    body = json.dumps({"name": name, "author": args.author or _git_name(), "invite": invite,
                       "code": path.read_text(encoding="utf-8")}).encode()
    req = urllib.request.Request(server + "/api/bots", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        res = json.loads(e.read() or b"{}")
        print(f"Rejected: {res.get('error', e.reason)}")
        if res.get("details"):
            print(res["details"])
        return 1
    except urllib.error.URLError as e:
        print(f"Couldn't reach {server}: {e.reason}")
        return 1
    keys = _load(KEYFILE)
    keys[name] = {"id": res["id"], "key": res["key"], "server": server}
    _save(KEYFILE, keys)
    print(f"Entered {name} (bot #{res['id']}). Its owner key is saved in {KEYFILE}.")
    return 0


def cmd_list(args) -> int:
    state = Ladder().state()
    print(f"{'#':<4}{'Bot':<24}{'Author':<14}{'Rating':>7}{'W-C-L':>10}{'Score':>7}{'bb/deal':>9}  Status")
    rank = 0
    for b in state["bots"]:
        rank += b["active"]
        bb = "-" if b["bb_deal"] is None else f"{b['bb_deal']:+.2f}"
        score = "-" if b["score"] is None else f"{b['score']:.2f}"
        status = "active" if b["active"] else (b["note"] or "retired")
        record = f"{b['wins']}-{b['chops']}-{b['losses']}"
        print(f"{rank if b['active'] else '-':<4}{b['name'][:23]:<24}{b['author'][:13]:<14}{b['rating']:>7}"
              f"{record:>10}{score:>7}{bb:>9}  {status}")
    print(f"\n{state['total_matches']} matches played")
    return 0


def cmd_add(args) -> int:
    from .core import SubmitError

    try:
        bot_id = Ladder().add_local(args.file, args.name, args.author or _git_name())
    except SubmitError as e:
        print(f"Rejected: {e.message}")
        if e.details:
            print(e.details)
        return 1
    print(f"Added {args.name} (bot #{bot_id}) from {args.file}. A running ladder picks it up within seconds.")
    return 0


def cmd_set_active(args, active: bool) -> int:
    ok = Ladder().set_active(args.name, active)
    print(f"{'Restored' if active else 'Retired'} {args.name}." if ok else f"No bot named {args.name}.")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m ladder", description="Bot ladder website for the poker arena.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the ladder website and match workers")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--tunnel", action="store_true", help="also open a public Cloudflare link")
    s.add_argument("--invite", help="invite code (default: saved one, or a new random one)")
    s.add_argument("--workers", type=int, default=2, help="matches played at the same time (default 2)")
    s.add_argument("--deals", type=int, default=1000, help="deals per half (official: 1000)")
    st = sub.add_parser("stop", help="stop a ladder started in the background")
    st.add_argument("--all", action="store_true", help="also close the public link")
    tn = sub.add_parser("tunnel", help="open the public link in its own process (survives ladder restarts)")
    tn.add_argument("--port", type=int, default=8787)
    a = sub.add_parser("add", help="host only: enter a bot from a folder on this machine")
    a.add_argument("file", help="the bot's agent.py (its folder may hold its own modules)")
    a.add_argument("--name", required=True)
    a.add_argument("--author", help="default: from git config")
    u = sub.add_parser("submit", help="upload a bot file to a ladder")
    u.add_argument("file")
    u.add_argument("--name", help="bot name (default: file name)")
    u.add_argument("--author", help="your name (default: from git config)")
    u.add_argument("--server", help="ladder address (default http://127.0.0.1:8787)")
    u.add_argument("--invite", help="invite code (default: this machine's ladder code)")
    sub.add_parser("list", help="print the leaderboard")
    r = sub.add_parser("retire", help="take a bot off the ladder (host only)")
    r.add_argument("name")
    r2 = sub.add_parser("restore", help="put a retired bot back on the ladder (host only)")
    r2.add_argument("name")
    args = ap.parse_args(argv)
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "stop":
        return cmd_stop(args)
    if args.cmd == "tunnel":
        return cmd_tunnel(args)
    if args.cmd == "submit":
        return cmd_submit(args)
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "add":
        return cmd_add(args)
    return cmd_set_active(args, args.cmd == "restore")


if __name__ == "__main__":
    sys.exit(main())
