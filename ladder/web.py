"""The ladder website: a small JSON API plus one static page. Standard library only."""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .core import Ladder, SubmitError

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 500_000
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
PAGES = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8")}


def make_server(ladder: Ladder, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "PokerLadder"
        sys_version = ""

        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy", CSP)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path in PAGES:
                name, ctype = PAGES[path]
                return self._send(200, (STATIC / name).read_bytes(), ctype)
            if path == "/api/state":
                return self._json(200, ladder.state())
            m = re.fullmatch(r"/api/bots/(\d+)", path)
            if m:
                detail = ladder.bot_detail(int(m.group(1)))
                return self._json(200, detail) if detail else self._json(404, {"error": "No such bot."})
            return self._json(404, {"error": "Not found."})

        do_HEAD = do_GET  # noqa: N815 (link previews and uptime checks)

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                return self._json(413, {"error": "Too big (limit 500 KB)."})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._json(400, {"error": "Bad request."})
            if not isinstance(body, dict):
                return self._json(400, {"error": "Bad request."})
            if path == "/api/bots":
                try:
                    bot_id, key = ladder.submit(str(body.get("name", "")), str(body.get("author", "")),
                                                str(body.get("code", "")), str(body.get("invite", "")))
                except SubmitError as e:
                    return self._json(400, {"error": e.message, "details": e.details})
                return self._json(201, {"id": bot_id, "key": key})
            m = re.fullmatch(r"/api/bots/(\d+)/(retire|log)", path)
            if m:
                bot_id, action, key = int(m.group(1)), m.group(2), str(body.get("key", ""))
                if action == "retire":
                    if ladder.retire(bot_id, key):
                        return self._json(200, {"ok": True})
                else:
                    text = ladder.owner_log(bot_id, key)
                    if text is not None:
                        return self._json(200, {"log": text})
                return self._json(403, {"error": "That owner key doesn't match this bot."})
            return self._json(404, {"error": "Not found."})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
