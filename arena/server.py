"""Serve a bot over HTTP so a friend can play against it without seeing the code."""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .host import BotHost


def _lan_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent; this just picks the outgoing interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def serve(bot_path: str, host: str = "0.0.0.0", port: int = 9001) -> None:
    bot_host = BotHost(bot_path)
    name = Path(bot_path).stem
    lock = threading.Lock()  # one message at a time, like a local bot

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self._send(200, {"name": name, "ok": True})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/message":
                self._send(404, {"error": "use POST /message"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                msg = json.loads(self.rfile.read(length))
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "bad json"})
                return
            with lock:
                reply = bot_host.handle(msg)
            self._send(200, reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving bot '{name}' on port {port}. Leave this running; Ctrl+C to stop.")
    ip = _lan_ip() if host in ("0.0.0.0", "") else host
    if ip:
        print(f"Your friend runs:  python3 -m arena match <their_bot> http://{ip}:{port}")
    print(f"To test it yourself: python3 -m arena match calling_station http://127.0.0.1:{port} --hands 100")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
