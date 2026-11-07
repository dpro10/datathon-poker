"""Loads a bot file and answers protocol messages. Shared by the subprocess runner and the HTTP server."""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


def load_bot_class(path: str | Path):
    path = Path(path).resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"bot_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Bot"):
        raise RuntimeError(f"{path.name} must define a class named Bot")
    return module.Bot


def normalize_action(a) -> dict:
    if isinstance(a, str):
        a = {"action": a}
    elif isinstance(a, (tuple, list)):
        a = {"action": a[0], "amount": a[1] if len(a) > 1 else None}
    if not isinstance(a, dict):
        raise TypeError(f"act() must return a dict like {{'action': 'call'}}, got {type(a).__name__}")
    action = str(a.get("action", "")).lower().strip()
    if action == "bet":
        action = "raise"
    out = {"action": action}
    if a.get("amount") is not None:
        out["amount"] = int(round(float(a["amount"])))
    return out


class BotHost:
    def __init__(self, path: str | Path):
        self.cls = load_bot_class(path)
        self.bot = None

    def handle(self, msg: dict) -> dict:
        mid = msg.get("id")
        kind = msg.get("type")
        try:
            if kind == "match_start":
                self.bot = self.cls()  # fresh bot for every half of every match
                hook = getattr(self.bot, "new_match", None)
                if hook:
                    hook(msg.get("info", {}))
                return {"id": mid, "ok": True}
            if self.bot is None:
                self.bot = self.cls()
            if kind == "act":
                return {"id": mid, **normalize_action(self.bot.act(msg["state"]))}
            if kind == "hand_over":
                hook = getattr(self.bot, "hand_over", None)
                if hook:
                    hook(msg["result"])
                return {"id": mid, "ok": True}
            if kind == "match_end":
                hook = getattr(self.bot, "match_end", None)
                if hook:
                    hook()
                return {"id": mid, "ok": True}
            return {"id": mid, "error": f"unknown message type {kind!r}"}
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return {"id": mid, "error": f"{type(e).__name__}: {e}"}
