"""The ladder: stores bots, plays duplicate matches between them nonstop, keeps Elo ratings."""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import secrets
import shutil
import sqlite3
import tempfile
import threading
import time
import traceback
import zipfile
from pathlib import Path

from arena import sandbox
from arena.match import BOTS_DIR, ROOT, BotSpec, run_match, safe_name
from pokerlib import Rules

DATA_DIR = ROOT / "ladder_data"
HOUSE_BOTS = ("calling_station", "maniac", "random_bot", "rock", "equity_bot")
STARTER_BOTS = ("calling_station", "maniac", "random_bot", "rock")  # equity_bot stays private
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,23}$")
MAX_CODE_BYTES = 200_000
KEEP_MATCH_DIRS = 400
AUTO_RETIRE_AFTER = 3  # matches in a row in which the bot crashed

SCHEMA = """
CREATE TABLE IF NOT EXISTS bots (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  author TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  owner_key_hash TEXT,
  created_at REAL NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  note TEXT,
  rating REAL NOT NULL DEFAULT 1500,
  matches INTEGER NOT NULL DEFAULT 0,
  wins INTEGER NOT NULL DEFAULT 0,
  losses INTEGER NOT NULL DEFAULT 0,
  draws INTEGER NOT NULL DEFAULT 0,
  chips INTEGER NOT NULL DEFAULT 0,
  hands INTEGER NOT NULL DEFAULT 0,
  timeouts INTEGER NOT NULL DEFAULT 0,
  errors INTEGER NOT NULL DEFAULT 0,
  illegal INTEGER NOT NULL DEFAULT 0,
  crashes INTEGER NOT NULL DEFAULT 0,
  crash_streak INTEGER NOT NULL DEFAULT 0,
  last_problem TEXT
);
CREATE TABLE IF NOT EXISTS matches (
  id INTEGER PRIMARY KEY,
  a INTEGER NOT NULL REFERENCES bots(id),
  b INTEGER NOT NULL REFERENCES bots(id),
  net_a INTEGER NOT NULL,
  hands INTEGER NOT NULL,
  t REAL,
  seed INTEGER,
  rating_a REAL,
  rating_b REAL,
  delta_a REAL,
  health TEXT,
  seconds REAL,
  dir TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS matches_by_a ON matches(a);
CREATE INDEX IF NOT EXISTS matches_by_b ON matches(b);
"""

MATCH_SELECT = ("SELECT m.*, a.name AS a_name, b.name AS b_name FROM matches m "
                "JOIN bots a ON a.id = m.a JOIN bots b ON b.id = m.b")

H2H_SQL = """
SELECT o.id AS opp_id, o.name AS opp_name, o.rating AS opp_rating, COUNT(*) AS n,
  SUM(CASE WHEN (m.a = :id AND m.net_a > 0) OR (m.b = :id AND m.net_a < 0) THEN 1 ELSE 0 END) AS wins,
  SUM(CASE WHEN (m.a = :id AND m.net_a < 0) OR (m.b = :id AND m.net_a > 0) THEN 1 ELSE 0 END) AS losses,
  SUM(CASE WHEN m.a = :id THEN m.net_a ELSE -m.net_a END) AS chips,
  SUM(m.hands) AS hands
FROM matches m JOIN bots o ON o.id = CASE WHEN m.a = :id THEN m.b ELSE m.a END
WHERE m.a = :id OR m.b = :id
GROUP BY o.id ORDER BY chips DESC
"""


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _scrub(text: str) -> str:
    """Remove this machine's folder paths (username, temp folders) from text shown on the site."""
    text = re.sub(r'File "[^"]*/bot\.py"', 'File "bot.py"', text)
    for root in {str(ROOT), str(ROOT.resolve())}:
        text = text.replace(root + "/", "")
    text = re.sub(r"/(?:private/)?var/folders/[^\s'\"]*/", "", text)
    text = re.sub(r"/Users/[^/\s'\"]+/", "~/", text)
    return text


def _tail(paths, lines: int = 60) -> str:
    out = []
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            out.append(f"--- {Path(p).name} ---\n" + "\n".join(text.splitlines()[-lines:]))
    return _scrub("\n".join(out))


class SubmitError(Exception):
    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.message = message
        self.details = details


class Ladder:
    def __init__(self, data_dir: Path = DATA_DIR, invite: str = "", hands: int = 400, timeout: float = 2.0,
                 time_bank: float = 60.0, workers: int = 2, rules: Rules | None = None):
        self.data = Path(data_dir)
        (self.data / "bots").mkdir(parents=True, exist_ok=True)
        (self.data / "matches").mkdir(parents=True, exist_ok=True)
        self.invite = invite
        self.hands = hands
        self.timeout = timeout
        self.time_bank = time_bank
        self.workers = workers
        self.rules = rules or Rules()
        self.sandboxed = sandbox.available()
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.data / "ladder.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.running: dict = {}
        self.busy_pairs: set = set()
        self.h2h: dict = {}
        for r in self.db.execute("SELECT a, b, COUNT(*) AS n FROM matches GROUP BY a, b"):
            k = frozenset((r["a"], r["b"]))
            self.h2h[k] = self.h2h.get(k, 0) + r["n"]
        self.smoke_slots = threading.Semaphore(2)
        self.stop = threading.Event()
        self.recorded = 0

    # ----- bots -----
    def ensure_house_bots(self) -> None:
        with self.lock:
            for name in HOUSE_BOTS:
                path = BOTS_DIR / f"{name}.py"
                if not path.exists():
                    continue
                row = self.db.execute("SELECT id FROM bots WHERE name = ?", (name,)).fetchone()
                if row is None:
                    self.db.execute("INSERT INTO bots (name, author, kind, path, created_at) "
                                    "VALUES (?, 'house', 'house', ?, ?)", (name, str(path), time.time()))
                else:
                    self.db.execute("UPDATE bots SET path = ? WHERE id = ? AND kind = 'house'", (str(path), row["id"]))
            self.db.commit()

    def submit(self, name: str, author: str, code: str, invite: str) -> tuple[int, str]:
        name, author = name.strip(), " ".join(author.split())
        if self.invite and not secrets.compare_digest(invite.strip().encode(), self.invite.encode()):
            raise SubmitError("Wrong invite code.")
        if not NAME_RE.match(name):
            raise SubmitError("Bot name: up to 24 letters, digits, dots, dashes or underscores (no spaces).")
        if not 1 <= len(author) <= 30:
            raise SubmitError("Add your name (up to 30 characters).")
        if not code.strip():
            raise SubmitError("The file is empty.")
        if len(code.encode()) > MAX_CODE_BYTES:
            raise SubmitError("The file is over 200 KB.")
        try:
            compile(code, f"{name}.py", "exec")
        except (SyntaxError, ValueError) as e:
            line = getattr(e, "lineno", None)
            raise SubmitError(f"Python syntax error{f' on line {line}' if line else ''}: {getattr(e, 'msg', e)}")
        if not re.search(r"^class\s+Bot\b", code, re.M):
            raise SubmitError("The file needs a class named Bot. Start from the template.")
        with self.lock:
            if self.db.execute("SELECT 1 FROM bots WHERE name = ?", (name,)).fetchone():
                raise SubmitError(f"The name {name} is taken. Try {name}-2.")

        staging = Path(tempfile.mkdtemp(prefix="upload_", dir=self.data / "bots"))
        path = staging / "bot.py"
        path.write_text(code, encoding="utf-8")
        test_dir = Path(tempfile.mkdtemp(prefix="smoke_"))
        try:
            spec = BotSpec("file", str(path), name, self.sandboxed)
            station = BotSpec("file", str(BOTS_DIR / "calling_station.py"), "calling_station", self.sandboxed)
            with self.smoke_slots:
                r = run_match(spec, station, self.rules, hands=20, seed=1, timeout=self.timeout,
                              init_timeout=20.0, out_dir=test_dir, time_bank=self.time_bank)
            h = r["health"][r["a"]]
            if h["crashed"] or h["errors"] or h["illegal"] or h["timeouts"]:
                log = _tail(sorted(test_dir.glob(f"{safe_name(name)}_*.log")), 40)
                details = _scrub("\n".join(h["problems"]))
                if log:
                    details += "\n\nWhat your bot printed:\n" + log
                raise SubmitError("Your bot failed a 20-hand test against calling_station.", details)
        except SubmitError:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(staging, ignore_errors=True)
            traceback.print_exc()
            raise SubmitError(f"Testing failed: {type(e).__name__}")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

        key = secrets.token_urlsafe(18)
        with self.lock:
            try:
                cur = self.db.execute("INSERT INTO bots (name, author, kind, path, owner_key_hash, created_at) "
                                      "VALUES (?, ?, 'upload', ?, ?, ?)", (name, author, str(path), _hash(key), time.time()))
            except sqlite3.IntegrityError:
                shutil.rmtree(staging, ignore_errors=True)
                raise SubmitError(f"The name {name} was just taken. Try {name}-2.")
            bot_id = cur.lastrowid
            final = self.data / "bots" / str(bot_id)
            staging.rename(final)
            self.db.execute("UPDATE bots SET path = ? WHERE id = ?", (str(final / "bot.py"), bot_id))
            self.db.commit()
        return bot_id, key

    def _owns(self, row, key: str) -> bool:
        return bool(key and row and row["owner_key_hash"]
                    and secrets.compare_digest(_hash(key), row["owner_key_hash"]))

    def retire(self, bot_id: int, key: str) -> bool:
        with self.lock:
            row = self.db.execute("SELECT * FROM bots WHERE id = ?", (bot_id,)).fetchone()
            if not self._owns(row, key):
                return False
            self.db.execute("UPDATE bots SET active = 0, note = 'retired by its owner' WHERE id = ?", (bot_id,))
            self.db.commit()
            return True

    def set_active(self, name: str, active: bool) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE bots SET active = ?, crash_streak = 0, note = ? WHERE name = ?",
                                  (int(active), None if active else "retired by the host", name))
            self.db.commit()
            return cur.rowcount > 0

    def owner_log(self, bot_id: int, key: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM bots WHERE id = ?", (bot_id,)).fetchone()
            if not self._owns(row, key):
                return None
            m = self.db.execute("SELECT dir FROM matches WHERE a = ? OR b = ? ORDER BY id DESC LIMIT 1",
                                (bot_id, bot_id)).fetchone()
        if not m or not m["dir"]:
            return "No matches yet."
        text = _tail(sorted(Path(m["dir"]).glob(f"{safe_name(row['name'])}_*.log")), 80)
        return text or "Your bot printed nothing in its latest match (print() output shows up here)."

    # ----- views -----
    def _public(self, r) -> dict:
        hands = r["hands"]
        return {
            "id": r["id"], "name": r["name"], "author": r["author"], "house": r["kind"] == "house",
            "active": bool(r["active"]), "note": r["note"], "rating": round(r["rating"]),
            "matches": r["matches"], "wins": r["wins"], "losses": r["losses"], "draws": r["draws"],
            "chips": r["chips"], "hands": hands,
            "bb100": round(r["chips"] / hands / self.rules.big_blind * 100, 1) if hands else None,
            "problems": r["timeouts"] + r["errors"] + r["illegal"] + r["crashes"],
            "last_problem": r["last_problem"], "created_at": r["created_at"],
        }

    @staticmethod
    def _match(r) -> dict:
        return {
            "id": r["id"], "a": r["a"], "b": r["b"], "a_name": r["a_name"], "b_name": r["b_name"],
            "net_a": r["net_a"], "hands": r["hands"], "t": r["t"],
            "delta_a": round(r["delta_a"] or 0, 1), "seconds": r["seconds"], "created_at": r["created_at"],
        }

    def state(self) -> dict:
        with self.lock:
            bots = [self._public(r) for r in self.db.execute("SELECT * FROM bots ORDER BY active DESC, rating DESC")]
            recent = [self._match(r) for r in self.db.execute(MATCH_SELECT + " ORDER BY m.id DESC LIMIT 30")]
            total = self.db.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            running = [dict(v) for v in self.running.values()]
        return {
            "bots": bots, "matches": recent, "running": running, "total_matches": total,
            "settings": {"hands": self.hands, "timeout": self.timeout, "time_bank": self.time_bank,
                         "rules": self.rules.to_dict(), "rules_text": self.rules.describe(),
                         "sandboxed": self.sandboxed, "invite_required": bool(self.invite)},
        }

    def bot_detail(self, bot_id: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM bots WHERE id = ?", (bot_id,)).fetchone()
            if not row:
                return None
            h2h = [dict(r) for r in self.db.execute(H2H_SQL, {"id": bot_id})]
            recent = [self._match(r) for r in self.db.execute(
                MATCH_SELECT + " WHERE m.a = ? OR m.b = ? ORDER BY m.id DESC LIMIT 20", (bot_id, bot_id))]
        for h in h2h:
            h["opp_rating"] = round(h["opp_rating"])
            h["bb100"] = round(h["chips"] / h["hands"] / self.rules.big_blind * 100, 1) if h["hands"] else None
        return {"bot": self._public(row), "h2h": h2h, "recent": recent}

    def starter_zip(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for folder in ("pokerlib", "arena"):
                for p in sorted((ROOT / folder).glob("*.py")):
                    z.write(p, f"poker-bot-starter/{folder}/{p.name}")
            z.write(BOTS_DIR / "_template.py", "poker-bot-starter/bots/my_bot.py")
            for name in STARTER_BOTS:
                z.write(BOTS_DIR / f"{name}.py", f"poker-bot-starter/bots/{name}.py")
            z.writestr("poker-bot-starter/requirements.txt", "numpy\n")
            z.writestr("poker-bot-starter/README.md", self._starter_readme())
        return buf.getvalue()

    def _starter_readme(self) -> str:
        return f"""# Poker bot starter kit

1. `pip install numpy` (Python 3.10 or newer).
2. Write your bot in `bots/my_bot.py`. The top of that file explains every field of the game state.
3. Test it: `python3 -m arena check my_bot` (illegal moves, crashes, speed).
4. Play it: `python3 -m arena match my_bot rock`
5. Upload `bots/my_bot.py` on the ladder page.

Game: heads-up short-deck (6 to A) 5-card Pot-Limit Omaha. {self.rules.describe()}
Ladder matches: {self.hands} hands, every deal played from both seats, {self.timeout:g}s per decision,
{self.time_bank:g}s of total thinking per half. Bots get Python, numpy and pokerlib; no internet,
no files outside their own folder.
"""

    # ----- matches -----
    def start(self) -> None:
        self.threads = [threading.Thread(target=self._worker, name=f"ladder-{i}", daemon=True)
                        for i in range(self.workers)]
        for t in self.threads:
            t.start()

    def shutdown(self, wait: float = 30.0) -> None:
        """Stop scheduling and let matches in progress finish (up to `wait` seconds)."""
        self.stop.set()
        deadline = time.time() + wait
        for t in getattr(self, "threads", []):
            t.join(max(0.0, deadline - time.time()))

    def _pick(self):
        with self.lock:
            bots = [dict(r) for r in self.db.execute(
                "SELECT id, name, path, rating, matches FROM bots WHERE active = 1")]
            if len(bots) < 2:
                return None
            busy = {i for pair in self.busy_pairs for i in pair}
            a = min(bots, key=lambda b: b["matches"] + (4 if b["id"] in busy else 0) + random.random())
            options = [b for b in bots if b["id"] != a["id"]
                       and frozenset((a["id"], b["id"])) not in self.busy_pairs]
            if not options:
                return None
            b = min(options, key=lambda o: (self.h2h.get(frozenset((a["id"], o["id"])), 0),
                                            abs(a["rating"] - o["rating"]) / 150 + random.random()))
            key = frozenset((a["id"], b["id"]))
            self.busy_pairs.add(key)
            self.running[key] = {"a": a["name"], "b": b["name"], "started": time.time()}
            return a, b, key

    def _worker(self) -> None:
        while not self.stop.is_set():
            picked = self._pick()
            if not picked:
                self.stop.wait(2.0)
                continue
            a, b, key = picked
            try:
                out = self.data / "matches" / f"{int(time.time() * 1000)}_{a['id']}_{b['id']}"
                r = run_match(BotSpec("file", a["path"], a["name"], self.sandboxed),
                              BotSpec("file", b["path"], b["name"], self.sandboxed),
                              self.rules, hands=self.hands, timeout=self.timeout, init_timeout=30.0,
                              out_dir=out, time_bank=self.time_bank)
                self._record(a["id"], b["id"], r, out)
            except Exception:
                traceback.print_exc()
                self.stop.wait(3.0)
            finally:
                with self.lock:
                    self.busy_pairs.discard(key)
                    self.running.pop(key, None)

    def _record(self, a_id: int, b_id: int, r: dict, out: Path) -> None:
        net = r["net_a"]
        score = 1.0 if net > 0 else 0.0 if net < 0 else 0.5
        t = r["t"] if math.isfinite(r["t"]) else None
        with self.lock:
            A = self.db.execute("SELECT * FROM bots WHERE id = ?", (a_id,)).fetchone()
            B = self.db.execute("SELECT * FROM bots WHERE id = ?", (b_id,)).fetchone()
            expected = 1 / (1 + 10 ** ((B["rating"] - A["rating"]) / 400))
            da = (32 if A["matches"] < 20 else 16) * (score - expected)
            db_ = (32 if B["matches"] < 20 else 16) * ((1 - score) - (1 - expected))
            for row, delta, chips, s in ((A, da, net, score), (B, db_, -net, 1 - score)):
                h = r["health"][row["name"]]
                crashed = int(bool(h["crashed"]))
                streak = row["crash_streak"] + 1 if crashed else 0
                active, note = row["active"], row["note"]
                if streak >= AUTO_RETIRE_AFTER and row["kind"] != "house":
                    active, note = 0, f"retired automatically after crashing in {streak} matches in a row"
                self.db.execute(
                    "UPDATE bots SET rating = rating + ?, matches = matches + 1, wins = wins + ?, losses = losses + ?, "
                    "draws = draws + ?, chips = chips + ?, hands = hands + ?, timeouts = timeouts + ?, "
                    "errors = errors + ?, illegal = illegal + ?, crashes = crashes + ?, crash_streak = ?, "
                    "last_problem = COALESCE(?, last_problem), active = ?, note = ? WHERE id = ?",
                    (delta, int(s == 1), int(s == 0), int(s == 0.5), chips, r["hands"], h["timeouts"],
                     h["errors"], h["illegal"], crashed, streak, _scrub(h["problems"][0]) if h["problems"] else None,
                     active, note, row["id"]))
            self.db.execute(
                "INSERT INTO matches (a, b, net_a, hands, t, seed, rating_a, rating_b, delta_a, health, seconds, dir, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (a_id, b_id, net, r["hands"], t, r["seed"], A["rating"] + da, B["rating"] + db_, da,
                 json.dumps(r["health"]), r["seconds"], str(out), time.time()))
            self.db.commit()
            k = frozenset((a_id, b_id))
            self.h2h[k] = self.h2h.get(k, 0) + 1
            self.recorded += 1
            prune = self.recorded % 50 == 0
        if prune:
            dirs = sorted(p for p in (self.data / "matches").iterdir() if p.is_dir())
            for p in dirs[:-KEEP_MATCH_DIRS]:
                shutil.rmtree(p, ignore_errors=True)
