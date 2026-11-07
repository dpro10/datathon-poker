"""The ladder: stores official-format bots, plays official matches between them nonstop,
keeps Elo ratings from the official verdicts (win 1, chop 0.5, loss 0)."""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import secrets
import shutil
import sqlite3
import tempfile
import threading
import time
import traceback
from pathlib import Path

from arena import sandbox

from . import official

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ladder_data"
BASELINES = ("random", "calling_station", "always_half", "always_pot")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,23}$")
MAX_CODE_BYTES = 300_000
KEEP_MATCH_DIRS = 400
AUTO_RETIRE_AFTER = 3  # matches in a row with a forfeit
SMOKE_DEALS = 40

SCHEMA = """
CREATE TABLE IF NOT EXISTS bots (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  author TEXT NOT NULL,
  kind TEXT NOT NULL,              -- house, local, upload
  path TEXT NOT NULL,
  owner_key_hash TEXT,
  created_at REAL NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  note TEXT,
  rating REAL NOT NULL DEFAULT 1500,
  matches INTEGER NOT NULL DEFAULT 0,
  wins INTEGER NOT NULL DEFAULT 0,
  chops INTEGER NOT NULL DEFAULT 0,
  losses INTEGER NOT NULL DEFAULT 0,
  points REAL NOT NULL DEFAULT 0,
  bb_sum REAL NOT NULL DEFAULT 0,  -- sum over matches of bb won per deal
  errors INTEGER NOT NULL DEFAULT 0,
  forfeits INTEGER NOT NULL DEFAULT 0,
  bank_outs INTEGER NOT NULL DEFAULT 0,
  think_max REAL NOT NULL DEFAULT 0,
  forfeit_streak INTEGER NOT NULL DEFAULT 0,
  last_problem TEXT
);
CREATE TABLE IF NOT EXISTS matches (
  id INTEGER PRIMARY KEY,
  a INTEGER NOT NULL REFERENCES bots(id),
  b INTEGER NOT NULL REFERENCES bots(id),
  verdict TEXT NOT NULL,
  points_a REAL NOT NULL,
  points_b REAL NOT NULL,
  mean_bb REAL NOT NULL,           -- a's bb per deal
  ci_low REAL, ci_high REAL,
  deals INTEGER NOT NULL,
  delta_a REAL, delta_b REAL,
  stats TEXT,
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
  SUM(CASE WHEN m.a = :id THEN m.points_a ELSE m.points_b END) AS points,
  SUM(CASE WHEN (m.a = :id AND m.points_a = 1) OR (m.b = :id AND m.points_b = 1) THEN 1 ELSE 0 END) AS wins,
  SUM(CASE WHEN m.points_a = 0.5 THEN 1 ELSE 0 END) AS chops,
  SUM(CASE WHEN m.a = :id THEN m.mean_bb ELSE -m.mean_bb END) AS bb_sum
FROM matches m JOIN bots o ON o.id = CASE WHEN m.a = :id THEN m.b ELSE m.a END
WHERE m.a = :id OR m.b = :id
GROUP BY o.id ORDER BY points * 1.0 / COUNT(*) DESC, bb_sum DESC
"""


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _scrub(text: str) -> str:
    """Remove this machine's folder paths (username, temp folders) from text shown on the site."""
    text = re.sub(r'File "[^"]*/(agent|bot)\.py"', r'File "\1.py"', text)
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
    def __init__(self, data_dir: Path = DATA_DIR, invite: str = "", deals: int = 1000, workers: int = 2):
        self.data = Path(data_dir)
        (self.data / "bots").mkdir(parents=True, exist_ok=True)
        (self.data / "matches").mkdir(parents=True, exist_ok=True)
        self.invite, self.deals, self.workers = invite, deals, workers
        self.sandboxed = sandbox.available()
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.data / "ladder_v2.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.running: dict = {}
        self.busy_pairs: set = set()
        self.challenges: collections.deque = collections.deque()  # matches people asked for, played first
        self.h2h: dict = {}
        for r in self.db.execute("SELECT a, b, COUNT(*) AS n FROM matches GROUP BY a, b"):
            k = frozenset((r["a"], r["b"]))
            self.h2h[k] = self.h2h.get(k, 0) + r["n"]
        self.smoke_slots = threading.Semaphore(2)
        self.stop = threading.Event()
        self.threads: list = []
        self.recorded = 0

    # ----- bots -----
    def ensure_house_bots(self) -> None:
        with self.lock:
            for name in BASELINES:
                path = official.OFFICIAL / "baseline-bots" / f"baseline_{name}.py"
                row = self.db.execute("SELECT id FROM bots WHERE name = ?", (name,)).fetchone()
                if row is None:
                    self.db.execute("INSERT INTO bots (name, author, kind, path, created_at) "
                                    "VALUES (?, 'organizers', 'house', ?, ?)", (name, str(path), time.time()))
                else:
                    self.db.execute("UPDATE bots SET path = ? WHERE id = ?", (str(path), row["id"]))
            self.db.commit()

    def add_local(self, path: str, name: str, author: str) -> int:
        """Host only: enter a bot from a folder on this machine (it may import its own modules)."""
        p = Path(path).resolve()
        if not p.is_file():
            raise SubmitError(f"No such file: {path}")
        self._check_name(name)
        self._smoke(p, name)
        with self.lock:
            cur = self.db.execute("INSERT INTO bots (name, author, kind, path, created_at) VALUES (?, ?, 'local', ?, ?)",
                                  (name, author, str(p), time.time()))
            self.db.commit()
            return cur.lastrowid

    def _check_name(self, name: str) -> None:
        if not NAME_RE.match(name):
            raise SubmitError("Bot name: up to 24 letters, digits, dots, dashes or underscores (no spaces).")
        with self.lock:
            if self.db.execute("SELECT 1 FROM bots WHERE name = ?", (name,)).fetchone():
                raise SubmitError(f"The name {name} is taken. Try {name}-2.")

    def _smoke(self, path: Path, name: str) -> None:
        """A short official half against calling_station. Any error or forfeit rejects the bot."""
        station = official.OFFICIAL / "baseline-bots" / "baseline_calling_station.py"
        test_dir = Path(tempfile.mkdtemp(prefix="smoke_"))
        try:
            with self.smoke_slots:
                half = official.play_half((str(path), name), (str(station), "calling_station"),
                                          official.je.new_match_seed(), 1, SMOKE_DEALS, test_dir, self.sandboxed)
            s = half.stats_agent1
            if s.errors or s.forfeited:
                details = _scrub("\n\n".join(s.error_messages[:3]))
                log = _tail(sorted(test_dir.glob(f"{name}_*.log")), 30)
                if log:
                    details += "\n\nWhat your bot printed:\n" + log
                raise SubmitError(f"Your bot made {s.errors} error(s) in a {SMOKE_DEALS}-hand test against "
                                  f"calling_station{' and forfeited' if s.forfeited else ''}.", details)
        except SubmitError:
            raise
        except Exception as e:
            traceback.print_exc()
            raise SubmitError(f"Testing failed: {type(e).__name__}")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def submit(self, name: str, author: str, code: str, invite: str) -> tuple[int, str]:
        name, author = name.strip(), " ".join(author.split())
        if self.invite and not secrets.compare_digest(invite.strip().encode(), self.invite.encode()):
            raise SubmitError("Wrong invite code.")
        self._check_name(name)
        if not 1 <= len(author) <= 30:
            raise SubmitError("Add your name (up to 30 characters).")
        if not code.strip():
            raise SubmitError("The file is empty.")
        if len(code.encode()) > MAX_CODE_BYTES:
            raise SubmitError("The file is over 300 KB.")
        try:
            compile(code, "agent.py", "exec")
        except (SyntaxError, ValueError) as e:
            line = getattr(e, "lineno", None)
            raise SubmitError(f"Python syntax error{f' on line {line}' if line else ''}: {getattr(e, 'msg', e)}")
        if not re.search(r"^def\s+choose_action\b", code, re.M):
            raise SubmitError("This doesn't look like the official agent.py (no choose_action function). "
                              "Start from github.com/tamu-datathon-org/heads-up-bundle.")
        staging = Path(tempfile.mkdtemp(prefix="upload_", dir=self.data / "bots"))
        path = staging / "agent.py"
        path.write_text(code, encoding="utf-8")
        try:
            self._smoke(path, name)
        except SubmitError:
            shutil.rmtree(staging, ignore_errors=True)
            raise
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
            self.db.execute("UPDATE bots SET path = ? WHERE id = ?", (str(final / "agent.py"), bot_id))
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

    def challenge(self, a_id: int, b_id: int, invite: str) -> tuple[bool, str]:
        """Queue a match between two bots; queued matches are played before scheduled ones."""
        if self.invite and not secrets.compare_digest(invite.strip().encode(), self.invite.encode()):
            return False, "Wrong invite code. Enter it in the Enter a bot form first."
        if a_id == b_id:
            return False, "Pick a different opponent."
        with self.lock:
            rows = {r["id"]: r for r in self.db.execute(
                "SELECT id, name, active FROM bots WHERE id IN (?, ?)", (a_id, b_id))}
            if len(rows) != 2 or not all(r["active"] for r in rows.values()):
                return False, "Both bots must be on the ladder (not retired)."
            pair = frozenset((a_id, b_id))
            if any(frozenset((c["a"], c["b"])) == pair for c in self.challenges):
                return False, "That match is already queued."
            if len(self.challenges) >= 12:
                return False, "The queue is full. Try again in a minute."
            self.challenges.append({"a": a_id, "b": b_id, "a_name": rows[a_id]["name"],
                                    "b_name": rows[b_id]["name"], "at": time.time()})
            return True, f"Queued {rows[a_id]['name']} vs {rows[b_id]['name']} (position {len(self.challenges)})."

    def set_active(self, name: str, active: bool) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE bots SET active = ?, forfeit_streak = 0, note = ? WHERE name = ?",
                                  (int(active), None if active else "retired by the host", name))
            self.db.commit()
            return cur.rowcount > 0

    def owner_log(self, bot_id: int, key: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM bots WHERE id = ?", (bot_id,)).fetchone()
            if not self._owns(row, key):
                return None
            m = self.db.execute("SELECT * FROM matches WHERE a = ? OR b = ? ORDER BY id DESC LIMIT 1",
                                (bot_id, bot_id)).fetchone()
        if not m:
            return "No matches yet."
        side = "agent1" if m["a"] == bot_id else "agent2"
        errors = [msg for half in json.loads(m["stats"] or "{}").get(side, []) for msg in half.get("error_messages", [])]
        text = _tail(sorted(Path(m["dir"]).glob(f"{row['name']}_half*.log")), 80) if m["dir"] else ""
        parts = []
        if errors:
            parts.append("Errors the judge recorded:\n" + _scrub("\n\n".join(errors[:5])))
        parts.append(text or "Your bot printed nothing in its latest match (print() output shows up here).")
        return "\n\n".join(parts)

    # ----- views -----
    @staticmethod
    def _public(r) -> dict:
        n = r["matches"]
        return {
            "id": r["id"], "name": r["name"], "author": r["author"], "house": r["kind"] == "house",
            "kind": r["kind"], "active": bool(r["active"]), "note": r["note"], "rating": round(r["rating"]),
            "matches": n, "wins": r["wins"], "chops": r["chops"], "losses": r["losses"],
            "points": r["points"], "score": round(r["points"] / n, 3) if n else None,
            "bb_deal": round(r["bb_sum"] / n, 2) if n else None,
            "errors": r["errors"], "forfeits": r["forfeits"], "bank_outs": r["bank_outs"],
            "think_max": round(r["think_max"], 2), "last_problem": r["last_problem"], "created_at": r["created_at"],
        }

    @staticmethod
    def _match(r) -> dict:
        return {
            "id": r["id"], "a": r["a"], "b": r["b"], "a_name": r["a_name"], "b_name": r["b_name"],
            "verdict": r["verdict"], "points_a": r["points_a"], "points_b": r["points_b"],
            "mean_bb": round(r["mean_bb"], 2), "ci": [r["ci_low"], r["ci_high"]], "deals": r["deals"],
            "seconds": r["seconds"], "created_at": r["created_at"],
        }

    def state(self) -> dict:
        with self.lock:
            bots = [self._public(r) for r in self.db.execute("SELECT * FROM bots ORDER BY active DESC, rating DESC")]
            recent = [self._match(r) for r in self.db.execute(MATCH_SELECT + " ORDER BY m.id DESC LIMIT 30")]
            total = self.db.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            running = [dict(v) for v in self.running.values()]
            queued = [{"a": c["a_name"], "b": c["b_name"]} for c in self.challenges]
        game = official.je
        return {
            "bots": bots, "matches": recent, "running": running, "queued": queued, "total_matches": total,
            "settings": {"deals": self.deals, "time_bank": game.TIME_BANK_SECONDS, "timeout": game.ACTION_TIMEOUT_SECONDS,
                         "error_limit": game.ERROR_LIMIT, "stack": __import__("game").STARTING_STACK,
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
            h["bb_deal"] = round(h["bb_sum"] / h["n"], 2) if h["n"] else None
            h["losses"] = h["n"] - h["wins"] - h["chops"]
        return {"bot": self._public(row), "h2h": h2h, "recent": recent}

    # ----- matches -----
    def start(self) -> None:
        self.threads = [threading.Thread(target=self._worker, name=f"ladder-{i}", daemon=True)
                        for i in range(self.workers)]
        for t in self.threads:
            t.start()

    def shutdown(self, wait: float = 30.0) -> None:
        self.stop.set()
        deadline = time.time() + wait
        for t in self.threads:
            t.join(max(0.0, deadline - time.time()))

    def _pick(self):
        with self.lock:
            bots = [dict(r) for r in self.db.execute(
                "SELECT id, name, path, rating, matches FROM bots WHERE active = 1")]
            by_id = {b["id"]: b for b in bots}
            for c in list(self.challenges):
                pair = frozenset((c["a"], c["b"]))
                if c["a"] not in by_id or c["b"] not in by_id:
                    self.challenges.remove(c)  # a bot was retired meanwhile
                    continue
                if pair in self.busy_pairs:
                    continue
                self.challenges.remove(c)
                a, b = by_id[c["a"]], by_id[c["b"]]
                self.busy_pairs.add(pair)
                self.running[pair] = {"a": a["name"], "b": b["name"], "started": time.time(), "requested": True}
                return a, b, pair
            if len(bots) < 2:
                return None
            busy = {i for pair in self.busy_pairs for i in pair}
            a = min(bots, key=lambda b: b["matches"] + (4 if b["id"] in busy else 0) + random.random())
            options = [b for b in bots if b["id"] != a["id"] and frozenset((a["id"], b["id"])) not in self.busy_pairs]
            if not options:
                return None
            b = min(options, key=lambda o: (self.h2h.get(frozenset((a["id"], o["id"])), 0),
                                            abs(a["rating"] - o["rating"]) / 150 + random.random()))
            if random.random() < 0.5:
                a, b = b, a
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
                m = official.play_match((a["path"], a["name"]), (b["path"], b["name"]), self.deals,
                                        out_dir=out, sandboxed=self.sandboxed)
                self._record(a["id"], b["id"], official.summarize(m), out)
            except Exception:
                traceback.print_exc()
                self.stop.wait(3.0)
            finally:
                with self.lock:
                    self.busy_pairs.discard(key)
                    self.running.pop(key, None)

    def _record(self, a_id: int, b_id: int, s: dict, out: Path) -> None:
        pa, pb = s["points"]["agent1"], s["points"]["agent2"]
        with self.lock:
            A = self.db.execute("SELECT * FROM bots WHERE id = ?", (a_id,)).fetchone()
            B = self.db.execute("SELECT * FROM bots WHERE id = ?", (b_id,)).fetchone()
            expected = 1 / (1 + 10 ** ((B["rating"] - A["rating"]) / 400))
            score = pa if pa + pb > 0 else 0.5  # both forfeited: no rating change
            da = (32 if A["matches"] < 20 else 16) * (score - expected)
            db_ = (32 if B["matches"] < 20 else 16) * ((1 - score) - (1 - expected))
            if pa + pb == 0:
                da = db_ = 0.0
            for row, side, delta, pts, bb in ((A, "agent1", da, pa, s["mean_bb"]), (B, "agent2", db_, pb, -s["mean_bb"])):
                halves = s["stats"][side]
                errors = sum(h["errors"] for h in halves)
                forfeit = any(h["forfeited"] for h in halves)
                bank_out = any(h["timed_out"] for h in halves)
                think = max(h.get("time_seconds", 0.0) for h in halves)
                problem = next((m for h in halves for m in h["error_messages"]), None)
                streak = row["forfeit_streak"] + 1 if forfeit else 0
                active, note = row["active"], row["note"]
                if streak >= AUTO_RETIRE_AFTER and row["kind"] != "house":
                    active, note = 0, f"retired automatically after forfeiting {streak} matches in a row"
                self.db.execute(
                    "UPDATE bots SET rating = rating + ?, matches = matches + 1, wins = wins + ?, chops = chops + ?, "
                    "losses = losses + ?, points = points + ?, bb_sum = bb_sum + ?, errors = errors + ?, "
                    "forfeits = forfeits + ?, bank_outs = bank_outs + ?, think_max = MAX(think_max, ?), "
                    "forfeit_streak = ?, last_problem = COALESCE(?, last_problem), active = ?, note = ? WHERE id = ?",
                    (delta, int(pts == 1), int(pts == 0.5), int(pts == 0), pts, bb, errors, int(forfeit),
                     int(bank_out), think, streak, _scrub(problem.splitlines()[0]) if problem else None,
                     active, note, row["id"]))
            self.db.execute(
                "INSERT INTO matches (a, b, verdict, points_a, points_b, mean_bb, ci_low, ci_high, deals, delta_a, "
                "delta_b, stats, seconds, dir, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (a_id, b_id, s["verdict"], pa, pb, s["mean_bb"], s["ci"][0], s["ci"][1], s["deals"], da, db_,
                 json.dumps(s["stats"]), s["seconds"], str(out), time.time()))
            self.db.commit()
            k = frozenset((a_id, b_id))
            self.h2h[k] = self.h2h.get(k, 0) + 1
            self.recorded += 1
            prune = self.recorded % 50 == 0
        if prune:
            dirs = sorted(p for p in (self.data / "matches").iterdir() if p.is_dir())
            for p in dirs[:-KEEP_MATCH_DIRS]:
                shutil.rmtree(p, ignore_errors=True)
