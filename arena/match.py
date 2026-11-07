"""Duplicate matches and round robins.

What makes it fair:
- Every deal is played twice: once as dealt, then again with the bots in each other's seats
  (same cards, same position). Card luck cancels out, so the result measures decisions.
- Each half runs in brand-new bot processes, so nothing seen in one half carries into the other.
- Both bots get the same time limit per decision. Slow, crashing or illegal answers become
  check (when free) or fold, and are reported. Three timeouts in a row stop the bot.
- A bot only sees its own cards, the board, and the opponent's cards at showdown.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import secrets
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from pokerlib import Rules, get_evaluator

from .engine import Hand, IllegalAction
from .transport import fetch_http_name, open_bot

ROOT = Path(__file__).resolve().parent.parent
BOTS_DIR = ROOT / "bots"
RUNS_DIR = ROOT / "runs"
PRACTICE_BOTS = ("calling_station", "maniac", "random_bot", "rock", "equity_bot")
MAX_TIMEOUTS_IN_A_ROW = 3


@dataclass
class BotSpec:
    kind: str    # "file" or "http"
    target: str  # file path or URL
    name: str
    sandbox: bool = False  # run inside the macOS sandbox (code we did not write)


def resolve_bot(spec: str) -> BotSpec:
    if spec.startswith(("http://", "https://")):
        url = spec.rstrip("/")
        return BotSpec("http", url, fetch_http_name(url) or re.sub(r"^https?://", "", url))
    p = Path(spec)
    if not p.exists() and p.suffix != ".py":
        p = BOTS_DIR / f"{spec}.py"
    if not p.exists():
        raise SystemExit(f"Bot not found: {spec} (use a file path, a name from bots/, or an http:// URL)")
    return BotSpec("file", str(p.resolve()), p.stem)


def default_bots() -> list[BotSpec]:
    return [resolve_bot(str(p)) for p in sorted(BOTS_DIR.glob("*.py")) if not p.name.startswith("_")]


def unique_names(specs: list[BotSpec]) -> list[BotSpec]:
    seen: dict[str, int] = {}
    out = []
    for s in specs:
        seen[s.name] = seen.get(s.name, 0) + 1
        out.append(s if seen[s.name] == 1 else BotSpec(s.kind, s.target, f"{s.name}#{seen[s.name]}", s.sandbox))
    return out


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:40]


@dataclass
class Health:
    timeouts: int = 0
    errors: int = 0
    illegal: int = 0
    crashed: bool = False
    problems: list = field(default_factory=list)
    times: list = field(default_factory=list)

    def note(self, text: str) -> None:
        if len(self.problems) < 4:
            self.problems.append(text)

    def merge(self, other: "Health") -> "Health":
        return Health(self.timeouts + other.timeouts, self.errors + other.errors,
                      self.illegal + other.illegal, self.crashed or other.crashed,
                      (self.problems + other.problems)[:4], self.times + other.times)

    def summary(self) -> dict:
        t = sorted(self.times)
        return {
            "timeouts": self.timeouts, "errors": self.errors, "illegal": self.illegal,
            "crashed": self.crashed, "decisions": len(t),
            "avg_ms": sum(t) / len(t) * 1000 if t else 0.0,
            "p95_ms": t[min(len(t) - 1, int(0.95 * len(t)))] * 1000 if t else 0.0,
            "max_ms": t[-1] * 1000 if t else 0.0,
            "problems": self.problems,
        }


def make_decks(seed: int, n: int) -> list[list[int]]:
    decks = []
    for i in range(n):
        d = list(range(36))
        random.Random(f"{seed}:{i}").shuffle(d)
        decks.append(d)
    return decks


def _record_problem(h: Health, status: str, bot, where: str, reply) -> None:
    if status == "timeout":
        h.timeouts += 1
        h.note(f"{where}: no answer in time")
    elif status == "error":
        h.errors += 1
        h.note(f"{where}: {(reply or {}).get('error', 'error')}")
    elif status == "crashed" and not h.crashed:
        h.crashed = True
        h.note(f"{where}: bot stopped ({bot.fatal or 'process exited'})")


def _play_half(seats: list[BotSpec], decks, rules: Rules, timeout: float, init_timeout: float,
               out_dir: Path | None, tag: str, on_hand, save_hands: bool, time_bank: float | None = None):
    ev = get_evaluator(rules)
    health = [Health(), Health()]
    in_a_row = [0, 0]
    thinking = [0.0, 0.0]
    bots = []
    nets: list[int] = []
    hand_log = None

    def ask(p: int, msg: dict, limit: float, where: str):
        reply, status = bots[p].request(msg, limit)
        if status == "ok":
            in_a_row[p] = 0
            return reply, status
        _record_problem(health[p], status, bots[p], where, reply)
        if status == "timeout":
            in_a_row[p] += 1
            if in_a_row[p] >= MAX_TIMEOUTS_IN_A_ROW and not bots[p].dead:
                bots[p].kill(f"{MAX_TIMEOUTS_IN_A_ROW} timeouts in a row")
                _record_problem(health[p], "crashed", bots[p], where, None)
        return reply, status

    try:
        for spec in seats:
            log_path = out_dir / f"{safe_name(spec.name)}_{tag}.log" if out_dir else None
            bots.append(open_bot(spec.kind, spec.target, log_path, spec.sandbox))
        if save_hands and out_dir:
            hand_log = open(out_dir / f"hands_{tag}.jsonl", "w", encoding="utf-8")
        info = {"rules": rules.to_dict(), "hands": len(decks), "session": tag, "time_limit": timeout}
        for p in (0, 1):
            ask(p, {"type": "match_start", "info": info}, init_timeout, "match start")
        for i, deck in enumerate(decks):
            hand = Hand(rules, ev, deck, button=i % 2, hand_id=i)
            while not hand.finished:
                s = hand.to_act
                state = hand.view(s)
                if time_bank is not None and thinking[s] > time_bank and not bots[s].dead:
                    bots[s].kill(f"used up its {time_bank:g}s of thinking time for this half")
                    _record_problem(health[s], "crashed", bots[s], f"hand {i}", None)
                t0 = time.perf_counter()
                reply, status = ask(s, {"type": "act", "state": state}, timeout, f"hand {i} {state['street']}")
                elapsed = time.perf_counter() - t0
                thinking[s] += elapsed
                health[s].times.append(elapsed)
                if status == "ok":
                    try:
                        hand.act(s, reply.get("action"), reply.get("amount"))
                        continue
                    except IllegalAction as e:
                        health[s].illegal += 1
                        health[s].note(f"hand {i} {state['street']}: {e}")
                hand.act(s, "check" if "check" in state["legal"] else "fold")
            for p in (0, 1):
                ask(p, {"type": "hand_over", "result": hand.result(p)}, timeout, f"hand {i} hand_over")
            nets.append(hand.net[0])
            if hand_log:
                hand_log.write(json.dumps(hand.record(seats[0].name, seats[1].name)) + "\n")
            if on_hand:
                on_hand()
    finally:
        for b in bots:
            try:
                b.close()
            except Exception:
                pass
        if hand_log:
            hand_log.close()
    return nets, health


def run_match(a: BotSpec, b: BotSpec, rules: Rules, hands: int = 2000, seed: int | None = None,
              timeout: float = 2.0, init_timeout: float = 30.0, out_dir: Path | None = None,
              save_hands: bool = False, progress: bool = False, time_bank: float | None = None) -> dict:
    """Duplicate match. Returns a summary dict from a's point of view."""
    a, b = unique_names([a, b])
    deals = max(1, hands // 2)
    seed = secrets.randbelow(10 ** 9) if seed is None else seed
    decks = make_decks(seed, deals)
    get_evaluator(rules)  # build the hand table once, before any bot process starts
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    done = [0]
    lock = threading.Lock()

    def on_hand():
        with lock:
            done[0] += 1

    started = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_play_half, [a, b], decks, rules, timeout, init_timeout, out_dir, "half1", on_hand,
                         save_hands, time_bank)
        f2 = pool.submit(_play_half, [b, a], decks, rules, timeout, init_timeout, out_dir, "half2", on_hand,
                         save_hands, time_bank)
        while progress and not (f1.done() and f2.done()):
            sys.stderr.write(f"\r  {a.name} vs {b.name}: {done[0]:,}/{2 * deals:,} hands")
            sys.stderr.flush()
            time.sleep(0.3)
        nets1, (ha1, hb1) = f1.result()
        nets2, (hb2, ha2) = f2.result()
    if progress:
        sys.stderr.write("\r" + " " * 78 + "\r")

    a_half1 = nets1                   # a sat in seat 0
    a_half2 = [-x for x in nets2]     # b sat in seat 0, so a's result is the negative
    pairs = [x + y for x, y in zip(a_half1, a_half2)]
    total = sum(pairs)
    sd = statistics.stdev(pairs) if len(pairs) > 1 else 0.0
    se = sd * math.sqrt(len(pairs))
    t = total / se if se > 0 else (0.0 if total == 0 else math.copysign(math.inf, total))
    played = 2 * len(pairs)
    result = {
        "a": a.name, "b": b.name, "hands": played, "deals": len(pairs), "seed": seed,
        "rules": rules.to_dict(), "time_limit": timeout,
        "net_a": total, "bb_per_100_a": total / played / rules.big_blind * 100,
        "se": se, "t": t, "ci95": [total - 1.96 * se, total + 1.96 * se],
        "halves_a": [sum(a_half1), sum(a_half2)],
        "health": {a.name: ha1.merge(ha2).summary(), b.name: hb1.merge(hb2).summary()},
        "seconds": round(time.time() - started, 1),
        "out_dir": str(out_dir) if out_dir else None,
    }
    if out_dir:
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def _verdict(t: float) -> str:
    if abs(t) >= 2.58:
        return "very likely a real edge"
    if abs(t) >= 1.96:
        return "probably a real edge"
    return "could be noise, play more hands"


def _health_table(health: dict, time_limit: float) -> list[str]:
    rows = [f"{'Bot health':<20}{'timeouts':>9}{'errors':>8}{'illegal':>9}{'crashed':>9}"
            f"{'avg ms':>9}{'p95 ms':>9}{'max ms':>9}"]
    notes = []
    for name, h in health.items():
        rows.append(f"{name[:19]:<20}{h['timeouts']:>9}{h['errors']:>8}{h['illegal']:>9}"
                    f"{'yes' if h['crashed'] else 'no':>9}{h['avg_ms']:>9.1f}{h['p95_ms']:>9.1f}{h['max_ms']:>9.1f}")
        for p in h["problems"]:
            notes.append(f"  {name}: {p}")
        if h["max_ms"] > 0.5 * time_limit * 1000:
            notes.append(f"  {name}: slowest decision used over half the {time_limit:g}s limit")
    return rows + notes


def format_match(r: dict) -> str:
    rules = Rules.from_dict(r["rules"])
    a, b, net = r["a"], r["b"], r["net_a"]
    lines = [
        f"{a} vs {b}: {r['hands']:,} hands ({r['deals']:,} deals, each played from both seats), "
        f"seed {r['seed']}, {r['seconds']}s",
        "Rules: " + rules.describe().replace("\n", "\n       "),
        f"Time limit: {r['time_limit']:g}s per decision",
        "",
    ]
    if net == 0:
        lines.append("Result: dead even")
    else:
        winner = a if net > 0 else b
        lines.append(f"Result: {winner} wins {abs(net):,} chips "
                     f"({abs(r['bb_per_100_a']):.1f} big blinds per 100 hands)")
        lo, hi = r["ci95"]
        lines.append(f"        95% range for {a}: {lo:+,.0f} to {hi:+,.0f} chips, t = {r['t']:.2f} ({_verdict(r['t'])})")
    h1, h2 = r["halves_a"]
    lines.append(f"        {a} made {h1:+,} in the first half and {h2:+,} after swapping seats (same cards)")
    lines.append("")
    lines += _health_table(r["health"], r["time_limit"])
    if r.get("out_dir"):
        lines.append(f"Logs: {r['out_dir']}")
    return "\n".join(lines)


def run_tournament(specs: list[BotSpec], rules: Rules, hands: int = 1000, seed: int | None = None,
                   timeout: float = 2.0, init_timeout: float = 30.0, jobs: int = 4,
                   out_dir: Path | None = None, save_hands: bool = False) -> dict:
    specs = unique_names(specs)
    if len(specs) < 2:
        raise SystemExit("A tournament needs at least two bots.")
    seed = secrets.randbelow(10 ** 9) if seed is None else seed
    get_evaluator(rules)
    pairs = list(combinations(range(len(specs)), 2))
    print(f"Round robin: {len(specs)} bots, {len(pairs)} matches of {2 * max(1, hands // 2):,} hands, "
          f"same cards in every match (seed {seed})")
    print("Rules: " + rules.describe().replace("\n", "\n       "))
    results: dict = {}

    def job(i: int, j: int):
        sub = out_dir / f"{safe_name(specs[i].name)}_vs_{safe_name(specs[j].name)}" if out_dir else None
        return i, j, run_match(specs[i], specs[j], rules, hands, seed, timeout, init_timeout, sub, save_hands)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(job, i, j) for i, j in pairs]
        for k, f in enumerate(as_completed(futures), 1):
            i, j, r = f.result()
            results[(i, j)] = r
            print(f"  [{k}/{len(pairs)}] {specs[i].name} {r['net_a']:+,} vs {specs[j].name}  (t = {r['t']:.1f})")

    names = [s.name for s in specs]
    chips = {n: 0 for n in names}
    hands_played = {n: 0 for n in names}
    won = {n: 0 for n in names}
    lost = {n: 0 for n in names}
    issues = {n: [] for n in names}
    grid = {n: {m: None for m in names} for n in names}
    for (i, j), r in results.items():
        a, b, net = names[i], names[j], r["net_a"]
        chips[a] += net
        chips[b] -= net
        hands_played[a] += r["hands"]
        hands_played[b] += r["hands"]
        grid[a][b], grid[b][a] = net, -net
        if net > 0:
            won[a] += 1
            lost[b] += 1
        elif net < 0:
            won[b] += 1
            lost[a] += 1
        for n in (a, b):
            h = r["health"][n]
            if h["timeouts"] or h["errors"] or h["illegal"] or h["crashed"]:
                issues[n].append(f"vs {b if n == a else a}: {h['timeouts']} timeouts, {h['errors']} errors, "
                                 f"{h['illegal']} illegal{', crashed' if h['crashed'] else ''}")

    order = sorted(names, key=lambda n: chips[n], reverse=True)
    print()
    print(f"{'Rank':<6}{'Bot':<22}{'Chips':>10}{'bb/100':>9}{'W-L':>8}  Health")
    for rank, n in enumerate(order, 1):
        bb100 = chips[n] / max(1, hands_played[n]) / rules.big_blind * 100
        print(f"{rank:<6}{n[:21]:<22}{chips[n]:>+10,}{bb100:>+9.1f}{f'{won[n]}-{lost[n]}':>8}  "
              f"{'ok' if not issues[n] else 'PROBLEMS'}")
    short = {n: n[:9] for n in names}
    print()
    print("Chips won by row bot against column bot:")
    print(" " * 11 + "".join(f"{short[m]:>10}" for m in order))
    for n in order:
        cells = "".join(f"{'':>10}" if grid[n][m] is None else f"{grid[n][m]:>+10,}" for m in order)
        print(f"{short[n]:<11}{cells}")
    for n in order:
        for line in issues[n]:
            print(f"  {n}: {line}")
    summary = {"seed": seed, "rules": rules.to_dict(), "standings": [
        {"bot": n, "chips": chips[n], "won": won[n], "lost": lost[n]} for n in order],
        "matches": [r for r in results.values()]}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nLogs: {out_dir}")
    return summary


def run_check(spec: BotSpec, rules: Rules, hands: int = 200, timeout: float = 2.0, seed: int | None = None,
              out_dir: Path | None = None) -> int:
    """Plays the bot against each practice bot and reports anything that would cost it in the real tournament."""
    print(f"Checking {spec.name}: {2 * max(1, hands // 2)} hands against each practice bot, "
          f"{timeout:g}s per decision")
    problems = 0
    slowest = 0.0
    for opp_name in PRACTICE_BOTS:
        if not (BOTS_DIR / f"{opp_name}.py").exists():
            continue
        opp = resolve_bot(opp_name)
        if opp.kind == spec.kind and Path(opp.target) == Path(spec.target):
            continue
        sub = out_dir / f"vs_{opp.name}" if out_dir else None
        r = run_match(spec, opp, rules, hands, seed, timeout, 30.0, sub)
        mine = r["health"][r["a"]]
        bad = mine["timeouts"] + mine["errors"] + mine["illegal"] + (1 if mine["crashed"] else 0)
        problems += bad
        slowest = max(slowest, mine["max_ms"])
        print(f"  vs {opp.name:<16}{r['net_a']:>+8,} chips ({r['bb_per_100_a']:>+7.1f} bb/100)  "
              f"{'ok' if not bad else 'PROBLEMS'}")
        for p in mine["problems"]:
            print(f"      {p}")
    print(f"Slowest decision: {slowest:.0f} ms of the {timeout * 1000:.0f} ms limit")
    if problems:
        print(f"Found {problems} problem(s). Bot logs are in {out_dir}" if out_dir else f"Found {problems} problem(s).")
        return 1
    print("No errors, illegal moves, timeouts or crashes.")
    return 0
