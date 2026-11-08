"""In-process duplicate-match harness that mirrors the official judge.

Bots are official-format modules (new_half / choose_action / on_hand_end), e.g.
agent.py itself or anything in dev/opponents/. For each half the bot module is
loaded fresh and gets a fresh memory dict, finished hands are delivered right
before the bot's next decision (and flushed at the end of the half), actions are
validated like the judge does, every call is timed against the 8 s bank, and the
match is scored with the judge's own combine_halves (95% CI on paired bb/deal).

    python dev/harness.py agent.py dev/opponents/tag.py --deals 1000 --seeds 1-8
"""
from __future__ import annotations

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # like the bot (sdbot/__init__.py)
    _os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import itertools
import os
import statistics
import sys
import time
import traceback
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import judge_engine as je  # noqa: E402
from game import (ERROR_LIMIT, TIME_BANK_SECONDS, Action, AgentConfig, Game,  # noqa: E402
                  HandResult, Observation, button_for_deal, default_action, make_deal)

_counter = itertools.count()
# Variance reduction for experiments (HARNESS_ALLIN_EV=1): a hand that ends in an all-in before
# the river is scored by its exact equity over the remaining board cards (sampled preflop)
# instead of the one runout dealt. Unbiased for expected winnings; off = the judge's scoring.
ALLIN_EV = os.environ.get("HARNESS_ALLIN_EV") == "1"


def allin_nets(game, res):
    """Expected nets of a hand that went all-in before the river (None if it did not)."""
    from game import BOARD_CARDS_VISIBLE
    if not res.went_to_showdown or not res.history:
        return None
    vis = BOARD_CARDS_VISIBLE[res.history[-1].street]
    if vis >= 5:
        return None
    import numpy as np
    from sdbot import evaluator as ev
    holes = [sorted(int(c) for c in game.deal.hole_cards[s]) for s in (0, 1)]
    board = [int(c) for c in game.deal.board[:vis]]
    rest = np.array(sorted(set(range(36)) - set(holes[0]) - set(holes[1]) - set(board)), dtype=np.int64)
    need = 5 - vis
    if need == 5:
        rng = np.random.default_rng(hash((game.hand_index, tuple(holes[0]), tuple(holes[1]))) & 0xFFFFFFFF)
        extra = np.sort(np.argsort(rng.random((3000, len(rest))), axis=1)[:, :5], axis=1)
        runs = rest[extra]
    else:
        idx = np.array(list(itertools.combinations(range(len(rest)), need)), dtype=np.int64)
        runs = np.concatenate([np.tile(board, (len(idx), 1)), rest[idx]], axis=1)
    tab = ev.board_tables(runs)
    s0 = tab[ev.hand_codes_one(holes[0])].max(axis=0)
    s1 = tab[ev.hand_codes_one(holes[1])].max(axis=0)
    p0 = float(np.mean(s0 > s1) + 0.5 * np.mean(s0 == s1))
    pot = game.pot
    n0 = pot * p0 - game.contributed[0]
    return (n0, -n0)
# Testing on a busy machine: a bigger bank keeps the bots' time governors from reacting to
# CPU contention, so strategy comparisons stay fair. Leave unset to mirror the judge.
BANK = float(os.environ.get("HARNESS_BANK", TIME_BANK_SECONDS))


def resolve(path: str) -> str:
    if os.path.isfile(path):
        return os.path.abspath(path)
    for cand in (os.path.join(HERE, "opponents", path), os.path.join(HERE, "opponents", path + ".py"),
                 os.path.join(ROOT, "baseline-bots", path + ".py")):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(path)


def load_module(path: str):
    path = resolve(path)
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    name = f"bot_{next(_counter)}_{os.path.splitext(os.path.basename(path))[0]}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Handle:
    """One bot for one half, with the judge's bookkeeping."""

    def __init__(self, path: str, config: AgentConfig, serialize: bool = True, time_bank: float = None):
        time_bank = BANK if time_bank is None else time_bank
        self.path = path
        self.config = config
        self.serialize = serialize
        self.time_bank = time_bank
        self.mod = load_module(path)
        self.memory: dict = {}
        self.pending: list = []
        self.time = 0.0
        self.max_call = 0.0
        self.calls = 0
        self.errors = 0
        self.forfeited = False
        self.timed_out = False
        self.first_error = None

    def _error(self, msg):
        self.errors += 1
        if self.first_error is None:
            self.first_error = msg
        if self.errors >= ERROR_LIMIT:
            self.forfeited = True

    def _timed(self, fn, *args):
        t = time.perf_counter()
        try:
            return fn(*args), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        finally:
            dt = time.perf_counter() - t
            self.time += dt
            self.max_call = max(self.max_call, dt)
            if dt > 1.0:
                self._error(f"call took {dt:.2f}s (> 1 s)")

    def start(self):
        _, err = self._timed(self.mod.new_half, self.config, self.memory)
        if err:
            self._error("new_half: " + err)
            self.forfeited = True

    def _deliver(self):
        results, self.pending = self.pending, []
        for r in results:
            if self.serialize:
                r = HandResult.from_dict(r.to_dict())
            _, err = self._timed(self.mod.on_hand_end, r, self.memory)
            if err:
                self._error("on_hand_end: " + err)

    def act(self, obs: Observation) -> Action:
        fallback = default_action(obs.to_call)
        if self.forfeited:
            return fallback
        if self.time >= self.time_bank:
            self.timed_out = True
            return fallback
        self.calls += 1
        self._deliver()
        o = Observation.from_dict(obs.to_dict()) if self.serialize else obs
        action, err = self._timed(self.mod.choose_action, o, self.memory)
        if err:
            self._error("choose_action: " + err)
            return fallback
        a = je.coerce_action(action.name if isinstance(action, Action) else action)
        if a is None or a not in obs.legal_actions:
            self._error(f"illegal action {action!r} (legal {[x.name for x in obs.legal_actions]})")
            return fallback
        return a

    def hand_end(self, result: HandResult):
        if not self.forfeited:
            self.pending.append(result)

    def flush(self):
        if not self.forfeited and self.time < self.time_bank:
            self._deliver()

    def stats(self) -> je.AgentStats:
        s = je.AgentStats(time_seconds=self.time, wall_seconds=self.time, calls=self.calls,
                          timed_out=self.timed_out, errors=self.errors, forfeited=self.forfeited,
                          error_messages=[self.first_error] if self.first_error else [])
        return s


def play_half(path1: str, path2: str, seed, half: int, deals: int, serialize=True, logs=None):
    config = AgentConfig(num_hands=deals, time_bank_seconds=BANK)
    a1 = Handle(path1, config, serialize)
    a2 = Handle(path2, config, serialize)
    a1_seat = 0 if half == 1 else 1
    bots = (a1, a2) if a1_seat == 0 else (a2, a1)
    for b in bots:
        b.start()
    nets = ([], [])
    for d in range(deals):
        button = button_for_deal(d)
        game = Game(make_deal(seed, d), button, hand_index=d)
        while not game.is_terminal:
            seat = game.to_act
            game.apply(bots[seat].act(game.observation(seat)))
        res = game.result()
        adj = allin_nets(game, res) if ALLIN_EV else None
        n0, n1 = adj if adj is not None else res.nets
        nets[0].append(n0)
        nets[1].append(n1)
        for seat in (0, 1):
            bots[seat].hand_end(game.hand_result(seat))
        if logs is not None:
            logs.append(je._hand_log(half, d, button, game))
    for b in bots:
        b.flush()
    hr = je.HalfResult(half=half, match_seed=str(seed), num_deals=deals, agent1_seat=a1_seat,
                       nets_agent1=nets[a1_seat], nets_agent2=nets[1 - a1_seat],
                       stats_agent1=a1.stats(), stats_agent2=a2.stats(),
                       agent1_name=os.path.basename(path1), agent2_name=os.path.basename(path2))
    hr.max_call = (a1.max_call, a2.max_call)
    return hr


def play_match(args):
    path1, path2, seed, deals = args[:4]
    serialize = args[4] if len(args) > 4 else True
    t = time.perf_counter()
    h1 = play_half(path1, path2, seed, 1, deals, serialize)
    h2 = play_half(path1, path2, seed, 2, deals, serialize)
    m = je.combine_halves(h1, h2)
    return {
        "a": os.path.basename(path1), "b": os.path.basename(path2), "seed": seed, "deals": deals,
        "mean": m.mean_bb, "stderr": m.stderr_bb, "ci": m.ci, "verdict": m.verdict, "points_a": m.points["agent1"],
        "time_a": (h1.stats_agent1.time_seconds, h2.stats_agent1.time_seconds),
        "time_b": (h1.stats_agent2.time_seconds, h2.stats_agent2.time_seconds),
        "calls_a": (h1.stats_agent1.calls, h2.stats_agent1.calls),
        "max_call_a": max(h1.max_call[0], h2.max_call[0]),
        "errors_a": h1.stats_agent1.errors + h2.stats_agent1.errors,
        "errors_b": h1.stats_agent2.errors + h2.stats_agent2.errors,
        "err_msg_a": (h1.stats_agent1.error_messages or h2.stats_agent1.error_messages or [None])[0],
        "err_msg_b": (h1.stats_agent2.error_messages or h2.stats_agent2.error_messages or [None])[0],
        "forfeit": (h1.stats_agent1.forfeited or h2.stats_agent1.forfeited, h1.stats_agent2.forfeited or h2.stats_agent2.forfeited),
        "timed_out": (h1.stats_agent1.timed_out or h2.stats_agent1.timed_out, h1.stats_agent2.timed_out or h2.stats_agent2.timed_out),
        "paired": m.paired_bb, "elapsed": time.perf_counter() - t,
    }


def parse_seeds(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part) if part.isdigit() else part)
    return out


def summarize(results, label_a=None):
    by_opp = {}
    for r in results:
        by_opp.setdefault(r["b"], []).append(r)
    lines = []
    tot_pts = 0
    for opp, rs in sorted(by_opp.items()):
        means = [r["mean"] for r in rs]
        pts = sum(r["points_a"] for r in rs)
        tot_pts += pts
        wins = sum(r["verdict"] == "AGENT1_WIN" for r in rs)
        losses = sum(r["verdict"] == "AGENT2_WIN" for r in rs)
        chops = len(rs) - wins - losses
        se = statistics.mean(r["stderr"] for r in rs)
        tmax = max(max(r["time_a"]) for r in rs)
        err = sum(r["errors_a"] for r in rs)
        lines.append(f"  vs {opp:28s} mean {statistics.mean(means):+7.3f} bb/deal (per-match se {se:.3f}) "
                     f"W/C/L {wins}/{chops}/{losses}  pts {pts:.1f}/{len(rs)}  maxT {tmax:.2f}s  err {err}")
    return "\n".join(lines) + f"\n  TOTAL points {tot_pts:.1f} / {len(results)}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bot")
    p.add_argument("opponents", nargs="+")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--seeds", default="1-4")
    p.add_argument("--procs", type=int, default=os.cpu_count())
    p.add_argument("--no-serialize", action="store_true")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)
    bot = resolve(args.bot)
    tasks = [(bot, resolve(o), s, args.deals, not args.no_serialize) for o in args.opponents for s in seeds]
    t = time.time()
    results = []
    with Pool(args.procs) as pool:
        for r in pool.imap_unordered(play_match, tasks):
            results.append(r)
            flag = ""
            if r["errors_a"] or r["errors_b"]:
                flag = f" ERR a={r['errors_a']} b={r['errors_b']}"
            print(f"{r['a']} vs {r['b']:24s} seed {r['seed']:>4}: {r['mean']:+7.3f} bb/deal ci [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
                  f"{r['verdict']:10s} time {r['time_a'][0]:.2f}/{r['time_a'][1]:.2f}s max {r['max_call_a']*1000:.0f}ms{flag}", flush=True)
            if r["err_msg_a"]:
                print("    first error (a):", r["err_msg_a"][:1500])
    print(f"\n{os.path.basename(bot)} summary ({time.time() - t:.0f}s):")
    print(summarize(results))


if __name__ == "__main__":
    main()
