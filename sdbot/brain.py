"""Top level: called by agent.py's three functions. Everything that must survive
between hands lives in `memory` (wiped by the judge each half)."""
from __future__ import annotations

import os
import time

import numpy as np

from . import evaluator as ev
from . import preflop as pf
from .betting import CALL, FOLD, HALF, POT, PRE, RIVER
from .hand import COARSE, HandTracker
from .model import OpponentModel, classify
from .search import DEFAULT_PARAMS, Search

DEBUG = os.environ.get("SDBOT_DEBUG") == "1"   # re-raise internal errors (testing only)


def _param_overrides() -> dict:
    """Experiments only: SDBOT_PARAMS='{"eta": 0.2}'. Never set in the tournament."""
    raw = os.environ.get("SDBOT_PARAMS")
    if not raw:
        return {}
    try:
        import json
        d = json.loads(raw)
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


PARAM_OVERRIDES = _param_overrides()

# Effort levels, richest first. The governor picks one per hand so that the time the
# judge will charge for the half stays well inside the bank.
LEVELS = [
    {"n_cand": 500, "flop_m": 24, "tree_raises": 2},
    {"n_cand": 360, "flop_m": 16, "tree_raises": 2},
    {"n_cand": 240, "flop_m": 10, "tree_raises": 1},
]
FALLBACK_LEVEL = len(LEVELS)
GRACE = 0.001          # judge: the first 1 ms of every request is free
OVERHEAD_REF = 0.0010  # HTTP + JSON + FastAPI cost per request on the reference machine
# internal seconds per request at each level on the reference machine (refined online)
COST_REF = [0.00075, 0.00060, 0.00045, 0.00005]


def _speed_ratio() -> float:
    """How slow this machine is compared with the one the constants were measured on."""
    try:
        rng = np.random.default_rng(5)
        boards = np.argsort(rng.random((24, 36)), axis=1)[:, :5]
        hands = np.sort(np.argsort(rng.random((300, 36)), axis=1)[:, :5], axis=1)
        codes = ev.hand_codes(hands)
        from .betting import BState
        best = 1e9
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(30):
                tab = ev.board_tables(boards)
                tab[codes].max(axis=1)
            for _ in range(800):
                s = BState(True)
                s.copy().apply(1)
            best = min(best, time.perf_counter() - t)
        return float(min(4.0, max(0.5, best / BENCH_REF)))
    except Exception:  # noqa: BLE001
        return 1.5


BENCH_REF = 0.0060    # seconds for the benchmark above on the reference machine
SPEED = float(os.environ["SDBOT_SPEED"]) if os.environ.get("SDBOT_SPEED") else _speed_ratio()  # env: tests only


def new_half(config, memory: dict) -> None:
    params = dict(DEFAULT_PARAMS, **PARAM_OVERRIDES)
    memory["model"] = OpponentModel(strength=float(params.get("prior_strength", 6.0)))
    memory["rng"] = np.random.default_rng(1234567)
    memory["hand"] = None
    memory["time"] = 0.0
    memory["num_hands"] = int(getattr(config, "num_hands", 1000) or 1000)
    memory["bank"] = float(getattr(config, "time_bank_seconds", 8.0) or 8.0)
    memory["params"] = params
    memory["level"] = 0
    memory["fallbacks"] = 0
    memory["hands_done"] = 0
    memory["net_bb"] = 0.0
    memory["charged"] = 0.0                 # estimated time the judge has charged us
    memory["requests"] = 0
    memory["pending_t"] = 0.0               # on_hand_end time riding on the next request
    memory["lvl_charge"] = [[0.0, 0] for _ in range(FALLBACK_LEVEL + 1)]
    memory["speed"] = SPEED
    memory["gaps"] = []                     # measured judge round trips (see _note_gap)
    memory["last_end"] = None


def overhead(memory: dict) -> float:
    """Per-request overhead the judge charges beyond our own compute: the larger of the
    speed-scaled reference and the median measured round trip."""
    est = OVERHEAD_REF * memory.get("speed", 1.0)
    g = memory.get("gaps")
    if g and len(g) >= 8:
        est = max(est, float(np.median(g)))
    return est


def _note_gap(obs, memory: dict, t_start: float) -> None:
    """When two of our requests come back to back in one hand with no opponent action in
    between (our call closed a street and we act first on the next), the time between
    our last reply and this call is the judge's round trip: network, JSON, plumbing."""
    last = memory.get("last_end")
    if last is None:
        return
    t_end, hand, n_hist = last
    if hand == obs.hand_index and len(obs.action_history) == n_hist:
        gap = t_start - t_end - memory.get("pending_t", 0.0)
        if 0.0 < gap < 0.05:
            g = memory["gaps"]
            g.append(gap)
            if len(g) > 64:
                del g[0]


def _charge(memory: dict, t_internal: float, level: int) -> None:
    c = max(0.0, t_internal + overhead(memory) - GRACE)
    memory["charged"] += c
    memory["requests"] += 1
    s = memory["lvl_charge"][level]
    s[0] += c
    s[1] += 1


def _level(memory: dict, hand_index: int) -> int:
    """Pick the richest level whose projected charge for the rest of the half fits the budget."""
    bank = memory["bank"]
    charged = memory["charged"]
    if charged > 0.8 * bank:
        return FALLBACK_LEVEL
    budget = 0.55 * bank
    hands_left = max(memory["num_hands"] - hand_index, 1)
    done = max(memory.get("hands_done", 0), 1)
    rate = memory["requests"] / done if memory.get("hands_done", 0) >= 20 else 3.0
    left = hands_left * rate
    speed = memory.get("speed", 1.0)
    ovh = overhead(memory)
    for lvl in range(FALLBACK_LEVEL + 1):
        tot, n = memory["lvl_charge"][lvl]
        guess = max(0.0, COST_REF[lvl] * speed + ovh - GRACE)
        per = (tot + 20 * guess) / (n + 20)          # measured, shrunk toward the guess
        if charged + left * per <= budget:
            return lvl
    return FALLBACK_LEVEL


def choose_action(obs, memory: dict):
    legal = [int(a) for a in obs.legal_actions]
    t0 = time.perf_counter()
    try:
        _note_gap(obs, memory, t0)
    except Exception:  # noqa: BLE001
        if DEBUG:
            raise
    try:
        a = _choose(obs, memory)
    except Exception:  # noqa: BLE001 - never let a bug cost an error
        if DEBUG:
            raise
        memory["fallbacks"] = memory.get("fallbacks", 0) + 1
        a = fallback(obs)
    finally:
        dt = time.perf_counter() - t0
        memory["time"] = memory.get("time", 0.0) + dt
        try:
            _charge(memory, dt + memory.get("pending_t", 0.0), memory.get("level", 0))
            memory["pending_t"] = 0.0
            memory["last_end"] = (time.perf_counter(), obs.hand_index, len(obs.action_history) + 1)
        except Exception:  # noqa: BLE001
            if DEBUG:
                raise
    if a not in legal:
        a = CALL if CALL in legal else legal[0]
    return a


def _choose(obs, memory: dict) -> int:
    model = memory["model"]
    h = memory.get("hand")
    if h is None or h.index != obs.hand_index:
        lvl = _level(memory, obs.hand_index)
        memory["level"] = lvl
        if lvl >= FALLBACK_LEVEL:
            memory["hand"] = None
            return fallback(obs)
        L = LEVELS[lvl]
        h = HandTracker(obs, L["n_cand"], memory["rng"], L["flop_m"], model)
        h.tree_raises = L["tree_raises"]
        memory["hand"] = h
    h.sync(obs)
    if h.desync:
        return fallback(obs)
    st = h.st
    e, u, w = h.range_now(st.street)
    params = memory["params"]
    if h.tree_raises != params.get("tree_raises", 2):
        params = dict(params, tree_raises=h.tree_raises)
    s = Search(h.probs_fn(st.street), e, st, params, weakness(memory))
    _, _, a, evs = s.ours(st, w)
    h.note_my_action(a)
    return a


def weakness(memory: dict) -> float:
    """How exploitable the opponent looks (0 = our equal, 1 = much weaker), from our
    win rate this half shrunk toward a prior."""
    p = memory["params"]
    n = memory.get("hands_done", 0)
    wr = (memory.get("net_bb", 0.0) + p["sigma_n0"] * p["sigma_wr0"]) / (n + p["sigma_n0"])
    return float(min(1.0, max(0.0, wr / p["sigma_full"])))


def on_hand_end(result, memory: dict) -> None:
    t0 = time.perf_counter()
    memory["net_bb"] = memory.get("net_bb", 0.0) + result.my_net / 2.0
    try:
        if memory.get("level", 0) < FALLBACK_LEVEL or memory.get("hand") is not None:
            _learn(result, memory)
    except Exception:  # noqa: BLE001
        if DEBUG:
            raise
        memory["fallbacks"] = memory.get("fallbacks", 0) + 1
    finally:
        dt = time.perf_counter() - t0
        memory["time"] = memory.get("time", 0.0) + dt
        memory["pending_t"] = memory.get("pending_t", 0.0) + dt
    memory["hands_done"] = memory.get("hands_done", 0) + 1


def _learn(result, memory: dict) -> None:
    model = memory["model"]
    h = memory.get("hand")
    if h is None or h.index != result.hand_index:
        # We never acted this hand: the opponent (small blind) folded at once.
        if not result.went_to_showdown and result.my_net == 1:
            model.observe_soft(0, FOLD, np.full(5, 0.2))
        return
    recs = list(h.opp_recs)
    p = h.pending
    if p is not None and not h.desync:
        # The hand ended right after our last action, so the opponent's reply is implied.
        if not result.went_to_showdown and result.winner == "me" and p["facing"]:
            recs.append({"cls": p["cls"], "act": FOLD, "street": p["street"], "pi": p["pi"]})
        elif result.went_to_showdown:
            recs.append({"cls": p["cls"], "act": CALL, "street": p["street"], "pi": p["pi"]})
    if result.opp_hole is not None and not h.desync:
        opp = [int(c) for c in result.opp_hole]
        h.board = [int(c) for c in result.board]
        cache = {}
        for r in recs:
            s = r["street"]
            if s not in cache:
                cache[s] = h.strength_of(opp, s)
            model.observe_known(r["cls"], r["act"], cache[s])
    else:
        for r in recs:
            model.observe_soft(r["cls"], r["act"], r["pi"])
    memory["hand"] = None


# ---------------------------------------------------------------------------
# Fallback: cheap, safe heuristic (used on errors, desyncs, or low time)
# ---------------------------------------------------------------------------

_FB_POS = {}   # n -> sorted random 5-subsets of range(n), for the fallback's random hands


def _fb_positions(n: int) -> np.ndarray:
    p = _FB_POS.get(n)
    if p is None:
        r = np.random.default_rng(n)
        p = np.sort(np.argsort(r.random((64, n)), axis=1)[:, :5], axis=1)
        _FB_POS[n] = p
    return p


def board_pct(hole, board) -> float:
    """Percentile of the hand we hold right now among 64 random hands on this board."""
    from .hand import bitmask, cards_not_in
    rem = cards_not_in(bitmask(hole) | bitmask(board))
    opp = rem[_fb_positions(len(rem))]
    t = ev.board_tables(np.asarray(board, dtype=np.int64)[None, :])[:, 0]
    mine = int(t[ev.hand_codes_one(hole)].max())
    theirs = t[ev.hand_codes(opp)].max(axis=1)
    return float(np.mean(theirs < mine) + 0.5 * np.mean(theirs == mine))


def fallback(obs) -> int:
    """Cheap, sane strategy for emergencies (errors, desyncs, an almost empty time bank).

    Preflop it uses the hand's percentile and position; after the flop, the percentile of
    the hand made right now among random hands on this board (~60 us)."""
    legal = [int(a) for a in obs.legal_actions]
    try:
        tc = obs.to_call
        price = tc / (obs.pot + tc) if tc > 0 else 0.0
        if len(obs.board) == 0:
            p = pf.pct_of(obs.hole_cards)
            if tc == 0:                                     # big blind after a limp
                return POT if (p > 0.75 and POT in legal) else CALL
            if obs.raises_this_street == 0:                 # small blind first action
                if p > 0.55 and POT in legal:
                    return POT
                return CALL if p > 0.10 else FOLD
            if p > 0.90 and POT in legal:
                return POT
            if p > 0.20 + 0.5 * price:
                return CALL
            return FOLD if FOLD in legal else CALL
        strength = board_pct(obs.hole_cards, obs.board)
        if len(obs.board) < 5:
            strength = 0.8 * strength + 0.2 * 0.5           # unfinished board: draws still to come
        if tc == 0:
            if strength > 0.85 and POT in legal:
                return POT
            if strength > 0.70 and HALF in legal:
                return HALF
            return CALL
        if strength > 0.93 and POT in legal:
            return POT
        if strength > 0.30 + price or tc <= 1:
            return CALL
        return FOLD if FOLD in legal else CALL
    except Exception:  # noqa: BLE001
        return CALL if obs.to_call == 0 or FOLD not in legal else FOLD
