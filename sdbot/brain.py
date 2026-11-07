"""Top level: called by agent.py's three functions. Everything that must survive
between hands lives in `memory` (wiped by the judge each half)."""
from __future__ import annotations

import os
import time

import numpy as np

from . import evaluator as ev
from . import preflop as pf
from .betting import CALL, FOLD, HALF, POT, PRE, RIVER
from .hand import HandTracker
from .model import OpponentModel, classify
from .search import DEFAULT_PARAMS, Search

DEBUG = os.environ.get("SDBOT_DEBUG") == "1"   # re-raise internal errors (testing only)

LEVELS = [  # (candidates, flop runouts) from richest to cheapest
    (600, 30),
    (400, 20),
    (250, 14),
]


def new_half(config, memory: dict) -> None:
    memory["model"] = OpponentModel()
    memory["rng"] = np.random.default_rng(1234567)
    memory["hand"] = None
    memory["time"] = 0.0
    memory["num_hands"] = int(getattr(config, "num_hands", 1000) or 1000)
    memory["bank"] = float(getattr(config, "time_bank_seconds", 8.0) or 8.0)
    memory["params"] = dict(DEFAULT_PARAMS)
    memory["level"] = 0
    memory["fallbacks"] = 0
    memory["hands_done"] = 0


def _level(memory, hand_index: int) -> int:
    """Pick how much work to do this hand so the whole half fits the time budget."""
    used = memory["time"]
    budget = 0.55 * memory["bank"]
    left = max(memory["num_hands"] - hand_index, 1)
    per_hand = (budget - used) / left
    avg = used / max(hand_index, 1)
    if used > 0.85 * memory["bank"] or per_hand <= 0:
        return 3
    if hand_index < 20:
        return 0
    if avg <= per_hand:
        return 0
    if avg * 0.65 <= per_hand:
        return 1
    return 2


def choose_action(obs, memory: dict):
    legal = [int(a) for a in obs.legal_actions]
    t0 = time.perf_counter()
    try:
        a = _choose(obs, memory)
    except Exception:  # noqa: BLE001 - never let a bug cost an error
        if DEBUG:
            raise
        memory["fallbacks"] = memory.get("fallbacks", 0) + 1
        a = fallback(obs)
    finally:
        memory["time"] = memory.get("time", 0.0) + time.perf_counter() - t0
    if a not in legal:
        a = CALL if CALL in legal else legal[0]
    return a


def _choose(obs, memory: dict) -> int:
    model = memory["model"]
    h = memory.get("hand")
    if h is None or h.index != obs.hand_index:
        lvl = _level(memory, obs.hand_index)
        memory["level"] = lvl
        if lvl >= 3:
            memory["hand"] = None
            return fallback(obs)
        n_cand, flop_m = LEVELS[lvl]
        h = HandTracker(obs, n_cand, memory["rng"], flop_m)
        memory["hand"] = h
    h.sync(obs, model)
    if h.desync or memory["level"] >= 3:
        return fallback(obs)
    st = h.st
    e, u, w = h.range_now(st.street)
    s = Search(h.probs_fn(st.street, model), e, st, memory["params"])
    _, _, a, evs = s.ours(st, w)
    h.note_my_action(a, model)
    return a


def on_hand_end(result, memory: dict) -> None:
    t0 = time.perf_counter()
    try:
        _learn(result, memory)
    except Exception:  # noqa: BLE001
        if DEBUG:
            raise
        memory["fallbacks"] = memory.get("fallbacks", 0) + 1
    finally:
        memory["time"] = memory.get("time", 0.0) + time.perf_counter() - t0
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

def fallback(obs) -> int:
    legal = [int(a) for a in obs.legal_actions]
    try:
        if len(obs.board) == 0:
            strength = pf.pct_of(obs.hole_cards)
        else:
            o = ev.score_one(obs.hole_cards, obs.board)
            cat = ev.CATEGORY_OF_ORD[o]
            strength = {0: 0.05, 1: 0.25, 2: 0.5, 3: 0.6, 4: 0.7, 5: 0.85, 6: 0.9, 7: 0.97, 8: 0.99}[int(cat)]
        if obs.to_call == 0:
            if strength > 0.8 and POT in legal:
                return POT
            return CALL
        price = obs.to_call / (obs.pot + obs.to_call)
        if strength > 0.9 and POT in legal:
            return POT
        if strength > price + 0.25 or obs.to_call <= 1:
            return CALL
        return FOLD if FOLD in legal else CALL
    except Exception:  # noqa: BLE001
        return CALL if obs.to_call == 0 or FOLD not in legal else FOLD
