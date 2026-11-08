"""Top level: called by agent.py's three functions. Everything that must survive
between hands lives in `memory` (wiped by the judge each half)."""
from __future__ import annotations

import os
import time

import numpy as np

from . import evaluator as ev
from . import preflop as pf
from .betting import CALL, FOLD, HALF, POT, PRE, RIVER, RULES, set_rules
from .hand import COARSE, MIX_MIN, HandTracker
from .model import DecoupledModel, OpponentModel, classify
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


def _preload() -> None:
    """Load the blueprint while the server starts (free) rather than in the first request
    (charged to the time bank)."""
    p = dict(DEFAULT_PARAMS, **PARAM_OVERRIDES)
    if p.get("bp_auto") or p.get("bp_model", 0) or p.get("mode") == "blueprint" or p.get("bp_streets") \
            or p.get("kl_kappa", 0) or p.get("blend_hi", 0):
        try:
            from . import blueprint
            bp = blueprint.get()
            if bp is not None:
                bp.pre_bin_rows(0, (CALL,))     # builds the preflop bin weights
        except Exception:  # noqa: BLE001
            pass


_preload()


def new_half(config, memory: dict) -> None:
    params = dict(DEFAULT_PARAMS, **PARAM_OVERRIDES)
    try:
        set_rules(getattr(config, "small_blind", 1), getattr(config, "big_blind", 2),
                  getattr(config, "starting_stack", 80), getattr(config, "raise_cap", 4))
    except (TypeError, ValueError):
        pass
    prior_probs = None
    if params.get("prior_file"):                # experiments only
        prior_probs = np.load(params["prior_file"])
    if params.get("model", "decoupled") == "decoupled":
        memory["model"] = DecoupledModel(prior_probs, strength=float(params.get("prior_strength", 6.0)),
                                         act_strength=float(params.get("act_strength", 8.0)),
                                         bin_strength=float(params.get("bin_strength", 2.0)),
                                         share=float(params.get("share", 0.0)))
    else:
        memory["model"] = OpponentModel(strength=float(params.get("prior_strength", 6.0)))
    memory["model_s"] = None
    if params.get("cp_detect"):
        memory["model_s"] = DecoupledModel(prior_probs, strength=float(params.get("prior_strength", 6.0)),
                                           act_strength=float(params.get("act_strength", 8.0)),
                                           bin_strength=float(params.get("bin_strength", 2.0)))
    memory["cp_s"] = 0.0
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


def _projection(memory: dict, hand_index: int):
    """(charged so far, budget, [projected total charge if the rest of the half ran at each level])."""
    bank = memory["bank"]
    charged = memory["charged"]
    budget = 0.55 * bank
    hands_left = max(memory["num_hands"] - hand_index, 1)
    done = max(memory.get("hands_done", 0), 1)
    rate = memory["requests"] / done if memory.get("hands_done", 0) >= 20 else 3.0
    left = hands_left * rate
    speed = memory.get("speed", 1.0)
    ovh = overhead(memory)
    proj = []
    for lvl in range(FALLBACK_LEVEL + 1):
        tot, n = memory["lvl_charge"][lvl]
        guess = max(0.0, COST_REF[lvl] * speed + ovh - GRACE)
        per = (tot + 20 * guess) / (n + 20)          # measured, shrunk toward the guess
        proj.append(charged + left * per)
    return charged, budget, proj


def _level(memory: dict, hand_index: int) -> int:
    """Pick the richest level whose projected charge for the rest of the half fits the budget."""
    charged, budget, proj = _projection(memory, hand_index)
    if charged > 0.8 * memory["bank"]:
        return FALLBACK_LEVEL
    for lvl in range(FALLBACK_LEVEL + 1):
        if proj[lvl] <= budget:
            return lvl
    return FALLBACK_LEVEL


def _rich_ok(memory: dict, hand_index: int) -> bool:
    """Room for a bigger search in big pots: the rest of the half at level 0 (whose measured
    cost includes the big-pot searches made so far) stays well inside the budget."""
    p = memory["params"]
    if not p.get("rich"):
        return False
    _, budget, proj = _projection(memory, hand_index)
    return proj[0] <= float(p.get("rich_slack", 0.8)) * budget


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
        if lvl == 0 and "l0" in memory["params"]:     # experiments only
            L = dict(L, **memory["params"]["l0"])
        h = HandTracker(obs, L["n_cand"], memory["rng"], L["flop_m"], model)
        h.tree_raises = L["tree_raises"]
        if lvl == 0 and _rich_ok(memory, obs.hand_index):
            r = memory["params"]["rich"]
            h.rich = (int(memory["params"].get("rich_pot", 16)), int(r["n_cand"]), int(r["flop_m"]))
        h.model_s = memory.get("model_s")
        ws = bp_model_weights(memory)
        mix = max(ws) > MIX_MIN
        # evidence for the weights: every hand at first or while the blueprint is in use;
        # once it has been ruled out, every 4th hand keeps watching at a quarter of the cost
        evid = bool(memory["params"].get("bp_auto")) and (
            mix or memory.get("hands_done", 0) < 200 or obs.hand_index % 4 == 0)
        if mix or evid:
            from . import blueprint
            h.bp = blueprint.get()
            if h.bp is not None:
                h.bp_pi, h.bp_ws, h.ev_on = max(ws), ws, evid
                h.bp_smooth = bool(memory["params"].get("bp_river_smooth"))
        memory["hand"] = h
    h.sync(obs)
    if h.desync:
        memory["desyncs"] = memory.get("desyncs", 0) + 1
        return fallback(obs)
    st = h.st
    if memory["params"].get("mode") == "blueprint" or st.street in memory["params"].get("bp_streets", ()):
        a = _blueprint_action(h, memory)
        if a is not None:
            h.note_my_action(a)
            return a
    e, u, w = h.range_now(st.street)
    params = memory["params"]
    if h.tree_raises != params.get("tree_raises", 2):
        params = dict(params, tree_raises=h.tree_raises)
    g_vecs, edges = (None, None)
    if params.get("leaf_bp") and st.street < RIVER:
        g_vecs, edges = _leaf_vectors(h, len(e))
    gw = h.bp_ws[st.street] if params.get("leaf_bp") == "auto" else None
    s = Search(h.probs_fn(st.street), e, st, params, weakness(memory), g_vecs, gw)
    if g_vecs is not None:
        s.spr_edges = edges
    if h.bp_pi > 0:
        s.child, s.root_node = h.bp.child, h.node_now()
    _, _, a, evs = s.ours(st, w)
    if params.get("kl_kappa", 0) > 0 or params.get("blend_hi", 0) > 0:
        a2 = _kl_choice(h, memory, evs)
        if a2 is not None:
            a = a2
    h.note_my_action(a)
    return a


def _blueprint_probs(h):
    """Blueprint probabilities over the 4 actions at the current node, or None."""
    from . import blueprint
    bp = blueprint.get()
    if bp is None:
        return None
    st = h.st
    node = bp.node(h.actions)
    if node < 0 or int(bp.street[node]) != st.street:
        return None
    if st.street == PRE:
        b = bp.bucket(0, my_cls=h.my_cls)
    else:
        b = bp.bucket(st.street, my_hs=h.street_data(st.street).my_hs)
    return bp.probs(node, b)


def _kl_choice(h, memory: dict, evs: dict):
    """Exploit only as far as the evidence justifies: sample from
        pi(a) ~ blueprint(a) * exp(EV(a) / tau),
    the strategy that maximizes EV minus tau * KL(pi || blueprint). tau scales with the pot
    and shrinks as we observe the opponent, so we start near equilibrium and move toward the
    search's best response once the opponent's tendencies are established."""
    p_bp = _blueprint_probs(h)
    if p_bp is None:
        return None
    p = memory["params"]
    legal = list(evs.keys())
    n = memory.get("hands_done", 0)
    pot = max(h.st.pot, 2)
    if p.get("blend_hi", 0) > 0:
        # Trust the search only as far as it is visibly beating this opponent: the
        # temperature falls from blend_hi (play the blueprint) to blend_lo (play the
        # search's best response) as our shrunk win rate rises toward blend_full bb/hand.
        wr = (memory.get("net_bb", 0.0) + p["blend_n0"] * p["blend_wr0"]) / (n + p["blend_n0"])
        sb = min(1.0, max(0.0, wr / p["blend_full"]))
        tau = pot * (p["blend_lo"] + (p["blend_hi"] - p["blend_lo"]) * (1.0 - sb))
    else:
        tau = p["kl_kappa"] * pot * p["kl_n0"] / (p["kl_n0"] + n)
    tau = max(tau, 1e-6)
    prior = np.array([p_bp[a] for a in legal]) + p.get("kl_floor", 0.02)
    ev = np.array([evs[a] for a in legal])
    logit = np.log(prior / prior.sum()) + (ev - ev.max()) / tau
    q = np.exp(logit - logit.max())
    q /= q.sum()
    return int(legal[int(memory["rng"].choice(len(legal), p=q))])


def _leaf_vectors(h, n_opp: int):
    """Blueprint leaf table rows for the current street: one vector over the opponent's
    candidates (or preflop strength bins) per SPR band. None if anything is missing."""
    from . import blueprint
    bp, lt = blueprint.get(), blueprint.leaf_table()
    if bp is None or lt is None:
        return None, None
    st = h.st
    s = st.street
    if list(lt["buckets"]) != bp.B[:3]:
        return None, None
    pos = 1 if st.button == 0 else 0
    if s == PRE:
        gm = int(lt["grp_pre"][bp.bucket(0, my_cls=h.my_cls)])
        rows = lt["pre_bins"][:, pos, gm, :]                       # (bands, NB)
    else:
        d = h.street_data(s)
        C, grp = (bp.cflop, lt["grp_flop"]) if s == 1 else (bp.cturn, lt["grp_turn"])
        gm = int(grp[bp.bucket(s, my_hs=d.my_hs)])
        go = h.candidate_groups(s, C, grp)
        rows = lt["g"][s, :, pos, gm, :][:, go]                    # (bands, K)
    if rows.shape[1] != n_opp:
        return None, None
    return [rows[b].astype(np.float64) for b in range(rows.shape[0])], lt["spr_edges"]


def _blueprint_action(h, memory: dict):
    """Play the CFR blueprint: sample from its strategy at this node for our bucket."""
    from . import blueprint
    bp = blueprint.get()
    if bp is None:
        return None
    st = h.st
    node = bp.node(h.actions)
    if node < 0 or int(bp.street[node]) != st.street:
        return None
    if st.street == PRE:
        b = bp.bucket(0, my_cls=h.my_cls)
    else:
        b = bp.bucket(st.street, my_hs=h.street_data(st.street).my_hs)
    p = bp.probs(node, b)
    if p is None:
        return None
    a = int(memory["rng"].choice(4, p=p))
    return a if a in st.legal() else None


def bp_model_weights(memory: dict) -> tuple:
    """Weight of the blueprint in the opponent model on each street (0 = the learned class
    model only). With bp_auto: the posterior mean of the weight w of the mixture
    w * blueprint + (1 - w) * learned, from how well each w predicted every action so far
    (one weight for the whole hand, or one per street with bp_streetwise)."""
    p = memory["params"]
    if not p.get("bp_auto"):
        w = float(p.get("bp_model", 0.0))
        return (w, w, w, w)
    ll = memory.get("pi_ll")                                 # (4, grid) log-likelihoods
    k = float(p.get("bp_prior_k", 0.0))                      # prior ~ exp(-k w): most opponents are not solvers
    if ll is None:
        if k <= 0:
            w = float(p.get("bp_model0", 0.5))
            return (w, w, w, w)
        ll = np.zeros((4, len(PI_GRID)))
    if not p.get("bp_streetwise"):
        ll = np.broadcast_to(ll.sum(axis=0), ll.shape)
    ll = ll / float(p.get("bp_temp", 1.0)) - k * PI_GRID[None, :]
    post = np.exp(ll - ll.max(axis=1, keepdims=True))
    w = post @ PI_GRID / post.sum(axis=1)
    w = np.clip(w, p.get("bp_lo", 0.02), p.get("bp_hi", 0.98))
    return tuple(float(x) for x in w)


def bp_model_weight(memory: dict) -> float:
    """The overall blueprint weight (for reports)."""
    return max(bp_model_weights(memory))


PI_GRID = np.linspace(0.0, 1.0, 21)


def _note_evidence(memory: dict, pairs) -> None:
    """Add the log-likelihood of every grid weight, per street."""
    if not pairs:
        return
    a = np.asarray(pairs)                                   # (n, 3): street, P(blueprint), P(learned)
    lik = np.log(PI_GRID[None, :] * a[:, 1:2] + (1.0 - PI_GRID[None, :]) * a[:, 2:3])
    ll = np.zeros((4, len(PI_GRID)))
    np.add.at(ll, a[:, 0].astype(np.int64), lik)
    # old evidence fades: the learned model improves as it sees more, so its early
    # mispredictions should not keep the blueprint weight up for long
    g = float(memory["params"].get("bp_decay", 1.0))
    memory["pi_ll"] = ll if memory.get("pi_ll") is None else g * memory["pi_ll"] + ll


def weakness(memory: dict) -> float:
    """How exploitable the opponent looks (0 = our equal, 1 = much weaker), from our
    win rate this half shrunk toward a prior."""
    p = memory["params"]
    n = memory.get("hands_done", 0)
    wr = (memory.get("net_bb", 0.0) + p["sigma_n0"] * p["sigma_wr0"]) / (n + p["sigma_n0"])
    sigma = float(min(1.0, max(0.0, wr / p["sigma_full"])))
    if p.get("sigma_bp") and p.get("bp_auto"):
        # an opponent the blueprint predicts well is not weak, whatever the early results say
        sigma *= 1.0 - float(np.mean(bp_model_weights(memory)))
    return sigma


def on_hand_end(result, memory: dict) -> None:
    t0 = time.perf_counter()
    memory["net_bb"] = memory.get("net_bb", 0.0) + result.my_net / float(RULES["bb"] or 2)
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
        if not result.went_to_showdown and result.my_net == RULES["sb"]:
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
    if h.bp_pi > 0 and h.ev_on and not h.desync:
        pairs = list(h.ev_pairs)
        if p is not None and len(recs) > len(h.opp_recs):
            r = recs[-1]
            legal = p["state"].legal()
            if r["act"] in legal:
                ev = h.evidence(r["street"], r["cls"], legal, legal.index(r["act"]), p.get("node", -1))
                if ev is not None:
                    pairs.append((r["street"],) + ev)
        _note_evidence(memory, pairs)
    w = memory["params"].get("memory_hands", 0)
    if w and hasattr(model, "decay"):
        model.decay(1.0 - 1.0 / float(w))
    models = [model]
    ms = memory.get("model_s")
    if ms is not None and not h.desync:
        models.append(ms)
        _change_point(memory, h.cp_ll)
        ms.decay(1.0 - 1.0 / float(memory["params"].get("cp_mem", 30)))
    if result.opp_hole is not None and not h.desync:
        opp = [int(c) for c in result.opp_hole]
        h.board = [int(c) for c in result.board]
        cache = {}
        for r in recs:
            s = r["street"]
            if s not in cache:
                cache[s] = h.strength_of(opp, s)
            for m in models:
                m.observe_known(r["cls"], r["act"], cache[s])
    else:
        for r in recs:
            for m in models:
                m.observe_soft(r["cls"], r["act"], r["pi"])
    memory["hand"] = None


def _change_point(memory: dict, x: float) -> None:
    """CUSUM on how much better a short-memory model predicts the opponent than the full
    model. A stationary opponent keeps the sum near 0 (the short model is noisier); after a
    change of style the short model adapts first and the sum climbs. Then most of the old
    observations are forgotten, so the full model re-learns quickly."""
    p = memory["params"]
    memory["cp_s"] = max(0.0, memory.get("cp_s", 0.0) + x)
    if memory["cp_s"] > float(p.get("cp_h", 8.0)) and memory.get("hands_done", 0) >= 40:
        memory["model"].decay(float(p.get("cp_keep", 0.2)))
        memory["cp_s"] = 0.0
        memory["cp_events"] = memory.get("cp_events", 0) + 1


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
        if strength > 0.30 + price or tc * 2 <= obs.big_blind:
            return CALL
        return FOLD if FOLD in legal else CALL
    except Exception:  # noqa: BLE001
        return CALL if obs.to_call == 0 or FOLD not in legal else FOLD
