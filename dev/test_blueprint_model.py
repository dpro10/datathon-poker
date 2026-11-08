"""The blueprint opponent model: tree nodes follow the official game, candidate buckets and
mixed probabilities are well formed.

    python dev/test_blueprint_model.py
"""
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from game import Action, Game, make_deal  # noqa: E402
from sdbot import blueprint  # noqa: E402
from sdbot.betting import BState, PRE  # noqa: E402
from sdbot.hand import HandTracker  # noqa: E402
from sdbot.model import DecoupledModel, classify  # noqa: E402
from sdbot.search import DEFAULT_PARAMS, Search  # noqa: E402

bp = blueprint.get()
assert bp is not None, "sdbot/data/blueprint.npz missing"
t = np.load(os.path.join(ROOT, "dev", "cfr", "tree.npz")) if os.path.exists(os.path.join(ROOT, "dev", "cfr", "tree.npz")) else None

# 1. Following the actions of random official games through the blueprint tree always lands
#    on a decision node of the right street, with exactly the game's legal actions.
rng = random.Random(11)
checked = 0
for h in range(3000):
    button = h % 2
    g = Game(make_deal("bpm", h), button, h)
    actions = []
    while not g.is_terminal:
        obs = g.observation(g.to_act)
        n = bp.node(actions)
        assert n >= 0 and bp.offset[n] >= 0, (h, actions)
        assert int(bp.street[n]) == int(obs.street), (h, actions)
        legal_tree = tuple(int(a) for a in np.nonzero(bp.child[n] >= 0)[0])
        assert legal_tree == tuple(int(a) for a in obs.legal_actions), (legal_tree, obs.legal_actions)
        if t is not None:
            assert int(t["player"][n]) == (0 if g.to_act == button else 1), (h, actions)
        legal = list(obs.legal_actions)
        w = [0.15 if a == Action.FOLD else 0.45 if a == Action.CHECK_CALL else 0.2 for a in legal]
        a = rng.choices(legal, w)[0]
        g.apply(a)
        actions.append(int(a))
        checked += 1
print(f"tree nodes match the game on {checked} decisions")

# 2. Rows are distributions over the legal actions.
for n in rng.sample(range(len(bp.offset)), 400):
    if bp.offset[n] < 0:
        continue
    legal = tuple(int(a) for a in np.nonzero(bp.child[n] >= 0)[0])
    s = int(bp.street[n])
    r = bp.rows(n, np.arange(bp.B[s]), legal)
    assert r.shape == (bp.B[s], len(legal)) and np.allclose(r.sum(axis=1), 1) and (r >= 0).all()
    if s == PRE:
        pb = bp.pre_bin_rows(n, legal)
        assert np.allclose(pb.sum(axis=1), 1)
print("blueprint rows are distributions")


# 3. A hand tracker with the blueprint model: buckets, mixed probabilities, evidence, search.
class Obs:
    pass


model = DecoupledModel()
for trial in range(30):
    rs = np.random.default_rng(trial)
    deck = rs.permutation(36)
    o = Obs()
    o.hole_cards, o.hand_index, o.is_button = [int(c) for c in deck[:5]], trial, bool(trial % 2)
    h = HandTracker(o, 400, rs, 16, model)
    h.bp, h.bp_pi, h.bp_ws, h.bps = bp, 0.6, (0.6, 0.6, 0.6, 0.6), [bp]
    st = BState(o.is_button)
    # play a short random line: preflop calls, then the flop
    while st.street == PRE and not st.over:
        a = 1
        if st.to_act == 1:
            h.st = st
            h._opp_action(a)
        h.actions.append(a)
        st.apply(a)
    if st.over:
        continue
    h.st = st
    h.board = [int(c) for c in deck[10:15]]
    for street in (1, 2, 3):
        d = h.street_data(street)
        b = h.candidate_buckets(street)
        assert b.shape == (d.K,) and b.min() >= 0 and b.max() < bp.B[street]
    n = h.node_now()
    P = h.probs_fn(st.street)(classify(st), st.legal(), n)
    assert P.shape[1] == len(st.legal()) and np.allclose(P.sum(axis=1), 1)
    e, u, w = h.range_now(st.street)
    s = Search(h.probs_fn(st.street), e, st, DEFAULT_PARAMS, 0.0)
    s.child, s.root_node = bp.child, h.node_now()
    _, ev, a, evs = s.ours(st, w)
    assert a in st.legal() and np.isfinite(ev)
    if st.to_act == 0:
        ev_pair = h.evidence(st.street, classify(st), st.legal(), 0, h.node_now())
        assert ev_pair is not None and 0 < ev_pair[0] <= 1 and 0 < ev_pair[1] <= 1
print("hand tracker + search with the blueprint model OK")
