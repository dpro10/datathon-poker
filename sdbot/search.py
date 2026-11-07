"""Expectimax over the rest of the current street.

Values are vectors over the opponent's candidate hands (or preflop strength bins):
V[k] = chips we expect to collect from now on minus chips we add from now on,
if the opponent holds candidate k. At our nodes we pick the action with the best
weighted mean (weights = our posterior over the opponent's hand at that node); at
the opponent's nodes the model gives P(action | candidate strength).

When the street ends with chips behind, a leaf estimates the rest of the hand:
    pot * (e + eta_pos * e * (1 - e)) + lam * min(stack, pot) * (2e - 1)|2e - 1|
(eta: positional realization, lam: money still to go in, won by the stronger hand).
"""
from __future__ import annotations

import numpy as np

from .betting import CALL, FOLD, BState
from .model import classify

DEFAULT_PARAMS = {
    "eta": 0.10,          # positional equity realization
    "lam": 0.15,          # future-street money won by the stronger hand
    "tree_raises": 2,     # raises explored beyond the root state
}


class Search:
    def __init__(self, probs_fn, e: np.ndarray, root: BState, params: dict):
        self.probs_fn = probs_fn
        self.e = e
        d = 2.0 * e - 1.0
        self.e_var = e * (1.0 - e)
        self.e_sq = d * np.abs(d)
        self.root_contrib = root.contrib[0]
        self.root_street = root.street
        self.root_raises = root.raises
        self.p = params
        self.max_raises = params.get("tree_raises", 2)
        self.nodes = 0

    def leaf(self, st: BState):
        P = st.pot
        S = min(st.stacks)
        r = min(1.0, S / P) if P > 0 else 0.0
        eta = self.p["eta"] * r * (1.0 if st.button == 0 else -1.0)
        return P * self.e + (P * eta) * self.e_var + (self.p["lam"] * min(S, P)) * self.e_sq

    def value(self, st: BState, w: np.ndarray):
        self.nodes += 1
        if st.over:
            inv = st.contrib[0] - self.root_contrib
            if st.folded == 0:
                return -float(inv)
            if st.folded == 1:
                return float(st.pot - inv)
            return st.pot * self.e - inv
        if st.street != self.root_street:
            return self.leaf(st) - (st.contrib[0] - self.root_contrib)
        if st.to_act == 0:
            return self.ours(st, w)[0]
        return self.theirs(st, w)

    def ours(self, st: BState, w: np.ndarray):
        legal = st.legal()
        if st.raises - self.root_raises >= self.max_raises:
            legal = tuple(a for a in legal if a <= CALL)
        sw = float(w.sum())
        best_v, best_ev, best_a = None, -1e18, legal[0]
        evs = {}
        for a in legal:
            ch = st.copy()
            ch.apply(a)
            v = self.value(ch, w)
            ev = float(np.dot(w, v)) / sw if isinstance(v, np.ndarray) else float(v)
            evs[a] = ev
            if ev > best_ev:
                best_v, best_ev, best_a = v, ev, a
        return best_v, best_ev, best_a, evs

    def theirs(self, st: BState, w: np.ndarray):
        legal = st.legal()
        cls = classify(st)
        P = self.probs_fn(cls, legal)
        if st.raises - self.root_raises >= self.max_raises and len(legal) > 2:
            # beyond the explored depth, a raise is treated as a call
            n = 2 if legal[0] == FOLD else 1
            full = P
            P = full[:, :n].copy()
            P[:, n - 1] += full[:, n:].sum(axis=1)
            legal = legal[:n]
        total = 0.0
        for j, a in enumerate(legal):
            p = P[:, j]
            if p.max() < 1e-3:
                continue
            ch = st.copy()
            ch.apply(a)
            total = total + p * self.value(ch, w * p)
        return total
