"""Expectimax over the rest of the current street.

Values are vectors over the opponent's candidate hands (or preflop strength bins):
V[k] = chips we expect to collect from now on minus chips we add from now on,
if the opponent holds candidate k. At our nodes we pick the action with the best
weighted mean (weights = our posterior over the opponent's hand at that node); at
the opponent's nodes the model gives P(action | candidate strength).

When the street ends with chips behind, a leaf estimates the rest of the hand
(fitted on self-play data, dev/leaf_data.py): with d = 2e - 1 and spr = stack / pot,
    pot * (e + eta_pos * e(1 - e) + min(spr, kappa_s) * (b_s * d + c_s * d|d|))
eta: positional realization; the min(spr, kappa) term is the money still to go in,
won by the stronger hand; s is the street that starts at the leaf.
"""
from __future__ import annotations

import numpy as np

from .betting import CALL, FOLD, HALF, BState
from .model import classify

DEFAULT_PARAMS = {
    "eta": 0.0,           # positional equity realization
    "leaf_scale": 1.0,    # multiplies the future-money term
    "tree_raises": 2,     # raises explored beyond the root state
    "prior_strength": 6.0,  # pseudo-counts per (class, bin) for the opponent model prior (EM model)
    "act_strength": 8.0,    # decoupled model: pseudo-counts for P(action | class)
    "bin_strength": 2.0,    # decoupled model: pseudo-counts for P(strength bin | action, class)
    "sigma_wr0": 1.0,     # prior win rate (bb/hand) for the opponent-weakness estimate
    "sigma_n0": 100.0,    # ...and its weight in hands
    "sigma_full": 2.0,    # win rate (bb/hand) at which the leaf is fully "weak opponent"
}

# Future-money curves per unit of min(spr, kappa), as a function of d = 2e - 1, for the
# street that starts at the leaf (index 1..3). Fitted with dev/leaf_data.py:
#   EVEN: self-play (equal skill), symmetric;  WEAK: vs the practice pool, convex.
LEAF_KAPPA = (0.0, 3.0, 2.0, 1.0)
LEAF_EVEN = ((0, 0), (0.658, -0.359), (0.262, 0.008), (0.063, 0.063))          # b*d + c*d|d|
LEAF_WEAK = ((0, 0, 0, 0), (1.427, -0.466, 0.222, 0.008), (0.628, 0.121, 0.013, -0.224),
             (0.257, 0.257, 0.076, -0.076))                                     # up1 d+ + up2 d+^2 + dn1 d- + dn2 d-^2


def future_curve(e: np.ndarray, street: int, sigma: float, even_c=None, weak_c=None) -> np.ndarray:
    if street < 1 or street > 3:
        return np.zeros_like(e)
    d = 2.0 * e - 1.0
    b, c = (even_c or LEAF_EVEN)[street]
    even = b * d + c * d * np.abs(d)
    u1, u2, d1, d2 = (weak_c or LEAF_WEAK)[street]
    dp, dn = np.maximum(d, 0.0), np.minimum(d, 0.0)
    weak = u1 * dp + u2 * dp * dp + d1 * dn + d2 * dn * dn
    return (1.0 - sigma) * even + sigma * weak


class Search:
    def __init__(self, probs_fn, e: np.ndarray, root: BState, params: dict, sigma: float = 0.0):
        self.probs_fn = probs_fn
        self.e = e
        self.e_var = e * (1.0 - e)
        self.sigma = sigma
        self.curve = future_curve(e, root.street + 1, sigma, params.get("leaf_even"), params.get("leaf_weak"))
        self.root_contrib = root.contrib[0]
        self.root_street = root.street
        self.root_raises = root.raises
        self.p = params
        self.max_raises = params.get("tree_raises", 2)
        self.nodes = 0

    def leaf(self, st: BState):
        P = st.pot
        S = min(st.stacks)
        spr = S / P if P > 0 else 0.0
        p = self.p
        eta = p["eta"] * min(1.0, spr) * (1.0 if st.button == 0 else -1.0)
        m = min(spr, LEAF_KAPPA[st.street]) * p["leaf_scale"] * P
        return P * self.e + (P * eta) * self.e_var + m * self.curve

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
            return self.ours(st, w, root=False)[0]
        return self.theirs(st, w)

    def ours(self, st: BState, w: np.ndarray, root: bool = True):
        legal = st.legal()
        if st.raises - self.root_raises >= self.max_raises:
            legal = tuple(a for a in legal if a <= CALL)
        elif not root and HALF in legal:
            legal = tuple(a for a in legal if a != HALF)    # deeper re-raises: pot size only
        sw = float(w.sum())
        best_v, best_ev, best_a = None, -1e18, legal[0]
        evs = {}
        for a in legal:
            if a == FOLD:
                v = -float(st.contrib[0] - self.root_contrib)
                self.nodes += 1
            else:
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
