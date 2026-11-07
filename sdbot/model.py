"""Opponent model: P(action | situation class, hand-strength percentile).

Situation classes (from the acting player's point of view):
  0 PRE_OPEN       small blind's first decision (fold / limp / raise)
  1 PRE_VS_LIMP    big blind after a limp (check / raise)
  2 PRE_VS_HALF    facing the first preflop raise, half-pot sized
  3 PRE_VS_POT     facing the first preflop raise, pot sized
  4 PRE_VS_RERAISE facing a re-raise preflop
  5 + 4*(street-1) + k for street = flop/turn/river, k:
       0 NOBET     nothing to call (check / bet)
       1 VS_HALF   facing a half-pot bet
       2 VS_POT    facing a pot bet
       3 VS_RAISE  facing a raise

Strength u in [0, 1] is the percentile of the hand's equity vs a random hand on
the current board (preflop: the precomputed preflop percentile). The model keeps
Dirichlet counts per (class, bin, action) with NBIN bins over u; probabilities
are linearly interpolated between bin centres so likelihoods are smooth in u.
"""
from __future__ import annotations

import numpy as np

from .betting import FLOP, PRE, BState, HALF, POT

NBIN = 5
NCLASS = 17
NACT = 4
CENTERS = (np.arange(NBIN) + 0.5) / NBIN
CLASS_NAMES = ["PRE_OPEN", "PRE_VS_LIMP", "PRE_VS_HALF", "PRE_VS_POT", "PRE_VS_RERAISE"] + [
    f"{s}_{k}" for s in ("FLOP", "TURN", "RIVER") for k in ("NOBET", "VS_HALF", "VS_POT", "VS_RAISE")]


def classify(st: BState) -> int:
    """Class of the decision facing st.to_act."""
    if st.street == PRE:
        if st.raises == 0:
            return 0 if st.to_call() > 0 else 1
        if st.raises == 1:
            return 3 if st.last_raise == POT else 2
        return 4
    base = 5 + 4 * (st.street - FLOP)
    if st.raises == 0:
        return base
    if st.raises == 1:
        return base + (2 if st.last_raise == POT else 1)
    return base + 3


# Prior policies (probabilities over F, C, H, P by strength bin, weak -> strong).
_P_OPEN = [[.55, .35, .03, .07], [.30, .50, .05, .15], [.15, .55, .08, .22], [.07, .45, .10, .38], [.03, .30, .10, .57]]
_P_LIMP = [[0, .90, .03, .07], [0, .85, .05, .10], [0, .78, .07, .15], [0, .65, .08, .27], [0, .45, .10, .45]]
_P_VSR1 = [[.60, .35, .02, .03], [.40, .52, .03, .05], [.25, .62, .04, .09], [.12, .65, .06, .17], [.05, .55, .10, .30]]
_P_VSRR = [[.72, .26, .00, .02], [.55, .42, .01, .02], [.38, .57, .01, .04], [.20, .70, .02, .08], [.08, .62, .04, .26]]
_P_NOBET = [[0, .80, .08, .12], [0, .80, .10, .10], [0, .70, .15, .15], [0, .50, .20, .30], [0, .35, .20, .45]]
_P_VSH = [[.75, .20, .01, .04], [.50, .45, .01, .04], [.30, .62, .03, .05], [.12, .70, .06, .12], [.04, .50, .10, .36]]
_P_VSP = [[.80, .17, .01, .02], [.58, .38, .01, .03], [.38, .55, .03, .04], [.18, .68, .04, .10], [.05, .55, .10, .30]]
_P_VSRA = [[.85, .14, .00, .01], [.70, .28, .00, .02], [.50, .46, .01, .03], [.28, .64, .02, .06], [.08, .60, .07, .25]]


def default_prior() -> np.ndarray:
    pri = np.zeros((NCLASS, NBIN, NACT))
    pri[0], pri[1], pri[2], pri[3], pri[4] = _P_OPEN, _P_LIMP, _P_VSR1, _P_VSR1, _P_VSRR
    pri[3] = np.array(_P_VSR1) * [1.15, 0.95, 1, 1]
    for s in range(3):
        b = 5 + 4 * s
        pri[b], pri[b + 1], pri[b + 2], pri[b + 3] = _P_NOBET, _P_VSH, _P_VSP, _P_VSRA
    pri /= pri.sum(axis=2, keepdims=True)
    return pri


def interp_weights(u: np.ndarray):
    """(lo bin index, weight of lo, weight of lo+1) for linear interpolation between bin centres."""
    x = np.clip((u - CENTERS[0]) / (CENTERS[1] - CENTERS[0]), 0.0, NBIN - 1 - 1e-9)
    lo = x.astype(np.int64)
    lo = np.minimum(lo, NBIN - 2)
    f = x - lo
    return lo, 1.0 - f, f


class OpponentModel:
    """Dirichlet counts over (class, bin, action); everything is plain numpy so it lives in memory."""

    def __init__(self, prior: np.ndarray | None = None, strength: float = 6.0):
        self.prior = default_prior() if prior is None else prior
        self.counts = self.prior * strength
        self.n_obs = np.zeros(NCLASS)
        self._cache_probs = None

    def probs(self) -> np.ndarray:
        if self._cache_probs is None:
            c = self.counts
            self._cache_probs = c / np.maximum(c.sum(axis=2, keepdims=True), 1e-12)
        return self._cache_probs

    def action_probs(self, cls: int, u: np.ndarray, legal) -> np.ndarray:
        """(K, len(legal)) probabilities of each legal action for hands with strengths u."""
        p = self.probs()[cls]                       # (NBIN, NACT)
        lo, wl, wh = interp_weights(u)
        pr = p[lo] * wl[:, None] + p[lo + 1] * wh[:, None]   # (K, NACT)
        pr = pr[:, list(legal)]
        pr = np.maximum(pr, 1e-4)
        return pr / pr.sum(axis=1, keepdims=True)

    def bin_dist(self, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Distribution over bins of strengths u with weights w (spread by interpolation)."""
        lo, wl, wh = interp_weights(u)
        d = np.bincount(lo, weights=w * wl, minlength=NBIN) + np.bincount(lo + 1, weights=w * wh, minlength=NBIN)
        s = d.sum()
        return d / s if s > 0 else np.full(NBIN, 1.0 / NBIN)

    def observe_known(self, cls: int, action: int, u: float, weight: float = 1.0):
        lo, wl, wh = interp_weights(np.array([u]))
        self.counts[cls, lo[0], action] += weight * wl[0]
        self.counts[cls, lo[0] + 1, action] += weight * wh[0]
        self.n_obs[cls] += weight
        self._cache_probs = None

    def observe_soft(self, cls: int, action: int, pi: np.ndarray, weight: float = 1.0):
        p = self.probs()[cls, :, action]
        post = pi * p
        s = post.sum()
        if s <= 0:
            return
        self.counts[cls, :, action] += weight * post / s
        self.n_obs[cls] += weight
        self._cache_probs = None

    def marginal(self, cls: int) -> np.ndarray:
        """Action frequencies for a uniformly distributed strength (for reports)."""
        return self.probs()[cls].mean(axis=0)
