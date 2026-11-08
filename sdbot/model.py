"""Opponent model: P(action | situation class, hand-strength percentile).

Situation classes (from the acting player's point of view; IP = the button, who acts
last after the flop):
  preflop
    0 SB open                  1 BB vs a limp            2 BB vs a half-pot open
    3 BB vs a pot open         4 SB vs a raise after limping
    5 SB vs a 3-bet            6 facing any further raise
  flop / turn / river: 7 + 8 * (street - 1) + k with k =
    0 OOP first to act (lead or check)      1 IP after a check (bet or check back)
    2 IP facing a half-pot lead             3 IP facing a pot lead
    4 OOP facing a half-pot bet (after checking)   5 OOP facing a pot bet
    6 OOP facing a raise (after leading)    7 IP facing a raise (check-raise, ...)

Strength u in [0, 1] is the percentile of the hand's equity vs a random hand on
the current board (preflop: the precomputed preflop percentile). The model keeps
Dirichlet counts per (class, bin, action) with NBIN bins over u; probabilities
are linearly interpolated between bin centres so likelihoods are smooth in u.
"""
from __future__ import annotations

import numpy as np

from .betting import FLOP, PRE, BState, HALF, POT

NBIN = 5
NPRE = 7
NPOST = 8
NCLASS = NPRE + 3 * NPOST    # 31
NACT = 4
CENTERS = (np.arange(NBIN) + 0.5) / NBIN
CLASS_NAMES = ["PRE_SB_OPEN", "PRE_BB_VS_LIMP", "PRE_BB_VS_HALF", "PRE_BB_VS_POT", "PRE_SB_VS_RAISE",
               "PRE_SB_VS_3BET", "PRE_VS_MORE"] + [
    f"{s}_{k}" for s in ("FLOP", "TURN", "RIVER") for k in
    ("OOP_FIRST", "IP_CHECKED_TO", "IP_VS_HALF", "IP_VS_POT", "OOP_VS_HALF", "OOP_VS_POT",
     "OOP_VS_RAISE", "IP_VS_RAISE")]


def classify(st: BState) -> int:
    """Class of the decision facing st.to_act."""
    ip = st.button == st.to_act
    if st.street == PRE:
        if st.raises == 0:
            return 0 if ip else 1          # the button is the small blind
        if st.raises == 1:
            if ip:
                return 4
            return 3 if st.last_raise == POT else 2
        if st.raises == 2 and ip:
            return 5
        return 6
    base = NPRE + NPOST * (st.street - FLOP)
    if st.raises == 0:
        return base + (1 if ip else 0)
    if st.raises == 1:
        pot = st.last_raise == POT
        if ip:
            return base + (3 if pot else 2)
        return base + (5 if pot else 4)
    return base + (7 if ip else 6)


# Prior policies (probabilities over F, C, H, P by strength bin, weak -> strong).
_P_OPEN = [[.55, .35, .03, .07], [.30, .50, .05, .15], [.15, .55, .08, .22], [.07, .45, .10, .38], [.03, .30, .10, .57]]
_P_LIMP = [[0, .90, .03, .07], [0, .85, .05, .10], [0, .78, .07, .15], [0, .65, .08, .27], [0, .45, .10, .45]]
_P_VSR1 = [[.60, .35, .02, .03], [.40, .52, .03, .05], [.25, .62, .04, .09], [.12, .65, .06, .17], [.05, .55, .10, .30]]
_P_VSRR = [[.72, .26, .00, .02], [.55, .42, .01, .02], [.38, .57, .01, .04], [.20, .70, .02, .08], [.08, .62, .04, .26]]
_P_NOBET = [[0, .80, .08, .12], [0, .80, .10, .10], [0, .70, .15, .15], [0, .50, .20, .30], [0, .35, .20, .45]]
_P_VSH = [[.75, .20, .01, .04], [.50, .45, .01, .04], [.30, .62, .03, .05], [.12, .70, .06, .12], [.04, .50, .10, .36]]
_P_VSP = [[.80, .17, .01, .02], [.58, .38, .01, .03], [.38, .55, .03, .04], [.18, .68, .04, .10], [.05, .55, .10, .30]]
_P_VSRA = [[.85, .14, .00, .01], [.70, .28, .00, .02], [.50, .46, .01, .03], [.28, .64, .02, .06], [.08, .60, .07, .25]]


def load_prior() -> np.ndarray:
    """The calibrated prior (dev/build_prior.py) if present, else the hand-made one."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "prior.npy")
    try:
        p = np.load(path)
        if p.shape == (NCLASS, NBIN, NACT) and np.isfinite(p).all():
            return p / p.sum(axis=2, keepdims=True)
    except (OSError, ValueError):
        pass
    return default_prior()


_PRIOR = None


def prior() -> np.ndarray:
    global _PRIOR
    if _PRIOR is None:
        _PRIOR = load_prior()
    return _PRIOR


def default_prior() -> np.ndarray:
    pri = np.zeros((NCLASS, NBIN, NACT))
    pri[0], pri[1], pri[2], pri[3], pri[4], pri[5], pri[6] = _P_OPEN, _P_LIMP, _P_VSR1, _P_VSR1, _P_VSR1, _P_VSRR, _P_VSRR
    pri[3] = np.array(_P_VSR1) * [1.15, 0.95, 1, 1]
    for s in range(3):
        b = NPRE + NPOST * s
        pri[b], pri[b + 1] = _P_NOBET, _P_NOBET
        pri[b + 2], pri[b + 3], pri[b + 4], pri[b + 5] = _P_VSH, _P_VSP, _P_VSH, _P_VSP
        pri[b + 6], pri[b + 7] = _P_VSRA, _P_VSRA
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

    def __init__(self, prior_probs: np.ndarray | None = None, strength: float = 6.0):
        self.prior = prior() if prior_probs is None else prior_probs
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


class DecoupledModel(OpponentModel):
    """P(action | class) from every observation; P(strength bin | action, class) only from
    showdowns (where we see the hand); combined with Bayes' rule:

        P(a | b, c) = P(a | c) P(b | a, c) / sum_a' P(a' | c) P(b | a', c)

    Unlike soft EM, a bet we folded to does not get attributed to strong hands just
    because the prior says bets are strong, so bluff-heavy opponents are learned from
    the showdowns we do see. Folds never show down, so their strength profile stays the
    prior's (weak hands fold).
    """

    def __init__(self, prior_probs: np.ndarray | None = None, strength: float = 6.0,
                 act_strength: float = 8.0, bin_strength: float = 2.0):
        super().__init__(prior_probs, strength)
        pri = self.prior                                       # (C, B, A)
        pb = np.full(NBIN, 1.0 / NBIN)
        marg = (pri * pb[None, :, None]).sum(axis=1)            # (C, A) under uniform strengths
        self.act = marg * act_strength                          # (C, A)
        shape = pri * pb[None, :, None]                         # (C, B, A) ~ P(b, a)
        shape = shape / np.maximum(shape.sum(axis=1, keepdims=True), 1e-12)   # P(b | a)
        self.binc = np.transpose(shape, (0, 2, 1)) * bin_strength              # (C, A, B)
        # prior pseudo-counts stay fixed; observed counts can fade (see decay)
        self.act0, self.binc0 = self.act.copy(), self.binc.copy()
        self.act_d = np.zeros_like(self.act)
        self.binc_d = np.zeros_like(self.binc)

    def decay(self, gamma: float) -> None:
        """Fade old observations (gamma < 1 per hand), so a change of style is re-learned."""
        if gamma >= 1.0:
            return
        self.act_d *= gamma
        self.binc_d *= gamma
        self.act = self.act0 + self.act_d
        self.binc = self.binc0 + self.binc_d
        self._cache_probs = None

    def probs(self) -> np.ndarray:
        if self._cache_probs is None:
            pa = self.act / np.maximum(self.act.sum(axis=1, keepdims=True), 1e-12)          # (C, A)
            pba = self.binc / np.maximum(self.binc.sum(axis=2, keepdims=True), 1e-12)       # (C, A, B)
            joint = pa[:, :, None] * pba                                                    # (C, A, B)
            theta = np.transpose(joint, (0, 2, 1))                                          # (C, B, A)
            theta = theta / np.maximum(theta.sum(axis=2, keepdims=True), 1e-12)
            self._cache_probs = theta
        return self._cache_probs

    def observe_known(self, cls: int, action: int, u: float, weight: float = 1.0):
        lo, wl, wh = interp_weights(np.array([u]))
        for arr in (self.act, self.act_d):
            arr[cls, action] += weight
        for arr in (self.binc, self.binc_d):
            arr[cls, action, lo[0]] += weight * wl[0]
            arr[cls, action, lo[0] + 1] += weight * wh[0]
        self.n_obs[cls] += weight
        self._cache_probs = None

    def observe_soft(self, cls: int, action: int, pi: np.ndarray, weight: float = 1.0):
        self.act[cls, action] += weight
        self.act_d[cls, action] += weight
        self.n_obs[cls] += weight
        self._cache_probs = None
