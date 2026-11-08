"""Per-hand tracking: the betting state, the opponent's range as weighted candidate
hands, and per-street equity / strength data.

Candidates are uniform random 5-card hands from the 31 cards we can't see; each has
a log-weight that accumulates log P(observed opponent action | its strength). A
candidate that collides with a board card is simply dropped (exact conditioning).
Preflop the range is kept over percentile bins; candidates are only built when the
hand reaches the flop (with the preflop actions replayed onto them).

Street data (computed once per street, the first time it's needed):
  e[k]   our equity vs candidate k over runouts of the remaining board
  evr[k] candidate k's equity vs a random hand (estimated by ranking it against
         the other candidates on every runout; flop/turn rank on a 251-level
         quantile scale of hand strength, the river ranks exactly)
  u[k]   percentile of evr[k] among candidates = the strength used by the model
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np

from . import evaluator as ev
from . import preflop as pf
from .betting import FLOP, PRE, RIVER, TURN, BState
from .model import classify

BOARD_N = (0, 3, 4, 5)
POOL_SIZE = 60_000
_rng0 = np.random.default_rng(20261107)
POS31 = np.sort(np.argsort(_rng0.random((POOL_SIZE, 31)), axis=1)[:, :5], axis=1)
PAIRS28 = np.array(list(combinations(range(28), 2)), dtype=np.int64)
BIT = np.array([1 << c for c in range(36)], dtype=np.uint64)
ALL36 = np.arange(36, dtype=np.uint64)
COARSE = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "coarse.npy")).astype(np.int64)
NO_COARSE = int(COARSE.max()) + 2
NO_EXACT = ev.NUM_ORD + 2


def bitmask_rows(cards: np.ndarray) -> np.ndarray:
    return np.bitwise_or.reduce(BIT[cards], axis=1)


def bitmask(cards) -> int:
    m = 0
    for c in cards:
        m |= 1 << int(c)
    return m


def cards_not_in(mask: int) -> np.ndarray:
    return np.nonzero(((np.uint64(mask) >> ALL36) & np.uint64(1)) == 0)[0]


class StreetData:
    __slots__ = ("street", "idx", "e", "evr", "u", "tables", "rmask", "cdf", "cnt", "nvalid", "evr_sorted",
                 "my_evr", "my_u", "K", "M", "probs", "coarse", "NO")


class HandTracker:
    def __init__(self, obs, n_cand: int, rng: np.random.Generator, flop_runouts: int, model):
        hole = sorted(int(c) for c in obs.hole_cards)
        self.index = obs.hand_index
        self.hole = np.array(hole, dtype=np.int64)
        self.hole_mask = bitmask(hole)
        self.me_button = bool(obs.is_button)
        self.st = BState(self.me_button)
        self.n_hist = 0
        self.rng = rng
        self.n_cand = n_cand
        self.flop_runouts = flop_runouts
        self.model = model
        self.cand = None                       # built lazily at the flop
        self.blogw = np.zeros(pf.NB)           # preflop range over percentile bins
        self.pre_recs: list[tuple] = []        # (cls, legal, j) of preflop opponent actions
        self.my_codes = ev.hand_codes_one(hole)
        self.my_cls = pf.class_of(hole)
        self.my_pre = float(pf.PCT[self.my_cls])
        self.sd: dict[int, StreetData] = {}
        self.board: list[int] = []
        self.opp_recs: list[dict] = []
        self.pending = None
        self.desync = False
        self.pre_probs: dict = {}

    # ----- candidates --------------------------------------------------------

    def _ensure_cands(self) -> None:
        if self.cand is not None:
            return
        rem = cards_not_in(self.hole_mask)                     # 31 cards
        start = int(self.rng.integers(0, POOL_SIZE - self.n_cand))
        self.cand = rem[POS31[start:start + self.n_cand]]
        self.cmask = bitmask_rows(self.cand)
        ci = pf.colex(self.cand)
        self.ccodes = pf.CODES_OF_HAND[ci].astype(np.intp)
        self.cpre = pf.PCT[pf.CLS_OF_HAND[ci]]
        self.logw = np.zeros(self.n_cand)
        for cls, legal, j in self.pre_recs:
            self.logw += np.log(self.model.action_probs(cls, self.cpre, legal)[:, j])

    # ----- street data -------------------------------------------------------

    def street_data(self, street: int) -> StreetData:
        d = self.sd.get(street)
        if d is not None:
            return d
        self._ensure_cands()
        n = BOARD_N[street]
        b = np.array(self.board[:n], dtype=np.int64)
        bmask = bitmask(b)
        idx = np.nonzero((self.cmask & np.uint64(bmask)) == 0)[0]
        rem = cards_not_in(self.hole_mask | bmask)
        if street == RIVER:
            runouts = b[None, :]
            rmask = np.zeros(1, dtype=np.uint64)
        elif street == TURN:
            runouts = np.empty((len(rem), 5), dtype=np.int64)
            runouts[:, :4] = b
            runouts[:, 4] = rem
            rmask = BIT[rem]
        else:
            m = min(self.flop_runouts, len(PAIRS28))
            pick = PAIRS28[self.rng.choice(len(PAIRS28), m, replace=False)]
            extra = rem[pick]
            runouts = np.empty((m, 5), dtype=np.int64)
            runouts[:, :3] = b
            runouts[:, 3:] = extra
            rmask = BIT[extra[:, 0]] | BIT[extra[:, 1]]
        tables = ev.board_tables(runouts)                       # (225, M)
        K, M = len(idx), runouts.shape[0]
        cs = tables[self.ccodes[idx]].max(axis=1)               # (K, M) int16
        ms = tables[self.my_codes].max(axis=0)                  # (M,)
        valid = (self.cmask[idx][:, None] & rmask[None, :]) == 0  # (K, M)
        nv = np.maximum(valid.sum(axis=1), 1)
        win2 = (ms > cs).astype(np.int16) * 2 + (ms == cs)      # 2 = win, 1 = tie
        e = (win2 * valid).sum(axis=1) / (2.0 * nv)
        # rank every candidate among the valid candidates on each runout via histograms
        coarse = street != RIVER
        NO = NO_COARSE if coarse else NO_EXACT
        cskey = COARSE[cs] if coarse else cs.astype(np.int64)
        offs = np.arange(M, dtype=np.int64) * NO
        keyed = cskey + offs[None, :]
        cnt = np.bincount(keyed[valid], minlength=M * NO)
        cdf = np.cumsum(cnt) - cnt
        cdf -= np.repeat(cdf[offs], NO)
        nvalid = K - (~valid).sum(axis=0)
        rank = (cdf[keyed] + 0.5 * (cnt[keyed] - 1)) / np.maximum(nvalid - 1, 1)
        evr = (rank * valid).sum(axis=1) / nv
        mk = (COARSE[ms] if coarse else ms.astype(np.int64)) + offs
        my_evr = float(np.mean((cdf[mk] + 0.5 * cnt[mk]) / np.maximum(nvalid, 1)))
        if street == RIVER:
            u = evr
        else:
            order = np.argsort(evr, kind="stable")
            u = np.empty(K)
            u[order] = np.arange(K) / max(K - 1, 1)
        d = StreetData()
        d.street, d.idx, d.e, d.evr, d.u, d.tables, d.rmask = street, idx, e, evr, u, tables, rmask
        d.cdf, d.cnt, d.nvalid, d.evr_sorted, d.K, d.M = cdf, cnt, nvalid, np.sort(evr), K, M
        d.coarse, d.NO, d.my_evr = coarse, NO, my_evr
        d.my_u = float(np.searchsorted(d.evr_sorted, my_evr) / max(K, 1))
        d.probs = {}
        self.sd[street] = d
        return d

    def strength_of(self, hand, street: int) -> float:
        """Percentile u of an arbitrary hand (e.g. the opponent's at showdown) on a street."""
        if street == PRE:
            return pf.pct_of(hand)
        d = self.sd.get(street)
        if d is None:
            d = self.street_data(street)
        codes = ev.hand_codes_one(sorted(hand))
        hs = d.tables[codes].max(axis=0)                       # (M,)
        valid = (d.rmask & np.uint64(bitmask(hand))) == 0
        if not valid.any():
            valid = np.ones_like(valid)
        k = (COARSE[hs] if d.coarse else hs.astype(np.int64)) + np.arange(d.M, dtype=np.int64) * d.NO
        rank = (d.cdf[k] + 0.5 * d.cnt[k]) / np.maximum(d.nvalid, 1)
        evr = float(rank[valid].mean())
        if street == RIVER:
            return evr
        lo = np.searchsorted(d.evr_sorted, evr, "left")
        hi = np.searchsorted(d.evr_sorted, evr, "right")
        return float((lo + 0.5 * (hi - lo)) / max(d.K, 1))

    # ----- range ---------------------------------------------------------------

    def probs_fn(self, street: int):
        """A cached P(action | strength) lookup for the search at this street."""
        model = self.model
        if street == PRE:
            cache, u = self.pre_probs, pf.BIN_CENTERS
        else:
            d = self.street_data(street)
            cache, u = d.probs, d.u

        def f(cls, legal):
            key = (cls, legal)
            P = cache.get(key)
            if P is None:
                P = model.action_probs(cls, u, legal)
                cache[key] = P
            return P
        return f

    def range_now(self, street: int):
        """(e, u, w) for the current street: equities, strengths and normalized weights."""
        if street == PRE:
            w = np.exp(self.blogw - self.blogw.max())
            return pf.EQ_BINS[self.my_cls], pf.BIN_CENTERS, w / w.sum()
        d = self.street_data(street)
        lw = self.logw[d.idx]
        w = np.exp(lw - lw.max())
        return d.e, d.u, w / w.sum()

    def _pi(self, street: int) -> np.ndarray:
        if street == PRE:
            return self.model.bin_dist(pf.BIN_CENTERS, np.exp(self.blogw - self.blogw.max()))
        d = self.street_data(street)
        lw = self.logw[d.idx]
        return self.model.bin_dist(d.u, np.exp(lw - lw.max()))

    # ----- replay ----------------------------------------------------------------

    def sync(self, obs) -> None:
        self.board = [int(c) for c in obs.board]
        hist = obs.action_history
        for rec in hist[self.n_hist:]:
            seat = 0 if rec.is_me else 1
            st = self.st
            if st.over or st.to_act != seat or int(rec.street) != st.street:
                self.desync = True
                break
            a = int(rec.action)
            if seat == 1:
                self._opp_action(a)
            added = st.apply(a)
            if added != rec.amount:
                self.desync = True
                break
        self.n_hist = len(hist)
        st = self.st
        if (st.over or st.to_act != 0 or st.street != int(obs.street) or st.pot != obs.pot
                or st.to_call(0) != obs.to_call or st.stacks[0] != obs.my_stack or st.stacks[1] != obs.opp_stack):
            self.desync = True

    def _opp_action(self, a: int) -> None:
        st = self.st
        cls = classify(st)
        legal = st.legal()
        if a not in legal:
            self.desync = True
            return
        j = legal.index(a)
        pi = self._pi(st.street)
        if st.street == PRE:
            self.blogw += np.log(self.probs_fn(PRE)(cls, legal)[:, j])
            self.pre_recs.append((cls, legal, j))
            if self.cand is not None:
                self.logw += np.log(self.model.action_probs(cls, self.cpre, legal)[:, j])
        else:
            d = self.street_data(st.street)
            self.logw[d.idx] += np.log(self.probs_fn(st.street)(cls, legal)[:, j])
        self.opp_recs.append({"cls": cls, "act": a, "street": st.street, "pi": pi})

    def note_my_action(self, a: int) -> None:
        """Remember the situation the opponent faces after our action, in case the hand
        ends before we see their reply (fold to our bet, check behind on the river, ...)."""
        st = self.st.copy()
        st.apply(a)
        self.pending = None
        if st.over or st.to_act != 1 or st.street != self.st.street:
            return
        self.pending = {"cls": classify(st), "street": st.street, "pi": self._pi(st.street),
                        "state": st, "facing": st.to_call(1) > 0}
