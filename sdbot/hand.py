"""Per-hand tracking: the betting state, the opponent's range as weighted candidate
hands, and per-street equity / strength data.

Candidates are uniform random 5-card hands from the 31 cards we can't see; each has
a log-weight that accumulates log P(observed opponent action | its strength). A
candidate that collides with a board card is simply dropped (exact conditioning).

Street data (computed once per street, the first time it's needed):
  e[k]   our equity vs candidate k over runouts of the remaining board
  evr[k] candidate k's equity vs a random hand (estimated by ranking it against
         the other candidates on every runout)
  u[k]   percentile of evr[k] among candidates = the strength used by the model
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from . import evaluator as ev
from . import preflop as pf
from .betting import CALL, FLOP, FOLD, PRE, RIVER, TURN, BState
from .model import classify

BOARD_N = (0, 3, 4, 5)
POOL_SIZE = 60_000
_rng0 = np.random.default_rng(20261107)
POS31 = np.sort(np.argsort(_rng0.random((POOL_SIZE, 31)), axis=1)[:, :5], axis=1)
PAIRS28 = np.array(list(combinations(range(28), 2)), dtype=np.int64)
ONE = np.uint64(1)
BIT = np.array([1 << c for c in range(36)], dtype=np.uint64)
OFF = 2048  # > number of ordinals, used to sort many columns at once


def bitmask_rows(cards: np.ndarray) -> np.ndarray:
    return np.bitwise_or.reduce(BIT[cards], axis=1)


def bitmask(cards) -> int:
    m = 0
    for c in cards:
        m |= 1 << int(c)
    return m


class StreetData:
    __slots__ = ("street", "idx", "e", "evr", "u", "tables", "rmask", "cdf", "cnt", "nvalid", "evr_sorted",
                 "my_evr", "my_u", "K", "M", "probs")


class HandTracker:
    def __init__(self, obs, n_cand: int, rng: np.random.Generator, flop_runouts: int):
        hole = sorted(int(c) for c in obs.hole_cards)
        self.index = obs.hand_index
        self.hole = np.array(hole, dtype=np.int64)
        self.hole_mask = bitmask(hole)
        self.me_button = bool(obs.is_button)
        self.st = BState(self.me_button)
        self.n_hist = 0
        self.rng = rng
        self.flop_runouts = flop_runouts
        rem = np.array([c for c in range(36) if not (self.hole_mask >> c) & 1], dtype=np.int64)
        start = int(rng.integers(0, POOL_SIZE - n_cand))
        self.cand = rem[POS31[start:start + n_cand]]
        self.cmask = bitmask_rows(self.cand)
        self.ccodes = ev.hand_codes(self.cand)
        self.cpre = pf.pct_many(self.cand)
        self.logw = np.zeros(n_cand)
        self.blogw = np.zeros(pf.NB)          # preflop range over percentile bins
        self.my_codes = ev.hand_codes_one(hole)
        self.my_cls = pf.class_of(hole)
        self.my_pre = float(pf.PCT[self.my_cls])
        self.sd: dict[int, StreetData] = {}
        self.board: list[int] = []
        self.opp_recs: list[dict] = []
        self.my_actions: list[tuple] = []
        self.pending = None
        self.desync = False
        self.pre_probs: dict = {}

    # ----- street data -----------------------------------------------------------

    def street_data(self, street: int) -> StreetData:
        d = self.sd.get(street)
        if d is not None:
            return d
        n = BOARD_N[street]
        b = np.array(self.board[:n], dtype=np.int64)
        bmask = np.uint64(bitmask(b))
        idx = np.nonzero((self.cmask & bmask) == 0)[0]
        known = self.hole_mask | int(bmask)
        rem = np.array([c for c in range(36) if not (known >> c) & 1], dtype=np.int64)
        if street == RIVER:
            runouts = b[None, :]
            rmask = np.zeros(1, dtype=np.uint64)
        elif street == TURN:
            runouts = np.concatenate([np.broadcast_to(b, (len(rem), 4)), rem[:, None]], axis=1)
            rmask = BIT[rem]
        else:
            m = min(self.flop_runouts, len(PAIRS28))
            pick = PAIRS28[self.rng.choice(len(PAIRS28), m, replace=False)]
            extra = rem[pick]
            runouts = np.concatenate([np.broadcast_to(b, (m, 3)), extra], axis=1)
            rmask = BIT[extra[:, 0]] | BIT[extra[:, 1]]
        tables = ev.board_tables(runouts)                       # (225, M)
        K, M = len(idx), runouts.shape[0]
        cs = tables[self.ccodes[idx]].max(axis=1).astype(np.int32)   # (K, M)
        ms = tables[self.my_codes].max(axis=0).astype(np.int32)      # (M,)
        valid = (self.cmask[idx][:, None] & rmask[None, :]) == 0      # (K, M)
        nv = np.maximum(valid.sum(axis=1), 1)
        win = (ms[None, :] > cs) + 0.5 * (ms[None, :] == cs)
        e = (win * valid).sum(axis=1) / nv
        # rank every candidate among the valid candidates on each runout, via per-runout
        # histograms of the ordinals: cdf[m, o] = #valid candidates scoring below o on runout m
        NO = ev.NUM_ORD + 2
        offs = np.arange(M, dtype=np.int64) * NO
        keyed = cs + offs[None, :]                               # (K, M)
        cnt = np.bincount(keyed[valid], minlength=M * NO)         # (M*NO,)
        cdf = np.cumsum(cnt) - cnt                                # exclusive prefix sums...
        cdf = cdf - np.repeat(cdf[offs], NO)                      # ...restarted on every runout
        nvalid = K - (~valid).sum(axis=0)                         # (M,)
        less = cdf[keyed]
        eq = cnt[keyed] - 1
        rank = (less + 0.5 * eq) / np.maximum(nvalid - 1, 1)[None, :]
        evr = (rank * valid).sum(axis=1) / nv
        mk = ms.astype(np.int64) + offs
        my_evr = float(np.mean((cdf[mk] + 0.5 * cnt[mk]) / np.maximum(nvalid, 1)))
        if street == RIVER:
            u = evr
        else:
            order = np.argsort(evr, kind="stable")
            u = np.empty(K)
            u[order] = np.arange(K) / max(K - 1, 1)
        evr_sorted = np.sort(evr)
        d = StreetData()
        d.street, d.idx, d.e, d.evr, d.u, d.tables, d.rmask = street, idx, e, evr, u, tables, rmask
        d.cdf, d.cnt, d.nvalid, d.evr_sorted, d.K, d.M = cdf, cnt, nvalid, evr_sorted, K, M
        d.probs = {}
        d.my_evr = my_evr
        d.my_u = float(np.searchsorted(evr_sorted, my_evr) / max(K, 1))
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
        hs = d.tables[codes].max(axis=0).astype(np.int64)                 # (M,)
        hm = np.uint64(bitmask(hand))
        valid = (d.rmask & hm) == 0
        if not valid.any():
            valid = np.ones_like(valid)
        NO = ev.NUM_ORD + 2
        k = hs + np.arange(d.M, dtype=np.int64) * NO
        rank = (d.cdf[k] + 0.5 * d.cnt[k]) / np.maximum(d.nvalid, 1)
        evr = float(rank[valid].mean())
        if street == RIVER:
            return evr
        lo = np.searchsorted(d.evr_sorted, evr, "left")
        hi = np.searchsorted(d.evr_sorted, evr, "right")
        return float((lo + 0.5 * (hi - lo)) / max(d.K, 1))

    # ----- range -----------------------------------------------------------------

    def probs_fn(self, street: int, model):
        """A cached P(action | strength) lookup for the search at this street."""
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

    # ----- replay ---------------------------------------------------------------

    def sync(self, obs, model) -> None:
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
                self._opp_action(a, model)
            else:
                self.my_actions.append((st.street, a))
            added = st.apply(a)
            if added != rec.amount:
                self.desync = True
                break
        self.n_hist = len(hist)
        st = self.st
        if (st.over or st.to_act != 0 or st.street != int(obs.street) or st.pot != obs.pot
                or st.to_call(0) != obs.to_call or st.stacks[0] != obs.my_stack or st.stacks[1] != obs.opp_stack):
            self.desync = True

    def _opp_action(self, a: int, model) -> None:
        st = self.st
        cls = classify(st)
        legal = st.legal()
        if a not in legal:
            self.desync = True
            return
        j = legal.index(a)
        if st.street == PRE:
            wb = np.exp(self.blogw - self.blogw.max())
            pi = model.bin_dist(pf.BIN_CENTERS, wb)
            self.logw += np.log(model.action_probs(cls, self.cpre, legal)[:, j])
            self.blogw += np.log(self.probs_fn(PRE, model)(cls, legal)[:, j])
        else:
            d = self.street_data(st.street)
            lw = self.logw[d.idx]
            w = np.exp(lw - lw.max())
            pi = model.bin_dist(d.u, w)
            self.logw[d.idx] += np.log(self.probs_fn(st.street, model)(cls, legal)[:, j])
        self.opp_recs.append({"cls": cls, "act": a, "street": st.street, "pi": pi})

    def note_my_action(self, a: int, model) -> None:
        """Remember the situation the opponent faces after our action, in case the hand
        ends before we see their reply (fold to our bet, check behind on the river, ...)."""
        st = self.st.copy()
        st.apply(a)
        self.pending = None
        if st.over or st.to_act != 1 or st.street != self.st.street:
            return
        cls = classify(st)
        if st.street == PRE:
            wb = np.exp(self.blogw - self.blogw.max())
            pi = model.bin_dist(pf.BIN_CENTERS, wb)
        else:
            d = self.street_data(st.street)
            lw = self.logw[d.idx]
            pi = model.bin_dist(d.u, np.exp(lw - lw.max()))
        self.pending = {"cls": cls, "street": st.street, "pi": pi, "state": st, "facing": st.to_call(1) > 0}
