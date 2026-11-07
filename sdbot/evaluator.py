"""Fast vectorized hand evaluation for Short Deck 5-card PLO.

A five-card short-deck hand's score depends only on its five ranks and on whether
all five cards share a suit. An Omaha hand is always 2 hole cards + 3 board cards,
so we index the rank part by (hole-pair rank combo, board-triple rank combo):
45 x 165 = 7,425 entries, built once from the official evaluator's own scoring
function, so the ordering (and every tie) is exactly the official one.

Scores are ordinals: 0 = impossible, 1..N in the official order, so comparing two
ordinals is the same as comparing two official omaha_score values.

For a fixed full board, the best score of a hole PAIR is a function of
(suit state of the pair, pair rank combo) only: 5 x 45 = 225 values. We call that
the board table. A five-card holding is then scored on that board with 10 lookups
(one per hole pair) and a max, which numpy does for thousands of hands at once.
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

import evaluator as _official  # the organizers' evaluator.py (also in the tournament image)

NUM_CARDS = 36
NUM_RANKS = 9

# ---------------------------------------------------------------------------
# Index tables
# ---------------------------------------------------------------------------
PAIRS5 = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # 10 x 2
TRIPLES5 = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # 10 x 3
TRIPLES4 = np.array(list(combinations(range(4), 3)), dtype=np.int64)  # 4 x 3
_TRIPLE3 = np.array([[0, 1, 2]], dtype=np.int64)

_pair_list = [(a, b) for a in range(NUM_RANKS) for b in range(a, NUM_RANKS)]          # 45
_trip_list = [(a, b, c) for a in range(NUM_RANKS) for b in range(a, NUM_RANKS)
              for c in range(b, NUM_RANKS)]                                           # 165
NP, NT = len(_pair_list), len(_trip_list)

PAIR_IDX = np.zeros((NUM_RANKS, NUM_RANKS), dtype=np.int64)
for i, (a, b) in enumerate(_pair_list):
    PAIR_IDX[a, b] = PAIR_IDX[b, a] = i
TRIP_IDX = np.zeros((NUM_RANKS,) * 3, dtype=np.int64)
for i, t in enumerate(_trip_list):
    for p in set(permutations(t)):
        TRIP_IDX[p] = i


def _build_rank_tables():
    raw_nf = np.zeros((NP, NT), dtype=np.int64)
    raw_fl = np.zeros((NP, NT), dtype=np.int64)
    for i, (a, b) in enumerate(_pair_list):
        for j, (c, d, e) in enumerate(_trip_list):
            ranks = sorted((a, b, c, d, e), reverse=True)
            counts = np.bincount(ranks, minlength=NUM_RANKS)
            if counts.max() > 4:
                continue  # five of one rank cannot exist
            raw_nf[i, j] = _official._score_ranks(tuple(ranks), False)
            if counts.max() == 1:  # a flush needs five distinct ranks
                raw_fl[i, j] = _official._score_ranks(tuple(ranks), True)
    values = np.unique(np.concatenate([raw_nf[raw_nf > 0], raw_fl[raw_fl > 0]]))
    to_ord = {int(v): k + 1 for k, v in enumerate(values)}
    nf = np.vectorize(lambda v: to_ord.get(int(v), 0))(raw_nf).astype(np.int32)
    fl = np.vectorize(lambda v: to_ord.get(int(v), 0))(raw_fl).astype(np.int32)
    return nf, fl, values


NF, FL, ORD_VALUES = _build_rank_tables()   # NF/FL: (45, 165) ordinals; ORD_VALUES[k-1] = official score
NF16, FL16 = NF.astype(np.int16), FL.astype(np.int16)
NUM_ORD = len(ORD_VALUES)
CATEGORY_OF_ORD = np.concatenate([[-1], ORD_VALUES >> 20]).astype(np.int32)  # ordinal -> category 0..8

# Card helpers as arrays.
RANK = np.arange(NUM_CARDS) >> 2
SUIT = np.arange(NUM_CARDS) & 3


# ---------------------------------------------------------------------------
# Hole-card side
# ---------------------------------------------------------------------------

def hand_codes(hands: np.ndarray) -> np.ndarray:
    """hands: (N, 5) card ints -> (N, 10) pair codes in [0, 225).

    code = suit_state * 45 + pair_rank_idx, suit_state = the pair's suit if both
    cards share it, else 4.
    """
    hands = np.asarray(hands, dtype=np.int64)
    c1 = hands[:, PAIRS5[:, 0]]
    c2 = hands[:, PAIRS5[:, 1]]
    r1, r2 = c1 >> 2, c2 >> 2
    s1, s2 = c1 & 3, c2 & 3
    state = np.where(s1 == s2, s1, 4)
    return state * NP + PAIR_IDX[r1, r2]


def hand_codes_one(hand) -> np.ndarray:
    return hand_codes(np.asarray(hand, dtype=np.int64).reshape(1, 5))[0]


# ---------------------------------------------------------------------------
# Board side
# ---------------------------------------------------------------------------

def _triples_of(boards: np.ndarray, triples: np.ndarray):
    """boards: (M, k) -> (trip_rank_idx (M, T), trip_suit_state (M, T)) with state 4 = not monotone."""
    a = boards[:, triples[:, 0]]
    b = boards[:, triples[:, 1]]
    c = boards[:, triples[:, 2]]
    ti = TRIP_IDX[a >> 2, b >> 2, c >> 2]
    sa, sb, sc = a & 3, b & 3, c & 3
    ts = np.where((sa == sb) & (sb == sc), sa, 4)
    return ti, ts


def board_tables(boards: np.ndarray) -> np.ndarray:
    """boards: (M, k) with k in {3, 4, 5} -> (225, M) int16: best ordinal per pair code.

    Column m is board m's table. With k = 5 this is the showdown table; with
    k = 3 or 4 it gives the hand made right now (best 2 + 3 of the visible cards).
    """
    boards = np.asarray(boards, dtype=np.int64)
    k = boards.shape[1]
    triples = TRIPLES5 if k == 5 else (TRIPLES4 if k == 4 else _TRIPLE3)
    ti, ts = _triples_of(boards, triples)            # (M, T)
    M = ti.shape[0]
    best_nf = NF16[:, ti].max(axis=2)                # (45, M)
    out = np.empty((5, NP, M), dtype=np.int16)
    out[:] = best_nf[None, :, :]
    m_idx, t_idx = np.nonzero(ts < 4)                # flush-capable (monotone) triples are rare
    if m_idx.size:
        s_idx = ts[m_idx, t_idx]
        vals = FL16[:, ti[m_idx, t_idx]]             # (45, n)
        for j in range(m_idx.size):
            col = out[s_idx[j], :, m_idx[j]]
            np.maximum(col, vals[:, j], out=col)
            out[s_idx[j], :, m_idx[j]] = col
    return out.reshape(5 * NP, M)


def scores(tables: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """tables: (225, M) from board_tables, codes: (N, 10) -> (N, M) ordinals."""
    return tables[codes].max(axis=1)


def scores_paired(tables: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """tables: (225, N), codes: (N, 10) -> (N,) ordinal of hand i on board i."""
    N = tables.shape[1]
    return tables[codes, np.arange(N)[:, None]].max(axis=1)


def score_one(hole, board) -> int:
    """Ordinal of one hand on one board (3, 4 or 5 cards)."""
    t = board_tables(np.asarray(board, dtype=np.int64).reshape(1, -1))
    return int(t[hand_codes_one(hole), 0].max())


def official_from_ord(o: int) -> int:
    return int(ORD_VALUES[o - 1]) if o > 0 else 0
