"""Hand strength for short-deck Omaha.

All 376,992 five-card hands from the 36-card deck get a precomputed strength value
(higher is better), stored in a table indexed by the hand's combinatorial index.
The table takes about a second to build and is cached in pokerlib/.cache/.

Omaha rule: a player's hand is the best of the 100 ways to combine exactly
2 of their 5 hole cards with exactly 3 of the 5 board cards.
"""
from __future__ import annotations

import os
import threading
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from .cards import NUM_CARDS, parse_cards
from .rules import Rules

NUM_HANDS = comb(NUM_CARDS, 5)  # 376,992
CATEGORY_NAMES = ("high_card", "pair", "two_pair", "trips", "straight",
                  "flush", "full_house", "quads", "straight_flush")
CATEGORY_BASE = 16 ** 5  # value = category_strength * CATEGORY_BASE + tiebreak

BINOM = np.array([[comb(n, k) for k in range(6)] for n in range(NUM_CARDS + 1)], dtype=np.int64)
_BINOM_LIST = BINOM.tolist()
_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_TABLE_VERSION = 1


def _cs(a, b):
    return np.minimum(a, b), np.maximum(a, b)


def sort5(c0, c1, c2, c3, c4):
    """Elementwise sorting network for five arrays; returns them in ascending order."""
    c0, c1 = _cs(c0, c1)
    c3, c4 = _cs(c3, c4)
    c2, c4 = _cs(c2, c4)
    c2, c3 = _cs(c2, c3)
    c1, c4 = _cs(c1, c4)
    c0, c3 = _cs(c0, c3)
    c0, c2 = _cs(c0, c2)
    c1, c3 = _cs(c1, c3)
    c1, c2 = _cs(c1, c2)
    return c0, c1, c2, c3, c4


def colex5(c0, c1, c2, c3, c4):
    """Index in [0, NUM_HANDS) of a sorted 5-card set c0 < c1 < c2 < c3 < c4."""
    return BINOM[c0, 1] + BINOM[c1, 2] + BINOM[c2, 3] + BINOM[c3, 4] + BINOM[c4, 5]


def _build_table(rules: Rules) -> np.ndarray:
    combos = np.array(list(combinations(range(NUM_CARDS), 5)), dtype=np.int64)
    n = len(combos)
    ranks = combos // 4
    suits = combos % 4
    rows = np.arange(n)

    counts = np.zeros((n, 9), dtype=np.int64)
    for j in range(5):
        counts[rows, ranks[:, j]] += 1
    present = counts > 0
    bitmask = (present.astype(np.int64) << np.arange(9)).sum(axis=1)

    is_flush = (suits == suits[:, :1]).all(axis=1)
    straight_high = np.full(n, -1, dtype=np.int64)
    for low in range(5):  # 6-T, 7-J, 8-Q, 9-K, T-A
        straight_high[bitmask == (0b11111 << low)] = low + 4
    straight_high[bitmask == ((1 << 8) | 0b1111)] = 3  # A-6-7-8-9 plays as the lowest straight
    is_straight = straight_high >= 0

    desc = -np.sort(-counts, axis=1)
    c1, c2 = desc[:, 0], desc[:, 1]
    code = {name: i for i, name in enumerate(CATEGORY_NAMES)}
    natural = np.full(n, code["high_card"], dtype=np.int64)
    natural[(c1 == 2) & (c2 == 1)] = code["pair"]
    natural[(c1 == 2) & (c2 == 2)] = code["two_pair"]
    natural[(c1 == 3) & (c2 == 1)] = code["trips"]
    natural[is_straight & ~is_flush] = code["straight"]
    natural[is_flush & ~is_straight] = code["flush"]
    natural[(c1 == 3) & (c2 == 2)] = code["full_house"]
    natural[c1 == 4] = code["quads"]
    natural[is_flush & is_straight] = code["straight_flush"]

    order = rules.category_order()
    strength = np.array([order.index(name) for name in CATEGORY_NAMES], dtype=np.int64)[natural]

    # Tiebreak: distinct ranks ordered by (count desc, rank desc), base 16, ranks stored as 1..9.
    score = np.where(present, counts * 16 + np.arange(9), -1)
    by_score = np.argsort(-score, axis=1, kind="stable")[:, :5]
    top = np.take_along_axis(score, by_score, axis=1)
    digits = np.where(top >= 0, by_score + 1, 0)
    tiebreak = (digits[:, 0] * 16 ** 4 + digits[:, 1] * 16 ** 3 + digits[:, 2] * 16 ** 2
                + digits[:, 3] * 16 + digits[:, 4])
    tiebreak = np.where(is_straight, straight_high + 1, tiebreak)

    table = np.zeros(NUM_HANDS, dtype=np.int32)
    table[colex5(*combos.T)] = strength * CATEGORY_BASE + tiebreak
    return table


_lock = threading.Lock()
_tables: dict[str, np.ndarray] = {}


def load_table(rules: Rules) -> np.ndarray:
    tag = rules.table_tag()
    with _lock:
        if tag in _tables:
            return _tables[tag]
        path = _CACHE_DIR / f"rank_table_{tag}_v{_TABLE_VERSION}.npy"
        table = None
        if path.exists():
            try:
                table = np.load(path)
                if table.shape != (NUM_HANDS,):
                    table = None
            except Exception:
                table = None
        if table is None:
            table = _build_table(rules)
            try:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                with open(tmp, "wb") as f:
                    np.save(f, table)
                os.replace(tmp, path)
            except OSError:
                pass
        _tables[tag] = table
        return table


class Evaluator:
    def __init__(self, rules: Rules | None = None):
        self.rules = rules or Rules()
        self.table = load_table(self.rules)
        self._order = self.rules.category_order()
        self._pairs = np.array(list(combinations(range(self.rules.hole_cards), 2)), dtype=np.int64)
        self._triples = {b: np.array(list(combinations(range(b), 3)), dtype=np.int64) for b in (3, 4, 5)}

    def eval5(self, cards) -> int:
        a, b, c, d, e = sorted(int(x) for x in cards)
        B = _BINOM_LIST
        return int(self.table[B[a][1] + B[b][2] + B[c][3] + B[d][4] + B[e][5]])

    def best_omaha(self, hole, board) -> int:
        """Best value using exactly 2 hole cards and 3 board cards (board needs 3 to 5 cards)."""
        hole = parse_cards(hole)
        board = parse_cards(board)
        if len(board) < 3:
            raise ValueError("need at least 3 board cards")
        best = -1
        for pair in combinations(hole, 2):
            for trip in combinations(board, 3):
                v = self.eval5(pair + trip)
                if v > best:
                    best = v
        return best

    def best_omaha_batch(self, holes, boards) -> np.ndarray:
        """Vectorized best_omaha. holes: (n, hole_cards) ints, boards: (n, 3..5) ints."""
        holes = np.asarray(holes, dtype=np.int64)
        boards = np.asarray(boards, dtype=np.int64)
        hp = holes[:, self._pairs]                     # (n, P, 2)
        bt = boards[:, self._triples[boards.shape[1]]]  # (n, T, 3)
        n, p, t = holes.shape[0], hp.shape[1], bt.shape[1]
        cols = [np.broadcast_to(hp[:, :, None, k], (n, p, t)) for k in range(2)]
        cols += [np.broadcast_to(bt[:, None, :, k], (n, p, t)) for k in range(3)]
        values = self.table[colex5(*sort5(*cols))]
        return values.reshape(n, p * t).max(axis=1)

    def category(self, value: int) -> str:
        return self._order[int(value) // CATEGORY_BASE]


_evaluators: dict[Rules, Evaluator] = {}


def get_evaluator(rules: Rules | None = None) -> Evaluator:
    rules = rules or Rules()
    ev = _evaluators.get(rules)
    if ev is None:
        ev = _evaluators[rules] = Evaluator(rules)
    return ev
