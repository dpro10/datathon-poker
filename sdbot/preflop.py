"""Preflop lookups from data/preflop.npz (built by dev/precompute_preflop.py).

pct   percentile of a hand's equity vs a random hand (0 = worst, 1 = best)
eqb   equity vs a random hand from each of NB percentile bins (bin j = [j/NB, (j+1)/NB))
"""
from __future__ import annotations

import os
from math import comb

import numpy as np

_DATA = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "preflop.npz"))
CLS_OF_HAND = _DATA["cls_of_hand"].astype(np.int32)   # (376992,)
EVR = _DATA["evr"].astype(np.float64)                 # (classes,)
PCT = _DATA["pct"].astype(np.float64)                 # (classes,)
EQ_BINS = _DATA["eq_bins"].astype(np.float64)         # (classes, NB)
NB = EQ_BINS.shape[1]
BIN_CENTERS = (np.arange(NB) + 0.5) / NB

BINOM = np.array([[comb(n, k) for k in range(6)] for n in range(37)], dtype=np.int64)


def colex(sorted_hands: np.ndarray) -> np.ndarray:
    h = sorted_hands
    return (BINOM[h[:, 0], 1] + BINOM[h[:, 1], 2] + BINOM[h[:, 2], 3]
            + BINOM[h[:, 3], 4] + BINOM[h[:, 4], 5])


def colex_one(cards) -> int:
    c = sorted(cards)
    return int(BINOM[c[0], 1] + BINOM[c[1], 2] + BINOM[c[2], 3] + BINOM[c[3], 4] + BINOM[c[4], 5])


def class_of(cards) -> int:
    return int(CLS_OF_HAND[colex_one(cards)])


def pct_of(cards) -> float:
    return float(PCT[class_of(cards)])


def pct_many(sorted_hands: np.ndarray) -> np.ndarray:
    return PCT[CLS_OF_HAND[colex(sorted_hands)]]


def _all_hand_codes() -> np.ndarray:
    """Pair codes of every 5-card hand, indexed by colex (uint8, 376992 x 10)."""
    from itertools import combinations
    from . import evaluator as ev
    hands = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    out = np.empty((len(hands), 10), dtype=np.uint8)
    out[colex(hands)] = ev.hand_codes(hands)
    return out


CODES_OF_HAND = _all_hand_codes()
