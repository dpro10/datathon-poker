"""Suit-isomorphism classes of 5-card hands and the colex index (shared by precompute + runtime)."""
from itertools import combinations, permutations
from math import comb

import numpy as np

BINOM = np.array([[comb(n, k) for k in range(6)] for n in range(37)], dtype=np.int64)


def colex(sorted_hands: np.ndarray) -> np.ndarray:
    """(N, 5) ascending card ints -> (N,) index in [0, 376992)."""
    h = sorted_hands
    return (BINOM[h[:, 0], 1] + BINOM[h[:, 1], 2] + BINOM[h[:, 2], 3]
            + BINOM[h[:, 3], 4] + BINOM[h[:, 4], 5])


def all_hands() -> np.ndarray:
    return np.array(list(combinations(range(36), 5)), dtype=np.int64)


def canonical_classes():
    hands = all_hands()                      # (376992, 5), each row ascending
    idx = colex(hands)
    order = np.argsort(idx)
    hands = hands[order]                     # now row i is the hand with colex index i
    assert (colex(hands) == np.arange(len(hands))).all()
    ranks, suits = hands >> 2, hands & 3
    best = None
    for perm in permutations(range(4)):
        p = np.array(perm)
        mapped = np.sort((ranks << 2) | p[suits], axis=1)
        key = mapped[:, 0] * 36 ** 4 + mapped[:, 1] * 36 ** 3 + mapped[:, 2] * 36 ** 2 + mapped[:, 3] * 36 + mapped[:, 4]
        best = key if best is None else np.minimum(best, key)
    uniq, cls, counts = np.unique(best, return_inverse=True, return_counts=True)
    reps = np.zeros(len(uniq), dtype=np.int64)
    reps[cls] = np.arange(len(hands))        # any member works as representative
    return hands, cls.astype(np.int32), counts, reps


if __name__ == "__main__":
    hands, cls, counts, reps = canonical_classes()
    print("hands", len(hands), "classes", len(counts), "max class size", counts.max(), "min", counts.min())
