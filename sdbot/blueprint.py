"""Equilibrium blueprint from the CFR solver (dev/cfr): lookup of the strategy for the
current betting node and our card bucket.

data/blueprint.npz holds
  child   (nodes, 4) int32   the exact betting tree (seat 0 = button), -1 = illegal
  street  (nodes,)   int8
  offset  (nodes,)   int64   first info set of each decision node (-1 for terminals)
  strat   (infosets, 4) uint8  average strategy, probabilities * 255 (255,255,255,255 = never reached)
  buckets (4,)       int32   B0..B3
  pre     (classes,) uint16  preflop bucket of every suit class
  cflop / cturn (B1 / B2, 8) float32  centroids of the final-strength histograms
Bucket features match the solver: flop/turn = 8-bin histogram of our strength (percentile
among random hands) over runouts; river = that strength in B3 equal bins.
"""
from __future__ import annotations

import os

import numpy as np

from . import preflop as pf

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "blueprint.npz")


class Blueprint:
    def __init__(self, path: str = _PATH):
        d = np.load(path)
        self.child = d["child"].astype(np.int64)
        self.street = d["street"].astype(np.int64)
        self.offset = d["offset"].astype(np.int64)
        self.strat = d["strat"]
        self.B = [int(x) for x in d["buckets"]]
        self.pre = d["pre"].astype(np.int64)
        self.cflop = d["cflop"].astype(np.float64)
        self.cturn = d["cturn"].astype(np.float64)

    def node(self, actions) -> int:
        n = 0
        for a in actions:
            n = int(self.child[n, a])
            if n < 0:
                return -1
        return n

    def bucket(self, street: int, my_cls: int | None = None, my_hs=None) -> int:
        if street == 0:
            return int(self.pre[my_cls])
        if street == 3:
            return min(self.B[3] - 1, int(float(my_hs[0]) * self.B[3]))
        h = np.bincount(np.minimum((np.asarray(my_hs) * 8).astype(np.int64), 7), minlength=8) / max(len(my_hs), 1)
        C = self.cflop if street == 1 else self.cturn
        return int(((C - h[None, :]) ** 2).sum(axis=1).argmin())

    def probs(self, node: int, bucket: int):
        """Probabilities over (FOLD, CALL, HALF, POT), or None if the info set was never trained."""
        if node < 0 or self.offset[node] < 0:
            return None
        row = self.strat[self.offset[node] + bucket].astype(np.float64)
        if row.min() >= 255 and row.max() >= 255 and row.sum() >= 1020:
            return None
        legal = self.child[node] >= 0
        p = np.where(legal, row, 0.0)
        s = p.sum()
        return p / s if s > 0 else None


_BP = None


def get() -> Blueprint | None:
    global _BP
    if _BP is None and os.path.exists(_PATH):
        try:
            _BP = Blueprint()
        except Exception:  # noqa: BLE001 - a broken file must never break the bot
            _BP = None
    return _BP
