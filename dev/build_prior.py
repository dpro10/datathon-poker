"""Build sdbot/data/prior.npy from calibration counts: the average (over opponents, equal
weight) of each opponent's smoothed P(action | class, bin)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from sdbot.model import CLASS_NAMES, default_prior  # noqa: E402

REALISTIC = ["tag", "eq_tight", "eq_loose", "eq_potodds", "eq_aggro", "eqbot", "nit", "diego_equity",
             "trapper", "adaptive", "lag", "bluffer"]


def main(path=os.path.join(HERE, "results", "calib.npz"), alpha=2.0):
    d = np.load(path)
    names, counts = list(d["names"]), d["counts"]
    hand = default_prior()
    acc = np.zeros_like(hand)
    n = 0
    for name in REALISTIC:
        if name not in names:
            print("missing", name)
            continue
        c = counts[names.index(name)] + alpha * hand
        acc += c / c.sum(axis=2, keepdims=True)
        n += 1
    prior = acc / n
    # actions that are never legal keep probability 0
    prior[:, :, 0] = np.where(hand[:, :, 0] == 0, 0.0, prior[:, :, 0])
    prior /= prior.sum(axis=2, keepdims=True)
    out = os.path.join(ROOT, "sdbot", "data", "prior.npy")
    np.save(out, prior.astype(np.float64))
    print("wrote", out, "from", n, "opponents")
    for c in range(len(CLASS_NAMES)):
        print(f"{CLASS_NAMES[c]:15s}", " | ".join("".join(f"{x:4.0%}" for x in prior[c, b]) for b in range(5)))


if __name__ == "__main__":
    main()
