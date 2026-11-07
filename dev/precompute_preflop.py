"""Precompute preflop tables -> sdbot/data/preflop.npz

  cls_of_hand[colex]  suit-isomorphism class of every 5-card hand (uint16)
  evr[cls]            equity vs one uniformly random hand (random board)
  pct[cls]            percentile of evr over all 376,992 hands (0 = worst, 1 = best)
  eq_bins[cls, j]     equity vs a random hand whose percentile is in bin j (NB bins of 1/NB)

Method: one shared pool of random deals (opponent hand + board), board tables built
once; each class uses every pool deal that has no card in common with it.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
from sdbot import evaluator as ev  # noqa: E402
from preflop_classes import canonical_classes, colex  # noqa: E402

P = int(os.environ.get("POOL", 1_000_000))
NB = 20
G = {}


def build_pool(seed=12345):
    rng = np.random.default_rng(seed)
    deals = np.argsort(rng.random((P, 36)), axis=1)[:, :10]
    opp = np.sort(deals[:, :5], axis=1)
    board = deals[:, 5:]
    mask = np.zeros(P, dtype=np.uint64)
    for j in range(10):
        mask |= (np.uint64(1) << deals[:, j].astype(np.uint64))
    tables = np.empty((P, 225), dtype=np.int16)
    opp_score = np.empty(P, dtype=np.int16)
    CH = 50_000
    for s in range(0, P, CH):
        t = ev.board_tables(board[s:s + CH])          # (225, ch)
        tables[s:s + CH] = t.T
        oc = ev.hand_codes(opp[s:s + CH])
        opp_score[s:s + CH] = ev.scores_paired(t, oc)
    return opp, mask, tables, opp_score


def init(pool):
    G.update(pool)


def work(args):
    lo, hi = args
    reps, hands = G["reps"], G["hands"]
    mask, tables, opp_score = G["mask"], G["tables"], G["opp_score"]
    opp_bin = G.get("opp_bin")
    out_evr = np.zeros(hi - lo)
    out_bins = np.zeros((hi - lo, NB))
    out_n = np.zeros(hi - lo)
    for i, c in enumerate(range(lo, hi)):
        h = hands[reps[c]]
        hm = np.uint64(0)
        for card in h:
            hm |= np.uint64(1) << np.uint64(card)
        use = np.nonzero((mask & hm) == 0)[0]
        codes = ev.hand_codes(h[None, :])[0]
        mine = tables[use[:, None], codes[None, :]].max(axis=1)
        theirs = opp_score[use]
        res = (mine > theirs) + 0.5 * (mine == theirs)
        out_evr[i] = res.mean()
        out_n[i] = len(use)
        if opp_bin is not None:
            b = opp_bin[use]
            out_bins[i] = np.bincount(b, weights=res, minlength=NB) / np.maximum(np.bincount(b, minlength=NB), 1)
    return lo, out_evr, out_bins, out_n


def sweep(pool_dict, ncls, procs):
    chunks = [(s, min(s + 200, ncls)) for s in range(0, ncls, 200)]
    evr = np.zeros(ncls)
    bins = np.zeros((ncls, NB))
    nn = np.zeros(ncls)
    with Pool(procs, initializer=init, initargs=(pool_dict,)) as pp:
        for lo, e, b, n in pp.imap_unordered(work, chunks):
            evr[lo:lo + len(e)] = e
            bins[lo:lo + len(e)] = b
            nn[lo:lo + len(e)] = n
    return evr, bins, nn


def main():
    t0 = time.time()
    hands, cls, counts, reps = canonical_classes()
    ncls = len(counts)
    print(f"{ncls} classes ({time.time() - t0:.1f}s)", flush=True)
    opp, mask, tables, opp_score = build_pool()
    print(f"pool of {P} deals built ({time.time() - t0:.1f}s)", flush=True)
    base = {"hands": hands, "reps": reps, "mask": mask, "tables": tables, "opp_score": opp_score}
    procs = os.cpu_count() or 4
    evr, _, nn = sweep(base, ncls, procs)
    print(f"pass 1 done ({time.time() - t0:.1f}s); samples/class min {nn.min():.0f} mean {nn.mean():.0f}", flush=True)
    # percentile of each class over all hands (weighted by class size)
    order = np.argsort(evr, kind="stable")
    w = counts[order].astype(np.float64)
    cum = np.cumsum(w) - 0.5 * w
    pct = np.empty(ncls)
    pct[order] = cum / w.sum()
    opp_cls = cls[colex(opp)]
    opp_bin = np.minimum((pct[opp_cls] * NB).astype(np.int64), NB - 1)
    evr2, bins, nn = sweep({**base, "opp_bin": opp_bin}, ncls, procs)
    print(f"pass 2 done ({time.time() - t0:.1f}s)", flush=True)
    out = os.path.join(ROOT, "sdbot", "data", "preflop.npz")
    np.savez_compressed(out, cls_of_hand=cls.astype(np.uint16), evr=evr.astype(np.float32),
                        pct=pct.astype(np.float32), eq_bins=bins.astype(np.float32),
                        counts=counts.astype(np.uint8))
    print("wrote", out, os.path.getsize(out), "bytes")
    # quick report
    print("evr range", evr.min(), evr.max(), "median", np.median(np.repeat(evr, counts)))
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        print(f"  pct {q}: evr {np.interp(q, np.sort(pct), np.sort(evr)):.4f}")
    best = np.argsort(-evr)[:8]
    for c in best:
        print("  top", ev_str(hands[reps[c]]), f"{evr[c]:.4f}", np.round(bins[c][[0, 10, 19]], 3))
    worst = np.argsort(evr)[:4]
    for c in worst:
        print("  bottom", ev_str(hands[reps[c]]), f"{evr[c]:.4f}")


def ev_str(h):
    import evaluator as off
    return off.cards_to_str(sorted(h.tolist(), reverse=True))


if __name__ == "__main__":
    main()
