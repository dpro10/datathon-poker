"""Check the C solver's showdown results (and river strength buckets) against the official evaluator."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import evaluator as off
off.warmup()
DT = np.dtype([("cards", np.uint8, 15), ("b", np.uint8, (2, 4)), ("result", np.int8)])
pool = np.fromfile(sys.argv[1], dtype=DT)
B3 = int(sys.argv[2]) if len(sys.argv) > 2 else 16
bad = 0
from sdbot import preflop as pf
pre = np.fromfile(os.path.join(HERE, "build", "pre.bin"), dtype=np.uint16)
badpre = 0
for d in pool[:3000]:
    c = [int(x) for x in d["cards"]]
    s0 = off.omaha_score(c[:5], c[10:]); s1 = off.omaha_score(c[5:10], c[10:])
    r = 1 if s0 > s1 else (-1 if s1 > s0 else 0)
    bad += r != int(d["result"])
    for p in range(2):
        badpre += pre[pf.colex_one(c[5 * p:5 * p + 5])] != d["b"][p][0]
print("deals", len(pool), "showdown mismatches", bad, "preflop bucket mismatches", badpre)
# river bucket sanity: correlation of bucket with exact river percentile computed by enumeration-free MC in python
rng = np.random.default_rng(1)
from sdbot import evaluator as ev
diffs = []
for d in pool[:300]:
    c = [int(x) for x in d["cards"]]
    board = np.array(c[10:])
    for p in range(2):
        h = c[5 * p:5 * p + 5]
        rem = np.array([x for x in range(36) if x not in h and x not in c[10:]])
        opp = np.array([rng.choice(rem, 5, replace=False) for _ in range(2000)])
        t = ev.board_tables(board[None])[:, 0]
        me = t[ev.hand_codes_one(h)].max(); th = t[ev.hand_codes(opp)].max(1)
        hs = np.mean(th < me) + 0.5 * np.mean(th == me)
        diffs.append(int(d["b"][p][3]) - min(B3 - 1, int(hs * B3)))
diffs = np.array(diffs)
print("river bucket vs python HS bucket: exact", np.mean(diffs == 0).round(3), "within 1", np.mean(np.abs(diffs) <= 1).round(3))
