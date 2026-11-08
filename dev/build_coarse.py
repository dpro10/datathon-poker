"""Quantile map from hand ordinals (1..NUM_ORD) to 0..250, for cheap approximate ranking.
Built from the distribution of best Omaha hands over random holdings and boards."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from sdbot import evaluator as ev
rng = np.random.default_rng(99)
n = 400_000
deals = np.argsort(rng.random((n, 36)), axis=1)[:, :10]
t = ev.board_tables(deals[:, 5:])
o = ev.scores_paired(t, ev.hand_codes(deals[:, :5]))
counts = np.bincount(o, minlength=ev.NUM_ORD + 1).astype(np.float64)
counts[1:] += 0.01                           # every real ordinal gets a level
cdf = np.cumsum(counts) - 0.5 * counts
cdf /= counts.sum()
coarse = np.clip(np.round(cdf * 250), 0, 250).astype(np.int16)
coarse[0] = 0
# keep the order strict where it matters most: never map a better hand below a worse one
coarse = np.maximum.accumulate(coarse)
np.save(os.path.join(ROOT, "sdbot", "data", "coarse.npy"), coarse)
print("levels used", len(np.unique(coarse)), "of", ev.NUM_ORD)
