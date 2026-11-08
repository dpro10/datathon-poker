# Experiment log

What was tried, how it was measured, and what it showed. Newest sections last.

## How to read the numbers

* A match is 1000 deals x 2 halves (duplicate: same cards, seats swapped), scored in big
  blinds per deal (bb/deal) for the first bot. The judge calls a WIN when the 95% interval
  of the paired-deal mean excludes 0.
* `dev/pool_eval.py` prints, per opponent, the mean over matches and **the average per-match
  standard error** (`se`, about 0.35 bb/deal for two strong bots). The uncertainty of a mean
  over n matches is about `se / sqrt(n)`: 16 matches -> +-0.09, 24 -> +-0.07.
* `HARNESS_BANK=40` gives every bot a 40 s bank so that a busy test machine does not push the
  time governors into cheaper levels (strategy comparisons stay fair). Leave it unset to
  mirror the judge.
* `HARNESS_ALLIN_EV=1` scores hands that went all-in before the river by their exact equity
  over the remaining board (sampled preflop) instead of the one runout dealt: unbiased for
  expected winnings, lower variance (checked against brute force with the official evaluator).
* Snapshots (`dev/snapshot.py`) freeze a version as `dev/versions/<name>/agent_<name>.py`;
  `v7` is the search bot, `bp1` the pure CFR blueprint (128/128/128/64 buckets, 1.29B iterations).

## Baselines

| Bot | vs pool (22 bots, seeds 1-2) | vs v7 | vs bp1 |
| --- | --- | --- | --- |
| v7 (search + learned model) | 44/44 W, +5.78 avg | 0 | -0.26 (16 seeds) |
| bp1 (pure blueprint) | 41/44 W, +1.76 avg | +0.30 (16 seeds) | 0 |

Exploitation is worth +0.2 to +11 bb/deal over the blueprint against every practice bot
(calling station +14.3 vs +3.0, sticky raiser +10.1 vs +1.7, eq_loose +8.8 vs +1.1). On the
public ladder v7 (maroon-sd) went 381W 32C 0L; the pure blueprint (maroon-bp) only +0.25
vs the strongest ladder bot where v7 had +0.83. The search must stay; its leak is against
equilibrium-style opponents.

## Blueprint self-play leaf values (dev/cfr/build_leaf_table.py) - rejected

Street-end leaves = pot x (equity + value beyond equity under blueprint self-play), by
SPR band, position and hand-type groups. After fixing the fit (column effects were summed
over every row, giving -12 pot values), the table is sane (+-0.4 pot) but the search with it
loses: -0.63 vs v7, -0.54 vs bp1 (8 seeds). Equilibrium continuation values do not describe
how the search itself plays the later streets. Option `leaf_bp` (off).

## Blueprint opponent model (`bp_model`, `bp_auto`)

The opponent's action probabilities at each search node come from a mixture
`w * blueprint(node, candidate bucket) + (1 - w) * learned class model`.

| Variant | vs bp1 | vs v7 | pool |
| --- | --- | --- | --- |
| w = 1 fixed (8 seeds) | +0.01 | -2.21 | - |
| w = 0.5 fixed (8 seeds) | -0.09 | +0.30 | - |
| `bp_auto`, one weight (24 seeds) | **+0.11** | **+0.06** | 44/44, +5.65 (seeds 1-2) |
| `bp_auto` + `bp_streetwise` (24 seeds) | -0.07 | -0.02 | 44/44, +5.67 (seeds 1-2) |

`bp_auto` fits w as the posterior mean over a 21-point grid from how well each w predicted
every opponent action (marginalized over our range for the opponent's hand). Typical fits:
0.9 vs bp1, 0.4 vs v7 (the mixture predicts v7 better than either model alone), 0.02 vs
pool bots. A model that only ever assumed the blueprint (w = 1) is crushed by v7, which is
why the weight must be learned.

Follow-ups (seeds as noted; `bp_auto` = one global weight, flat prior, no decay):

* **Pool cost.** Over pool seeds 1-4 (4 matches per bot), `bp_auto` averages -0.13 bb/deal
  below v7 (se 0.05 across the 22 bots; biggest drops vs adaptive, lag, always_half, bluffer).
  Traces show why: early in the half the learned model is still its prior, so the blueprint
  predicts some simple bots better (0.78 weight after 10 hands vs always_half, whose half-pot
  raises look like the blueprint's favourite open) and that early evidence keeps the weight
  up for ~100 hands, while the blueprint's predictions in the spots that matter (folds) are
  wrong for that bot. Candidate fix under test: fade old evidence (`bp_decay` 0.98 per hand)
  and a prior ~exp(-4 w) on the weight (`bp_prior_k`).
* **Informed exploiter** (`dev/exploit_test.py`: a search bot given a prior fitted to the
  candidate's real behaviour): gains +0.13 vs `bp_auto` (12 seeds), against +0.68 vs v7.
* `sigma_bp` (12 seeds): -0.06 vs bp1, +0.07 vs v7 - no better than plain `bp_auto`; off.
* `leaf_bp: auto` (12 seeds): +0.02 vs bp1, +0.20 vs v7, slower - inconclusive; off.

Ideas not yet tried (for later):

* Finer strength bins at the top of the learned model (e.g. edges 0, .2, .4, .6, .8, .9,
  .95, 1): "bets the river only with the nuts" is invisible to 5 equal bins. Needs the prior
  rebuilt (calibrate.py / build_prior.py bin the same way as `model.interp_weights`).
* Spend more of the time bank on big pots (a richer search level when the pot is large).
* River abstraction by equity against opponent hand clusters (blockers / nut potential) for
  the next blueprint; longer training on more cores.

## v8: the new default (05:35 UTC, Nov 8)

`bp_auto` + `bp_decay` 0.98 + `bp_prior_k` 4 ("D") became the default, snapshot `v8`.

| Test | v7 | `bp_auto` (A) | D = v8 |
| --- | --- | --- | --- |
| vs bp1, seeds 25-40 (all-in EV) | - | -0.09 | -0.06 |
| vs v7, seeds 25-40 (all-in EV) | 0 | +0.10 | +0.14 |
| 8-bot pool subset, seeds 7-10 | +4.43 | +4.37 | +4.34 |
| ... vs always_half alone | +6.82 | +6.10 | +6.84 |

A over 40 seeds: +0.03 vs bp1, +0.07 vs v7. The decay + prior removes the always_half
loss (the blueprint no longer holds weight for the first ~100 hands against simple bots);
the remaining ~0.1 bb/deal margin cost against weak bots is within 2 se and changes no match
outcome (every practice match is still won). Note that the learned model's prior was
calibrated on these same practice bots, which favours the learned model in this test.

Experiments started before 05:35 used the v7 defaults unless their parameters say otherwise
(round 4b's river-smoothing run started at the switch; treat it as A-based).

* v8 through the official judge over HTTP (`dev/http_match.sh`, calling station, seed 7,
  machine load ~8 on 4 cores): charged 3.82 s / 4.50 s of the 8 s bank, 0 errors, +14.1.
* `bp_river_smooth` (seeds 25-40, all-in EV): -0.04 vs bp1, +0.09 vs v7 - same as without
  (A: -0.09 / +0.10, D: -0.06 / +0.14); off.
* Bigger search everywhere (`l0`: 900 candidates, 40 flop runouts, ~2.5x the time; v8
  defaults, seeds 25-40, bank 40): +0.06 vs bp1, +0.18 vs v7 (v8: -0.06 / +0.14). More
  compute helps a little but does not fit the 8 s bank everywhere -> `rich` option: the
  bigger search only in pots of 16+ chips at the flop, while the governor shows slack.
* `share` 10 (cross-street tendency sharing) on top of v8, pool seeds 5-6: +0.09 bb/deal
  (se 0.07 over 22 bots; +0.03 without lag's +1.23). Follow-up queued.
* v8 + KL blend toward the blueprint (`blend_hi` 1, `blend_wr0` 0.5, `blend_n0` 100,
  `blend_full` 0.4), seeds 25-40: +0.21 vs v7 but -0.17 vs bp1 (2 losses in 16), and +3.88
  on the 8-bot subset (v8 +4.34) - rejected.
