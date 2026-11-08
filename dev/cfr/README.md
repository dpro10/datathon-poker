# CFR blueprint for heads-up short-deck 5-card PLO

An approximate equilibrium strategy for the datathon game, computed offline with
external-sampling Monte Carlo CFR over the **exact** betting tree (no bet-size abstraction:
the game only has fold / call / half-pot / pot with 4 raises per street, and stacks reset
every hand, so the tree is the same for every deal). Only the cards are abstracted.

| Step | Command | Output |
| --- | --- | --- |
| Betting tree | `python tree.py` | `tree.npz` (41,490 nodes, 15,952 decisions) |
| Tables + preflop clusters | `python export.py 128` | `build/tables.bin`, `build/pre.bin`, `build/pre_class_bucket.npy` |
| Build the solver | `gcc -O3 -march=native -o solver solver.c -lpthread -lm` | `solver` |
| Flop / turn buckets | `python fit_buckets.py 300000 128 128` | `build/cent_{flop,turn}_128.f32` |
| Deal pool | `./solver pool 4000000 build/pool_a.bin build/cent_flop_128.f32 128 build/cent_turn_128.f32 128 64` | 4M deals with buckets + showdown winner |
| Check the pool | `python verify_pool.py build/pool_a.bin 64` | 0 mismatches vs the official `omaha_score` |
| Train | `./solver train build/pool_a.bin 128 128 128 64 <seconds> 4 build/bp_a` | `build/bp_a.strategy.f32` (+ regrets/sums to resume) |
| Pack for the bot | `python export_blueprint.py build/bp_a 128 128 128 64` | `sdbot/data/blueprint.npz` |

Card abstraction (imperfect recall; the bot computes the same features live in `sdbot/hand.py`):

- **Preflop**: 128 k-means clusters of the 19,998 suit classes by their equity against each
  5% strength slice of opponent hands (`sdbot/data/preflop.npz`).
- **Flop / turn**: 128 k-means clusters of an 8-bin histogram of the hand's final strength
  (percentile among random opponent hands) over runouts: 24 random turn+river pairs on the
  flop, every river card on the turn. This separates draws from made hands with the same
  average equity.
- **River**: the strength percentile in 64 equal bins.

Training speed on 4 cores: about 330,000 iterations (two traversals each) per second; the
first 100M iterations are discounted linearly, as in Pluribus.

| Blueprint | Buckets (pre/flop/turn/river) | Info sets | Iterations | Pool | Where |
| --- | --- | --- | --- | --- | --- |
| bp_a | 128 / 128 / 128 / 64 | 1.31M | 1.29B (1 h) | 4M deals | `sdbot/data/blueprint.npz` |
| bp_b | 256 / 256 / 256 / 96 | 2.26M | 2.29B (2.5 h, shared CPU) | 6M deals | `sdbot/data/blueprint_b.npz` |

Since v10 the bot's opponent model averages the two blueprints' predictions (`bp_extra`): two
abstractions approximate the same equilibrium with different errors, so the average predicts
an unknown solver better than either alone.

Only the packed strategies are kept in git; the regret / strategy-sum files needed to resume
training (`build/bp_*.regret.f32`, `.sum.f64`) are large and stay local.

## Using the blueprint in the bot

* **Opponent model** (`bp_auto`, see `dev/RESULTS.md`): the search predicts the opponent's
  replies with a mixture of the blueprint (per candidate hand: its bucket from the same
  features, at the exact tree node) and the learned class model; the mixing weight is fitted
  online from how well each weight predicted the opponent's actions. `dev/test_blueprint_model.py`
  checks that the tree nodes the bot follows match the official game.
* **Playing the blueprint** (`mode: blueprint`, `bp_streets`) and **KL-regularized blends**
  with the search (`kl_kappa`, `blend_hi`): experiments, off by default.
* **Leaf values from blueprint self-play** (`build_leaf_table.py`, `leaf_bp`): the solver's
  `simulate` mode records every street transition of blueprint self-play; the table of
  "value beyond equity" by street / SPR band / position / hand-type groups did not help the
  search (it does not play the blueprint on later streets). Off by default; the table is not
  shipped (`python build_leaf_table.py build/sim_a.f32 128 128 128` rebuilds it).
