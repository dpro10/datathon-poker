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
