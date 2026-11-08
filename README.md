# Datathon poker: tournament bot + practice arena

This repo is a valid submission for the TAMU Datathon **Heads Up** challenge (short-deck
5-card Pot-Limit Omaha). `agent.py` and `requirements.txt` sit at the root as the rules
require, next to unmodified copies of the organizers' `game.py`, `evaluator.py`,
`judge_engine.py`, `test_agent.py`, `baseline-bots/`, `tools/` and `docker/`
(from tamu-datathon-org/heads-up-bundle @ c83ab6e).

## The bot (`agent.py` + `sdbot/`)

`agent.py` keeps the organizers' server plumbing byte-for-byte; its three functions call
`sdbot/brain.py`, and every call is wrapped so a bug can only ever produce a safe legal
action, never an error.

| Piece | What it does |
| --- | --- |
| `sdbot/evaluator.py` | Vectorized evaluator: a 45 x 165 rank table (hole pair x board triple) reduced to a 225-entry table per board. Identical to `omaha_score` on every tested hand (146k checks), ~2000x faster in batches. |
| `sdbot/preflop.py` + `data/preflop.npz` | All 19,998 suit-classes of 5-card hands: equity vs a random hand, percentile, and equity vs each 5% slice of opponent hands (Monte Carlo, ~175k samples per class). |
| `sdbot/hand.py` | The opponent's range as ~500 weighted candidate hands; each opponent action reweights them by how likely that action is for each hand. Per street: our equity vs every candidate, each candidate's strength percentile and its blueprint bucket. |
| `sdbot/model.py` | Opponent model learned during the half: 31 situations (street, position, bet faced) x 5 strength bins x 4 actions. Action frequencies come from every hand, the strength behind each action from showdowns (so bluffs we folded to aren't explained away). Prior calibrated on a pool of practice bots. |
| `sdbot/blueprint.py` + `data/blueprint.npz` | An approximate equilibrium of the game from our own CFR solver (`dev/cfr/`: external-sampling MCCFR in C over the exact betting tree, 1.3B iterations, 128/128/128/64 card buckets). It is a second opponent model - what a solver would do with each candidate hand at the exact betting node - mixed with the learned one as `w * blueprint + (1 - w) * learned`. The weight is fitted to each opponent online (posterior over a grid from how well each `w` predicted every action, old evidence fading, prior leaning to 0): about 0.9 against a solver, 0.4 against an adaptive search bot, 0 against simple bots. |
| `sdbot/search.py` | Expectimax over the rest of the betting round, using the model for the opponent's replies; leaves estimate the rest of the hand with curves fitted on simulated play, blended between "equal opponent" and "weaker opponent" by how much we're winning. |
| `sdbot/brain.py` | Ties it together; a time governor (startup CPU benchmark + measured judge round trips) keeps the time the judge charges near or below 55% of the 8 s bank, with a cheap board-aware fallback strategy. |

Results in local testing (1000-deal duplicate matches, judge's 95% rule): it wins every
match against a 22-bot practice pool (baselines, equity-threshold bots, TAG/LAG/nit, a
balanced bluffer, adaptive exploiters). The blueprint opponent model (v8) fixed the previous version's one measured leak: v7 lost
0.26 bb/deal to the pure CFR blueprint, while the auto-weighted model is even with it over
40 seeds (+0.03) and slightly ahead of v7 (+0.07); the shipped settings (evidence fading,
prior leaning to 0) did as well or better on the 16 seeds where both were run, and cost
about 0.1 bb/deal of margin against the weakest practice bots. An opponent given a prior
fitted to the bot's own behaviour gains +0.13 against it (+0.68 against v7). Details,
methods and everything that was tried: `dev/RESULTS.md`.

Through the real judge over HTTP, with the bot in its tournament-like container (read-only,
non-root, 1 CPU, fresh container per half), v7's worst case (a calling station, ~4 decisions
per hand) was charged 3.2-3.5 s of the 8 s bank with 0 errors, and the official smoke test
passes. The blueprint model adds ~0.1-0.3 ms per decision; the time governor absorbs it.

**Before submitting:** set `PARTICIPANT` in `agent.py` to your team name.

```sh
pip install -r requirements-dev.txt
python -m pytest -q test_agent.py                        # official tests
python agent.py & python baseline-bots/baseline_random.py & python judge_engine.py --deals 200
python dev/pool_eval.py agent.py --seeds 1-2             # every practice bot, in-process (fast)
python dev/stress.py                                     # extreme opponents: errors / desyncs / timing
python dev/test_rules.py                                 # changed blinds / stacks / raise cap
dev/http_match.sh dev/opponents/tag.py 7                 # one match through the real judge over HTTP
dev/docker_match.sh dev/opponents/tag.py 7               # same, with the bot in its container
```

`dev/` (excluded from the Docker image) also holds the tools used to build the bot:
`harness.py` (in-process judge clone), `opponents/`, `precompute_preflop.py`,
`calibrate.py` + `build_prior.py` (opponent prior), `leaf_data.py` (leaf fits),
`snapshot.py` (freeze a version to play against), `sweep.py` (parameter A/B tests).

## Practice arena (older tooling)

A fair practice arena for the TAMU Datathon **Heads Up** challenge: short-deck (6 to A),
5-card Pot-Limit Omaha, two bots, round robin.

## Setup

```sh
pip install numpy            # Python 3.10+
python3 tests/test_all.py    # 17 checks, about 3 seconds
```

## Daily use

```sh
cp bots/equity_bot.py bots/diego.py          # start from the baseline (or bots/_template.py)
python3 -m arena check diego                 # errors, illegal moves, speed vs the practice bots
python3 -m arena match diego friend          # 2,000-hand match, same cards for both
python3 -m arena tournament                  # every bot in bots/ plays every other bot
python3 -m arena match diego equity_bot --seed 7   # same seed = same cards, to compare versions
```

Results show chips won, big blinds per 100 hands, a 95% range and a t-score.
Under t = 2 means the difference could still be luck, so play more hands.

## Playing your friend

**Same repo (simplest):** both bots go in `bots/`, then `python3 -m arena match diego friend`.

**Keep code private:** your friend runs their bot as a server on their laptop:

```sh
python3 -m arena serve bots/friend.py        # prints the address to use
```

and you run `python3 -m arena match diego http://<their-ip>:9001`. For full symmetry, both of
you serve and either one runs `python3 -m arena match http://<ip-a>:9001 http://<ip-b>:9002`.
If campus Wi-Fi blocks laptop-to-laptop connections, put both laptops on one phone hotspot.

## Ladder website

**Moved 2026-11-07:** the live ladder now runs on Vercel at https://testyourpokerbot.us (code in
`~/testyourpokerbot`: Vercel Sandbox VMs for matches, Neon Postgres for data). The Mac version below
is stopped; its data was imported there.

A site where people upload bots that play each other nonstop, with a live leaderboard.

```sh
python3 -m ladder serve --tunnel                      # prints the public link and the invite code
python3 -m ladder submit bots/diego.py --name diego-v2 # upload from the terminal
python3 -m ladder list                                # leaderboard in the terminal
python3 -m ladder retire <name>                       # host: take a bot off (restore puts it back)
python3 -m ladder stop
```

- Anyone with the link can watch; uploading needs the invite code.
- Every upload plays a 20-hand test first; broken bots are rejected with their error output.
- Uploaded bots run in a macOS sandbox: no internet, no files outside their own folder,
  none of your environment variables, CPU and file-size limits.
- The house practice bots stay on the ladder as a baseline. The starter kit people can
  download leaves out `equity_bot.py`.
- Data (bots, results, invite code) lives in `ladder_data/`, which is not committed.
- The public link works only while this Mac is awake and online (the server keeps it from
  idling to sleep, but closing the lid still sleeps it). The link changes on every restart.

## Why it's fair

- **Same cards:** every deal is played twice, the second time with the bots in each other's
  seats (same cards, same position). Card luck cancels, so the result measures decisions.
- **No memory between halves:** each half runs in brand-new bot processes.
- **Same clock:** both bots get the same time limit per decision (`--timeout`).
- **Hidden cards stay hidden:** a bot sees its own cards, the board, and the opponent's
  cards only at showdown.
- **Failures cost the bot that failed:** a slow, crashing or illegal answer is played as
  check (when free) or fold and reported. Three timeouts in a row stop the bot.
  Last year's official runner gave each match one try with no retries, so treat crashes as fatal.

Honor rule: bots must not save files or use the network to pass information between halves.

## Rules: CHECK THESE AGAINST THE OFFICIAL GUIDE

Defaults are guesses until we read the guide. Change them with flags (or in `pokerlib/rules.py`):

| Setting | Default | Flag |
| --- | --- | --- |
| Blinds | 1 / 2 | `--sb`, `--bb` |
| Ante | 0 | `--ante` |
| Stack | 200 (100 big blinds), reset every hand | `--stack` |
| Flush vs full house | flush wins | `--full-house-beats-flush` |
| Trips vs straight | trips win | `--straight-beats-trips` |
| Time per decision | 2 s | `--timeout` |

Fixed: A-6-7-8-9 is the lowest straight; a hand uses exactly 2 hole cards and 3 board cards.

## Writing a bot

A bot is a file with a `Bot` class. The arena calls `act(state)` on your turn; see
`bots/_template.py` for every field. Return `{"action": "fold" | "check" | "call" | "raise",
"amount": <raise-to total for this street>}`. Optional hooks: `new_match(info)`,
`hand_over(result)` (includes the opponent's cards after a showdown).

Helpers in `pokerlib`: `equity_vs_random(hole, board)` (about 3 ms for 500 samples),
`made_hand(hole, board)`, `get_evaluator(rules).best_omaha(hole, board)`.

## Practice bots

| Bot | Plays like |
| --- | --- |
| `calling_station` | never folds, never raises |
| `maniac` | raises the maximum every time |
| `random_bot` | random legal actions |
| `rock` | strong hands only, no bluffs |
| `equity_bot` | the baseline: win chance vs price, adjusts to aggression |

## Layout

- `pokerlib/`: cards, rules, hand evaluation (table of all 376,992 hands), equity.
- `arena/`: game engine, match runner, bot processes, HTTP serving, command line.
- `runs/`: summaries and each bot's printed output (`*.log`) for every run.
