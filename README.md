# Datathon poker arena

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
