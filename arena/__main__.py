"""python3 -m arena <match | tournament | check | serve> ..."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from pokerlib import Rules

from .match import (RUNS_DIR, default_bots, format_match, resolve_bot, run_check, run_match,
                    run_tournament, safe_name)


def _add_common(p: argparse.ArgumentParser, hands: int) -> None:
    p.add_argument("--hands", type=int, default=hands,
                   help=f"hands per match (default {hands}); every deal is played from both seats")
    p.add_argument("--timeout", type=float, default=2.0,
                   help="seconds allowed per decision (default 2.0); set this to the official limit")
    p.add_argument("--seed", type=int, help="card seed: the same seed deals the same cards")
    p.add_argument("--save-hands", action="store_true", help="write every hand to the run folder")
    g = p.add_argument_group("rules (set these to match the official guide)")
    g.add_argument("--sb", type=int, default=1, help="small blind (default 1)")
    g.add_argument("--bb", type=int, default=2, help="big blind (default 2)")
    g.add_argument("--ante", type=int, default=0, help="ante per player (default 0)")
    g.add_argument("--stack", type=int, default=200, help="starting stack every hand (default 200)")
    g.add_argument("--straight-beats-trips", action="store_true",
                   help="use if the guide ranks a straight above three of a kind (default: trips win)")
    g.add_argument("--full-house-beats-flush", action="store_true",
                   help="use if the guide ranks a full house above a flush (default: flush wins)")


def _rules(args) -> Rules:
    return Rules(small_blind=args.sb, big_blind=args.bb, ante=args.ante, starting_stack=args.stack,
                 flush_beats_full_house=not args.full_house_beats_flush,
                 trips_beat_straight=not args.straight_beats_trips)


def _run_dir(label: str):
    return RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{label}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m arena",
                                 description="Fair heads-up arena for short-deck 5-card Pot-Limit Omaha bots.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("match", help="play two bots against each other")
    m.add_argument("bot_a", help="a .py file, a name from bots/, or http://host:port")
    m.add_argument("bot_b")
    _add_common(m, 2000)

    t = sub.add_parser("tournament", help="round robin (default: every bot in bots/)")
    t.add_argument("bots", nargs="*")
    t.add_argument("--jobs", type=int, default=min(4, max(1, (os.cpu_count() or 2) // 2)),
                   help="matches to run at the same time")
    _add_common(t, 1000)

    c = sub.add_parser("check", help="test a bot against the practice bots before submitting")
    c.add_argument("bot")
    _add_common(c, 200)

    s = sub.add_parser("serve", help="host your bot so a friend can play it without seeing your code")
    s.add_argument("bot")
    s.add_argument("--port", type=int, default=9001)
    s.add_argument("--host", default="0.0.0.0", help="0.0.0.0 makes it reachable on your network (default)")

    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from .server import serve

        spec = resolve_bot(args.bot)
        if spec.kind != "file":
            raise SystemExit("serve needs a bot file")
        serve(spec.target, args.host, args.port)
        return 0

    rules = _rules(args)
    if args.cmd == "match":
        a, b = resolve_bot(args.bot_a), resolve_bot(args.bot_b)
        out = _run_dir(f"{safe_name(a.name)}_vs_{safe_name(b.name)}")
        r = run_match(a, b, rules, args.hands, args.seed, args.timeout, out_dir=out,
                      save_hands=args.save_hands, progress=sys.stderr.isatty())
        print(format_match(r))
        return 0
    if args.cmd == "tournament":
        specs = [resolve_bot(x) for x in args.bots] if args.bots else default_bots()
        run_tournament(specs, rules, args.hands, args.seed, args.timeout, jobs=args.jobs,
                       out_dir=_run_dir("tournament"), save_hands=args.save_hands)
        return 0
    if args.cmd == "check":
        spec = resolve_bot(args.bot)
        return run_check(spec, rules, args.hands, args.timeout, args.seed,
                         out_dir=_run_dir(f"check_{safe_name(spec.name)}"))
    return 2


if __name__ == "__main__":
    sys.exit(main())
