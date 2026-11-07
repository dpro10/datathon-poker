"""Run: python3 tests/test_all.py   (or pytest, if installed)"""
import itertools
import random
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from arena.engine import Hand, IllegalAction  # noqa: E402
from arena.match import BotSpec, resolve_bot, run_match  # noqa: E402
from pokerlib import CATEGORY_BASE, Rules, equity_vs_random, get_evaluator, parse_cards  # noqa: E402
from pokerlib.evaluator import sort5  # noqa: E402

EV = get_evaluator(Rules())


# ---------- hand evaluation ----------
def test_sort_network():
    for perm in itertools.permutations(range(5)):
        out = sort5(*[np.array([v]) for v in perm])
        assert [int(c[0]) for c in out] == [0, 1, 2, 3, 4]


def test_category_counts():
    order = Rules().category_order()
    u, c = np.unique(EV.table // CATEGORY_BASE, return_counts=True)
    counts = {order[k]: int(v) for k, v in zip(u, c)}
    assert counts == {"straight_flush": 24, "quads": 288, "full_house": 1728, "flush": 480,
                      "straight": 6120, "trips": 16128, "two_pair": 36288, "pair": 193536,
                      "high_card": 122400}
    assert (EV.table > 0).all()


def test_default_rank_order():
    flush = EV.eval5(parse_cards("6h 8h 9h Jh Kh"))
    full_house = EV.eval5(parse_cards("As Ah Ad Ks Kd"))
    trips = EV.eval5(parse_cards("6s 6h 6d 7c 8c"))
    broadway = EV.eval5(parse_cards("Ts Jh Qd Kc Ac"))
    assert flush > full_house and trips > broadway


def test_flipped_rank_order():
    ev = get_evaluator(Rules(flush_beats_full_house=False, trips_beat_straight=False))
    assert ev.eval5(parse_cards("As Ah Ad Ks Kd")) > ev.eval5(parse_cards("6h 8h 9h Jh Kh"))
    assert ev.eval5(parse_cards("6s 7h 8d 9c Tc")) > ev.eval5(parse_cards("As Ah Ad 7c 8c"))


def test_ace_low_straight():
    wheel = EV.eval5(parse_cards("As 6h 7d 8c 9s"))
    six_high = EV.eval5(parse_cards("6s 7h 8d 9c Ts"))
    assert EV.category(wheel) == "straight" and wheel < six_high
    assert EV.category(EV.eval5(parse_cards("Ah 6h 7h 8h 9h"))) == "straight_flush"


def test_kickers():
    assert EV.eval5(parse_cards("Ks Kh Ad 9c 7c")) > EV.eval5(parse_cards("Ks Kh Qd Jc Tc"))
    assert EV.eval5(parse_cards("Ks Kh 7d 7c Ac")) > EV.eval5(parse_cards("Ks Kh 7d 7c Qc"))
    assert EV.eval5(parse_cards("Ks Kh 7d 7c Ac")) == EV.eval5(parse_cards("Kc Kd 7s 7h Ad"))


def test_omaha_uses_exactly_two_hole_cards():
    # Four spades in hand + one on board is not a flush; A-K-Q-J-T would need four hole cards.
    v = EV.best_omaha("As Ks Qs Js 6d", "Ts 9h 8h 7c 6c")
    assert v == EV.eval5(parse_cards("Qs Js Ts 9h 8h"))
    # Four hearts on board + one in hand is not a flush either.
    assert EV.category(EV.best_omaha("Ah Ks Qd Jc 7s", "6h 8h 9h Th Kd")) != "flush"
    # Two hearts in hand + three on board is.
    assert EV.category(EV.best_omaha("Ah 7h Qd Jc 7s", "6h 8h 9c Th Kd")) == "flush"


def test_batch_matches_single():
    rng = random.Random(1)
    holes, boards, singles = [], [], []
    for _ in range(300):
        d = rng.sample(range(36), 10)
        holes.append(d[:5])
        boards.append(d[5:])
        singles.append(EV.best_omaha(d[:5], d[5:]))
    assert EV.best_omaha_batch(np.array(holes), np.array(boards)).tolist() == singles


def test_equity_matches_exact_enumeration():
    hole, board = parse_cards("Ah Kh 9c 8d 7s"), parse_cards("Qh Jh 6c 6d Tc")
    rest = [c for c in range(36) if c not in hole + board]
    opps = np.array(list(itertools.combinations(rest, 5)))
    assert len(opps) == comb(26, 5)
    boards = np.broadcast_to(np.array(board), (len(opps), 5))
    mine = EV.best_omaha_batch(np.broadcast_to(np.array(hole), (len(opps), 5)), boards)
    theirs = EV.best_omaha_batch(opps, boards)
    exact = float(np.mean((mine > theirs) + 0.5 * (mine == theirs)))
    mc = equity_vs_random(hole, board, n=20000, rng=np.random.default_rng(3))
    assert abs(mc - exact) < 4 * 0.5 / np.sqrt(20000), (mc, exact)
    assert equity_vs_random("Js Ts 9c 8c 7c", "As Ks Qs 7h 6d", n=500) == 1.0  # royal flush


# ---------- betting engine ----------
def test_pot_limit_amounts():
    h = Hand(Rules(), EV, list(range(36)), button=0)
    lg = h.legal(0)
    assert lg["actions"] == ["fold", "call", "raise"] and lg["to_call"] == 1
    assert (lg["min_raise_to"], lg["max_raise_to"]) == (4, 6)  # pot raise from the small blind is to 6
    h.act(0, "call")
    lg = h.legal(1)
    assert lg["actions"] == ["check", "raise"] and (lg["min_raise_to"], lg["max_raise_to"]) == (4, 6)
    h.act(1, "check")
    assert h.street == 1 and len(h.board) == 3 and h.to_act == 1  # big blind acts first after the flop
    lg = h.legal(1)
    assert (lg["min_raise_to"], lg["max_raise_to"]) == (2, 4)    # bet the pot = 4
    h.act(1, "raise", 4)
    lg = h.legal(0)
    assert lg["to_call"] == 4 and (lg["min_raise_to"], lg["max_raise_to"]) == (8, 16)
    try:
        h.act(0, "raise", 17)
        raise AssertionError("raise above pot accepted")
    except IllegalAction:
        pass


def test_fold_preflop():
    h = Hand(Rules(), EV, list(range(36)), button=1)
    h.act(1, "fold")
    assert h.finished and h.net == [1, -1] and not h.showdown


def test_all_in_runs_out_the_board():
    h = Hand(Rules(), EV, list(range(36)), button=0)
    while not h.finished:
        lg = h.legal(h.to_act)
        if "raise" in lg["actions"]:
            h.act(h.to_act, "raise", lg["max_raise_to"])
        else:
            h.act(h.to_act, "call")
    assert h.showdown and len(h.board) == 5 and abs(h.net[0]) in (0, 200) and sum(h.net) == 0


def test_random_play_conserves_chips():
    rng = random.Random(7)
    rules = Rules()
    for i in range(4000):
        deck = list(range(36))
        rng.shuffle(deck)
        h = Hand(rules, EV, deck, button=i % 2, hand_id=i)
        while not h.finished:
            s = h.to_act
            lg = h.legal(s)
            a = rng.choice(lg["actions"])
            amt = rng.randint(lg["min_raise_to"], lg["max_raise_to"]) if a == "raise" else None
            h.act(s, a, amt)
            assert min(h.stacks) >= 0
        assert sum(h.net) == 0 and max(abs(x) for x in h.net) <= rules.starting_stack


# ---------- arena (real bot processes) ----------
def test_duplicate_mirror_is_exactly_even():
    r = run_match(resolve_bot("calling_station"), resolve_bot("calling_station"), Rules(), hands=200, seed=5)
    assert r["net_a"] == 0 and r["halves_a"][0] == -r["halves_a"][1]
    assert all(h["timeouts"] == h["errors"] == h["illegal"] == 0 for h in r["health"].values())


def test_practice_bots_run_clean():
    r = run_match(resolve_bot("equity_bot"), resolve_bot("maniac"), Rules(), hands=100, seed=1)
    for h in r["health"].values():
        assert h["timeouts"] == h["errors"] == h["illegal"] == 0 and not h["crashed"], h


def _temp_bot(folder: Path, name: str, body: str) -> BotSpec:
    p = folder / f"{name}.py"
    p.write_text(textwrap.dedent(body))
    return BotSpec("file", str(p), name)


def test_bad_bots_are_contained():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        raiser = _temp_bot(d, "raises_error", """
            class Bot:
                def act(self, s):
                    if s["hand_id"] % 2:
                        raise ValueError("boom")
                    return {"action": "check" if "check" in s["legal"] else "call"}
        """)
        illegal = _temp_bot(d, "illegal", """
            class Bot:
                def act(self, s):
                    return {"action": "raise", "amount": 1}
        """)
        dies = _temp_bot(d, "dies", """
            import os
            class Bot:
                def act(self, s):
                    if s["hand_id"] == 3:
                        os._exit(1)
                    return {"action": "check" if "check" in s["legal"] else "call"}
        """)
        slow = _temp_bot(d, "slow", """
            import time
            class Bot:
                def act(self, s):
                    time.sleep(1.0)
                    return {"action": "call"}
        """)
        station = resolve_bot("calling_station")
        r = run_match(raiser, station, Rules(), hands=20, seed=1)
        assert r["health"]["raises_error"]["errors"] > 0
        r = run_match(illegal, station, Rules(), hands=20, seed=1)
        assert r["health"]["illegal"]["illegal"] > 0
        r = run_match(dies, station, Rules(), hands=20, seed=1)
        assert r["health"]["dies"]["crashed"]
        t0 = time.time()
        r = run_match(slow, station, Rules(), hands=20, seed=1, timeout=0.2)
        assert r["health"]["slow"]["crashed"] and r["health"]["slow"]["timeouts"] >= 3
        assert time.time() - t0 < 15
        assert r["health"]["calling_station"]["timeouts"] == 0


def test_bot_served_over_http():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([sys.executable, "-m", "arena", "serve", "calling_station",
                               "--host", "127.0.0.1", "--port", str(port)],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        remote = resolve_bot(f"http://127.0.0.1:{port}")
        assert remote.name == "calling_station"
        r = run_match(resolve_bot("maniac"), remote, Rules(), hands=40, seed=2)
        assert all(h["timeouts"] == h["errors"] == h["illegal"] == 0 and not h["crashed"]
                   for h in r["health"].values())
    finally:
        server.kill()
        server.wait()


# ---------- sandbox and ladder ----------
def test_sandbox_blocks_escapes():
    from arena import sandbox

    if not sandbox.available():
        return
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        spy = _temp_bot(d, "spy", f"""
            import os, socket, subprocess
            def blocked(fn):
                try:
                    fn()
                    return False
                except Exception:
                    return True
            class Bot:
                def act(self, s):
                    checks = [
                        blocked(lambda: open({str(Path.home() / '.zshrc')!r}).read()),
                        blocked(lambda: os.listdir({str(Path.home())!r})),
                        blocked(lambda: open({str(ROOT / 'README.md')!r}).read()),
                        blocked(lambda: open({str(ROOT / 'bots' / 'equity_bot.py')!r}).read()),
                        blocked(lambda: open('/private/tmp/sandbox_escape_test', 'w').write('x')),
                        blocked(lambda: socket.create_connection(('1.1.1.1', 80), timeout=2)),
                        blocked(lambda: subprocess.run(['/bin/ls'], capture_output=True)),
                        not any(k.endswith(('KEY', 'TOKEN', 'SECRET')) for k in os.environ),
                    ]
                    if not all(checks):
                        raise RuntimeError(f"escape possible: {{checks}}")
                    open('scratch.txt', 'w').write('ok')
                    return {{"action": "check" if "check" in s["legal"] else "call"}}
        """)
        spy.sandbox = True
        r = run_match(spy, resolve_bot("calling_station"), Rules(), hands=4, seed=1)
        assert r["health"]["spy"]["errors"] == 0 and not r["health"]["spy"]["crashed"], r["health"]["spy"]


def test_ladder_end_to_end():
    """Official-format ladder: baselines, uploads checked by a short official half, live matches."""
    from ladder.core import Ladder, SubmitError

    good = (ROOT / "official" / "baseline-bots" / "baseline_calling_station.py").read_text()
    crashing = good.replace("def choose_action(obs: Observation, memory: dict) -> Action:",
                            "def choose_action(obs: Observation, memory: dict) -> Action:\n    raise ValueError('bug')")
    assert crashing != good
    with tempfile.TemporaryDirectory() as d:
        lad = Ladder(data_dir=Path(d), invite="code-1", deals=60, workers=2)
        lad.ensure_house_bots()
        for bad in [("x", "me", good, "wrong"), ("bad name!", "me", good, "code-1"),
                    ("notabot", "me", "print('hi')\n", "code-1"), ("crash", "me", crashing, "code-1")]:
            try:
                lad.submit(*bad)
                raise AssertionError(f"accepted {bad[0]}")
            except SubmitError:
                pass
        bot_id, key = lad.submit("mine", "me", good, "code-1")
        lad.start()
        deadline = time.time() + 60
        while time.time() < deadline and lad.state()["total_matches"] < 8:
            time.sleep(0.5)
        lad.shutdown()
        state = lad.state()
        assert state["total_matches"] >= 8
        assert sum(b["matches"] for b in state["bots"]) == 2 * state["total_matches"]
        assert all(b["errors"] == 0 and b["forfeits"] == 0 for b in state["bots"])
        assert lad.bot_detail(bot_id)["bot"]["name"] == "mine"
        assert lad.owner_log(bot_id, "wrong") is None and lad.owner_log(bot_id, key) is not None
        assert lad.retire(bot_id, key) and not lad.bot_detail(bot_id)["bot"]["active"]


def test_challenges_play_first():
    from ladder.core import Ladder

    with tempfile.TemporaryDirectory() as d:
        lad = Ladder(data_dir=Path(d), invite="code-1", deals=30, workers=1)
        lad.ensure_house_bots()
        ids = {b["name"]: b["id"] for b in lad.state()["bots"]}
        a, b = ids["random"], ids["always_pot"]
        assert not lad.challenge(a, b, "wrong")[0]
        assert not lad.challenge(a, a, "code-1")[0]
        assert lad.challenge(a, b, "code-1")[0]
        assert not lad.challenge(b, a, "code-1")[0]  # same pair already queued
        assert lad.state()["queued"] == [{"a": "random", "b": "always_pot"}]
        lad.start()
        deadline = time.time() + 30
        while time.time() < deadline and lad.state()["total_matches"] < 1:
            time.sleep(0.2)
        lad.shutdown()
        first = lad.db.execute("SELECT a, b FROM matches ORDER BY id LIMIT 1").fetchone()
        assert {first["a"], first["b"]} == {a, b} and lad.state()["queued"] == []


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"ok    {name} ({time.time() - t0:.1f}s)")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
