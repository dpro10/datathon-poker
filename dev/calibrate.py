"""Measure opponents' real behaviour P(action | class, strength bin) from full-information logs.

Plays agent.py against each opponent, then replays every hand: for each opponent decision
we know its true hand, so we compute its strength u exactly the way the bot does at a
showdown (HandTracker.strength_of) and count (class, bin, action).

    python dev/calibrate.py --deals 600 --out dev/results/calib.npz  opp1 opp2 ...
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import harness  # noqa: E402
from evaluator import cards_from_str  # noqa: E402
from game import Action  # noqa: E402
from sdbot.betting import BState  # noqa: E402
from sdbot.hand import HandTracker  # noqa: E402
from sdbot.model import NACT, NBIN, NCLASS, classify, interp_weights  # noqa: E402


class _Obs:
    def __init__(self, hole, idx, button):
        self.hole_cards, self.hand_index, self.is_button = hole, idx, button


def analyze(logs, opp_seat_for_half, rng):
    counts = np.zeros((NCLASS, NBIN, NACT))
    for lg in logs:
        opp_seat = opp_seat_for_half[lg["half"]]
        me_seat = 1 - opp_seat
        holes = [cards_from_str(" ".join(h)) for h in lg["hole_cards"]]
        board = cards_from_str(" ".join(lg["board"]))
        me_button = lg["button_seat"] == me_seat
        st = BState(me_button)
        h = HandTracker(_Obs(holes[me_seat], 0, me_button), 300, rng, 24)
        h.board = board
        cache = {}
        for a in lg["actions"]:
            seat = 0 if a["seat"] == me_seat else 1
            act = int(Action[a["action"]])
            if seat == 1:
                cls = classify(st)
                s = st.street
                if s not in cache:
                    cache[s] = h.strength_of(holes[opp_seat], s)
                lo, wl, wh = interp_weights(np.array([cache[s]]))
                counts[cls, lo[0], act] += wl[0]
                counts[cls, lo[0] + 1, act] += wh[0]
            st.apply(act)
    return counts


def run(args):
    opp, seed, deals = args
    bot = harness.resolve("agent.py") if os.path.exists("agent.py") else os.path.join(ROOT, "agent.py")
    logs = []
    harness.play_half(os.path.join(ROOT, "agent.py"), harness.resolve(opp), seed, 1, deals, logs=logs)
    harness.play_half(os.path.join(ROOT, "agent.py"), harness.resolve(opp), seed, 2, deals, logs=logs)
    rng = np.random.default_rng(seed)
    # agent2 (opponent) is seat 1 in half 1 and seat 0 in half 2
    return opp, analyze(logs, {1: 1, 2: 0}, rng)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("opponents", nargs="+")
    p.add_argument("--deals", type=int, default=600)
    p.add_argument("--seeds", default="101")
    p.add_argument("--out", default=os.path.join(HERE, "results", "calib.npz"))
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tasks = [(o, s, a.deals) for o in a.opponents for s in harness.parse_seeds(a.seeds)]
    per = {}
    with Pool(os.cpu_count()) as pool:
        for opp, c in pool.imap_unordered(run, tasks):
            per[opp] = per.get(opp, 0) + c
            print("done", opp, flush=True)
    names = sorted(per)
    np.savez(a.out, names=np.array(names), counts=np.stack([per[n] for n in names]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
