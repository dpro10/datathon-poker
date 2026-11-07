"""Collect (street, position, pot, stack, true equity) -> realized value of the rest of the hand,
from logged hands of agent.py vs a set of opponents. Saves dev/results/leaf.npz."""
import os, sys
from itertools import combinations
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import harness
from evaluator import cards_from_str
from game import Action
from sdbot import evaluator as ev
from sdbot.betting import BState

def exact_equity(h1, h2, board_known):
    known = set(h1) | set(h2) | set(board_known)
    rem = [c for c in range(36) if c not in known]
    need = 5 - len(board_known)
    runs = np.array([list(board_known) + list(r) for r in combinations(rem, need)], dtype=np.int64)
    t = ev.board_tables(runs)
    a = t[ev.hand_codes_one(sorted(h1))].max(axis=0)
    b = t[ev.hand_codes_one(sorted(h2))].max(axis=0)
    return float(np.mean((a > b) + 0.5 * (a == b)))

def rows_from_logs(logs, me_seat_for_half):
    out = []
    for lg in logs:
        me = me_seat_for_half[lg["half"]]
        holes = [cards_from_str(" ".join(h)) for h in lg["hole_cards"]]
        board = cards_from_str(" ".join(lg["board"]))
        st = BState(lg["button_seat"] == me)
        final_net = lg["nets"][me]
        seen_street = 0
        for a in lg["actions"] + [None]:
            if st.street != seen_street and not st.over:
                # a new street began with chips behind: record it
                s = st.street
                n = (0, 3, 4, 5)[s]
                e = exact_equity(holes[me], holes[1 - me], board[:n])
                out.append((s, int(st.button == 0), st.pot, min(st.stacks), e, final_net + st.contrib[0]))
                seen_street = s
            if a is None:
                break
            st.apply(int(Action[a["action"]]))
    return out

def run(args):
    opp, seed, deals = args[:3]
    logs = []
    harness.play_half(os.path.join(ROOT, "agent.py"), harness.resolve(opp), seed, 1, deals, logs=logs)
    harness.play_half(os.path.join(ROOT, "agent.py"), harness.resolve(opp), seed, 2, deals, logs=logs)
    return rows_from_logs(logs, {1: 0, 2: 1})

if __name__ == "__main__":
    opps = sys.argv[1:]
    reps = int(os.environ.get("REPS", "1"))
    tasks = [(o, 300 + i + 100 * j, 600) for i, o in enumerate(opps) for j in range(reps)]
    rows = []
    with Pool(os.cpu_count()) as p:
        for r in p.imap_unordered(run, tasks):
            rows += r
    arr = np.array(rows, dtype=np.float64)
    np.savez(os.path.join(HERE, "results", os.environ.get("OUT", "leaf.npz")), rows=arr)
    print("rows", len(arr))
