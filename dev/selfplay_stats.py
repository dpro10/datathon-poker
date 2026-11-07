"""Behaviour stats from logged hands, split by street and position (IP = button postflop)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import harness
from game import Action
from sdbot.betting import BState
from evaluator import cards_from_str

def stats(logs, me_seat_for_half):
    S = {}
    def add(k, v):
        a = S.setdefault(k, [0, 0]); a[0] += v; a[1] += 1
    for lg in logs:
        me = me_seat_for_half[lg["half"]]
        st = BState(lg["button_seat"] == me)
        for a in lg["actions"]:
            seat = 0 if a["seat"] == me else 1
            act = int(Action[a["action"]])
            who = "me" if seat == 0 else "op"
            pos = "IP" if st.button == seat else "OOP"
            if st.street > 0:
                key = f"s{st.street} {who} {pos}"
                if st.to_call(seat) == 0 and st.raises == 0:
                    add(key + " bet-when-free", int(act >= 2))
                elif st.to_call(seat) > 0 and st.raises == 1:
                    add(key + " fold-vs-bet", int(act == 0))
                    add(key + " raise-vs-bet", int(act >= 2))
                elif st.to_call(seat) > 0 and st.raises >= 2:
                    add(key + " fold-vs-raise", int(act == 0))
                    add(key + " reraise-vs-raise", int(act >= 2))
            else:
                key = f"s0 {who} {'SB' if st.button == seat else 'BB'}"
                if st.raises == 0 and st.to_call(seat) > 0:
                    add(key + " open-raise", int(act >= 2)); add(key + " open-fold", int(act == 0))
                elif st.to_call(seat) > 0:
                    add(key + f" fold-vs-r{min(st.raises,3)}", int(act == 0)); add(key + f" raise-vs-r{min(st.raises,3)}", int(act >= 2))
            st.apply(act)
    return S

if __name__ == "__main__":
    opp = sys.argv[1]
    logs = []
    for seed in (11, 12):
        harness.play_half(os.path.join(ROOT, "agent.py"), harness.resolve(opp), seed, 1, 800, logs=logs)
        harness.play_half(os.path.join(ROOT, "agent.py"), harness.resolve(opp), seed, 2, 800, logs=logs)
    S = stats(logs, {1: 0, 2: 1})
    # who wins at showdown / how hands end, by position of 'me'
    for k in sorted(S):
        v, n = S[k]
        if "me" in k:
            print(f"  {k:28s} {v / n:6.1%}  (n={n})")
