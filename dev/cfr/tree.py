"""Enumerate the exact betting tree of one hand (it is the same for every deal, because
stacks reset each hand) and export it as flat arrays for the CFR trainer and the bot.

Node arrays (int32):
  player[n]   seat to act (0 = small blind / button, 1 = big blind), -1 for terminals
  street[n]   0..3
  child[n,a]  child node for action a in (FOLD, CALL, HALF, POT), -1 if illegal
  kind[n]     0 decision, 1 fold terminal, 2 showdown terminal
  win[n]      fold terminals: the seat that collects the pot
  contrib[n,s] chips seat s has put in when the node is reached
Seats here are absolute: 0 is the button (small blind), 1 the big blind.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from sdbot.betting import BState  # noqa: E402


def build():
    player, street, child, kind, win, contrib = [], [], [], [], [], []

    def new(st):
        n = len(player)
        player.append(-1); street.append(st.street); child.append([-1, -1, -1, -1])
        kind.append(0); win.append(-1)
        # BState seat 0 is "me"; build with me = button so seat 0 == button
        contrib.append([st.contrib[0], st.contrib[1]])
        return n

    root_state = BState(True)       # seat 0 = button
    stack = [(root_state, None, None)]
    root = None
    # iterative DFS
    def rec(st):
        n = new(st)
        if st.over:
            kind[n] = 2 if st.showdown else 1
            if not st.showdown:
                win[n] = 1 - st.folded
            return n
        player[n] = st.to_act
        for a in st.legal():
            ch = st.copy()
            ch.apply(a)
            child[n][a] = rec(ch)
        return n

    sys.setrecursionlimit(10000)
    root = rec(root_state)
    return (np.array(player, np.int32), np.array(street, np.int32), np.array(child, np.int32),
            np.array(kind, np.int32), np.array(win, np.int32), np.array(contrib, np.int32))


if __name__ == "__main__":
    player, street, child, kind, win, contrib = build()
    dec = kind == 0
    print("nodes", len(player), "decision", int(dec.sum()), "fold-terminal", int((kind == 1).sum()),
          "showdown-terminal", int((kind == 2).sum()))
    for s in range(4):
        print(f"  street {s}: decision nodes {int((dec & (street == s)).sum())}")
    np.savez(os.path.join(HERE, "tree.npz"), player=player, street=street, child=child, kind=kind, win=win, contrib=contrib)
