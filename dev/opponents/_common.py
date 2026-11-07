"""Helpers for practice opponents: Monte Carlo equity vs one random hand (fast evaluator)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from game import Action, Street  # noqa: E402,F401
from sdbot import evaluator as ev  # noqa: E402

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def equity(hole, board, n, rng):
    hole = list(hole)
    board = list(board)
    known = set(hole) | set(board)
    rem = np.array([c for c in range(36) if c not in known], dtype=np.int64)
    need = 5 + (5 - len(board))
    picks = rem[np.argsort(rng.random((n, len(rem))), axis=1)[:, :need]]
    boards = np.concatenate([np.broadcast_to(np.array(board, dtype=np.int64), (n, len(board))), picks[:, 5:]], axis=1)
    t = ev.board_tables(boards)
    mine = t[ev.hand_codes_one(hole)].max(axis=0)
    theirs = ev.scores_paired(t, ev.hand_codes(picks[:, :5]))
    return float(np.mean((mine > theirs) + 0.5 * (mine == theirs)))


def can_raise(obs):
    return Action.RAISE_POT in obs.legal_actions


def pick(obs, want):
    """Return `want` if legal, else the closest legal fallback."""
    if want in obs.legal_actions:
        return want
    if want in RAISES:
        return Action.CHECK_CALL
    if want == Action.FOLD:
        return Action.CHECK_CALL
    return obs.legal_actions[0]
