"""Monte Carlo equity: how often a hand wins (ties count half) against one random hand."""
from __future__ import annotations

import numpy as np

from .cards import NUM_CARDS, parse_cards
from .evaluator import Evaluator, get_evaluator
from .rules import Rules


def equity_vs_random(hole, board=(), n: int = 600, rng: np.random.Generator | None = None,
                     evaluator: Evaluator | None = None, rules: Rules | None = None, dead=()) -> float:
    """Win probability against a uniformly random opponent hand, running the board out at random.

    hole, board, dead: card strings ("As Kd ...") or ints. Standard error is about 0.5/sqrt(n).
    """
    ev = evaluator or get_evaluator(rules)
    rng = rng if rng is not None else np.random.default_rng()
    hole = parse_cards(hole)
    board = parse_cards(board)
    h = len(hole)
    known = set(hole) | set(board) | set(parse_cards(dead))
    remaining = np.array([c for c in range(NUM_CARDS) if c not in known], dtype=np.int64)
    k = h + (5 - len(board))
    # argsort of random keys gives each row an independent, uniformly random ordered sample.
    picks = remaining[np.argsort(rng.random((n, len(remaining))), axis=1)[:, :k]]
    opp = picks[:, :h]
    fixed = np.broadcast_to(np.array(board, dtype=np.int64), (n, len(board)))
    boards = np.concatenate([fixed, picks[:, h:]], axis=1)
    mine = ev.best_omaha_batch(np.broadcast_to(np.array(hole, dtype=np.int64), (n, h)), boards)
    theirs = ev.best_omaha_batch(opp, boards)
    return float(np.mean((mine > theirs) + 0.5 * (mine == theirs)))


def made_hand(hole, board, evaluator: Evaluator | None = None) -> str:
    """Category of the best hand right now ("pair", "flush", ...). Needs 3+ board cards."""
    ev = evaluator or get_evaluator()
    return ev.category(ev.best_omaha(hole, board))
