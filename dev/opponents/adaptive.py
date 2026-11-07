"""An adaptive opponent: tracks how often we bet and fold, and adjusts its bluffing and calling."""
import numpy as np

from _common import Action, Street, can_raise, equity


def new_half(config, memory):
    memory.update(rng=np.random.default_rng(71), opp_bets=1, opp_chances=3, opp_folds=1, opp_faced=3, hand=-1, seen=0)


def _track(obs, memory):
    if memory["hand"] != obs.hand_index:
        memory["hand"], memory["seen"] = obs.hand_index, 0
    hist = obs.action_history
    for i in range(memory["seen"], len(hist)):
        a = hist[i]
        if a.is_me:
            continue
        prev = hist[i - 1] if i > 0 else None
        facing = prev is not None and not prev.is_me is False and prev.action in (Action.RAISE_HALF, Action.RAISE_POT) and prev.street == a.street
        if prev is not None and prev.is_me and prev.street == a.street and prev.action in (Action.RAISE_HALF, Action.RAISE_POT):
            memory["opp_faced"] += 1
            memory["opp_folds"] += a.action == Action.FOLD
        else:
            memory["opp_chances"] += 1
            memory["opp_bets"] += a.action in (Action.RAISE_HALF, Action.RAISE_POT)
    memory["seen"] = len(hist)


def choose_action(obs, memory):
    _track(obs, memory)
    rng = memory["rng"]
    eq = equity(obs.hole_cards, obs.board, 200, rng)
    bet_rate = memory["opp_bets"] / memory["opp_chances"]
    fold_rate = memory["opp_folds"] / memory["opp_faced"]
    price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if obs.street == Street.PREFLOP:
        if eq > 0.57 and can_raise(obs):
            return Action.RAISE_POT
        if eq > 0.46 or obs.to_call == 0:
            return Action.CHECK_CALL
        return Action.FOLD
    if obs.to_call == 0:
        bluff = min(0.5, max(0.0, fold_rate - 0.3))
        if can_raise(obs) and (eq > 0.65 or rng.random() < bluff):
            return Action.RAISE_POT if eq > 0.78 or rng.random() < 0.5 else Action.RAISE_HALF
        return Action.CHECK_CALL
    margin = 0.10 - 0.25 * max(0.0, bet_rate - 0.3)
    if eq > 0.82 and can_raise(obs):
        return Action.RAISE_POT
    return Action.CHECK_CALL if eq > price + margin else Action.FOLD


def on_hand_end(result, memory):
    pass
