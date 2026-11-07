"""Port of bots/equity_bot.py (the repo's arena baseline) to the official 4-action format."""
import numpy as np

from _common import Action, can_raise, equity

SAMPLES = [300, 500, 500, 400]
BET = [0.56, 0.62, 0.68, 0.72]
RAISE = [0.60, 0.75, 0.80, 0.85]
POT_SIZE_AT = 0.80
CALL_MARGIN = 0.02
TIGHT_DISCOUNT = 0.10


def new_half(config, memory):
    memory["rng"] = np.random.default_rng()
    memory["opp_decisions"] = 0
    memory["opp_raises"] = 0
    memory["seen"] = 0


def _size(eq):
    return Action.RAISE_POT if eq >= POT_SIZE_AT else Action.RAISE_HALF


def choose_action(obs, memory):
    # count opponent actions we have not seen yet this hand
    opp = [a for a in obs.action_history if not a.is_me]
    new = opp[memory["seen"]:] if memory.get("hand") == obs.hand_index else opp
    memory["hand"] = obs.hand_index
    memory["seen"] = len(opp)
    for a in new:
        memory["opp_decisions"] += 1
        memory["opp_raises"] += a.action in (Action.RAISE_HALF, Action.RAISE_POT)
    s = int(obs.street)
    eq = equity(obs.hole_cards, obs.board, SAMPLES[s], memory["rng"])
    if obs.to_call == 0:
        if can_raise(obs) and eq >= BET[s]:
            return _size(eq)
        return Action.CHECK_CALL
    aggression = (memory["opp_raises"] + 1) / (memory["opp_decisions"] + 3)
    eq -= TIGHT_DISCOUNT * max(0.0, 1 - aggression / 0.4)
    if can_raise(obs) and eq >= RAISE[s]:
        return _size(eq)
    price = obs.to_call / (obs.pot + obs.to_call)
    if eq >= price + CALL_MARGIN:
        return Action.CHECK_CALL
    return Action.FOLD


def on_hand_end(result, memory):
    pass
