"""Aggressive equity bot: bets and raises often, bluffs sometimes."""
from eqbot import make

new_half, choose_action, on_hand_end = make(
    bet=[0.50, 0.52, 0.55, 0.60], pot=[0.55, 0.62, 0.66, 0.72], **{"raise": [0.55, 0.66, 0.72, 0.78]},
    margin=[-0.02, 0.00, 0.00, 0.02], bluff=0.25, seed=13)
