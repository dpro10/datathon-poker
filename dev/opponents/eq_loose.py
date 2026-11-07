"""Loose-passive equity bot: calls a lot, bets only good hands."""
from eqbot import make

new_half, choose_action, on_hand_end = make(
    bet=[0.60, 0.65, 0.70, 0.75], pot=[0.70, 0.80, 0.85, 0.90], **{"raise": [0.65, 0.82, 0.86, 0.90]},
    margin=[-0.08, -0.06, -0.05, -0.03], seed=12)
