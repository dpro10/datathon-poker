"""Pure pot-odds bot: calls whenever equity beats the price, raises only near-nut hands."""
from eqbot import make

new_half, choose_action, on_hand_end = make(
    bet=[0.62, 0.75, 0.80, 0.85], pot=[0.65, 0.85, 0.88, 0.92], **{"raise": [0.66, 0.88, 0.90, 0.93]},
    margin=[0.0, 0.0, 0.0, 0.0], samples=250, seed=14)
