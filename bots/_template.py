"""Start here: copy this file (or equity_bot.py) to bots/<yourname>.py, then run
    python3 -m arena check bots/<yourname>.py

The arena calls act(state) on your turn. Example state:
{
  "hand_id": 17, "street": "flop",            # preflop, flop, turn, river
  "hole": ["As", "Kd", "Qh", "Jc", "Ts"],     # your 5 cards; the deck is 6 through A
  "board": ["9s", "8h", "7d"],
  "button": true,                             # true: you posted the small blind, act first preflop and last after
  "pot": 12,                                  # everything in the middle, including this street's bets
  "my_stack": 194, "opp_stack": 194,          # chips still behind
  "my_bet": 0, "opp_bet": 0,                  # chips put in on this street
  "to_call": 0,
  "legal": ["check", "raise"],
  "min_raise_to": 2, "max_raise_to": 12,      # only when "raise" is legal; totals for this street
  "history": [{"street": "preflop", "actor": "opp", "action": "raise", "amount": 6}, ...]
}
Return {"action": "fold" | "check" | "call" | "raise", "amount": <raise-to total, only for raise>}.
Anything illegal, too slow or crashing is played as check (when free) or fold.
Omaha rule: a hand uses exactly 2 of your 5 cards and exactly 3 board cards.
Helpers: from pokerlib import equity_vs_random, made_hand, get_evaluator, Rules
"""


class Bot:
    def new_match(self, info):
        """Start of each half of a match. info: rules, hands, session, time_limit."""
        self.rules = info.get("rules", {})

    def act(self, state):
        if "check" in state["legal"]:
            return {"action": "check"}
        return {"action": "call"}

    def hand_over(self, result):
        """After every hand: your net chips, the board, and the opponent's cards if it went to showdown."""
        pass
