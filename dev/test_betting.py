"""Replay random official games and check BState tracks game.py exactly."""
import os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from game import Game, make_deal, Action
from sdbot.betting import BState

rng = random.Random(5)
n_checked = 0
for h in range(20000):
    button = h % 2
    g = Game(make_deal("t", h), button, h)
    states = {0: BState(button == 0), 1: BState(button == 1)}  # each seat's own view (me = that seat)
    while not g.is_terminal:
        seat = g.to_act
        obs = g.observation(seat)
        st = states[seat]
        assert st.to_act == 0, (h, "to_act")
        assert st.pot == obs.pot and st.to_call() == obs.to_call and st.bets[0] == obs.my_bet and st.bets[1] == obs.opp_bet
        assert st.stacks[0] == obs.my_stack and st.stacks[1] == obs.opp_stack and st.raises == obs.raises_this_street
        assert tuple(int(a) for a in obs.legal_actions) == st.legal(), (st.legal(), obs.legal_actions)
        for a in obs.legal_actions:
            if a in (Action.RAISE_HALF, Action.RAISE_POT):
                assert obs.raise_to[a] == st.raise_to(int(a)), (h, a)
        assert obs.min_raise_to == st.bets[1] + max(st.last_inc, 2)
        legal = list(obs.legal_actions)
        w = [0.15 if a == Action.FOLD else 0.45 if a == Action.CHECK_CALL else 0.2 for a in legal]
        a = rng.choices(legal, w)[0]
        g.apply(a)
        for s2 in (0, 1):
            st2 = states[s2]
            who = 0 if s2 == seat else 1
            assert st2.to_act == who
            st2.apply(int(a))
        n_checked += 1
    res = g.result()
    for s2 in (0, 1):
        st2 = states[s2]
        assert st2.over
        assert st2.showdown == res.went_to_showdown
        assert st2.pot == res.pot
print("ok, decisions checked:", n_checked)
