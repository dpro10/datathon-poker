"""The bot must follow changed match settings (the guide says they may be tuned):
patch game.py's constants, send the matching config, and check for desyncs/errors."""
import os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import game
import harness
from game import Action, AgentConfig, Game, make_deal, button_for_deal

def run(stack, sb, bb, cap, n=400):
    game.STARTING_STACK, game.SMALL_BLIND, game.BIG_BLIND, game.RAISE_CAP_PER_STREET = stack, sb, bb, cap
    cfg = AgentConfig(starting_stack=stack, big_blind=bb, small_blind=sb, num_hands=n, raise_cap=cap)
    a = harness.Handle(os.path.join(ROOT, "agent.py"), cfg)
    a.start()
    rng = random.Random(stack + cap)
    for d in range(n):
        g = Game(make_deal("rules", d), button_for_deal(d), d)
        while not g.is_terminal:
            s = g.to_act
            obs = g.observation(s)
            if s == 0:
                g.apply(a.act(obs))
            else:
                legal = obs.legal_actions
                g.apply(rng.choice(legal) if rng.random() < 0.5 else Action.CHECK_CALL)
        a.hand_end(g.hand_result(0))
    a.flush()
    m = a.memory
    print(f"stack {stack} blinds {sb}/{bb} cap {cap}: errors {a.errors} desyncs {m.get('desyncs', 0)} "
          f"fallbacks {m.get('fallbacks', 0)} max call {a.max_call * 1000:.1f} ms")
    return a.errors + m.get("desyncs", 0) + m.get("fallbacks", 0)

if __name__ == "__main__":
    bad = 0
    for args in ((80, 1, 2, 4), (200, 1, 2, 4), (40, 1, 2, 4), (160, 2, 4, 3), (100, 1, 2, 5), (80, 1, 2, 2)):
        bad += run(*args)
    print("ALL OK" if bad == 0 else f"PROBLEMS: {bad}")
