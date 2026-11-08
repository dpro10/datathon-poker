"""Robustness stress: many hands vs random/extreme opponents; report errors, internal
fallbacks, desyncs and the slowest decision. Run with SDBOT_DEBUG=1 to surface exceptions."""
import os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import harness
from game import Action, AgentConfig, Game, make_deal, button_for_deal

def opp_policy(kind, rng):
    def f(obs):
        legal = obs.legal_actions
        raises = [a for a in legal if a in (Action.RAISE_HALF, Action.RAISE_POT)]
        if kind == "allin" and raises:
            return Action.RAISE_POT
        if kind == "minraise" and raises:
            return Action.RAISE_HALF
        if kind == "fold" and Action.FOLD in legal:
            return Action.FOLD
        if kind == "chaos":
            return rng.choice(legal)
        if kind == "mostly_raise":
            return rng.choice(raises) if raises and rng.random() < 0.8 else (Action.CHECK_CALL)
        return Action.CHECK_CALL
    return f

def main(n_hands=3000):
    worst = 0.0
    for kind in ("chaos", "allin", "minraise", "fold", "mostly_raise", "call"):
        rng = random.Random(kind)
        cfg = AgentConfig(num_hands=n_hands)
        a = harness.Handle(os.path.join(ROOT, "agent.py"), cfg)
        a.start()
        pol = opp_policy(kind, rng)
        for d in range(n_hands):
            g = Game(make_deal(f"stress-{kind}", d), button_for_deal(d), d)
            while not g.is_terminal:
                s = g.to_act
                obs = g.observation(s)
                g.apply(a.act(obs) if s == 0 else pol(obs))
            a.hand_end(g.hand_result(0))
        a.flush()
        m = a.memory
        worst = max(worst, a.max_call)
        print(f"{kind:13s} errors {a.errors}  fallbacks {m.get('fallbacks', 0)}  desyncs {m.get('desyncs', 0)}  "
              f"max call {a.max_call * 1000:.1f} ms  internal {m['time']:.2f}s  level {m['level']}"
              + (f"\n   first error: {a.first_error[:800]}" if a.first_error else ""))
    print(f"slowest decision overall: {worst * 1000:.1f} ms")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
