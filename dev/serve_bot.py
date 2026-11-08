"""Serve any official-format bot module (new_half / choose_action / on_hand_end) over HTTP
with the organizers' plumbing, so dev opponents can play through judge_engine.py.

    PORT=5009 python dev/serve_bot.py dev/opponents/eq_tight.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
path = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(path))
spec = importlib.util.spec_from_file_location("served_bot", path)
bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot)

# Reuse agent.py's plumbing, swapping in this bot's three functions.
spec2 = importlib.util.spec_from_file_location("plumbing", os.path.join(ROOT, "agent.py"))
plumb = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(plumb)
plumb.new_half, plumb.choose_action, plumb.on_hand_end = bot.new_half, bot.choose_action, bot.on_hand_end
plumb.AGENT_NAME = os.path.basename(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(plumb.app, host="0.0.0.0", port=int(os.environ.get("PORT", "5009")), log_level="warning", access_log=False)
