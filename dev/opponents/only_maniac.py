"""The switcher's maniac style alone."""
import os
os.environ["SWITCH_PERIOD"] = "1000000"; os.environ["SWITCH_START"] = "1"
import sys
sys.modules.pop("switcher", None)
import switcher  # noqa: E402
new_half, choose_action, on_hand_end = switcher.new_half, switcher.choose_action, switcher.on_hand_end
