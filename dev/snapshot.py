"""Freeze the current bot as dev/versions/<name>/agent_<name>.py using package sdbot_<name>,
so different versions can play each other in one process."""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main(name):
    dst = os.path.join(HERE, "versions", name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    pkg = f"sdbot_{name}"
    shutil.copytree(os.path.join(ROOT, "sdbot"), os.path.join(dst, pkg),
                    ignore=shutil.ignore_patterns("__pycache__"))
    src = open(os.path.join(ROOT, "agent.py")).read()
    src = re.sub(r"from sdbot import", f"from {pkg} import", src)
    src = src.replace('AGENT_NAME = "SDBot"', f'AGENT_NAME = "SDBot-{name}"')
    open(os.path.join(dst, f"agent_{name}.py"), "w").write(src)
    print("snapshot ->", os.path.join(dst, f"agent_{name}.py"))


if __name__ == "__main__":
    main(sys.argv[1])
