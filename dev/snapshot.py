"""Freeze the current bot as dev/versions/<name>/agent_<name>.py using package sdbot_<name>,
so different versions can play each other in one process."""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def freeze(pkg_dir):
    """A snapshot must ignore the experiment overrides meant for the bot under test."""
    path = os.path.join(pkg_dir, "brain.py")
    s = open(path).read()
    s = s.replace('os.environ.get("SDBOT_PARAMS")', 'None').replace('os.environ.get("SDBOT_SPEED")', 'None')
    open(path, "w").write(s)


def set_defaults(pkg_dir, **kv):
    """Bake parameter defaults into a snapshot (e.g. mode='blueprint')."""
    path = os.path.join(pkg_dir, "search.py")
    s = open(path).read()
    s += f"\n\n# baked into this snapshot by dev/snapshot.py\nDEFAULT_PARAMS.update({kv!r})\n"
    open(path, "w").write(s)


def main(name, **defaults):
    dst = os.path.join(HERE, "versions", name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    pkg = f"sdbot_{name}"
    shutil.copytree(os.path.join(ROOT, "sdbot"), os.path.join(dst, pkg),
                    ignore=shutil.ignore_patterns("__pycache__"))
    freeze(os.path.join(dst, pkg))
    if defaults:
        set_defaults(os.path.join(dst, pkg), **defaults)
    src = open(os.path.join(ROOT, "agent.py")).read()
    src = re.sub(r"from sdbot import", f"from {pkg} import", src)
    src = src.replace('AGENT_NAME = "SDBot"', f'AGENT_NAME = "SDBot-{name}"')
    open(os.path.join(dst, f"agent_{name}.py"), "w").write(src)
    print("snapshot ->", os.path.join(dst, f"agent_{name}.py"))


if __name__ == "__main__":
    if sys.argv[1] == "--freeze-all":
        for v in sorted(os.listdir(os.path.join(HERE, "versions"))):
            freeze(os.path.join(HERE, "versions", v, f"sdbot_{v}"))
            print("froze", v)
    else:
        import json
        main(sys.argv[1], **(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}))
