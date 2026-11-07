"""Child process that hosts one bot and talks JSON lines on stdin/stdout. The arena starts it."""
import json
import sys


def main() -> int:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr  # the bot's print() goes to its log file, not the protocol
    from arena.host import BotHost

    try:
        host = BotHost(sys.argv[1])
    except Exception as e:
        import traceback

        traceback.print_exc()
        protocol_out.write(json.dumps({"id": None, "fatal": f"{type(e).__name__}: {e}"}) + "\n")
        protocol_out.flush()
        return 1
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        reply = host.handle(msg)
        protocol_out.write(json.dumps(reply) + "\n")
        protocol_out.flush()
        if msg.get("type") == "match_end":
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
