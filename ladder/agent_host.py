"""Child process: runs one official-format bot (agent.py) and answers the judge's four routes
over stdin/stdout instead of HTTP. The route handling mirrors the template's SERVER PLUMBING,
so a bot behaves here exactly as it does behind its FastAPI server.

    python agent_host.py <path/to/agent.py> [--limits]
"""
import importlib.util
import json
import os
import sys
import traceback
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "official")


def _stub_web_modules() -> None:
    """agent.py imports FastAPI and uvicorn for its server; here no server runs."""
    fastapi = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def _route(self, *args, **kwargs):
            return lambda fn: fn

        get = post = put = delete = patch = _route

    class Request:
        pass

    responses = types.ModuleType("fastapi.responses")

    class JSONResponse:
        def __init__(self, content=None, status_code=200, **kwargs):
            self.content, self.status_code = content, status_code

    fastapi.FastAPI, fastapi.Request, fastapi.responses = FastAPI, Request, responses
    responses.JSONResponse = JSONResponse
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: None
    sys.modules.update({"fastapi": fastapi, "fastapi.responses": responses, "uvicorn": uvicorn})


def _limits() -> None:
    import resource

    for limit, value in ((resource.RLIMIT_CPU, 900), (resource.RLIMIT_FSIZE, 64 * 2 ** 20)):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


class Host:
    def __init__(self, module):
        from game import Action, AgentConfig, HandResult, Observation

        self.m = module
        self.Action, self.AgentConfig, self.HandResult, self.Observation = Action, AgentConfig, HandResult, Observation
        memory = getattr(module, "MEMORY", None)
        self.memory = memory if isinstance(memory, dict) else {}
        self.started = False

    @staticmethod
    def _error(exc: Exception) -> dict:
        return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}

    def _deliver(self, raw_results) -> list:
        errors = []
        for raw in raw_results or []:
            try:
                self.m.on_hand_end(self.HandResult.from_dict(raw), self.memory)
            except Exception as exc:
                errors.append(self._error(exc))
        return errors

    def handle(self, method: str, path: str, body):
        if method == "GET" and path == "/":
            return 200, {"participant": str(getattr(self.m, "PARTICIPANT", "?")),
                         "agent_name": str(getattr(self.m, "AGENT_NAME", "?"))}
        if method != "POST":
            return 404, {"error": "not found"}
        if not self.started and path != "/new-half":
            return 409, {"error": "call /new-half first"}
        if not isinstance(body, dict):
            return 400, {"error": "the request body must be a JSON object"}
        if path == "/new-half":
            self.memory.clear()
            self.started = True
            try:
                self.m.new_half(self.AgentConfig.from_dict(body.get("config", {})), self.memory)
            except Exception as exc:
                return 500, {**self._error(exc), "hand_end_errors": []}
            return 200, {"ok": True}
        if path == "/act":
            errors = self._deliver(body.get("hand_results"))
            try:
                obs = self.Observation.from_dict(body["observation"])
                action = self.m.choose_action(obs, self.memory)
            except Exception as exc:
                return 500, {**self._error(exc), "hand_end_errors": errors}
            if isinstance(action, self.Action):
                answer = action.name
            elif action is None or isinstance(action, (str, int, float, bool)):
                answer = action
            else:
                answer = repr(action)
            return 200, {"action": answer, "hand_end_errors": errors}
        if path == "/hand-end":
            return 200, {"ok": True, "hand_end_errors": self._deliver(body.get("results"))}
        return 404, {"error": "not found"}


def main() -> int:
    if "--limits" in sys.argv[2:]:
        _limits()
    out = sys.stdout
    sys.stdout = sys.stderr  # anything the bot prints goes to its log
    path = os.path.abspath(sys.argv[1])
    sys.path[:0] = [OFFICIAL, os.path.dirname(path)]  # official game.py / evaluator.py win, like the tournament
    try:
        _stub_web_modules()
        spec = importlib.util.spec_from_file_location("agent_under_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent_under_test"] = module
        spec.loader.exec_module(module)
        for fn in ("new_half", "choose_action", "on_hand_end"):
            if not callable(getattr(module, fn, None)):
                raise AttributeError(f"agent file has no function {fn}()")
        host = Host(module)
    except BaseException as exc:
        traceback.print_exc()
        out.write(json.dumps({"id": None, "fatal": f"{type(exc).__name__}: {exc}"}) + "\n")
        out.flush()
        return 1
    while True:
        line = sys.stdin.readline()
        if not line:
            return 0
        try:
            msg = json.loads(line)
            status, data = host.handle(msg.get("method", ""), msg.get("path", ""), msg.get("body"))
        except Exception as exc:  # a bug here must not look like the bot's fault
            status, data = 500, {"error": f"host error: {exc}"}
            msg = msg if isinstance(msg, dict) else {}
        out.write(json.dumps({"id": msg.get("id"), "status": status, "data": data}) + "\n")
        out.flush()


if __name__ == "__main__":
    sys.exit(main())
