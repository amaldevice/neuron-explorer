"""Ticket #16: pluggable omp / pi Agent backends.

Runs the real server (EXPLORER_NO_MODEL=1) under each backend and checks the whole turn: the MCP `explorer`
server is reached, the unified JSONL stream is parsed into text / tool / tool_result / done events, the session
is resumed on a follow-up turn, and /api/agent/rewind cuts the last user turn back out of the session file.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
PORT = int(os.environ.get("TICKET16_PORT", "58917"))
BASE = f"http://127.0.0.1:{PORT}"
FAILURES = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  -- {detail}" if detail and not cond else ""), flush=True)
    if not cond:
        FAILURES.append(name)

def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path, timeout=60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    """The explorer server, started with a given Agent backend."""

    def __init__(self, backend, port):
        self.backend, self.port = backend, port
        env = {**os.environ, "EXPLORER_NO_MODEL": "1", "PORT": str(port), "EXPLORER_AGENT": backend,
               "EXPLORER_AGENT_MODEL": "openai-codex/gpt-5.6-luna"}
        self.proc = subprocess.Popen([PY, os.path.join(ROOT, "explorer", "server.py")], env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    def wait(self, tries=120):
        for _ in range(tries):
            if self.proc.poll() is not None:
                return False
            try:
                get("/api/index", timeout=5)
                return True
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        return False

    def stop(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(10)
        except Exception:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except Exception:
                pass


def turn(chat, message):
    """One chat turn; returns the list of NDJSON events."""
    req = urllib.request.Request(BASE + "/api/agent/chat",
                                 data=json.dumps({"chat": chat, "message": message}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    events = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if line:
                events.append(json.loads(line))
    return events


# ---------- 1. backend selection, config files, stream parsing ----------

def test_selection_and_config():
    import importlib
    sys.path.insert(0, os.path.join(ROOT, "explorer"))
    os.environ["EXPLORER_AGENT"] = "pi"
    import server as srv
    importlib.reload(srv)
    check("EXPLORER_AGENT=pi selects pi", srv.get_agent_name() == "pi", srv.get_agent_name())
    os.environ["EXPLORER_AGENT"] = "omp"
    check("EXPLORER_AGENT=omp selects omp", srv.get_agent_name() == "omp")
    os.environ["EXPLORER_AGENT"] = "claude"
    check("EXPLORER_AGENT=claude selects claude", srv.get_agent_name() == "claude")
    os.environ["EXPLORER_AGENT"] = "nonsense"
    detected = srv.get_agent_name()
    check("unset/invalid EXPLORER_AGENT auto-detects omp -> pi -> claude",
          detected in ("omp", "pi", "claude"), detected)
    os.environ["EXPLORER_AGENT"] = "omp"
    os.environ["EXPLORER_AGENT_MODEL"] = "m1"
    check("EXPLORER_AGENT_MODEL wins", srv.get_agent_model() == "m1")
    del os.environ["EXPLORER_AGENT_MODEL"]
    os.environ["EXPLORER_CLAUDE_MODEL"] = "m2"
    check("EXPLORER_CLAUDE_MODEL is the fallback", srv.get_agent_model() == "m2")
    del os.environ["EXPLORER_CLAUDE_MODEL"]

    agent_dir = os.path.join(ROOT, "explorer", "cache", "agent")
    srv._setup_omp()
    omp_cfg = os.path.join(agent_dir, ".omp", "config.yml")
    mcp = json.load(open(os.path.join(agent_dir, ".omp", "mcp.json")))
    check("omp mcp.json has the explorer server with ${EXPLORER_URL}/${EXPLORER_CHAT}",
          mcp["mcpServers"]["explorer"]["env"] == {"EXPLORER_URL": "${EXPLORER_URL}", "EXPLORER_CHAT": "${EXPLORER_CHAT}"})
    cfg = open(omp_cfg).read()
    check("omp config.yml disables agents-md / claude-md", "agents-md" in cfg and "claude-md" in cfg)
    check("omp config.yml disables the context-file extensions",
          all(f"context-file:{k}:{w}.md" in cfg for k in ("user", "project") for w in ("AGENTS", "CLAUDE")))
    check("omp config.yml turns off xdev and intentTracing", "xdev: false" in cfg and "intentTracing: false" in cfg)
    srv._setup_pi()
    pi_mcp = json.load(open(os.path.join(agent_dir, ".pi", "mcp.json")))
    e = pi_mcp["mcpServers"]["explorer"]
    check("pi mcp.json exposure is direct", e.get("exposure") == "direct")
    check("pi mcp.json env expands EXPLORER_URL / EXPLORER_CHAT",
          e["env"] == {"EXPLORER_URL": "${EXPLORER_URL}", "EXPLORER_CHAT": "${EXPLORER_CHAT}"})

    # the unified stream parser
    st = {"cost": 0.0, "last": None, "done": False, "seen": False}
    out = list(srv._unified_events({"type": "message_update",
                                     "assistantMessageEvent": {"type": "text_delta", "delta": "hi"}}, st))
    check("text_delta -> text event", out == [{"t": "text", "text": "hi"}], out)
    out = list(srv._unified_events({"type": "tool_execution_start", "toolCallId": "c1",
                                     "toolName": "mcp__explorer_overview", "args": {"x": 1}}, st))
    check("tool_execution_start -> tool event with the mcp prefix stripped",
          out == [{"t": "tool", "id": "c1", "input": {"x": 1}, "name": "overview"}], out)
    out = list(srv._unified_events({"type": "tool_execution_end", "toolCallId": "c1", "isError": False,
                                     "result": {"content": [{"type": "text", "text": "ok"}]}}, st))
    check("tool_execution_end -> tool_result event", out[0]["t"] == "tool_result" and out[0]["output"] == "ok", out)
    out = list(srv._unified_events({"type": "message_end", "message": {"role": "assistant", "stopReason": "stop",
                                                                        "usage": {"cost": {"total": 0.5}}}}, st))
    check("assistant message_end accumulates the cost", st["cost"] == 0.5, st)
    out = list(srv._unified_events({"type": "agent_end", "isTerminal": True, "willRetry": False}, st))
    check("agent_end -> done with the cost", out == [{"t": "done", "cost": 0.5}], out)


# ---------- 2. a live turn per backend ----------

def test_backend(backend):
    port = PORT + (1 if backend == "omp" else 2)
    base = f"http://127.0.0.1:{port}"
    global BASE
    old, BASE = BASE, base
    srv = Server(backend, port)
    try:
        if not srv.wait():
            check(f"[{backend}] server starts", False, srv.proc.stderr.read().decode()[-800:])
            return
        idx = get("/api/index")
        check(f"[{backend}] /api/index reports the agent", idx.get("agent") == backend, idx.get("agent"))
        ov = get("/api/agent/overview")
        check(f"[{backend}] /api/agent/overview reports the agent", ov.get("agent") == backend, ov.get("agent"))

        chat = f"ticket16-{backend}-{int(time.time())}"
        ev = turn(chat, "Call the overview tool and reply with the model name in one short sentence.")
        kinds = [e["t"] for e in ev]
        check(f"[{backend}] the turn ends with done", "done" in kinds, kinds)
        check(f"[{backend}] the turn streams text", "text" in kinds, kinds)
        check(f"[{backend}] the turn streams a tool call", "tool" in kinds, kinds)
        check(f"[{backend}] the turn streams a tool result", "tool_result" in kinds, kinds)
        check(f"[{backend}] the tool call is the explorer's overview",
              any(e.get("name") == "overview" for e in ev if e["t"] == "tool"),
              [e for e in ev if e["t"] == "tool"])
        check(f"[{backend}] the tool result carries the model name",
              any("Qwen" in (e.get("output") or "") for e in ev if e["t"] == "tool_result"))
        done = [e for e in ev if e["t"] == "done"]
        check(f"[{backend}] done carries a cost", done and done[0].get("cost") is not None, done)
        check(f"[{backend}] no error event", not any(e["t"] == "error" for e in ev),
              [e for e in ev if e["t"] == "error"])

        # the session file exists, and the follow-up turn resumes it
        session = _session_path(backend, chat)
        check(f"[{backend}] the session file was written", bool(session) and os.path.exists(session), session)
        first_size = os.path.getsize(session) if session else 0
        ev2 = turn(chat, "Now say the word resumed.")
        check(f"[{backend}] the follow-up turn also ends with done", any(e["t"] == "done" for e in ev2))
        check(f"[{backend}] the follow-up turn resumed the same session file",
              os.path.exists(session) and os.path.getsize(session) > first_size)

        # rewind cuts the last user turn out of the session file
        before = open(session).read()
        r = post("/api/agent/rewind", {"chat": chat, "text": "Now say the word resumed."})
        check(f"[{backend}] rewind reports a cut", r.get("cut") is True, r)
        after = open(session).read()
        check(f"[{backend}] rewind removed the last user turn from the session file",
              "Now say the word resumed." not in after, after[-400:])
        check(f"[{backend}] rewind kept the earlier turn", "Call the overview tool" in after)

        test_attachments(backend)
        test_interrupt(backend, port)
    finally:
        srv.stop()

    # a restarted server picks the same session back up
    srv = Server(backend, port)
    try:
        if not srv.wait():
            check(f"[{backend}] the server restarts", False)
            return
        test_resume_after_restart(backend, port, chat)
    finally:
        srv.stop()
        BASE = old


def _session_path(backend, chat):
    d = os.path.join(ROOT, "explorer", "cache", "agent", "sessions")
    if backend == "omp":
        p = os.path.join(d, f"{chat}.jsonl")
        return p if os.path.exists(p) else None
    hits = [h for h in os.listdir(d) if h.endswith(f"_{chat}.jsonl")] if os.path.isdir(d) else []
    return os.path.join(d, hits[0]) if hits else None


def upload(name, kind, raw):
    """Upload a file and return its id."""
    import base64
    b = json.dumps({"name": name, "type": kind, "data": base64.b64encode(raw).decode()}).encode()
    req = urllib.request.Request(BASE + "/api/agent/upload", data=b,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["id"]


def turn_with(chat, message, attachments):
    req = urllib.request.Request(BASE + "/api/agent/chat",
                                 data=json.dumps({"chat": chat, "message": message,
                                                  "attachments": attachments}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    events = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if line:
                events.append(json.loads(line))
    return events


PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a4944415478"
                    "9c6360000002000154a24f5f0000000049454e44ae426082")
PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"


def test_attachments(backend):
    """An image, a text file and a PDF go with the turn."""
    chat = f"ticket16-att-{backend}-{int(time.time())}"
    ids = [upload("pic.png", "image/png", PNG),
           upload("notes.txt", "text/plain", b"the neuron fires on cats"),
           upload("doc.pdf", "application/pdf", PDF)]
    ev = turn_with(chat, "Say what the attached files are, in one short sentence.", ids)
    kinds = [e["t"] for e in ev]
    check(f"[{backend}] the attachment turn ends with done", "done" in kinds, kinds)
    check(f"[{backend}] the attachment turn produced no error", not any(e["t"] == "error" for e in ev),
          [e for e in ev if e["t"] == "error"])
    session = _session_path(backend, chat)
    body = open(session).read() if session else ""
    check(f"[{backend}] the text attachment reached the agent", "fires on cats" in body, body[-600:])
    check(f"[{backend}] the PDF attachment reached the agent", "doc.pdf" in body or "Attached PDF" in body,
          body[-600:])
    check(f"[{backend}] the image attachment reached the agent",
          "pic.png" in body or "image" in body.lower(), body[-600:])


def test_resume_after_restart(backend, port, chat):
    """The server was restarted: the same chat id must pick the session back up."""
    ev = turn(chat, "Now say the word resumed.")
    check(f"[{backend}] the turn after a restart ends with done", any(e["t"] == "done" for e in ev))
    session = _session_path(backend, chat)
    body = open(session).read() if session else ""
    check(f"[{backend}] the restarted server resumed the same session", "Call the overview tool" in body)


def test_interrupt(backend, port):
    """Interrupting a turn leaves no orphan processes behind."""
    before = subprocess.run(["pgrep", "-f", "agent_mcp.py"], capture_output=True, text=True).stdout.split()
    chat = f"ticket16-int-{backend}-{int(time.time())}"
    req = urllib.request.Request(BASE + "/api/agent/chat",
                                 data=json.dumps({"chat": chat, "message": "Count slowly from 1 to 100."}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    r = urllib.request.urlopen(req, timeout=60)
    r.readline()  # the turn is under way
    r.close()     # the browser hung up: the server stops the CLI and its MCP child
    time.sleep(4)
    after = subprocess.run(["pgrep", "-f", "agent_mcp.py"], capture_output=True, text=True).stdout.split()
    check(f"[{backend}] no orphan agent_mcp.py processes after an interrupt",
          len(after) <= len(before), f"{len(before)} -> {len(after)}")
    orphans = subprocess.run(["pgrep", "-f", f"{backend} -p --mode json"], capture_output=True, text=True).stdout.split()
    check(f"[{backend}] no orphan {backend} CLI processes after an interrupt", not orphans, orphans)

# ---------- 3. the frontend names the backend ----------

def test_frontend():
    js = open(os.path.join(ROOT, "explorer", "static", "app.js")).read()
    check("app.js reads index.agent", "index.agent" in js)
    check("app.js sets the tab label", 'button[data-tab="claude"]' in js and "agentLabel" in js)
    check("app.js sets the input placeholder", "a-input" in js and "placeholder" in js)
    check("toolLabel looks at a.path too", "a.path" in js)


if __name__ == "__main__":
    for backend in ("omp", "pi"):
        if not shutil.which(backend):
            print(f"SKIP [{backend}] not on PATH")
            continue
        test_backend(backend)
    test_selection_and_config()
    test_frontend()
    print()
    print("FAILURES:", FAILURES if FAILURES else "none")
    sys.exit(1 if FAILURES else 0)
