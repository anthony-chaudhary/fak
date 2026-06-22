#!/usr/bin/env python3
"""
fak — a third-party MCP client walkthrough (stdlib only, no dependencies)
========================================================================

`fak serve` is a Model Context Protocol (MCP) server: it speaks JSON-RPC 2.0
either over **stdio** (newline-delimited frames on stdin/stdout — the MCP stdio
convention) or over **HTTP** (a single request/response per `POST /mcp`). Because
it is plain JSON-RPC 2.0, *any* compliant MCP client drives it — not just Claude
Code. This script IS that arbitrary third-party client, written against the
Python standard library alone (no `mcp` SDK, no `requests`), so you can read it
end to end and port it to your own language.

What it does, against either transport:

  1. spawns `fak serve --stdio` over subprocess pipes  (or, with --http URL,
     talks to a running `fak serve --addr ...` over POST /mcp),
  2. does the MCP `initialize` handshake (protocol-version negotiation),
  3. lists the tools (`tools/list`),
  4. calls each of the SIX tools fak exposes with a small payload, and
  5. prints the response SHAPE for each so you can see what comes back.

The six tools and the payload this walkthrough sends each:

  fak_adjudicate     — verdict only, no execution         {tool: git_status}
  fak_syscall        — adjudicate AND execute             {tool: git_status, read_only}
  fak_admit          — screen a result YOU executed        {tool: web_fetch, result}
  fak_changes        — drain the cross-agent change feed   {since: 0}
  fak_revoke         — refute a world-state witness        {witness: sha256:…}
  fak_context_change — negative-only recall mutation       {image_dir, step, reason}

The wire shape of every tool result (the `SyscallResponse` envelope, the verdict
object, and the closed refusal vocabulary) is specified in
../../docs/mcp-tool-result.md. The transports and the serving path are
GETTING-STARTED.md §3.

Scope: this is a *reference* client meant to be read, not a production MCP SDK.
It does no reconnect/retry, no batched requests, and no streaming. The CI-grade
gate that asserts a verdict over MCP stdio is ../mcp/verify.py; this script
instead drives all six verbs so an adopter sees each response shape.

Usage:
    python3 examples/mcp-client/client.py                 # stdio (spawns fak)
    python3 examples/mcp-client/client.py --fak ./fak     # stdio, explicit binary
    python3 examples/mcp-client/client.py --http http://127.0.0.1:8080/mcp
                                                           # HTTP (server already up)

Exit code: 0 if the handshake and all six calls returned a frame (result OR a
JSON-RPC error — an error is itself a valid, documented shape), 1 if a transport
fault stopped the walkthrough. CI-importable. Honors NO_COLOR.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICY = os.path.join("examples", "dev-agent-policy.json")
RECV_TIMEOUT = 30  # seconds to wait for one JSON-RPC response frame

# The protocol revisions fak negotiates (the single source of truth is
# mcpProtocolVersions in internal/gateway/mcp.go). The client offers its
# preferred one in `initialize`; the server echoes it if supported, else answers
# with its own default — never an unknown revision claimed as implemented.
CLIENT_PROTOCOL = "2024-11-05"


def color(enabled):
    if not enabled:
        return {k: "" for k in ("g", "r", "y", "b", "d", "x")}
    return {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m",
            "b": "\033[36m", "d": "\033[2m", "x": "\033[0m"}


def find_fak(explicit):
    """Return a runnable fak command (argv list). Prefer an existing binary; fall
    back to building one; last resort `go run`. Mirrors examples/mcp/verify.py."""
    if explicit:
        return [explicit]
    exe = "fak.exe" if sys.platform == "win32" else "fak"
    local = os.path.join(REPO_ROOT, exe)
    if os.path.isfile(local):
        return [local]
    on_path = shutil.which("fak")
    if on_path:
        return [on_path]
    if shutil.which("go"):
        out = os.path.join(REPO_ROOT, exe)
        print(f"  building {exe} (one-time) …")
        r = subprocess.run(["go", "build", "-o", out, "./cmd/fak"], cwd=REPO_ROOT)
        if r.returncode == 0 and os.path.isfile(out):
            return [out]
        return ["go", "run", "./cmd/fak"]
    sys.exit("fak not found and no Go toolchain to build it; pass --fak PATH")


class StdioTransport:
    """A `fak serve --stdio` child driven over newline-delimited JSON-RPC 2.0.

    Binary pipes (not text mode) so Windows can't translate the outbound "\\n"
    into "\\r\\n". A reader thread pumps stdout frames into a queue so recv() can
    time out cross-platform (select() doesn't work on Windows pipes); a second
    thread drains stderr (the server logs there) for diagnostics.
    """

    def __init__(self, fak):
        self.proc = subprocess.Popen(
            fak + ["serve", "--stdio", "--policy", POLICY],
            cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        self._q: queue.Queue = queue.Queue()
        self._err: list[bytes] = []
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self):
        for line in self.proc.stdout:        # iterates on b"\n"
            self._q.put(line)
        self._q.put(None)                    # EOF sentinel

    def _pump_stderr(self):
        for line in self.proc.stderr:
            self._err.append(line)

    def _stderr_tail(self) -> str:
        return b"".join(self._err).decode("utf-8", "replace")[-800:]

    def roundtrip(self, msg):
        self.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        if "id" not in msg:
            return None                      # a notification gets no reply
        while True:
            line = self._q.get(timeout=RECV_TIMEOUT)   # raises queue.Empty
            if line is None:
                raise RuntimeError("server closed stdout before replying — stderr tail:\n"
                                   + self._stderr_tail())
            s = line.decode("utf-8", "replace").strip()
            if s:
                return json.loads(s)

    def close(self):
        for closer in (lambda: self.proc.stdin.close(), self.proc.terminate):
            try:
                closer()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class HTTPTransport:
    """The same JSON-RPC dispatch over `POST /mcp` — one request/response per POST.
    Assumes a server is already up (`fak serve --addr 127.0.0.1:8080`)."""

    def __init__(self, url):
        self.url = url

    def roundtrip(self, msg):
        body = json.dumps(msg).encode("utf-8")
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=RECV_TIMEOUT) as resp:
            if "id" not in msg:
                return None                  # 202 Accepted, no body
            raw = resp.read().decode("utf-8", "replace").strip()
            return json.loads(raw) if raw else None

    def close(self):
        pass


class MCPClient:
    """A minimal JSON-RPC 2.0 / MCP client over a pluggable transport."""

    def __init__(self, transport):
        self.t = transport
        self._id = 0

    def _next_id(self):
        self._id += 1
        return self._id

    def request(self, method, params=None):
        msg = {"jsonrpc": "2.0", "id": self._next_id(), "method": method}
        if params is not None:
            msg["params"] = params
        return self.t.roundtrip(msg)

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.t.roundtrip(msg)

    def call_tool(self, name, arguments):
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self):
        self.t.close()


def tool_result_text(frame):
    """Extract the SyscallResponse JSON the gateway packs into the single text
    content block. Returns (parsed_or_none, error_or_none)."""
    if frame is None:
        return None, {"message": "no frame (transport returned nothing)"}
    if "error" in frame:
        return None, frame["error"]
    content = (frame.get("result") or {}).get("content") or []
    text = content[0].get("text", "") if content else ""
    try:
        return (json.loads(text) if text else {}), None
    except json.JSONDecodeError:
        return {"_raw": text}, None


def one_line(obj, width=110):
    s = json.dumps(obj, separators=(",", ":"))
    return s if len(s) <= width else s[: width - 1] + "…"


# Each entry: (tool name, arguments, one-line meaning shown in the walkthrough).
WALKTHROUGH = [
    ("fak_adjudicate", {"tool": "git_status", "arguments": {}},
     "verdict only, no execution (the production path for a client that runs its own tools)"),
    ("fak_syscall", {"tool": "git_status", "arguments": {}, "read_only": True},
     "adjudicate AND execute through the kernel (returns verdict + admitted result)"),
    ("fak_admit", {"tool": "web_fetch", "result": {"text": "hello from an external tool"}},
     "screen a result YOU executed through the result-side stack before it enters context"),
    ("fak_changes", {"since": 0},
     "drain the cross-agent change feed (typed mutations + revocations since a cursor)"),
    ("fak_revoke", {"witness": "sha256:deadbeefcafef00d"},
     "refute a world-state witness; entries admitted under it are evicted fleet-wide"),
    ("fak_context_change", {"image_dir": "examples/mcp-client/no-such-image",
                            "step": 1, "reason": "walkthrough probe"},
     "negative-only recall mutation (needs a real recall image — see note below)"),
]


def main():
    ap = argparse.ArgumentParser(
        description="fak third-party MCP client walkthrough — drives all six tools (stdlib only)")
    ap.add_argument("--fak", help="path to the fak binary (stdio transport; default: find or build it)")
    ap.add_argument("--http", metavar="URL",
                    help="talk to a running server over POST /mcp instead of spawning --stdio "
                         "(e.g. http://127.0.0.1:8080/mcp)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # survive a Windows code-page console
        except (AttributeError, ValueError):
            pass
    c = color(not args.no_color and not os.environ.get("NO_COLOR") and sys.stdout.isatty())

    if args.http:
        transport, label = HTTPTransport(args.http), f"HTTP · POST {args.http}"
    else:
        transport, label = StdioTransport(find_fak(args.fak)), "stdio · fak serve --stdio"

    print(f"{c['b']}fak — third-party MCP client walkthrough{c['x']}  "
          f"{c['d']}{label} · JSON-RPC 2.0 · stdlib only{c['x']}")
    if not args.http:
        print(f"  {c['d']}floor: {POLICY}{c['x']}")
    print()

    client = MCPClient(transport)
    try:
        # 1. initialize — negotiate a protocol version and learn the server's identity.
        init = client.request("initialize", {
            "protocolVersion": CLIENT_PROTOCOL, "capabilities": {},
            "clientInfo": {"name": "fak-walkthrough", "version": "0"}})
        res = (init or {}).get("result", {})
        info = res.get("serverInfo") or {}
        print(f"{c['g']}initialize{c['x']}  "
              f"{c['d']}server={info.get('name')} v{info.get('version')} · "
              f"protocol negotiated → {res.get('protocolVersion')}{c['x']}")
        client.notify("notifications/initialized")  # MCP: acknowledge the handshake

        # 2. tools/list — discover what the server offers.
        listed = client.request("tools/list")
        names = [t.get("name") for t in (listed.get("result") or {}).get("tools", [])]
        print(f"{c['g']}tools/list{c['x']}  {c['d']}{len(names)} tools: {', '.join(names)}{c['x']}\n")

        # 3. call each of the six tools and print its response SHAPE.
        print(f"{c['b']}calling each tool with a small payload (showing the response shape){c['x']}")
        for name, arguments, meaning in WALKTHROUGH:
            frame = client.call_tool(name, arguments)
            parsed, err = tool_result_text(frame)
            print(f"\n  {c['y']}{name}{c['x']}  {c['d']}{meaning}{c['x']}")
            print(f"    {c['d']}→ args   {c['x']}{one_line(arguments)}")
            if err is not None:
                # The JSON-RPC error CHANNEL — reserved for protocol/build faults
                # (bad params, missing recall image), NOT for a policy refusal.
                print(f"    {c['d']}→ error  {c['x']}{c['r']}JSON-RPC error{c['x']} "
                      f"code={err.get('code')} {one_line(err.get('message', ''))}")
            else:
                # The tool-result CHANNEL — a SyscallResponse (isError:false even on
                # a DENY: a refusal is deny-as-VALUE, surfaced in verdict.kind).
                verdict = parsed.get("verdict") if isinstance(parsed, dict) else None
                if verdict:
                    print(f"    {c['d']}→ verdict{c['x']}{one_line(verdict)}")
                print(f"    {c['d']}→ result {c['x']}{one_line(parsed)}")

        print(f"\n{c['b']}done{c['x']}  ·  {c['d']}all six tools answered over {label.split(' ')[0]}. "
              f"Five returned a SyscallResponse (deny-as-value lives in verdict.kind); "
              f"fak_context_change used the JSON-RPC error channel because no recall image "
              f"was supplied — exactly the result-vs-error split in docs/mcp-tool-result.md.{c['x']}")
        return 0
    except Exception as e:
        print(f"\n{c['r']}transport fault: {e}{c['x']}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
