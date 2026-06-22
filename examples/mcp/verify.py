#!/usr/bin/env python3
"""
fak — MCP stdio adjudication proof (zero dependencies)
======================================================

The MCP README claims your coding agent (Claude Code, Cursor, any MCP client)
can route a proposed tool call through the kernel *before* running it: point it at
`fak serve --stdio` and the `fak_*` tools appear, each call adjudicated against a
reviewable capability floor. This script *proves* that handshake end to end over
the **real MCP stdio transport** — newline-delimited JSON-RPC 2.0 over
stdin/stdout, the exact path `.mcp.json` wires — with **no model, no API key, no
GPU, no network**. Just the `fak` binary (or a Go toolchain to build it) and the
Python standard library.

It is the stdio sibling of [`../wire-proof/verify.py`](../wire-proof/verify.py),
which proves the same gate over the HTTP wire. This one covers the transport an
actual editor integration uses, which nothing else in the repo exercises.

How: spawn `fak serve --stdio --policy examples/dev-agent-policy.json`, then run
four checks and tear the server down:

  A. initialize        — the JSON-RPC handshake negotiates a protocol version and
                         returns serverInfo name "fak-gateway".
  B. tools/list        — discovery returns the fak_* adjudication tools your agent
                         will call (fak_adjudicate / fak_syscall / fak_admit).
  C. fak_adjudicate    — a shared-history mutation (git_push) is refused by the
                         floor: verdict DENY, reason POLICY_BLOCK. A DENY is
                         deny-as-VALUE (a normal tool result), never a JSON-RPC error.
  D. fak_adjudicate    — a read (git_status) is permitted: verdict ALLOW (the floor
                         is not a blanket deny).

Scope: this exercises the call-side capability gate over MCP stdio only — the same
layer as ../adjudication-demo and ../wire-proof. It does NOT exercise result-side
containment (the context-MMU quarantine / IFC taint ledger reached via fak_admit /
fak_syscall) or the deliberately non-load-bearing result detector; see ../../README.md
and ../../CLAIMS.md for the full, honest scope.

Usage:
    python3 examples/mcp/verify.py [--fak PATH] [--no-color]

Exit code: 0 if all four checks pass, 1 otherwise. CI-usable. Honors NO_COLOR.
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICY = os.path.join("examples", "dev-agent-policy.json")
SERVER_INFO_NAME = "fak-gateway"
# Versions the server negotiates (internal/gateway/mcp.go); it echoes a supported
# one or falls back to its default. We accept any of the known-good set.
SUPPORTED_PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18"}
# The adjudication verbs an MCP client relies on (tools/list returns more).
EXPECT_TOOLS = {"fak_adjudicate", "fak_syscall", "fak_admit"}
RECV_TIMEOUT = 30  # seconds to wait for one JSON-RPC response line


def color(enabled):
    if not enabled:
        return {k: "" for k in ("g", "r", "y", "b", "d", "x")}
    return {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m",
            "b": "\033[36m", "d": "\033[2m", "x": "\033[0m"}


def find_fak(explicit):
    """Return a runnable fak command (argv list). Prefer an existing binary; fall
    back to building one; last resort `go run` (slower). Mirrors wire-proof."""
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


class Server:
    """A `fak serve --stdio` child driven over newline-delimited JSON-RPC 2.0.

    Binary pipes (not text mode) so Windows can't translate the outbound `\\n`
    into `\\r\\n`. A reader thread pumps stdout lines into a queue so recv() can
    time out cross-platform (select() doesn't work on Windows pipes); a second
    thread drains stderr (the server logs there) into a buffer for diagnostics.
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

    def stderr_tail(self) -> str:
        return b"".join(self._err).decode("utf-8", "replace")[-800:]

    def send(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def recv(self):
        """Return the next JSON-RPC frame (dict). Skips blank lines; raises on
        timeout or a closed stream."""
        while True:
            line = self._q.get(timeout=RECV_TIMEOUT)   # raises queue.Empty
            if line is None:
                raise RuntimeError("server closed stdout before replying — stderr tail:\n"
                                   + self.stderr_tail())
            s = line.decode("utf-8", "replace").strip()
            if not s:
                continue
            return json.loads(s)

    def request(self, rid, method, params=None):
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        return self.recv()

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)   # a notification has no id and gets no reply

    def adjudicate(self, rid, tool):
        """Call fak_adjudicate for `tool` and return its WireVerdict dict.
        The MCP tool result carries the SyscallResponse JSON as a text block;
        a DENY is a successful result (isError:false), never a protocol error."""
        r = self.request(rid, "tools/call",
                         {"name": "fak_adjudicate", "arguments": {"tool": tool, "arguments": {}}})
        if "error" in r:
            raise RuntimeError(f"fak_adjudicate({tool}) returned a JSON-RPC error: {r['error']}")
        content = (r.get("result") or {}).get("content") or []
        text = content[0].get("text", "") if content else ""
        return json.loads(text).get("verdict", {}) if text else {}

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


def main():
    ap = argparse.ArgumentParser(description="fak MCP stdio adjudication proof (no model/key/GPU)")
    ap.add_argument("--fak", help="path to the fak binary (default: find or build it)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # survive a Windows code-page console
        except (AttributeError, ValueError):
            pass
    c = color(not args.no_color and not os.environ.get("NO_COLOR") and sys.stdout.isatty())

    fak = find_fak(args.fak)
    print(f"{c['b']}fak — MCP stdio adjudication proof{c['x']}  "
          f"{c['d']}newline-delimited JSON-RPC over stdin/stdout · no model, key, or GPU{c['x']}")
    print(f"  {c['d']}floor: {POLICY}{c['x']}\n")

    srv = Server(fak)
    fails = []
    try:
        # A. initialize — the handshake negotiates a protocol and names the server.
        try:
            r = srv.request(1, "initialize",
                            {"protocolVersion": "2024-11-05", "capabilities": {},
                             "clientInfo": {"name": "py-verifier", "version": "0"}})
            res = r.get("result", {})
            name = (res.get("serverInfo") or {}).get("name")
            proto = res.get("protocolVersion")
            ok_a = name == SERVER_INFO_NAME and proto in SUPPORTED_PROTOCOLS
            if ok_a:
                srv.notify("notifications/initialized")
                print(f"  {c['g']}✓{c['x']} A  initialize handshake  "
                      f"{c['d']}serverInfo={name} · protocol {proto}{c['x']}")
            else:
                fails.append(f"A: initialize unexpected (name={name!r} protocol={proto!r})")
                print(f"  {c['r']}✗ A  initialize: name={name!r} protocol={proto!r}{c['x']}")
        except Exception as e:
            fails.append(f"A: initialize failed: {e}")
            print(f"  {c['r']}✗ A  initialize failed: {e}{c['x']}")

        # B. tools/list — the fak_* tools an agent will call are discoverable.
        try:
            r = srv.request(2, "tools/list")
            names = {t.get("name") for t in (r.get("result") or {}).get("tools", [])}
            ok_b = EXPECT_TOOLS <= names
            if ok_b:
                print(f"  {c['g']}✓{c['x']} B  tools/list exposes the adjudication verbs  "
                      f"{c['d']}{', '.join(sorted(EXPECT_TOOLS))} (+{len(names) - len(EXPECT_TOOLS)} more){c['x']}")
            else:
                fails.append(f"B: tools/list missing {sorted(EXPECT_TOOLS - names)} (got {sorted(names)})")
                print(f"  {c['r']}✗ B  tools/list missing {sorted(EXPECT_TOOLS - names)}{c['x']}")
        except Exception as e:
            fails.append(f"B: tools/list failed: {e}")
            print(f"  {c['r']}✗ B  tools/list failed: {e}{c['x']}")

        # C. fak_adjudicate git_push — a shared-history mutation is refused.
        try:
            v = srv.adjudicate(3, "git_push")
            ok_c = v.get("kind") == "DENY" and v.get("reason") == "POLICY_BLOCK"
            if ok_c:
                print(f"  {c['g']}✓{c['x']} C  fak_adjudicate refuses git_push  "
                      f"{c['d']}DENY ({v.get('reason')}/{v.get('disposition')}){c['x']}")
            else:
                fails.append(f"C: git_push expected DENY/POLICY_BLOCK, got {v}")
                print(f"  {c['r']}✗ C  git_push expected DENY/POLICY_BLOCK, got {v}{c['x']}")
        except Exception as e:
            fails.append(f"C: fak_adjudicate(git_push) failed: {e}")
            print(f"  {c['r']}✗ C  fak_adjudicate(git_push) failed: {e}{c['x']}")

        # D. fak_adjudicate git_status — a read is allowed (not a blanket deny).
        try:
            v = srv.adjudicate(4, "git_status")
            ok_d = v.get("kind") == "ALLOW"
            if ok_d:
                print(f"  {c['g']}✓{c['x']} D  fak_adjudicate allows git_status  "
                      f"{c['d']}ALLOW{c['x']}")
            else:
                fails.append(f"D: git_status expected ALLOW, got {v}")
                print(f"  {c['r']}✗ D  git_status expected ALLOW, got {v}{c['x']}")
        except Exception as e:
            fails.append(f"D: fak_adjudicate(git_status) failed: {e}")
            print(f"  {c['r']}✗ D  fak_adjudicate(git_status) failed: {e}{c['x']}")

        print()
        if fails:
            print(f"{c['b']}summary:{c['x']} {c['r']}FAILED{c['x']}  ·  " + "  ·  ".join(fails))
            return 1
        print(f"{c['b']}summary:{c['x']} {c['g']}PASS{c['x']}  ·  the kernel adjudicated every proposed "
              f"call over the MCP stdio transport, with no model, key, or GPU.\n"
              f"{c['d']}  this is the path your editor's MCP client uses (.mcp.json wires `fak serve --stdio`).{c['x']}")
        return 0
    finally:
        srv.close()


if __name__ == "__main__":
    sys.exit(main())
