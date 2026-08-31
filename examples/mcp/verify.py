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
six checks and tear the server down:

  A. initialize        — the JSON-RPC handshake negotiates a protocol version and
                         returns serverInfo name "fak-gateway".
  B. tools/list        — bootstrap discovery returns the schema-light tools exposed
                         eagerly (fak_adjudicate / fak_syscall / fak_tools_search).
  C. fak_tools_search  — deferred discovery finds fak_admit without inflating the
                         bootstrap tools/list response.
  D. fak_admit         — a benign client-run result enters the result-side stack and
                         returns the typed DEFER/OK admission envelope intact.
  E. fak_adjudicate    — a shared-history mutation (git_push) is refused by the
                         floor: verdict DENY, reason POLICY_BLOCK. A DENY is
                         deny-as-VALUE (a normal tool result), never a JSON-RPC error.
  F. fak_adjudicate    — a read (git_status) is permitted: verdict ALLOW (the floor
                         is not a blanket deny).

Scope: this exercises both call-side adjudication and a benign result-side admission
over MCP stdio. It proves the context-MMU/IFC entrypoint is discoverable and routable;
it does not claim that the deliberately non-load-bearing result detector catches every
poisoned payload. See ../../README.md and ../../CLAIMS.md for the full, honest scope.

Usage:
    python3 examples/mcp/verify.py [--fak PATH] [--no-color]

Exit code: 0 if all six checks pass, 1 otherwise. CI-usable. Honors NO_COLOR.
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
# Schema-light bootstrap tools stay eager. Result admission is intentionally deferred
# and must remain discoverable through fak_tools_search (#3231).
EXPECT_BOOTSTRAP_TOOLS = {"fak_adjudicate", "fak_syscall", "fak_tools_search"}
DEFERRED_TOOL = "fak_admit"
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

    def call_tool(self, rid, name, arguments):
        """Call one MCP tool and decode its JSON text result."""
        r = self.request(rid, "tools/call", {"name": name, "arguments": arguments})
        if "error" in r:
            raise RuntimeError(f"{name} returned a JSON-RPC error: {r['error']}")
        content = (r.get("result") or {}).get("content") or []
        text = content[0].get("text", "") if content else ""
        if not text:
            raise RuntimeError(f"{name} returned no text result")
        return json.loads(text)

    def adjudicate(self, rid, tool):
        """Call fak_adjudicate for `tool` and return its WireVerdict dict.
        A DENY is a successful result (isError:false), never a protocol error."""
        return self.call_tool(rid, "fak_adjudicate",
                              {"tool": tool, "arguments": {}}).get("verdict", {})

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

        # B. tools/list — only the schema-light bootstrap tools are eager.
        try:
            r = srv.request(2, "tools/list")
            names = {t.get("name") for t in (r.get("result") or {}).get("tools", [])}
            missing = EXPECT_BOOTSTRAP_TOOLS - names
            ok_b = not missing and DEFERRED_TOOL not in names
            if ok_b:
                print(f"  {c['g']}✓{c['x']} B  tools/list exposes schema-light bootstrap tools  "
                      f"{c['d']}{', '.join(sorted(EXPECT_BOOTSTRAP_TOOLS))}; {DEFERRED_TOOL} deferred{c['x']}")
            else:
                fails.append(f"B: bootstrap missing {sorted(missing)} or {DEFERRED_TOOL} was eager "
                             f"(got {sorted(names)})")
                print(f"  {c['r']}✗ B  unexpected tools/list surface: {sorted(names)}{c['x']}")
        except Exception as e:
            fails.append(f"B: tools/list failed: {e}")
            print(f"  {c['r']}✗ B  tools/list failed: {e}{c['x']}")

        # C. fak_tools_search — deferred fak_admit remains discoverable.
        try:
            found = srv.call_tool(3, "fak_tools_search",
                                  {"query": DEFERRED_TOOL, "detail_level": "name"})
            found_names = {t.get("name") for t in found.get("tools", [])}
            ok_c = DEFERRED_TOOL in found_names
            if ok_c:
                print(f"  {c['g']}✓{c['x']} C  fak_tools_search discovers deferred fak_admit  "
                      f"{c['d']}routable without eager schema load{c['x']}")
            else:
                fails.append(f"C: {DEFERRED_TOOL} not found (got {sorted(found_names)})")
                print(f"  {c['r']}✗ C  {DEFERRED_TOOL} not found: {sorted(found_names)}{c['x']}")
        except Exception as e:
            fails.append(f"C: fak_tools_search failed: {e}")
            print(f"  {c['r']}✗ C  fak_tools_search failed: {e}{c['x']}")

        # D. fak_admit — route a benign client-run result through result admission.
        try:
            benign = {"status": "ok", "source": "mcp-stdio-verifier"}
            admitted = srv.call_tool(4, "fak_admit",
                                     {"tool": "git_status", "result": benign,
                                      "trace_id": "mcp-stdio-verifier"})
            verdict = admitted.get("verdict", {})
            result = admitted.get("result", {})
            decoded = json.loads(result.get("content", "null"))
            ok_d = (verdict.get("kind") == "DEFER" and result.get("status") == "OK"
                    and decoded == benign and (result.get("meta") or {}).get("ifc_taint") == "tainted")
            if ok_d:
                print(f"  {c['g']}✓{c['x']} D  fak_admit screens a benign result  "
                      f"{c['d']}DEFER · result OK · IFC taint recorded{c['x']}")
            else:
                fails.append(f"D: unexpected admission envelope {admitted}")
                print(f"  {c['r']}✗ D  unexpected fak_admit result: {admitted}{c['x']}")
        except Exception as e:
            fails.append(f"D: fak_admit failed: {e}")
            print(f"  {c['r']}✗ D  fak_admit failed: {e}{c['x']}")

        # E. fak_adjudicate git_push — a shared-history mutation is refused.
        try:
            v = srv.adjudicate(5, "git_push")
            ok_e = v.get("kind") == "DENY" and v.get("reason") == "POLICY_BLOCK"
            if ok_e:
                print(f"  {c['g']}✓{c['x']} E  fak_adjudicate refuses git_push  "
                      f"{c['d']}DENY ({v.get('reason')}/{v.get('disposition')}){c['x']}")
            else:
                fails.append(f"E: git_push expected DENY/POLICY_BLOCK, got {v}")
                print(f"  {c['r']}✗ E  git_push expected DENY/POLICY_BLOCK, got {v}{c['x']}")
        except Exception as e:
            fails.append(f"E: fak_adjudicate(git_push) failed: {e}")
            print(f"  {c['r']}✗ E  fak_adjudicate(git_push) failed: {e}{c['x']}")

        # F. fak_adjudicate git_status — a read is allowed (not a blanket deny).
        try:
            v = srv.adjudicate(6, "git_status")
            ok_f = v.get("kind") == "ALLOW"
            if ok_f:
                print(f"  {c['g']}✓{c['x']} F  fak_adjudicate allows git_status  "
                      f"{c['d']}ALLOW{c['x']}")
            else:
                fails.append(f"F: git_status expected ALLOW, got {v}")
                print(f"  {c['r']}✗ F  git_status expected ALLOW, got {v}{c['x']}")
        except Exception as e:
            fails.append(f"F: fak_adjudicate(git_status) failed: {e}")
            print(f"  {c['r']}✗ F  fak_adjudicate(git_status) failed: {e}{c['x']}")

        print()
        if fails:
            print(f"{c['b']}summary:{c['x']} {c['r']}FAILED{c['x']}  ·  " + "  ·  ".join(fails))
            return 1
        print(f"{c['b']}summary:{c['x']} {c['g']}PASS{c['x']}  ·  the kernel admitted a client result and adjudicated proposed "
              f"calls over the MCP stdio transport, with no model, key, or GPU.\n"
              f"{c['d']}  this is the path your editor's MCP client uses (.mcp.json wires `fak serve --stdio`).{c['x']}")
        return 0
    finally:
        srv.close()


if __name__ == "__main__":
    sys.exit(main())
