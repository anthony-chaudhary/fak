#!/usr/bin/env python3
"""
fak — over-the-wire adjudication proof (zero dependencies)
==========================================================

The integration index claims an agent "drops in" behind `fak serve` with no
agent-side code change: every tool call it proposes passes through the capability
floor first. This script *proves* that over HTTP, with **no model, no API key, no
GPU, no ollama** — just the `fak` binary (or a Go toolchain to build it) and the
Python standard library.

How: `fak serve` with no `--base-url` serves a deterministic offline mock planner.
We start it, then check three things the index documents:

  A. OpenAI wire, inline adjudication — a normal POST /v1/chat/completions comes
     back as a standard chat completion whose proposed tool call carries the
     kernel's verdict inline, in a top-level "fak" block. (The gate is just there.)
  B. Structural DENY — POST /v1/fak/adjudicate on a tool that is NOT on the
     allow-list is refused by structure: verdict DENY, reason POLICY_BLOCK.
  C. Not a blanket deny — an allow-listed tool returns ALLOW.

This is the call-side capability gate only (see ../adjudication-demo/README.md for
the same gate driven by a real model, and the repo README for containment + the
honest scope of the result detector).

Usage:
    python3 examples/wire-proof/verify.py [--fak PATH] [--no-color]

Exit code: 0 if all three checks pass, 1 otherwise. CI-usable. Honors NO_COLOR.
"""
from __future__ import annotations
import argparse, json, os, shutil, socket, subprocess, sys, time
import urllib.error, urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICY = os.path.join("examples", "customer-support-readonly-policy.json")
BOOT_TIMEOUT = 30   # seconds to wait for the server to answer /healthz


def color(enabled):
    if not enabled:
        return {k: "" for k in ("g", "r", "y", "b", "d", "x")}
    return {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m",
            "b": "\033[36m", "d": "\033[2m", "x": "\033[0m"}


def find_fak(explicit):
    """Return a runnable fak command (list of argv[0..]). Prefer an existing binary;
    fall back to building one; last resort `go run` (slower, compiles each call)."""
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
        return ["go", "run", "./cmd/fak"]   # building failed; try the slow path
    sys.exit("fak not found and no Go toolchain to build it; pass --fak PATH")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def post_json(url, obj):
    body = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def wait_healthy(base):
    deadline = time.time() + BOOT_TIMEOUT
    while time.time() < deadline:
        try:
            h = json.load(urllib.request.urlopen(base + "/healthz", timeout=2))
            if h.get("ok"):
                return h
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.3)
    return None


def main():
    ap = argparse.ArgumentParser(description="fak over-the-wire adjudication proof (no model/key/GPU)")
    ap.add_argument("--fak", help="path to the fak binary (default: find or build it)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    # UTF-8 the output streams so the ✓/✗/→ glyphs survive a Windows code-page console.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    c = color(not args.no_color and not os.environ.get("NO_COLOR") and sys.stdout.isatty())

    fak = find_fak(args.fak)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    print(f"{c['b']}fak — over-the-wire adjudication proof{c['x']}  "
          f"{c['d']}offline mock planner · no model, key, or GPU{c['x']}")

    proc = subprocess.Popen(
        fak + ["serve", "--addr", f"127.0.0.1:{port}", "--policy", POLICY],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        health = wait_healthy(base)
        if not health:
            out = ""
            try:
                proc.terminate(); out = (proc.communicate(timeout=5)[0] or "")
            except Exception:
                pass
            print(f"{c['r']}✗ server did not become healthy in {BOOT_TIMEOUT}s{c['x']}\n{c['d']}{out[-800:]}{c['x']}")
            return 1
        print(f"  {c['d']}serving: engine={health.get('engine')} planner={health.get('planner')} on {base}{c['x']}\n")

        fails = []

        # A. OpenAI wire — the proposed tool call is adjudicated inline, in a "fak" block.
        chat = post_json(base + "/v1/chat/completions",
                         {"model": "mock", "messages": [{"role": "user", "content": "refund my last order"}]})
        adj = (chat.get("fak") or {}).get("adjudications") or []
        tool_calls = chat.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
        ok_a = bool(tool_calls) and bool(adj) and isinstance(adj[0].get("verdict"), dict) and adj[0]["verdict"].get("kind")
        if ok_a:
            v = adj[0]
            print(f"  {c['g']}✓{c['x']} A  OpenAI wire carries inline adjudication  "
                  f"{c['d']}{v['tool']} → {v['verdict']['kind']} (admitted={v.get('admitted')}){c['x']}")
        else:
            fails.append("A: /v1/chat/completions response carried no 'fak.adjudications' verdict")
            print(f"  {c['r']}✗ A  no inline adjudication in the chat response{c['x']}  {c['d']}{str(chat)[:160]}{c['x']}")

        # B. Structural DENY — a non-allow-listed tool is refused by structure.
        deny = post_json(base + "/v1/fak/adjudicate", {"tool": "refund_payment", "arguments": {"amount": 500}})
        vd = deny.get("verdict", {})
        ok_b = vd.get("kind") == "DENY" and vd.get("reason") == "POLICY_BLOCK"
        if ok_b:
            print(f"  {c['g']}✓{c['x']} B  unsanctioned tool refused by structure  "
                  f"{c['d']}refund_payment → DENY ({vd.get('reason')}/{vd.get('disposition')}){c['x']}")
        else:
            fails.append(f"B: refund_payment expected DENY/POLICY_BLOCK, got {vd}")
            print(f"  {c['r']}✗ B  expected DENY/POLICY_BLOCK, got {vd}{c['x']}")

        # C. Not a blanket deny — an allow-listed tool is allowed.
        allow = post_json(base + "/v1/fak/adjudicate", {"tool": "search_kb", "arguments": {"q": "refund window"}})
        va = allow.get("verdict", {})
        ok_c = va.get("kind") == "ALLOW"
        if ok_c:
            print(f"  {c['g']}✓{c['x']} C  allow-listed tool permitted  {c['d']}search_kb → ALLOW{c['x']}")
        else:
            fails.append(f"C: search_kb expected ALLOW, got {va}")
            print(f"  {c['r']}✗ C  expected ALLOW, got {va}{c['x']}")

        print()
        if fails:
            print(f"{c['b']}summary:{c['x']} {c['r']}FAILED{c['x']}  ·  " + "  ·  ".join(fails))
            return 1
        print(f"{c['b']}summary:{c['x']} {c['g']}PASS{c['x']}  ·  the gate adjudicated every proposed call over the wire, "
              f"with no model, key, or GPU.\n"
              f"{c['d']}  swap the offline mock for your real engine by adding --base-url; nothing else changes.{c['x']}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
