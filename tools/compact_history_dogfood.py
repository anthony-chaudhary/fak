#!/usr/bin/env python3
"""End-to-end dogfood of fak's 100k-session value lever.

Proves, on the REAL `fak serve --provider anthropic --base-url <mock> --compact-history-budget`
wire path (not a unit test), that the cache-prefix-preserving history compaction:

  1. SHEDS history tokens on a large (100k-token-class) Claude-Code-shaped session body, and
  2. Keeps the cache_control prefix BYTE-IDENTICAL between OFF and ON, so the upstream
     prompt-cache hit survives (the whole point — naive compaction would break the cache and
     cost MORE on a long session), and
  3. Forwards a still-valid body the upstream accepts (the session keeps working).

It does this by standing up a mock Anthropic upstream that records the EXACT bytes fak
forwards, sending the same big body with the budget OFF then ON, and diffing.

Usage:  python compact_dogfood.py --fak ./fak-demo.exe --out evidence.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
import threading
import http.server
import socketserver
import urllib.request
import hashlib

# --- mock Anthropic upstream: records the raw body fak forwards, returns a valid message ---
class Recorder:
    def __init__(self):
        self.bodies = []  # list of raw request bytes fak forwarded upstream

def make_handler(rec):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            rec.bodies.append(body)
            resp = json.dumps({
                "id": "msg_dogfood", "type": "message", "role": "assistant",
                "model": "claude-mock", "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
    return H

def start_upstream(rec):
    httpd = socketserver.TCPServer(("127.0.0.1", 0), make_handler(rec))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port

# --- build a realistic large Claude-Code-shaped /v1/messages body (100k-token class) ---
def build_body(n_turns):
    """A system array ending in a cache_control breakpoint + n_turns of UNIQUE alternating
    turns; turn 0 (user) carries a per-message breakpoint. Unique content per turn so the
    gateway's repetition-loop breaker never short-circuits before forwarding."""
    system = [{
        "type": "text",
        "text": "You are a coding agent. " + ("Project context and standing instructions. " * 200),
        "cache_control": {"type": "ephemeral"},
    }]
    msgs = []
    msgs.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": "Initial task framing #0. " + ("early cached project context. " * 60),
            "cache_control": {"type": "ephemeral"},
        }],
    })
    for i in range(1, n_turns):
        role = "assistant" if i % 2 == 0 else "user"
        # UNIQUE body per turn (index in the text) so no two turns are verbatim-equal.
        msgs.append({
            "role": role,
            "content": [{"type": "text",
                         "text": f"conversation turn #{i} unique-{i*7919}. " + ("detail line. " * 40)}],
        })
    return {"model": "claude-mock", "max_tokens": 64, "system": system, "messages": msgs}

def est_tokens(raw):  # match EstimateAnthropicTokens ~4 chars/token
    return len(raw) // 4

def post_messages(port, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-api-key": "test",
                 "anthropic-version": "2023-06-01"},
        method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()

def wait_ready(port, deadline=20):
    end = time.time() + deadline
    while time.time() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1).read()
            return True
        except Exception:
            try:  # some builds expose readiness differently; a refused conn means not up yet
                urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1).read()
                return True
            except Exception:
                time.sleep(0.2)
    return False

def run_case(fak, upstream_port, budget, body):
    # Each case gets a FRESH fak serve on its own port so state never leaks between OFF/ON.
    import random
    fak_port = random.randint(20000, 39000)
    env = dict(os.environ)
    args = [fak, "serve", "--addr", f"127.0.0.1:{fak_port}",
            "--provider", "anthropic",
            "--base-url", f"http://127.0.0.1:{upstream_port}",
            "--compact-history-budget", str(budget)]
    proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if not wait_ready(fak_port):
            out, err = b"", b""
            try:
                proc.terminate()
                out, err = proc.communicate(timeout=5)
            except Exception:
                pass
            raise RuntimeError(f"fak serve (budget={budget}) not ready on {fak_port}\nSTDERR:\n{err.decode(errors='replace')[:2000]}")
        resp = post_messages(fak_port, body)
        time.sleep(0.3)  # let the recorder thread land the body
        return resp, fak_port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fak", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--turns", type=int, default=900)
    ap.add_argument("--budget", type=int, default=4000)
    args = ap.parse_args()

    rec = Recorder()
    httpd, up_port = start_upstream(rec)
    try:
        body = build_body(args.turns)
        body_raw = json.dumps(body).encode()
        in_tokens = est_tokens(body_raw)

        # CASE OFF (budget 0): body must be forwarded byte-for-byte.
        rec.bodies.clear()
        run_case(args.fak, up_port, 0, body)
        if not rec.bodies:
            raise RuntimeError("OFF case: upstream received no body (gateway short-circuited?)")
        off_fwd = rec.bodies[-1]

        # CASE ON (budget N): body must be compacted but cache prefix byte-identical.
        rec.bodies.clear()
        run_case(args.fak, up_port, args.budget, body)
        if not rec.bodies:
            raise RuntimeError("ON case: upstream received no body")
        on_fwd = rec.bodies[-1]
    finally:
        httpd.shutdown()

    # --- analysis ---
    # Find the cache prefix boundary: the byte range through the LAST cache_control in the
    # ORIGINAL inbound body. The forwarded OFF body should equal the inbound body (passthrough).
    # We assert the ON body's prefix (up to the last cache_control marker) is byte-identical to OFF's.
    marker = b'"cache_control"'
    last_cc_off = off_fwd.rfind(marker)
    # Prefix we compare: everything up to and including the end of the message holding the last
    # cache_control. A robust, conservative proxy: compare up to the last cache_control offset
    # (this lies strictly inside the protected prefix by construction).
    prefix_len = last_cc_off + len(marker)
    off_prefix = off_fwd[:prefix_len]
    on_prefix = on_fwd[:prefix_len]
    prefix_identical = (off_prefix == on_prefix)

    off_valid = is_valid_messages(off_fwd)
    on_valid = is_valid_messages(on_fwd)
    stub_present = b"[fak] compacted" in on_fwd

    off_tokens = est_tokens(off_fwd)
    on_tokens = est_tokens(on_fwd)
    shed = off_tokens - on_tokens
    shed_pct = round(100.0 * shed / off_tokens, 1) if off_tokens else 0.0

    result = {
        "schema": "fak-compact-dogfood/1",
        "wire": "fak serve --provider anthropic --base-url <mock-upstream> --compact-history-budget",
        "input": {"turns": args.turns, "budget_tokens": args.budget, "inbound_est_tokens": in_tokens,
                  "inbound_bytes": len(body_raw)},
        "off": {"forwarded_bytes": len(off_fwd), "est_tokens": off_tokens, "valid_messages": off_valid,
                "byte_identical_to_inbound": off_fwd == body_raw},
        "on": {"forwarded_bytes": len(on_fwd), "est_tokens": on_tokens, "valid_messages": on_valid,
               "compaction_stub_present": stub_present},
        "shed": {"tokens": shed, "percent": shed_pct},
        "cache_prefix": {"compared_prefix_bytes": prefix_len,
                         "byte_identical_off_vs_on": prefix_identical,
                         "off_prefix_sha256": hashlib.sha256(off_prefix).hexdigest(),
                         "on_prefix_sha256": hashlib.sha256(on_prefix).hexdigest()},
    }
    # The headline pass: tokens shed > 0, prefix byte-identical, ON body still valid.
    passed = (shed > 0 and prefix_identical and on_valid and off_valid and stub_present)
    result["verdict"] = "PASS" if passed else "FAIL"

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(0 if passed else 1)

def is_valid_messages(raw):
    try:
        obj = json.loads(raw)
        return isinstance(obj.get("messages"), list) and len(obj["messages"]) >= 1
    except Exception:
        return False

if __name__ == "__main__":
    main()
