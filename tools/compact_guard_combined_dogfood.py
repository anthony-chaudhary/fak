#!/usr/bin/env python3
"""Combined end-to-end dogfood: ONE `fak serve` wrap delivers BOTH halves of the value at
session length — the default-deny capability FLOOR and the 100k-session cost LEVER.

On a single `fak serve --provider anthropic --base-url <mock> --compact-history-budget`
instance it proves, in one capture:

  (1) SECURITY FLOOR  — POST /v1/fak/adjudicate with a dangerous tool call (git_push) is
      DENY (POLICY_BLOCK); a benign read (git_status) is ALLOW. The floor is not a blanket
      deny.
  (2) COST LEVER      — POST /v1/messages with a 100k+-token Claude-Code-shaped body is
      compacted on the wire: the mock upstream records a much smaller body whose
      cache_control prefix is byte-identical to the OFF-budget passthrough.

This is epic #747's headline: security AND the long-session cost lever on the same wrap,
with no agent-side change.

Usage:  python combined_dogfood.py --fak ./fak-demo.exe --out evidence.json
"""
import argparse
import json
import subprocess
import time
import threading
import http.server
import socketserver
import urllib.request
import hashlib
import random
import sys

class Recorder:
    def __init__(self): self.bodies = []

def make_handler(rec):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            rec.bodies.append(self.rfile.read(n))
            resp = json.dumps({"id":"msg_x","type":"message","role":"assistant","model":"claude-mock",
                "content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":None,
                "usage":{"input_tokens":10,"output_tokens":2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
    return H

def start_upstream(rec):
    httpd = socketserver.TCPServer(("127.0.0.1", 0), make_handler(rec))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]

def build_body(n_turns):
    system = [{"type":"text","text":"You are a coding agent. "+("Project context. "*200),
               "cache_control":{"type":"ephemeral"}}]
    msgs = [{"role":"user","content":[{"type":"text","text":"Task #0. "+("early cached context. "*60),
             "cache_control":{"type":"ephemeral"}}]}]
    for i in range(1, n_turns):
        role = "assistant" if i % 2 == 0 else "user"
        msgs.append({"role":role,"content":[{"type":"text","text":f"turn #{i} unique-{i*7919}. "+("detail. "*40)}]})
    return {"model":"claude-mock","max_tokens":64,"system":system,"messages":msgs}

def post(port, path, body, headers):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def wait_ready(port, deadline=20):
    end = time.time()+deadline
    while time.time() < end:
        for p in ("/healthz","/metrics"):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{p}", timeout=1).read()
                return True
            except Exception:
                pass
        time.sleep(0.2)
    return False

def adjudicate(port, tool, args, read_only=False):
    return post(port, "/v1/fak/adjudicate",
                {"tool":tool, "arguments":args, "read_only":read_only},
                {"Content-Type":"application/json"})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fak", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--turns", type=int, default=900)
    ap.add_argument("--budget", type=int, default=4000)
    args = ap.parse_args()

    rec = Recorder()
    httpd, up_port = start_upstream(rec)
    fak_port = random.randint(20000, 39000)
    anthropic_hdr = {"Content-Type":"application/json","x-api-key":"test","anthropic-version":"2023-06-01"}
    result = {"schema":"fak-combined-dogfood/1",
              "wire":"one `fak serve --provider anthropic --base-url <mock> --compact-history-budget` instance"}
    try:
        # First measure the OFF passthrough body for the cache-prefix reference (budget 0).
        off_proc = subprocess.Popen([args.fak,"serve","--addr",f"127.0.0.1:{fak_port}",
            "--provider","anthropic","--base-url",f"http://127.0.0.1:{up_port}",
            "--policy","examples/dev-agent-policy.json",
            "--compact-history-budget","0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not wait_ready(fak_port):
            _,err = off_proc.communicate(timeout=5)
            raise RuntimeError("OFF serve not ready: "+err.decode(errors="replace")[:1500])
        body = build_body(args.turns)
        rec.bodies.clear()
        post(fak_port, "/v1/messages", body, anthropic_hdr)
        time.sleep(0.3)
        off_fwd = rec.bodies[-1]
        off_proc.terminate()
        off_proc.wait(timeout=5)

        # Now the ON instance: prove FLOOR (adjudicate) AND LEVER (compact) on one server.
        fak_port2 = random.randint(20000, 39000)
        on_proc = subprocess.Popen([args.fak,"serve","--addr",f"127.0.0.1:{fak_port2}",
            "--provider","anthropic","--base-url",f"http://127.0.0.1:{up_port}",
            "--policy","examples/dev-agent-policy.json",
            "--compact-history-budget",str(args.budget)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not wait_ready(fak_port2):
            _,err = on_proc.communicate(timeout=5)
            raise RuntimeError("ON serve not ready: "+err.decode(errors="replace")[:1500])

        # (1) FLOOR
        deny = adjudicate(fak_port2, "git_push", {"args":"origin main"})
        allow = adjudicate(fak_port2, "git_status", {"args":""}, read_only=True)
        def verdict(d):
            for k in ("verdict","decision","status"):
                if isinstance(d, dict) and k in d:
                    return d[k]
            v = d.get("result", d)
            return v.get("verdict") if isinstance(v, dict) else d
        deny_v = json.dumps(deny)
        allow_v = json.dumps(allow)
        floor_pass = ("DENY" in deny_v.upper() or "POLICY_BLOCK" in deny_v.upper()) and \
                     ("ALLOW" in allow_v.upper())

        # (2) LEVER
        rec.bodies.clear()
        post(fak_port2, "/v1/messages", body, anthropic_hdr)
        time.sleep(0.3)
        on_fwd = rec.bodies[-1]
        on_proc.terminate()
        on_proc.wait(timeout=5)
    finally:
        httpd.shutdown()

    def et(b):
        return len(b)//4
    marker = b'"cache_control"'
    prefix_len = off_fwd.rfind(marker) + len(marker)
    prefix_identical = off_fwd[:prefix_len] == on_fwd[:prefix_len]
    shed = et(off_fwd) - et(on_fwd)
    shed_pct = round(100.0*shed/et(off_fwd), 1) if et(off_fwd) else 0.0

    result.update({
        "floor": {"deny_call":"git_push", "deny_response_excerpt": deny_v[:240],
                  "allow_call":"git_status", "allow_response_excerpt": allow_v[:240],
                  "pass": floor_pass},
        "lever": {"inbound_est_tokens": et(json.dumps(body).encode()),
                  "off_forwarded_tokens": et(off_fwd), "on_forwarded_tokens": et(on_fwd),
                  "tokens_shed": shed, "percent_shed": shed_pct,
                  "cache_prefix_byte_identical": prefix_identical,
                  "off_prefix_sha256": hashlib.sha256(off_fwd[:prefix_len]).hexdigest(),
                  "on_prefix_sha256": hashlib.sha256(on_fwd[:prefix_len]).hexdigest(),
                  "compaction_stub_present": b"[fak] compacted" in on_fwd,
                  "pass": shed > 0 and prefix_identical and b"[fak] compacted" in on_fwd},
    })
    result["verdict"] = "PASS" if (result["floor"]["pass"] and result["lever"]["pass"]) else "FAIL"
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["verdict"] == "PASS" else 1)

if __name__ == "__main__":
    main()
