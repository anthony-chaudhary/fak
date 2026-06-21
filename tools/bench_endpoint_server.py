#!/usr/bin/env python3
"""bench_endpoint_server.py -- zero-dependency OpenAI-compatible STAND-IN endpoint.

The friction in standing up a Tailscale benchmark endpoint is "install + load a
model first". This removes it: a stdlib-only (http.server) server that speaks just
enough of the OpenAI API to (a) make the endpoint go READY in fleet_endpoints, and
(b) round-trip the harness's transport + latency checks -- with NO model download.
It is the "testing" half of the benchmark-endpoint goal; for real model
benchmarking, point the same port at ollama / llama.cpp / local_shim.py instead.

  GET  /v1/models            -> { data: [ {id: <model>} ] }   (run_local_model.sh's readiness probe)
  POST /v1/chat/completions  -> deterministic echo reply in OpenAI shape (+ usage, latency ms)
  GET  /healthz              -> "ok"

SAFETY: bind to the node's TAILSCALE IP (tailnet-only) -- never 0.0.0.0 on a
personal laptop. Default host is 127.0.0.1 so a bare run exposes nothing; you must
pass the tailscale IP explicitly to make it reachable by the driver.

  # on the endpoint (its own tailscale IP):
  python tools/bench_endpoint_server.py --host 100.x.x.x --port 8131
  # tear down: Ctrl-C  (nothing persists)
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "bench-echo"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/healthz":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/v1/models"):
            self._send(200, {"object": "list",
                             "data": [{"id": self.server.model, "object": "model",
                                       "owned_by": "fleet-bench"}]})
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        t0 = time.time()
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            req = {}
        msgs = req.get("messages", [])
        last_user = next((m.get("content", "") for m in reversed(msgs)
                          if m.get("role") == "user"), "")
        reply = f"[bench-echo] received {len(msgs)} message(s); last user said: {last_user!r}"
        prompt_toks = sum(len(str(m.get("content", "")).split()) for m in msgs)
        compl_toks = len(reply.split())
        self._send(200, {
            "id": f"chatcmpl-bench-{int(t0 * 1000)}",
            "object": "chat.completion",
            "created": int(t0),
            "model": req.get("model", self.server.model),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": prompt_toks, "completion_tokens": compl_toks,
                      "total_tokens": prompt_toks + compl_toks},
            "x_bench_latency_ms": round((time.time() - t0) * 1000, 3),
        })

    def log_message(self, fmt, *args):
        # one concise line per request to stderr (server-side liveness trace)
        print("[bench] %s - %s" % (self.address_string(), fmt % args))


def main():
    ap = argparse.ArgumentParser(description="zero-dep OpenAI-compatible stand-in endpoint")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (use the node's TAILSCALE IP for tailnet-only reach; never 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8131)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.model = args.model
    where = "TAILNET-REACHABLE" if not args.host.startswith("127.") else "localhost-only"
    print(f"[bench] OpenAI-compatible stand-in on http://{args.host}:{args.port}/v1 "
          f"({where}) model={args.model} -- Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[bench] stopped")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
