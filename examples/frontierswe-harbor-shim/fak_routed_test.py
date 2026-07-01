#!/usr/bin/env python3
"""Self-contained gate for the FrontierSWE fak-routing shim (no fak, no harbor_ext, no GPU).

This is the runnable witness for examples/frontierswe-harbor-shim (epic #1706, C6): it
proves the shim's one job — reroute the wrapped agent's model base_url to a fak gateway and
change NOTHING else — against a stub harbor_ext agent and a mock OpenAI-compatible endpoint
that records it received the traffic. Run: python3 fak_routed_test.py  (exit 0 = pass).

Everything here is Python stdlib, so `make ci`'s Go driver can shell out to it wherever
python3 exists, and it stays runnable by hand anywhere else.
"""
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import fak_routed


# --- a stub that mimics a harbor_ext CLI-harness agent -------------------------------------
# It carries the same surface the shim cares about: model_name + override_timeout_sec taken
# positionally (as FrontierSWE instantiates agents), a vendor base_url attribute, and the
# job.yaml `kwargs` splatted in. Its run() POSTs to {base_url}/chat/completions so a test can
# prove where the traffic actually went.
VENDOR_BASE_URL = "https://api.vendor.example/v1"


class StubHarnessAgent:
    def __init__(self, model_name, override_timeout_sec=None, base_url=VENDOR_BASE_URL, **kwargs):
        self.model_name = model_name
        self.override_timeout_sec = override_timeout_sec
        self.base_url = base_url
        self.kwargs = kwargs

    def run(self, prompt):
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps({"model": self.model_name, "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())


# --- a mock OpenAI-compatible endpoint that records the calls it received ------------------
class _Recorder:
    def __init__(self):
        self.paths = []


def _make_server(recorder):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            recorder.paths.append(self.path)
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):  # silence the default stderr access log
            pass

    return HTTPServer(("127.0.0.1", 0), Handler)


def _fail(label, msg):
    print(f"FAIL {label}: {msg}")
    return False


def test_routes_traffic_and_changes_only_base_url():
    """The shim overrides the wrapped agent's base_url to the gateway, leaves every other
    field identical to an un-routed instance, and its traffic lands on the gateway."""
    recorder = _Recorder()
    server = _make_server(recorder)
    port = server.server_address[1]
    gateway = f"http://127.0.0.1:{port}/v1"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    ok = True
    try:
        # A raw instance is the reference for "what did NOT change".
        raw = StubHarnessAgent("anthropic/claude-opus-4-6", 72000, effort_level="max")

        agent = fak_routed.FakRoutedAgent(
            "anthropic/claude-opus-4-6",
            72000,
            wrapped=StubHarnessAgent,
            fak_base_url=gateway,
            allow_internet=False,
            effort_level="max",
        )

        # The one intended delta: base_url now points at the gateway, not the vendor.
        if agent.wrapped.base_url != gateway:
            ok &= _fail("override", f"base_url={agent.wrapped.base_url!r}, want {gateway!r}")
        if raw.base_url == gateway:
            ok &= _fail("reference", "raw agent already pointed at the gateway (bad fixture)")

        # Everything else is byte-identical to the un-routed agent — the parity invariant.
        for field in ("model_name", "override_timeout_sec", "kwargs"):
            if getattr(agent.wrapped, field) != getattr(raw, field):
                ok &= _fail("only-delta", f"{field} changed: {getattr(agent.wrapped, field)!r} != {getattr(raw, field)!r}")

        # The base-url ENV seams are pointed at the gateway for the child harness process.
        import os
        for key in fak_routed._BASE_URL_ENV_KEYS:
            if os.environ.get(key) != gateway:
                ok &= _fail("env", f"{key}={os.environ.get(key)!r}, want {gateway!r}")

        # Traffic actually goes to the gateway (proves the reroute is live, not cosmetic).
        agent.run("hello")
        if recorder.paths != ["/v1/chat/completions"]:
            ok &= _fail("traffic", f"gateway saw {recorder.paths!r}, want ['/v1/chat/completions']")

        agent.restore_env()
        for key in fak_routed._BASE_URL_ENV_KEYS:
            if os.environ.get(key) == gateway:
                ok &= _fail("restore", f"{key} still pinned to the gateway after restore_env()")
    finally:
        server.shutdown()
    return ok


def test_import_path_string_resolves():
    """`wrapped` accepts FrontierSWE's 'module:Class' import_path form, not just a class."""
    agent = fak_routed.FakRoutedAgent(
        "m",
        1,
        wrapped="fak_routed_test:StubHarnessAgent",
        fak_base_url="http://127.0.0.1:8080/v1",
    )
    # Assert by class name, not identity: run as __main__, importlib loads fak_routed_test
    # as a second module, so the resolved class is a distinct-but-equivalent object.
    if type(agent.wrapped).__name__ != "StubHarnessAgent":
        return _fail("import_path", f"resolved to {type(agent.wrapped)!r}, want StubHarnessAgent")
    agent.restore_env()
    return True


def test_default_gateway_is_loopback():
    """With no fak_base_url, the shim falls back to the in-sandbox loopback default."""
    agent = fak_routed.FakRoutedAgent("m", 1, wrapped=StubHarnessAgent)
    ok = fak_routed._is_in_sandbox(agent.fak_base_url) or _fail(
        "default", f"default gateway {agent.fak_base_url!r} is not in-sandbox"
    )
    agent.restore_env()
    return ok


def test_no_internet_rejects_external_gateway():
    """allow_internet=false must refuse an external gateway URL (no off-sandbox leak)."""
    try:
        fak_routed.FakRoutedAgent(
            "m", 1, wrapped=StubHarnessAgent, fak_base_url="https://gateway.example.com/v1", allow_internet=False
        )
    except ValueError:
        return True
    return _fail("no-internet", "external gateway under allow_internet=false did NOT raise")


def test_internet_allows_external_gateway():
    """When allow_internet=true, an external gateway is permitted (the constraint is lifted)."""
    agent = fak_routed.FakRoutedAgent(
        "m", 1, wrapped=StubHarnessAgent, fak_base_url="https://gateway.example.com/v1", allow_internet=True
    )
    ok = agent.wrapped.base_url == "https://gateway.example.com/v1" or _fail(
        "internet", f"base_url={agent.wrapped.base_url!r}"
    )
    agent.restore_env()
    return ok


def main():
    tests = [
        test_routes_traffic_and_changes_only_base_url,
        test_import_path_string_resolves,
        test_default_gateway_is_loopback,
        test_no_internet_rejects_external_gateway,
        test_internet_allows_external_gateway,
    ]
    ok = True
    for t in tests:
        result = t()
        ok &= result
        print(f"{'ok  ' if result else 'FAIL'} {t.__name__}")
    print("ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
