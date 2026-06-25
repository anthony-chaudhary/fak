#!/usr/bin/env python3
"""
Runnable OpenAI Agents SDK guardrail-mapping demo for fak.

It starts `fak serve` offline, calls /v1/fak/adjudicate before a tool, calls
/v1/fak/admit after a tool, and prints the behavior an Agents SDK tool guardrail
would return. No OpenAI key, model, network, or Agents SDK install is required.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from guarded_tool import (
    FakGuardrailClient,
    emit_agents_trace_if_available,
    guarded_call,
    output_guardrail_decision,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICY = os.path.join("examples", "dev-agent-policy.json")
BOOT_TIMEOUT = 30
FAKE_SECRET = "sk-liveAA00bb11cc22dd33ee44ff55"


def find_fak(explicit: str | None) -> list[str]:
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
        print(f"  building {exe} (one-time)")
        result = subprocess.run(["go", "build", "-o", out, "./cmd/fak"], cwd=REPO_ROOT)
        if result.returncode == 0 and os.path.isfile(out):
            return [out]
        return ["go", "run", "./cmd/fak"]
    raise SystemExit("fak not found and no Go toolchain to build it; pass --fak PATH")


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_healthy(base: str) -> dict | None:
    deadline = time.time() + BOOT_TIMEOUT
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/healthz", timeout=2) as resp:
                health = json.load(resp)
                if health.get("ok"):
                    return health
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.2)
    return None


def fake_git_status() -> dict[str, str]:
    return {"text": "On branch main\nnothing to commit, working tree clean"}


def should_not_run() -> dict[str, str]:
    raise AssertionError("denied tool executed")


def print_decision(label: str, decision, *, executed: bool | None = None) -> None:
    verdict = decision.verdict or {}
    suffix = "" if executed is None else f" executed={str(executed).lower()}"
    print(
        f"  {label}: behavior={decision.behavior} "
        f"verdict={verdict.get('kind')} reason={verdict.get('reason', '')}{suffix}"
    )
    print("    trace_metadata=" + json.dumps(decision.trace_metadata(), sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="fak + OpenAI Agents SDK guardrail adapter demo")
    parser.add_argument("--fak", help="path to the fak binary")
    parser.add_argument("--kernel", help="use an already-running fak base URL instead of starting one")
    parser.add_argument("--emit-agents-trace", action="store_true",
                        help="emit a best-effort Agents SDK custom span if the SDK is installed")
    args = parser.parse_args()

    proc: subprocess.Popen | None = None
    base = args.kernel
    if base is None:
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        fak = find_fak(args.fak)
        proc = subprocess.Popen(
            fak + ["serve", "--addr", f"127.0.0.1:{port}", "--policy", POLICY],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        health = wait_healthy(base)
        if not health:
            out = ""
            try:
                proc.terminate()
                out = proc.communicate(timeout=5)[0] or ""
            except Exception:
                proc.kill()
            print(f"server did not become healthy in {BOOT_TIMEOUT}s\n{out[-800:]}")
            return 1
    else:
        health = wait_healthy(base)
        if not health:
            print(f"could not reach fak at {base}")
            return 1

    client = FakGuardrailClient(base)
    failures: list[str] = []

    print("fak + OpenAI Agents SDK guardrail adapter demo")
    print(f"  kernel={base} policy={POLICY}")

    try:
        denied = guarded_call(client, "git_push", {}, should_not_run, trace_id="agents-deny-1")
        deny_decision = denied["input"]
        print_decision("input guardrail blocks git_push", deny_decision, executed=denied["tool_executed"])
        if deny_decision.behavior != "reject_content" or denied["tool_executed"]:
            failures.append("git_push should be rejected before tool execution")

        allowed = guarded_call(
            client,
            "git_status",
            {},
            lambda: fake_git_status(),
            trace_id="agents-allow-1",
            read_only=True,
            witness="git:HEAD",
        )
        input_decision = allowed["input"]
        output_decision = allowed["output"]
        print_decision("input guardrail allows git_status", input_decision, executed=allowed["tool_executed"])
        print_decision("output guardrail admits git_status result", output_decision)
        if input_decision.behavior != "allow" or output_decision.behavior != "allow":
            failures.append("git_status should run and its clean result should be admitted")

        poisoned = client.admit(
            "web_fetch",
            {"page": f"config loaded. api_key={FAKE_SECRET} was found in env"},
            trace_id="agents-quarantine-1",
        )
        poisoned_decision = output_guardrail_decision(poisoned)
        print_decision("output guardrail quarantines web_fetch result", poisoned_decision)
        if poisoned_decision.behavior != "reject_content":
            failures.append("poisoned web_fetch result should map to reject_content")

        emitted = False
        if args.emit_agents_trace:
            emitted = emit_agents_trace_if_available(
                "fak.guardrail",
                poisoned_decision.trace_metadata(),
            )
        print(f"  agents_sdk_custom_span_emitted={str(emitted).lower()}")

        if failures:
            print("summary: FAIL - " + "; ".join(failures))
            return 1
        print("summary: PASS - denied call did not run, clean call ran, poisoned result was quarantined")
        return 0
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.communicate(timeout=5)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
