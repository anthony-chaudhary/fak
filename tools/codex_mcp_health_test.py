#!/usr/bin/env python3
"""Tests for codex_mcp_health -- the Codex<->fak stdio MCP transport diagnostic (GH #1445).

Drives every classification branch against fixtures (no process spawn): a
"server OK + transport dead" case (the #1445 amplifier), a "server dead" case, a
"stale children present" case, and the all-clear. Plus the pure parsers: the smoke
output classifier (against the REAL fak_read response shape) and the PowerShell
Win32_Process inventory parser. The reap path is asserted to act ONLY on the
caller-supplied PID list, never on the inventory.

Run: `python tools/codex_mcp_health_test.py`  (exit 0 = all pass),
or `python -m pytest tools/codex_mcp_health_test.py -q`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codex_mcp_health as h  # noqa: E402


# A real fak_read VERSION smoke response, exactly as the server emits it (id=2 line).
# Captured shape: content[0].text is a JSON string with verdict.kind / result.status /
# result.meta.engine and a nested result.content JSON carrying the file body.
def _smoke_stdout(verdict="ALLOW", status="OK", engine="fakread", body="0.34.0\n") -> str:
    inner = {
        "verdict": {"kind": verdict, "by": "monitor"},
        "result": {
            "status": status,
            "content": json.dumps({"content": body, "file_path": "VERSION"}),
            "meta": {"engine": engine},
        },
        "trace_id": "gw-2",
    }
    init_line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"serverInfo": {"name": "fak-gateway", "version": "0.34.0"}},
        }
    )
    call_line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"content": [{"type": "text", "text": json.dumps(inner)}], "isError": False},
        }
    )
    return init_line + "\n" + call_line + "\n"


def check(name: str, cond: bool) -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return cond


# --- smoke output parsing -------------------------------------------------------


def test_parse_smoke_ok() -> bool:
    res = h.parse_smoke_output(_smoke_stdout())
    return (
        check("smoke ok=True on a clean ALLOW/OK/fakread/0.34.0 response", res.ok)
        and check("smoke verdict==ALLOW", res.verdict == "ALLOW")
        and check("smoke status==OK", res.status == "OK")
        and check("smoke engine==fakread", res.engine == "fakread")
        and check("smoke content carries 0.34.0", "0.34.0" in res.content)
    )


def test_parse_smoke_wrong_verdict() -> bool:
    res = h.parse_smoke_output(_smoke_stdout(verdict="DENY"))
    return check("smoke ok=False when verdict != ALLOW", not res.ok) and check(
        "smoke mismatch names the verdict", "DENY" in res.reason
    )


def test_parse_smoke_jsonrpc_error() -> bool:
    err = json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -32602, "message": "boom"}})
    res = h.parse_smoke_output(err + "\n")
    return check("smoke ok=False on a JSON-RPC error response", not res.ok) and check(
        "error reason mentions JSON-RPC error", "JSON-RPC error" in res.reason
    )


def test_parse_smoke_no_call_response() -> bool:
    # Only the initialize line, no id=2 tools/call response (a torn-down session).
    only_init = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})
    res = h.parse_smoke_output(only_init + "\n")
    return check("smoke ok=False with no id=2 response", not res.ok) and check(
        "reason names the missing tools/call response", "id=2" in res.reason
    )


# --- PowerShell inventory parsing ----------------------------------------------


def test_inventory_parse_list() -> bool:
    payload = json.dumps(
        [
            {"ProcessId": 111, "CommandLine": r"C:\work\fak\fak.exe serve --stdio --policy x.json"},
            {"ProcessId": 222, "CommandLine": r"C:\work\fak\fak.exe guard --split=auto"},
            {"ProcessId": 333, "CommandLine": r"C:\work\fak\fak.exe serve --stdio"},
        ]
    )
    rows = h.parse_powershell_inventory(payload)
    pids = sorted(c.pid for c in rows)
    return (
        check("inventory keeps only serve --stdio rows (drops guard)", pids == [111, 333])
        and check("inventory captures the command line", "serve --stdio" in rows[0].command)
    )


def test_inventory_parse_single_object() -> bool:
    # ConvertTo-Json emits a bare object (not an array) for a single match.
    one = json.dumps({"ProcessId": 999, "CommandLine": "fak.exe serve --stdio --policy p"})
    rows = h.parse_powershell_inventory(one)
    return check("inventory accepts a single bare object", len(rows) == 1 and rows[0].pid == 999)


def test_inventory_parse_empty() -> bool:
    return check("inventory empty string -> no rows", h.parse_powershell_inventory("") == [])


# --- classification: the four closed states ------------------------------------


def test_classify_transport_dead_server_ok() -> bool:
    # The #1445 case: server smoke passes but the in-session transport is dead.
    smoke = h.parse_smoke_output(_smoke_stdout())
    diag = h.classify(smoke, transport_alive=False, stale_children=[])
    return (
        check("server OK + transport dead -> TRANSPORT_DEAD_SERVER_OK",
              diag.state == h.TRANSPORT_DEAD_SERVER_OK)
        and check("state is in the closed set", diag.state in h.DIAGNOSTIC_STATES)
        and check("next_step says reconnect / do not retry", "reconnect" in diag.next_step.lower())
    )


def test_classify_server_dead() -> bool:
    bad = h.SmokeResult(ok=False, reason="smoke timed out")
    # Even with the transport alive, a dead server dominates.
    diag = h.classify(bad, transport_alive=True, stale_children=[])
    return (
        check("failed smoke -> SERVER_DEAD", diag.state == h.SERVER_DEAD)
        and check("server_ok False", diag.server_ok is False)
        and check("next_step says fix the server", "fix the server" in diag.next_step.lower())
    )


def test_classify_stale_children() -> bool:
    smoke = h.parse_smoke_output(_smoke_stdout())
    kids = [h.ChildProc(pid=111, command="fak.exe serve --stdio")]
    diag = h.classify(smoke, transport_alive=True, stale_children=kids)
    return (
        check("server OK + transport alive + strays -> STALE_CHILDREN",
              diag.state == h.STALE_CHILDREN)
        and check("the stray child rides along", diag.stale_children[0].pid == 111)
        and check("next_step says reap explicitly", "--reap" in diag.next_step)
    )


def test_classify_reconnect_ok() -> bool:
    smoke = h.parse_smoke_output(_smoke_stdout())
    diag = h.classify(smoke, transport_alive=True, stale_children=[])
    return (
        check("server OK + transport alive + no strays -> RECONNECT_OK",
              diag.state == h.RECONNECT_OK)
        and check("next_step says no action", "no action" in diag.next_step.lower())
    )


def test_server_dead_dominates_stale_children() -> bool:
    # A dead server with strays present must still surface SERVER_DEAD (precedence).
    bad = h.SmokeResult(ok=False, reason="spawn failed")
    kids = [h.ChildProc(pid=42, command="fak.exe serve --stdio")]
    diag = h.classify(bad, transport_alive=False, stale_children=kids)
    return check("dead server dominates even with strays", diag.state == h.SERVER_DEAD)


# --- reap acts only on the explicit list ---------------------------------------


def test_reap_only_explicit_list(monkey=None) -> bool:
    # reap_children must act solely on the PIDs passed -- never derive from inventory.
    seen: list[int] = []

    def fake_run(cmd, **kwargs):
        # cmd is taskkill /PID <pid> /F on win32; record the pid.
        for i, tok in enumerate(cmd):
            if tok == "/PID":
                seen.append(int(cmd[i + 1]))

        class R:
            returncode = 0
            stdout = "SUCCESS"
            stderr = ""

        return R()

    orig_run = h.subprocess.run
    orig_kill = h.os.kill
    try:
        h.subprocess.run = fake_run  # type: ignore[assignment]
        h.os.kill = lambda pid, sig: seen.append(pid)  # type: ignore[assignment]
        results = h.reap_children([5, 7])
    finally:
        h.subprocess.run = orig_run  # type: ignore[assignment]
        h.os.kill = orig_kill  # type: ignore[assignment]
    return (
        check("reap touched exactly the passed PIDs", sorted(seen) == [5, 7])
        and check("reap returns one row per pid", len(results) == 2)
    )


def test_reap_empty_list_is_noop() -> bool:
    return check("reap([]) is a no-op", h.reap_children([]) == [])


def main() -> int:
    tests = [
        test_parse_smoke_ok,
        test_parse_smoke_wrong_verdict,
        test_parse_smoke_jsonrpc_error,
        test_parse_smoke_no_call_response,
        test_inventory_parse_list,
        test_inventory_parse_single_object,
        test_inventory_parse_empty,
        test_classify_transport_dead_server_ok,
        test_classify_server_dead,
        test_classify_stale_children,
        test_classify_reconnect_ok,
        test_server_dead_dominates_stale_children,
        test_reap_only_explicit_list,
        test_reap_empty_list_is_noop,
    ]
    ok = True
    for t in tests:
        print(t.__name__)
        ok = t() and ok
    print("\n" + ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
