#!/usr/bin/env python3
r"""codex_mcp_health -- a transport health diagnostic for the Codex <-> fak stdio MCP seam.

Background (GH #1445): after the ``fak_read`` policy/engine fix in ``805b0cc9`` the
configured ``fak.exe`` answers a raw MCP stdio smoke cleanly --

    fak.exe serve --stdio --policy examples/dev-agent-policy.json
    -> fak_read on VERSION returns verdict=ALLOW, status=OK, engine=fakread, content=0.34.0

-- yet the *live* Codex MCP tool path could remain stuck reporting ``Transport closed``.
Once a session is in that state, repeated ``mcp__fak`` calls keep failing without ever
re-exercising the (now healthy) server: a spam amplifier. Stale ``fak serve --stdio``
children can also be left behind by the repeated failed spawns.

This tool separates the two operands a stuck session conflates -- *is the configured
server healthy?* (a fresh out-of-session spawn we drive ourselves) and *is the current
in-session transport alive?* (a fact only the caller knows) -- and folds them, plus a
stray-child inventory, into ONE actionable diagnostic from a closed state set so the
agent can suppress unchanged retries instead of spam-failing:

    TRANSPORT_DEAD_SERVER_OK  the configured server answers a fresh smoke, but the
                              in-session transport is dead -> reconnect (respawn the
                              MCP client); do NOT keep retrying the dead transport.
    SERVER_DEAD               a fresh smoke against the configured server FAILS ->
                              the server itself is broken; fix it before reconnecting.
    RECONNECT_OK              server answers AND the in-session transport is alive ->
                              nothing to do; retries are expected to succeed.
    STALE_CHILDREN            stray ``fak serve --stdio`` children were inventoried and
                              offered for an explicit reap (advisory; never force-killed
                              blindly).

The smoke speaks the MCP-over-stdio wire protocol directly (newline-delimited
JSON-RPC: ``initialize`` -> ``notifications/initialized`` -> ``tools/call`` fak_read),
the SAME framing ``internal/gateway/mcp.go`` serves, so it exercises the real server.

The Windows process inventory uses the PowerShell ``Get-CimInstance Win32_Process``
evidence command from the issue; it is gated on ``sys.platform == 'win32'`` and degrades
to an empty inventory (with a noted reason) elsewhere. The reap is never automatic:
``inventory_stale_children`` only reports; ``reap_children`` acts solely on an explicit
list of PIDs the caller passes.

CLI:
    python tools/codex_mcp_health.py                       # smoke + inventory, table
    python tools/codex_mcp_health.py --json                # machine diagnostic
    python tools/codex_mcp_health.py --transport-dead      # assert in-session transport dead
    python tools/codex_mcp_health.py --reap PID [PID ...]  # explicit reap of named children
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- Closed diagnostic state set ------------------------------------------------

TRANSPORT_DEAD_SERVER_OK = "TRANSPORT_DEAD_SERVER_OK"
SERVER_DEAD = "SERVER_DEAD"
RECONNECT_OK = "RECONNECT_OK"
STALE_CHILDREN = "STALE_CHILDREN"

DIAGNOSTIC_STATES = (
    TRANSPORT_DEAD_SERVER_OK,
    SERVER_DEAD,
    RECONNECT_OK,
    STALE_CHILDREN,
)

# One actionable next step per state, so the agent knows what changed and what to do.
NEXT_STEP = {
    TRANSPORT_DEAD_SERVER_OK: (
        "Configured fak stdio server is healthy but the in-session Codex transport is "
        "dead. Reconnect / respawn the MCP client; do NOT retry the dead transport."
    ),
    SERVER_DEAD: (
        "The configured fak stdio server failed its own smoke. Fix the server "
        "(policy/binary/version) before reconnecting; retrying the transport is futile."
    ),
    RECONNECT_OK: (
        "Server answers and the in-session transport is alive. No action; mcp__fak "
        "calls should succeed."
    ),
    STALE_CHILDREN: (
        "Stray 'fak serve --stdio' children are present. Review the inventory and reap "
        "explicitly with --reap <pid ...>; do not blind-kill."
    ),
}

DEFAULT_POLICY = "examples/dev-agent-policy.json"
EXPECTED_VERSION = "0.34.0"


# --- Smoke: drive the configured server over the MCP stdio wire protocol --------


def repo_root() -> Path:
    """Best-effort repo root (the dir holding VERSION / examples), from this file."""
    here = Path(__file__).resolve().parent
    for cand in (here.parent, here):
        if (cand / "VERSION").exists() and (cand / "examples").is_dir():
            return cand
    return here.parent


def fak_binary(root: Path) -> str:
    """Path to the fak binary to smoke (fak.exe on Windows, fak elsewhere)."""
    name = "fak.exe" if sys.platform == "win32" else "fak"
    return str(root / name)


def smoke_frames() -> bytes:
    """The three newline-delimited JSON-RPC frames of a fak_read VERSION smoke."""
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"},
    }
    inited = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    call = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "fak_read", "arguments": {"file_path": "VERSION"}},
    }
    lines = [json.dumps(init), json.dumps(inited), json.dumps(call)]
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass
class SmokeResult:
    """The outcome of a fresh out-of-session smoke against the configured server."""

    ok: bool
    verdict: str = ""
    status: str = ""
    engine: str = ""
    content: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "verdict": self.verdict,
            "status": self.status,
            "engine": self.engine,
            "content": self.content,
            "reason": self.reason,
        }


def parse_smoke_output(stdout: str) -> SmokeResult:
    """Classify the server's stdout (one JSON-RPC response per line) into a SmokeResult.

    The fak_read tool result is the response with id==2; its content[0].text is a JSON
    string carrying verdict.kind / result.status / result.meta.engine and a nested
    result.content JSON whose own .content field is the file body. A pure function so a
    fixture can drive every classification branch without spawning a process.
    """
    call_resp = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("id") == 2:
            call_resp = obj
    if call_resp is None:
        return SmokeResult(ok=False, reason="no tools/call response (id=2) in server output")
    if "error" in call_resp:
        err = call_resp["error"]
        return SmokeResult(ok=False, reason=f"server returned JSON-RPC error: {err}")
    result = call_resp.get("result", {})
    if result.get("isError"):
        return SmokeResult(ok=False, reason="tool result isError=true")
    content_list = result.get("content") or []
    if not content_list:
        return SmokeResult(ok=False, reason="tool result has empty content")
    text = content_list[0].get("text", "")
    try:
        inner = json.loads(text)
    except json.JSONDecodeError:
        return SmokeResult(ok=False, reason="tool result text is not JSON")
    verdict = (inner.get("verdict") or {}).get("kind", "")
    res = inner.get("result") or {}
    status = res.get("status", "")
    engine = (res.get("meta") or {}).get("engine", "")
    # result.content is itself a JSON string {"content": "...","file_path": "..."}.
    file_body = ""
    raw_content = res.get("content", "")
    try:
        file_body = (json.loads(raw_content) or {}).get("content", "")
    except (json.JSONDecodeError, TypeError):
        file_body = raw_content
    ok = (
        verdict == "ALLOW"
        and status == "OK"
        and engine == "fakread"
        and EXPECTED_VERSION in (file_body or "")
    )
    reason = "" if ok else (
        f"smoke mismatch: verdict={verdict!r} status={status!r} engine={engine!r} "
        f"content={file_body!r} (want ALLOW/OK/fakread/{EXPECTED_VERSION})"
    )
    return SmokeResult(
        ok=ok,
        verdict=verdict,
        status=status,
        engine=engine,
        content=(file_body or "").strip(),
        reason=reason,
    )


def run_server_smoke(
    root: Path | None = None,
    policy: str = DEFAULT_POLICY,
    timeout: float = 30.0,
) -> SmokeResult:
    """Spawn a FRESH `fak serve --stdio` and drive the fak_read VERSION smoke.

    A spawn/timeout failure is itself a SmokeResult(ok=False) with the transport-level
    reason -- a server we cannot even run is a dead server, not an exception to leak.
    """
    root = root or repo_root()
    binary = fak_binary(root)
    if not Path(binary).exists():
        return SmokeResult(ok=False, reason=f"fak binary not found at {binary}")
    cmd = [binary, "serve", "--stdio", "--policy", policy]
    try:
        proc = subprocess.run(
            cmd,
            input=smoke_frames(),
            capture_output=True,
            cwd=str(root),
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return SmokeResult(ok=False, reason=f"spawn failed: {exc}")
    except subprocess.TimeoutExpired:
        return SmokeResult(ok=False, reason=f"smoke timed out after {timeout}s")
    out = proc.stdout.decode("utf-8", errors="replace")
    res = parse_smoke_output(out)
    if not res.ok and not res.reason:
        res.reason = f"server exited rc={proc.returncode} with no parsable response"
    return res


# --- Stray-child inventory (Windows-gated) --------------------------------------


@dataclass
class ChildProc:
    pid: int
    command: str

    def to_dict(self) -> dict:
        return {"pid": self.pid, "command": self.command}


def parse_powershell_inventory(stdout: str) -> list[ChildProc]:
    """Parse the JSON emitted by the Win32_Process query into ChildProc rows.

    Accepts either a single object or a list (PowerShell ConvertTo-Json emits a bare
    object for one match, an array for many). Only rows whose CommandLine names a
    `serve --stdio` are kept -- the stray-MCP-child signature.
    """
    stdout = stdout.strip()
    if not stdout:
        return []
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    out: list[ChildProc] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        cmd = row.get("CommandLine") or row.get("commandline") or ""
        pid = row.get("ProcessId", row.get("processid"))
        if pid is None:
            continue
        if "serve" in cmd and "--stdio" in cmd:
            out.append(ChildProc(pid=int(pid), command=cmd))
    return out


def inventory_stale_children(root: Path | None = None) -> tuple[list[ChildProc], str]:
    """Inventory stray `fak serve --stdio` children. Reports only -- never kills.

    Returns (children, note). On non-Windows the inventory is empty with a note naming
    the gated platform; the diagnostic still folds the smoke result. The PowerShell
    evidence command is the one from GH #1445.
    """
    root = root or repo_root()
    if sys.platform != "win32":
        return [], f"process inventory gated to win32 (platform={sys.platform})"
    fak_path = fak_binary(root)
    # Emit ProcessId + CommandLine as JSON for every fak.exe; filter to serve --stdio.
    ps = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.ExecutablePath -eq '{fak_path}' }} | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return [], f"inventory probe failed: {exc}"
    children = parse_powershell_inventory(proc.stdout)
    note = "" if children else "no stray fak serve --stdio children found"
    return children, note


def reap_children(pids: list[int]) -> list[dict]:
    """Explicitly reap the named child PIDs. Acts ONLY on the caller-supplied list.

    Never derives its own kill list from the inventory -- the inventory reports, a human
    or agent decides, this executes that decision. Returns one result row per PID.
    """
    results: list[dict] = []
    for pid in pids:
        try:
            if sys.platform == "win32":
                proc = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    text=True,
                    timeout=15.0,
                )
                ok = proc.returncode == 0
                detail = (proc.stdout or proc.stderr).strip()
            else:
                os.kill(pid, 15)
                ok, detail = True, "SIGTERM sent"
        except (OSError, subprocess.SubprocessError) as exc:
            ok, detail = False, str(exc)
        results.append({"pid": pid, "reaped": ok, "detail": detail})
    return results


# --- Fold: smoke + transport + inventory -> one diagnostic ----------------------


@dataclass
class Diagnostic:
    state: str
    next_step: str
    server_ok: bool
    transport_alive: bool
    smoke: SmokeResult
    stale_children: list[ChildProc] = field(default_factory=list)
    inventory_note: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "next_step": self.next_step,
            "server_ok": self.server_ok,
            "transport_alive": self.transport_alive,
            "smoke": self.smoke.to_dict(),
            "stale_children": [c.to_dict() for c in self.stale_children],
            "inventory_note": self.inventory_note,
        }


def classify(
    smoke: SmokeResult,
    transport_alive: bool,
    stale_children: list[ChildProc],
    inventory_note: str = "",
) -> Diagnostic:
    """Fold the three operands into one closed-set diagnostic.

    Precedence: a dead server dominates (SERVER_DEAD) -- nothing downstream can succeed
    until the server is fixed, so even stray children wait. With a healthy server: a dead
    in-session transport is the #1445 case (TRANSPORT_DEAD_SERVER_OK); else a present
    stray-child set is the next signal (STALE_CHILDREN); else all-clear (RECONNECT_OK).
    """
    if not smoke.ok:
        state = SERVER_DEAD
    elif not transport_alive:
        state = TRANSPORT_DEAD_SERVER_OK
    elif stale_children:
        state = STALE_CHILDREN
    else:
        state = RECONNECT_OK
    return Diagnostic(
        state=state,
        next_step=NEXT_STEP[state],
        server_ok=smoke.ok,
        transport_alive=transport_alive,
        smoke=smoke,
        stale_children=stale_children,
        inventory_note=inventory_note,
    )


def diagnose(
    root: Path | None = None,
    policy: str = DEFAULT_POLICY,
    transport_alive: bool = True,
    timeout: float = 30.0,
) -> Diagnostic:
    """End-to-end: smoke the configured server, inventory strays, fold a diagnostic."""
    root = root or repo_root()
    smoke = run_server_smoke(root=root, policy=policy, timeout=timeout)
    children, note = inventory_stale_children(root=root)
    return classify(smoke, transport_alive, children, note)


# --- CLI ------------------------------------------------------------------------


def render_table(diag: Diagnostic) -> str:
    lines = [
        f"diagnostic: {diag.state}",
        f"  server_ok        : {diag.server_ok}",
        f"  transport_alive  : {diag.transport_alive}",
        f"  smoke            : verdict={diag.smoke.verdict or '-'} "
        f"status={diag.smoke.status or '-'} engine={diag.smoke.engine or '-'} "
        f"content={diag.smoke.content or '-'}",
    ]
    if diag.smoke.reason:
        lines.append(f"  smoke_reason     : {diag.smoke.reason}")
    if diag.stale_children:
        lines.append(f"  stale_children   : {len(diag.stale_children)}")
        for c in diag.stale_children:
            lines.append(f"    - pid={c.pid} {c.command}")
    elif diag.inventory_note:
        lines.append(f"  stale_children   : 0 ({diag.inventory_note})")
    lines.append(f"  next_step        : {diag.next_step}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="policy file for the smoke")
    ap.add_argument("--json", action="store_true", help="emit the machine diagnostic")
    ap.add_argument(
        "--transport-dead",
        action="store_true",
        help="assert the in-session Codex transport is dead (the #1445 case)",
    )
    ap.add_argument("--timeout", type=float, default=30.0, help="smoke timeout (s)")
    ap.add_argument(
        "--reap",
        nargs="+",
        type=int,
        metavar="PID",
        help="explicitly reap the named child PIDs (no inventory-derived kill)",
    )
    args = ap.parse_args(argv)

    if args.reap:
        results = reap_children(args.reap)
        if args.json:
            print(json.dumps({"reaped": results}, indent=2))
        else:
            for r in results:
                print(f"reap pid={r['pid']}: {'OK' if r['reaped'] else 'FAIL'} {r['detail']}")
        return 0 if all(r["reaped"] for r in results) else 1

    diag = diagnose(
        policy=args.policy,
        transport_alive=not args.transport_dead,
        timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(diag.to_dict(), indent=2))
    else:
        print(render_table(diag))
    # Exit non-zero on a state that needs the agent to STOP retrying as-is.
    return 0 if diag.state == RECONNECT_OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
