#!/usr/bin/env python3
"""Dogfood recently-added local features and leave repeatable evidence.

This is a small, deterministic dogfood runner for features that just landed in
this repo and otherwise tend to rot as isolated commands:

  * `fak loop append|status` and the loopmgr hash-chained ledger
  * `fak vcache score|prove-recall` and the vCache score/refutation surfaces
  * `fak benchmarks list|run vcache` and the benchmark catalog registry
  * `internal/callavoid` avoided-call economics tests
  * `internal/promptmmu` tool-pruning tests
  * `tools/code_slop_scorecard.py` as a real scorecard consumer
  * `tools/dogfood_coverage.py` as the live dogfood scorecard

It writes its evidence under `.fak/recent-feature-dogfood/<stamp>/`, which is
ignored local run state. Wrap it with `fak loop run` when scheduling:

  go run ./cmd/fak loop run --loop recent-feature-dogfood/manual -- \
    python tools/recent_feature_dogfood.py

The runner is intentionally local and no-network by default. Scorecards may
report ACTION/debt and still pass this dogfood run as long as their machine
payload is well-formed. This checks that the feature can be used repeatably; it
does not require the repo's debt to already be zero.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fak.recent-feature-dogfood.v1"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fak_cmd(root: Path) -> list[str]:
    configured = os.environ.get("FAK_BIN")
    if configured:
        return split_configured_command(configured)
    for rel in ("fak.exe", "fak", "tools/.bin/fak"):
        p = root / rel
        if p.exists() and supports_recent_fak([str(p)], root):
            return [str(p)]
    found = shutil.which("fak")
    if found and supports_recent_fak([found], root):
        return [found]
    return ["go", "run", "./cmd/fak"]


def split_configured_command(configured: str) -> list[str]:
    configured = configured.strip()
    if not configured:
        return []
    unquoted = strip_outer_quotes(configured)
    expanded = os.path.expandvars(os.path.expanduser(unquoted))
    for candidate in (configured, unquoted, expanded):
        if candidate and Path(candidate).exists():
            return [candidate]
    if not any(ch.isspace() for ch in configured):
        return [configured]
    tokens = shlex.split(configured, posix=(os.name != "nt"))
    if os.name == "nt":
        tokens = [strip_outer_quotes(t) for t in tokens]
    return tokens


def strip_outer_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def supports_recent_fak(cmd: list[str], root: Path) -> bool:
    """Return true only for a fak binary new enough for this dogfood packet."""
    probe_ledger = Path(tempfile.gettempdir()) / f"fak-loop-probe-{os.getpid()}.jsonl"
    probes = [
        (cmd + ["loop", "status", "--ledger", str(probe_ledger), "--json"], "fak.loop-status.v1"),
        (cmd + ["vcache", "score", "--json"], "fak.vcache.score.v1"),
    ]
    for argv, schema in probes:
        try:
            proc = subprocess.run(
                argv,
                cwd=str(root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if proc.returncode != 0:
            return False
        try:
            payload = json.loads(proc.stdout)
        except ValueError:
            return False
        if not isinstance(payload, dict) or payload.get("schema") != schema:
            return False
    try:
        proc = subprocess.run(
            cmd + ["benchmarks", "list", "--offline", "--json"],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        return False
    ok, _ = validate_payload("benchmarks_offline", payload)
    if not ok:
        return False
    return True


@dataclass(frozen=True)
class Probe:
    key: str
    description: str
    command: list[str]
    allowed_exits: tuple[int, ...] = (0,)
    json_source: str = ""  # "stdout" or an artifact path
    validator: str = ""
    required: bool = True


def build_suite(root: Path, out_dir: Path, *, include_go_tests: bool = True) -> list[Probe]:
    fak = fak_cmd(root)
    py = sys.executable or "python"
    ledger = out_dir / "loop-smoke.jsonl"
    vcache_score_artifact = out_dir / "vcache-score.json"
    # CLI JSON probe input for `fak callavoid account` (#799): an amplifying window the
    # CLI must grade B. Written here so the probe drives the operator CLI surface, not
    # just the package tests, via --in (no stdin plumbing in the runner).
    callavoid_input = out_dir / "callavoid-account-input.json"
    callavoid_input.write_text(
        json.dumps({"execute": 4, "memo_hit": 6}) + "\n", encoding="utf-8"
    )
    suite = [
        Probe(
            key="loop-append",
            description="append one canonical loop fire row",
            command=fak + [
                "loop", "append",
                "--ledger", str(ledger),
                "--loop", "recent-feature-dogfood/local",
                "--kind", "fire",
                "--source", "recent_feature_dogfood",
                "--summary", "recent feature dogfood smoke",
                "--json",
            ],
            json_source="stdout",
            validator="loop_event",
        ),
        Probe(
            key="loop-status",
            description="fold the dogfood loop ledger",
            command=fak + ["loop", "status", "--ledger", str(ledger), "--json"],
            json_source="stdout",
            validator="loop_status",
        ),
        Probe(
            key="vcache-score",
            description="run vCache 2x scorecard with dogfood telemetry (OBSERVED, not PLANNED)",
            command=fak + [
                "vcache", "score",
                "--telemetry", "experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl",
                "--json", "--out", str(vcache_score_artifact),
            ],
            json_source=str(vcache_score_artifact),
            validator="vcache_score",
        ),
        Probe(
            key="vcache-recall-refutation",
            description="prove the default M4 single-unit recall is refused",
            command=fak + ["vcache", "prove-recall", "--json"],
            allowed_exits=(1,),
            json_source="stdout",
            validator="vcache_recall_refuted",
        ),
        Probe(
            key="benchmarks-list-offline",
            description="list the offline benchmark catalog",
            command=fak + ["benchmarks", "list", "--offline", "--json"],
            json_source="stdout",
            validator="benchmarks_offline",
        ),
        Probe(
            key="benchmarks-run-vcache",
            description="run the benchmark catalog's vCache scorecard entry",
            command=fak + ["benchmarks", "run", "vcache"],
            json_source="stdout",
            validator="vcache_score",
        ),
        Probe(
            key="code-slop-scorecard",
            description="run the code-slop scorecard as a machine consumer",
            command=[py, "tools/code_slop_scorecard.py", "--json", "--no-toolchain"],
            allowed_exits=(0, 1),
            json_source="stdout",
            validator="code_slop_scorecard",
        ),
        Probe(
            key="dogfood-coverage-scorecard",
            description="run the live dogfood coverage scorecard",
            command=[py, "tools/dogfood_coverage.py", "--json"],
            json_source="stdout",
            validator="dogfood_coverage",
        ),
        Probe(
            key="callavoid-account-cli",
            description="run the avoided-call economics scorecard through the fak CLI (#799)",
            command=fak + ["callavoid", "account", "--in", str(callavoid_input), "--json"],
            json_source="stdout",
            validator="callavoid_account",
        ),
        Probe(
            key="cache-value-ledger",
            description="run cache-value ledger regression gate with 75.1% floor (#1114)",
            command=fak + ["nightrun", "score", "--floor", "0.751", "--json"],
            allowed_exits=(0,),
            json_source="stdout",
            validator="nightrun_score",
        ),
    ]
    if include_go_tests:
        suite.extend([
            Probe(
                key="go-test-loopmgr",
                description="unit-test the loop ledger package",
                command=["go", "test", "./internal/loopmgr"],
                validator="exit_only",
            ),
            Probe(
                key="go-test-promptmmu",
                description="unit-test the prompt tool-pruning spine",
                command=["go", "test", "./internal/promptmmu"],
                validator="exit_only",
            ),
            Probe(
                key="go-test-vcachescore",
                description="unit-test the vCache scorecard package",
                command=["go", "test", "./internal/vcachescore"],
                validator="exit_only",
            ),
            Probe(
                key="go-test-benchcatalog",
                description="unit-test the benchmark catalog registry",
                command=["go", "test", "./internal/benchcatalog"],
                validator="exit_only",
            ),
            Probe(
                key="go-test-callavoid",
                description="unit-test the avoided-call economics leaf",
                command=["go", "test", "./internal/callavoid"],
                validator="exit_only",
            ),
            Probe(
                key="go-test-fak-loop-vcache-benchmarks",
                description="unit-test the loop, vCache, and benchmark CLI surfaces",
                command=["go", "test", "./cmd/fak", "-run", "TestLoop|TestRunVCache|TestReadVCache|TestBenchmarks"],
                validator="exit_only",
            ),
        ])
    return suite


def run_completed(command: list[str], root: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def load_probe_json(probe: Probe, proc: subprocess.CompletedProcess[str], root: Path) -> tuple[Any | None, str]:
    if not probe.json_source:
        return None, ""
    if probe.json_source == "stdout":
        text = proc.stdout
    else:
        path = Path(probe.json_source)
        if not path.is_absolute():
            path = root / path
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return None, f"read json artifact: {exc}"
    try:
        return json.loads(text), ""
    except ValueError as exc:
        return None, f"decode json: {exc}"


def _nested_int(payload: Any, path: tuple[str, ...]) -> int | None:
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, int) and not isinstance(cur, bool):
        return cur
    return None


def validate_payload(name: str, payload: Any) -> tuple[bool, str]:
    if name == "exit_only":
        return True, "exit code accepted"
    if name == "loop_event":
        ok = isinstance(payload, dict) and payload.get("schema") == "fak.loop-event.v1" and payload.get("hash")
        return bool(ok), "loop event hash present" if ok else "missing loop event schema/hash"
    if name == "loop_status":
        loops = payload.get("loops") if isinstance(payload, dict) else None
        ok = payload.get("schema") == "fak.loop-status.v1" and isinstance(loops, list) and len(loops) >= 1
        return bool(ok), "loop status folded at least one loop" if ok else "missing loop status row"
    if name == "vcache_score":
        ok = (
            isinstance(payload, dict)
            and payload.get("schema") == "fak.vcache.score.v1"
            and payload.get("two_x_better") is True
            and payload.get("active_multiplier", 0) >= 2
            and payload.get("active_source") == "telemetry"
        )
        return bool(ok), "vCache score proves 2x-ready from OBSERVED dogfood telemetry" if ok else "vCache score did not prove 2x from telemetry"
    if name == "vcache_recall_refuted":
        ok = (
            isinstance(payload, dict)
            and str(payload.get("status")).lower() == "refuted"
            and payload.get("decision") == "cold_prefill"
            and int(payload.get("break_even_siblings") or 0) >= 301
        )
        return bool(ok), "default recall refutation proved" if ok else "recall refutation payload unexpected"
    if name == "benchmarks_offline":
        entries = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
        vcache = next((row for row in entries if row.get("Name") == "vcache" or row.get("name") == "vcache"), None)
        need = (vcache or {}).get("Need") or (vcache or {}).get("need")
        run = str((vcache or {}).get("Run") or (vcache or {}).get("run") or "")
        ok = vcache is not None and need == "offline" and "fak vcache bench --json" in run
        return (
            bool(ok),
            f"offline benchmark catalog includes vcache ({len(entries)} entries)"
            if ok else "offline benchmark catalog missing vcache scorecard entry",
        )
    if name == "code_slop_scorecard":
        debt = _nested_int(payload, ("corpus", "slop_debt"))
        ok = isinstance(payload, dict) and payload.get("schema") == "fleet-code-slop-scorecard/1" and debt is not None
        return bool(ok), f"code-slop payload reports slop_debt={debt}" if ok else "missing code-slop payload/debt"
    if name == "dogfood_coverage":
        debt = payload.get("dogfood_debt") if isinstance(payload, dict) else None
        ok = isinstance(payload, dict) and payload.get("schema") == "dogfood-coverage/1" and isinstance(debt, int)
        return bool(ok), f"dogfood coverage reports debt={debt}" if ok else "missing dogfood coverage payload/debt"
    if name == "callavoid_account":
        amp = payload.get("amplification") if isinstance(payload, dict) else None
        ok = (
            isinstance(payload, dict)
            and payload.get("schema") == "fak.callavoid.turns.v1"
            and payload.get("status") == "amplifying"
            and isinstance(amp, (int, float))
            and amp > 1
        )
        return bool(ok), f"callavoid CLI grades the window amplifying (amp={amp})" if ok else "callavoid CLI payload unexpected"
    if name == "nightrun_score":
        ok = (
            isinstance(payload, dict)
            and isinstance(payload.get("multi_turn_sessions"), int)
            and isinstance(payload.get("multi_turn_turns"), int)
            and isinstance(payload.get("realized_reuse_ratio"), (int, float))
            and payload.get("vs_naive_multiple_excluded") is True
        )
        ratio = payload.get("realized_reuse_ratio", 0) if isinstance(payload, dict) else 0
        mt_turns = payload.get("multi_turn_turns", 0) if isinstance(payload, dict) else 0
        return bool(ok), f"nightrun score: reuse ratio {ratio:.1%} over {mt_turns} multi-turn turns" if ok else "nightrun score payload invalid"
    return False, f"unknown validator {name}"


def summarize_payload(name: str, payload: Any) -> Any:
    if name == "benchmarks_offline" and isinstance(payload, list):
        entries = [row for row in payload if isinstance(row, dict)]
        names = [str(row.get("Name") or row.get("name")) for row in entries[:10]]
        return {"entries": len(entries), "names": names}
    if not isinstance(payload, dict):
        return payload
    base = {k: payload.get(k) for k in ("schema", "ok", "verdict", "finding", "status") if k in payload}
    if name == "loop_event":
        base.update({"seq": payload.get("seq"), "kind": payload.get("kind"), "hash": payload.get("hash")})
    elif name == "loop_status":
        loops = payload.get("loops") or []
        base.update({
            "loops": len(loops),
            "loop_ids": [loop.get("loop_id") for loop in loops[:5] if isinstance(loop, dict)],
        })
    elif name == "vcache_score":
        base.update({
            "two_x_better": payload.get("two_x_better"),
            "active_multiplier": payload.get("active_multiplier"),
            "grade": payload.get("grade"),
            "score": payload.get("score"),
        })
    elif name == "vcache_recall_refuted":
        base.update({
            "decision": payload.get("decision"),
            "break_even_siblings": payload.get("break_even_siblings"),
            "loss_ratio": payload.get("loss_ratio"),
        })
    elif name == "code_slop_scorecard":
        corpus = payload.get("corpus") or {}
        base.update({
            "score": corpus.get("score"),
            "grade": corpus.get("grade"),
            "slop_debt": corpus.get("slop_debt"),
        })
    elif name == "dogfood_coverage":
        base.update({
            "coverage": payload.get("coverage"),
            "grade": payload.get("grade"),
            "dogfood_debt": payload.get("dogfood_debt"),
            "audit_rows": payload.get("audit_rows"),
        })
    elif name == "nightrun_score":
        base.update({
            "multi_turn_turns": payload.get("multi_turn_turns"),
            "realized_reuse_ratio": payload.get("realized_reuse_ratio"),
            "multi_turn_sessions": payload.get("multi_turn_sessions"),
        })
    return base


def run_probe(
    probe: Probe,
    root: Path,
    *,
    timeout: int,
    runner: Callable[[list[str], Path, int], subprocess.CompletedProcess[str]] = run_completed,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = runner(probe.command, root, timeout)
    except subprocess.TimeoutExpired:
        dur_ms = int((time.monotonic() - started) * 1000)
        return {
            "key": probe.key,
            "description": probe.description,
            "ok": False,
            "required": probe.required,
            "exit_code": 124,
            "duration_ms": dur_ms,
            "command": probe.command,
            "reason": f"timed out after {timeout}s",
        }
    except OSError as exc:
        dur_ms = int((time.monotonic() - started) * 1000)
        return {
            "key": probe.key,
            "description": probe.description,
            "ok": False,
            "required": probe.required,
            "exit_code": 127,
            "duration_ms": dur_ms,
            "command": probe.command,
            "reason": str(exc),
        }

    dur_ms = int((time.monotonic() - started) * 1000)
    exit_ok = proc.returncode in probe.allowed_exits
    payload, json_err = load_probe_json(probe, proc, root)
    if json_err:
        valid, reason = False, json_err
    elif not exit_ok:
        valid, reason = False, f"exit {proc.returncode}, want one of {list(probe.allowed_exits)}"
    else:
        valid, reason = validate_payload(probe.validator, payload)
    return {
        "key": probe.key,
        "description": probe.description,
        "ok": bool(exit_ok and valid),
        "required": probe.required,
        "exit_code": proc.returncode,
        "allowed_exits": list(probe.allowed_exits),
        "duration_ms": dur_ms,
        "command": probe.command,
        "reason": reason,
        "stdout_tail": (proc.stdout or "")[-1500:],
        "stderr_tail": (proc.stderr or "")[-1500:],
        "payload": summarize_payload(probe.validator, payload) if probe.json_source else None,
    }


def run_suite(
    root: Path,
    out_dir: Path,
    *,
    timeout: int,
    include_go_tests: bool,
    runner: Callable[[list[str], Path, int], subprocess.CompletedProcess[str]] = run_completed,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    probes = [
        run_probe(p, root, timeout=timeout, runner=runner)
        for p in build_suite(root, out_dir, include_go_tests=include_go_tests)
    ]
    failed_required = [p for p in probes if p["required"] and not p["ok"]]
    report = {
        "schema": SCHEMA,
        "ok": not failed_required,
        "verdict": "OK" if not failed_required else "ACTION",
        "finding": "recent_features_dogfooded" if not failed_required else "recent_feature_dogfood_failed",
        "reason": (
            f"{len(probes) - len(failed_required)}/{len(probes)} probes passed"
            if not failed_required
            else f"{len(failed_required)} required probe(s) failed: "
            + ", ".join(p["key"] for p in failed_required)
        ),
        "workspace": str(root),
        "out_dir": str(out_dir),
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "probes": probes,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def render(report: dict[str, Any]) -> str:
    lines = [
        f"recent-feature-dogfood: {report['verdict']} ({report['finding']})",
        f"  {report['reason']}",
        f"  evidence: {report['out_dir']}",
    ]
    for probe in report["probes"]:
        mark = "OK " if probe["ok"] else "FAIL"
        lines.append(f"  [{mark}] {probe['key']:<30} exit={probe['exit_code']}  {probe['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dogfood recently-added fak features locally.")
    ap.add_argument("--workspace", default="", help="repo root (default: auto)")
    ap.add_argument("--out-dir", default="", help="evidence dir (default: .fak/recent-feature-dogfood/<utc>)")
    ap.add_argument("--timeout", type=int, default=120, help="per-probe timeout seconds")
    ap.add_argument("--no-go-tests", action="store_true", help="skip the Go unit-test probes")
    ap.add_argument("--json", action="store_true", help="emit the report JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        root / ".fak" / "recent-feature-dogfood" / utc_stamp()
    )
    report = run_suite(
        root,
        out_dir,
        timeout=args.timeout,
        include_go_tests=not args.no_go_tests,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
