#!/usr/bin/env python3
"""Cache-value ledger for continuous dogfood runs with 2x regression gate (issue #1075).

Tracks cache-value metrics from dogfood runs in a persistent ledger and enforces
a 2x regression guard: current multiplier must stay >= 2x threshold. This provides
continuous visibility into cache-value performance and prevents silent regressions.

Usage:
    python tools/cache_value_ledger.py --ledger experiments/agent-live/cache-value-ledger.jsonl --check 2x
    python tools/cache_value_ledger.py --ledger experiments/agent-live/cache-value-ledger.jsonl --append

Exit 0 = OK (2x threshold maintained). Exit 1 = REGRESSION (sub-2x multiplier).
Exit 2 = HARNESS ERROR (binary unavailable or ledger unreadable).

Ledger format (JSONL):
    {"schema":"fak.cache-value-ledger.v1","utc":"2026-06-27T12:00:00Z",
     "active_multiplier":3.76,"two_x_better":true,"grade":"A",
     "economics":{"hit_rate":0.9553,"cache_read_tokens":10163712,"rebate_token_equiv":9147340.8,
                  "cost_token_equiv":1491490.2,"baseline_token_equiv":10638831,"multiplier":7.13}}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "fak.cache-value-ledger.v1"
SCORE_SCHEMA = "fak.vcache.score.v1"


@dataclass(frozen=True)
class CacheValueEntry:
    """One cache-value ledger entry, minimal and reproducible."""
    utc: str
    active_multiplier: float
    two_x_better: bool
    grade: str
    economics_hit_rate: float | None
    economics_cache_read_tokens: float | None
    economics_rebate_token_equiv: float | None
    economics_cost_token_equiv: float | None
    economics_baseline_token_equiv: float | None
    economics_multiplier: float | None


def repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def fak_cmd() -> list[str]:
    configured = os.environ.get("FAK_BIN", "").strip()
    if configured:
        return [configured]
    for rel in ("fak.exe", "fak", "tools/.bin/fak"):
        p = repo_root() / rel
        if p.exists():
            return [str(p)]
    found = shutil.which("fak")
    if found:
        return [found]
    return ["go", "run", "./cmd/fak"]


def run_vcache_score(timeout: int = 120) -> tuple[int, dict]:
    """Run `fak vcache score --json` and return (exit_code, payload)."""
    cmd = fak_cmd() + ["vcache", "score", "--json"]
    proc = subprocess.run(
        cmd, cwd=str(repo_root()), capture_output=True, text=True, timeout=timeout
    )
    try:
        payload = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    return proc.returncode, payload


def extract_metrics(score_payload: dict) -> CacheValueEntry:
    """Extract cache-value metrics from the vCache score payload."""
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    active_multiplier = score_payload.get("active_multiplier", 0.0)
    two_x_better = score_payload.get("two_x_better", False)
    grade = score_payload.get("grade", "F")
    econ = score_payload.get("economics") or {}
    return CacheValueEntry(
        utc=utc,
        active_multiplier=float(active_multiplier),
        two_x_better=bool(two_x_better),
        grade=str(grade),
        economics_hit_rate=econ.get("hit_rate"),
        economics_cache_read_tokens=econ.get("cache_read_tokens"),
        economics_rebate_token_equiv=econ.get("rebate_token_equiv"),
        economics_cost_token_equiv=econ.get("cost_token_equiv"),
        economics_baseline_token_equiv=econ.get("baseline_token_equiv"),
        economics_multiplier=econ.get("multiplier"),
    )


def append_entry(ledger_path: Path, entry: CacheValueEntry) -> None:
    """Append one cache-value entry to the ledger file (JSONL)."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": SCHEMA,
        "utc": entry.utc,
        "active_multiplier": entry.active_multiplier,
        "two_x_better": entry.two_x_better,
        "grade": entry.grade,
        "economics": {
            "hit_rate": entry.economics_hit_rate,
            "cache_read_tokens": entry.economics_cache_read_tokens,
            "rebate_token_equiv": entry.economics_rebate_token_equiv,
            "cost_token_equiv": entry.economics_cost_token_equiv,
            "baseline_token_equiv": entry.economics_baseline_token_equiv,
            "multiplier": entry.economics_multiplier,
        },
    }
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def read_ledger(ledger_path: Path) -> list[CacheValueEntry]:
    """Read all entries from the ledger file, ignoring malformed rows."""
    if not ledger_path.exists():
        return []
    entries = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            econ = row.get("economics") or {}
            entries.append(CacheValueEntry(
                utc=row.get("utc", ""),
                active_multiplier=float(row.get("active_multiplier", 0)),
                two_x_better=bool(row.get("two_x_better", False)),
                grade=str(row.get("grade", "F")),
                economics_hit_rate=econ.get("hit_rate"),
                economics_cache_read_tokens=econ.get("cache_read_tokens"),
                economics_rebate_token_equiv=econ.get("rebate_token_equiv"),
                economics_cost_token_equiv=econ.get("cost_token_equiv"),
                economics_baseline_token_equiv=econ.get("baseline_token_equiv"),
                economics_multiplier=econ.get("multiplier"),
            ))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return entries


def check_regression(entry: CacheValueEntry, threshold: float) -> tuple[bool, str]:
    """Check if the current entry meets the 2x regression threshold."""
    if not isinstance(entry.active_multiplier, (int, float)):
        return False, f"active_multiplier is not a number: {entry.active_multiplier!r}"
    if entry.active_multiplier < threshold:
        return False, f"active_multiplier {entry.active_multiplier:.2f} < 2x threshold {threshold}"
    return True, f"active_multiplier {entry.active_multiplier:.2f} >= 2x threshold {threshold}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True,
                    help="path to cache-value ledger file (JSONL)")
    ap.add_argument("--check", type=float, default=2.0,
                    help="check current score against threshold (default 2.0)")
    ap.add_argument("--append", action="store_true",
                    help="append current metrics to ledger")
    ap.add_argument("--timeout", type=int, default=120,
                    help="timeout for `fak vcache score` (seconds)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable verdict")
    args = ap.parse_args(argv)

    ledger_path = Path(args.ledger).resolve()
    try:
        code, payload = run_vcache_score(args.timeout)
    except (subprocess.SubprocessError, OSError, FileNotFoundError) as e:
        msg = f"cache-value-ledger: could not run fak vcache score: {e}"
        if args.json:
            print(json.dumps({"schema": SCHEMA, "ok": False, "error": str(e)}))
        else:
            print(msg, file=sys.stderr)
        return 2

    if code != 0:
        msg = f"cache-value-ledger: fak vcache score exited {code}, want 0"
        if args.json:
            print(json.dumps({"schema": SCHEMA, "ok": False, "error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 2

    if payload.get("schema") != SCORE_SCHEMA:
        msg = f"cache-value-ledger: wrong schema {payload.get('schema')!r}, want {SCORE_SCHEMA!r}"
        if args.json:
            print(json.dumps({"schema": SCHEMA, "ok": False, "error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 2

    entry = extract_metrics(payload)
    ok, reason = check_regression(entry, args.check)

    if args.append and ok:
        append_entry(ledger_path, entry)
        reason += f" (appended to {ledger_path})"

    if args.json:
        verdict = {
            "schema": SCHEMA,
            "ok": ok,
            "threshold": args.check,
            "active_multiplier": entry.active_multiplier,
            "two_x_better": entry.two_x_better,
            "grade": entry.grade,
            "reason": reason,
            "economics": {
                "hit_rate": entry.economics_hit_rate,
                "cache_read_tokens": entry.economics_cache_read_tokens,
                "rebate_token_equiv": entry.economics_rebate_token_equiv,
                "cost_token_equiv": entry.economics_cost_token_equiv,
                "baseline_token_equiv": entry.economics_baseline_token_equiv,
                "multiplier": entry.economics_multiplier,
            },
        }
        print(json.dumps(verdict, indent=2))
    else:
        if ok:
            print(f"cache-value-ledger: OK -- {reason}")
        else:
            print(f"cache-value-ledger: REGRESSION -- {reason}", file=sys.stderr)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())