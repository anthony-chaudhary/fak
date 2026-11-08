#!/usr/bin/env python3
"""Concept-coverage drift watcher - did the concept corpus re-open since the last look?

The concept-disambiguation scorecard (``tools/concept_disambiguation_scorecard.py``)
computes ``coverage_debt`` against the confusable-token universe it discovers in the
tree. That number is a SNAPSHOT: it says nothing about whether it has MOVED since the
last time anyone looked. And it moves - proven 2026-11-07/08, where merges re-opened
debt 2 twice (discovered 4170 -> 4173). This watcher closes that loop:

  --record-baseline   run the scorecard once and durably record its numbers (git head
                      + discovered / covered / coverage-debt + standing) to the
                      baseline file. Run this after retiring a drift batch.
  (default) check     run the scorecard again, diff against the baseline, write a
                      drift receipt next to the baseline. Exit 0 when debt did not
                      grow; exit 3 when coverage-debt RE-OPENED (delta > 0), printing
                      the unpositioned tokens per family.
  --simulate          prove the mechanism without touching real data: run the
                      scorecard on the intact catalog, then again on a temp copy
                      with the rows file covering the most tokens removed, and
                      require the debt to grow with every newly-uncovered token
                      belonging to the removed file. Exit 0 when the self-witness
                      holds.
  --auto-ticket       on a real drift, best-effort create a positioning ticket via
                      ``fak-run fak-ticket create concept-coverage-drift-<ts> ...``.

The scorecard runs as a SUBPROCESS (``python tools/concept_disambiguation_scorecard.py
--json``) so the watcher never re-implements discovery; its stdout is read through a
pipe (clean bytes), and a BOM / UTF-16 header is tolerated for manually redirected
output. Read-only over the tree; the only writes are the baseline + receipt JSONs.
Deterministic, ASCII, stdlib only.

Run from the repo ROOT::

    python tools/concept_drift_watcher.py --record-baseline
    python tools/concept_drift_watcher.py                # check vs baseline
    python tools/concept_drift_watcher.py --simulate     # self-witness
    python tools/concept_drift_watcher.py --auto-ticket  # check + position tickets

Exit codes: 0 clean / recorded / simulated OK - 2 operational error - 3 debt
re-opened - 4 simulate failed to detect its own drift.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASELINE_SCHEMA = "fak-concept-coverage-drift-baseline/1"
RECEIPT_SCHEMA = "fak-concept-coverage-drift-receipt/1"
SCORECARD_NAME = "concept_disambiguation_scorecard.py"
DEFAULT_DATA_DIR_REL = "tools/concept_disambiguation_scorecard.data"
DEFAULT_BASENAME = "drift-baseline.json"
RECEIPT_PREFIX = "drift-receipt-"
TICKET_LANE = "tools"
SCORECARD_TIMEOUT_S = 1800
GIT_TIMEOUT_S = 30
TICKET_TIMEOUT_S = 300
MAX_TICKET_TOKENS = 20

EXIT_OK = 0
EXIT_OPERATIONAL = 2
EXIT_DRIFT = 3
EXIT_SIMULATE_FAILED = 4

_NORM_RE = re.compile(r"[^a-z0-9]")


# ---------------------------------------------------------------------------
# Small pure helpers.
# ---------------------------------------------------------------------------

def norm_token(s: Any) -> str:
    """Collapse a name to its comparable token: lowercase, keep only [a-z0-9] -
    the same normalization the scorecard uses, so a watcher-side count agrees
    with the scorecard's coverage bookkeeping."""
    if not isinstance(s, str):
        return ""
    return _NORM_RE.sub("", s.lower())


def row_tokens(r: dict[str, Any]) -> set[str]:
    """Every normalized name one catalog row answers to: canonical + grounding +
    aliases. The local row_tokens-equivalent used to decide which rows file's
    removal uncovers the most tokens."""
    toks = {norm_token(r.get("canonical", "")), norm_token(r.get("grounding", ""))}
    for a in r.get("aliases") or []:
        toks.add(norm_token(a))
    return {t for t in toks if t}


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp_iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_compact(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y%m%d-%H%M")


def decode_json_bytes(raw: bytes) -> Any:
    """Parse the scorecard's JSON payload from raw stdout bytes.

    The pipe path is clean UTF-8; the BOM / UTF-16 branches cover manually
    redirected output (PowerShell 5 ``>`` writes UTF-16 with a BOM)."""
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        text = raw.decode("utf-8")
    return json.loads(text.lstrip("\ufeff").lstrip())


def git_head(workspace: Path) -> str:
    """The workspace HEAD sha, best-effort ('' when not a git checkout)."""
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(workspace),
                           capture_output=True, timeout=GIT_TIMEOUT_S)
        if p.returncode == 0:
            return p.stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


# ---------------------------------------------------------------------------
# Scorecard subprocess + snapshot extraction.
# ---------------------------------------------------------------------------

def run_scorecard(workspace: Path, data_dir: Path | None = None,
                  timeout: int = SCORECARD_TIMEOUT_S
                  ) -> tuple[dict[str, Any] | None, str]:
    """Run the sibling scorecard as a subprocess and parse its --json payload.

    A non-zero exit is NOT an error here: the scorecard exits 1 whenever the
    namespace is not ok - which is exactly the drift state this watcher measures.
    Only a missing / undecodable payload is an operational failure."""
    script = Path(__file__).resolve().parent / SCORECARD_NAME
    cmd = [sys.executable, str(script), "--json", "--workspace", str(workspace)]
    if data_dir is not None:
        cmd += ["--data", str(data_dir)]
    try:
        p = subprocess.run(cmd, cwd=str(workspace), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"scorecard timed out after {timeout}s"
    except OSError as exc:
        return None, f"cannot launch scorecard: {exc}"
    try:
        payload = decode_json_bytes(p.stdout)
    except (ValueError, UnicodeDecodeError) as exc:
        tail = (p.stderr or b"")[-400:].decode("utf-8", "replace").strip()
        return None, f"cannot parse scorecard JSON (exit {p.returncode}): {exc}; stderr: {tail}"
    if not isinstance(payload, dict):
        return None, "scorecard payload is not a JSON object"
    return payload, ""


def extract_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """The drift-relevant slice of a scorecard payload: the coverage counters,
    the per-verdict standing, and the unpositioned token list (family + token)."""
    c = payload.get("corpus") or {}
    cov = c.get("coverage") or {}
    return {
        "discovered": int(cov.get("discovered", 0)),
        "covered": int(cov.get("covered", 0)),
        "coverage_debt": int(cov.get("coverage_debt", 0)),
        "standing": c.get("standing") or {},
        "uncovered": cov.get("uncovered") or [],
    }


def uncovered_by_family(uncovered: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group the unpositioned tokens by owning family, deterministically sorted."""
    out: dict[str, list[str]] = {}
    for u in uncovered:
        fam = str(u.get("family") or "?")
        tok = str(u.get("token") or "")
        if tok:
            out.setdefault(fam, []).append(tok)
    for fam in out:
        out[fam].sort()
    return dict(sorted(out.items()))


def audit_error(payload: dict[str, Any] | None) -> str:
    """The scorecard's own AUDIT_ERROR reason, when the payload says the read failed."""
    if payload and payload.get("verdict") == "AUDIT_ERROR":
        return f"scorecard audit error: {payload.get('reason')}"
    return ""


# ---------------------------------------------------------------------------
# Baseline + receipt IO.
# ---------------------------------------------------------------------------

def load_baseline(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, f"no baseline at {path} - run --record-baseline first"
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return None, f"cannot read baseline {path.name}: {exc}"
    if not isinstance(doc, dict):
        return None, f"baseline {path.name} is not a JSON object"
    return doc, ""


def write_receipt(receipt: dict[str, Any], baseline_path: Path) -> Path:
    """The drift receipt lands NEXT TO the baseline it diffs against."""
    ts = receipt["recorded_at"].replace(":", "")
    out = baseline_path.parent / f"{RECEIPT_PREFIX}{ts}.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Modes.
# ---------------------------------------------------------------------------

def record_baseline(workspace: Path, data_dir: Path | None, baseline_path: Path) -> int:
    payload, err = run_scorecard(workspace, data_dir)
    if err:
        print(f"drift-watcher: {err}", file=sys.stderr)
        return EXIT_OPERATIONAL
    reason = audit_error(payload)
    if reason:
        print(f"drift-watcher: {reason}", file=sys.stderr)
        return EXIT_OPERATIONAL
    snap = extract_snapshot(payload)
    baseline = {
        "schema": BASELINE_SCHEMA,
        "recorded_at": stamp_iso(),
        "git_head": git_head(workspace),
        "discovered": snap["discovered"],
        "covered": snap["covered"],
        "coverage_debt": snap["coverage_debt"],
        "standing": snap["standing"],
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"baseline recorded -> {baseline_path}")
    print(f"  git_head={baseline['git_head'] or '(no git head)'} "
          f"discovered={snap['discovered']} covered={snap['covered']} "
          f"coverage_debt={snap['coverage_debt']}")
    return EXIT_OK


def check_mode(workspace: Path, data_dir: Path | None, baseline_path: Path,
               auto_ticket: bool) -> int:
    baseline, err = load_baseline(baseline_path)
    if err:
        print(f"drift-watcher: {err}", file=sys.stderr)
        return EXIT_OPERATIONAL
    payload, err = run_scorecard(workspace, data_dir)
    if err:
        print(f"drift-watcher: {err}", file=sys.stderr)
        return EXIT_OPERATIONAL
    reason = audit_error(payload)
    if reason:
        print(f"drift-watcher: {reason}", file=sys.stderr)
        return EXIT_OPERATIONAL
    snap = extract_snapshot(payload)
    b_debt = int(baseline.get("coverage_debt", 0))
    b_disc = int(baseline.get("discovered", 0))
    debt_delta = snap["coverage_debt"] - b_debt
    disc_delta = snap["discovered"] - b_disc
    drifted = debt_delta > 0
    ubf = uncovered_by_family(snap["uncovered"])
    head = git_head(workspace)
    n_uncovered = sum(len(v) for v in ubf.values())
    if drifted:
        why = (f"coverage-debt re-opened {b_debt} -> {snap['coverage_debt']} "
               f"(+{debt_delta}) since baseline recorded {baseline.get('recorded_at', '?')} "
               f"at {baseline.get('git_head') or 'unknown head'}; "
               f"{n_uncovered} token(s) unpositioned")
    else:
        why = (f"coverage-debt held at {snap['coverage_debt']} (baseline {b_debt}); "
               f"discovered {snap['discovered']} ({disc_delta:+d}); no re-opening")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "kind": "check",
        "recorded_at": stamp_iso(),
        "workspace": str(workspace),
        "git_head": head,
        "baseline_file": str(baseline_path),
        "baseline": {k: baseline.get(k) for k in
                     ("recorded_at", "git_head", "discovered", "covered",
                      "coverage_debt", "standing")},
        "current": {k: snap[k] for k in ("discovered", "covered", "coverage_debt",
                                         "standing")},
        "delta": {"coverage_debt": debt_delta, "discovered": disc_delta},
        "drift": drifted,
        "verdict": "DRIFT" if drifted else "OK",
        "uncovered_by_family": ubf,
        "reason": why,
    }
    rp = write_receipt(receipt, baseline_path)
    if drifted:
        print(f"DRIFT: coverage-debt re-opened {b_debt} -> {snap['coverage_debt']} "
              f"(+{debt_delta}); discovered {b_disc} -> {snap['discovered']} "
              f"({disc_delta:+d}) at {head or '(no git head)'}")
        for fam, toks in ubf.items():
            shown = ", ".join(toks[:MAX_TICKET_TOKENS])
            if len(toks) > MAX_TICKET_TOKENS:
                shown += f" (+{len(toks) - MAX_TICKET_TOKENS} more)"
            print(f"  {fam}: {shown}")
        print(f"receipt: {rp}")
        if auto_ticket:
            create_drift_ticket(workspace, debt_delta, [t for v in ubf.values() for t in v])
        return EXIT_DRIFT
    print(f"OK: coverage-debt held at {snap['coverage_debt']} (baseline {b_debt}); "
          f"discovered {snap['discovered']} ({disc_delta:+d}); no re-opening")
    print(f"receipt: {rp}")
    return EXIT_OK


def create_drift_ticket(workspace: Path, n_reopened: int, tokens: list[str]) -> None:
    """Best-effort positioning ticket on a real drift. Never fatal: a missing
    fak-run or a refused create is reported and swallowed."""
    exe = shutil.which("fak-run")
    if not exe:
        print("auto-ticket: fak-run not on PATH - ticket skipped (best-effort)")
        return
    shown = ", ".join(tokens[:MAX_TICKET_TOKENS])
    if len(tokens) > MAX_TICKET_TOKENS:
        shown += f" (+{len(tokens) - MAX_TICKET_TOKENS} more)"
    intent = f"position {n_reopened} re-opened tokens: {shown}"
    tid = f"concept-coverage-drift-{stamp_compact()}"
    cmd = [exe, "fak-ticket", "create", tid, "--lane", TICKET_LANE, "--intent", intent]
    try:
        p = subprocess.run(cmd, cwd=str(workspace), capture_output=True,
                           timeout=TICKET_TIMEOUT_S)
        if p.returncode == 0:
            print(f"auto-ticket: created {tid}")
        else:
            tail = (p.stderr or p.stdout or b"")[-300:].decode("utf-8", "replace").strip()
            print(f"auto-ticket: fak-run exited {p.returncode} ({tail}) - ticket not created")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"auto-ticket: skipped ({exc})")


def file_tokens(rows_file: Path) -> set[str]:
    """The normalized tokens one rows file covers (row_tokens-equivalent, summed
    over its rows)."""
    try:
        doc = json.loads(rows_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return set()
    out: set[str] = set()
    for r in (doc or {}).get("rows") or []:
        if isinstance(r, dict):
            out |= row_tokens(r)
    return out


def pick_victim(data_dir: Path) -> tuple[Path, set[str]]:
    """The rows file whose removal uncovers the most tokens: argmax over
    |row_tokens|, ties broken by file name (deterministic)."""
    candidates: list[tuple[set[str], Path]] = []
    for p in sorted(data_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        candidates.append((file_tokens(p), p))
    if not candidates:
        raise ValueError(f"no rows-*.json files in {data_dir}")
    candidates.sort(key=lambda tp: (-len(tp[0]), tp[1].name))
    toks, victim = candidates[0]
    return victim, toks


def simulate(workspace: Path, data_dir: Path | None) -> int:
    """Self-witness, as a DIFF between two scorecard runs: one on the intact
    catalog, one on a TEMP COPY with the biggest rows file removed.

    The drift shows when the post-run debt exceeds the pre-run debt and EVERY
    newly-uncovered token belongs to the removed file. (Not every token of the
    removed file becomes uncovered: row surfaces that were never DISCOVERED tree
    tokens - e.g. glossed canonicals below the presence threshold - do not enter
    the coverage universe at all.) Proves the mechanism end to end without
    touching the real catalog."""
    src = (data_dir or (repo_root() / DEFAULT_DATA_DIR_REL)).resolve()
    if not src.is_dir():
        print(f"drift-watcher: no data directory at {src}", file=sys.stderr)
        return EXIT_OPERATIONAL
    payload0, err = run_scorecard(workspace, src)
    if err:
        print(f"drift-watcher: {err}", file=sys.stderr)
        return EXIT_OPERATIONAL
    reason = audit_error(payload0)
    if reason:
        print(f"drift-watcher: {reason}", file=sys.stderr)
        return EXIT_OPERATIONAL
    snap0 = extract_snapshot(payload0)
    with tempfile.TemporaryDirectory(prefix="concept-drift-sim-") as td:
        dst = Path(td) / "data"
        shutil.copytree(src, dst)
        try:
            victim, removed = pick_victim(dst)
        except ValueError as exc:
            print(f"drift-watcher: cannot simulate: {exc}", file=sys.stderr)
            return EXIT_OPERATIONAL
        victim.unlink()
        payload1, err = run_scorecard(workspace, dst)
        if err:
            print(f"drift-watcher: {err}", file=sys.stderr)
            return EXIT_OPERATIONAL
        reason = audit_error(payload1)
        if reason:
            print(f"drift-watcher: {reason}", file=sys.stderr)
            return EXIT_OPERATIONAL
        snap1 = extract_snapshot(payload1)
        u0 = {str(u.get("token") or "") for u in snap0["uncovered"]}
        u1 = {str(u.get("token") or "") for u in snap1["uncovered"]}
        new_uncovered = sorted(u1 - u0)
        foreign = sorted(t for t in new_uncovered if t not in removed)
        debt_delta = snap1["coverage_debt"] - snap0["coverage_debt"]
        detected = debt_delta > 0 and bool(new_uncovered) and not foreign
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "kind": "simulate",
            "recorded_at": stamp_iso(),
            "workspace": str(workspace),
            "git_head": git_head(workspace),
            "removed_file": victim.name,
            "removed_tokens": sorted(removed),
            "pre": {"discovered": snap0["discovered"], "covered": snap0["covered"],
                    "coverage_debt": snap0["coverage_debt"]},
            "post": {"discovered": snap1["discovered"], "covered": snap1["covered"],
                     "coverage_debt": snap1["coverage_debt"]},
            "delta": {"coverage_debt": debt_delta},
            "newly_uncovered": new_uncovered,
            "drift_detected": detected,
            "verdict": "DRIFT-DETECTED" if detected else "NO-DRIFT",
            "reason": (f"self-witness holds: removing {victim.name} "
                       f"({len(removed)} tokens) re-opened coverage-debt "
                       f"{snap0['coverage_debt']} -> {snap1['coverage_debt']} "
                       f"(+{debt_delta}); all {len(new_uncovered)} newly-uncovered "
                       f"token(s) belong to the removed file"
                       if detected else
                       f"self-witness FAILED: removing {victim.name} moved debt "
                       f"{snap0['coverage_debt']} -> {snap1['coverage_debt']} "
                       f"(delta {debt_delta:+d}); newly-uncovered={len(new_uncovered)}; "
                       f"foreign tokens (not in removed file): {foreign[:20]}"),
        }
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return EXIT_OK if detected else EXIT_SIMULATE_FAILED


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Concept-coverage drift watcher: diff the scorecard's "
                    "coverage-debt against a recorded baseline; exit 3 when debt "
                    "re-opens.")
    ap.add_argument("--workspace", default="",
                    help="workspace root (default: repo root)")
    ap.add_argument("--data", default="",
                    help=f"data directory (default: {DEFAULT_DATA_DIR_REL})")
    ap.add_argument("--baseline", default="",
                    help=f"baseline JSON (default: <data-dir>/{DEFAULT_BASENAME})")
    ap.add_argument("--record-baseline", action="store_true",
                    help="run the scorecard and record the baseline")
    ap.add_argument("--simulate", action="store_true",
                    help="self-witness: remove the biggest rows file in a temp "
                         "copy and require the drift to show")
    ap.add_argument("--auto-ticket", action="store_true",
                    help="on drift, best-effort create a positioning ticket via "
                         "fak-run")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    data_dir = Path(args.data).resolve() if args.data else None
    default_data = workspace / DEFAULT_DATA_DIR_REL
    baseline_path = (Path(args.baseline).resolve() if args.baseline
                     else (data_dir or default_data) / DEFAULT_BASENAME)

    if args.record_baseline:
        return record_baseline(workspace, data_dir, baseline_path)
    if args.simulate:
        return simulate(workspace, data_dir)
    return check_mode(workspace, data_dir, baseline_path, args.auto_ticket)


if __name__ == "__main__":
    raise SystemExit(main())
