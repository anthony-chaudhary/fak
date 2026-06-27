#!/usr/bin/env python3
r"""bench-signal — turn a benchmark THROUGHPUT regression (a tok/s drop vs a pinned
perf baseline) into a specific, deduped, lane-routed GitHub issue the dispatch loop
can resolve. The PERF dual of tools/score_signal.py (a scorecard-debt regression)
and a sibling of tools/gate_signal.py (a non-scorecard CI finding).

  score_signal.py  : scorecard control-pane debt rose   -> a deduped issue
  gate_signal.py   : a non-scorecard CI finding (red)    -> a deduped issue
  bench_signal.py  : a benchmark tok/s dropped (this)    -> a deduped issue

Same "turn a CI signal into an actionable item, IF RELEVANT" discipline as the
siblings: relevance filter -> label-scoped per-benchmark marker dedup -> worst-drop-
first hard CAP -> DRY-RUN by default, `--live` the explicit arm. Only the input and
the relevance threshold differ — perf is noisier than scorecard debt, so the filter
is stricter (below).

THE SOURCE — experiments/benchmark/catalog.json (schema benchmark/catalog.v1), the
committed, machine-scoped benchmark corpus: a flat `runs[]` of
{run_id, machine_id, model, precision, baseline_tok_per_sec, peak_tok_per_sec,
speedup, timestamp, tags, path}. The CURRENT number for a benchmark is the latest
(by timestamp) run's `peak_tok_per_sec` (fallback `baseline_tok_per_sec`). The
PINNED reference is a committed tools/bench_baseline.json (mirrors
tools/scorecard_baseline.json), keyed by `<machine_id>/<benchmark_key>`.

MACHINE-SCOPING (the issue's open question) — the regression key is
`<machine_id>/<benchmark_key>`, so a drop on `cpu-server-a` NEVER dedups against (or
compares against) the same benchmark on `gcp-g2-l4`. `benchmark_key` is a stable
slug of the model + precision, so the same model at a different quant is tracked
apart. A node-specific slowdown surfaces as its own ticket on its own machine row.

THE NOISE THRESHOLD (the issue's open question) — perf is noisier than scorecard
debt, so a drop fires ONLY when BOTH hold:
  * a RELATIVE drop >= --min-drop-pct (default 15.0%) — absorbs scale (a 5% wobble
    on a fast benchmark is noise, not a regression), and
  * an ABSOLUTE drop >= --min-abs tok/s (default 1.0) — absorbs tiny-number noise (a
    15% drop on a 2 tok/s micro-benchmark is 0.3 tok/s, below the floor).
The HIGHER of peak/baseline tok/s is the value compared. A null/missing current OR
baseline number is SKIPPED (a measurement gap the bench owns, not a regression).

DEDUP — a perf regression is RECURRING work (re-file after a close). Dedups against
OPEN bench-signal-labelled issues only, by a per-benchmark marker
(`<!-- fak-bench-signal: <machine>/<key> -->`): at most one OPEN issue per
machine+benchmark, re-fileable after a close. Label-scoped fetch bounds the index
EXACTLY. A hard CAP (--max-issues, worst-drop-first) caps a multi-benchmark storm.

PINNING — `--pin` writes/refreshes tools/bench_baseline.json from the CURRENT
catalog (mirrors the scorecard `--pin`), so an intentional perf change (or the first
ever run) re-floors the baseline instead of re-filing forever.

SAFE BY DEFAULT: dry-run. `--live` is the explicit opt-in that creates the issues.

    python tools/bench_signal.py                       # dry-run: plan, file nothing
    python tools/bench_signal.py --json                # machine-readable plan
    python tools/bench_signal.py --pin                 # re-floor the baseline
    python tools/bench_signal.py --min-drop-pct 20 --live

Exit codes: 0 = ran clean (including "nothing to signal") · 2 = infra error (gh
missing / not authed / the catalog or baseline could not be read).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fak-bench-signal/1"
BASELINE_SCHEMA = "fak-bench-signal.baseline/1"
SIGNAL_LABEL = "bench-signal"
PERF_LABEL = "performance"
MARKER_RE = re.compile(r"fak-bench-signal:\s*([A-Za-z0-9_:./\-]+)")

CATALOG_PATH = "experiments/benchmark/catalog.json"
BASELINE_PATH = "tools/bench_baseline.json"

DEFAULTS = {
    "min_drop_pct": 15.0,   # relative drop (%) below baseline to be actionable
    "min_abs": 1.0,         # absolute drop (tok/s) floor — absorbs tiny-number noise
    "max_issues": 5,        # hard cap on issues filed per run (anti-storm)
    "issue_scan_limit": 800,  # open bench-signal issues fetched for the dedup index
}


# ============================================================================
# Pure helpers (no I/O) — unit-tested directly.
# ============================================================================
def benchmark_key(run: dict[str, Any]) -> str:
    """A stable per-benchmark slug from the model + precision — the same model at a
    different quant tracks apart. Excludes the machine (that is the other half of the
    full key) and the timestamp (a benchmark is the same across runs)."""
    model = str(run.get("model") or "model")
    prec = str(run.get("precision") or "")
    raw = f"{model}-{prec}" if prec and prec.lower() not in ("n/a", "none") else model
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9.]+", "-", raw.lower())).strip("-") or "bench"


def full_key(machine_id: str, bench_key: str) -> str:
    """The machine-scoped regression key: a drop on one machine never dedups against
    (or compares against) the same benchmark on another."""
    return f"{_slug(machine_id)}/{bench_key}"


def _slug(text: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9._-]+", "-", str(text).lower())).strip("-") or "x"


def run_value(run: dict[str, Any]) -> float | None:
    """The throughput number for a run: the HIGHER of peak/baseline tok/s (peak is
    the headline; baseline is a fallback). None when neither is a real number, so a
    measurement gap is skipped rather than read as a 0 regression."""
    vals = []
    for k in ("peak_tok_per_sec", "baseline_tok_per_sec"):
        v = run.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            vals.append(float(v))
    return max(vals) if vals else None


def current_by_key(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The CURRENT measured value per machine-scoped key: the latest (by timestamp)
    run that has a real tok/s number. Runs without a number are skipped so a stale
    real datum is not overwritten by a newer null one."""
    out: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        val = run_value(run)
        if val is None:
            continue
        machine = str(run.get("machine_id") or "")
        if not machine:
            continue
        key = full_key(machine, benchmark_key(run))
        ts = str(run.get("timestamp") or "")
        prev = out.get(key)
        if prev is None or ts >= prev["timestamp"]:
            out[key] = {
                "key": key, "machine_id": machine, "model": run.get("model"),
                "precision": run.get("precision"), "value": val,
                "timestamp": ts, "run_id": run.get("run_id"),
            }
    return out


def regressions(current: dict[str, dict[str, Any]], baseline: dict[str, Any], *,
                min_drop_pct: float, min_abs: float) -> list[dict[str, Any]]:
    """Every machine-scoped benchmark whose current value dropped past BOTH the
    relative %-drop and the absolute-tok/s floor vs the pinned baseline, worst-drop
    first. A key absent from the baseline (never pinned) or with a null current value
    is skipped — there is no regression to act on without two real numbers."""
    pins = baseline.get("baselines") or {}
    base_commit = str(baseline.get("commit") or "")
    out: list[dict[str, Any]] = []
    for key, cur in current.items():
        base = pins.get(key)
        if not isinstance(base, (int, float)) or isinstance(base, bool) or base <= 0:
            continue
        cur_val = cur["value"]
        drop_abs = base - cur_val
        if drop_abs <= 0:
            continue  # improved or flat — not a regression
        drop_pct = 100.0 * drop_abs / base
        if drop_pct < min_drop_pct or drop_abs < min_abs:
            continue  # within the noise band on at least one axis
        out.append({
            **cur, "baseline": float(base), "current": cur_val,
            "drop_abs": drop_abs, "drop_pct": drop_pct,
            "baseline_commit": base_commit,
        })
    out.sort(key=lambda r: (-r["drop_pct"], r["key"]))
    return out


def marker(key: str) -> str:
    """The load-bearing dedup anchor: one stable HTML-comment per machine+benchmark."""
    return f"<!-- fak-bench-signal: {key} -->"


def open_issue_keys(issues: list[dict[str, Any]]) -> set[str]:
    """The set of benchmark keys already tracked by an OPEN bench-signal issue."""
    keys: set[str] = set()
    for iss in issues:
        for m in MARKER_RE.findall(iss.get("body") or ""):
            keys.add(m.strip())
    return keys


def render_issue(cand: dict[str, Any], today: str) -> dict[str, Any]:
    """Build the {title, body, labels} an issue is created from. The per-benchmark
    marker is the load-bearing dedup anchor; the body names the benchmark, machine,
    before->after tok/s, the baseline commit, the re-measure path (for lane routing),
    and the `#N`-stamp worker contract."""
    key = cand["key"]
    machine = cand["machine_id"]
    model = cand.get("model") or "(model)"
    prec = cand.get("precision") or ""
    base, cur = cand["baseline"], cand["current"]
    drop_pct, drop_abs = cand["drop_pct"], cand["drop_abs"]
    base_commit = cand.get("baseline_commit") or ""
    run_id = cand.get("run_id") or ""
    prec_s = f" ({prec})" if prec and str(prec).lower() not in ("n/a", "none") else ""

    title = (f"bench-signal: {model}{prec_s} on {machine} dropped "
             f"-{drop_pct:.1f}% ({base:.1f}->{cur:.1f} tok/s)")

    # The re-measure path is named in the doc-link convention so the dispatch router
    # path-confirms the `tools` lane (the perf-feeder + baseline live under tools/).
    remeasure_path = "tools/bench_signal.py"

    do_lines = [
        f"1. Reproduce the benchmark for `{model}{prec_s}` on `{machine}` "
        f"(latest run `{run_id}`).",
        "2. Find the regression's cause (a kernel/dispatch change, a config drift, a "
        "host change) and restore throughput; change nothing the benchmark does not "
        "cover.",
        f"3. Re-measure on `{machine}` to PROVE the drop is recovered; if the drop is "
        f"an INTENTIONAL trade-off, re-floor the baseline with "
        f"`python {remeasure_path} --pin` and close as accepted.",
        "4. Ship a commit citing this issue's `#N` in the subject + a `(fak <leaf>)` "
        "trailer, so the witness can bind and close it.",
    ]

    body = (
        f"> Auto-filed by **bench-signal** (`{remeasure_path}`, {today}) from a "
        f"throughput regression in `{CATALOG_PATH}`. A measured tok/s drop vs the "
        f"pinned perf baseline — **needs a worker**; if the drop is an accepted "
        f"trade-off, re-pin the baseline (`python {remeasure_path} --pin`).\n\n"
        f"**Benchmark:** `{model}{prec_s}`  ·  **Machine:** `{machine}`  ·  "
        f"**Key:** `{key}`\n"
        f"**Change:** **-{drop_pct:.1f}%** ({base:.1f} -> {cur:.1f} tok/s, a "
        f"{drop_abs:.1f} tok/s drop) vs the pinned baseline"
        + (f" @{base_commit}" if base_commit else "")
        + f" (latest run `{run_id}`).\n\n"
        f"**Why this fired:** the drop cleared BOTH the relative-% and absolute-tok/s "
        f"noise floors — a real regression, not measurement wobble. The key is "
        f"machine-scoped (`<machine>/<benchmark>`), so this is specific to `{machine}` "
        f"and does not implicate the same benchmark on another node.\n\n"
        f"**What to do**\n"
        + "\n".join(do_lines)
        + "\n\n---\n"
        f"_bench-signal is the perf dual of score-signal. The corpus is "
        f"`{CATALOG_PATH}`; the pinned baseline is `{BASELINE_PATH}`. After this "
        f"issue is closed, a future regression of the same machine+benchmark files a "
        f"FRESH issue — only on an armed `--live` run._\n"
        + marker(key)
    )

    return {
        "title": title, "body": body, "labels": [SIGNAL_LABEL, PERF_LABEL],
        "key": key, "drop_pct": drop_pct, "drop_abs": drop_abs,
    }


def plan_issues(regs: list[dict[str, Any]], open_keys: set[str], *,
                max_issues: int, today: str,
                ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """relevance (already in regressions()) -> dedup vs OPEN issues -> CAP. Returns
    (issues, skip_stats). Deterministic: regressions are worst-drop-first, deduped by
    key, then capped."""
    stats = {"already-open": 0, "within-run-dup": 0, "over-cap": 0}
    out: list[dict[str, Any]] = []
    seen_run: set[str] = set()
    for cand in regs:
        key = cand["key"]
        if key in seen_run:
            stats["within-run-dup"] += 1
            continue
        seen_run.add(key)
        if key in open_keys:
            stats["already-open"] += 1
            continue
        out.append(render_issue(cand, today))
    if len(out) > max_issues:
        stats["over-cap"] = len(out) - max_issues
        out = out[:max_issues]
    return out, stats


def build_baseline(current: dict[str, dict[str, Any]], commit: str, today: str,
                   ) -> dict[str, Any]:
    """The pinned baseline written by --pin: the current measured value per key, so a
    later run dropping past the threshold fires (and an intentional change re-floors).
    Mirrors tools/scorecard_baseline.json's shape."""
    return {
        "schema": BASELINE_SCHEMA,
        "commit": commit,
        "pinned": today,
        "baselines": {k: round(v["value"], 4) for k, v in sorted(current.items())},
    }


# ============================================================================
# I/O boundary — gh + the catalog/baseline files. Thin wrappers.
# ============================================================================
def gh_json(args: list[str], timeout: int = 60) -> Any:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def fetch_open_issues(limit: int) -> list[dict[str, Any]]:
    """Open issues carrying the bench-signal label — the dedup index (label-scoped,
    exact bound; cf. score_signal.fetch_open_issues)."""
    return gh_json([
        "issue", "list", "--state", "open", "--label", SIGNAL_LABEL,
        "--limit", str(limit), "--json", "number,title,body",
    ])


def ensure_labels() -> None:
    """Idempotently create the marker labels so `gh issue create` never fails."""
    wanted = [
        (SIGNAL_LABEL, "0e8a16",
         "Auto-filed by bench-signal (tools/bench_signal.py) from a benchmark "
         "throughput regression; needs a worker"),
        (PERF_LABEL, "fbca04", "Performance / throughput work"),
    ]
    for name, color, desc in wanted:
        try:
            proc = subprocess.run(
                ["gh", "label", "create", name, "--color", color,
                 "--description", desc, "--force"],
                capture_output=True, text=True, encoding="utf-8", timeout=30)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"warning: could not run `gh label create {name}`: {e}",
                  file=sys.stderr)
            continue
        if proc.returncode != 0:
            print(f"warning: could not ensure '{name}' label "
                  f"(issue creation may fail): {proc.stderr.strip()[:200]}",
                  file=sys.stderr)


def create_issue(issue: dict[str, Any]) -> str:
    args = ["issue", "create", "--title", issue["title"], "--body", issue["body"]]
    for lab in issue["labels"]:
        args += ["--label", lab]
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"gh issue create -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


def load_json_file(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return payload


def git_commit(root: Path) -> str:
    try:
        proc = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, encoding="utf-8", timeout=15)
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


# ============================================================================
# Driver.
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="bench-signal: a benchmark throughput regression -> a deduped issue.")
    ap.add_argument("--workspace", default=".", help="repo root. Default: cwd.")
    ap.add_argument("--catalog", default="", help=f"benchmark catalog JSON "
                    f"(default {CATALOG_PATH}).")
    ap.add_argument("--baseline", default="", help=f"pinned perf baseline JSON "
                    f"(default {BASELINE_PATH}).")
    ap.add_argument("--min-drop-pct", type=float,
                    help=f"relative drop %% below baseline to file "
                         f"(default {DEFAULTS['min_drop_pct']}).")
    ap.add_argument("--min-abs", type=float,
                    help=f"absolute tok/s drop floor (default {DEFAULTS['min_abs']}).")
    ap.add_argument("--max-issues", type=int,
                    help=f"hard cap on issues filed, worst-drop-first "
                         f"(default {DEFAULTS['max_issues']}).")
    ap.add_argument("--pin", action="store_true",
                    help="re-floor the baseline from the current catalog and exit.")
    ap.add_argument("--live", action="store_true",
                    help="actually create the issues (default: dry-run).")
    ap.add_argument("--json", action="store_true", help="machine-readable output.")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve()
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    catalog_path = Path(args.catalog) if args.catalog else root / CATALOG_PATH
    baseline_path = Path(args.baseline) if args.baseline else root / BASELINE_PATH
    min_drop_pct = args.min_drop_pct if args.min_drop_pct is not None else DEFAULTS["min_drop_pct"]
    min_abs = args.min_abs if args.min_abs is not None else DEFAULTS["min_abs"]
    max_issues = args.max_issues if args.max_issues is not None else DEFAULTS["max_issues"]

    try:
        catalog = load_json_file(catalog_path)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"refuse: could not load the benchmark catalog {catalog_path}: {e}",
              file=sys.stderr)
        return 2
    runs = catalog.get("runs") or []
    current = current_by_key(runs if isinstance(runs, list) else [])

    # --pin: re-floor the baseline from the current catalog and exit (no gh).
    if args.pin:
        baseline = build_baseline(current, git_commit(root), today)
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"bench-signal: pinned {len(baseline['baselines'])} benchmark baseline(s) "
              f"-> {baseline_path}")
        return 0

    try:
        baseline = load_json_file(baseline_path)
    except FileNotFoundError:
        print(f"bench-signal {today} — no pinned baseline at {baseline_path}; "
              f"run `python {BASELINE_PATH and 'tools/bench_signal.py'} --pin` first. "
              f"Nothing to signal.", file=sys.stderr if args.json else sys.stdout)
        if args.json:
            print(json.dumps({"schema": SCHEMA, "date": today, "mode": "dry-run",
                              "note": "no pinned baseline", "planned": []}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as e:
        print(f"refuse: could not load the pinned baseline {baseline_path}: {e}",
              file=sys.stderr)
        return 2

    regs = regressions(current, baseline, min_drop_pct=min_drop_pct, min_abs=min_abs)

    # The OPEN-issue dedup index is mandatory (same contract as the siblings).
    errors: list[str] = []
    try:
        issues = fetch_open_issues(DEFAULTS["issue_scan_limit"])
        open_keys = open_issue_keys(issues)
    except Exception as e:  # noqa: BLE001
        print(f"refuse: cannot fetch open issues for the dedup index ({e})",
              file=sys.stderr)
        return 2

    to_file, skip_stats = plan_issues(regs, open_keys, max_issues=max_issues, today=today)

    filed: list[dict[str, Any]] = []
    if args.live and to_file:
        ensure_labels()
        for issue in to_file:
            try:
                url = create_issue(issue)
            except Exception as e:  # noqa: BLE001
                errors.append(f"create[{issue['key']}]: {e}")
                continue
            filed.append({**issue, "issue_url": url})

    result = {
        "schema": SCHEMA,
        "date": today,
        "mode": "live" if args.live else "dry-run",
        "catalog_runs": len(runs) if isinstance(runs, list) else 0,
        "benchmarks_measured": len(current),
        "baseline_commit": baseline.get("commit", ""),
        "open_signal_issues": sorted(open_keys),
        "skipped": skip_stats,
        "planned": [
            {"title": i["title"], "key": i["key"], "drop_pct": round(i["drop_pct"], 2),
             "labels": i["labels"]}
            for i in to_file],
        "filed": [{"title": f["title"], "key": f["key"],
                   "issue_url": f.get("issue_url", "")} for f in filed],
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # Human report.
    print(f"bench-signal {today} — {result['mode']}  "
          f"({result['benchmarks_measured']} measured benchmark(s) of "
          f"{result['catalog_runs']} catalog runs; baseline @{result['baseline_commit'] or '?'})")
    sk = ", ".join(f"{k}={v}" for k, v in skip_stats.items() if v) or "none"
    print(f"  deduped/dropped: {sk}")
    if result["open_signal_issues"]:
        print(f"  already tracked (open): {', '.join(result['open_signal_issues'])}")
    if not to_file:
        print(f"  → no NEW actionable perf regression past -{min_drop_pct:.0f}% / "
              f"{min_abs:.1f} tok/s. Nothing to file.")
    else:
        verb = "FILED" if args.live else "would file"
        print(f"  → {verb} {len(to_file)} issue(s) (cap {max_issues}, "
              f"min-drop {min_drop_pct:.0f}% & {min_abs:.1f} tok/s):")
        for i in to_file:
            f = next((x for x in filed if x["key"] == i["key"]), None)
            mark = ""
            if args.live:
                mark = f"  {f['issue_url']}" if f else "  (create failed)"
            print(f"     [-{i['drop_pct']:>4.1f}%] {i['title']}{mark}")
    if errors:
        print("  errors:")
        for e in errors:
            print(f"     ! {e}")
    if not args.live and to_file:
        print("\n  dry-run — file these for real with:  "
              "python tools/bench_signal.py --live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
