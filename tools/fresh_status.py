#!/usr/bin/env python3
"""fresh_status.py — fold git + benchmarks + work + industry into ONE rollup.

The repo has rich PER-domain status tooling — ``release_status.py``,
``industry_scorecard.py``, ``plan_audit.py``, the 12-scorecard
``scorecard_control_pane.py``, and ``bench_catalog.py`` (``catalog.json``) — and
each emits a clean control-pane envelope (``schema/ok/verdict/finding/reason/
next_action``). What NOTHING does is fold the four TOP-LEVEL domains into one
"where do we stand right now?" view with git context at the center. Today that
answer takes 4-5 separate commands plus a hand-read of STATUS.md.

This is that fold — the cross-domain control pane. It is the sibling of
``scorecard_control_pane.py`` (which folds the scorecard FAMILY) one level up:
that one answers "is the repo getting better or worse on quality?"; this one
answers "what is the state of git, the benchmarks, the work, and our competitive
standing this minute?".

  python tools/fresh_status.py            # human rollup
  python tools/fresh_status.py --json      # machine control-pane payload
  python tools/fresh_status.py --check     # advisory gate: non-zero only on a
                                           # HARD failure or stale benchmarks
  python tools/fresh_status.py --write-doc # regenerate the committed snapshot doc

Four domain panes, each a PURE fold over data that already exists — never a
re-measure:

  * git        — HEAD sha + branch + dirty count + ahead/behind vs upstream + push
                 lag. The "crystal clear on git" center. A HARD pane (git is ground
                 truth). Push lag (age of the oldest unpushed commit) is the
                 keep-git-up-to-date velocity gate: past --push-lag-mins it trips
                 ACTION, so committed work that silently stopped reaching origin
                 fails --check instead of sitting unnoticed.
  * benchmarks — experiments/benchmark/catalog.json: run/machine counts, newest-run
                 staleness, and a per-benchmark provenance tag via the shared
                 bench_provenance classifier (measured | modeled | functional |
                 unknown). FUNCTIONAL separates correctness/agent-live/load-only
                 witnesses from throughput numbers; UNKNOWN is the fail-closed
                 residue, surfaced loudly and never silently treated as measured —
                 the same honesty discipline the check_provenance_labels.py gate
                 enforces. A HARD pane.
  * work       — tools/plan_audit.py --json (plans shipped vs remaining). A SOFT
                 pane: zero plans is a valid zero-state, not an error; if the plan
                 surface is absent it degrades to SKIP without tripping the rollup.
  * industry   — tools/industry_scorecard.py --json (parity_debt + grade vs SOTA).
                 A SOFT pane (the sub-tool can be slow / scan a data dir).

Verdict ladder (mirrors scorecard_control_pane.fold): a HARD pane that errors is
ACTION; benchmark staleness past the threshold is ACTION; otherwise OK. SOFT
panes that can't report degrade to SKIP and never trip the rollup. ``--check``
exits non-zero ONLY on ACTION — the same advisory contract as the scorecard
ratchet (debt may stay/fall, a gate fails only on a real regression).

Pure-stdlib Python, repo-root resolved like the other honesty gates, no network
by default. The benchmark/provenance/freshness math is split from the live
sub-tool runner so the fold is unit-testable (tools/fresh_status_test.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_provenance  # noqa: E402

SCHEMA = "fak-fresh-status/1"
CATALOG_REL = "experiments/benchmark/catalog.json"
AUTHORITY_REL = "BENCHMARK-AUTHORITY.md"

# A benchmark catalog whose newest run is older than this is STALE — the rollup's
# one freshness gate. Generous (the bench cadence is per-major-refresh, not
# per-commit); the point is to catch a catalog that quietly stopped being fed.
STALE_DAYS = 30

# Committed-but-unpushed work IS git falling behind: a push that silently failed
# leaves local commits piling up ahead of origin. Past this lag the git pane trips
# ACTION so the rollup/--check nudges a push — the "keep git up to date" velocity
# gate. Generous on purpose: in healthy operation work is pushed within minutes
# (ahead returns to 0 and there is no lag), so this only fires on a genuinely
# stalled or failed push, never on a normal in-flight commit.
DEFAULT_PUSH_LAG_ACTION_SECONDS = 45 * 60


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


# --- git helpers (the scorecard_control_pane pattern) ----------------------

def _git_line(args: list[str], root: Path) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    if p.returncode != 0:
        return ""
    return p.stdout.strip()


def head_commit(root: Path) -> str:
    return _git_line(["rev-parse", "--short", "HEAD"], root) or "unknown"


def git_pane(root: Path, *, now: datetime | None = None,
             push_lag_action_seconds: int = DEFAULT_PUSH_LAG_ACTION_SECONDS) -> dict[str, Any]:
    """HEAD sha + branch + dirty count + ahead/behind vs upstream + push lag — the center.

    A HARD pane: if ``git rev-parse`` cannot answer at all we report ERROR (the
    rollup has no ground truth). A missing UPSTREAM is NOT an error — a detached
    or local-only branch just reports ahead/behind (and push lag) as null.

    The push-lag dimension is the velocity signal. ahead/behind are STOCKS; "keep
    git up to date" is about TIME, so when local commits are AHEAD of upstream we
    also measure how long the OLDEST unpushed commit has been waiting to reach
    origin. Past ``push_lag_action_seconds`` the pane trips ACTION so the rollup
    and ``--check`` nudge a push. ``now`` is injectable so the lag is deterministic
    in tests.
    """
    now = now or datetime.now(timezone.utc)
    sha = _git_line(["rev-parse", "--short", "HEAD"], root)
    branch = _git_line(["rev-parse", "--abbrev-ref", "HEAD"], root)
    if not sha:
        return {
            "key": "git", "label": "git", "ok": False, "verdict": "ERROR",
            "reason": "git rev-parse HEAD failed — not a repo or git unavailable",
            "sha": None, "branch": None, "dirty": None, "ahead": None, "behind": None,
            "push_lag_seconds": None, "oldest_unpushed_ts": None,
        }
    porcelain = _git_line(["status", "--porcelain"], root)
    dirty = len([ln for ln in porcelain.splitlines() if ln.strip()]) if porcelain else 0
    ahead = behind = None
    counts = _git_line(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"], root)
    if counts:
        parts = counts.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            behind, ahead = int(parts[0]), int(parts[1])
    # Push lag: age of the OLDEST unpushed commit's committer date. Only meaningful
    # when an upstream exists AND we are ahead of it; otherwise there is nothing
    # waiting to be pushed and the lag is None (not zero — "no upstream" and "fully
    # pushed" both legitimately have no lag).
    push_lag_seconds: int | None = None
    oldest_unpushed_ts: int | None = None
    if ahead:
        cts = _git_line(["log", "--format=%ct", "@{upstream}..HEAD"], root)
        stamps = [int(s) for s in cts.split() if s.isdigit()] if cts else []
        if stamps:
            oldest_unpushed_ts = min(stamps)
            push_lag_seconds = max(0, int(now.timestamp()) - oldest_unpushed_ts)
    stale_push = push_lag_seconds is not None and push_lag_seconds > push_lag_action_seconds
    bits = [f"{sha} ({branch or 'detached'})"]
    bits.append(f"{dirty} dirty" if dirty else "clean tree")
    if ahead is not None:
        bits.append(f"+{ahead}/-{behind} vs upstream")
    if push_lag_seconds is not None:
        mins = push_lag_seconds // 60
        bits.append(f"{ahead} unpushed, oldest {mins}m old — push to origin"
                    if stale_push else f"{ahead} unpushed, oldest {mins}m")
    return {
        "key": "git", "label": "git", "ok": not stale_push,
        "verdict": "ACTION" if stale_push else "OK",
        "reason": ", ".join(bits),
        "sha": sha, "branch": branch or None, "dirty": dirty,
        "ahead": ahead, "behind": behind,
        "push_lag_seconds": push_lag_seconds, "oldest_unpushed_ts": oldest_unpushed_ts,
    }


# --- benchmark pane: pure folds over the catalog ----------------------------
# Provenance is classified by the shared, authority-grounded bench_provenance
# module (4-way: measured | modeled | functional | unknown), so the rollup, the
# catalog stamp, and any future consumer all read one verdict. See that module's
# docstring for the taxonomy and the priority ladder.


def _parse_run_ts(ts: str) -> datetime | None:
    """Parse a catalog run timestamp (ISO ``2026-06-25T05:00:17Z`` or the compact
    ``20260625T050017Z`` form the bench harness emits) to an aware UTC datetime."""
    if not ts:
        return None
    raw = ts.strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # tolerant ISO fallback (handles fractional seconds / offsets)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fold_benchmarks(catalog: dict[str, Any] | None, *, now: datetime,
                    stale_days: int = STALE_DAYS) -> dict[str, Any]:
    """Fold catalog.json into the benchmark pane (pure — no I/O).

    Counts runs/machines, finds the newest run, computes staleness, and tags each
    run measured/modeled/functional/unknown via the shared bench_provenance
    classifier (catalog `provenance` stamp first, else tags/run_id, fail-closed to
    unknown). A malformed/absent catalog is a HARD ERROR (the benchmark surface is
    a load-bearing claim).
    """
    empty_prov = {t: 0 for t in bench_provenance.TAGS}
    if not isinstance(catalog, dict):
        return {
            "key": "benchmarks", "label": "benchmarks", "ok": False, "verdict": "ERROR",
            "reason": "catalog.json missing or unreadable — no benchmark rollup",
            "runs": None, "machines": None, "newest": None, "age_days": None,
            "provenance": empty_prov,
        }
    runs = catalog.get("runs") or []
    machines = catalog.get("machines") or {}
    prov = bench_provenance.classify_all(runs)
    newest_dt: datetime | None = None
    newest_ts = ""
    for r in runs:
        dt = _parse_run_ts(str(r.get("timestamp") or ""))
        if dt and (newest_dt is None or dt > newest_dt):
            newest_dt, newest_ts = dt, str(r.get("timestamp"))
    age_days = None
    stale = False
    if newest_dt is not None:
        age_days = round((now - newest_dt).total_seconds() / 86400.0, 1)
        stale = age_days > stale_days
    n_runs, n_machines = len(runs), len(machines)
    prov_line = bench_provenance.summary_line(prov)
    if not runs:
        return {
            "key": "benchmarks", "label": "benchmarks", "ok": False, "verdict": "ACTION",
            "reason": f"catalog has 0 runs across {n_machines} machine(s) — nothing benchmarked",
            "runs": 0, "machines": n_machines, "newest": None, "age_days": None,
            "stale": False, "provenance": prov,
        }
    verdict = "ACTION" if stale else "OK"
    ok = not stale
    reason = (f"{n_runs} runs / {n_machines} machines; newest {newest_ts} "
              f"({age_days}d ago); {prov_line}")
    if stale:
        reason += f" — STALE (>{stale_days}d since newest run; catalog may have stopped being fed)"
    return {
        "key": "benchmarks", "label": "benchmarks", "ok": ok, "verdict": verdict,
        "reason": reason, "runs": n_runs, "machines": n_machines,
        "newest": newest_ts or None, "age_days": age_days, "stale": stale,
        "provenance": prov,
    }


# --- work + industry panes: SOFT folds over sub-tool payloads ----------------

def fold_work(plan_payload: dict[str, Any] | None, error: str = "") -> dict[str, Any]:
    """Fold plan_audit.py --json into the work pane. SOFT: zero plans / an absent
    plan surface degrade to SKIP, never an error that trips the rollup."""
    if error or not isinstance(plan_payload, dict):
        return {
            "key": "work", "label": "work", "ok": True, "verdict": "SKIP",
            "reason": f"no plan surface ({error or 'plan_audit unavailable'}); "
                      "work pane skipped (not a failure)",
            "total_plans": None, "shipped": None, "remaining": None,
        }
    counts = plan_payload.get("counts") or {}
    total = counts.get("total_plans")
    shipped = counts.get("shipped") if isinstance(counts.get("shipped"), int) else None
    remaining = counts.get("remaining") if isinstance(counts.get("remaining"), int) else None
    if not isinstance(total, int) or total == 0:
        return {
            "key": "work", "label": "work", "ok": True, "verdict": "SKIP",
            "reason": "0 phased plans tracked in this clone (valid zero-state)",
            "total_plans": total if isinstance(total, int) else 0,
            "shipped": shipped, "remaining": remaining,
        }
    parts = [f"{total} plans"]
    if shipped is not None:
        parts.append(f"{shipped} shipped")
    if remaining is not None:
        parts.append(f"{remaining} remaining")
    return {
        "key": "work", "label": "work", "ok": True, "verdict": "OK",
        "reason": ", ".join(parts), "total_plans": total,
        "shipped": shipped, "remaining": remaining,
    }


def fold_industry(ind_payload: dict[str, Any] | None, error: str = "") -> dict[str, Any]:
    """Fold industry_scorecard.py --json into the industry pane. SOFT: an absent /
    failed sub-tool degrades to SKIP. ``parity_debt`` lives under ``corpus``."""
    if error or not isinstance(ind_payload, dict):
        return {
            "key": "industry", "label": "industry", "ok": True, "verdict": "SKIP",
            "reason": f"industry scorecard unavailable ({error or 'no payload'}); skipped",
            "parity_debt": None, "grade": None,
        }
    corpus = ind_payload.get("corpus") or {}
    debt = corpus.get("parity_debt")
    grade = corpus.get("grade")
    if not isinstance(debt, int):
        return {
            "key": "industry", "label": "industry", "ok": True, "verdict": "SKIP",
            "reason": "industry scorecard reported no parity_debt; skipped",
            "parity_debt": None, "grade": grade if isinstance(grade, str) else None,
        }
    return {
        "key": "industry", "label": "industry", "ok": True, "verdict": "OK",
        "reason": f"parity-debt {debt} vs SOTA, grade {grade or '?'}",
        "parity_debt": debt, "grade": grade if isinstance(grade, str) else None,
    }


# --- the fold: four panes -> one control-pane payload -----------------------

def fold(panes: list[dict[str, Any]], *, workspace: str, commit: str,
         generated_at: str) -> dict[str, Any]:
    """Fold the domain panes into one control-pane payload + verdict.

    HARD failures (a pane with verdict ERROR or a benchmark ACTION) trip the
    rollup to ACTION; SOFT SKIPs never do. The verdict ladder mirrors
    scorecard_control_pane.fold so a loop runner reads the same envelope.
    """
    {p["key"]: p for p in panes}
    actionable = [p for p in panes if p.get("verdict") in ("ERROR", "ACTION")]
    skipped = [p for p in panes if p.get("verdict") == "SKIP"]

    if actionable:
        ok, verdict, finding = False, "ACTION", "needs_attention"
        reason = "; ".join(f"{p['label']}: {p['reason']}" for p in actionable)
        # Point next_action at the heaviest concern: an unreadable git/catalog
        # before a staleness, before anything else.
        first = actionable[0]
        if first["key"] == "git" and first.get("verdict") == "ERROR":
            next_action = "fix the git context (not a repo / git unavailable) before trusting any other pane"
        elif first["key"] == "git":
            mins = (first.get("push_lag_seconds") or 0) // 60
            next_action = (f"push to origin — {first.get('ahead')} commit(s) have been unpushed "
                           f"for {mins}m; committed work is not reaching the remote")
        elif first["key"] == "benchmarks" and first.get("stale"):
            next_action = ("refresh the benchmark catalog — run the relevant bench + "
                           "`python tools/bench_catalog.py build`")
        elif first["key"] == "benchmarks":
            next_action = "populate experiments/benchmark/catalog.json (no runs registered)"
        else:
            next_action = f"resolve the {first['label']} pane: {first['reason']}"
    else:
        ok, verdict, finding = True, "OK", "all_green"
        live = [p for p in panes if p.get("verdict") == "OK"]
        reason = "; ".join(f"{p['label']}: {p['reason']}" for p in live)
        if skipped:
            reason += f" ({len(skipped)} pane(s) skipped: " \
                      + ", ".join(p["label"] for p in skipped) + ")"
        next_action = "rollup is green; nothing required"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "commit": commit,
        "generated_at": generated_at,
        "panes": {p["key"]: p for p in panes},
        "pane_order": [p["key"] for p in panes],
    }


# --- live runner ------------------------------------------------------------

def run_tool(root: Path, script: str, *args: str, python: str, timeout: int
             ) -> tuple[dict[str, Any] | None, str]:
    """Run ``tools/<script> --json`` and parse its JSON. ``(payload|None, error)``."""
    script_path = root / "tools" / script
    if not script_path.exists():
        return None, f"missing tool: tools/{script}"
    try:
        proc = subprocess.run(
            [python, str(script_path), *args],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    try:
        return json.loads(proc.stdout), ""
    except ValueError:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        return None, f"non-JSON output (exit {proc.returncode}): {tail[0][:160]}"


def load_catalog(root: Path) -> dict[str, Any] | None:
    try:
        return json.loads((root / CATALOG_REL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def enrich_catalog_engines(root: Path, catalog: dict[str, Any] | None) -> dict[str, Any] | None:
    """Stamp each run's local artifact engine fields onto ``artifact_engines``.

    The catalog's per-run ``provenance`` stamp is written on a bench node and can be
    clobbered when a peer rebuilds the (shared) catalog from a clone with different
    local artifacts. So the rollup does NOT depend on that stamp surviving: when a
    run's artifact dir IS local here, we read its engine fields directly and let the
    classifier apply the highest-trust signal (Rule 1) at fold time. A run whose
    artifacts are gitignored/remote keeps whatever stamp the catalog carries (or
    falls back to tags/run_id). Mutates a shallow copy; the on-disk catalog is
    untouched (this tool is read-only).
    """
    if not isinstance(catalog, dict):
        return catalog
    out = dict(catalog)
    runs = []
    for r in catalog.get("runs") or []:
        r = dict(r)
        rel = str(r.get("path") or "").replace("\\", "/")
        run_dir = (root / rel) if rel else None
        if run_dir and run_dir.is_dir():
            engines = []
            for jf in sorted(run_dir.glob("*.json")):
                try:
                    m = json.loads(jf.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(m, dict):
                    eng = m.get("engine") or m.get("generated_by")
                    if isinstance(eng, str) and eng:
                        engines.append(eng)
            if engines:
                r["artifact_engines"] = engines
        runs.append(r)
    out["runs"] = runs
    return out


def collect(root: Path, *, python: str = "", timeout: int = 60,
            now: datetime | None = None,
            push_lag_action_seconds: int = DEFAULT_PUSH_LAG_ACTION_SECONDS
            ) -> list[dict[str, Any]]:
    """Collect all four domain panes from the live tree (git + 1 file read + 2
    sub-tools). ``now`` is injectable so freshness is deterministic in tests."""
    python = python or sys.executable
    now = now or datetime.now(timezone.utc)
    git = git_pane(root, now=now, push_lag_action_seconds=push_lag_action_seconds)
    bench = fold_benchmarks(enrich_catalog_engines(root, load_catalog(root)), now=now)
    plan_payload, plan_err = run_tool(root, "plan_audit.py", "--json", python=python, timeout=timeout)
    work = fold_work(plan_payload, plan_err)
    ind_payload, ind_err = run_tool(root, "industry_scorecard.py", "--json", python=python, timeout=timeout)
    industry = fold_industry(ind_payload, ind_err)
    return [git, bench, work, industry]


# --- render + doc -----------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    lines = [
        f"fresh status — {payload['verdict']} ({payload['finding']})  @{payload['commit']}",
        f"  generated {payload['generated_at']}",
        "",
    ]
    for key in payload["pane_order"]:
        p = payload["panes"][key]
        mark = {"OK": "✓", "SKIP": "·", "ACTION": "✗", "ERROR": "✗"}.get(p["verdict"], "?")
        lines.append(f"  {mark} {p['label']:<11} {p['reason']}")
    lines.extend(["", f"  → {payload['next_action']}"])
    return "\n".join(lines)


def render_doc(payload: dict[str, Any], *, date: str) -> str:
    """The committed snapshot note body (docs/notes/FRESH-STATUS-<date>.md)."""
    g = payload["panes"].get("git", {})
    b = payload["panes"].get("benchmarks", {})
    prov = b.get("provenance", {}) if isinstance(b, dict) else {}
    rows = []
    for key in payload["pane_order"]:
        p = payload["panes"][key]
        rows.append(f"| {p['label']} | {p['verdict']} | {p['reason']} |")
    lines = [
        f"# Fresh status snapshot ({date})",
        "",
        "> Regenerated by `python tools/fresh_status.py --write-doc`. This is a",
        "> committed front door for the cross-domain rollup that folds git +",
        "> benchmarks + work + industry into one control-pane payload. Re-run the",
        "> tool for the live state; this note is the last pinned snapshot.",
        "",
        f"**Overall:** {payload['verdict']} ({payload['finding']}) — {payload['next_action']}",
        "",
        f"- **git:** HEAD `{g.get('sha')}` on `{g.get('branch')}`, "
        f"{g.get('dirty')} dirty file(s)"
        + (f", +{g.get('ahead')}/-{g.get('behind')} vs upstream"
           if g.get('ahead') is not None else ""),
        f"- **benchmarks:** {b.get('runs')} runs / {b.get('machines')} machines; "
        f"newest {b.get('newest')} ({b.get('age_days')}d ago); provenance "
        f"{bench_provenance.summary_line(prov)}",
        "",
        "## Panes",
        "",
        "| Pane | Verdict | Detail |",
        "|---|---|---|",
        *rows,
        "",
        "_Provenance discipline (`tools/bench_provenance.py`, authority-grounded +"
        " adversarially verified): **measured** = a real wall-clock; **modeled** = a"
        " closed-form work floor; **functional** = a correctness / agent-live /"
        " load-only witness that is NOT a throughput number; **unknown** = the"
        " fail-closed residue, surfaced loudly and never silently counted as"
        " measured — the same honesty floor `tools/check_provenance_labels.py`"
        " enforces._",
        "",
    ]
    return "\n".join(lines)


# --- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cross-domain fresh-status rollup (git + benchmarks + work + industry).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--check", action="store_true",
                    help="advisory gate: exit non-zero only on a HARD pane failure or stale benchmarks")
    ap.add_argument("--write-doc", action="store_true",
                    help="regenerate the committed snapshot note under docs/notes/")
    ap.add_argument("--date", default="", help="snapshot date YYYY-MM-DD for --write-doc (default: today UTC)")
    ap.add_argument("--timeout", type=int, default=60, help="per-sub-tool timeout seconds")
    ap.add_argument("--push-lag-mins", type=int, default=DEFAULT_PUSH_LAG_ACTION_SECONDS // 60,
                    help="trip the git pane to ACTION when the oldest unpushed commit is older "
                         "than this many minutes (the keep-git-up-to-date velocity gate)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    now = datetime.now(timezone.utc)
    panes = collect(root, timeout=args.timeout, now=now,
                    push_lag_action_seconds=max(0, args.push_lag_mins) * 60)
    payload = fold(panes, workspace=str(root), commit=head_commit(root),
                   generated_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"))

    if args.write_doc:
        date = args.date or now.strftime("%Y-%m-%d")
        doc_path = root / "docs" / "notes" / f"FRESH-STATUS-{date}.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(render_doc(payload, date=date), encoding="utf-8")
        if not args.json:
            print(f"wrote snapshot -> {doc_path.relative_to(root)}")

    if args.check:
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(render(payload))
        # Advisory: non-zero ONLY on ACTION (a HARD pane failure or stale bench).
        return 0 if payload["verdict"] != "ACTION" else 1

    if args.json:
        print(json.dumps(payload, indent=2))
    elif not args.write_doc:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
