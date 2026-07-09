#!/usr/bin/env python3
"""commit_quality_scorecard — score the fleet's own recent commit output.

The scorecard family measures every surface a reviewer cares about EXCEPT the one
the fleet produces most: its own commits. Commit quality WAS measured — but only by
two standalone, run-by-hand meters that never fold into the portfolio ratchet:

  * ``commit_subject_coverage.py`` — what fraction of recent ship subjects are
    witness-gradeable (``type(scope): <verb> …``; a noun-led subject ABSTAINs and an
    ABSTAIN on a landed commit is immutable — the dominant cause of closed-but-
    unwitnessed issues).
  * ``commit_stamp_doctor.py`` — what fraction carry a ``(fak <leaf>)`` ship-stamp
    the ``dos verify`` referee can BIND (and whether the stamped leaf is a real lane).

This card is the fold-of-folds: it COMPOSES those two pure cores (single source of
truth — it imports them, never re-implements the verb set or the trailer regex) into
one portfolio-grade ``commit_debt`` + A–F grade, so the agentic-dev output surface is
scored on every ``scorecard_control_pane`` pass — the unbounded scoring/rescoring loop.

It obeys the five scorecard laws:

  1. Deterministic — pure ``git log`` read over the last N commits; at a fixed HEAD
     two clones score identically. No clock, no network, no gh/dos calls.
  2. Cross-checked against reality, ungameable — the facts are the immutable committed
     subjects and the declared lanes in ``dos.toml``. You cannot lower ``commit_debt``
     by editing a data file; you lower it by writing gradeable, stamped commits.
  3. One headline ``commit_debt`` integer — the count of recent SHIP commits that a
     referee cannot bind (ABSTAIN-prone subject) or attribute (unstamped) or that
     stamp a phantom leaf (off-lane). Bookkeeping / merges / release bumps are exempt.
  4. A pure core, an impure shell — the KPIs are pure functions over the fetched
     ``(sha, subject)`` rows; a thin git shell gathers them.
  5. A control-pane payload — emits ``corpus.commit_debt`` / ``corpus.grade`` so the
     fold and any loop runner read it uniformly.

    python tools/commit_quality_scorecard.py                # the work-list
    python tools/commit_quality_scorecard.py --json
    python tools/commit_quality_scorecard.py --last 100
    python tools/commit_quality_scorecard.py --compare baseline.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-commit-quality-scorecard/1"
DEFAULT_LAST = 50


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _load_sibling(name: str):
    """Import a sibling tools/ module by path — single source of truth, no re-impl."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_subjcov = _load_sibling("commit_subject_coverage")
_stampdoc = _load_sibling("commit_stamp_doctor")


def grade_letter(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


# ---- fact gathering (impure shell) -----------------------------------------

CommitFetcher = Callable[[Path, int], list[tuple[str, str]]]


def recent_commits(root: Path, n: int) -> list[tuple[str, str]]:
    """The last n commits as [(short_sha, subject), …], newest first (merges kept —
    each KPI applies its own exemption). Empty on any git error (card degrades to a
    clean read rather than crashing the fold)."""
    try:
        proc = subprocess.run(
            ["git", "log", f"-{int(n)}", "--format=%h%x09%s"],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows: list[tuple[str, str]] = []
    for line in (proc.stdout or "").splitlines():
        if "\t" in line:
            sha, subj = line.split("\t", 1)
            if subj.strip():
                rows.append((sha.strip(), subj.strip()))
    return rows


def head_sha(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


# ---- KPIs (pure functions over the fetched rows) ---------------------------


def kpi_subject_gradeable(rows: list[tuple[str, str]]) -> dict[str, Any]:
    """Reuse commit_subject_coverage.coverage — fraction of ship subjects the DOS
    commit-audit witness can grade (a leading verb after ``type(scope):``). Each
    ABSTAIN-prone subject is one HARD defect; fix by leading the description with a
    recognized verb (never by weakening the verb set)."""
    by_subject = {subj: sha for sha, subj in reversed(rows)}  # newest wins on dup
    cov = _subjcov.coverage([subj for _, subj in rows])
    defects = []
    for a in cov.get("abstain_subjects", []):
        subj = a.get("subject", "")
        sha = by_subject.get(subj, "?")
        defects.append(f"{sha}: {subj[:70]} — {a.get('reason', 'non-gradeable subject')}")
    total = cov.get("total", 0)
    gradeable = cov.get("gradeable", 0)
    score = 100.0 if not total else round(100.0 * gradeable / total, 1)
    detail = (f"{gradeable}/{total} ship subjects witness-gradeable"
              if total else "no gradeable ship commits in window")
    return {"kpi": "subject-gradeable", "group": "witness", "score": score,
            "detail": detail, "defects": defects, "soft": []}


def kpi_ship_stamped(rows: list[tuple[str, str]]) -> dict[str, Any]:
    """Reuse commit_stamp_doctor.classify — fraction of SHIP commits carrying a
    ``(fak <leaf>)`` / ``fak/<leaf>:`` stamp the ``dos verify`` referee can bind.
    Bookkeeping (merges, snapshots, plan rollups) and release bumps are exempt (they
    name work as narrative, not a per-leaf ship). Each unstamped ship = one HARD
    defect; fix by stamping the leaf, never by relaxing the trailer grammar."""
    ship = 0
    stamped = 0
    defects = []
    for sha, subj in rows:
        kind = _stampdoc.classify(subj)
        if kind in ("bookkeeping", "release"):
            continue
        ship += 1
        if kind in ("stamped-trailer", "stamped-direct"):
            stamped += 1
        else:
            defects.append(f"{sha}: {subj[:70]} — no (fak <leaf>) ship-stamp")
    score = 100.0 if not ship else round(100.0 * stamped / ship, 1)
    detail = (f"{stamped}/{ship} ship commits carry a bindable stamp"
              if ship else "no ship commits in window")
    return {"kpi": "ship-stamped", "group": "witness", "score": score,
            "detail": detail, "defects": defects, "soft": []}


def kpi_stamp_on_lane(rows: list[tuple[str, str]], root: Path) -> dict[str, Any]:
    """A stamped leaf must bind to something the taxonomy knows: a declared lane in
    ``dos.toml`` OR a real ``cmd/<leaf>/`` demo dir. A leaf outside both (``(fak
    gatway)`` typo) binds ``dos verify fak <leaf>`` to a phantom nobody queries — the
    silent half of the trust gap. Each off-lane stamp is one HARD defect. When
    ``dos.toml`` is unreadable the recognized set is empty and this check is SKIPPED
    (scores 100), never failed — a parse error must not manufacture debt."""
    lanes = _stampdoc.declared_lanes(str(root))
    recognized = lanes | _stampdoc.cmd_demo_leaves(str(root))
    defects = []
    stamped_total = 0
    if lanes:
        for sha, subj in rows:
            if _stampdoc.classify(subj) in ("stamped-trailer", "stamped-direct"):
                stamped_total += 1
                leaf = _stampdoc.stamp_leaf(subj)
                if leaf and leaf not in recognized:
                    defects.append(f"{sha}: (fak {leaf}) binds no declared lane / cmd demo")
    score = 100.0 if not stamped_total else round(
        100.0 * (stamped_total - len(defects)) / stamped_total, 1)
    detail = (f"{stamped_total - len(defects)}/{stamped_total} stamps bind a real lane"
              if stamped_total else "no declared lanes to check (dos.toml)" if not lanes
              else "no stamped commits in window")
    return {"kpi": "stamp-on-lane", "group": "honesty", "score": score,
            "detail": detail, "defects": defects, "soft": []}


# KPI weights (sum to 1.0). The two bindability halves dominate; off-lane is a
# smaller honesty check that only fires when dos.toml is readable.
WEIGHTS = {"subject-gradeable": 0.40, "ship-stamped": 0.40, "stamp-on-lane": 0.20}


def build_payload(*, root: str, rows: list[tuple[str, str]], last: int,
                  sha: str) -> dict[str, Any]:
    rootp = Path(root)
    kpis = [
        kpi_subject_gradeable(rows),
        kpi_ship_stamped(rows),
        kpi_stamp_on_lane(rows, rootp),
    ]
    commit_debt = sum(len(k["defects"]) for k in kpis)
    composite = round(sum(WEIGHTS[k["kpi"]] * k["score"] for k in kpis), 1)
    grade = grade_letter(composite)
    ok = commit_debt == 0
    kpi_summ = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))
    if ok:
        verdict = "OK"
        finding = f"all {len(rows)} recent commits are referee-bindable"
        reason = (f"commit_debt 0; score {composite}/100 (grade {grade}); every ship "
                  f"subject is witness-gradeable and stamped")
        next_action = "none — keep leading subjects with a verb and stamping the leaf"
    else:
        verdict = "ACTION"
        worst = max(kpis, key=lambda k: len(k["defects"]))
        finding = f"{commit_debt} recent ship commit(s) a referee cannot bind"
        reason = (f"commit_debt {commit_debt}; score {composite}/100 (grade {grade}); "
                  f"heaviest KPI: {worst['kpi']} ({len(worst['defects'])} defect(s))")
        next_action = (f"fix the {worst['kpi']} defects — "
                       + ("lead the description with a recognized verb"
                          if worst["kpi"] == "subject-gradeable"
                          else "stamp the leaf `(fak <leaf>)`"
                          if worst["kpi"] == "ship-stamped"
                          else "correct the stamp leaf to a declared lane"))
    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": root,
        "commit": sha,
        "window": last,
        "scanned": len(rows),
        "corpus": {
            "score": composite,
            "grade": grade,
            "commit_debt": commit_debt,
        },
        "kpis": kpis,
        "kpi_summary": kpi_summ,
    }


def collect(root: Path, *, last: int,
            fetcher: CommitFetcher | None = None) -> dict[str, Any]:
    fetch = fetcher or recent_commits
    rows = fetch(root, last)
    return build_payload(root=str(root), rows=rows, last=last, sha=head_sha(root))


def render(p: dict[str, Any]) -> str:
    c = p.get("corpus", {})
    lines = [
        f"commit-quality: {p.get('verdict')} ({'ok' if p.get('ok') else 'ACTION'})  "
        f"score {c.get('score')}/100 grade {c.get('grade')} · "
        f"COMMIT-DEBT {c.get('commit_debt')} · window {p.get('window')} @ {p.get('commit')}",
        f"  {p.get('reason')}",
        "",
        f"  {'score':>5} {'debt':>4}  kpi              detail",
    ]
    for b in p.get("kpi_summary", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['kpi']:<16} {b['detail']}")
    worklist = []
    for k in p.get("kpis", []):
        worklist.extend(k["defects"])
    if worklist:
        lines.append("")
        lines.append("  work-list (bind these — verb-led subject / (fak <leaf>) stamp):")
        for d in worklist[:15]:
            lines.append(f"    - {d}")
        if len(worklist) > 15:
            lines.append(f"    … +{len(worklist) - 15} more")
    return "\n".join(lines)


def compare(cur: dict[str, Any], base_path: str) -> str:
    try:
        base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"compare: cannot read baseline {base_path}: {e}"
    b = int(base.get("corpus", {}).get("commit_debt", 0))
    c = int(cur.get("corpus", {}).get("commit_debt", 0))
    delta = c - b
    if delta < 0:
        verdict = "3x" if b and c * 3 <= b else "2x" if b and c * 2 <= b else "DOWN"
    elif delta == 0:
        verdict = "FLAT"
    else:
        verdict = "REGRESSED"
    return f"commit_debt {b} -> {c} (delta {delta:+d}) [{verdict}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score the fleet's own recent commit output (read-only, git-log).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--last", type=int, default=DEFAULT_LAST,
                    help=f"commits to scan (default {DEFAULT_LAST})")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the commit_debt delta vs a saved --json baseline")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(root, last=max(1, args.last))
    if args.compare:
        print(compare(payload, args.compare))
        return 0 if payload.get("ok") else 1
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
