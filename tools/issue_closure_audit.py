#!/usr/bin/env python3
"""DOS-witnessed GitHub-ticket closure auditor — the loop's checking layer.

The RSI supervisor stack measures *plan units* (``dos_supervisor_canary_audit``
folds plan ``done_units``). Nothing measured whether an *issue* actually got
**truly resolved**: an issue can be closed with no witnessed commit (a CLAIM),
or a doc commit can merely *mention* ``#N`` while the issue stays open. Without
this layer "the supervisor resolves tickets truly" is unfalsifiable, and the
loop can reward-hack its improvement metric — close issues / grow the queue
while real closure sits at zero (the failure ``.claude/rsi-loop-dod.md`` forbids:
*every acceleration layer has a matching checking layer*).

This auditor binds each issue to its resolving commit(s) and grades them through
the **DOS witness rung** (``dos commit-audit``), not self-report. An issue is
``TRUE_RESOLVED`` only if it is closed AND a commit that *claims* to resolve it
passes ``dos commit-audit`` with a diff-witness — the change actually did the
kind of thing claimed. Everything else closed is ``CLAIMED_CLOSED`` (the honest
gap). ``closure_rate`` is therefore a number from a witness the loop did not
author, exactly mirroring the ``dos improve`` keep-bit discipline.

Read-only. It never edits, labels, or closes an issue. It folds three
read-back surfaces the running process did not author: ``gh issue list``, the
repo's own ``git log``, and ``dos commit-audit``.

IMPORTANT: run from the repo ROOT. ``dos`` resolves its lane taxonomy from the
nearest ``dos.toml``; from ``fak/`` it scaffolds a throwaway generic config with
the wrong lanes. ``collect`` always resolves ``--workspace`` to the repo root.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fleet-issue-closure-audit/1"

# git log record/field separators (unit/record sep — safe inside commit prose).
_FIELD = "\x1f"
_RECORD = "\x1e"
_GIT_PRETTY = f"--pretty=format:%H{_FIELD}%s{_FIELD}%b{_RECORD}"

# A resolving reference: the GitHub-closing verbs in front of #N. We deliberately
# DO NOT trust GitHub's own closedByPullRequestsReferences here — in this repo it
# is always empty (no PR-keyword workflow), so the binding must be reconstructed
# from the commit text the repo actually writes.
_RESOLVE_RE = re.compile(r"\b(?:close|fixe?|resolve)[sd]?\s+#(\d+)\b", re.IGNORECASE)
# Any issue reference: '#' + digits at a word boundary. The leading boundary
# excludes token noise like 'xoxb-...#118' embedded in a longer non-word run only
# when it is genuinely glued; a normal 'Relates to #118' still matches (and is
# classified MENTION because it is not a _RESOLVE_RE hit and not in the subject).
_REF_RE = re.compile(r"(?<![\w-])#(\d+)\b")
# House close convention (NOUN form): '(issue #N)' / '(issues #N, #M)'. The repo
# writes this in commit bodies when a change resolves a ticket (e.g. efc0e78
# '(issue #501)', e60b92e '(issues #558, #565)'). _RESOLVE_RE only catches the
# VERB form (close/fix/resolve #N), so without this rung a genuinely-witnessed
# close written in the noun form is mis-bucketed CLAIMED_CLOSED — or a shipped fix
# whose issue is still OPEN is missed as OPEN_WITNESSED. The literal 'issue(s)'
# anchor before the #N list keeps it TIGHT: a bare prose '#499' / 'the #500 gap'
# is NOT bound (those stay MENTION), so the witness is never loosened — a
# newly-bound commit still has to pass `dos commit-audit` to count as resolved.
_ISSUE_NOUN_RE = re.compile(r"\bissues?\b[\s:]*((?:#\d+[\s,]*(?:and\s+)?)+)", re.IGNORECASE)

RESOLVING = "resolving"
MENTION = "mention"

# Witness rung: a commit "truly resolves" only with a diff-witness.
_WITNESS_OK = "diff-witnessed"
_VERDICT_OK = "OK"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def run_text(cmd: list[str], cwd: Path, *, timeout: int = 60) -> dict[str, Any]:
    """Run a command, return {stdout, stderr, returncode} (no JSON parse).

    Force UTF-8 decode with replacement: git/dos emit non-ASCII commit prose
    (em dashes, arrows) that the Windows default cp1252 codec cannot decode,
    which otherwise crashes the subprocess reader thread mid-audit.
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": 1, "_error": str(exc)}
    return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


def read_json_from_text(text: str) -> dict[str, Any]:
    """Mirror of dos_supervisor_status.read_json_from_text — tolerant JSON scrape."""
    text = (text or "").strip()
    if text:
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            return obj
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


# ---------------------------------------------------------------------------
# Pure binding parser: git log text -> {issue_number: [CommitRef]}
# ---------------------------------------------------------------------------

def classify_refs(subject: str, body: str) -> dict[int, str]:
    """Classify every #N in one commit as RESOLVING or MENTION.

    RESOLVING wins if ANY of: the GitHub-closing VERB form (close/fix/resolve #N,
    in subject or body) matches; the repo's house NOUN form ('issue #N' / 'issues
    #N, #M') matches; or #N appears in the subject line. Otherwise the reference
    is a MENTION (Relates-to, bare prose/bullet body refs) and never counts toward
    TRUE_RESOLVED. Binding tighter never loosens the witness — a newly-bound commit
    still has to pass `dos commit-audit` to grade TRUE_RESOLVED / OPEN_WITNESSED.
    """
    subject = subject or ""
    body = body or ""
    text = subject + "\n" + body
    out: dict[int, str] = {}

    subject_refs = {int(m) for m in _REF_RE.findall(subject)}
    resolve_refs = {int(m) for m in _RESOLVE_RE.findall(text)}
    noun_refs: set[int] = set()
    for run in _ISSUE_NOUN_RE.findall(text):
        noun_refs.update(int(m) for m in _REF_RE.findall(run))
    resolving = resolve_refs | subject_refs | noun_refs

    for n in _REF_RE.findall(text):
        num = int(n)
        if num in resolving:
            out[num] = RESOLVING
        else:
            out.setdefault(num, MENTION)
    # A resolving classification always wins over a mention for the same issue.
    for num in resolving:
        out[num] = RESOLVING
    return out


def parse_git_log(log_text: str) -> dict[int, list[dict[str, Any]]]:
    """Parse the separator-delimited git log into issue -> [commit ref] map."""
    refs: dict[int, list[dict[str, Any]]] = {}
    for record in (log_text or "").split(_RECORD):
        record = record.strip("\n")
        if not record.strip():
            continue
        parts = record.split(_FIELD)
        if len(parts) < 2:
            continue
        sha = parts[0].strip()
        subject = parts[1] if len(parts) > 1 else ""
        body = parts[2] if len(parts) > 2 else ""
        for num, kind in classify_refs(subject, body).items():
            refs.setdefault(num, []).append({"sha": sha, "subject": subject.strip(), "kind": kind})
    return refs


def git_issue_refs(workspace: Path, max_commits: int) -> dict[int, list[dict[str, Any]]]:
    res = run_text(["git", "log", f"-{int(max_commits)}", _GIT_PRETTY], workspace)
    return parse_git_log(res.get("stdout", ""))


def git_total_commits(workspace: Path) -> int | None:
    """Total commits reachable from HEAD, or None if git can't answer.

    Used to detect when the --max-commits log window is *narrower* than the
    repo's history: a closed issue whose only resolving commit fell outside the
    window is unbindable and would grade CLAIMED for the wrong reason. The
    coverage block surfaces that instead of silently under-binding.
    """
    res = run_text(["git", "rev-list", "--count", "HEAD"], workspace)
    text = (res.get("stdout") or "").strip()
    if res.get("returncode") != 0 or not text.isdigit():
        return None
    return int(text)


# ---------------------------------------------------------------------------
# Injectable I/O seams (real implementations; tests pass fakes)
# ---------------------------------------------------------------------------

IssueFetcher = Callable[[Path], list[dict[str, Any]]]
CommitAuditor = Callable[[str, Path], dict[str, Any]]


def fetch_issues(workspace: Path, *, limit: int = 1000) -> list[dict[str, Any]]:
    res = run_text(
        [
            "gh", "issue", "list", "--state", "all", "--limit", str(limit),
            "--json", "number,state,stateReason,closedAt,title,labels",
        ],
        workspace,
    )
    text = (res.get("stdout") or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def _first_audit_record(text: str) -> dict[str, Any]:
    """`dos commit-audit --json` emits a JSON ARRAY (one record per commit).

    The ref is positional and single, so we want the first/only element. Fall
    back to the tolerant object scrape if the CLI ever emits a bare object.
    """
    text = (text or "").strip()
    if text:
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        if isinstance(obj, dict):
            return obj
    return read_json_from_text(text)


def audit_commit(sha: str, workspace: Path) -> dict[str, Any]:
    # The ref is POSITIONAL (`dos commit-audit <ref>`), not `--ref` (that is the
    # MCP parameter name; the CLI rejects it with exit 2 / empty stdout).
    payload = _first_audit_record(
        run_text(["dos", "commit-audit", sha, "--workspace", str(workspace), "--json"], workspace)["stdout"]
    )
    return {
        "sha": sha,
        "verdict": payload.get("verdict"),
        "witness": payload.get("witness"),
        "claim_kind": payload.get("claim_kind"),
    }


def commit_is_witnessed(audit: dict[str, Any]) -> bool:
    return str(audit.get("verdict")) == _VERDICT_OK and str(audit.get("witness")) == _WITNESS_OK


# ---------------------------------------------------------------------------
# The grader (pure): issues + refs + audits -> the standard payload
# ---------------------------------------------------------------------------

TRUE_RESOLVED = "TRUE_RESOLVED"
CLAIMED_CLOSED = "CLAIMED_CLOSED"
CLOSED_NOT_PLANNED = "CLOSED_NOT_PLANNED"
OPEN_WITNESSED = "OPEN_WITNESSED"
OPEN = "OPEN"


def _is_closed(issue: dict[str, Any]) -> bool:
    return str(issue.get("state", "")).upper() == "CLOSED"


def _state_reason(issue: dict[str, Any]) -> str:
    return str(issue.get("stateReason") or "").upper()


def grade_issue(
    issue: dict[str, Any],
    issue_refs: list[dict[str, Any]],
    audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Grade one issue into exactly one bucket using the witness rung."""
    resolving = [r for r in issue_refs if r.get("kind") == RESOLVING]
    witnessed = [
        r for r in resolving if commit_is_witnessed(audits.get(r["sha"], {}))
    ]
    closed = _is_closed(issue)
    reason = _state_reason(issue)

    if closed and reason == "NOT_PLANNED":
        bucket = CLOSED_NOT_PLANNED
    elif closed and witnessed:
        bucket = TRUE_RESOLVED
    elif closed:
        bucket = CLAIMED_CLOSED
    elif witnessed:
        bucket = OPEN_WITNESSED
    else:
        bucket = OPEN

    return {
        "number": issue.get("number"),
        "title": str(issue.get("title") or "")[:80],
        "state": str(issue.get("state") or ""),
        "state_reason": issue.get("stateReason") or "",
        "bucket": bucket,
        "resolving_commits": [r["sha"][:7] for r in resolving],
        "witnessed_commits": [r["sha"][:7] for r in witnessed],
        "mentions": [r["sha"][:7] for r in issue_refs if r.get("kind") == MENTION],
    }


def compute_coverage(
    *,
    issues_fetched: int,
    issue_limit: int,
    commits_scanned: int,
    max_commits: int,
    total_commits: int | None,
) -> dict[str, Any]:
    """Detect whether either load surface hit its cap (so closures may be unseen).

    `gh issue list` returns newest-first, so a fetch that returned exactly the
    limit almost certainly dropped older issues — and the oldest issues are
    disproportionately the *closed* ones, the population this auditor grades.
    Likewise a git-log window narrower than the repo's history can fail to bind
    a closed issue's resolving commit. Either case makes `closure_rate` a number
    over a *slice*, not the backlog — so we surface it loudly rather than letting
    `issues_audited` read as comprehensive.
    """
    issues_truncated = issues_fetched >= issue_limit
    commits_truncated = (
        total_commits is not None and total_commits > commits_scanned
    ) or (total_commits is None and commits_scanned >= max_commits)
    notes: list[str] = []
    if issues_truncated:
        notes.append(
            f"gh fetch returned {issues_fetched} issue(s) = the --issue-limit cap; "
            f"older issues (disproportionately the closed ones) may be unseen — "
            f"raise --issue-limit"
        )
    if commits_truncated:
        scanned = min(commits_scanned, total_commits) if total_commits else commits_scanned
        notes.append(
            f"git-log window scanned {scanned} of "
            f"{total_commits if total_commits is not None else '?'} commit(s); a "
            f"resolving commit older than the window can't bind — raise --max-commits"
        )
    return {
        "complete": not (issues_truncated or commits_truncated),
        "issues_truncated": issues_truncated,
        "commits_truncated": commits_truncated,
        "issues_fetched": issues_fetched,
        "issue_limit": issue_limit,
        "commits_total": total_commits,
        "commits_window": max_commits,
        "notes": notes,
    }


def build_payload(
    *,
    workspace: str,
    issues: list[dict[str, Any]],
    refs: dict[int, list[dict[str, Any]]],
    audits: dict[str, dict[str, Any]],
    audit_error: str | None = None,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graded = [grade_issue(i, refs.get(int(i.get("number") or 0), []), audits) for i in issues]
    counts: dict[str, int] = {
        b: 0 for b in (TRUE_RESOLVED, CLAIMED_CLOSED, CLOSED_NOT_PLANNED, OPEN_WITNESSED, OPEN)
    }
    for g in graded:
        counts[g["bucket"]] = counts.get(g["bucket"], 0) + 1

    true_resolved = counts[TRUE_RESOLVED]
    claimed = counts[CLAIMED_CLOSED]
    denom = true_resolved + claimed
    closure_rate = round(true_resolved / denom, 4) if denom else None

    coverage = coverage or {"complete": True, "notes": []}
    incomplete = not coverage.get("complete", True)
    coverage_note = "; ".join(coverage.get("notes") or [])

    # `ok` is checked FIRST by fleet_control_pane.classify_loop_status and
    # short-circuits the verdict string, so it must carry the real signal:
    # the audit ran, COVERED the backlog, AND no closed ticket lacks a witness.
    if audit_error:
        ok = False
        verdict = "AUDIT_ERROR"
        finding = "tooling_error"
        reason = audit_error
        next_action = "fix the gh/git/dos read-back error, then re-run the closure audit"
    elif claimed > 0:
        # A real closure-honesty gap is the loudest signal; report it even if
        # coverage was also incomplete (the claimed set can only grow with more
        # coverage, never shrink), but flag the partial coverage in the reason.
        ok = False
        verdict = "ACTION"
        finding = "claimed_closed"
        open_witnessed = counts[OPEN_WITNESSED]
        reason = (
            f"{claimed} issue(s) are CLOSED with no diff-witnessed resolving commit "
            f"(closure_rate={closure_rate})"
            + (f"; {open_witnessed} OPEN_WITNESSED issue(s) are closable now (fix shipped + witnessed)"
               if open_witnessed else "")
            + (f"; NOTE partial coverage — {coverage_note}" if incomplete else "")
        )
        next_action = (
            "investigate the CLAIMED_CLOSED issues: each was closed without a DOS-witnessed "
            "commit — reopen, bind a (fak <leaf>) commit, or confirm it was a non-code close"
            + ("; and close the OPEN_WITNESSED issues whose fix already shipped"
               if open_witnessed else "")
        )
    elif incomplete:
        # No claimed gap *in what we saw* — but we did not see the whole backlog,
        # so "all closures witnessed" would be a claim over a slice. Refuse to
        # report OK on partial coverage; surface the load truncation as the action.
        ok = False
        verdict = "ACTION"
        finding = "incomplete_coverage"
        reason = (
            f"closure audit did not cover the full backlog, so closure_rate="
            f"{closure_rate} is over a slice, not all closed issues — {coverage_note}"
        )
        next_action = (
            "re-run with a higher --issue-limit / --max-commits (the wired loop now defaults "
            "above the current repo size) so every closed issue and its resolving commit load"
        )
    elif counts[OPEN_WITNESSED] > 0:
        ok = True
        verdict = "OK"
        finding = "shipped_but_open"
        reason = (
            f"{counts[OPEN_WITNESSED]} issue(s) have a diff-witnessed resolving commit but are "
            f"still OPEN; closure_rate={closure_rate}"
        )
        next_action = "close the OPEN_WITNESSED issues — the fix already shipped and is witnessed"
    else:
        ok = True
        verdict = "OK"
        finding = "closures_witnessed"
        reason = (
            f"all closed issues are witnessed or non-code closes; "
            f"true_resolved={true_resolved}, closure_rate={closure_rate}"
        )
        next_action = "no closure-honesty action needed; re-run after the next supervisor tick"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "closure_rate": closure_rate,
        "regression_rate": None,  # honest placeholder: needs the verdict-journal trail a live cycle writes
        "coverage": coverage,
        "counts": counts,
        "totals": {
            "issues_audited": len(graded),
            "closed_audited": true_resolved + claimed + counts[CLOSED_NOT_PLANNED],
        },
        "issues": sorted(graded, key=_issue_sort_key),
    }


def _issue_sort_key(g: dict[str, Any]) -> tuple[int, int]:
    # Surface the actionable buckets first; then by issue number desc (recent).
    order = {CLAIMED_CLOSED: 0, OPEN_WITNESSED: 1, TRUE_RESOLVED: 2, OPEN: 3, CLOSED_NOT_PLANNED: 4}
    return (order.get(g.get("bucket"), 9), -int(g.get("number") or 0))


# ---------------------------------------------------------------------------
# Wiring + CLI
# ---------------------------------------------------------------------------

def collect(
    workspace: Path,
    *,
    max_commits: int = 2000,
    issue_limit: int = 1000,
    fetcher: IssueFetcher | None = None,
    auditor: CommitAuditor | None = None,
) -> dict[str, Any]:
    root = workspace.resolve()
    fetch = fetcher or (lambda ws: fetch_issues(ws, limit=issue_limit))
    do_audit = auditor or audit_commit

    issues = fetch(root)
    refs = git_issue_refs(root, max_commits)
    total_commits = git_total_commits(root)
    coverage = compute_coverage(
        issues_fetched=len(issues),
        issue_limit=issue_limit,
        commits_scanned=max_commits,
        max_commits=max_commits,
        total_commits=total_commits,
    )

    audit_error: str | None = None
    if not issues:
        audit_error = "gh returned no issues (auth/network?) — cannot grade closure"

    # Only audit commits that RESOLVE an issue we actually fetched. Dedup by sha.
    fetched_nums = {int(i.get("number") or 0) for i in issues}
    to_audit: set[str] = set()
    for num, commit_refs in refs.items():
        if num in fetched_nums:
            for r in commit_refs:
                if r.get("kind") == RESOLVING:
                    to_audit.add(r["sha"])
    audits: dict[str, dict[str, Any]] = {sha: do_audit(sha, root) for sha in sorted(to_audit)}

    return build_payload(
        workspace=str(root), issues=issues, refs=refs, audits=audits,
        audit_error=audit_error, coverage=coverage,
    )


def render(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    lines = [
        f"issue-closure audit: {payload.get('verdict')} ({payload.get('finding')})",
        f"closure_rate={payload.get('closure_rate')}  next: {payload.get('next_action')}",
        (
            f"buckets: true={counts.get(TRUE_RESOLVED, 0)} "
            f"claimed={counts.get(CLAIMED_CLOSED, 0)} "
            f"open_witnessed={counts.get(OPEN_WITNESSED, 0)} "
            f"open={counts.get(OPEN, 0)} "
            f"not_planned={counts.get(CLOSED_NOT_PLANNED, 0)}"
        ),
    ]
    coverage = payload.get("coverage") or {}
    if not coverage.get("complete", True):
        for note in coverage.get("notes") or []:
            lines.append(f"  ! partial coverage: {note}")
    # Show only the actionable rows so the console stays a dos-top-style glance.
    actionable = [g for g in payload.get("issues", []) if g.get("bucket") in (CLAIMED_CLOSED, OPEN_WITNESSED)]
    if actionable:
        lines.append("  attention:")
        for g in actionable[:15]:
            commits = ",".join(g.get("witnessed_commits") or g.get("resolving_commits") or []) or "-"
            lines.append(f"    #{g['number']:<5} {g['bucket']:<14} [{commits}] {g['title']}")
    return "\n".join(lines)


def render_md(payload: dict[str, Any], *, date: str) -> str:
    counts = payload.get("counts") or {}
    out = [
        f"# Issue closure audit — {date}",
        "",
        f"- **closure_rate**: `{payload.get('closure_rate')}` "
        f"(TRUE_RESOLVED / (TRUE_RESOLVED + CLAIMED_CLOSED))",
        f"- **verdict**: `{payload.get('verdict')}` — {payload.get('reason')}",
        f"- **next**: {payload.get('next_action')}",
    ]
    coverage = payload.get("coverage") or {}
    if not coverage.get("complete", True):
        out.append(f"- **coverage**: ⚠ partial — {'; '.join(coverage.get('notes') or [])}")
    elif coverage:
        out.append(
            f"- **coverage**: complete "
            f"(issues_fetched={coverage.get('issues_fetched')}, "
            f"commits_total={coverage.get('commits_total')})"
        )
    out += [
        "",
        "| bucket | count |",
        "|---|---|",
    ]
    for b in (TRUE_RESOLVED, CLAIMED_CLOSED, OPEN_WITNESSED, OPEN, CLOSED_NOT_PLANNED):
        out.append(f"| {b} | {counts.get(b, 0)} |")
    out += ["", "## Per-issue", "", "| # | bucket | witnessed | title |", "|---|---|---|---|"]
    for g in payload.get("issues", []):
        commits = ",".join(g.get("witnessed_commits") or []) or "-"
        out.append(f"| {g['number']} | {g['bucket']} | {commits} | {g['title']} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DOS-witnessed GitHub ticket closure auditor (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--max-commits", type=int, default=2000, help="git log window to scan for issue refs")
    ap.add_argument("--issue-limit", type=int, default=1000, help="max issues to fetch from gh")
    ap.add_argument("--md", default="", help="write a dated markdown report to this path")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, max_commits=args.max_commits, issue_limit=args.issue_limit)

    if args.md:
        # Date is supplied by git (deterministic, no wall-clock dep in the tool itself).
        date = run_text(["git", "log", "-1", "--format=%cs"], workspace)["stdout"].strip() or "unknown"
        Path(args.md).write_text(render_md(payload, date=date), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
