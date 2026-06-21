#!/usr/bin/env python3
r"""Render the issue-resolution prompt the issue-targeted dispatch worker runs.

The generic ``/dos-kernel:dos-dispatch-loop`` worker resolves units from the
*plan portfolio* — but this public repo ships no ``PLAN-*.md`` (the supervisor
reports ``PLAN_SURFACE_EMPTY``), so a bare dispatch-loop worker has no
issue-level work surface and closes nothing. The backlog is in GitHub *issues*,
not plans.

This module is the missing bridge: given ONE concrete open issue, it renders a
scoped resolution prompt (generalised from the proven ``top20`` per-issue prompt
shape) that tells a headless ``claude -p`` worker to read the issue, do the
smallest correct change that resolves it, run the lane's own gate, and — the
load-bearing rule — **reference ``#N`` in the commit subject**. That ``#N`` is
what binds the shipped commit back to the issue: ``issue_closure_audit.py``'s
``_RESOLVE_RE`` / subject-``#N`` rung reconstructs the commit→issue link from the
subject (this repo runs no PR-keyword workflow), so without the citation a
resolved issue can never be witnessed-closed. With it, the deterministic
``issue_resolve_witnessed.py`` arm picks the issue up and closes it, each close
re-verified by ``dos commit-audit``.

The prompt is deliberately conservative and honest-block-first: an issue the
worker cannot fully land on this host must end with a final report naming the
exact missing gate, never a fabricated pass. It hard-codes the repo's
non-negotiable git laws (trunk-only, commit-by-path, sign-off) so an unattended
worker cannot drift off them.

Pure/testable: ``render_prompt`` takes plain data and returns a string; the CLI
wrapper fetches the issue via ``gh`` and prints (or writes) the prompt.

    python tools/issue_worker_prompt.py --issue 465 --lane docs           # print
    python tools/issue_worker_prompt.py --issue 465 --lane docs --out p.md # write
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-issue-worker-prompt/1"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def fetch_issue(number: int, *, workspace: Path, timeout: int = 60) -> dict[str, Any]:
    """Fetch the issue's title/body/labels via ``gh``. On any failure return a
    minimal record (number only) so the prompt still renders — a worker can read
    the live issue itself with ``gh issue view`` as its first step."""
    try:
        proc = subprocess.run(
            ["gh", "issue", "view", str(number), "--json",
             "number,title,body,labels,state"],
            cwd=str(workspace), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"number": number, "_error": str(exc)}
    if proc.returncode != 0:
        return {"number": number, "_error": (proc.stderr or proc.stdout).strip()[-300:]}
    try:
        doc = json.loads(proc.stdout)
    except ValueError:
        return {"number": number, "_error": "gh issue view produced no JSON"}
    return doc if isinstance(doc, dict) else {"number": number}


def _labels(issue: dict[str, Any]) -> str:
    labs = issue.get("labels") or []
    names = [l.get("name") for l in labs if isinstance(l, dict) and l.get("name")]
    return ", ".join(names) if names else "(none)"


def render_prompt(issue: dict[str, Any], lane: str, *, workspace: str) -> str:
    """Render the resolution prompt for ONE issue. Pure: no I/O."""
    n = issue.get("number")
    title = (issue.get("title") or "").strip() or f"issue #{n}"
    body = (issue.get("body") or "").strip()
    # Keep the embedded body bounded — the worker re-reads the live issue anyway.
    if len(body) > 1800:
        body = body[:1800] + "\n…(truncated — read the full issue with `gh issue view`)"
    labels = _labels(issue)

    return f"""your goal: resolve GitHub issue #{n} ({title}) with the smallest correct \
change that genuinely closes it, then ship it on `main` citing `#{n}` in the \
commit subject — OR end with a final report naming the exact gate you could not \
reach and why. Do NOT fabricate a pass.

read first: run `gh issue view {n}` for the live issue, then orient with \
`AGENTS.md` (build/test/run + the hard rules) and `llms.txt` (the doc map). \
This issue routed to the `{lane}` lane (its file-tree). Labels: {labels}.

issue body (verbatim, may be truncated — re-read live):
---
{body or '(no body — read the title and `gh issue view` for the full thread)'}
---

how to work it:
- Take the lane lease first if siblings may collide: \
`dos arbitrate --workspace . --lane {lane} --kind cluster` (honor a REFUSE — pick \
nothing and stop; do not --force onto a held lane).
- Make the SMALLEST change that resolves the issue's actual ask. Prefer one leaf \
/ one file. If the issue is a docs/observability/test ask, that is often a single \
file. If it is a large epic you cannot land whole, land the smallest honest \
increment and say in your report what remains.
- Run the gate yourself before claiming done: the lane's own test \
(`go test ./... -count=1` for the touched package, or the doc/lint check the \
issue names). A claim with no gate run is not done.

git laws (enforced below the agent — breaking them refuses your commit):
- Work on `main` ONLY. Never branch / new-worktree (the OFF_TRUNK guard refuses it).
- `git commit -s -- <explicit paths>` — sign-off (DCO), commit BY PATH only. \
NEVER `git add -A` (shared multi-session tree — a blanket add steals a sibling's \
in-flight files). Stage only the files you wrote.
- **Reference `#{n}` in the commit subject** (e.g. `fix(scope): … (#{n})`). This is \
what binds your commit to the issue so the closure auditor can witness it — a \
resolved issue with no `#{n}` in the subject never closes.
- No push / tag / force-push / history-rewrite / reset / clean / \
checkout-of-tracked-files. Just commit on main.

acceptance (your stop condition): a committed change on `main` whose subject \
cites `#{n}` and whose gate you actually ran is green — OR a final report that \
names the specific missing artifact/host capability and the smallest next step. \
Honesty over a green-looking lie: the repo keeps a witness ledger and a \
self-authored "done" is re-checked against git.

workspace: {workspace}. lane: {lane}. issue: #{n}.
"""


def build(number: int, lane: str, *, workspace: Path) -> dict[str, Any]:
    issue = fetch_issue(number, workspace=workspace)
    prompt = render_prompt(issue, lane, workspace=str(workspace))
    return {
        "schema": SCHEMA,
        "issue": number,
        "lane": lane,
        "title": issue.get("title"),
        "fetch_error": issue.get("_error"),
        "prompt": prompt,
        "prompt_chars": len(prompt),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render the issue-resolution prompt for one open issue.")
    ap.add_argument("--issue", type=int, required=True, help="issue number to resolve")
    ap.add_argument("--lane", required=True, help="the lane the issue routed to")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--out", default=None, help="write the prompt to this file (else stdout)")
    ap.add_argument("--json", action="store_true", help="emit the full record as JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    rec = build(args.issue, args.lane, workspace=root)
    if args.out:
        Path(args.out).write_text(rec["prompt"], encoding="utf-8")
        print(f"wrote {args.out} ({rec['prompt_chars']} chars) for issue #{args.issue}")
    elif args.json:
        print(json.dumps(rec, indent=2))
    else:
        print(rec["prompt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
