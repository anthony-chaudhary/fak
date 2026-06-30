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
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _no_window_creationflags() -> int:
    """``creationflags`` that stop a console child (the ``gh issue view`` below) from
    popping a visible window when this renders windowless (pythonw) from a scheduled
    dispatch tick; ``0`` on POSIX. Mirrors dispatch_worker.no_window_creationflags,
    kept local so this leaf prompt module imports only stdlib."""
    return 0x08000000 if os.name == "nt" else 0

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
        # gh emits UTF-8; force it. Without encoding=, Windows decodes with cp1252
        # and a single non-cp1252 byte in the issue body (an em-dash, smart quote,
        # or emoji — the house style is em-dash-heavy) raises UnicodeDecodeError in
        # subprocess's reader thread, leaving proc.stdout None. errors="replace"
        # keeps a stray byte on any platform from crashing the decode.
        proc = subprocess.run(
            ["gh", "issue", "view", str(number), "--json",
             "number,title,body,labels,state"],
            cwd=str(workspace), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            creationflags=_no_window_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"number": number, "_error": str(exc)}
    if proc.returncode != 0:
        return {"number": number, "_error": (proc.stderr or proc.stdout or "").strip()[-300:]}
    try:
        doc = json.loads(proc.stdout)
    except (ValueError, TypeError):
        # ValueError: malformed JSON. TypeError: stdout is None (a reader-thread
        # decode error on a node still on the old cp1252 path). Either way the
        # docstring's promise holds — return the minimal record, never crash.
        return {"number": number, "_error": "gh issue view produced no JSON"}
    return doc if isinstance(doc, dict) else {"number": number}


def _labels(issue: dict[str, Any]) -> str:
    labs = issue.get("labels") or []
    names = [lab.get("name") for lab in labs if isinstance(lab, dict) and lab.get("name")]
    return ", ".join(names) if names else "(none)"


def _generation_intent(issue: dict[str, Any]) -> str:
    labs = issue.get("labels") or []
    names = {lab.get("name") for lab in labs if isinstance(lab, dict) and lab.get("name")}
    if "gen/now" in names:
        return "now — immediate trunk-safe product/operator work; do not wait for a future architecture bet"
    if "gen/next" in names:
        return "next — near-term foundation; keep gated or dogfooded until promotion evidence lands"
    if "gen/second-next" in names:
        return "second-next — architectural option; preserve assumptions and compatibility evidence"
    if "gen/future" in names:
        return "future — research or long-horizon option; do not treat it as lower priority by default"
    return "unclassified — read docs/generation.md and avoid guessing; keep needs-triage if the horizon is unclear"


def _generation_frame(issue: dict[str, Any]) -> str:
    labs = issue.get("labels") or []
    names = {lab.get("name") for lab in labs if isinstance(lab, dict) and lab.get("name")}
    if "gen/now" in names:
        return ("stream=gen/now; allowed risk=low, trunk-safe, reversible; "
                "proof bar=focused test or captured command output before closing; "
                "scope width=one leaf or one operator surface; "
                "expected artifact=shipped code, doc, report, or configuration on main")
    if "gen/next" in names:
        return ("stream=gen/next; allowed risk=moderate only behind a gate, dogfood path, or handoff contract; "
                "proof bar=contract test plus promotion evidence that names what moves it toward now; "
                "scope width=near-term foundation, one prompt/dispatch/report seam; "
                "expected artifact=agent-runnable schema, prompt frame, gated behavior, or operator readout")
    if "gen/second-next" in names:
        return ("stream=gen/second-next; allowed risk=architectural exploration, never default exposure; "
                "proof bar=simulation, compatibility policy, or dependency edge with demotion criteria; "
                "scope width=cross-generation option or interface boundary; "
                "expected artifact=design memo, compatibility test, lifecycle model, or prototype behind an explicit gate")
    if "gen/future" in names:
        return ("stream=gen/future; allowed risk=research and option valuation only, not current-product commitment; "
                "proof bar=sourced memo, kill criteria, or decision model with assumptions stated; "
                "scope width=long-horizon market, standards, benchmark, or portfolio option; "
                "expected artifact=research note, scorecard, simulator spec, or narrative that preserves non-goals")
    return ("stream=unclassified; allowed risk=triage only; "
            "proof bar=classify the horizon from issue evidence before implementation; "
            "scope width=label/milestone repair or a clarification note; "
            "expected artifact=updated labels, milestone, or final report naming why classification is blocked")


def render_prompt(issue: dict[str, Any], lane: str, *, workspace: str) -> str:
    """Render the resolution prompt for ONE issue. Pure: no I/O."""
    n = issue.get("number")
    title = (issue.get("title") or "").strip() or f"issue #{n}"
    body = (issue.get("body") or "").strip()
    # Keep the embedded body bounded — the worker re-reads the live issue anyway.
    if len(body) > 1800:
        body = body[:1800] + "\n…(truncated — read the full issue with `gh issue view`)"
    labels = _labels(issue)
    generation = _generation_intent(issue)
    generation_frame = _generation_frame(issue)

    return f"""your goal: resolve GitHub issue #{n} ({title}) with the smallest correct \
change that genuinely closes it, then ship it on `main` citing `#{n}` in the \
commit subject — OR end with a final report naming the exact gate you could not \
reach and why. Do NOT fabricate a pass.

read first: run `gh issue view {n}` for the live issue, then orient with \
`AGENTS.md` (build/test/run + the hard rules) and `llms.txt` (the doc map). \
Then run `python tools/memory_read.py` for the committed fleet memory (lane \
quirks, known blockers, host gotchas) — a Claude worker gets this auto-injected, \
an opencode worker does NOT, so this read is how both backends start warm (it is \
a harmless no-op if the mirror is absent). \
This issue routed to the `{lane}` lane (its file-tree). Labels: {labels}.
Generation intent: {generation}. Generation is orthogonal to priority, shared trunk, and runtime feature gates.
Generation frame: {generation_frame}.
When closing generation work, name promotion evidence, demotion/retirement evidence, and at least one invalidating assumption in the artifact or final report.

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
- **Reference `#{n}` in the subject AND end it with a `(fak {lane})` trailer**, \
lead with a verb (e.g. `fix({lane}): … (#{n}) (fak {lane})`; use add/fix/implement/\
test, NEVER a noun-led description). The `#{n}` binds your commit to the issue; the \
verb-led subject + `(fak {lane})` trailer is what makes `dos commit-audit` grade it \
`diff-witnessed` instead of ABSTAIN — and the closure auditor closes the issue ONLY \
on a witnessed commit. Miss either the `#{n}` or the trailer and your resolved issue \
never closes.
- No push / tag / force-push / history-rewrite / reset / clean / \
checkout-of-tracked-files. Just commit on main.

acceptance (your stop condition): a committed change on `main` whose subject \
cites `#{n}` and whose gate you actually ran is green — OR a final report that \
names the specific missing artifact/host capability and the smallest next step. \
Honesty over a green-looking lie: the repo keeps a witness ledger and a \
self-authored "done" is re-checked against git. If you discovered a durable fact \
worth keeping (a lane quirk, a host gotcha, a blocker), surface it explicitly in \
your final message so an operator or Claude peer can record it to the memory \
mirror — an opencode worker has no auto-memory write path of its own.

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
        "issue_record": issue,
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
