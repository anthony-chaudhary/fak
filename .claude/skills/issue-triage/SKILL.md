---
name: issue-triage
description: One repeatable pass over the open GitHub issue backlog — classify every open issue (needs-priority / needs-kind / needs-area, orphaned P0-P1, stale, dormant question, likely-duplicate), rank them into a deterministic "do next" order, propose the mechanical gardening moves (mark stale, close dormant questions), and apply them only on operator approval. The helper is read-only; writing labels, comments, or closes is gated. Use when the operator says "triage the issues", "what should I work on next", "garden the backlog", "the issue labels are a mess", "close stale issues", or on a /loop cadence to keep the backlog honest.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Grep, Glob
argument-hint: "[--since-days N] [--scope priority|kind|area|orphans|stale|dup|question]   (apply: issue-actions-*.json)"
output_root: docs/_audits
metadata:
  opencode: claude-only   # #422: read-only allowed-tools boundary is load-bearing and Claude-only — exclude from the opencode skills.paths scan
---

# /issue-triage — classify, rank, and garden the open issue backlog

> Wraps `tools/issue_triage.py` (pure-stdlib). The helper is **read-only** — it
> fetches via `gh`, classifies, ranks, and proposes actions, but never edits an
> issue. Applying the proposed labels / comments / closes is a separate,
> operator-approved step that runs `gh issue edit|comment|close`. This mirrors
> plan-audit's read-only discipline and the release skill's dry-run-first rule:
> the helper decides what is *true* and what *could* be done; the operator
> decides what *is* done.

A backlog rots the same way an index drifts: labels go missing, priorities
never get set, questions hang open forever, duplicates accrete. This skill is
the gardening pass for GitHub issues — the issue analogue of `curate-cluster`.
One pass: **gather → rank → propose → (approve) → apply → verify**.

The triage ranking is a transparent integer score, not a model judgment (see
the helper's module docstring for the exact formula). The score is surfaced in
the report so the ordering is auditable.

## Project contract

Reads `.claude/project.yaml`. Required keys:

- `python` — interpreter path (default: `python`).
- `helpers.issue_triage` — the read-only fetch/classify/rank/actions script.
- `audits_dir` — where reports + the actions manifest are written (default:
  `docs/_audits/`, shared with plan-audit — these are process residue, kept out
  of the root index scan).

If `helpers.issue_triage` is absent, print one line pointing at the missing
helper and stop. Do not improvise a substitute.

The label taxonomy is the project's, baked into the helper (priority/P0|P1|P2,
kind, area, workflow). Override thresholds or label sets via the helper's
`--config <json>` if the taxonomy changes; the skill itself is taxonomy-agnostic.

## Step 1 — Gather and rank (read-only)

```bash
<p> <h.issue_triage> --markdown --out <audits_dir>/issue-triage-<YYYY-MM-DD>.md
<p> <h.issue_triage> --actions  --out <audits_dir>/issue-actions-<YYYY-MM-DD>.json
```

Use today's date in UTC. Use `--out` (not a shell `>` redirect) — PowerShell
redirects re-encode to UTF-16 and mangle the `·` / `—` glyphs in the report.
One report per day; overwrite if it exists.

Optional scoping:

- `--since-days N` — only issues touched in the last N days (use after a burst
  of filing, to triage just the new wave).
- `--scope <tag>` — filter the rows to one bucket: `priority | kind | area |
  orphans | stale | dup | question`. Drops the other sections so the report is
  focused.

Read the markdown report back (the helper prints `wrote <path>`, not the body).
Do not page through `gh issue list` by hand — the helper already folded the whole
open backlog into the ranked model.

## Step 2 — Read the report, lead with the load-bearing finding

The report's counts line is the headline. Lead the operator summary with the
**single largest gap**, in this priority order:

1. **Orphan P0/P1** (`orphan` count) — high-priority work with no claimant.
   This is the most expensive backlog rot. Name the top 3 by score.
2. **needs-priority** — the canonical "labels were never set" gap. A pass that
   cuts that number is a real win; quote the project's own current baseline from
   the report rather than a fixed number.
3. **stale / dormant-question** — these carry **mechanical** proposed actions
   (mark stale, close). They are the gardening moves the operator can approve
   in one batch.
4. **likely-duplicate** clusters — confirm-and-close candidates.

Do **not** paste the whole report into chat — link the file. Surface only the
0–3 findings that matter and the one batch the operator can act on now.

## Step 3 — Distinguish mechanical from judgment-call actions

The `--actions` JSON carries two kinds of entries:

- **mechanical** (`cmd` is set) — `close-dormant-question` and `mark-stale`.
  These are defensible from the issue's own signals (a `question` idle ≥ 30 d,
  or any non-P0/P1 idle ≥ 60 d). The operator can approve these as a batch.
- **review-only** (`cmd: null`, `kind: "review"`) — `needs-priority`,
  `needs-kind`, `needs-area`, `orphan`, `likely-dup`. The helper **cannot**
  decide these algorithmically (what priority is the work? is this really a
  dup?). They are surfaced for the operator to decide per-issue.

Never fabricate a `cmd` for a review-only action. If the operator sets a
priority by hand, you build that single `gh issue edit <n> --add-label
priority/Px` command on the spot — you do not promote the row to "mechanical".

## Step 4 — Apply mechanical actions only on approval (dry-run first)

Before running anything, print the exact commands you will run, grouped by
kind. The manifest's `cmd` strings are PowerShell-shell-ready (single-quoted
`--comment`, double-quoted `--reason`); run them verbatim.

```bash
# Dry-run review — show, don't run. Confirm the counts match the manifest.
# Then, per batch, on explicit operator "yes":
gh issue close <n> --reason "not planned" --comment "..."   # dormant-question
gh issue edit <n> --add-label "stale" --add-comment '...'   # mark-stale
```

Rules:

- **Batch by kind**, and confirm each batch separately ("3 dormant-question
  closes — ok?"). Do not run a single blanket "apply everything".
- **One issue at a time for any review-only action** the operator resolves
  live. If they say "#98 is P0, already claimed" → `gh issue edit 98
  --add-label in-progress`, nothing else.
- **Re-fetch after applying** — re-run Step 1's `--actions` and confirm the
  applied rows dropped out (their tags changed). The pass is done when the
  mechanical-action count is zero AND the operator has either resolved or
  explicitly deferred each review-only row.

## Step 5 — Verify (witness, don't assume)

```bash
gh issue view <n> --json labels,state,comments   # spot-check 2–3 applied rows
<p> <h.issue_triage> --actions --out <audits_dir>/issue-actions-<YYYY-MM-DD>.json
```

The second run is the witness: the applied issues should no longer carry the
tag the action addressed (a closed dormant-question is no longer open; a
marked-stale issue now has the `stale` label). If a row's tag survived, the
`gh` write failed silently — re-check and re-run that one.

The pass is done when: the report's mechanical-action count is zero, the
operator has resolved or deferred every review-only row they chose to handle
this pass, and the spot-checks confirm the GitHub state matches the manifest.

## Running it on a cadence

This is a single pass. The backlog re-rots, so drive it with `/loop` (e.g.
`/loop issue-triage --since-days 30`, weekly) — each pass only touches issues
that aged into a bucket since the last one. The helper is idempotent: an issue
already correctly labeled and claimed contributes nothing to the action
manifest, and a pass with nothing to propose commits nothing and exits.

## Where this misleads / honest limits

- **The score is a heuristic, not a priority oracle.** A P2 bug can matter more
  than a P1 enhancement; the formula can't see that. Treat the ranking as
  "where to look first", not "what to build first".
- **Stale/dormant thresholds are calendar-based, not signal-based.** A quiet
  issue that is actively blocked (waiting on a dependency) is not stale — the
  operator must read the close candidates before approving. The manifest's
  comment text says "reply to keep open" for exactly this reason.
- **Duplicate detection is title-token Jaccard.** It catches "same words, same
  scope" pairs but misses semantic dups with different wording, and can
  false-positive on a prolific scope. Always confirm before closing as dup;
  never auto-close on the `likely-dup` tag alone.
- **It only sees labels + timestamps.** It cannot read the issue body for
  severity, dependencies, or whether the reporter still cares. The operator's
  judgment is the last rung; this skill gets the queue into shape so that
  judgment is spent on the right rows.
- **Closed issues are out of scope.** This triages the *open* backlog. Auditing
  closed-issue quality (wrong reason codes, premature closes) is a different
  skill.
