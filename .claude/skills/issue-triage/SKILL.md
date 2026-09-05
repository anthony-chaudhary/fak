---
name: issue-triage
description: One repeatable pass over the open GitHub issue backlog — classify every open issue (needs-priority / needs-kind / needs-area, orphaned P0-P1, stale, dormant question), rank them into a deterministic "do next" order, propose the mechanical gardening moves (mark stale, close dormant questions), and apply them only on operator approval. The helper is read-only; writing labels, comments, or closes is gated. Use when the operator says "triage the issues", "what should I work on next", "garden the backlog", "the issue labels.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Grep, Glob
argument-hint: "[--since-days N] [--scope priority|kind|area|orphans|stale|dup|question]   (apply: issue-actions-*.json)"
output_root: docs/_audits
metadata:
  opencode: claude-only   # #422: read-only allowed-tools boundary is load-bearing and Claude-only — exclude from the opencode skills.paths scan
---

# /issue-triage — classify, rank, and garden the open issue backlog

> Lead with current first-class `fak console issues`, `fak-dev issue`, and
> `fak dispatch` surfaces. The Python triage helper remains a read-only legacy/debug
> fallback for report fields not yet exposed by those commands. Mutations stay a separate,
> operator-approved step.

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
fak console issues --state open --json > issues-ranked.json

# Legacy/debug comparison for extra historical report fields:
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

### Default medium-scale scout: OpenCode Go Ox Alpha

For a broad repository review or a medium-scale gardening batch (25 or more
issues/candidates), use `opencode-go/ox-alpha-free` as a **read-only proposal
stage** when that route is available. This is the default model-assisted path,
not authority to mutate GitHub or the tree. If the route is unavailable, keep
the deterministic report and continue without a model; do not silently
substitute a paid model.

Run the scout with OpenCode's permission-constrained `plan` agent. Never pass
`--auto`, and tell the model that shell commands are read-only too:

```bash
opencode run --agent plan --model opencode-go/ox-alpha-free --format json \
  "Read-only review. Inspect the ranked issue report and repository evidence. Do not edit files, create/comment/close issues, or run any mutating command. Return candidate findings only, each with: claim; exact witness command; observed output/count; affected paths or issue numbers; dedupe query and results; current parent issue state; proposed labels/milestone; confidence; and disconfirming evidence."
```

Save the session ID and the candidate packet beside the dated audit report.
Treat model prose as an allegation until an independent pass reruns every
witness. The measured rationale and failure examples are indexed in
[`docs/notes/OX-ALPHA-DOGFOOD-2026-08-22.md`](../../../docs/notes/OX-ALPHA-DOGFOOD-2026-08-22.md#medium-scale-issue-review-and-gardening-audit).

Ox Alpha's audited failure mode is confident conflation: one true global smell
can be combined with stale parent state, non-versioned residue, incomplete
inventories, or incorrect attribution. Therefore the scout and gardener must
not be the same self-certifying pass.
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
4. **duplicate clusters** — run `fak-dev issue dedup` (the body-aware census; the
   triage helper no longer detects dups). Confirm-and-close candidates.

Before proposing dispatch for a selected issue, validate and price it through the
first-class Go surfaces:

```bash
gh issue view N --repo owner/name --json number,title,body,labels > issue.json
fak-dev issue contract --from-issues issue.json --json
fak issue-queue --from-issues issue.json --json
```

Use `tools/issue_contract.py` or `tools/issue_lane_router.py` only to debug a discrepancy
with those commands; do not teach them as the default operator path.

For every model-proposed candidate, the independent reconciliation pass must:

1. Rerun the exact witness command and preserve its output.
2. Check that the measurement actually supports the claim's scope and causal
   language (for example, filesystem residue outside `go list ./...` cannot
   explain a Go package build failure).
3. Read back every named parent/duplicate issue's current state.
4. Preserve the complete inventory behind any count; samples cannot support a
   “full list” acceptance condition.
5. Reject candidates that combine different leaves, non-versioned cleanup, or
   comment bookkeeping into one issue.
6. Materialize each survivor as a candidate JSON packet, then run:

```bash
fak-dev issue contract --file candidate.json --dedupe-checked \
  --strict-witness --strict-scale --strict-project-work \
  --strict-born-routed --strict-model-tier --json
```

A held verdict stays `needs-triage` / `triage-only`; it is not worker-ready.
Create only passing candidates, with priority, generation, class, and milestone
set at creation. After creation, read the issue back from GitHub and compare its
title, body, labels, milestone, and parent against the accepted packet. A
separate reconciliation pass may retain, rewrite, or close model-filed
candidates; never use the filing trajectory's own summary as proof.
Do **not** paste the whole report into chat — link the file. Surface only the
0–3 findings that matter and the one batch the operator can act on now.

## Step 3 — Distinguish mechanical from judgment-call actions

The `--actions` JSON carries two kinds of entries:

- **mechanical** (`cmd` is set) — `close-dormant-question` and `mark-stale`.
  These are defensible from the issue's own signals (a `question` idle ≥ 30 d,
  or any non-P0/P1 idle ≥ 60 d). The operator can approve these as a batch.
- **review-only** (`cmd: null`, `kind: "review"`) — `needs-priority`,
  `needs-kind`, `needs-area`, `orphan`. The helper **cannot** decide these
  algorithmically (what priority is the work?). They are surfaced for the
  operator to decide per-issue.

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
fak console issues --state open --json > issues-ranked-after.json
gh issue view <n> --json labels,state,comments   # spot-check 2–3 applied rows
<p> <h.issue_triage> --actions --out <audits_dir>/verify-actions.json
```

The first-class before/after issue models are the truth source; the fallback action manifest
is an additional mutation witness when that helper was used.

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
- **Duplicate detection lives in `fak issue dedup`, not this helper.** The
  body-aware census (title + title+body simhash) catches semantic dups the old
  title-token Jaccard pass missed. It is advisory: it ranks merge proposals with
  per-pair evidence and suppresses oversized shared-template families (epic
  siblings are not duplicates of each other). Always confirm before closing as
  dup; it never writes to GitHub.
- **It only sees labels + timestamps.** It cannot read the issue body for
  severity, dependencies, or whether the reporter still cares. The operator's
  judgment is the last rung; this skill gets the queue into shape so that
  judgment is spent on the right rows.
- **Closed issues are out of scope.** This triages the *open* backlog. Auditing
  closed-issue quality (wrong reason codes, premature closes) is a different
  skill.
