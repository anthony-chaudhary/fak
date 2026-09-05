---
name: ticket-scope
description: "One repeatable pass that decides whether a GitHub ticket is a single dispatchable unit of agent work — or names exactly which of the six scope axes it fails and how to fix it. Wraps the native scope toolkit (`fak-dev issue contract` for structure/size/routing, `fak dispatch issue-smallness-lint` for atomicity, `fak-dev issue cohort` for batch/wave placement) and reads back one verdict per issue: DISPATCHABLE, or TRIAGE (add the missing section), DECOMPOSE (S2+ epic → leaves), or SPLIT. Use when assessing whether a ticket is dispatchable."
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Grep, Glob
argument-hint: "[--issue N | --body-file F | --open | --batch issues.json]   (default: --open backlog report)"
metadata:
  opencode: claude-only   # read-only allowed-tools boundary is load-bearing and Claude-only
---

# /ticket-scope — is this ticket one dispatchable unit of work?

> Wraps the native **ticket-scope toolkit** — `fak-dev issue contract`,
> `fak dispatch issue-smallness-lint`, `fak-dev issue cohort`. Every verb is
> **read-only**: it fetches via `gh`, lints, and reports. Editing the issue body,
> splitting an epic into child issues, or relabeling is a separate,
> operator-approved step (`fak-dev issue create`, `gh issue edit`). This mirrors
> [`issue-triage`](../issue-triage/SKILL.md)'s discipline: the tool decides what is
> *true*; the operator decides what *is done*.

The concept and the six axes live in **[docs/ticket-scope.md](../../docs/ticket-scope.md)** —
read it once. This skill is the executable pass over that contract.

A ticket's **scope** is the contract for one unit of agent work: what one worker
changes, proves, and finishes in one session. Wrong scope strands a worker — a
two-feature ticket no commit can witness, an epic that should be ten leaves, a
ticket whose write boundary collides with a live lease. One pass:
**gather → check six axes → verdict → propose fix → (approve) → apply → verify.**

## The six axes and the verb that measures each

| # | Axis | Verb | Fails with → fix |
|---|------|------|------------------|
| 1 | Structure | `fak-dev issue contract` | `ISSUE_SCOPE_INCOMPLETE` → **TRIAGE** (add the missing section) |
| 2 | Size (S0–S4) | `fak-dev issue contract` | `ISSUE_NOT_DISPATCH_LEAF` (S2+) → **DECOMPOSE** |
| 3 | Atomicity | `fak dispatch issue-smallness-lint` | `fail` (≥3 deliverables / witness≠1) → **SPLIT** |
| 4 | Write-scope | `dos arbitrate` · `fak orient` | unknown paths / collision → **TRIAGE** |
| 5 | Cohort | `fak-dev issue cohort` | split-first queue → **DECOMPOSE** |
| 6 | Work-class | `fak orient` · work-class axis | fenced/private lane → **ROUTE** |

A ticket is **DISPATCHABLE** only when all six agree.

## Gather

Choose the target from the argument:

- `--issue N` — one live issue (fetched via `gh`).
- `--body-file F` — a local draft (`-` = stdin), before it is ever filed.
- `--open` (default) — the whole open backlog, as a report.
- `--batch issues.json` — a cohort of candidates (a JSON array of
  `{number,title,body}`), for wave/split-first planning.

For a live single issue, cache the body once so both verbs read the same bytes:

```sh
gh issue view <N> --json number,title,body > /tmp/ticket-scope-<N>.json
```

## Check — run the two (or three) verbs

**Axes 1, 2, 4, 6 — structure, size, routing, class:**

```sh
fak-dev issue contract --from-issues /tmp/ticket-scope-<N>.json --json
# exit 0 = dispatchable on these axes; exit 3 = triage-only/refused.
# Read: missing_sections, missing_fields, the scale readout (declared/derived/
# effective size + witness-KIND match), model_tier readout, and the closed reason.
# Add --strict-scale to hold an undeclared size or under-scale witness.
```

**Axis 3 — atomicity / smallness:**

```sh
fak dispatch issue-smallness-lint --issue <N> --json
# verdict pass/warn/fail: fail on ≥3 deliverables (split) or witness count ≠ 1.
# Use --body-file for a draft, --open for the backlog, --open --scorecard for a card.
```

**Axis 5 — cohort (only with `--batch`):**

```sh
fak-dev issue cohort --from-issues issues.json --json --max-wave 8
# waves[] = concurrency-safe (disjoint-tree); split_first[] = oversized/non-leaf;
# triage[] = the rest; duplicate marker keys reported. Planner, not a gate.
```

For the **backlog** (`--open`), lead with the smallness report — it scans every
open issue in one call — then run `fak-dev issue contract --from-issues` over the
flagged rows to attach the structure/size reason:

```sh
fak dispatch issue-smallness-lint --open --json
```

## Verdict — one line per ticket

Fold the readouts into exactly one verdict, worst axis first:

- **DISPATCHABLE** — contract exit 0, smallness `pass`, write-scope bounded, lane
  takeable. Ready for a worker / a dispatch wave.
- **TRIAGE** — `ISSUE_SCOPE_INCOMPLETE` or unknown paths. Name the missing
  `sections`/`fields`; this is completable in place.
- **DECOMPOSE** — S2+ (`ISSUE_NOT_DISPATCH_LEAF`) or lands in the cohort
  split-first queue. Hand to [`/dos-replan`](../dos-replan/SKILL.md): one epic
  issue + N leaf issues, never a monolith.
- **SPLIT** — smallness `fail`. Two deliverables → two tickets; no witness or
  multiple witnesses → rewrite to exactly one.
- **ROUTE / FENCE** — routes to a private/exclusive or `class:frontdoor` lane;
  needs an operator label edit, not a body rewrite (an overlay backfills body
  sections only — it cannot clear a label-driven reason).

Report the verdict, the deciding reason token, and the one fix. Do **not** edit
the issue in this step.

## Propose → (approve) → apply → verify

State the proposed writes explicitly and stop for approval:

- **TRIAGE** — the exact section text to add (`Current state` / `Scope` /
  `Done condition` / `Witness` / `Likely files`). Apply with `gh issue edit`.
- **SPLIT / DECOMPOSE** — the proposed child titles + one-line scopes. File with
  `fak-dev issue create` (the sanctioned smooth path — one issue at a time, deduped
  by failure-signature, never one per turn), then link parent↔children.
- **ROUTE** — the label edit for an operator to make.

After applying, **re-run the same verb** on the edited issue (or the new
children) to prove the axis now passes. A proposal is not a fix until the verb
that failed returns clean.

## Guardrails

- **Read-only by default.** The three verbs never write to GitHub. Writes are a
  separate, operator-approved step — respect the reversibility gate.
- **Don't confuse size and atomicity.** A ticket can be a leaf (S1) and still fail
  smallness (two leaves stapled together), or be atomic and still be an epic.
  Both must pass.
- **One witness is load-bearing.** A ticket a worker cannot *prove* done is out of
  scope no matter how small — witness count ≠ 1 is a hard `fail`.
- **Ship the deciding evidence, not a summary.** Quote the reason token and the
  readout the verb emitted, so the verdict is auditable.
