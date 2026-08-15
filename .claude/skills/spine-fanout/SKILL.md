---
name: spine-fanout
description: Apply the two new-work defaults — ship the minimal WORKING end-to-end spine first (or file the spine as its own issue), then fan out the 3..50+ follow-on QA/dogfood/productization backlog at creation time via `fak issue fanout`. Use when starting any new feature/leaf/verb/demo, when a spine just shipped, when asked to "fan out", "file follow-ons", "create the e2e spin", or at the end of a super-loop turn that landed new work.
allowed-tools: Read, Bash, Write
---

# spine-fanout — spine first, then fan out

Doctrine: [`docs/spine-first-defaults.md`](../../../docs/spine-first-defaults.md).
Two defaults for any new unit of work; this skill is the mechanical checklist.

## Step 1 — classify the moment

- **Starting new work** → go to Step 2 (spine gate).
- **A spine just shipped** (commit/demo/test exists) → go to Step 3 (fan out).

## Step 2 — the spine gate

Ship the smallest runnable end-to-end path through the REAL seam this session.
Before broad proof or optimization, write the outcome in one sentence, connect one
representative applied path, and capture its command/test result. Then expand the
failure/edge/platform/concurrency/soak envelope; optimize only after the working
path provides the real baseline. Do not count a component benchmark, comparison
matrix, or disconnected mock as the spine.

Before arming the issue, write both contract sections:

- `Core through-line`: shortest change -> real seam -> observable outcome -> witness path.
- `Gold-plating boundary`: tempting breadth, polish, abstraction, or optimization that the
  named outcome and witness still work without; route valuable items to follow-ons.

Counterfactual: if omitting an item still leaves the named user able to traverse the spine
and the witness able to prove the outcome, it is gold plating for this issue. Required
safety, compatibility, and fail-closed behavior stay core.

| Work shape | Minimal spine bar |
|---|---|
| user-facing | LCD demo (`docs/run-the-demos.md`): one command, deterministic, no key/network/GPU, `-selfcheck` |
| library leaf / verb | a test driving the real object + one captured live run (`--json`/`--dry-run` ok) |
| process/doctrine | the enforcing machinery (gate/skill/verb), not a memo |

**Not achievable this session with high confidence?** File the spine itself as
the first issue — `gen/now`, milestoned at creation, missing witness named in
the body. Never silently defer the spine. Then continue to Step 3 with
`--spine` set to that issue ref only if a partial witness exists; otherwise
stop after filing (the fan-out waits for a spine).

## Step 3 — prove outward from the working spine

Keep the captured spine command/test green while adding the issue's required
proof envelope. Order the queue: correctness and fail-closed paths, edge cases,
platform/concurrency/soak evidence, then end-to-end optimization against the real
baseline. If new evidence breaks the spine, restore the applied path before
continuing the matrix.

## Step 4 — generate the fan-out (3..50+)

```bash
fak issue fanout --title "<feature>" --leaf <leaf> \
    --spine "<commit sha | demo cmd | doc path>" [--parent '#<epic>'] --json > /tmp/fanout.json
fak issue cohort --from-plan /tmp/fanout.json   # optional: leased, concurrency-safe waves
```

The verb refuses to run without `--spine` — that refusal is Step 2 talking.
Areas: qa, dogfood, product, observability, integration, docs, release
(`--areas` filters; `--max` caps, floor 3).

## Step 5 — file the issues

For each candidate (or each wave): create the GitHub issue with the candidate's
title and contract sections as the body, **milestone + labels at creation**
(`fanout`, the area label, priority from the candidate, generation label).
Dedupe first: search existing issues for the `fanout-<leaf>-` marker key and
comment instead of re-filing. On this host run `gh` through PowerShell, never
the Bash tool.

## Step 6 — close the loop

- Link the fan-out issues from the epic/parent; note the spine SHA there.
- If the session ends with candidates unfiled, commit the plan JSON output into
  the epic/issue body or a `docs/notes/` entry (with the INDEX.md line) so the
  fan-out is recoverable — a plan that lives only in a transcript is lost.
