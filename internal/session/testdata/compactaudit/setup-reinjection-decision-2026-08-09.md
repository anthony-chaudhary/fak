---
title: "Setup-reinjection dedupe: refuted — the counter stays a rollout-level restatement counter (2026-08-09)"
description: "Decision record for #5255: the 71%-duplicated instructions class is cross-fire restatement whose post-fire copy is the window's only resident copy, emitted by the upstream CLI. Deduplicating it would strip the agent's instructions; no safe change moves the counter down."
---

# Setup-reinjection dedupe: refuted (#5255, 2026-08-09)

[#5255](https://github.com/anthony-chaudhary/fak/issues/5255) read the
`instructions` class's 71% `dup_bytes` share as re-paid context and asked for a
dedupe. The 2026-08-03 attribution on the issue thread refuted that reading; this
note pins the decision in-repo so the counter is not re-mined into the same
inference. No detector behavior changed.

## Reproduce (this run)

```
fak session compact-audit --since 2026-06-15 --json --scrub --aggregate-only
```

Run 2026-08-09 from a `git archive HEAD` export at `5238e6a97a` (the live tree was
peer-red in `internal/gateway`), exit 0. Scrubbed: aggregate-only, lengths and
content hashes only, no payload bodies.

## The counter tracks corpus growth, not a regression

| quantity | 2026-07-18 witness | 2026-08-03 investigation | this run |
|---|---|---|---|
| fires with post-fire telemetry | 1,510 | 1,510 | 1,746 |
| `instructions` rows / bytes | 325 / 4,743,754 | 325 / 4,743,754 | 380 / 5,025,431 |
| `instructions` dup_rows / dup_bytes | 134 / 3,390,235 (71.5%) | 134 / 3,390,235 | 138 / 3,507,819 (69.8%) |
| `DUPLICATE_SETUP_REINJECTION` windows | 85 | 85 | 87 |
| `message/developer` dup rows / bytes | 1 / 2,121 | — | 1 / 2,121 |
| `message/user` dup rows / bytes | 1 / 5,877 | — | 1 / 5,877 |

## Why the dedupe is refuted (2026-08-03 attribution, 2,448 rollouts)

Three independent measurements from the issue-thread investigation, each pointing
the same way:

1. **Within a post-fire window, every instruction payload occurs exactly once** —
   225 (payload, window) pairs, all multiplicity 1. Zero in-window repeats.
2. **The fires that append setup are precisely those whose rebuilt window omits
   it.** The 84 appending fires carry a `replacement_history` with 0 setup-bearing
   bytes; the 1,431 non-appending fires carry the setup inside
   `replacement_history` already. The re-emission is the only copy in the window.
3. **The "duplicate" is a cross-window comparison against a copy the cut already
   discarded.** `regrowthTracker.seen` is session-scoped across fires by design.
   Double residency was tested directly and refuted: same-role
   `replacement_history` overlap ≥ 2,048 chars in 0 of 84 appending fires.

The emitter is not in this repo: every affected session's `session_meta` names
`originator=codex-tui`, and no fak source emits the payloads. Family A (55.0% of
dup bytes) is the upstream CLI's `role=developer` harness/skill rebuild — needs an
upstream report, not a patch here. Family B (45.0%) is the `role=user` session
prefix, dominated by this repo's own AGENTS.md plus the environment block.

## Decision

- `DUPLICATE_SETUP_REINJECTION` **stays a rollout-level restatement counter**. Its
  semantics are now stated at the definition (`compactregrowth.go`,
  `AnomalyDuplicateSetup`) and pinned by the sole-copy assertion in
  `TestRegrowthDuplicatedSetupFlagged`: the flagged duplicate is the window's only
  instructions row, never a license to drop the reinjection.
- The DoD's "witness the window count and dup_bytes drop" is **retired as
  structurally unmeetable**: on this evidence the only change that moves the
  counter down deletes the sole resident copy of the agent's instructions. The
  counter growing 85 → 87 with the corpus is the healthy reading.
- The real lever is **size, not dedupe**: the ~34 kB Family B payload is AGENTS.md
  resident in *every* window, pre- and post-fire — epic #3229's resident-budget
  axis, not this issue.

## Generation close (promotion / demotion / invalidating assumption)

- **Promotion evidence (what would reopen this):** an in-window instructions
  duplicate (multiplicity ≥ 2) appearing in the aggregate, or an upstream CLI
  change that makes the post-compaction rebuild referenceable instead of
  re-emitted.
- **Demotion/retirement evidence:** the three refuting measurements above, plus
  bit-for-bit reproduction of the ticket's numbers at two later HEADs.
- **Invalidating assumption:** the corpus stays `originator=codex-tui`. A harness
  whose rebuilt window *retained* the earlier copy and still re-appended would be
  true double residency — this counter cannot distinguish that case today, and the
  family split (A/B) was measured once on the 2026-08-03 corpus, not re-derived in
  this run.
