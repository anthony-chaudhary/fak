---
title: "Generation classification: fleet-status roll-up digest (#2042)"
description: "Grooming artifact classifying #2042 (post fak rollup as the single delta-gated ops digest + de-cluster the shared Slack channel) as gen/now, with promotion, demotion/retirement, and invalidating-assumption evidence, the applied label/milestone intake, and the smallest-next implementation increment for a code-lane worker."
---

# Generation Classification — Fleet-Status Roll-up Digest (issue #2042)

Grooming artifact for issue #2042
(`feat(fleet-status): post the roll-up as the single ops digest (delta-gated) +
de-cluster the shared channel`). It classifies the generation horizon from issue
evidence so a later, implementation-authorized worker starts warm. It is **not**
a resolution: #2042's witnessable acceptance (`fak rollup --slack --dry-run`
resolves channel/token and reports what it would post while sending nothing, plus
the delta-gate suppressing an unchanged GREEN) is not built, so #2042 stays
**open**.

Parent: pillar 1 ("one report") of the fleet-status epic
[#2038](https://github.com/anthony-chaudhary/fak/issues/2038).
Snapshot date: 2026-07-05.

## Why this is a classification note and not the feature

The issue routed to the **`docs` lane** (its file-tree is docs-only:
`docs/**`, `README.md`, `INDEX.md`, `llms.txt`). Its acceptance, however, is a
new `--slack` flag and delta-gate on `fak rollup` — Go changes in
`cmd/fak/rollup.go` and `internal/` (a new post path + a persisted last-posted
digest for the delta compare). Those paths are **outside** the docs lane's lease
tree, so implementing them here would collide with a code lane and violate the
lane the arbiter granted (`dos arbitrate --lane docs` tree =
`docs/**`, not `cmd/**`/`internal/**`). The generation intent on this routing was
**unclassified**, and the triage frame for that state authorizes classification +
a note only — not a cross-lane code edit. So this worker does the in-lane,
gate-passable half (classify + record the map) and leaves the code to a
code-lane worker, warmed by the increment map below.

## Classification

| Field | Value |
|---|---|
| Issue | #2042 |
| Stream | `gen/now` |
| Milestone | `Generation G0 - Now / Immediate` |
| Also carries | `generation` |
| Parent epic | #2038 (`epic(fleet-status)`), pillar 1 "one report" |
| Intake status | **applied** — labels `generation` + `gen/now` and milestone bound to #2042 on 2026-07-05 |

### Why `gen/now` (from issue evidence)

`gen/now` is the generation contract's "improves the current product, operator
loop, or trunk hygiene with a clear witness and no dependency on a future
architecture bet" stream, and its worked example is literally *"add a report
field that immediately helps the operator decide today's dispatch lane, with a
captured command output as witness."* #2042 is exactly that shape:

- It **improves the current operator loop today**: it collapses the N clustered
  per-metric spam posts on `FAK_SCOREBOARD_CHANNEL` (scoreboard 08:17, capacity
  08:47, backlog 09:11, + watchdog errors inside an 80-minute window) down to
  one delta-gated digest the operator can read in a glance.
- Its witness is a **captured command output**: `fak rollup --slack --dry-run`
  resolves channel/token and prints what it would post, sending nothing —
  mirroring the existing `tools/fleet_slack_status.py --dry-run` discipline.
- It has **no future-architecture dependency**: the consolidation substrate
  already exists and ships today (`cmd/fak/rollup.go` + `internal/execrollup`
  fold to one GREEN/WATCH/RED page with a ranked what-needs-you list), and the
  Slack config resolver already exists (`internal/slackenv` — env-then-file
  token/channel lookup reused by every outbound publisher). The change is a
  `--slack` post path plus a delta-gate over already-shipped parts.
- Both moves are **additive and reversible** and inert-by-default: `--dry-run`
  sends nothing; the delta-gate only *suppresses* posts; de-clustering repoints
  existing `FAK_*_CHANNEL` feeders, which the routing decision already frames as
  additive/reversible
  ([`docs/decisions/scorecards-channel-routing.md`](../decisions/scorecards-channel-routing.md),
  "post on a real delta" + "graduating a metric is additive and reversible").

### Why not the other streams

- **Not `gen/next`**: it needs no new schema, no default-exposure gate, and no
  dogfood run before agents/operators can rely on it. `--slack` is an explicit
  operator flag over shipped folds; there is no default-off runtime gate to
  earn. (The `chat.update` upsert-a-pinned-digest *stretch* named in the issue
  Notes — which needs a persisted `ts` store — is the one part that could edge
  toward `gen/next`; it is a stretch on this child, not the core ask, and does
  not move the child's stream.)
- **Not `gen/second-next`**: no cross-generation compatibility policy, adapter
  conformance, or simulation — it composes existing single-repo seams.
- **Not `gen/future`**: the next action is code against named seams, not
  research, a hardware witness, or a market/standards analogue.

## Promotion evidence (what closes #2042 / confirms `gen/now`)

- `fak rollup --slack --dry-run` runs, resolves channel + token via
  `internal/slackenv` (env `FAK_ROLLUP_CHANNEL` or the existing dispatch
  channel, scoreboard token), prints the resolved target and the rendered digest
  it *would* post, and sends nothing — captured command output as the witness.
- A GREEN roll-up byte-identical to the last posted one is suppressed by the
  delta-gate; a WATCH/RED verdict, or any changed verdict / what-needs-you list,
  posts. Witnessed by a focused Go test in `cmd/fak` (or `internal/execrollup`)
  that folds two inputs and asserts post-vs-suppress on the delta.
- Observed operator-surface before/after: ops-channel post count per day drops
  to ~1 digest (+ true escalations), with the per-metric streams still reaching
  their own `FAK_*_CHANNEL` rooms.

## Demotion / retirement evidence (what pushes #2042 back or closes it)

- The delta-gate cannot be made honest without persisting the last-posted digest
  somewhere durable, and that store needs its own schema + dogfood gate before it
  is safe-by-default → the delta-gate slice demotes to `gen/next`; the plain
  `--slack --dry-run` post path can still land as `gen/now`.
- The clustered feeders turn out to already route off the shared channel (the
  "spam" evidence in #2038 is stale after a sibling repoint) → the de-cluster
  half is a no-op and retires; only the digest-post half remains.
- The `fak rollup` fold itself is superseded by a different consolidation owner
  from the #2038 epic (pillar 2/3 lands a new report shape) → retire the
  `--slack` post into that owner rather than double-posting.

## Invalidating assumption

This classification assumes the issue's "spam" evidence — several per-metric
cards clustered on `FAK_SCOREBOARD_CHANNEL` inside an 80-minute window — still
reflects the *live* channel routing and has not already been de-clustered by a
sibling under epic #2038 (e.g. an ask-2/ask-3 child) landed after 2026-07-05. If
the feeders already route to their own rooms, move 2 (de-cluster) is done and
#2042 collapses to just the digest-post half. A future worker must refresh
`gh issue view 2042`, re-read the current `.github/workflows/*-feed.yml` channel
targets, and confirm `fak rollup` has not already grown a `--slack` path before
dispatching from this note.

## Smallest next increment (advisory, for a code-lane worker)

The full feature is two moves; a defensible smallest honest slice, deferring the
rest, is **move 1 only, dry-run first**:

1. Add `--slack` + `--dry-run` flags to `runRollup` (`cmd/fak/rollup.go`).
   Resolve channel via `slackenv.Lookup("FAK_ROLLUP_CHANNEL")` (fall back to the
   existing dispatch channel key) and the scoreboard token via the same
   resolver. On `--dry-run`, print the resolved channel/token-source and the
   rendered `execrollup.Render(r)` body, and **send nothing** — mirror
   `tools/fleet_slack_status.py --dry-run`. Gate the test on that captured
   output (channel resolved, body rendered, zero network).
2. Add the delta-gate: hash the rendered digest (verdict + what-needs-you list),
   compare against the last-posted hash from a small state file, and post only
   on a real delta or a non-GREEN verdict. Witness with a focused fold test
   asserting suppress-on-identical-GREEN and post-on-changed-verdict.
3. Move 2 (de-cluster) is a separate, purely-config increment: repoint the
   `FAK_*_CHANNEL` feeders that currently default onto `FAK_SCOREBOARD_CHANNEL`
   to their own rooms, per the migration path in
   [`docs/decisions/scorecards-channel-routing.md`](../decisions/scorecards-channel-routing.md).

Do not wire a live `chat.postMessage` ahead of the `--dry-run` path and its
delta-gate — an ungated digest post reintroduces exactly the spam the issue
removes. The `chat.update` upsert-a-pinned-digest form (issue Notes) is a stretch
for a later increment, not this slice.
