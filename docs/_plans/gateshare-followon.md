---
name: gateshare-followon
description: "Four independent follow-on tasks from docs/ops-unsticking.md, each self-contained for a headless worker. Disjoint file trees: dos-kernel arbitrate fail-open, refusal-vocab re-base, AGENTS.md byte ratchet, weekly naive-arm control."
---

# gateshare follow-on — four disjoint tasks

Spawn one headless `claude -p` worker per task. All four are **independent** and
touch **disjoint files**, so they can run concurrently on `main` in the same
worktree. Do not let any worker commit a file another worker owns.

| # | Task | Repo | Owns |
|---|---|---|---|
| 1 | MCP `dos_arbitrate` fails open | `dos-kernel` | `dos_mcp/server.py`, arbitrate tests |
| 2 | Re-base `refusal_vocab_size` | `fak` | `internal/heavinessscore/*` |
| 3 | `AGENTS.md` byte ratchet | `fak` | `docs/context-budget/*`, gate config |
| 4 | Weekly naive-arm control | `fak` | new `internal/naivecontrol/*` |

Shared rules for every worker: no `git add -A`, no whole-tree staging, no
force-push, no branch creation, no touching another task's files, never discard a
dirty file you did not create. `docs/ops-unsticking.md` is READ-ONLY for all four —
it is the evidence base, and a worker editing it would invalidate the other three.

---

## Task 1 — MCP `dos_arbitrate` must fail closed (repo: dos-kernel)

**Why.** `dos_arbitrate` over MCP calls
`live_leases(list(live_leases or []))` in `dos_mcp/server.py`. With `live_leases`
omitted it arbitrates against an empty set, so **every lane reads free**. Observed
live: it returned `"cluster lane 'tools' free — admitted"` for a lane
`dos lease-lane acquire` refused the same second
(`dos-kernel#246`). The CLI form reads the WAL and agrees with the authority; only
the MCP tool fails open. An agent calling the convenient tool picks held lanes every
time, and the correctness of the whole admission stack is irrelevant while the
tool it actually calls lies.

**Watch out.** `docs/agentic-issue-dispatch.md` in the **fak** repo records that a
prior fix to this — forcing all three readers to agree — was **retracted by its
author**: the dead pid in a lease is the expected steady state (it names the
short-lived acquiring CLI, not the worker), and making the readers agree would
reintroduce the double-booking regression. Do not "fix" it that way.

**Do.**
1. Reproduce: call `dos_arbitrate` over MCP with no `live_leases` and show it
   admitting a lane the lease store holds.
2. Make the MCP path read the real live set by default, and **fail CLOSED** when
   the store is unreadable. An unreadable store must refuse, not admit.
3. Keep `--leases '[]'` (or the equivalent) as the only way to opt into an empty
   world, and make that opt-in explicit in the tool schema.
4. Add tests: the admitting case must now refuse; the explicit-opt-in empty-world
   case must still admit; an unreadable store must refuse.
5. If you conclude the correct fix is instead to **remove or rename** the tool so
   agents cannot call it, that is an acceptable outcome — say so and argue it.

**Done when.** A test fails on `main` before the change and passes after, and the
observed-live disagreement above can no longer be reproduced. Witness the commit
SHA.

---

## Task 2 — Re-base `refusal_vocab_size` (repo: fak)

**Why.** The KPI reads **0.000 / score 0** against a soft line of 12 while the
vocabulary holds **149** reasons (`grep -cE '^\[reasons\.[A-Z0-9_]+\]' dos.toml`).
It read 0.000 at 86 reasons too. A KPI that reads zero carries no information, so
re-running the meter cannot help — it needs a new scale. This is the one saturated
axis in an otherwise healthy scorecard (Grade B, pressure 139).

**Do.**
1. Read `internal/heavinessscore/heavinessscore.go` and find why the score floors
   at 0 — a ratio that clamps, or a soft line the value exceeds so far that the
   curve is flat.
2. Re-base the KPI so it discriminates across the observed range (12 → ~150) while
   preserving its meaning: more structured refusals = heavier operator surface.
   A log or piecewise scale is fine; the requirement is that 86 and 149 must not
   score the same.
3. Keep the metric name and JSON key stable if a consumer exists — check for
   consumers first (`grep -rn refusal_vocab_size`).
4. Regenerate `docs/OPERATOR-HEAVINESS.md` with
   `go run ./cmd/fak/ operator heaviness --markdown > docs/OPERATOR-HEAVINESS.md`.
5. Tests: assert the new scale separates 86 from 149, and that the soft line at 12
   still scores near the top of the range.

**Done when.** The two historical readings score differently, the doc is
regenerated, and the grade is reported honestly even if it drops. Witness the
commit SHA. **Do not** make the number look better than it is — if the honest
result is a worse grade, that is the deliverable.

---

## Task 3 — Put a ratchet on `AGENTS.md` bytes (repo: fak)

**Why.** `AGENTS.md` is **52,301 bytes / 521 lines**, forced into turn 1 of every
session. There is **no ratchet on its size** — `docs/context-budget/agents-md-floor.md:149`
says so outright. The sibling floors in that same directory DO have ratchets
(`mcp-tool-floor.md` via `floorgate.go`, `skill-description-floor.md` via #5444).
The largest slice is the only unguarded one. History: 5,563 bytes (2026-06-21) →
83,681 (2026-08-23 peak) → 21,538 after the #8705 trim → 52,301 today. It was
**cut 3.9x and regrew 2.4x in 34 days**. Issue #3535 asks for a sectioned loader
to cut the forced read; that loader **does not exist** (the only mention of
"sectioned loader" in the tree is a forward-looking comment at
`internal/promptlint/breath/doc.go:72`).

**Do.**
1. Add an `AGENTS.md` byte floor to the existing `docs/context-budget/` gate
   family, following the `skill-description-floor` pattern: a counted baseline, a
   stable key (`KIND<TAB>path`, a COUNT not a line number), and a hard error on an
   unparseable baseline row rather than a skipped line.
2. Set the floor at the **current** 52,301 bytes so the gate is honest on day one —
   do NOT set it at a pre-trim figure and pretend the debt is already paid.
3. Make the failure message actionable: name the byte count, the overage, and the
   sections most likely to have regrown.
4. Tests: a doc that grows past the floor fails; a doc at the floor passes; fixing
   one of two findings tightens the floor; a corrupt baseline row is a hard error.
5. Separately: report whether the sectioned loader (#3535) is feasible now. Do not
   build it — measure and write a dated note under `docs/notes/` saying what it
   would take and what it would save.

**Done when.** The gate exists, is tested, and `AGENTS.md` is not modified by this
task (this task adds the ratchet; shrinking the file is a different change).
Witness the commit SHA.

---

## Task 4 — Stand up the naive arm as a measured control (repo: fak)

**Why.** Every claim in `docs/ops-unsticking.md` is inference until the simple
path is measured on the same ruler as the elaborate one. The job repo learned this
the expensive way: it relabelled its naive fanout era as `"human-driven ... not a
fair apples-to-apples baseline"`, structurally excluding the simple path's output
from the comparison while scoring the loop's output against a target. A path with
no number cannot be credited when it wins, and cannot be defended when it loses.

The naive arm is one command: `fak issue-orchestrator --top 10 --max-waves 1` —
45 characters, against 54,000 tokens of control-plane `SKILL.md` and 55,232 lines
of Go.

**Do.**
1. Add a `internal/naivecontrol/` package that runs the naive arm on a schedule
   and records, per run: arm, timestamp, picks offered, picks shipped, commits
   landed (git-ancestry verified, not self-reported), wall time, cost. Shipped
   counts MUST come from git ancestry — the job repo recorded "a lying dispatch
   driver whose SHIPPED/BLOCKED tokens were overridden by git-ancestry 5/5 times",
   so a self-reported verdict is not admissible here.
2. Emit the SAME metrics the orchestrated path emits, so the two are directly
   comparable. If a metric does not exist on the orchestrated side, say so rather
   than inventing a proxy.
3. Record a **missing-measurement state** for any field that cannot be read. Do
   NOT default a missing field to 0 — the DOA incident was 350 dead workers
   indistinguishable from background noise precisely because everything read as a
   benign default.
4. Add a single comparison surface that prints both arms side by side.
5. Tests: shipped counts are git-verified; a missing metric renders UNKNOWN not 0;
   the comparison is correct when one arm has no data.

**Done when.** One weekly run of the naive arm produces a row that a reader can
put next to an orchestrated row and draw a conclusion from. Witness the commit
SHA.
