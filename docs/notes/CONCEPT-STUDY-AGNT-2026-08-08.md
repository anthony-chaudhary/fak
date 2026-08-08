# Concept study — agnt-gg/agnt (2026-08-08)

**Source:** `https://github.com/agnt-gg/agnt`
**Pinned:** `66208c56f99ce42a0d167b23da5588180cc72169` (2026-08-08 01:15:40 -0400) — every
`path:line@sha` below resolves against this commit, not against their tree.
**Licence verdict (repo-wide):** **INSPIRE only.** See [Licence gate](#licence-gate).

`AGNT` was ambiguous input. It resolved to `agnt-gg/agnt` — the literal name match and the
nearer domain neighbour to fak (an agent substrate with a kernel, a tool surface and a
context budget). The runner-up reading was the AGNTCY org; it was not studied.

## What was read

Acquisition: shallow clone into the session scratchpad, `--filter=blob:none`, then
`core.longpaths` + `git restore --source=HEAD :/` after the initial checkout aborted on a
long Windows path. Completeness verified by count: 2267 HEAD paths, 2267 files on disk,
zero HEAD paths missing. Clone later deepened to 476 commits so the trajectory read
(step 2.4) was possible at all — `--depth 1` drops the log.

Trajectory over the last 250 commits, by directory churn: `services/orchestrator` 222 ·
`tools/library` 93 · `services/ai` 92 · `services/evolution` 41. The context, tool-surface
and cache work this study borrows from is where they are actively investing, not dead code.

**Completeness critic.** Repo-wide grep of `backend/src` for rationale markers
(`WHY THIS EXISTS`, `DESIGN CONTRACT`, `SAFETY CONTRACT`, `THE PROBLEM`,
`Measured (on|over|against)`, `INVARIANT`) surfaced 40 rationale-bearing modules; every one
with a plausible fak borrow surface was opened. Justified skips: `frontend/src` (823 files,
Vue 3 desktop UI — fak is a Go CLI/kernel with no GUI; the *producer* of their monitoring
panel, `contextManifest.js`, was read instead, since that is the half with a borrow
surface), `electron/`, `mobile/`, `main.js`/`preload.js` (desktop packaging),
`docs/_API-DOCUMENTATION.md` (9,875-line HTTP surface reference), `DESIGN.md` (a pure
visual design system — palette, type, spacing), `stream/example_workflows` (fixtures).

**Deviation from the skill, stated explicitly.** `study-repo` step 2 mandates parallel
`Explore`/`Agent` readers for the deep read. This session's operating instructions forbid
calling the Agent tool unless the user requests it, and the user did not. The depth floor
was met instead by batched-parallel `Read`/`Grep`/`Bash` calls performed directly, followed
by the completeness-critic sweep above. Coverage was met; the fan-out mechanism was not
used.

**Their worldview** (step 2.5, reconstructed from README non-goals, config defaults and the
measurements in their comments): local-first, single-tenant, desktop-app ergonomics, 15+
providers including CLI/OAuth-backed Claude Code and Codex. Stated non-goals: *"a hosted
chatbot, a prompt playground, a SaaS-only automation tool, public multi-tenant hosting,
a zero-config cloud product."* One operator pays the provider bill and sees every
false-positive personally. That single fact explains most of what is worth borrowing:
cache economics are a product surface rather than an optimisation, an over-blocking gate is
a support burden they absorb the same day, and no design may require downloading a model.

A second, method-level observation: their comments carry the *measurement that forced the
design* (corpus size, false-rejection counts, the alternatives that lost), including
negative results. That is what made this repo cheap to borrow from and is worth imitating
independently of any single technique.

## Licence gate

`LICENSE.md` @ `66208c56` — "AGNT Community Core License (AGNT Open License)" v1.0,
self-described as *"Source-Available, Fair-Use Internal-Commercial License"*. It explicitly
permits *"Study and learn from the codebase for educational purposes"* and forbids
*"Rebrand, redistribute, or fork AGNT under another name"*. `package.json` records
`"license": "Custom - See LICENSE.MD"`.

**Therefore: INTEGRATE is forbidden repo-wide and the gate collapses to one answer.** No
bytes may be vendored. Every borrow below is clean-room reimplementation in Go, citing the
source for provenance. Their licence sanctions exactly this use.

## Candidates

Witness classes: **PARTIAL/ABSENT** → filed · **PRESENT-on-axis** → fak already covers the
axis (read, not assumed) · **DIVERGENT** → fak chose otherwise for a reason that still
holds, tradeoff stated.

| # | Borrow | Source `path:line@66208c56` | Axis | Their-worldview reason | fak seam | Witness | Filed |
|---|---|---|---|---|---|---|---|
| C3 | Pick a gate predicate by replaying admitted calls, counting false rejections | `backend/src/services/orchestrator/toolArgGuard.js:1-46` | Predicate chosen by counterfactual replay over calls that **succeeded**, not by taste | An over-block is a same-day support burden for the one operator | `internal/guardaccuracy/complaints.go:20` | PARTIAL | #5896 |
| C1 | Resident-block gates must read only conversation-stable inputs, enforced by withholding the volatile one | `backend/src/services/orchestrator/system-prompts/promptElements.js:16-46` | **Presence**-volatility of a resident block (not value-volatility) | They pay the cache-write premium directly and visibly | `internal/agent/anthropic_cachebp.go:33` | PARTIAL | #5897 |
| C4 | Scope action rules to sink-bearing args; leave data/DLP rules unscoped | `backend/src/services/security/toolCapabilities.js:1-38` | Prose *describing* an action must not match an action rule | Their median payload is prose flowing through the tool surface | `internal/adjudicator/reversibility.go:435` | PARTIAL | #5898 |
| C6 | Persist a per-(provider, model) estimate→billed ratio, non-self-amplifying | `backend/src/services/orchestrator/calibrationStore.js:1-138` | Cross-session learned estimator correction, keyed by provider | 15+ providers; CLI/OAuth ones inject bytes they never see | `internal/agent/anthropic_footprint.go:21` | ABSENT | #5899 |
| C5 | Separate chars-per-token divisors for JSON schemas and prose | `backend/src/utils/contextManager.js:16-32` | One shared ratio mis-sizes the tool surface and cascades into eviction | A 295-tool surface is most of their prompt | `internal/agent/anthropic_footprint.go:34` | PARTIAL | #5900 |
| C13 | Set-based token containment over the existing FTS index for near-dups | `backend/src/utils/memorySimilarity.js:1-30` | Beats embeddings on LLM-paraphrased text with no model, no vector column | A desktop app cannot ask a user to download a model | `internal/gateway/mcp_defer.go:68` | PARTIAL | #5901 |
| C8 | Gate repeat extraction on a structural outcome signature, durations excluded | `backend/src/services/evolution/ExtractionGate.js:1-84` | Suppress the repetitive case with no tunable threshold; never the anomalous one | Personal, visible LLM spend on their own machine | `internal/dojocal/select.go:28` | PARTIAL | #5902 |
| C12 | Per-file EOL reconciliation, mixed-file healing, platform-constant banned by test | `backend/src/utils/lineEndings.js:1-46` | Existing endings win; **mixed** files heal to dominant | Their agent edits the user's own checkout; a flip lands in `git diff` | `internal/agentsindex/agentsindex.go:10` | PARTIAL | #5903 |
| C9 | Pin the recovery/discovery tool at the head of any tool-budget cap | `backend/src/services/orchestrator/toolSelector.js:12-56` | A capability unreachable on a channel configured before it shipped, with no healing path | Saved per-channel tool selections are an enumerated snapshot | `internal/gateway/mcp_defer.go:55` | **PRESENT-on-axis** | — |
| C15 | Unpriceable cost is NULL, never zero | `backend/src/services/execution/LedgerRecorder.js:1-140` | A zero is indistinguishable from "this was free" | Invisible workflow spend was a months-long bug for them | `internal/dispatchtick/admission_edf.go:40` | **PRESENT-on-axis** | — |
| C2 | Chunked eviction to a low-water mark with a persisted monotone watermark | `backend/src/utils/contextManager.js:548-613` | Eviction as a cache-prefix-stability problem | They own and re-render the whole message array each turn | `internal/agent/anthropic_elide.go:1-30` | **DIVERGENT** | — |
| C7 | Separate recurring (per-turn standing) context cost from one-off spend | `backend/src/utils/contextEconomics.js:1-130` | Per-item recurrence attribution ("re-sent 42 times") | One operator reading one panel | `internal/cachevaluereport/` | **DIVERGENT** | — |
| C11 | Hard-deadline process exit independent of drain success | `backend/src/utils/gracefulShutdown.js:1-46` | Once shutdown starts, the process exits regardless of what hangs | Long-lived Electron backend holding Socket.IO/SSE streams open by design | — | **DIVERGENT** | — |

### The three earned dismissals

**C2 — DIVERGENT.** AGNT persists an eviction watermark as a unit count and replays it every
turn, because they own the message array and re-render it; recomputing the minimum cut each
turn slid the window and destroyed the prefix (measured: 27 prefix rewrites over 40 turns,
6 after the fix). fak solves the same problem *at a different layer and arguably better*:
`internal/agent/anthropic_elide.go:12-27` and `anthropic_compact.go` are **request-side wire
transforms** that splice on the original bytes, so the first-breakpoint head prefix is proven
byte-identical and later breakpoints cascade-burst as a documented trade. Adopting a
persisted watermark would put mutable per-conversation state into a transform whose stated
trust property is that it *"never touches the decoded `req.Messages` the kernel adjudicates"*.
The tradeoff is real and fak's side of it still holds.

**C7 — DIVERGENT.** AGNT's `buildEconomics` prices a fixed prefix at cached vs uncached rates
so *"the gap between these two is exactly what a broken prefix costs, per turn"*, and returns
`null` rather than a fabricated `$0.00`. fak already prices cache economics far more heavily —
`internal/cacheprice` (breakeven, dividend, readmission, retention) and
`internal/cachevaluereport` (cohorts, census, billing mode, markdown P&L). The one uncovered
sliver is *per-item recurrence attribution* — "this 15k block has been re-sent 42 times".
That framing serves one human reading one panel; fak's consumer is a fleet scorecard, and the
session/cohort grain is the right one for it. Not filed; recorded here so a later reader can
re-check the call rather than inherit it.

**C11 — DIVERGENT.** Their invariant — *"once shutdown starts, the process exits — within
`hardDeadlineMs` at the very latest, no matter what hangs"* — exists because `server.close()`
waits for every open connection and their app holds SSE/Socket.IO streams open by design,
producing an orphan holding the port and a supervisor misreading it as a crash. They note it
never reproduced on Windows, where `.kill()` is `TerminateProcess` and the handler never runs.
fak is a CLI and kernel, not a long-lived socket server; the pathology has no analogue. fak's
`internal/dispatchaging` hard deadline is a *scheduling* force-serve rung, a different thing
with the same name — which is exactly the kind of umbrella-name collision this study's
witness step is meant to catch.

### Method finding (no code to copy)

Their comments state the measurement that *forced* each design — corpus sizes, false-rejection
counts, and the alternatives that lost, including the embarrassing ones (bag-of-words hashing
scoring worse than a dumb prefix). This is what let a reader grade a borrow at the axis in one
pass. Recorded as an observation about documentation practice, not filed: it clears no
ship-alone bar and fak's rationale comments are already dense in the same style.

### Dogfood note

The capability witness could not use `fak_feature_query`. Three calls, phrased at three
different axes, each returned the whole index (~1.0–1.2M characters, over the response
limit). The entire step-6 witness was redone with raw `Grep` over `internal/` plus
`gh issue list --search`. That is why every ABSENT above was confirmed by reading fak's code
on the seam rather than by a ranker miss. Filed as **#5958** — a *boundedness* defect
(`internal/selfquery/selfquery.go:255-256` caps only when `Limit > 0`, and every default
caller passes `0`), distinct from #5901's *ranking-quality* borrow.

Separately: while this note's sibling issue bodies were being written, a tool call was
refused with `NEVER_AMEND_SHARED` because the *text being written* discussed history
rewriting. No such operation was in flight. Filed as **#5960**. Adversarial verification
corrected the first reading of it: this is **not** the #5898 pathology. gitgate already reads
only the sink-bearing argument, so it is not an argument-scoping miss — the correct argument
contains prose that `internal/gitgate` lexes as shell (backtick spans, heredoc bodies). #5898
would ship in full and leave the refusal standing; the correction is recorded on #5898.

Three defects in fak's own tooling surfaced this way and were filed after adversarial
verification: **#5958** (feature-query boundedness), **#5959** (`issue_lane_router.py`
accepts a bare `--apply-labels-write` and silently discards it — the reason the eight borrows
above were labelled by hand), **#5960** (gitgate refuses prose about the trunk rules). A
fourth candidate — a gap in this skill's own fan-out wording — was refuted 3/3 and dropped.

## Companions

- [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) — the capability-witness
  discipline this study reused for step 6; each filed issue above is a grounded candidate it
  can grade further.
- [`study-repo`](../../.claude/skills/study-repo/SKILL.md) — the pass this note records.
- Epic **#3229** (context/token budget) anchors #5897, #5899 and #5900.
