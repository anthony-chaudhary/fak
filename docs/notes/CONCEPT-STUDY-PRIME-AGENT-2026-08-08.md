# Concept study — PrimeIntellect-ai/prime-agent (2026-08-08)

**Source:** <https://github.com/PrimeIntellect-ai/prime-agent>
**Pinned:** `a18809e00ea30638584d87b3afea7285a9d7296c` (2026-08-07, `add privacy-safe agent analytics (#521)`)
**License:** MIT (`LICENSE:1` — © 2025 Mario Zechner, © 2026 Prime Intellect). fak is
Apache-2.0, so MIT→Apache vendoring is permitted with attribution; every borrow below is
still routed **INSPIRE** because the source is TypeScript and the value is the technique.
**Entry point requested:** the `prime-agent` launch post. `WebFetch` is refused at the
capability floor here (`TRUST_VIOLATION`), so the post itself was never read — the repo it
names was acquired instead and every claim below is grounded in code, not the pitch.

## What prime-agent is

An MIT-licensed TypeScript coding/research agent built on two abstractions: the **RLM**
(persistent IPython as the model's single tool; subagents and context are function calls
in a REPL) and the **Continual Harness** (supplemental prompts, memories, skills, and
subagent specs as durable state the agent refines with small evidence-backed edits).
Hard fork of `pi-mono`; ~7.2k stars.

## Their worldview (reconstructed, with evidence)

- **Built for long-horizon research evals, not just chat.** `README.md:81` — "especially
  for evaluations in research". The autonomous loop refuses the agent's own self-report:
  *"Do not end the session yourself; the verifier/evaluator decides completion"*
  (`packages/coding-agent/src/core/autonomous.ts:46`).
- **They optimize attach/boot latency.** `scripts/boot-bench.mjs`,
  `bench-daemon-startup.mjs`, `bench-attach-bytes.mjs`; the daemon protocol carries a
  `slim_attach` capability that omits summary/messages for thin clients
  (`src/modes/daemon/daemon-protocol.ts:283`). Their user detaches and reattaches often.
- **Explicit non-goal: sandboxing.** `README.md:65-66` — worker/kernel isolation is
  "**not** a security sandbox". Trust is assumed at the repo boundary. This is the single
  largest divergence from fak's kernel/admission posture and it explains several rows below.
- **Local-first blast radius.** Refinement state is session-local by default; global is an
  explicit, policy-gated act (`src/core/refinement/refinement.ts:140-148`).
- **Convergent evolution worth noting:** their `AGENTS.md:209-256` independently reinvents
  fak's shared-tree commit discipline — commit only your own paths, never `git add -A`,
  never `git stash`/`reset --hard`/`clean -fd`, never force-push. Two projects reaching the
  same rules from the same multi-agent-one-worktree pressure. No borrow; a validation.

## Read coverage

Serial deep read (the skill's parallel fan-out was not used — a standing session
instruction bars spawning subagents unless the operator asks; depth was met by direct
reading instead).

**Opened:** `README.md`, `AGENTS.md`, `LICENSE`; `src/core/` —
`prompt-admission.ts`, `session-lease.ts`, `orphan-process-journal.ts`, `output-guard.ts`,
`resolve-config-value.ts`, `side-question.ts`, `rlm-max-depth.ts`, `autonomous.ts`,
`context-tree.ts`, `telemetry.ts`, `agent-messages.ts`, `compaction/{compaction,
branch-summarization}.ts`, `refinement/refinement.ts`; `src/modes/daemon/`
—`mutation-drain-latch.ts`, `daemon-protocol.ts`; `src/core/kernel/boot-gate.ts`;
`packages/ai/` layout + `test/cross-provider-handoff.test.ts`; `scripts/` inventory;
`skills/` inventory.

**Completeness critic — skipped with justification:** `packages/tui` (74 files) and
`src/modes/interactive` (73) are terminal rendering; fak's surface is a Go CLI + MCP with
no TUI to borrow into. `examples/extensions` (109) are samples, not load-bearing.
`packages/ai/src/providers/*` are TS provider adapters superseded by fak's own Go
accounts/routing layer — but their *test matrix* names the edge classes and one row below
comes from it. `prime-agent-runtime` (4 Python files) is a thin IPython kernel shim.
A first pass had also left the RLM kernel, compaction, and agent-messaging unopened; the
critic caught that and they were opened before reading stopped.

## Candidate table

Witness is **on-axis**: a hit on the umbrella capability was not accepted as PRESENT.

| # | Borrow | Source `path:line@a18809e0` | Axis | Their-worldview reason | Witness (fak seam) | Route | Filed |
|---|---|---|---|---|---|---|---|
| 1 | Lock/lease identity = `pid` + process **start time** | `src/core/session-lease.ts:140,177,249` | Proving the *original* holder is gone vs. a PID recycled onto another process | Daemon sessions outlive the terminal; a wedged lease strands a saved session forever | **PARTIAL** — fak guards PID reuse by *image name* (`internal/safecommit/lockprobe.go:117-136`), and the allowlist (`pwsh`,`node`,`claude`,`git`,`bash`,`cmd`) is the ambient fleet process set, so a recycled PID reads committer-like and the lock becomes permanently un-reapable | inspire | **#5892** |
| 2 | Worktree-effect no-progress (status + `diff --binary HEAD` + untracked content hash) | `src/core/autonomous.ts:296-311,370,374,421` | Futility measured on the **artifact**, not the output stream | They forbid agent self-reported completion, so they need an oracle that cannot be narrated | **PARTIAL** — `internal/gateway/noprogress.go:47-57` has exactly two verdicts, both keyed on planner-output digests / refused calls; a loop with *varying* output and *admitted* calls that writes nothing trips neither | inspire | **#5893** |
| 3 | Orphan-process journal: fsync'd append log of spawned PIDs + start-id, identity re-checked before kill | `src/core/orphan-process-journal.ts:20-88` | Never SIGKILL a recycled PID | Kernel/worker processes must be reapable after a crash without collateral kills | **UNWITNESSED** — my first guess (`internal/orphanscan`) was wrong: that is a dead-*code* detector (`internal/orphanscan/orphanscan.go:1-11`). Real seam is `internal/procguard/deadowner.go`, not yet read | inspire | not yet — needs its own witness; named as a non-scope second site in #5892 |
| 4 | Cross-provider context-handoff test matrix: every provider's transcript (thinking + tool_call + tool_result) fed to every other provider | `packages/ai/test/cross-provider-handoff.test.ts:1-25` | Does a transcript *authored by* model A remain *consumable by* model B (tool-call-id formats, thinking blocks) | They ship many providers and users switch models mid-project | **UNWITNESSED** — fak rotates accounts (`accounts_rotate`, `accounts_launch -rotate`) and has session portability (`internal/sessionimage`, `internal/resume/stopped`), but whether any test feeds a model-A transcript to model B was not established. Do **not** file as a gap until witnessed | inspire | not yet |
| 5 | Memory writes default to **session-local**; global is explicit and policy-gated | `src/core/refinement/refinement.ts:32,140-148,893-895` | Blast radius of a written lesson | A wrong global lesson poisons every future session; local keeps a bad inference cheap | **UNWITNESSED** — fak has ~10 memory packages (`memgate`, `recall`, `fleetmemory`, `memorystability`…); scope-defaulting not yet checked against any | inspire | not yet |
| 6 | Invertible memory edits: `before`/`after` snapshot per edit + mechanical `rollbackProposal` | `src/core/refinement/refinement.ts:85-102,804-836` | One-command revert of a lesson that turned out wrong | Self-modification is only safe if it is reversible | **UNWITNESSED** | inspire | not yet |
| 7 | Auto-refine: cheap yes/no reviewer gates the expensive planner, triggered on turn-interval **or at compaction** | `src/core/refinement/refinement.ts:110-121,949-998` | Capturing a lesson at the moment context is about to be discarded | Compaction is when knowledge is provably lost; ask then, cheaply | **UNWITNESSED** | inspire | not yet |
| 8 | Memory entries carry `evidence` + `expectedOutcome` | `src/core/refinement/refinement.ts:50-57,78-83` | A memory that is **falsifiable** — a later pass can grade whether the lesson helped | Closes the self-improvement loop instead of accumulating unfalsifiable advice | **UNWITNESSED** — fak's memory convention has *Why:*/*How to apply:*, which is rationale but not a testable prediction | inspire | not yet |
| 9 | Bounded agent-to-agent messaging: family reach (parent/sibling/child only), token-bucket rate limit, max-pending backpressure, size cap, `steer` vs `follow_up` delivery | `src/core/agent-messages.ts:12-24` | The *envelope* that makes inter-agent steering admissible rather than a trust hole | Agents orchestrate each other without routing through the user | **PARTIAL — corrected on the second pass.** The provisional DIVERGENT rested on a wrong premise: `SendMessage`→`TRUST_VIOLATION` is *this harness's* session policy, not fak's design. fak ships its own A2A bus — `internal/a2achan/a2achan.go:96` (`DefaultQueueCap = 1024`), refusing as a value at `:159-161`. It bounds **depth**, not **rate**: no token bucket, no per-pair key, no refund, and the `abi.ReasonRateLimited` token there is documented as an "honest in-set mapping" onto the closed vocabulary rather than a real limiter | inspire | **#5915** |
| 10 | Own-vs-total token attribution across a subagent tree, no double-count, cumulative across compaction | `src/core/context-tree.ts:14-81` | "Which subtree burned the budget" without parent/child double counting | Recursive subagents make naive totals meaningless | **UNWITNESSED** — fak has ctxresidency + microagent budget work (epic #3229) | inspire | not yet |
| 11 | `!command` credential indirection (config value is a shell command; stdout is the secret, cached) | `src/core/resolve-config-value.ts:17-23,68-86` | Credential **by reference** rather than at rest | Users keep keys in `op`/`pass`/keychain, not in a config file | **UNWITNESSED** — fak has `accounts` API-key storage + `accounts_launch -apikeyenv` | inspire | not yet |
| 12 | Dependency `min-release-age=7` cooldown, enforced in `.npmrc` **and** dependabot | `AGENTS.md:44-48` | Supply chain: never install a package published hours ago | A compromised release is usually yanked within days | **UNWITNESSED** | inspire | not yet |
| 13 | Capability-negotiated daemon protocol: `DAEMON_PROTOCOL_VERSION`/`SCHEMA_REVISION`, per-command `minProtocol` + capability, both-direction compat tests | `src/modes/daemon/daemon-protocol.ts:52,60,613-689`; `AGENTS.md:35-42` | Wire evolution without lockstep client/daemon upgrade | Users run an old TUI against a new daemon constantly | **UNWITNESSED** | inspire | not yet |
| 14 | Side-question: clone live context into an ephemeral tool-less agent, answer, discard | `src/core/side-question.ts:42-102` | Introspecting a running agent's context with **zero transcript pollution** | Ask "why did you do that" without perturbing the run | **UNWITNESSED** — fak has `fak_trajquery`; whether it can query a *live* session without appending was not checked | inspire | not yet |
| 15 | Measured kernel-boot concurrency gate with the empirical cap in the comment ("256 collapses to ~28% boot at N=200; core*4 holds 100%") | `src/core/kernel/boot-gate.ts:1-40` | Startup-storm admission bounded by *measurement*, not a guessed constant | Fan-out of 200 subagent kernels is a normal day | **PARTIAL (weak)** — fak bounds fleet fan-out via `min(FAK_MAX_WORKERS, host_cap, seats, dos target)`; the borrow is only the *documented measurement* habit | inspire | no — too thin to ship alone |
| 16 | Runtime stdout purity: stray `process.stdout.write` redirected to stderr | `src/core/output-guard.ts:9-34` | Machine-readable channel purity enforced at runtime, not by convention | A stray print corrupts JSON/RPC mode | **UNWITNESSED** — fak has many `-json` surfaces | inspire | not yet |
| 17 | `git --no-optional-locks` on read-only status/diff | `src/core/autonomous.ts:394,407` | Fingerprinting a tree without taking git's index lock | — | **PRESENT-on-axis** — already fak practice: `internal/commitlane/status.go:316`, `stageddelete.go:213`, `internal/releasestale/releasestale.go:381` | — | no |
| 18 | `prompt-admission.ts` | `src/core/prompt-admission.ts:1-42` | — | — | **FALSE FRIEND** — name collides with fak's admission kernel; it is 42 lines of abort-signal plumbing, nothing to do with capability admission | — | no |
| 19 | Rename-then-delete stale-lease reclaim (never `rm` in place) | `src/core/session-lease.ts:218-230` | A concurrent reader never observes a half-deleted lease | Crash-safe reclamation under concurrency | **UNWITNESSED** — relevant to fak's documented refusal to `rm .git/index.lock`; folded into #5892's context, not separately witnessed | inspire | not yet |

## Honest state of this pass

Two borrows were witnessed hard against fak source and filed (#5892, #5893). **Thirteen
rows are UNWITNESSED** (3-8, 10-14, 16, 19) — captured with their axis and grounded
source anchor, but fak's side was not read, so none of them is either a claimed gap or an
earned dismissal. They are candidates for a follow-up pass, not backlog; the skill's
depth floor was met on the *source* read, not on thirteen fak-side witnesses.

Four rows resolved without filing: one PRESENT-on-axis (17), one false friend (18), one
too thin to ship alone (15), and one provisional DIVERGENT (9) whose tradeoff is stated
but whose fak-side rationale was not read in code — so (9) is a *deferred* dismissal, not
an earned one.

The memory cluster (rows 5-8) is the most valuable unmined seam: scope-defaulting,
invertibility, capture-at-compaction, and falsifiable expected-outcome are four separable
axes on a subsystem fak has ten packages for. That is the natural next study.

## Second pass — same `@a18809e0`, fak-side witnesses (2026-08-08)

A follow-up pass on the identical SHA, aimed at the honest gap the first pass declared:
thirteen rows captured with a source anchor but no fak-side witness. This pass opened the
files the first read missed and witnessed six candidates to a conclusion.

**Newly opened:** `src/modes/daemon/daemon-catalog-process.ts`,
`src/modes/daemon/command-recovery-journal.ts`, `src/core/mcp/mcp-manager.ts`,
`src/core/model-resolver.ts`, `test.sh`, `prime-agent.sh`, and the compaction cut-point
logic in `src/core/compaction/compaction.ts` (opened by pass 1 but never carried a row).

| # | Borrow | Source `path:line@a18809e0` | Axis | Their-worldview reason | Witness (fak seam) | Route | Filed |
|---|---|---|---|---|---|---|---|
| 20 | Crash notice injected **into the transcript** as a hidden `custom_message`: "uncertain … work was not replayed. Inspect external side effects before continuing." | `src/modes/daemon/daemon-catalog-process.ts:268-277` | Whether the **model** — not just the operator — learns that a prior turn was interrupted | Unattended runs with no human to inspect side effects; the agent is the only reader present | **PARTIAL** — fak detects the interrupted turn and lands it on metrics only: `internal/agent/loop.go:463-465` sets `m.ResumedPendingTurn`, commented "Read-only here". The write-ahead machinery is sound (`internal/agent/chat.go:562,864,1034`); the disclosure to the resumed agent is absent | inspire | **#5913** |
| 21 | Credential-free test entry point: stash the auth store under an `EXIT` trap, `unset` ~35 provider env vars, single source of truth named in-file | `test.sh:5-20,26-60`; `prime-agent.sh:26-63`; `AGENTS.md:32` | Whether the suite **can** reach real credentials or spend paid tokens — enforced at the entry point, not per test | Many providers, contributors with live keys, and a stated rule ("Do not use real provider APIs, real API keys, or paid tokens") made satisfiable by a faux provider | **PARTIAL** — `Makefile:91` is `go test ./...` with no fence; hygiene is per-test and opt-in (`internal/fleetaccounts/discover_apikey_test.go:169`, `internal/dispatchtick/main_test.go:13`), and `internal/accounts/credrefresh_test.go:122` documents the need in a comment. The enumeration half already exists in `internal/envconfiglint` | inspire | **#5914** |
| 22 | Per-`(sender→target)` token bucket (3/1000 ms) **refunded on delivery failure**, layered over the size + depth caps | `src/core/agent-messages.ts:12-15,334-365,463-506`; `src/modes/daemon/daemon-mode.ts:5435` | Bounding send **rate** against a *live* peer, vs bounding queue **depth** against a dead one | Shallow star topology of unattended agents where a child looping on `agent_message.send` is a normal failure | **PARTIAL** — refines row 9 above from provisional-DIVERGENT to witnessed. `internal/a2achan/a2achan.go:96,159-161` caps depth at 1024 (closing #3480's growth leak, correctly) but never engages against a peer that drains promptly. No time-based limiter, no per-pair key, no quota to refund | inspire | **#5915** |
| 23 | Recovery refuses to replay: a mutation logged `received` with no `result` is uncertain and is **never re-issued** | `src/modes/daemon/command-recovery-journal.ts:73-89` | What to do with a mutation whose outcome is unknown after a crash | Their daemon mutations are arbitrary side-effecting shell/tool commands with no replay contract, so a replay can double-apply | **DIVERGENT (earned)** — fak answers the opposite way *on purpose*: its write-ahead checkpoint is a typed cursor into a deterministic, audited journal (`internal/agent/chat.go:562,864`, symmetric clear at `:1034`), so re-entering the turn is the correct and replayable behavior. Same problem, opposite answers, each right for its user world. Only the **disclosure** half (row 20) transfers | — | no — record only |
| 24 | Compaction cut point never lands on a `toolResult` | `src/core/compaction/compaction.ts:310-348,397-459` | Never orphaning a `tool_use`/`tool_result` pair across a compaction boundary | A dropped pair makes the next provider call malformed | **PRESENT-on-axis** — and fak fails *safer*: `internal/agent/anthropic_compact.go:456,511,519-520` computes `keptWindowOrphansToolUse` and returns `CompactOutcome{Reason: CompactReasonWindowNoDrop}` — declining to compact at all rather than snapping the boundary | — | no |
| 25 | Cache-read tokens excluded from the autonomous spend budget | `src/core/autonomous.ts:186-194` | Not charging a re-read cache hit against a budget meant to bound real spend | Long-horizon reruns re-read the same context constantly; counting it would terminate healthy runs early | **PRESENT-on-axis** — `internal/cachevaluereport/` exists for exactly this accounting, with the exclusion at `internal/agent/chat.go:278` | — | no |
| 26 | Confused-deputy defense: a user-redefined catalog server name does **not** inherit the official server's OAuth token | `src/core/mcp/mcp-manager.ts:113-117,136-138` | A name collision escalating into token theft | Users add MCP servers by name from a shared catalog; a name is not an identity | **PRESENT-by-construction** — fak's `accounts` layer stores an env-var *reference*, never a secret keyed to a server name, so the attack shape cannot arise. Recorded because the *absence* of the vulnerability is structural, not defended | — | no |
| 27 | Absolute reserve-token compaction trigger (`reserveTokens: 16384`, `keepRecentTokens: 20000`), not a percentage | `src/core/compaction/compaction.ts:128-132,229-233` | Trigger stability across models with different window sizes | They route across many providers; a percentage means a different absolute headroom per model | **not yet** — fak's compaction is a budget regime (the ~70% figure), but whether the trigger is proportional or absolute was not read to a conclusion. Do not file as a gap until witnessed | inspire | not yet |
| 28 | Structural `no-silent-catch` lint | `AGENTS.md` lint rules | A swallowed error that never reaches a verdict | — | **not yet** — fak has 10+ lint packages (`codelint`, `boundarylint`, `envconfiglint`, `architest`, `guardaudit`); none was read deeply enough to support either verdict. Honest UNWITNESSED, not an ABSENT | inspire | not yet |
| 29 | `RLM_MAX_DEPTH` defaults to **1** despite the "Recursive LM" framing | `src/core/agent-session.ts:1588-1590` | — | — | **WORLDVIEW-FINDING (note-only)** — the shipped default is a shallow *star*, not deep recursion. Their recursion is a marketing frame over a one-level fan-out, which is also why their per-pair message bucket (row 22) is sized for siblings rather than a deep tree. Useful when reading their other defaults; no fak axis to witness | — | no |

### What the second pass changed

- **Row 9's dismissal was wrong and is retracted.** It rested on "fak refuses subagent
  messaging outright (`SendMessage`→`TRUST_VIOLATION`)" — but that is *this harness's*
  session policy, not fak's design. fak ships `internal/a2achan`, so the correct verdict is
  PARTIAL-on-axis, now filed as #5915. This is precisely the ego-dismissal failure the
  skill warns about, caught only because the second pass read fak's code instead of
  inferring fak's stance from a harness refusal.
- **Two genuinely new borrows** (#5913, #5914) came from files the first read never opened.
- **Three candidates died honestly** — two PRESENT-on-axis (24, 25; on 24 fak is arguably
  stronger) and one PRESENT-by-construction (26).
- **One earned DIVERGENT** (23), with both rationales stated.
- Rows 27-28 are recorded as **not yet witnessed**, not as gaps. Rows 3-8, 10-14, 16 and 19
  from the first pass remain unwitnessed; the memory cluster (5-8) is still the largest
  unmined seam and still the natural next study.

## Companions

- [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) — the per-capability
  witness+file this pass hands off to for rows 3-16.
- [`study-repo`](../../.claude/skills/study-repo/SKILL.md) — the pass that produced this note.
- Filed: #5892 (safecommit lock identity), #5893 (gateway worktree-effect no-progress).
