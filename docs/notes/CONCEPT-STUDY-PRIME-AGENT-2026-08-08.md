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
| 9 | Bounded agent-to-agent messaging: family reach (parent/sibling/child only), token-bucket rate limit, max-pending backpressure, size cap, `steer` vs `follow_up` delivery | `src/core/agent-messages.ts:12-24` | The *envelope* that makes inter-agent steering admissible rather than a trust hole | Agents orchestrate each other without routing through the user | **DIVERGENT (provisional)** — fak refuses subagent messaging outright (`SendMessage`→`TRUST_VIOLATION`). Their envelope is designed under an explicit non-sandbox assumption (`README.md:65`), which fak does not share. The *mechanism* (reach bound + backpressure) is still the shape a future admissible design would take. Provisional because fak's refusal rationale was not read in code | inspire | no — record only |
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

## Companions

- [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) — the per-capability
  witness+file this pass hands off to for rows 3-16.
- [`study-repo`](../../.claude/skills/study-repo/SKILL.md) — the pass that produced this note.
- Filed: #5892 (safecommit lock identity), #5893 (gateway worktree-effect no-progress).
