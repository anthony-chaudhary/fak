# Concept study — can1357/oh-my-pi (2026-08-08)

**Source:** <https://github.com/can1357/oh-my-pi>
**Pinned:** `896bf5f33e0b67bdd0cf951c82739a28e75d0823`
**License:** MIT (`LICENSE:1` — © 2025 Mario Zechner, © 2025-2026 Can Bölük). fak is
Apache-2.0, so MIT→Apache vendoring is permitted with attribution; every borrow below is
still routed **INSPIRE**, because the source is TypeScript/Rust and the value is the
technique, not the bytes.
**Epic:** #5917. **Filed:** #5918–#5927 (ten leaves).

## What oh-my-pi is

An MIT-licensed TypeScript/Bun + Rust coding-agent harness — TUI, 60+ providers, 31 tools,
~80k lines of Rust across supporting crates. A hard fork of `badlogic/pi-mono`.

**Same upstream as the prime-agent study** filed the same day
([`CONCEPT-STUDY-PRIME-AGENT-2026-08-08.md`](CONCEPT-STUDY-PRIME-AGENT-2026-08-08.md),
#5892/#5893): both are descendants of `pi-mono`. The borrow sets are disjoint by
construction — that pass took lock identity, no-progress oracles, and the memory cluster;
this one takes isolation cost, the tool-call stream, compaction arithmetic, credentials,
and retry safety.

## Their worldview (reconstructed, with evidence)

- **A full harness for one developer on a laptop.** fak is a management plane wrapped
  around *someone else's* agent. This asymmetry is the single most load-bearing fact in the
  whole study: it eliminated an entire class of otherwise attractive borrows (structural
  summary reads, hashline `[src/foo.ts#0A1B]` anchors, the selector grammar, the fs scan
  cache keyed on full effective `WalkOptions`) — fak does not own the read/edit tools, so
  there is nothing to borrow them into. Recognizing this early is what kept the candidate
  list honest instead of long.
- **Fan-out is normal and disk is scarce.** `crates/pi-iso/src/lib.rs` is one isolation PAL
  with eight copy-on-write backends, because they cannot control the user's filesystem and
  a per-agent full copy is fatal at their fan-out.
- **The model is steerable mid-generation, not only at tool boundaries.**
  `docs/ttsr-injection-lifecycle.md` aborts a tool call while its arguments are still
  streaming.
- **Credentials are substituted, not removed** (`docs/secrets.md`), so the model can still
  operate on a value it must never see.
- **Compaction is priced, not heuristic** (`docs/compaction.md`): a prune below the
  placeholder cost *grows* context, so there is a hard floor.

## Read coverage

Serial deep read. The skill's parallel fan-out was **not** used — a standing session
instruction bars spawning subagents unless the operator asks — so depth was met by reading
18 subsystem design docs end to end plus the highest-value source file directly.

**Opened:** `LICENSE`, `README.md` (map only); `docs/` —
`ttsr-injection-lifecycle.md`, `advisor-watchdog.md`, `compaction.md`,
`non-compaction-retry-policy.md`, `secrets.md`, `memory.md`, `agent-hub.md`,
`task-agent-discovery.md`, `blob-artifact-architecture.md`, `approval-mode.md`,
`fs-scan-cache-architecture.md`, `hooks.md`, `tools/read.md`; and
`crates/pi-iso/src/lib.rs` (the single highest-value read of the pass).

**Completeness critic — skipped with justification:** the TUI and interactive-mode
packages are terminal rendering with no fak surface to land in; the provider adapters are
superseded by fak's Go accounts/routing layer; `crates/pi-iso`'s per-backend files
(`apfs.rs`, `btrfs.rs`, `zfs.rs`, `linux_reflink.rs`, `overlayfs.rs`, `projfs.rs`,
`windows_block_clone.rs`, `rcopy.rs`, `diff.rs`) were **not** read in full — the seam and
the fallback ladder in `lib.rs` are what #5918 needs, and per-backend syscall detail is
work for #5919 when that backend is actually built. That is a deliberate deferral, not a
gap in the decision.

**Fak-side witnesses were verified against the working tree**, not recalled: every seam
`path:line` cited in a filed issue was grepped and confirmed before the issue was created.

## Candidate table

Witness is **on-axis**: a hit on the umbrella capability was never accepted as PRESENT.

| # | Borrow | Source `path:line@896bf5f` | Axis | Their-worldview reason | Witness (fak seam) | Route | Filed |
|---|---|---|---|---|---|---|---|
| 1 | CoW isolation backend behind one PAL, `diff` delegating to `git diff` | `crates/pi-iso/src/lib.rs` | Per-worker tree materialization **bytes**, not creation frequency | Laptop fan-out; they cannot choose the user's filesystem, so eight backends + a universal fallback | **PARTIAL** — `internal/workerworktree/workerworktree.go:359` (`Prepare`) is always a full `git worktree add`, priced at ~450MB/worker (`coldreap.go:11-12`). The warm pool (`pool.go:24-36`, #3572) amortizes *creation*; #3242 attacks *build*; nothing attacks bytes | inspire | **#5918** |
| 2 | Windows `FSCTL_DUPLICATE_EXTENTS_TO_FILE` block clone + probe-and-degrade ladder | `crates/pi-iso/src/lib.rs` | Same axis, on the platform the fleet runs on | A worker that cannot start is worse than one that starts with a full copy | **ABSENT** — no reflink/clonefile/block-clone call site anywhere | inspire | **#5919** |
| 3 | Stream-rule matching over tool-argument deltas, per-tool-call buffer isolation | `docs/ttsr-injection-lifecycle.md` §2 | **How early** a rule can observe a tool call | They treat generation as interruptible, not atomic | **ABSENT** (witnessed twice) — fak's guard is `PreToolUse`-boundary; `sessionsteer` is lifecycle-triggered; `adjudicator` decides at admission | inspire | **#5920** |
| 4 | Mid-stream abort → discard partial → re-prompt, with 4 staleness guards | `docs/ttsr-injection-lifecycle.md` §3–4 | What the model is told when a rule fires | Cancel a doomed call before it costs a full generation | **ABSENT** for the mechanism; the *frame* is a deliberate inversion (below) | inspire | **#5921** |
| 5 | Anti-nag ledger: `once`/`after-gap` in **completed turns**, persisted, restored on resume | `docs/ttsr-injection-lifecycle.md` §5, §7 | Repetition pressure on context | An unbounded rule trains the model to ignore the channel | **PARTIAL** — `sessionsteer` injects at SessionStart / persists at Stop; `goalpark`/`seatpark` bound *waits*, not *advice repetition* | inspire | **#5922** |
| 6 | Reversible secret placeholders: per-install **HMAC** base, case-variant independence, restore before execution | `docs/secrets.md` | Whether the model can **operate on** a credential it must never **see** | Users paste secrets constantly; removal breaks the plan, substitution does not | **PARTIAL** — `internal/secretgate/secretgate.go:1-16` quarantines (removes) under a CAS-pinned handle and delegates detection to the single `canon.SecretPatterns` list. Strong on detection, absent on reversibility | inspire | **#5923** |
| 7 | `AgentToolResult.useless` — the **tool** flags its own output prunable | `docs/compaction.md` | Who knows a result is worthless | Inverts the information flow: one bit from the party uniquely qualified to supply it | **PARTIAL** — `compactcohere` reasons about *when* pruning is cache-safe; no producer-side value signal exists | inspire | **#5924** |
| 8 | `MIN_PRUNE_TOKENS` floor; never cut at a `toolResult` | `docs/compaction.md` | The **arithmetic** of one prune decision | A prune below the placeholder cost grows context *and* churns cache — worse on both axes at once | **PARTIAL** — timing is PRESENT (`compactcohere.go:10` TTL, `:127` `Classify`); the floor and structural cut validation are ABSENT | inspire | **#5925** |
| 9 | `retry.usageAware*` — preflight remaining quota, reserve %, three-way policy, **fail-open** on unknown | `docs/non-compaction-retry-policy.md` | Deciding **before** the spend vs after the 429 | Coding-plan quotas are knowable in advance; spending the last request is avoidable | **PARTIAL** — `internal/accounts/cooldown.go:88,97` parses `resets at` **out of the refusal text**; the signature `ParseReset(message string)` is itself the proof the input is the loss | inspire | **#5926** |
| 10 | Replay-safety gate: visible text/images/tool calls/server-tool blocks block replay; thinking-only and whitespace-only are discardable | `docs/non-compaction-retry-policy.md` | Is a retry safe to **replay** — orthogonal to whether the error was transient | Conflating the two is how a retry duplicates a side effect | **PARTIAL** — `attemptbudget`/`sessionreplay`/`goalpark` are strong and error classification is well developed (#3457), but no *content-shaped* predicate was found. #5927 makes verification its first task | inspire | **#5927** |
| 11 | Advisor emission guard (NFKC → content-free filter → 4096 FIFO dedupe), `immuneTurns`, bounded `syncBacklog` | `docs/advisor-watchdog.md` | Keeping a second reviewer model's advice channel from becoming noise | A watchdog that repeats itself is ignored; one that blocks the primary is worse | **PARTIAL** — the dedupe/backpressure shape generalizes, but fak has no second-reviewer channel to attach it to yet | inspire | no — folded into #5922's rationale; refile if a reviewer channel lands |
| 12 | Advisor **unsafe-output quarantine** — ≥3 hazard classes, or new instruction-override + quoted destructive command → discard the whole turn; 1st silent re-prime, 2nd warn + reset | `docs/advisor-watchdog.md` | Treating your own reviewer model's output as untrusted input | A second model is an injection surface, not an oracle | **Genuinely novel** — no fak analogue; also no fak reviewer channel to defend | inspire | no — record only; the idea outlives the missing surface |
| 13 | `prewalk` — strong model hands off to a cheap model at first edit/write; skipped only when model identity **and** effective thinking level both match | `docs/task-agent-discovery.md` | Cost routing at a **semantic** boundary (plan→implement), not a turn count | Planning is worth a big model; typing the edit is not | **UNWITNESSED** — fak has model routing and roles; whether any handoff triggers on first write was not established | inspire | not yet |
| 14 | `snapcompact` — history rendered to per-model-tuned PNG bitmap frames, priced against vision billing geometry | `docs/compaction.md` | Tokens-per-unit-history under a *vision* price sheet | They own the renderer, so they can arbitrage two billing models | **Genuinely novel; watch** — depends on owning the transcript renderer, which fak does not | — | no — watch item |
| 15 | Blob (content-addressed, global) vs artifact (session-local monotonic int) split | `docs/blob-artifact-architecture.md` | Addressing durable content | — | **PRESENT-on-axis** — `internal/blobfs` is a durable CAS implementing `abi.Resolver`/`abi.CASPinner`/`abi.PageOutBackend`; strictly stronger | — | no |
| 16 | Approval tiers read/write/exec with arg-dependent policy functions + `override` | `docs/approval-mode.md` | Admission granularity | — | **PRESENT-on-axis** — `internal/adjudicator` capability-floor rungs + `internal/laneadmit` lane/tree/lease collision model are richer | — | no |
| 17 | Structural-summary reads, hashline anchors, selector grammar, fs scan cache | `docs/tools/read.md`, `docs/fs-scan-cache-architecture.md` | Read-tool ergonomics | Their agent owns its own read/edit tools | **N/A — DIVERGENT by architecture.** fak is a management plane around someone else's agent and does not own these tools. The tradeoff is deliberate and still holds: owning them would mean owning the harness | — | no |
| 18 | Task depth cap that **removes** the `task` tool at the cap (rather than refusing at call time) | `docs/task-agent-discovery.md` | Making an unavailable capability unrepresentable vs refusable | A tool the model can see but never use wastes tokens and invites retries | **UNWITNESSED** — fak defers cold tools (#3232) and has a toolcall runaway floor (#2887); whether depth removes rather than refuses was not checked | inspire | not yet |

## The one deliberate inversion — and the learning behind it

The source's mid-stream injection is a **prohibition** frame, literally
`<system-interrupt reason="rule_violation" rule="..." path="...">`.

fak already holds the opposite position, and holds it *with a scorer*:
`internal/negframe` backs `fak score negframe`, whose stated law is that steer prose leads
with the affordance, because "a directive framed as a prohibition makes the reader invert a
negative to find the action" (`cmd/fak/negframescore.go:76-78`). Critically, negframe hands
back a positive rewrite **only where the reframe is mechanically unambiguous**, and
deliberately refuses to auto-rewrite judgement-tier negations
(`experiments/negframe-steerability-ab/README.md:44-49`).

Transplanted to the interception channel, that gives fak's rule, which #5921 encodes as a
registration-time constraint rather than a style guideline:

> **Interrupt only when you can substitute the action to take. Where there is no
> unambiguous substitute, do not interrupt.**

A scolding interrupt costs a full generation and returns nothing actionable. So fak ships
the source's lifecycle and race discipline, and refuses its frame. #5921's DoD makes this
checkable by pointing the existing `negframe` classifier at the rendered injection text and
requiring **zero** findings — the discipline becomes a gate, not an opinion.

## Honest state of this pass

Ten borrows were witnessed against fak source and filed (#5918–#5927), under epic #5917.
Every filed issue carries a source `path:line@896bf5f`, a verified fak seam `path:line`,
the axis, the on-axis verdict, the INSPIRE + license verdict, a first checkable step, and
an explicit Definition of Done.

Four rows resolved without filing and are **earned**, not assumed: two PRESENT-on-axis
(15, 16 — fak is stronger), one DIVERGENT-by-architecture with its tradeoff stated (17),
and one watch item that depends on a surface fak does not have (14). Two rows (11, 12) are
real but have no seam to attach to until a second-reviewer channel exists — recorded rather
than filed, so they are not lost. Two rows (13, 18) are **UNWITNESSED**: the source side is
grounded but fak's side was not read, so neither is a claimed gap nor an earned dismissal.

The unmined seam most worth a follow-up is cost routing at semantic boundaries (row 13) —
`prewalk` is one instance of a general pattern fak's management-plane position is unusually
well placed to exploit.

## Companions

- [`study-repo`](../../.claude/skills/study-repo/SKILL.md) — the pass that produced this note.
- [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) — the per-capability
  witness+file this pass hands rows 13 and 18 off to.
- [`CONCEPT-STUDY-PRIME-AGENT-2026-08-08.md`](CONCEPT-STUDY-PRIME-AGENT-2026-08-08.md) —
  the sibling `pi-mono` descendant studied the same day (#5892, #5893).
- Related epics: #3165 (per-worker worktree isolation) parents #5918/#5919; #3229 (context
  budget) is the natural home for #5924/#5925.
