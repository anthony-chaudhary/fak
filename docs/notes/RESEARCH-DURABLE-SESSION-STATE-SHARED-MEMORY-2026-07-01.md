---
title: "fak research: durable session-state recording to shared memory — use cases, SOTA, and the kill-safe session (2026-07-01)"
description: "Why long agent sessions feel brittle, why nobody wants to stop one that is 'almost there', what the field (durable execution, agent memory systems, practitioner folk remedies) does about it, what fak already ships, and the composed design that closes the gap: a per-turn journal, a residual meter, a restart-parity probe, and a witnessed promotion gate into an arbitrated shared store."
---

# Durable session state → shared memory: the kill-safe session

> Date: 2026-07-01. Status: research + design note — no code ships with this note; every
> proposal names its next checkable step. Companions:
> [PORTABLE-SESSION-IMAGE-AND-SNAPSHOT](PORTABLE-SESSION-IMAGE-AND-SNAPSHOT-2026-06-24.md),
> [VERIFIED-RESUME-PACKET-CHECKPOINT (#636)](VERIFIED-RESUME-PACKET-CHECKPOINT-2026-06-26.md),
> [BUDGET-TRIGGERED-SESSION-RESET](BUDGET-TRIGGERED-SESSION-RESET-2026-06-25.md),
> [SESSION-CONTROL-STATE-AS-FIRST-CLASS](SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md),
> [MEMORY-VIEW-CONTRACT (#904)](MEMORY-VIEW-CONTRACT-2026-06-26.md),
> [CONTEXT-IS-NOT-MEMORY](../CONTEXT-IS-NOT-MEMORY.md),
> [COMPACTION-VS-INDUSTRY](COMPACTION-VS-INDUSTRY-2026-06-25.md).

## 0. The problem, stated as feelings

The operator named three feelings, and they are the right spec:

1. **Brittleness.** Hours of orientation — the task model, the decisions, the dead ends —
   live in one context window. One crash, one bad compaction, one overflow and it is gone.
2. **"Almost there."** Nobody wants to stop a long session that is close to done, because
   stopping *feels* like losing the accumulated context. So degraded sessions get pushed
   past the point where a fresh session would outperform them.
3. **"Losing the context."** The fear is not losing the *transcript* (transcripts persist);
   it is losing the *distilled working state* — which the transcript contains only
   diffusely, and which nothing else holds at all.

All three are one structural fact: **state whose only copy is a context window is a
liability, and no instrument reports the size of that liability.** The "almost there"
feeling is a sunk-cost fallacy only when the state *is* recoverable elsewhere; when it
isn't, the reluctance is rational. The fix is therefore not a better resume command — it
is making the liability continuously small and continuously *measured*, so that stopping
is provably cheap. Sessions end; recorded state persists.

## 1. Use cases — what "record state to shared memory" must actually serve

Seven distinct readers, with different write triggers and trust requirements. Designs that
serve only one of these (e.g. crash recovery) miss the rest.

| # | Use case | Write trigger | What must be written | Reader | Trust rung needed |
|---|---|---|---|---|---|
| U1 | Crash recovery | *before* the crash (periodic/per-turn — a Stop hook never fires on a crash) | pointer to transcript + where work stood | fresh session / human | witnessed > self-report |
| U2 | Planned handoff / budget reset | chosen boundary (compact, budget, end-of-turn) | carryover: durable facts, task recap, next action, verbatim tail | successor session | self-report acceptable, graded better |
| U3 | Parallel peers sharing state | continuous | claims, leases, progress facts | concurrent sessions | must be non-forgeable (`dos_status` has no `claimed` field by construction) |
| U4 | Cross-host / cross-model migration | on demand | full logical session image, no model-specific state | another host/model | integrity-checked (sha256, fail-closed) |
| U5 | Durable-fact promotion | the moment a value crosses into durable store | facts that stay true (truth-duration gate) | all future sessions | provenance + write-time classification |
| U6 | Operator supervision | continuous | liveness, progress, region, resume plan | human / supervisor loop | ledger-verified only |
| U7 | Post-hoc audit | every adjudication | verdict trace | reviewer / referee | append-only, witnessed |

## 2. SOTA readout (mid-2026, sourced)

### 2.1 Durable execution — the record-once discipline

The workflow-engine world solved "long process survives crashes" a decade ago and is now
being applied to agents wholesale. Temporal (official OpenAI Agents SDK and LangGraph
integrations) persists every non-deterministic result — each LLM call, each tool result —
to an event history and *replays* code against recorded results after a crash
([temporal.io/blog](https://temporal.io/blog/announcing-openai-agents-sdk-integration)).
DBOS checkpoints each step's output as a Postgres row — a library, no server
([docs.dbos.dev/architecture](https://docs.dbos.dev/architecture)). Restate journals every
non-deterministic step per invocation and pins executions to the deployed code version so
a months-suspended agent resumes against the code it started on
([restate.dev](https://docs.restate.dev/ai/patterns/durable-agents)). Inngest memoizes per
step. Azure Durable Task checkpoints at every `await` and bounds unbounded agent loops
with continue-as-new
([learn.microsoft.com](https://learn.microsoft.com/en-us/azure/durable-task/sdks/durable-agents-microsoft-agent-framework)).

The convergent invariant, across all five: **record the expensive non-deterministic result
once, at the seam where it happens, and never re-ask during recovery.** The open
weaknesses are also convergent: intra-step loss (work inside a long step dies with the
process), external side effects that cannot be rolled back, code/model version drift
across suspend gaps, and log corruption exactly at the crash boundary.

Framework checkpointing is the weaker cousin: LangGraph writes a full state snapshot per
super-step keyed by thread (with time-travel forks) but nothing *within* a node
([langchain docs](https://docs.langchain.com/oss/python/langgraph/persistence)); AutoGen
`save_state()` is DIY-cadence and famously saves an empty dict for custom agents unless
overridden; Letta takes the strongest position — agent state lives in a database *always*,
so there is no serialization step to forget
([letta.com](https://www.letta.com/blog/ai-agents-stack/)).

Research is pushing the process-snapshot line: CRIU-based sandbox checkpoint/rollback with
inter-checkpoint deltas (DeltaBox, [arXiv:2605.22781](https://arxiv.org/abs/2605.22781)),
copy-on-write forking of live environments (TClone,
[arXiv:2605.17320](https://arxiv.org/abs/2605.17320)), agents-as-OS-processes with scoped
checkpoint/fork/commit-to-image (Agent libOS,
[arXiv:2606.03895](https://arxiv.org/abs/2606.03895)), and the security caveat that a
checkpoint-restore rewinds local state but not external side effects (ACRFence,
[arXiv:2603.20625](https://arxiv.org/abs/2603.20625) — already triaged in
[ACRFENCE-SOUND-RESTORE](ACRFENCE-SOUND-RESTORE-2026-06-25.md)). Event-sourced agent state
is being formalized (ESAA, [arXiv:2602.23193](https://arxiv.org/abs/2602.23193); "The Log
is the Agent", [arXiv:2605.21997](https://arxiv.org/html/2605.21997) — content-addressed
caches so replay makes zero new model calls).

### 2.2 Agent memory systems — what gets written, when, by whom

The products have converged on four moves ([Letta sleep-time
compute](https://arxiv.org/abs/2504.13171), [Mem0](https://arxiv.org/abs/2504.19413),
[Zep/Graphiti](https://arxiv.org/abs/2501.13956),
[A-MEM](https://arxiv.org/abs/2502.12110), [LangMem](https://www.langchain.com/blog/langmem-sdk-launch),
[Anthropic effective-context-engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents),
[Manus lessons](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)):

1. **Write distilled facts, not transcripts.** Extraction → ADD/UPDATE/DELETE adjudication
   (Mem0), atomic linked notes with retroactive evolution (A-MEM), size-capped in-context
   blocks continuously rewritten (Letta).
2. **Move writes to background/idle compute** — Letta's sleep-time agent, LangMem's
   background manager, ChatGPT's consolidation pass. Write quality stops competing with
   the live task for attention.
3. **Small always-loaded index over on-demand detail** — Claude Code's MEMORY.md pointer
   index is the shipped instance of the pattern.
4. **Externalize to files/ledgers because compaction is destructive** — Manus's file
   system as restorable memory (drop page content, keep the URL) and todo.md recitation;
   Cognition's finding that models develop *context anxiety* near the window limit and
   spontaneously dump summary files ([inkeep.com/blog/context-anxiety](https://inkeep.com/blog/context-anxiety)).

And the field's two named open problems are exactly the ones that matter here:
**staleness** (only Zep has principled temporal semantics — bi-temporal edges that
invalidate rather than delete) and **recall-time verification plus multi-writer conflict
handling — essentially nobody does either.** A memory write is an unaudited privilege
escalation from one conversation into all future ones (AgentPoison,
[arXiv:2407.12784](https://arxiv.org/abs/2407.12784); MINJA,
[arXiv:2503.03704](https://arxiv.org/abs/2503.03704)), and every mainstream store handles
concurrent writers by last-write-wins or git merge.

### 2.3 Practitioner reality — the pain is documented, not vibes

Compaction data loss is endemic and first-hand: "I lost 3 hours of Claude Code work to
compaction" with 200+ corroborating comments
([dev.to/gonewx](https://dev.to/gonewx/i-lost-3-hours-of-claude-code-work-to-compaction-never-again-468o)),
auto-compact firing mid-refactor and the agent then editing functions explicitly marked
dangerous, Codex resume rehydrating *the wrong summary* and continuing the wrong task
([openai/codex#8310](https://github.com/openai/codex/issues/8310)). Long-context
degradation ("context rot") is measured across all tested frontier models, so "restart
early, restart often" is now standard advice — yet users still push rotted sessions
because "starting over feels like giving up"
([mindstudio.ai](https://www.mindstudio.ai/blog/context-rot-ai-coding-agents-explained)).
The folk-remedy ecosystem is exactly a hand-rolled version of §2.1 + §2.2: handoff
documents written before quality degrades, `progress.md` "under 50 lines, written for the
next agent session", commits as checkpoints, manual `/compact` at chosen boundaries with
preserve-instructions, JSONL transcript backups. What practitioners say is still missing:
trustworthy summaries, a signal for *when* to checkpoint, and cross-session/parallel
shared memory.

The sharpest practitioner distinction: crash loss is usually *perception* (transcripts
persist; resume exists), but compaction loss is *real* (destructive, unrecoverable without
external snapshots). The feeling of brittleness is calibrated to the worst case, and the
worst case is real.

## 3. What fak already ships — the honest map

fak is unusually far along here; the pieces exist but are not composed into one write path.

| Piece | What it does | Rung | Gap |
|---|---|---|---|
| `tools/session_checkpoint.py` | periodic + Stop-hook off-host record: git snapshot, transcript pointer, private/public scrub router | crash floor (U1) | `in_flight` is a self-report; cadence is coarse; #636 designs the witnessed overlay but the fold is unbuilt |
| `internal/trajectory` | per-turn witnessed verdict trace (JSONL export) | audit (U7) | records *kernel verdicts*, not task state |
| `dos_status` digest | `{liveness, progress, region, resume}`, no `claimed` field by construction | peer/supervisor truth (U3, U6) | not folded into the checkpoint yet (#636) |
| `internal/sessionimage` + `internal/snapshot` | portable, integrity-checked, model-agnostic session image; dump/restore across hosts and models; quarantine survives the boundary | migration (U4) | on-demand, not continuous; nothing triggers a dump automatically |
| `internal/sessionreset` (`BuildSeed`) | budget-triggered carryover: durability-classified facts + task recap + warm-prefix descriptor + verbatim tail, auto-continue | planned handoff (U2) | fires only on budget exhaustion; carryover quality is ungraded |
| `internal/ctxplan` + `internal/recall` | O(1) resident view over a lossless content-addressed store; page-fault recovery for elided spans | the "context is a cache" substrate | store is per-session; not the shared cross-session surface |
| `internal/ctxmmu` truth-duration gate | write-time classification: ephemeral expires, durable persists ([CONTEXT-IS-NOT-MEMORY](../CONTEXT-IS-NOT-MEMORY.md)) | promotion policy (U5) | not wired to the *shared* memory store's write path |
| `internal/memview` (#904) | typed views with digest+span provenance, taint inheritance, invalidate-on-digest-change | staleness answer | contract shipped; the shared store's records don't carry it yet |
| `dos_recall` | re-verifies a recalled memory's claims against git + working tree at read time | recall-time verification — the thing §2.2 says nobody does | mis-aimed at the forked store; not on the default recall path |
| `dos_arbitrate` / lanes | multi-writer admission by disjoint file trees | multi-writer conflict handling — the other thing nobody does | memory tree is not a lane; concurrent sessions write the store unarbitrated |
| agent memory (MEMORY.md + files) | distilled facts, pointer index, mirrored off-host | the shared store itself (U5) | write trigger is agent judgment mid-task — exactly the SOTA weakness; store is forked (two divergent roots) |

Read the table's last column top to bottom and the diagnosis writes itself: **every
weakness is a missing composition, not a missing primitive.** fak even owns the field's
two "nobody does this" items — recall-time verification and multi-writer arbitration —
as shipped, tested mechanisms that simply aren't pointed at the session-state problem.

## 4. The gaps, precisely

- **G1 — Write cadence.** The only writer that survives a crash is the periodic tick
  (coarse, minutes); the Stop hook never fires on a crash; memory writes happen when the
  agent thinks of it. Nothing writes at the *turn boundary* — the natural seam where
  durable-execution systems journal, and the same quantum
  [SESSION-LIFECYCLE-IS-SERVING-ADMISSION](SESSION-LIFECYCLE-IS-SERVING-ADMISSION-2026-06-26.md)
  already names as the session's admission/preemption unit.
- **G2 — No residual meter.** Nothing measures how much live-window state is *not yet*
  recoverable from durable stores. The "almost there" feeling has no number, so the
  stop/continue decision is made on anxiety instead of data. (`tools/ctxwin.py` measures
  window *composition*; nobody measures window *liability*.)
- **G3 — Promotion is judgment, not policy.** Durable facts reach shared memory when the
  agent remembers to write them — the exact write-trigger weakness the SOTA products have.
  The truth-duration gate exists (`ctxmmu`) but the shared store's write path doesn't run it.
- **G4 — The shared store is unarbitrated and unverified.** Concurrent sessions
  last-write-win the memory tree; records carry no provenance digest, so `dos_recall`-style
  re-verification has nothing mechanical to bind to; and the store itself is forked
  (`.claude` vs `.claude-gem7-netra`), so even the verifier aims at the wrong root.
- **G5 — Resumability is never graded.** Checkpoints, carryover seeds, and handoff notes
  are written and *trusted*. No referee ever checks that a fresh session, given only the
  durable state, actually reaches the same next action the live session would have taken.
  This is the repo's core discipline — a claim is not shipped until witnessed — applied
  nowhere to the claim "this checkpoint suffices to resume."

## 5. The proposal ladder

Ordered so each rung is independently shippable and independently checkable. P1–P3 attack
the feelings directly; P4–P5 make the shared store trustworthy; P6 is hygiene that gates P4.

**P1 — Turn-boundary session journal (the WAL).** Extend the checkpoint writer with a
per-turn append: `{turn_seq, next_action, task_delta, decisions[], artifacts_touched[]}`,
sourced per #636 from witnessed columns (`dos_status`, trajectory tail) with the
self-report demoted to a labelled fallback. The durable-execution invariant, imported:
record once at the seam, never re-derive during recovery. A crash then loses at most one
turn, and U1/U2/U6 all read the same journal instead of three different reconstructions.
*Next step:* the #636 fold (`--witnessed --run-id`) plus a `--per-turn` mode writing to the
private route; grade the write cost (it must be O(delta), not O(transcript)).

**P2 — The residual meter and the kill-safe verdict.** Define
`residual(session) := live-window content not reconstructable from (git ∪ journal ∪
shared memory ∪ ctxplan store)`, reported as a token count and a share of window. Expose
it continuously (exit summary, `/metrics`, the status line). A session with residual ≈ 0
is `KILL_SAFE`; a session accumulating residual gets an alarm *before* auto-compaction
fires, not after. This converts the brittleness feeling into an instrument — and note the
inversion: today's tooling measures how *full* the window is; the liability is how much of
it is *unrecorded*. *Next step:* a definition test over real transcripts (`ctxwin.py`
extended with a reconstructability column) before any live wiring; the ~56% tool_result
share already measured is the upper-bound candidate for "cheaply reconstructable".

**P3 — The restart-parity probe (the referee for "almost there").** The claim "I can't
stop, I'd lose the context" is checkable: spawn a throwaway fresh agent from the durable
packet alone (journal + carryover seed + repo), ask it for its next action, and compare —
via the shipped `resume.FoldNextAction` fold — against the live session's declared next
action. `PARITY` means stopping is provably cheap; `DIVERGED` names exactly what the
durable state is missing (and *that* delta is what P1 should journal next). This is fak's
disinterested-referee stance aimed at the sunk-context fallacy: replace the feeling with a
verdict a worker cannot forge. It also directly answers the practitioner ask ("knowing
when to checkpoint" / trustworthy summaries) that §2.3 found unmet in the field — graded,
not trusted, summaries. *Next step:* offline harness first — replay N historical sessions,
cut them at k% marks, measure parity vs. the recorded continuation; publishes a curve of
"how early could this session have safely stopped."

**P4 — The witnessed promotion gate.** Every write into shared memory runs the `ctxmmu`
truth-duration classifier (ephemeral → refuse promotion, it expires with the session) and
lands as a `memview`-shaped record: source digest + span, producer, taint, freshness.
`dos_recall` then has a mechanical binding to re-verify at read time, closing the
write-poisoning / staleness loop end to end — write-time classification, provenance at
rest, verification at recall. No mainstream memory product has all three; fak has all
three *pieces* shipped. *Next step:* one contributor in the memory write path that emits
the provenance envelope; a test that a stale record (source digest changed) is refused at
recall rather than injected.

**P5 — Shared memory as an arbitrated lane.** The memory tree becomes a `dos` lane;
concurrent sessions take shared leases to append, exclusive to consolidate/rewrite.
Consolidation (the MemGPT/Letta sleep-time rewrite) runs as a background loop under an
exclusive lease, so live sessions never race the rewriter. This applies the repo's
existing answer for code trees to the store the field leaves last-write-wins. *Next step:*
add the lane to `dos.toml`; the consolidation loop is `memory-compact` (already a skill)
run under the lease.

**P6 — Heal the fork.** One canonical store; the other becomes a read-only mirror or is
retired; `dos_recall` re-aimed at the canonical root. Gates P4/P5 — provenance and
arbitration over a forked root verify the wrong thing. *Next step:* diff the two roots,
pick survivors, leave a tombstone pointer in the retired one.

## 6. What this buys, in the operator's terms

- **Brittleness** → the crash floor moves from "last periodic tick, self-reported" to
  "last turn boundary, witnessed." The liability is bounded and the bound is visible.
- **"Almost there"** → a number (residual) and a verdict (restart parity). When parity
  holds, stopping costs nothing *and you can prove it*; when it diverges, the diff says
  exactly what to journal before stopping. The rational core of the reluctance is honored
  — and then dissolved by construction.
- **"Losing the context"** → the context window becomes what the KV cache already is in
  [PORTABLE-SESSION-IMAGE](PORTABLE-SESSION-IMAGE-AND-SNAPSHOT-2026-06-24.md): *a cache,
  not state* — rebuildable from the journal + store, so compaction and resets stop being
  destructive events and become cache evictions.

## 7. Honest fences

- Nothing in §5 is shipped by this note. P1's schema exists as a design (#636); P2/P3 need
  their offline definition tests before any live wiring claims value.
- P3's parity check compares *next actions*, not full mental state; `PARITY` is evidence,
  not proof, that the checkpoint suffices. Over-summarized handoffs that pass a shallow
  parity check but re-explore dead ends are a known failure mode — the probe must include
  at least one "what would you NOT do" question to catch lost negative knowledge.
- The residual definition (P2) has a hard sub-problem: "reconstructable" is
  model-relative. The offline harness must fix a reconstruction procedure before the
  number means anything.
- Off-host writes ride the existing scrub router (`session_checkpoint.py`'s private/public
  gates) — P1 adds cadence, not new leak surface, but the per-turn volume raises the cost
  of a gate bug; the needle scan stays mandatory on every append.
- External side effects remain outside every checkpoint scheme (the ACRFence lesson): a
  resumed session must re-verify the world, not assume the journal's last view of it.
