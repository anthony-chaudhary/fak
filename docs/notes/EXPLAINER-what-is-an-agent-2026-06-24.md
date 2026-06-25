---
title: "What is an agent? The parts the word smears, and the seams that pull them apart"
description: "A disambiguation map. The word 'agent' conflates roughly a dozen separable concerns; fak gives each a typed seam, the way an OS named process, thread, and file descriptor. The horizontal complement to the five-loop ladder."
---

# What is an agent? The seams the word smears

> Date: 2026-06-24.
> Scope: a mapping document, not a benchmark. It pins vocabulary against shipped
> seams and is honest in §6 about which seams are wired and which are foundations.
> It is the **horizontal** complement to
> [`engineering-is-building-loops`](../explainers/engineering-is-building-loops.md):
> that doc decomposes an agent by **scale** (nested loops); this one decomposes it
> by **part** (the components inside any one loop) and then names the **cuts** the
> word collapses. Read together they are a 2-D map.

## 0. The word is doing too much work

Ask three engineers what "the agent" is and you get the model, the loop around the
model, and the running session. None is wrong. That is the problem. "Agent" has
become a smear word: it names a brain, a control loop, a lifetime, a fleet, and a
product, and a sentence rarely says which.

There is a precedent for fixing this. "A program" used to smear together the
process, the thread, the address space, the file descriptor, and the scheduler,
until an operating system gave each one a name, a type, and a syscall. After that
you could swap a thread without forking a process, share an address space across
threads, and revoke a descriptor without killing the program. The names were the
engineering.

fak's bet is that an agent kernel does the same job for "agent." If a tool call is
a syscall, then an agent is not one thing; it is a small set of separable parts
held apart by typed seams, and keeping them apart is what lets you swap, gate, and
prove each independently. This doc is the parts list and the cuts.

## 1. Five questions hide inside one word

"What is an agent" is not one question. The field, and fak's own docs, answer it at
least five different ways, and they blur because each reaches for the same tired
words ("layer", "loop", "level"). Pin the five apart first, because most confusion
is a category error between two of these, not a wrong answer inside one.

| Decomposition | The question it answers | The units | Owned by |
|---|---|---|---|
| by **scale** | how much runs in one address space, and how many of each | inner / turn / session / fleet / RSI | [`engineering-is-building-loops`](../explainers/engineering-is-building-loops.md) |
| by **part** | what components one agent is built from | engine, history, view, prompt, tool, loop, gate, … | **this doc** |
| by **optimization target** | what you change to make it better | skill / trajectory / substrate | [`RESEARCH-three-layers`](RESEARCH-three-layers-of-agent-optimization-2026-06-24.md) |
| by **memory layer** | how the KV / agent-memory is organized | routing / addressing / fusion / semantics | [`MEMORY-LAYERS-EXPLAINER`](../MEMORY-LAYERS-EXPLAINER.md) |
| by **integration layer** | where you wire it in | gateway / kernel-ABI / backend | [`agent-integration-architecture`](../fak/agent-integration-architecture.md) |

These axes are orthogonal. A "turn loop" (a scale) is not a "trajectory layer" (an
optimization target); the four-layer KV memory (an organization) is not the
five-loop ladder (a scale). When someone says "the agent layer," ask which of the
five they mean. The rest of this doc develops the **part** axis, then the cuts that
run across all of them.

## 2. The parts list

One agent decomposes into five groups of parts. For each part: a one-line
definition, the fak seam that *is* it (cited by symbol and test, never by line
number — line numbers rot), and the single sharpest "it is not."

### A. Substrate — what persists across turns

- **History.** The lossless, append-only record of everything that happened. Two
  shipped surfaces: the running `agent.Message` list (roles `system/user/assistant/tool`)
  and the content-addressed `ctxplan.Store` of `Span`s the planner views; the
  durable decision twin is the hash-chained `internal/journal` ledger.
  *It is not the context window* — the window is a rendering of this.
- **Session.** The durable lifetime of one agent run, carrying a typed drive state
  (`session.State`: `RunState`, `Budget`, `Priority`, `Pace`, `Rev`) read once per
  turn by `session.Table.Decide`, dumpable to and restorable from a `recall` core
  image via `Table.Restore`. *It is not a turn* (it spans many), *and not identity*
  (the TraceID is only the key it is filed under).
- **Identity.** The stable handle for *who* a task is: a deterministic signature
  over `(project, cwd, canonical first directive)` from `tools/task_identity.py`,
  plus the share/taint provenance on `abi.Ref` (`ShareScope`, `TaintLabel`).
  *It is not the session* — fifteen re-homed sessions of one `/goal` collapse to
  one identity; fifteen distinct goals under a byte-identical preamble must not.

### B. View — what the model sees this turn

- **Context window.** The O(1) materialized view of tokens fed to the model now,
  re-derived each turn as a 0/1 knapsack (`ctxplan.Optimize`) over candidate spans
  under a token budget, producing a `Selected`/`Elided` partition realized by
  `ctxplan.Materialize`. *It is not the history, and not lossy compaction* — an
  elided span keeps a page-back-in handle, proven by `TestFaithfulnessVsCompaction`
  and `TestDemandPageServesAnElidedSpan`.
- **System prompt.** A standing instruction expressed as *tokens* the model may
  ignore: `agent.SystemPrompt` lowered to a `RoleSystem` message and force-resident
  via `ctxplan.Forecast.Pins`. *It is not the gate* — pinning guarantees the bytes
  stay in the window, never that the model obeys them.
- **Skill.** A named, versioned *procedure* — a `SKILL.md` playbook with
  front-matter, loaded into context as instruction-data, that composes existing
  tools. The kernel face is a procedural-memory cache (`contextq.SkillContextRecord`,
  HIT on identical re-invocation per `TestSkillProcedureMemoryHitOnReInvocation`).
  *It is not a tool* — it sits above tools and emits them.

### C. Compute — what proposes

- **Engine.** The replaceable inference driver that turns a `ToolCall` into a
  `Result`: `abi.EngineDriver`, registered by id (`abi.RegisterEngine`), selected
  per call by the `ToolCall.Engine` route string the kernel reads in `routeFor`.
  Swapping `mock → cassette → inkernel → remote` is one route string; the loop runs
  byte-identically across engines (`TestAgentLoopOnInKernelEngine`).
  *It is not the agent* — it is the brain the agent submits to.
- **Intermediate output.** One planner turn is a single `agent.Completion` whose
  `Message` carries *three* separately-typed payloads at once: reasoning
  (`Thinking`/`RedactedThinking`), an action proposal (`ToolCalls`, raw args kept
  verbatim), and a final answer (`Content`). `TestAnthropicPreservesThinkingBlocks`
  shows all three parsing off one response. *It is not one undifferentiated reply* —
  the loop routes each kind differently.

### D. Control — what disposes and drives

- **Tool.** A single named entry in the operation table: a `Tool` string on a
  `ToolCall` the kernel dispatches through one `Syscall`. *It is not a procedure* —
  the model never calls the effector directly; it emits a name the kernel dispatches
  after gating.
- **Tool-call loop.** The harness-owned iteration (`agent.RunArm`) that asks the
  model for one completion, appends it, and branches: tool calls get executed and
  fed back, a tool-call-free turn is the final answer. *It is not the model* —
  `Planner.Complete` is a stateless one-turn oracle; the agency lives in the loop,
  proven by `TestRunArmTurnBudgetCapsRun` (the loop, not the weights, honors a
  budget).
- **Adjudicator (the hard gate).** The in-kernel reference monitor at the syscall
  boundary: the model proposes a call, the kernel disposes a `Verdict` by folding a
  rank-ordered chain through a default-deny lattice, citing one of a closed
  12-reason vocabulary (`abi.CoreReasonCount == 12`, `TestEmptyPolicyDefaultDeny`).
  *It is not the system prompt* — a prompt asks; this is code on the call path the
  model cannot route around.
- **Quarantine (the observe gate).** The write-time gate every *result* folds
  through before its bytes may enter context (`ctxmmu.MMU.Admit`, `VerdictQuarantine`);
  a held page returns only after a witness `Clear` and a re-screen (`TestPageInGatedByClear`).
  *It is not the call-side adjudicator* — that gates whether the tool runs, pre-call;
  this gates whether the produced result enters context, post-call
  (`TestQuarantineDistinctFromDeny`).

### E. Plurality — more than one

- **Communicator.** When there is more than one agent, "multi-agent" smears two
  disjoint rank spaces. One moves bytes: `model.DistComm`, a real cross-process
  tensor collective (`AllReduceSum`/`AllGather` over a socket, bit-exact in
  `TestDistCommAllReduceSumMatchesLocal`). One moves zero bytes: independent agent
  sessions coordinating only through file-tree leases and witnessed commits (the
  dispatch loop), with `modelroute.ReduceAllReduce` borrowing the `all_reduce` name
  for a scalar fold and explicitly disowning the tensor meaning in its own comment.
  *They are not the same primitive* — one is a process group; the other is a
  coordination pattern.

```
                 THE PARTS OF ONE AGENT  (the horizontal axis)

   A. SUBSTRATE        B. VIEW             C. COMPUTE        D. CONTROL
   what persists       what the model      what proposes     what disposes/drives
                       sees this turn
   +-------------+     +-------------+     +-----------+     +---------------+
   | history     |     | context     |     | engine    |     | tool          |
   | session     | --> | window      | --> | (model)   | --> | loop          |
   | identity    |     | system prompt|    |           |     | adjudicator   |
   |             |     | skill       |     | intermed. |     | quarantine    |
   +-------------+     +-------------+     | output    |     +---------------+
        |              (a VIEW over        +-----------+          |
        |               substrate)           (PROPOSE) -----------+ (DISPOSE)
        |                                                         |
        +----- E. PLURALITY: many of the above, two rank spaces --+
               (DistComm moves bytes | agent communicator moves none)
```

## 3. The cuts that run across the parts

The parts list tells you *what* the pieces are. The real clarity is in the binary
distinctions the single word "agent" collapses. Each cut below is a line the kernel
draws with a type, and a power you only get once the line exists.

1. **Substrate vs view.** History is lossless and durable; the context window is a
   lossy, re-derived projection of it. The field calls both "context." The cut is
   `ctxplan.Store` versus `ctxplan.Optimize`'s output. *Power: you can drop a span
   from the window without losing it* — it pages back on demand.
2. **Soft steering vs hard gate.** A system prompt and a skill are tokens the model
   may ignore; the adjudicator is code it cannot bypass. The word "steering" covers
   both. *Power: where a prompt can only make a request, a gate makes a guarantee.*
   This is the load-bearing cut; most "safety" claims live or die on which side
   they are on.
3. **Propose vs dispose.** The model's output is a proposal; the action is a
   disposition. "The agent did X" smears them. `Submit` adjudicates before any
   engine or network is touched, and a denied call never reaches dispatch. *Power:
   you can prove an action that should not have run never ran.*
4. **Replaceable vs durable.** The engine is swappable per call; identity and
   session persist across the swap. "Which agent / which model" smears them.
   *Power: you can change the brain without losing the agent.*
5. **Reasoning vs action vs answer.** Three kinds of intermediate output, not one
   "response." Only the action is adjudicated; reasoning is echoed but never
   executed; the answer is terminal only when no tool calls ride with it. *Power:
   you never confuse "the agent thought X" with "the agent did X."*
6. **Primitive vs composed.** A tool is one syscall; a skill is a procedure over
   tools. "Capability" smears them. *Power: the gate has a small, fixed surface to
   reason about — it sees tool calls, not playbooks.*
7. **Turn vs session vs identity.** One model round-trip, the durable lifetime that
   spans many, and the stable name for who is running — three things, one word
   ("session"). The session note already splits *cold resume* (re-attach a context
   image) from a *warm flip* (`PAUSED → RUNNING`). *Power: you can dial a live run
   without restarting it, and re-home it without renaming it.*
8. **Drive vs audit.** Drive state is what happens *next* and is writable; the audit
   plane is what already happened and is read-only. Both get called "session state."
   They meet at the TraceID and never overlap. *Power: control and forensics do not
   contaminate each other.*
9. **One rank space vs the fleet rank space.** A tensor collective moves activations
   between ranks; the agent communicator moves no payload between sessions. *Power:
   you cost and reason about "distributed" correctly instead of importing GPU
   intuitions into a lease-coordinated fleet.*
10. **Residency vs reuse vs legality.** The cache trichotomy: where bytes sit, whether
    two requests may share computed state, and whether that sharing is still legal.
    [`SCALING-LAWS §6`](SCALING-LAWS-OF-AGENTS-2026-06-19.md) already pins this
    ("the word cache names at least seven different things"); it is the same move as
    this doc, applied to memory. *Power: "the cache is hot" stops being mistaken for
    "this reuse is safe."*

## 4. Glossary: seven words that mean several things each

The clearest single artifact is a list of the overloaded terms with their sharpest
split. These are not hypothetical; each sense below is a real type or doc in this
tree.

- **agent** → (1) fak itself, the *kernel*; (2) the external untrusted AI loop fak
  sits in front of, the *guest*; (3) `fak agent`, the CLI verb for fak's own demo
  loop; (4) the `internal/agent` package, which is mostly wire servers and the loop.
  The trust-framing rule: the kernel is the reference monitor, the guest is the
  untrusted program. Never let one word name both.
- **session** → (1) `model.Session`, a token *decoder* over a KV cache; (2)
  `recall.Session`, a reloaded *core image*; (3) `session.Table`/`State`, the live
  *drive state* (the canonical one); (4) `gateway.SessionState`, its *wire* form;
  (5) `agent.SessionPlanner`, a per-session *context planner*. Qualify always:
  decoder, core image, drive state, planner.
- **context** → (1) the *window*, a fixed token budget; (2) the *view*, what the
  planner made resident this turn; (3) the *admission target*, what "enters context"
  means at the `ctxmmu` gate. "Fits in the window" is not "entered the view."
- **model** → (1) the *routed LLM*, a provider+weights the policy selects (really an
  engine binding — `model-routing.md` concedes "it selects an engine, not a
  receiver"); (2) `internal/model.Model`, fak's own *in-kernel transformer*. "fak
  runs a model" (owns weights) is not "fak routes to a model" (picks an endpoint).
- **memory** → (1) *working memory*, the KV cache; (2) *durable/recall memory*,
  cross-session facts that earn promotion; (3) *procedural memory*, a cached skill
  view. "Working memory" should only ever mean the KV cache.
- **tool vs skill** → a *tool* is an effect-bearing call the adjudicator gates per
  invocation; a *skill* is a host-side procedure, upstream of the gate, that issues
  tool calls. fak gates tools, not skills.
- **steering** → (1) *loop steering*, a deny disposition redirecting the loop
  (defensive, kernel-owned); (2) *planner bias*, benefit signals over which spans go
  resident; (3) *adversarial steering*, an attacker's prompt pushing the model.
  Reserve "steer" for the first; call the others "bias" and "manipulate."

## 5. The map, and what the seams buy

Place the part axis (§2) against the scale axis (the loops doc) and you get the
2-D map. A given seam lives in one cell: the adjudicator is *control at the inner
ring*; the session table is *substrate at the session ring*; the communicator is
*plurality at the fleet ring*.

```
            CONCERN (this doc, the horizontal)  -->
 SCALE      substrate     view          compute       control        plurality
 (loops,    -----------   -----------   -----------   ------------   -----------
  vertical)
 inner      Ref/taint     -             EngineDriver  adjudicator     -
 turn       Message log   ctxplan view  Completion    loop / ctxmmu   -
 session    session.Table recall image  route policy  Decide gate     -
 fleet      identity      -             -             dispatch        DistComm /
                                                      witness         agent comm.
 RSI        journal       -             -             shipgate        -
```

Keeping the seams apart is not tidiness; each separated cut is a concrete power the
smeared word cannot give you:

- swap the engine without losing the agent (engine ⊥ identity/session);
- make a guarantee where a prompt can only make a request (soft ⊥ hard);
- prove a denied action never ran (propose ⊥ dispose);
- dial a live session without a restart (drive ⊥ audit);
- cost and cache correctly (residency ⊥ reuse ⊥ legality);
- never read "thought X" as "did X" (reasoning ⊥ action).

## 6. Honest fences

This is a map of seams at different maturities, not a claim that every part is
shipped and wired. The load-bearing gaps, stated plainly:

- **The context window is a view, but the live loop that feeds it is off by
  default.** `ctxplan` is proven as a pure planner; the seam that feeds a real model
  is guarded (`FAK_CTXPLAN_SEAM`, `TestCtxSeamIsOffByDefault`) and the buffered turn
  is the only one planned.
- **History as a lossless content store is a seam, not the live default.** The
  shipped lossless surface for a live session is the linear `agent.Message` list; the
  durable `journal` records the decision stream, not full message content; the
  content-addressed `ctxplan.Store` adapters are a higher tier.
- **The hard gate is only as hard as where it sits.** A vDSO fast-path runs before
  the chain, so a fast-path-served tool skips the monitor; `DeniedTools()` exists to
  lint exactly that hole.
- **The observe gate is real; its detector is an evadable floor.** `ScreenBytes`
  catches fixed secret shapes and literal injection strings; the strong
  `abi.SemanticScreen` seam ships inert (no registered caller).
- **Session drive state is consumed on the harness loop, not yet the served turn.**
  An operator `POST …/pause` records and reads back; the gateway's served passthrough
  turn does not yet gate on `Decide`. Priority is a field with no contended scheduler.
- **Identity is a Python module, not an `abi` Go type.** `task_identity.py` answers
  "who is this task"; `abi.ShareScope`/`Taint` answer "what may its results be shared
  as." They are not wired into one principal, and the primitive is not yet the sole
  consumer at every call site.
- **The agent communicator has no first-class Go type.** The zero-byte rank space is
  a lease-and-witness pattern in the dispatch loop, not an abstraction the way
  `DistComm` is.

None of these weakens the map. The point of naming a seam is precisely to say where
the line is and how far the wiring has reached.

## 7. Related

- [`engineering-is-building-loops`](../explainers/engineering-is-building-loops.md) —
  the **scale** axis (the five nested loops). This doc is its horizontal complement.
- [`SCALING-LAWS-OF-AGENTS`](SCALING-LAWS-OF-AGENTS-2026-06-19.md) — the cost terms
  of a fleet, and the cache trichotomy (§3 cut 10).
- [`SESSION-CONTROL-STATE-AS-FIRST-CLASS`](SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md) —
  the drive state behind cuts 7 and 8.
- [`RESEARCH-three-layers`](RESEARCH-three-layers-of-agent-optimization-2026-06-24.md) —
  the **optimization-target** decomposition (skill / trajectory / substrate).
- [`MEMORY-LAYERS-EXPLAINER`](../MEMORY-LAYERS-EXPLAINER.md) and
  [`CONTEXT-IS-NOT-MEMORY`](../CONTEXT-IS-NOT-MEMORY.md) — the memory and
  context/memory cuts.
- [`agent-integration-architecture`](../fak/agent-integration-architecture.md) — the
  **integration-layer** decomposition (gateway / kernel ABI / backend).
