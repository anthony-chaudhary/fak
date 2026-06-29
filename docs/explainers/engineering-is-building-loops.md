---
title: "Engineering is building loops; fak is the kernel they run on"
description: "Modern engineering is increasingly the act of building agentic loops. fak is the in-process kernel those loops run on, safe and fast for the same reason."
slug: engineering-is-building-loops
keywords:
  - agentic loops
  - agent kernel
  - tool call as syscall
  - observe orient decide act verify
  - recursive self-improvement
  - issue-dispatch loop
  - context planner
  - addressable KV cache
  - capability gate
  - witness-gated
---

# Engineering is becoming the act of building loops. fak is the kernel they run on.

A decade ago the unit of work was a function. You named an input, named an output, wrote the steps between them, and you were done. The thing either returned the right value or it didn't.

That unit is changing. More and more of the work now looks like a loop: observe, orient, decide, act, verify, and run it again until some condition holds. An agent harness is a loop. A CI bot that keeps proposing fixes until the suite is green is a loop. A dispatch fleet chewing through a backlog is a loop. A system that tunes itself is a loop. The function still exists, but it is now the body of a loop someone else is running.

Here is the problem. Most people build each of those loops by hand on top of a raw model API. The model both proposes the next action and executes it. Context gets rebuilt blindly every turn. Nothing gates the act before it happens. Nothing remembers cheaply across turns. Nothing checks that a "kept" change is actually better. Every loop re-implements the dangerous, expensive parts, badly, in slightly different ways.

A loop is only as trustworthy and as fast as the thing underneath it. That thing is what fak is.

## The loop you already have, and the loop you wish you had

Strip an agent down and the inner cycle is the same five steps every time:

- **observe** — a tool produces some bytes
- **orient** — assemble the context for this turn
- **decide** — is this action allowed?
- **act** — run it
- **verify** — did it help; keep or revert

When you hand-roll this, the model is in charge of all five. It decides what is allowed, so the only safety is "please don't." It picks what goes in context, so the window fills with junk. It grades its own work, so "done" means whatever it says it means.

fak takes the structural steps away from the model and gives them to a kernel. The model still proposes. The kernel disposes. That single boundary is the whole idea, and it is why fak can be both safer and faster at the same time: the same gate that refuses a bad action also lets a known-good one reuse work it has already done.

## Model proposes, kernel disposes: the syscall seam

The lowest loop is one tool call. In fak a tool call is a syscall, not a function the model just runs.

The kernel exposes a typed seam: `Submit` then `Reap` (the `Kernel` interface in `internal/abi/types.go`). `Submit` adjudicates the call and returns a handle and a verdict immediately, before any engine or network is touched (`Kernel.Submit` in `internal/kernel/kernel.go`). `Reap` carries the slow part, the actual engine round-trip, and folds the result through admission (`Kernel.Reap`). The kernel's own package comment is blunt about where the cost lives: "Adjudication happens entirely at Submit and touches neither the engine nor the network."

The decision is made by an in-process reference monitor. No spawned hook, no IPC, the same call stack as the tool invocation (`Adjudicator.Adjudicate` in `internal/adjudicator/decide.go`). Three properties make it trustworthy:

- **Default-deny.** An empty policy, an unknown tool, or every link abstaining all resolve to `Deny` with `DEFAULT_DENY` (`TestEmptyPolicyDefaultDeny`). No affirmative allow means no dispatch.
- **Provable refusal only.** A `Deny` cites one reason from a closed twelve-word vocabulary (`CoreReasonCount = 12` in `internal/abi/reasons.go`). What it cannot prove, it defers rather than guesses.
- **Deny never reaches the engine.** A denied call returns a `DenyResult` on the `Reap` path before `engine.Complete` is ever called (witness `TestDenyNeverReachesDispatch`).

The refusal is bounded too: a self-modify denial returns only the offending glob, not the whole policy or the argument values (`TestSelfModifyDeniedWithBoundedWitness`). And the policy itself is a version-tagged JSON manifest loaded at runtime, so an adopter configures which tools an agent may call by editing one reviewable file, never by forking the kernel (`internal/policy`; CLAIMS.md).

One honest measurement, framed honestly: the in-process decide path runs at about 2,427 ns versus 6.913 ms for a spawned hook doing the same work, roughly 2,849x at n=100 (STATUS.md, BENCHMARK-AUTHORITY.md). That is a subsystem regression sentinel proving the gate is not accidentally paying a process boundary. It is not a throughput or production-readiness number, and the repo says so in the same breath (CLAIMS.md). The point that carries here is narrow and real: making the gate an in-process call instead of a spawn is what lets you put it on every single tool call without the loop grinding to a halt.

A note on the async story, because it is easy to over-read. The ABI freezes a `StatusPending` and a typed `Completion` for future async dispatch (`internal/abi/types.go`), but no current engine returns them. `Reap` blocks on a synchronous `engine.Complete`. The seam is real and frozen; the async operations are not shipped (ARCHITECTURE.md). Worth saying plainly so nobody plans around a future that is not here yet.

## The turn loop: orient and observe without trusting the bytes

Around the syscall sits the turn. Two organs run here.

**Orient** is the context planner (`internal/ctxplan`). It treats this turn's window as an O(1) materialized view over the full, lossless history. Each turn it runs a bounded 0/1-knapsack over candidate spans (`Optimize` in `internal/ctxplan/plan.go`; greedy-density by default, with exact-DP and submodular-coverage objectives also available), keeping pins such as the system prompt, the active goal, and the last user turn as hard constraints, and everything else by benefit. A span it drops is not summarized away; it keeps a content-address and pages back in on a forecast miss (`DemandPage` in `internal/ctxplan/fault.go`). A candidate index bounds the per-turn work from Θ(N²) toward Θ(c·N), measured on an 851-turn replay as 100.1K full-scan candidate-scorings versus 68.0K bounded (CLAIMS.md).

The caveat belongs in the same breath: the live-loop wiring is a guarded seam, off by default (`FAK_CTXPLAN_SEAM`, `internal/agent/ctxplan_seam.go`). The numbers come from a real transcript replay, not a live-serving benchmark.

**Observe** is the result quarantine (`Admit` in `internal/ctxmmu/mmu.go`). Every tool result passes a write-time gate before it can enter context. Poison or secret-shaped bytes are held out and replaced in place with a stub pointer; oversize benign results page out to a sub-2KB pointer (CLAIMS.md). A held page can only be paged back in after an explicit witness `Clear()` and a fresh re-screen of the bytes, so a clearance alone cannot launder poison, even across a process restart (`TestQuarantineSurvivesTheSessionBoundary`).

This is where safe and fast turn out to be the same mechanism. When the gate quarantines a result, a deeper bridge evicts that result's K/V span from the attention cache (`AdmitResult` in `internal/kvmmu/kvmmu.go`), and the cache ends up bit-identical to a run that never saw the span: max|Δ| = 0.0 against a non-vacuous poison control near 0.33 (`TestWriteTimeEvictEqualsNeverSaw`; CLAIMS.md). The trick is that RoPE is linear in position, so survivors after the evicted span are re-rotated by the position delta and land exactly where a fresh prefill would put them (`internal/model/kvcache.go`). The same boundary that contains the poison is the one that lets the cache reuse the clean prefix.

The big honest caveat lives here. The architecture is sound — zero leaks after quarantine — but the detector fak inherits is roughly 100% evadable and false-positive prone on real context: one real session sealed 2 of 59 pages, both benign base64 images (CLAIMS.md). The load-bearing guarantee is the capability floor plus the containment, not detection (CLAIMS.md). And recurrent-state hybrid caches such as Gated-DeltaNet cannot quarantine mid-span at all; there the boundary fails loud with a typed error rather than silently corrupting (`RecurrentEvictUnsupportedError` in `internal/model/kvcache.go`).

## Loops all the way down

Step back and the same observe/decide/act/verify shape repeats at five sizes. Each layer hands the next a primitive it can trust.

- **Inner — the tool call.** Primitive: the syscall seam plus the adjudicator. Model proposes via `Submit`, kernel disposes via the verdict, denial never reaches the engine. (`internal/kernel`, `internal/adjudicator`)
- **Turn — one agent step.** Primitive: ctxplan orients, ctxmmu observes. Context is a planned O(1) view; results are gated before they enter it. (`internal/ctxplan`, `internal/ctxmmu`)
- **Session — many turns over time.** Primitive: the KV cache and the durable core-dump. A finished session is a page table over a content-addressed swap, and a quarantine survives the process boundary. (`internal/recall`)
- **Fleet — many sessions in parallel.** Primitive: witness-gated dispatch. Capped workers resolve issues, and an issue closes only on a per-SHA `dos commit-audit`, never on self-report. (`tools/issue_resolve_witnessed.py`, `docs/dispatch-loop.md`)
- **RSI — the loop that improves the loop.** Primitive: the non-forgeable keep-bit. A change is kept only on measured gain AND a green suite AND clean truth, all derived from runs the loop performs itself. (`internal/shipgate`, `internal/rsiloop`)

The diagram below is the same idea, nested.

```
                    LOOPS ALL THE WAY DOWN
        (each ring = observe -> orient -> decide -> act -> verify)

  +=========================================================================+
  | RSI LOOP  (the loop that improves the loop)                             |
  |   primitive: non-forgeable keep-bit  shipgate.Evaluate / internal/rsiloop|
  |   KEEP only if  gain AND suite-green AND truth-clean  (all measured)     |
  |                                                                         |
  |  +===================================================================+  |
  |  | FLEET LOOP  (many sessions, one backlog)                          |  |
  |  |   primitive: witness-gated dispatch  docs/dispatch-loop.md        |  |
  |  |   spawn under cap -> ship #N commit -> per-SHA audit -> close     |  |
  |  |                                                                   |  |
  |  |  +=============================================================+  |  |
  |  |  | SESSION LOOP  (many turns over time)                        |  |  |
  |  |  |   primitive: KV cache + durable core-dump  internal/recall  |  |  |
  |  |  |   quarantine survives the process boundary (Clear + rescreen)|  |  |
  |  |  |                                                             |  |  |
  |  |  |  +=======================================================+  |  |  |
  |  |  |  | TURN LOOP  (one agent step)                           |  |  |  |
  |  |  |  |   ORIENT: ctxplan  O(1) view over lossless history    |  |  |  |
  |  |  |  |   OBSERVE: ctxmmu  result gate before context entry   |  |  |  |
  |  |  |  |                                                       |  |  |  |
  |  |  |  |  +=================================================+  |  |  |  |
  |  |  |  |  | INNER LOOP  (one tool call = one syscall)       |  |  |  |  |
  |  |  |  |  |   MODEL PROPOSES ........ kernel.Submit         |  |  |  |  |
  |  |  |  |  |       |                  (adjudicate, no engine)|  |  |  |  |
  |  |  |  |  |       v                                         |  |  |  |  |
  |  |  |  |  |   DECIDE ............... adjudicator verdict    |  |  |  |  |
  |  |  |  |  |       |                  default-deny, 12 reasons|  |  |  |  |
  |  |  |  |  |       v                                         |  |  |  |  |
  |  |  |  |  |   KERNEL DISPOSES ...... kernel.Reap            |  |  |  |  |
  |  |  |  |  |                          deny never reaches engine|  |  |  |  |
  |  |  |  |  +=================================================+  |  |  |  |
  |  |  |  +=======================================================+  |  |  |
  |  |  +=============================================================+  |  |
  |  +===================================================================+  |
  +=========================================================================+

  Same boundary, every ring:  a decision no participant can move by
  narrating a number.  That is what makes each loop SAFE -- and, by reusing
  the work it already trusts, FAST.
```

## The session loop: cache as durable loop state

A session is just the turn loop run many times. Its state is the KV cache, and fak treats a finished session as a core dump (`internal/recall`). The context-MMU already paged every heavy or poisoned result out to a content-addressed store at write time, so what is left is a small page table — roles, digests, quarantine state — over a frozen swap device.

Reload it in a fresh process and the moat holds: a page sealed at write time is refused on page-in unless `Clear()` ran and the bytes re-pass the gate (`TestQuarantineSurvivesTheSessionBoundary`; CLAIMS.md). Two independent gates, so poison cannot be laundered by clearance alone.

There is a durability axis too. Benign results are classified at write time as turn, session, or durable, and only durable facts cross into the persisted core image under `PromotionEnforce` (CLAIMS.md). The default is audit-only (`PromotionWarn`) while callers migrate. This closes the benign over-promotion arm of OWASP Memory-Poisoning T1, where an ephemeral observation silently becomes a permanent bias. It does not close the adversarial arm, which still rests on the same evadable trust gate (CLAIMS.md).

## The fleet loop: dispatch nobody has to trust

Scale out and you get a fleet: many sessions resolving a backlog at once. fak's issue-dispatch loop is a staged pipeline — gate, route, spawn, prompt, witness, close, harvest, surface — driven on a 10/15/30-minute cadence by three scheduled tasks (`docs/dispatch-loop.md`).

The interesting part is that no stage trusts a worker's word. A spawn passes only if the host is safe, an account is free, and the live worker count is under the cap, with any failed check refusing the spawn (`dispatch_preflight.py`). The live worker count is `MAX(kernel lease count, OS process scan)`, so neither a stale lease nor an orphan process can hide load. An issue closes only after the close arm re-runs `dos commit-audit <sha>` per SHA at close time and confirms the commit is reachable from origin/main (`tools/issue_resolve_witnessed.py`). The commit-to-issue link is reconstructed only from the commit text — a closing verb (`close/fix/resolve #N`), `#N` in the subject, or the house `issue #N` noun form — so a fix that names no issue number can never be witnessed-closed (`docs/dispatch-loop.md`).

The headline metric is computed from git evidence, not self-report: `closure_rate = TRUE / (TRUE + CLAIMED)` over `dos commit-audit` verdicts (`docs/dispatch-loop.md`). A closed issue whose commit fails witness stays in `CLAIMED_CLOSED` and never inflates the numerator. A durable curve in `.dispatch-runs/progress.jsonl` — operator-local and gitignored — records every witnessed close, so the count itself is reconstructable from evidence rather than asserted.

Everything is dry-run until `--live` (`docs/dispatch-loop.md`). The honest edge: a silent human close on the same commit could be mis-attributed to the loop, and the opencode backend is single-shot by design, with replan owned by the supervisor, not the worker.

## The loop that improves the loop

The outermost loop tunes fak itself. This is where the keep-bit has to be unforgeable, because the thing being graded is the grader.

There is a one-shot, `cmd/rsicycle`, that takes the four witnesses — before, after, suite-green, truth-clean — as flags. It is honest about being hand-fed. The true loop, `internal/rsiloop` plus `cmd/rsiloop`, derives every one of those witnesses from a run it performs itself (`docs/rsi-loop.md`). It forks a detached git worktree off main, rewrites the tunable, measures the KPI by actually running the probe, takes suite-green from a real build and vet, and takes truth-clean from `git status` (`internal/rsiloop/worktree.go`). The loop author supplies none of them.

The keep-bit lives in one place. `shipgate.Evaluate` sets it only on the conjunction of strict gain, a green suite, and clean truth (`internal/shipgate/shipgate.go`). A zero, unevaluated witness can never report `Kept()` true (`TestKeepBitNonForgeable`). Even a large metric gain is reverted if the suite is red or truth is dirty (`TestKeepBitNeedsAllThree`). Telemetry observers run after the row is journaled and can never re-gate the decision (`TestRunObserved_ObservesEveryVerdictWithoutChangingIt`). The loop never mutates main; a kept change advances only the in-memory baseline and the journal, and landing it is a separate human step (`docs/rsi-loop.md`; CLAIMS.md).

Caveats, stated: only the demo tunable (`DefaultCacheSize`) is wired today; a real subsystem plugs in its own proposer and measurer. The default suite-green check is `go build` plus `go vet`, the Windows-safe proxy, weaker than a full `go test` (which a production run overrides with WSL).

Notice the discipline repeats. The RSI keep-bit, the fleet's per-SHA close, the syscall's provable refusal — they are the same rule at three scales: a decision a participant cannot move by narrating a number.

## Below the tool call: loops inside the syscall

A tool call is not the bottom of the stack. It is the body of smaller loops, and the smallest one runs one token at a time inside the kernel's own address space. The decode loop that *emits* a tool call is `Session.Generate` plus `Session.Step` in `internal/model/kv.go`: prefill the prompt once, then loop, taking `argmax(logits)` for each next token and advancing the KV cache. Greedy, synchronous, owned by fak at the Go call site. The honest edge: this is greedy only. The ABI reserves an async/speculative seam but no engine returns it, and turning it on would re-open the greedy-path proofs.

Under the decode loop is the forward pass: embed, then the attention and MLP layer stack, then a final norm to logits (`Session.token` and `Session.Prefill` in `internal/model/kv.go`). The live decode path can be kernel-selected by flag (f32, Q8_0, Q4_K). The correctness oracle is a separate, cacheless `Forward` (`internal/model/forward.go`) that runs CPU f32 only; it is proven bit-exact against a HuggingFace argmax oracle on SmolLM2-135M, the llama family. Other model families and the GPU and Q8 device paths are held to the weaker argmax-exact plus logit-cosine gate, not bit-identity, and several family oracles are still open for want of fixtures.

Under the forward pass is the KV cache as a first-class kernel object. `KVCache` in `internal/model/kvcache.go` keeps the pre-RoPE keys in `Kraw` so `Evict` can compact the survivors and re-rotate them to their new positions, landing bit-exact to a cache that never saw the evicted span. This is the same RoPE-is-linear trick the turn loop relies on, exposed one rung lower. The boundary is honest where it stops: recurrent-state hybrids such as Gated-DeltaNet cannot evict mid-span and fail loud with `RecurrentEvictUnsupportedError`, never silent corruption.

Under that is the compute HAL (`internal/compute`). It lifts seven CPU-monoculture assumptions into the type system, so adding a GPU, XPU, or NPU is a new `Backend` registration rather than an edit to the forward loop. The CPU reference backend is byte-identical by design (`cpuBackend.Class()` returns `Reference` in `internal/compute/cpuref.go`); every device backend (CUDA, Vulkan, Metal) is `Approx` class, held only to argmax-exact plus a per-backend empirical cosine threshold. A new device needs its own correctness study, not just a recompile.

The bottom rung is borrowed, and this is the firm ceiling. The hardware scheduling loop — device-firmware kernel queuing, occupancy, VRAM paging, graph replay — belongs to CUDA, Vulkan, and Metal. fak exposes the device through the `Backend` interface and gates the correctness class of the results. It does not own or prove the launch queue or device-memory allocation. So the honest reach below the tool call is narrow and clear: fak owns the decode loop, the forward pass, and the KV cache as in-process kernel objects; it ships the HAL contract; it sits on the hardware loop.

## Beyond RSI: loops that pick the work and improve the improver

RSI is not the top of the ladder either. Above it sit loops that improve the kernel indirectly, and the honest tags matter more here because several are conceptual.

The first is meta-RSI: a loop that would tune the improvement *policy* itself, not just one tunable. In fak this is mostly conceptual. The mechanical piece is real — `shipgate.Gate` in `internal/shipgate/shipgate.go` counts consecutive non-keeps and returns `ESCALATE` after K — but on escalation the loop exits to a human, and nothing yet feeds that judgment back to retune fak's own keep-gate. Treat meta-RSI as conceptual with a shipped escalation breaker.

The intake loop picks the work. `tools/idea_scout.py` scans arXiv and GitHub for ideas adjacent to agent-kernel work, dedups three ways, and files cap-bounded triage issues that feed the dispatch backlog. It runs dry-run by default, with a transparent integer relevance score and a gitignored seen-cache. This is the loop that decides what the fleet works on next.

The multi-surface scorecard family applies RSI's discipline to surfaces that are not code. `tools/scorecard_control_pane.py` folds a family of deterministic per-surface scorecards (docs, code, appeal, seo, industry, product, persona, agent-readiness, and more) into one debt integer, with `--check` as the CI ratchet against a pinned baseline. Every score is re-derived from disk and the Go toolchain on each run, so a number cannot be edited into looking better. It is the same no-narration rule, pointed at repo health.

Two loops at the top are conceptual and must be labeled as such. The ecosystem and conformance loop names a frozen, additive-only ABI (`internal/abi/testdata/abi_v0.1.golden`, machine-checked by `TestABIGoldenFreeze`) and a `fak-certified` mark documented in `GOVERNANCE.md` and `TRADEMARK.md`, but the conformance suite a third party would run is declared, not shipped, and the second-implementation trigger has not occurred. The market loop is instrumentation only: `tools/industry_scorecard.py --stale` surfaces SOTA bars due a re-check against a dated, sourced taxonomy, but the action of updating fak to match the field is human-directed, never autonomous. fak surfaces the gap and escalates; it does not auto-reposition.

## The orthogonal loops: the same rule at every ring

The five nested loops are one axis: scale, or how much of the stack lives in one address space. There is a second axis the nesting hides. Some concerns are not a rung — they are threads that recur at *every* rung. Reading the loops this way turns one ladder into a grid.

```
                 TWO AXES, NOT ONE LADDER

   VERTICAL = SCALE (how much of the stack is one address space)
   ORTHOGONAL = INVARIANTS that recur at EVERY scale (->)

                 trust   cost   memory  observ.  human
   SCALE         /witness /econ  /durab  /feedbk  /govern
   -----------   ------   ----   ------  -------  -----
   ecosystem  =  [CONCEPTUAL: frozen ABI + fak-certified mark]
   meta-RSI   =  [CONCEPTUAL: enforce-tune; ESCALATE breaker is shipped]
   RSI        =  keep-bit  ->  ->  ->  ->  ->   (shipgate.Evaluate)
   fleet      =  per-SHA   ->  ->  ->  ->  ->   (dos commit-audit)
   session    =  sealed    ->  ->  ->  ->  ->   (internal/recall)
   turn       =  Clear+    ->  ->  ->  ->  ->   (ctxplan / ctxmmu)
              =  rescreen
   tool-call  =  provable  ->  ->  ->  ->  ->   (adjudicator)
              =  refusal
   ...........................................................
   = = = = = = = = = below the tool call = = = = = = = = = = =
   decode     =  OWNED    Session.Generate / Step   (kv.go)
   forward    =  OWNED    Prefill / token           (kv.go, forward.go)
   KV cache   =  OWNED    Evict + Kraw re-RoPE       (kvcache.go)
   compute HAL=  SHIPPED  Backend; CPU=Reference,    (internal/compute)
              =           CUDA/Vulkan/Metal=Approx
   hardware   =  BORROWED device firmware schedules; fak only
              =           registers the device + gates correctness

   VERTICAL  -> how DEEP fak owns the stack (one address space)
   ORTHOGONAL-> the SAME rule, EVERY ring (trust is one of 5 threads)
   distinctive = the CROSSING POINT: most scales, same invariant,
                 one kernel. (0/29 primitives novel; assembly is it.)
```

There are five threads. Trust and witness is the one the rest of this doc traces: provable refusal at the syscall, quarantine `Clear()`-plus-rescreen at the turn, sealed pages at the session, per-SHA `dos commit-audit` at the fleet, the non-forgeable keep-bit (`shipgate.Evaluate`) at RSI. It takes two forms — the witness discipline at the inner, fleet, and RSI rings (a decision no participant can move by narrating a number) and the structural containment gate at the turn and session rings (a sealed page opens only on `Clear()` and a fresh re-screen, never on a say-so) — but both are structural, not a promise.

Cost and economy thread the same ladder: O(1) bounded context reconstruction at the turn (`internal/ctxplan`), shared-prefix reuse across a session, and per-aspect or ensemble model routing at the call (`internal/modelroute`). Today the routing is deterministic over the request's shape, and cost is a post-hoc lens (`EstimateSavings`); cost-guided live dispatch is named as future wiring, not shipped.

Memory and durability are the time axis: results are classified at write time, and in enforce mode the gate refuses to promote ephemeral observations into the durable image (audit-only by default — `Admit` in `internal/ctxmmu/mmu.go`, `Page.Durability` in `internal/recall/recall.go`). The same `Evict` primitive that contains poison is the one a TTL-driven forgetting policy would ride.

Observability and feedback thread through the typed per-turn `Turn` record carried at every scale (`internal/trajectory/trajectory.go`), feeding the scorer seam and the measured witnesses RSI keeps on. Human governance recurs as the operator's hand on the loop: policy authored by a person, not the model; dry-run-until-`--live` at the fleet; the `ESCALATE` verdict that hands control back at RSI.

The reframe is two sentences. The vertical axis is *how much of the stack is one address space*: fak owns from the KV and decode loop up through the fleet and RSI loops in a single in-process kernel, borrowing only the hardware scheduler below and leaving the ecosystem loop above as aspiration. The orthogonal axis is *the same rule, every ring*: the observe, decide, act, and verify shape repeats across the scales, and trust is only one of the threads doing it.

This is also the cleanest statement of what makes fak distinctive, and it stays inside the prior-art honesty the repo leads with (0 of 29 primitives novel). Plenty of systems own a deep vertical slice — a serving engine owns decode and the KV cache. Plenty enforce one cross-cutting policy — a guardrail enforces trust. fak is the one substrate present at the most scales while carrying the same trust-and-reuse invariant through all of them. The contribution is the crossing point, not either axis alone.

## The external map: loop engineering, and the one claim fak can own

Outside this repo the same idea now has a name: *loop engineering*. Its core primitive is the **Ralph loop** — Geoffrey Huntley's `while :; do cat PROMPT.md | agent; done`: run a model over and over in fresh context against a plan file until the work is done. The pattern has gone mainstream — OpenAI Codex's `/goal`, Vercel's `ralph-loop-agent`, goose, and Google's ADK each ship a version of it — so any reader evaluating fak now arrives with this frame. It is worth saying exactly where fak sits inside it.

The frame has one load-bearing weakness, and it is the same one this whole doc is about. A Ralph loop has to decide when to stop, and in the basic form the *model* decides: it reads its own output and reports "done." That is the self-assessment trap at the scale of a whole loop — the thing being graded is the grader. fak's thesis (model proposes, kernel disposes) is the answer to precisely that weakness. The one claim fak can own here is narrow and real: **a Ralph loop whose exit-gate is a real adjudication, not a self-report.**

Here is the canon mapped onto the rings this doc already built:

| SOTA primitive | What it is | fak ring / primitive |
|---|---|---|
| Ralph loop (`while :; cat PROMPT.md \| agent`) | iterate to verified done in fresh context | the **Turn** ring, driven by the durable loop ledger `fak loop run -- CMD` (hash-chained fire/admit/start, an admission governor gating the always-on loop). A dedicated `fak loop drive` front-end is tracked in #1175. |
| external verification exit-gate | "done" judged by an oracle, not the model | the **RSI** ring's non-forgeable keep-bit (`shipgate.Evaluate`) and the **fleet** ring's per-SHA `dos commit-audit` — both shipped. Lifting that same gate to a per-turn dos exit-gate is tracked in #1174. |
| state on disk, not context | the plan-file is memory | the **Session** ring: a finished session is a durable core-dump over a content-addressed swap (`internal/recall`). A first-class `GOAL.md` plan-file is tracked in #1176. |
| meta-agent search / DGM / SICA | agents searching agent design space | an evidence-gated variant archive — tracked in #1177; today only the demo tunable is wired in `internal/rsiloop`. |
| meta-RSI (tune the improver) | retune the keep-policy itself | the ladder's labeled-conceptual apex: the `ESCALATE` breaker (`shipgate.Gate`) is shipped; feeding that judgment back to retune the keep-gate is not. |
| cross-model review | a peer model refutes before ship | a scout review rung — tracked in #1185. |
| repo-contained safety | no irreversible out-of-repo effects | `fak guard` containment — default-deny adjudication on every tool call plus the network-egress floor (`fak egress`) — is shipped; the loop-facing containment contract is tracked in #1187. |
| spec-anchor not metric (Kitchen Loop) | converge to a spec, avoid goodhart | the dos witness criterion: a change is kept on a witness derived against the spec, never a metric the agent can move (#1174/#1177). |

Read the third column honestly. The witnessed exit-gate is shipped today — the RSI keep-bit (`shipgate.Evaluate`) and the fleet's per-SHA `dos commit-audit` are the real, non-forgeable adjudications the rest of this doc traces, and the durable loop ledger (`fak loop run`) and `fak guard` containment are shipped too. The rest of the column — a dedicated `fak loop drive` front-end (#1175), a per-turn dos exit-gate (#1174), a first-class `GOAL.md` plan-file (#1176), an evidence-gated variant archive (#1177), a cross-model scout review rung (#1185), and the loop-facing guard-containment contract (#1187) — is tracked work under the verified-loop epic (#1173), not yet shipped. The mapping is the bridge; the issues are the build-out.

### Why "verified" is the whole point

Two failure modes haunt the Ralph loop, and dos refuses both by construction rather than by promise.

The first is the **self-assessment trap**: a model that grades its own work will, often enough, call a half-finished or wrong result "done." dos answers this with a decision no participant can move by narrating a number — a `Deny` cites one reason from a closed twelve-word vocabulary, a kept change sets the keep-bit only on measured gain AND a green suite AND clean truth, and an issue closes only on a per-SHA commit-audit reachable from origin/main. The agent cannot talk its way past any of these.

The second is **goodharting**: optimize against a metric and the loop learns to game the metric instead of doing the work. This is the Kitchen Loop's warning, and the fix is the one fak already uses — anchor the exit-gate to a *spec witness*, not a movable score. The RSI keep-bit is gated on a witness the loop derives from a run it performs itself (a real build, a real `git status`, a real probe), not a number the proposer hands in. A change that games the KPI but fails the suite or dirties the tree is reverted regardless of how good the metric looks (`TestKeepBitNeedsAllThree`).

That is the whole bet, stated against the external names: the Ralph loop is the right primitive, and the missing piece — the part every hand-rolled version re-implements badly — is an exit-gate the model cannot forge. fak is that exit-gate.

**Sources.** The Ralph loop (Geoffrey Huntley); OpenAI Codex `/goal`; Vercel's [`ralph-loop-agent`](https://github.com/vercel-labs/ralph-loop-agent); the [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) and [SICA](https://github.com/MaximeRobeyns/self_improving_coding_agent) / [ADAS](https://github.com/ShengranHu/ADAS) for meta-agent search; the [Kitchen Loop](https://arxiv.org/pdf/2603.25697) for the spec-anchor / anti-goodhart criterion.

## Why an engineer should care

You can build all of this yourself. People do. But every one of those loops re-implements the same dangerous, expensive scaffolding, and the failure modes are quiet: an action that should have been refused, a context full of poison, a "kept" change that was never better, a closed issue that was never fixed.

fak's bet is that this scaffolding is a kernel, not a library you copy into each project. You inherit the spine — the gate, the quarantine, the durable cache, the witness, the keep-bit — and you write the only part that is actually yours: what the loop is *for*.

One last honest note, the one the repo leads with. A prior-art audit found 0 of 29 primitives here novel (CLAIMS.md). The contribution is not any single mechanism. It is the assembly: a fused, fail-closed, witness-gated kernel with the tool call promoted to an in-process syscall. The parts are old. Wiring them into one boundary that is safe and fast for the same reason is the thing.

## Read next

- [Policy in the kernel](policy-in-the-kernel.md) — why a default-deny check on the call path beats an external recognizer that fails open.
- [Addressable KV cache](addressable-kv-cache.md) — how mid-run causal span eviction stays bit-exact.
- [The O(1) context window](o1-context-window-economics.md) — when reconstructing a bounded context each turn beats leaning on the prefix cache.
- [The cross-platform spine](cross-platform-spine.md) — the third axis: the *same* kernel and the *same* invariants across the whole deployment substrate, from an IoT node to a hyperscaler, the way this doc's scale axis runs from one tool call to the fleet.
- [The issue-dispatch loop](../dispatch-loop.md) — the witness-gated fleet loop in full.
- [The RSI loop](../rsi-loop.md) — the self-improvement loop and its non-forgeable keep-bit.
