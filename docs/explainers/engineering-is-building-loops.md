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

## Why an engineer should care

You can build all of this yourself. People do. But every one of those loops re-implements the same dangerous, expensive scaffolding, and the failure modes are quiet: an action that should have been refused, a context full of poison, a "kept" change that was never better, a closed issue that was never fixed.

fak's bet is that this scaffolding is a kernel, not a library you copy into each project. You inherit the spine — the gate, the quarantine, the durable cache, the witness, the keep-bit — and you write the only part that is actually yours: what the loop is *for*.

One last honest note, the one the repo leads with. A prior-art audit found 0 of 29 primitives here novel (CLAIMS.md). The contribution is not any single mechanism. It is the assembly: a fused, fail-closed, witness-gated kernel with the tool call promoted to an in-process syscall. The parts are old. Wiring them into one boundary that is safe and fast for the same reason is the thing.

## Read next

- [Policy in the kernel](policy-in-the-kernel.md) — why a default-deny check on the call path beats an external recognizer that fails open.
- [Addressable KV cache](addressable-kv-cache.md) — how mid-run causal span eviction stays bit-exact.
- [The O(1) context window](o1-context-window-economics.md) — when reconstructing a bounded context each turn beats leaning on the prefix cache.
- [The issue-dispatch loop](../dispatch-loop.md) — the witness-gated fleet loop in full.
- [The RSI loop](../rsi-loop.md) — the self-improvement loop and its non-forgeable keep-bit.
