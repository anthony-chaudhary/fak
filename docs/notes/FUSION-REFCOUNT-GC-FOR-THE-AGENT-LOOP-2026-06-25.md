# Fusion, ref-counting, and GC for the agent loop

*2026-06-25 design note. Lens: treat the agentic loop the way a language runtime treats a
heap — roots, ref-counts, reachability, collection — and ask which of those the kernel
already has and which it is missing.*

## The one-sentence claim

> Every turn and every tool result is a **heap object**. A pinned goal, an active
> slash-command, an open task is a **GC root**. Compaction is **garbage collection**.
> Today the kernel runs GC with no roots and no ref-count *from intent*, so it reclaims
> context by a recency heuristic instead of by reachability — which can free a live object
> (drop context the agent still needs) and retain a dead one (carry stale tool output
> forever).

The surprising part, after mapping the tree: **the collector already exists.** What is
missing is the single edge from an *intentional root* to the *heap objects it keeps alive*.
That edge is what "pinnable goal" really means, and wiring it makes the existing compaction
sound instead of heuristic.

## What the kernel already has (the machine)

The runtime vocabulary the goal asked for is mostly shipped — at the byte and KV-tensor
depths:

| GC / fusion concept | Where it already lives | What it does |
|---|---|---|
| **ref-count by identity** | `abi.CASPinner.Pin/Unpin` (`internal/abi/registry.go:935`) | refcount by digest; content-dedup means N holders share one digest; survives until the **last** unpin |
| **bounded heap + sweep** | `blob.Store` (`pins map[string]int`, `internal/blob/store.go:147`), `blobfs.Store`, `ctxmmu.MMU` held-ledger | pinned digests excluded from LRU; unpinned oldest-first dropped to a byte cap |
| **commit / collect hook** | `abi.ProvisionalSink.Promote/Rollback(txn,epoch)` (`internal/abi/types.go:146`) | the retract-or-keep contract a speculation epoch drives |
| **TTL + revive** | `cachemeta.Lifecycle` (`internal/cachemeta/lifecycle.go:111`) | `Filling→Resident→Expiring→Expired`; `Touch` **revives** Expiring→Resident (access proves still hot) |
| **interning / dedup** | `vdso` (tool,args,world-ver), `vcachestar`, content-address `blob.Put` | identical work → one stored result; "a hit equals a fresh call" |
| **reachability feedback** | `ctxplan.Outcome{Hits,Faults,Wasted}` (`internal/ctxplan/learn.go:25`) | witnessed: `Hits`=referenced=live, `Wasted`=resident-untouched=garbage, `Faults`=freed-too-early |
| **the heap itself** | `ctxplan.Span` over a lossless `Store`; `Plan.Selected`/`Elided` | resident vs cold-but-recoverable; eliding a span is a page-out, faulting it back is a page-in |
| **pin = exempt from collection** | `ctxplan.Forecast.Pins` + `Plan.Selection.Pinned` (`internal/ctxplan/forecast.go:34`, `plan.go:126`) | spans that MUST stay resident regardless of score; charged against budget first |

The `ctxplan.Pins` doc comment is the tell. It already says, verbatim, that a pin is *"the
spans a turn cannot proceed without (the system prompt, **the active goal**, the last user
turn)."* The code names the active goal as the canonical pin — and then never wires one in.

## The gap, located exactly

`SessionPlanner.pins()` (`internal/agent/ctxplan_session.go:140-151`) is the **complete
current root-set**, and it is three hardcoded structural pins:

```go
var pins []string
if sp.systemPin   != "" { pins = append(pins, sp.systemPin) }    // the system prompt
if sp.firstUserPin != "" { pins = append(pins, sp.firstUserPin) } // the original ask
if sp.lastUserPin  != "" { pins = append(pins, sp.lastUserPin) }  // the latest ask
return pins
```

There is no `goalPin`. There is no edge from *what the agent is trying to do* to *what the
collector must keep*. The forecast that consumes these pins is authored every turn at
`ctxplan_session.go:158` (and the heuristic twin at `ctxplan_seam.go:208`). So the collector
runs each turn over a root-set that knows the conversation's *shape* (first/last/system) but
not its *purpose*.

That is why compaction today is **mark-sweep with no mark phase**: it elides by score
(relevance + recency + durability + utility), never by *reachability from a goal*. A long
session pursuing one goal will happily elide the span that goal depends on if that span
stopped scoring well three turns ago — and will keep twelve tool results that nothing alive
still references, because nothing told it they were dead.

## What "ref-count of a turn / tool-use" actually is

The goal's sharpest question. A heap object in the window is **retained** by exactly three
kinds of referent — and its ref-count is how many are currently live:

1. **A root needs it.** An open goal/pin/task that depends on this object. (The missing
   edge.) Example: goal "serve GLM-5.2 on DGX3" retains every tool result about the node's
   state until the goal is discharged.
2. **A later turn cites it.** Turn 7's reasoning quotes turn 3's grep output ⇒ turn 3 is
   retained by turn 7. This is `ctxplan.Outcome.Hits` run *backwards*: a hit is a live
   inbound reference.
3. **A pending consumer will read it.** A dispatched tool call whose result will feed the
   next reasoning step — a producer→consumer edge. This is the **fusable** case (below).

Ref-count zero ⇒ **provably** free, not "probably stale." That word *provably* is the whole
prize: it is the difference between the current heuristic elision and a sound collection.

### The safety argument is an asymmetry

A false *free* silently loses information the agent needs — catastrophic, and invisible
until the agent flails. A false *retain* costs a few tokens — cheap, bounded, visible on the
S/N metric. So the collector must be **conservative**: over-retain when unsure, exactly like
a conservative GC treats anything pointer-shaped as a root, and exactly like the kernel's own
`NeverAdmits`/`FallbackDeny` floor over-refuses when unsure. The bias toward retention is
what makes it safe to run automatically. This is the same decision the kernel already makes
at the wire and KV depths — now made at the **object/reachability depth**, the one layer that
can actually see what keeps what alive.

## Fusion proper — the part the goal led with

Garbage collection reclaims dead objects. *Fusion* is the stronger move: never materialize an
intermediate at all when it has exactly one consumer. Two loop-level fusions fall out:

**1. Turn fusion (producer→consumer elision).** When turn N is a pure tool dispatch whose
result is read by exactly turn N+1's reasoning and never referenced again (ref-count 1,
producer→consumer), the result need not persist as its own heap object. It collapses into
"turn N+1 concluded X (via a tool call)." This is precisely a compiler fusing a producer into
its consumer to elide the intermediate buffer. The kernel already ships the heuristic version
of this — `gateway.maybeCompactInboundTools` / `maybeCompactAnthropicRaw`
(`internal/gateway/messages.go:357,383`) drop old turns beyond the cached prefix. Fusion is
the **sound** version: a ref-count of 1 proves the intermediate is unobserved elsewhere, so
collapsing it is information-preserving, not a budget guess. `ProvisionalSink.Promote` is the
natural commit point — the fused result is promoted into its consumer turn, the standalone
object is collected.

**2. Tenuring — "a command that becomes known to be xyz."** The goal's second example is
**generational promotion**. A slash-command invoked once is a *young* object. A command run
every loop (`/conflation-score`, `/quality-score`) is *tenured* — proven long-lived, so you
stop re-deriving its context each turn and keep a compact promoted form: its rollup. The
"effective storage for rollup/aggregate/GC" the goal asks for **is** the tenured generation.
`cachemeta.Lifecycle` already has the mechanism (`Touch` revives a hot entry; cold entries
demote and expire); it just is not pointed at *commands* yet. The file-memory store
(`MEMORY.md` + topic files) is, in this frame, the **tenured heap** — the generation that
survives session-GC — and its hand-maintained "every topic file referenced exactly once"
bijection (`check_memory.py`) is literally a **ref-count == 1 invariant** enforced by witness.

## The pinnable goal, concretely

`/goal` today is a harness Stop hook: session-scoped, in-memory, model-judged (Haiku re-reads
the transcript). It is not persisted and it is invisible to the kernel — the collector cannot
see it. To make a goal *pinnable* in the fak sense is to **promote it from ephemeral
transcript state into a kernel root**:

- **Store it where the scheduler can read it.** Add a goal/root field to `session.State`
  (`internal/session/session.go:164`) — the PCB that `Table.Snapshot()` already exposes and
  that, per the agent-OS roadmap, *nothing yet reads*. A pinned goal is the most natural first
  consumer of that Snapshot: a root with a priority and a budget.
- **Wire its edge into the collector.** One branch in `SessionPlanner.pins()`:
  `if sp.goalPin != "" { pins = append(pins, sp.goalPin) }` — and a producer that maps the
  active goal's content into the span IDs it transitively retains (start trivial: pin the
  goal span itself; later, pin spans whose descriptors overlap the goal's content-tokens, the
  same matching `ctxplan.Index.Probe` already does for intents).
- **Witness "done" instead of judging it.** A fak goal's discharge is checked by
  `dos hook stop` (git/effect evidence), not only by the model re-reading itself — the
  consistency-vs-grounding upgrade the DOS goal-gate already documents.

The payoff: when the goal is pinned as a root, the spans it depends on are reachable *by
construction* and the collector cannot elide them no matter how their recency score decays —
and when the goal is discharged, its whole retained sub-graph drops to ref-count zero in one
unpin and is collected. Pinning a goal and freeing its working set become the *same*
mechanism, viewed from the two ends of the object's life.

## Recommendation — what to build first

Build the **smallest edge that proves the frame**: wire the active goal into the per-turn
pin set, end to end, and measure that it changes what survives compaction.

1. Carry the active goal text into `SessionPlanner` (a `goalPin` span, durability `session`).
2. Add the one branch in `pins()` so the goal is charged into the resident view first.
3. Reuse the existing `ctxplan.Outcome` loop as the ref-count witness: a goal-pinned span that
   shows up in `Wasted` for K turns is a false-retain (tune down); a span that `Faults` while
   a goal that needs it is live is a false-free (the bug this whole note is about — it should
   become impossible once the edge exists).
4. Surface it on the S/N metric already shipped (`signalnoise.go`, `signal = Hits ∪ pins`) so
   the goal-pin's effect on signal density is visible, not asserted.

Everything heavier — generational tenuring of commands, sound turn-fusion via
`ProvisionalSink`, a scheduler that honors goal-root priority over `Table.Snapshot()` — is a
real follow-on, but each is an *extension* of a primitive that already ships. The frame's
whole value is that it does not require a new collector. It requires connecting the collector
the kernel already has to the one thing it cannot currently see: **what the agent is for.**

---

*Grounding: `internal/abi/registry.go:935` (CASPinner), `internal/abi/types.go:146`
(ProvisionalSink), `internal/ctxplan/forecast.go:34` + `learn.go:25` + `plan.go:126`,
`internal/agent/ctxplan_session.go:140` (the pin set), `internal/cachemeta/lifecycle.go:111`
(TTL+revive), `internal/session/session.go:164` + `table.go:134` (the PCB). See also the
memory note `fak-refcount-gc-design-frame` and `fak-agent-os-positioning` (the keystone
"nothing reads Snapshot" gap).*
