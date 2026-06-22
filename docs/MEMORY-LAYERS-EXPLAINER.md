---
title: "The four layers of agent memory, explained by fak"
description: "How routing, addressing, fusion, and semantics are four different KV-cache problems, and why fak's paradigm change lives only at the semantics layer."
---

# The four layers of agent memory — routing, addressing, fusion, semantics

*Why "the KV cache is shared now" is four different problems wearing one name — and which one fak actually changes.*

## TL;DR

When people say agent memory (the KV cache — the key/value tensors a transformer
caches per token so it needn't recompute them — and the context window) is becoming a
**shared, networked tier**, they are compressing four genuinely different problems
into one sentence. The four are **routing** (where a cell physically lives and how a
request finds it), **addressing** (the stable name two readers use for the same
cell), **fusion** (whether the bytes share one arena for zero-copy access), and
**semantics** (whether a cell can be coherently mutated, isolated, attributed, and
capability-gated across a trust boundary, *and proven*). They sit at different
layers and answer different questions about the same KV-cache cell.

The serving world has been pouring effort into the first three. **fak's paradigm
change is at the fourth**, and the fourth alone. Routing, addressing, and fusion all
take the cell *as it is* — a frozen, append-only, single-writer scratchpad — and move
it, name it, or co-locate it. fak changes **what the cell is**: it makes the cell
mutable-in-the-middle, isolatable, attributable, and gated. The other three operate
on whatever object you hand them; fak hands them a better object.

The one-line test for the lane: **if a claim is true of a frozen single-writer cache
that merely got moved, named, or co-located, it's a routing/addressing/fusion claim —
not fak's differentiator.** fak's differentiator is always a sentence that is only
true once the cell can be *coherently mutated, isolated, attributed, or gated across
a trust boundary.*

> This explainer is the expanded, standalone version of
> `DISAGGREGATED-AGENT-MEMORY.md` (private companion — not published) §2.5. That doc is
> the strategy note; this one is the teaching artifact you can hand someone cold.
> Honesty discipline is the same as [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md): where it names
> something unbuilt it says `[GAP]`.

---

## The four questions, side by side

Each layer asks a *different question about the same cell.* Hold the cell fixed and
walk down the column:

| Layer | The question | Operates on | Who owns it today | fak's paradigm change? |
|---|---|---|---|---|
| **Routing / placement** | *Where does this cell live, and how does a request reach it?* | the cell as an opaque blob | the fabric + serving scheduler — Mooncake, NVIDIA Dynamo, LMCache, vLLM prefix-routing | **No** — networking layer, *below* the change |
| **Addressing / naming** | *By what stable name do two readers refer to the same cell?* | the cell's identity | content-addressing / a content-addressable store, CAS (an established technique) | **No** — a precondition fak *uses*, not invents |
| **Fusion / co-residence** | *Do the bytes share one arena, for zero-copy access?* | the cell's storage layout | whoever owns the KV arena — vLLM on the GPU (via CUDA, NVIDIA's GPU-compute platform); or fak v0.2's in-kernel model owning *its own* arena | **No** — a deployment property, orthogonal to meaning |
| **Semantics / mutation & trust** | *Can a writer coherently edit, isolate, attribute, and gate this cell across a trust boundary — and prove it?* | the cell's **meaning and provenance** | **largely unowned — this is fak's layer** | **Yes** |

The crucial column is the last one. Three "No"s and one "Yes," and the Yes is at the
layer nobody else is standing on.

---

## A picture: the stack, and where each player sits

The four layers stack. Lower layers move bytes; upper layers govern meaning. A
request enters at the top of an agent's intent and resolves *down* through naming and
placement to physical bytes — but **trust flows the other way**: a value's meaning and
provenance are decided at the semantics layer regardless of where the bytes ended up.

```
                          THE QUESTION EACH LAYER ANSWERS
   ┌───────────────────────────────────────────────────────────────────────┐
   │  SEMANTICS    "may this cell be edited / isolated / trusted / acted on, │  ← fak's
   │  & TRUST       and can I PROVE it?"                                     │    paradigm
   │               coherent middle-eviction · quarantine · provenance ·     │    change
   │               capability floor · arbitration                           │    (the object
   │               ── owner: largely UNOWNED. fak is here. ──               │     itself)
   ├───────────────────────────────────────────────────────────────────────┤
   │  FUSION       "do the bytes live in one arena for zero-copy?"          │  ┐
   │  & CO-RESIDENCE  vLLM owns KV in CUDA · fak v0.2 in-kernel model owns   │  │
   │               its OWN arena · external co-residence = [GAP] copy-CAS   │  │ operate
   ├───────────────────────────────────────────────────────────────────────┤  │ on the
   │  ADDRESSING   "what stable NAME do two readers share for one cell?"    │  │ cell
   │  & NAMING     content-addressing / CAS · digest, not heap pointer      │  │ AS-IS
   ├───────────────────────────────────────────────────────────────────────┤  │ (move /
   │  ROUTING      "WHERE does the cell live, how does a request find it?"  │  │ name /
   │  & PLACEMENT  Mooncake · NVIDIA Dynamo · LMCache · vLLM prefix-routing │  │ co-locate)
   │               ── crowded, well-funded, nearly solved ──               │  ┘
   └───────────────────────────────────────────────────────────────────────┘
        resolve DOWN  ↓   (intent → name → place → bytes)
        trust flows  ↑    (meaning/provenance decided at the top, wherever bytes land)
```

Read the right margin: the bottom three layers take the cell *as-is* and relocate,
rename, or co-locate it. Only the top layer rewrites the contract of the cell.

---

## Layer by layer

### 1. Routing / placement — "after the paradigm, not the paradigm"

*"The KV cache needs to exist **somewhere**, and a request needs to **find** it"* is a
real, hard problem. It is also a **networking/placement** problem, and — this is the
load-bearing observation — it is **invariant to what the cell means.**

Mooncake's KVCache-centric scheduler, NVIDIA Dynamo's prefix-aware router, LMCache's
tiered DRAM+SSD (main memory plus solid-state disk) store, vLLM's prefix-cache-aware
routing: these are excellent at
*getting the right bytes to the right GPU.* They say **nothing** about whether a
writer may evict a poisoned span from the middle of a shared sequence, whether reader
B is allowed to page that span in, or whether the value B reads was written by a
trusted author. You can bolt fak's semantics onto any of these routers — the router
places the cell; fak governs what may be done to it.

The layering is one-directional and worth memorizing: **routing assumes the cell, fak
defines it.** A pitch that positions fak *against* a KV router has made the category
error. The correct relationship is "fak rides above your router."

> This is the "existence is a networking issue, after the paradigm we change" point.
> A cache existing somewhere on the fabric, and a request finding it, is the problem
> the serving engines already own. It is *downstream* of the question fak changes.

### 2. Addressing / naming — a precondition fak uses, not invents

For two readers to reuse one cell, the cell needs a **stable name** that doesn't
depend on one process's heap layout. The established answer is **content-addressing**:
name a value by the digest of its bytes (a CAS — content-addressable store). A result
written by agent A is then reachable by agent B *by digest*, with no shared pointer.

fak **uses** this — the CAS is the substrate under the vDSO (virtual dynamic shared
object — a fast, safe read path borrowed from the OS-kernel term) tier-2 cache and the
context-MMU (memory management unit — the hardware/OS unit that maps and protects
memory; fak's is the software analogue for agent context) page-out
(`internal/ctxmmu/mmu.go`) — but content-addressing is not fak's
invention or its differentiator. It is table stakes: the naming precondition that any
shared tier needs before the *interesting* (semantics) questions even arise. Naming a
cell tells you nothing about whether you may *mutate* or *trust* it.

### 3. Fusion / co-residence — a deployment property, orthogonal to meaning

*"Do the cell's bytes live in one memory arena, so they can be shared without a
copy?"* This is **fusion** (or co-residence). It is a property of *where the bytes
physically sit relative to the compute*, not of what they mean.

**There are two different zero-copies, and conflating them is the trap.** Most "the
cache is zero-copy" claims you hear are the *first* kind:

- **Intra-engine zero-copy (ubiquitous, and fak does it too).** Within one engine, a
  request that shares a prefix *points at* already-resident KV pages instead of
  re-copying them — one allocator owns the pages and hands out references. vLLM's
  PagedAttention, SGLang's paged pool, and fak's own in-kernel arena all do this.
  This is real, it is everywhere, and it is genuinely zero-copy. fak scores a `●`
  here: **v0.2 fuses a real forward pass into the kernel**, so the model owns *its
  own* KV arena as a kernel Go structure (`internal/model.KVCache`).
- **Cross-engine zero-copy (the integration seam, not yet built).** Sharing *one* KV
  arena across a **trust/process boundary** — fak reading and mutating the bytes that
  a *separate* vLLM/CUDA process owns, in place, with no copy. The shipped path here
  is copy-CAS; the genuinely-shared-arena version is the unbuilt rung. The `Ref`/
  `Resolver`/`RegionBackend` seam is **frozen precisely so this is a backend swap**,
  not a rewrite — see [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) for the exact stub language.

So the `●` and the open rung are the *same layer* at two different scopes: zero-copy
inside an arena fak owns (done) vs zero-copy into an arena another engine owns (the
seam we integrate with later — **not** ruled out, just not yet wired).

Either way, fusion is a **deployment** choice: it changes copy cost and latency, not
whether a cell can be coherently edited or trusted. A fused arena with no semantics is
still a frozen single-writer scratchpad — just a faster one.

**Why the cross-engine rung is genuinely harder than "wire up a pointer" — three
gates, increasingly fundamental:**

1. *Allocator ownership.* Intra-engine zero-copy works because one allocator owns the
   pages. Cross-engine, vLLM owns its KV in its own CUDA/Python space; sharing it
   needs a handle *into that allocator* (CUDA IPC — inter-process communication —
   handles, a shared VMM — virtual memory management — pool, a pinned
   host segment both map). That's the ~120h engineering boundary — real, but a matter
   of work, not a law.
2. *Byte-exact layout agreement.* Zero-copy means both sides read the *same bytes* as
   the same tensor: paged-block layout, head-dim order, dtype, RoPE (rotary position
   embedding — how token positions are encoded into the keys) convention must
   all match, or you're re-packing (a copy). This is intra-model even before it's
   cross-engine — which is *also* why the cross-**architecture** dream (one pool across
   Claude *and* Gemini) stays a non-starter at the tensor layer no matter how clever
   the allocator trick.
3. *The deep one — the semantics fak wants needs the bytes structured a certain way at
   **write** time, which only the owner controls.* fak's provable forgetting re-rotates
   survivors using the pre-RoPE key (`Kraw`) it keeps. A foreign engine keeps
   *post-RoPE* K in its paged pool — it never stored `Kraw`. So fak reading vLLM's
   pages zero-copy would get *visibility* but **not** the substrate for bit-exact
   middle-eviction, because the information the exact re-rotation needs was thrown away
   before fak ever saw the bytes. **Zero-copy read access to a foreign arena buys the
   cheap layer (placement/visibility) and specifically *not* the layer fak exists for.**

That third gate is the real reason the matrix marks the cross-engine cell as an open
rung rather than a quick win: it's not that zero-copy is impossible — it's that
zero-copy *into someone else's arena* doesn't, by itself, deliver the provable
semantics, because the proof depends on a write-time structuring decision the foreign
engine didn't make. The path forward is the integration the seam is frozen for (fak
either owns the write side, or the foreign engine exposes enough to reconstruct
`Kraw`), and it is firmly on the roadmap — these adjacent layers are ones we **get to
and integrate with**, not ones we rule out.

**One subtle, one-directional dependency — fusion doesn't *give* you semantics, but
owning the bytes is a precondition for *proving* them.** The layers are orthogonal in
the direction that matters for positioning (fusion is not a semantics claim), but there
is a real asymmetry the other way: you cannot do **bit-exact** middle-eviction on a KV
arena you don't own. fak's provable forgetting (§4) re-rotates survivors *inside the
arena it owns* — an engine that holds its KV behind a paged CUDA pool it doesn't expose
can recompute or approximate, but it cannot hand you a byte-identity proof, because the
bytes aren't its to re-derive deterministically. So fusion-of-the-model-into-the-kernel
isn't the *differentiator* (it's a deployment property), but it is the **substrate that
makes the semantics layer's strongest claim — provable, not asserted — physically
possible.** Read the dependency the right way: semantics is what's scarce and valuable;
owning the arena is the price of admission to proving it, not the prize.

### 4. Semantics / mutation & trust — the layer fak changes

This is the only layer where fak changes *what the cell is.* The shared-memory
questions that have no good answer once memory crosses a trust boundary:

- **Coherent middle-mutation.** Remove a tool result from the *middle* of a kept
  sequence and have the survivors stay byte-correct. fak keeps the pre-RoPE key
  (`Kraw`) so `KVCache.Evict` re-rotates survivors in one pass — **byte-identical to
  never-having-seen it** (`internal/kvmmu`, proven token-for-token vs HuggingFace).
  Page-shared engines recompute the tail; llama.cpp's K-shift only *approximates*.
- **Isolation / quarantine.** A secret- or injection-shaped write is held out of
  context and the model is made *incapable of attending to it* (`internal/ctxmmu` +
  `kvmmu.AdmitResult`), and the seal **survives the process boundary**.
- **Provenance / verification.** Every value is source-stamped (`internal/ifc`), a
  kernel-authored classifier takes authorship of trust away from the model
  (`internal/provenance`), and a witness gate fails closed on an unwitnessed claim
  (`internal/witness`, the in-process `dos_verify`).
- **Capability / access-control.** A deployable, version-tagged JSON policy floor
  (`--policy FILE`, `internal/policy`) — not a compiled-in constant.
- **Arbitration.** `dos_arbitrate` keeps two writers off the same region.

None of these is something a router, a name, or a co-resident arena gives you for
free. They are the contract of a cell that *means* something — and they are what fak
has been building for the single-agent kernel all along.

---

## The analogy: Docker ↔ Kubernetes (similar, adjacent, different layer)

The cleanest way to feel "related but you must not conflate them" is the container
stack — because it is the same shape of confusion, and most engineers have already
made (and recovered from) the mistake once.

- **Docker** answers *"what is the unit, and what's inside it"* — the image, its
  layers, its content-addressed digest, its isolation boundary. It defines the
  **object's identity and contents.**
- **Kubernetes** answers *"where do the units run, how are they found, how do they
  scale and fail over"* — scheduling, service discovery, placement. It **routes and
  orchestrates** the objects Docker defined.

People conflate them constantly ("isn't K8s just Docker at scale?"), and they *are*
genuinely adjacent — but they are **different layers**, and K8s *assumes* a
well-defined image; it does not redefine what an image is. Map it across:

| Containers | Disaggregated KV memory | Who's here |
|---|---|---|
| **Docker** — defines the image: identity (digest), contents, isolation boundary | **the semantics layer** — defines the cell: coherent mutation, isolation, provenance, capability | **fak** (the "Docker": defines the *object*) |
| **Kubernetes** — schedules / discovers / scales the images | **the KV router / fabric** — places & finds the cells | Mooncake, Dynamo, LMCache, vLLM-routing (the "K8s": *routes* the object) |

The punchline has the same shape as *"K8s is not a better Docker, it's a different
layer that runs on top"*:

> **A KV router is not a better memory MMU. It's a different layer that runs on top of
> one. fak is the Docker-layer of agent memory — it defines the cell that the routing
> layer then schedules.**

Confusing the two is exactly the error of thinking you can replace Docker with
Kubernetes. You can't: K8s needs an image to schedule, and a KV router needs a cell to
place.

### One caution the analogy invites

Don't over-read it into *"fak is the packaging and the router is the real system."*
The container analogy is about **which layer owns which question**, not about
importance. If anything the agent-memory case **inverts** the usual hype gradient: the
routing layer is the crowded, well-funded, nearly-solved part, while the semantics
layer — the "Docker" here — is the unsolved, unowned one. The analogy maps *layers*,
not *value*.

---

## Where each system actually sits — the layer matrix

Place the named systems against the four layers and the picture the lane has been
arguing for becomes visible at a glance: the lower three layers are *crowded*, and the
top layer is *empty except for fak*. `●` = a primary, owned competence; `◐` = present
but not the system's focus; `○` = not addressed; `[GAP]` = fak's own unbuilt rung.

```
                          ROUTING      ADDRESSING    FUSION        SEMANTICS
                          (where /     (stable       (zero-copy    (coherent mutation,
                          find it)     name)         arena)        isolation, trust — PROVEN)
   ─────────────────────────────────────────────────────────────────────────────────────
   Mooncake / Kimi          ●            ●             ◐             ○
   NVIDIA Dynamo            ●            ◐             ◐             ○
   LMCache                  ●            ●             ◐             ○
   vLLM (paged + routing)   ●            ◐             ●             ○
   SGLang (RadixAttention)  ◐            ●             ●             ○
   llama.cpp                ○            ◐             ●             ◐  (K-shift: approx, not exact)
   ─────────────────────────────────────────────────────────────────────────────────────
   fak                      ◐ rides      ●  CAS        ● own arena   ●  THE owned layer
                            above a        (uses,        (in-kernel    coherent middle-evict
                            router         not its       model) +      (bit-exact) · quarantine
                            (§1)           moat)         [GAP] ext.    · provenance · capability
                                                         co-residence  · arbitration · PROVABLE
```

Three things to read off it:

1. **The semantics column is empty above the line.** Every serving engine scores `○`
   there — not because they're weak, but because it isn't the layer they're built at.
   llama.cpp's `◐` is the honest exception: its K-shift *attempts* in-place edits but
   only *approximates* (~1e-6 drift), which is exactly the gap between "asserted" and
   "proven" that the semantics layer is about.
2. **The lower-left is saturated.** Routing and addressing are `●` across nearly every
   row. That is the crowded, well-funded competition — and it is *not* the column fak
   needs to out-engineer to be useful (it scores `◐`/uses-not-owns there). But "doesn't
   need to win it" is **not** "rules it out" — see the next section: several of these
   lower rungs are actually *cheaper in fak's context* than in a serving engine's, and
   they're on the roadmap to integrate, not to avoid.
3. **fak's only solid-bottom score is a borrowed one.** Its fusion `●` (own in-kernel
   arena) isn't the differentiator — per the dependency above, it's the *substrate*
   that lets the one `●` that matters, the semantics column, be **provable** rather
   than merely claimed.

The matrix is the four-word test rendered as a scoreboard: a win in the first three
columns is a win at a layer many systems already own; the column that is fak's to win
is the one no row above the line has even entered.

---

## Why the lower layers are *easier* in fak's context, not harder

It would be a mistake to read the `◐`/`[GAP]` marks as "fak is weak on the lower
layers." The opposite is closer to true: **several of those rungs are cheaper for fak
than for a serving engine, because fak operates one layer up — at the agent syscall
boundary — where the structure a serving engine has to *guess* is still present and
typed.**

Here is the crux. A general serving engine sees an **anonymous stream of tokens.** By
the time a prompt reaches it, the structure has been erased at the API boundary, so the
engine must *reverse-engineer* everything the lower layers need:

- *Whose cache is this?* → it has no idea. It sees a token sequence, not a principal.
- *Which requests share a prefix?* → it must **guess**, via radix-tree matching over
  raw token IDs.
- *What is safe to evict?* → it must **guess**, via LRU (least-recently-used eviction)
  under memory pressure — a bet
  about future reuse it has no real basis for.
- *Where are the semantic seams* (this span is a tool result, that span is reasoning)?
  → invisible; it's all one flat stream.

fak's context is the inverse. It sits at the **tool-call / agent-loop boundary**, so the
structure was never thrown away — it is **given**:

- **It's a specific user/agent, not a stream.** The `Ref` is agent-scoped and tainted
  at mint time (`internal/ifc`, the gateway seam). fak doesn't infer ownership — it is
  handed *this principal's* memory, with identity attached. Addressing and isolation
  stop being inference problems and become bookkeeping.
- **It's a state machine, not a token soup.** An agent loop is a *known sequence of
  typed transitions* — tool call → result admitted/transformed/quarantined → next turn.
  fak knows the turn boundaries, knows which span is a tool result versus reasoning,
  knows what write invalidates which prior read (the `FLEET-SWEEP` scoped-invalidation
  eraser is exactly this). The serving engine sees none of that and has to guess where
  the seams are.

So when the matrix says fak "rides above a router" or marks content-addressing as
"uses, not its moat," that's not modesty about a hard problem — it's that **in fak's
context these are easy or already done.** Content-addressing isn't bolted on; the cache
is digest-named by construction. Cross-node distribution isn't a retrofit; the
`Ref`/`Resolver`/`RegionBackend` indirection was *frozen up front* precisely so a
fabric backend is a swap, where a CUDA-owning engine would have to tear open its hot
path to add one. Even the hard cross-engine rung (§3, gate 3) is *fak-favorable*: the
easy path to provable eviction is fak owning the write side, which it already does
because it already keeps `Kraw`.

The deeper reason this is true — and it is the reason the whole layering matters: **the
semantics layer is only ownable by something that still has the identity and the
state-machine structure.** That structure is exactly what the agent syscall boundary
*preserves* and the token-serving boundary *destroys*. So fak is not at the top layer
by luck or by avoiding the bottom ones; it is at the top layer because it stands at the
boundary where the information the top layer needs hasn't been erased yet — and that
same vantage point is what makes the lower rungs cheap rather than hard. **A serving
engine guesses the cache's owner and shape from tokens; fak is handed both.**

---

## The payoff: a whole new line of optimizations the vantage opens up

This is the part that makes the layering more than defensive positioning. Once you own
the cell *and* you know whose it is *and* you know the state machine it belongs to, a
class of optimizations becomes possible that simply cannot exist on an anonymous token
stream. They are not "fak is faster at the same thing" — they are *things the other
layer cannot do at all.* A few terms first, so this reads for anyone:

- **KV cache** — the key/value tensors a transformer stores per token so it doesn't
  recompute them; think of it as the model's short-term working memory for a session.
- **Prompt injection** — a malicious instruction smuggled inside data the model reads
  (a web page, a tool result) that hijacks the model into doing something it shouldn't.
- **Latency** — delay; **microsecond (µs)** = one millionth of a second,
  **millisecond (ms)** = one thousandth. A model *turn* (one round of the model
  thinking) is typically hundreds of ms; a memory operation can be µs — a thousand-fold
  difference, which is the whole point below.
- **State machine** — a system that is always in one of a set of known states and moves
  between them on defined events. An agent loop is one: *waiting → tool call → result
  admitted/rejected → next turn.* Knowing the machine means knowing exactly where you
  are and what may legally happen next.

### 1. Filtering *before* the write, not scrubbing *after* — the µs security filter

The dominant way to handle a bad tool result today is **after-the-fact**: let it into
the model's context, then try to detect and clean up the damage (re-prompt, re-scan,
hope). That is a cleanup crew. fak's vantage lets it be a **filter at the doorway**:
because the tool result crosses the syscall boundary *before* it is ever written into
the KV cache, the policy check runs on the way in. fak's in-process adjudication —
the decision of allow / deny / repair / quarantine — runs in **~1,300 nanoseconds** for
the cheapest detection-scan layer (about 1.3 µs; the composed normgate+ctxmmu chain is
29–87 µs, witnessed on M3 Pro 2026-06-20 — see `MAC-M3PRO-KERNEL-BENCH-2026-06-20.md`),
versus the *hundreds of milliseconds* an extra model turn costs to notice and undo a bad
write after the fact.

The difference is categorical, not incremental: a known-bad pattern (a secret-shaped
blob, an injection signature, a tool the policy forbids) is **stopped at the door for
the price of a memory compare**, so the poisoned bytes never enter the cache at all. An
after-the-fact system has already paid to ingest the poison and now pays again to chase
it. This is the firewall-vs-cleanup-crew distinction, and it is only available because
fak sees the write *before* it lands — which it does because it sits at the boundary,
holds the policy as data (`internal/policy`), and owns the arena the write would go
into. `[SHIPPED]` — this is the gate the README headline measures.

### 2. Exact rewind and cheap branching — because the turns are known and the bytes are owned

A serving engine cannot cleanly "go back to how things were three turns ago," because it
doesn't know where turn boundaries are (it sees flat tokens) and it doesn't keep the
information needed to undo a rotation exactly. fak keeps both:

- It knows the **turn boundaries** (the state machine gives them).
- It keeps the **pre-rotation key** (`Kraw` — the key vector *before* position
  information is baked in; "rotation"/**RoPE**, rotary position embedding, is how a
  model encodes *where* a token sits in the sequence). Keeping `Kraw` is what lets fak
  re-derive the cache after a removal **bit-for-bit** — byte-identical to a cache that
  never saw the removed span — instead of approximately.

So fak can **rewind** to the exact state at turn *N* and **branch** — fork the session
into two futures that share everything up to the fork and diverge after — with a
`Clone()` (copy the cache cheaply) plus an `Evict()` (drop a span exactly). For an agent
doing tree-of-thought search or exploring two tool strategies, that means: *try branch A,
and if it dead-ends, snap back to the fork and try branch B from precisely the same
state*, with no recompute of the shared prefix and no drift. `[SHIPPED]` — `Clone()` /
`Evict()` are the proven primitives (`TOOL-RESULT-TREE-KV-RESULTS.md`); the dynamic
per-turn rewind/branch *policy* that drives them is the natural next rung.

### 3. Speculative and transactional turns — run it provisionally, keep it only if it's good

Because fak knows it is *between* defined states, it can run a turn **provisionally** —
a **transaction** (a unit of work that either fully commits or fully undoes, never half).
Let the model take a speculative action, and:

- if the outcome is good → **commit** it (make it permanent);
- if it's bad → **roll it back** so it's as if it never happened — including evicting any
  KV the speculative turn produced.

This is ordinary in databases ("begin transaction … commit/rollback") and almost unheard
of for an agent's working memory, because you can only offer it if you can *exactly*
retract a write — which loops back to owning the arena and keeping `Kraw`. fak's
envelope already carries the provisional lifecycle for this: a `SpeculationContext`, a
transaction id (`TxnID`), and `Promote`/`Rollback` verbs, with a driver that retracts
squashed effects (`internal/spec`, `ARCHITECTURE.md` §2.6/§3.4). `[SEAM SHIPPED]` — the
lifecycle and the speculative-execution driver exist; wiring richer keep/revert policies
on top is the open work.

### 4. Structure-aware eviction — drop what a span *is*, not what an LRU *guesses*

A serving engine evicts by **LRU** (least-recently-used — throw out whatever hasn't been
touched in a while), a blind guess about future reuse. fak knows what each span *is* — a
tool result, a reasoning step, a system prompt — so it can evict by *meaning*: drop the
stale tool result whose data has since been superseded, keep the system prompt, and do it
**exactly** (the survivors stay byte-correct). Eviction stops being a cache-pressure
heuristic and becomes a *policy decision* — "this span is no longer valid," not "this
span looks cold." `[SHIPPED]` primitive (span-exact eviction); meaning-driven eviction
*policy* is the additive rung.

### 5. Per-principal everything — quota, redaction, audit — because identity is attached

The bottom-layer engines have no principal (no notion of *who* a cache belongs to), so
they cannot natively answer "how much memory is *this user* using," "redact *this
tenant's* data and prove it's gone," or "show the audit trail for *this agent's* writes."
fak's `Ref` is **agent-scoped and provenance-stamped** at mint time (the gateway tags
every value with who produced it and how trusted it is), so per-user quota, per-tenant
provable redaction (the *provable forgetting* of §4), and per-agent audit are natural,
not bolted-on. `[SHIPPED]` stamping + isolation; the management surface over them is the
build-out.

> **The through-line.** Every one of these is the *same trick*: an optimization that is
> impossible-or-guessed on an anonymous, unowned token stream becomes *exact and cheap*
> the moment you have (a) the identity (whose cell), (b) the state machine (which turn,
> what's legal next), and (c) the owned arena with `Kraw` (the power to undo a write
> bit-for-bit). The four-layer picture isn't only about *not over-claiming* the routing
> layer — it's about *what the semantics layer lets you build* that the layers below
> structurally cannot. The filter-at-the-door, the exact rewind, the transactional turn:
> these are the new line, and they all trace back to standing where the structure hasn't
> been erased.

---

## Why this matters — the failure mode it prevents

Every time a routing win gets re-told as a fak win, the lane drifts toward "fak is a
faster/cheaper KV cache" — a claim that is (a) false (fak is parity-to-behind on raw
throughput; `CLAIMS.md` is explicit) and (b) a crowded loser even if it were true. The
four-layer split is the guardrail. Run the one-line test on every sentence:

```
   Is this sentence true of a FROZEN, SINGLE-WRITER cache
   that merely got MOVED / NAMED / CO-LOCATED?
        │                                  │
       YES                                 NO
        │                                  │
   routing/addressing/fusion        only true once the cell can be
   claim — fine to state, but       coherently MUTATED / ISOLATED /
   NOT the fak differentiator       ATTRIBUTED / GATED across a trust
                                    boundary — THIS is the fak claim
```

Keep the "throughput is solved, semantics isn't" framing honest by never letting a
placement win cross the line into a semantics claim.

---

## See also

- `DISAGGREGATED-AGENT-MEMORY.md` (private companion — not published) — the strategy note;
  §2.5 is the compact form of this explainer, §2 maps the six memory semantics (S1–S6)
  to shipped primitives, §3 the cross-agent / cross-tenant / cross-node axes.
- [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — the honest claims ledger; the exact `[SHIPPED]`/`[STUB]`
  language behind every primitive cited here.
- `HYBRID-AI-MEMORY.md` (private companion — not published) — applies the four-word test to the
  **device↔cloud** seam: hybrid AI's "may this cell cross to the cloud" is a *semantics*
  question (the locality/residency axis), not a routing/addressing/fusion one.
- `RADIXATTENTION-EXPLAINER.md` (private companion — not published) — a worked case at the
  *addressing* layer (prefix reuse by name) where fak adds a *semantics* operation
  (provable eviction) the routing-only engines structurally cannot.
- [`TOOL-RESULT-TREE-KV-RESULTS.md`](benchmarks/TOOL-RESULT-TREE-KV-RESULTS.md) — the
  coherent-middle-mutation result, token-for-token vs HuggingFace.
