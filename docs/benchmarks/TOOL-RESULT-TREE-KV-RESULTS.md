# TOOL-RESULT-TREE-KV — why a regular KV cache can't *faithfully* store a tool result, and what fak does instead

> Companion to `KV-QUARANTINE-BRIDGE-RESULTS.md` (the byte-gate drives the KV-gate),
> `RADIXATTENTION-RESULTS.md` / `RADIXATTENTION-EXPLAINER.md` (automatic prefix
> reuse + policy eviction), and the in-kernel model lane. This doc isolates one
> claim that kept getting stated loosely — *"regular KV caching doesn't properly
> store tool results / breaks the tree"* — and pins down the **precise, defensible**
> version of it, because the loose version is refutable and the precise version is
> not. Every number below is a `go test` line.

## The loose claim, and why it's wrong as stated

The intuition is right but the slogan over-reaches. "Regular KV caching breaks the
tree / can't store tool results" is **false** taken literally, and a vLLM / SGLang /
llama.cpp engineer can refute it in one breath:

- **vLLM (PagedAttention)** ref-counts physical KV blocks and does **copy-on-write
  on write-divergence**: forking a sequence (parallel sampling, beam search) shares
  the parent's blocks, and the first write to a shared block allocates+copies a fresh
  one. That *is* per-branch isolation — branch A's later writes cannot touch branch B.
- **SGLang (RadixAttention)** literally *is* a tree. A tool result that is the unique
  **leaf** of one branch lives in that branch's own trie node; dropping that node
  touches neither the shared parent nor any sibling. (fak's own `radixkv.EvictNode`
  does exactly this — node-granular pruning — and nothing more.)
- **llama.cpp** ships `llama_kv_cache_seq_cp` (branch a sequence), `seq_rm` (remove an
  arbitrary, including *middle*, token range), and `seq_add` (the K-shift graph op that
  **re-rotates** surviving keys to their new positions). So "a standard cache can only
  re-prefill or share-and-corrupt" is simply not true — llama.cpp removes-from-the-
  middle-and-shifts as a normal operation.

So the tree, branch isolation, and even middle removal are all things shipped engines
do. If the claim were "regular caches can't do agent trees," it would be **refuted**.
It isn't the claim worth making.

## The precise claim that survives every comparator

There are exactly **two** capabilities here that fak has and no shipped KV cache
offers, plus one governance property. They are narrow and they are real.

### Core 1 — bit-exact removal of a tool result from the *middle* of a kept sequence

A tool result is untrusted. The moment a poison/quarantine verdict lands, you want
that result **gone** — and you want what's left to be **byte-identical to a run that
never saw it**, including every token that came *after* it. That last clause is the
hard part, and it is where every other engine stops short:

| engine | branch a sequence | remove a *middle* span | result after middle removal |
|---|---|---|---|
| vLLM PagedAttention | yes (block CoW) | **no API** — append-only, post-RoPE blocks | must recompute the tail |
| SGLang RadixAttention | yes (trie) | only the unique **leaf**; no middle-span surgery | n/a (recompute) |
| llama.cpp | yes (`seq_cp`) | yes (`seq_rm` + `seq_add`/K-shift) | **not bit-exact** — K-shift δ-rotates *already-rotated* keys (a composed rotation), drifts ~1e-6, enough to flip a greedy token; llama.cpp does not claim bit-exactness |
| HF transformers `DynamicCache` | (logical) | no reposition primitive | re-prefill |
| **fak** | yes (eager `Clone`) | **yes** (`KVCache.Evict`) | **bit-identical to never-saw** |

Why fak alone is bit-exact: it keeps the **pre-RoPE** key alongside the post-RoPE one.
`model.KVCache` stores `Kraw` — "the SAME entries pre-RoPE, so a span can be
repositioned" (`internal/model/kv.go:21`), stashed in `blockStep` *before* RoPE is
applied (`kv.go:366`). When `Evict(from, n)` removes a middle span, each surviving
later token must move to a lower absolute position; fak re-derives its key from `Kraw`
in a **single** rotation at the new index (`kv.go:74-87`) rather than composing a
delta onto the already-rotated key. The design note states the trap exactly: composing
two rotations "is mathematically equal but drifts ~1e-6 — enough to flip a greedy
token" (`kv.go:26-29`). That single-rotation-from-pre-RoPE store is the machinery
llama.cpp's K-shift lacks, and it is why fak's middle removal is bit-exact where K-shift
is only approximate.

**Proven**, two rungs, kept distinct:

- *Numerics, vs HuggingFace* — `TestKVQuarantineEqualsNeverSaw` (`internal/model/evict_test.go`):
  a poison span is prefilled with a query *after* it, evicted from the middle, and the
  greedy continuation matches the HF "never-saw" run **token-for-token**; the reposition
  invariant `K[i] == RoPE(Kraw[i], i)` holds at `max|Δ|=0` (byte-exact on amd64). It also
  proves the *boundary*: a span evicted *after* downstream tokens already attended to it
  is **not** un-seen — which is why quarantine must be write-time. This rung is
  **fixture-gated** (it `t.Skip`s where the 538 MB HF export is absent).
- *Wiring, unconditional* — `TestLedgerRenumberAfterMiddleEvict` and
  `TestWriteTimeEvictEqualsNeverSaw` (`internal/kvmmu/kvmmu_test.go`): on a synthetic
  model (no weights, runs anywhere), a *middle* segment is evicted and the surviving
  distribution is `max|Δ|=0` against a reference that never saw it, with the real
  `ctxmmu` poison verdict driving the eviction. The structural property is
  weights-independent, so the synthetic witness is faithful for the *mechanism*.

### Core 2 — provably byte-disjoint sibling branches

In a fan-out (one orchestrator context → N sub-agents), each branch runs its own tool
call and accumulates its own untrusted results. fak forks a branch by **eager `Clone`**
of the parent cache (`SessionFromPrefix` → `KVCache.Clone`, `kv.go:95,169`), which
`make()`s and copies every layer's `K`/`Kraw`/`V`. So branch A's backing arrays are
**physically disjoint** from branch B's: `A.Cache.Evict(...)` mutates only A's slices
and *cannot reach B's bytes, even in principle*.

This is witnessed by the new `TestTreeSiblingQuarantineIsolation`
(`internal/model/tree_sibling_test.go`): a shared parent prefix is cloned into two
sibling branches; branch A runs a **poisoned** tool result and quarantines it; branch B
runs a **benign** one and keeps it; then both decode the same query.

| measurement | max\|Δ\| | meaning |
|---|---|---|
| branch A: quarantined result vs **never-saw-poison** | **0.000e+00** | A's own tool result cleanly removed |
| **sibling B vs B-built-in-isolation** | **0.000e+00** | A's quarantine left sibling B byte-untouched — the tree held |
| control: A-poison-kept vs A-clean | 2.713e-01 | poison genuinely perturbs → the scrub is non-vacuous |
| control: B-with-result vs B-without | 3.861e-01 | B's result genuinely matters → the isolation is non-trivial |
| control: A vs B | >0 | the branches are genuinely distinct → isolation isn't vacuously true |

**Read this honestly.** The `sibling B = 0` line is true **by construction**: an eager
deep copy guarantees disjoint arrays, so "disjoint arrays stay disjoint" is a
*regression guard* against an accidental shallow clone/aliasing, not a deep theorem.
fak does **not solve** the shared-cell policy-eviction problem — it **dissolves** it by
never sharing the cells in the first place, paying memory to do so. That is exactly the
trade: where vLLM/SGLang share blocks zero-copy and would have to CoW-copy a branch
*before* they could surgically evict from it (and even then could not do the bit-exact
middle removal of Core 1, having no pre-RoPE key), fak copies up front and gets the
guarantee for free thereafter. **The eager copy is more memory and is faithfully
labeled as such** (`RADIXATTENTION-RESULTS.md` already calls it "a copy, not SGLang's
zero-copy page share").

### The governance axis (kept distinct from the mechanism)

fak exposes eviction as a **policy** operation, not just a memory-pressure one:
`radixkv.EvictNode` prunes a named subtree regardless of LRU/recency
(`radixkv.go:270-286`), and `kvmmu.AdmitResult` evicts a tool-result span write-time on
a real detector verdict (`internal/kvmmu/kvmmu.go`). vLLM/SGLang eviction is
LRU / refcount / preemption only — there is no "evict *this* span because policy says
so" verb. **But do not conflate the two fak mechanisms**: `radixkv.EvictNode` is
node-granular tree pruning that re-RoPEs **nothing**; the bit-exact *middle-span*
surgery of Core 1 and the disjoint-sibling guarantee of Core 2 come from the
**per-session `Clone` + `KVCache.Evict`** path, not from the radix tree.

## The one-sentence version a hostile reviewer can't call false

> fak is the only one of these KV caches that can remove a tool-result span from the
> *middle* of a kept sequence and leave the cache **bit-identical to never having seen
> it** — because it alone retains the pre-RoPE keys needed to re-rotate the survivors in
> a single exact rotation (proven token-for-token vs HuggingFace) — and it isolates
> concurrent agent branches by eager copy so one branch's quarantine is provably
> invisible to its siblings (`max|Δ|=0`), at the labeled cost of the copy that
> zero-copy page-sharing engines avoid.

## Scope — what this is NOT (labeled, not hidden)

- **Structural capability, not throughput.** Nothing here claims fak is faster or
  higher-throughput than vLLM/SGLang/llama.cpp. fak pays an eager copy where they
  zero-copy share; the win is a *guarantee*, on a different axis. (This is the same
  axis-honesty `MODEL-BASELINE-RESULTS.md` and the head-to-head doc enforce; the
  prefix-reuse *speed* numbers elsewhere are fak-vs-fak ablations and are **not** invoked
  here.)
- **Comparator limits are about today's shipped engines, not a theorem.** A paged
  engine *could in principle* retain a pre-RoPE key and copy-on-write a branch before a
  span eviction. Nothing proves no engine *ever* will. The claim is "no shipped KV cache
  does this," backed by their documented mechanisms (post-RoPE-only paged pools; K-shift
  that admits drift), not an impossibility result.
- **Not yet wired into the live `fak agent` loop.** Today `fak agent` quarantines a
  poisoned tool result at the **byte/context** layer — `ctxmmu` holds the bytes out of
  the prompt (`internal/agent/loop.go`: the `Quarantines` counter, `r.Meta["admit"] ==
  "quarantined"`, "held out of context") — and drives the model over the OpenAI-compatible
  HTTP seam. The KV-level eviction (`kvmmu`) and the tree-CoW fork (`SessionFromPrefix`)
  are **proven kernel primitives** that the live loop does not yet call; wiring them in
  needs the in-kernel tokenizer + chat-template rung the in-kernel lane already scopes.
- **The two rungs prove different things.** `TestTreeSiblingQuarantineIsolation` (new,
  synthetic) proves Clone-disjointness + a *tail*-evict (its branch-A evict removes the
  span before the query is appended, so it re-RoPEs no survivor — it is **not** a
  middle-re-RoPE witness). The bit-exact *middle*-span re-RoPE is proven separately, over
  real HF weights, by `TestKVQuarantineEqualsNeverSaw`. The writeup attributes each
  sub-claim to its own proof and does not let either borrow the other's strength.

## Adversarial verification (panel, default-REFUTE)

House discipline (`STATUS.md` §5; the same posture `KV-QUARANTINE-BRIDGE-RESULTS.md`
used): six independent skeptics, each on a distinct lens, each instructed to **default
to refuting** the claim. Result: **0 refuted, 0 unqualified-hold, 5 NEEDS_QUALIFICATION**
(the 6th, the over-claim-framing lens, hung in a tool loop; the SGLang skeptic
independently confirmed no throughput over-claim is present, covering that ground). The
panel did not break the claim — it **sharpened** it, and this doc is the sharpened form.

| Lens | Verdict | The qualification it forced (now incorporated) |
|---|---|---|
| **vLLM PagedAttention CoW** | NEEDS_QUALIFICATION | vLLM *does* isolate forked branches (block CoW); drop "shared-cell engines can't isolate." The real gap is **bit-exact middle removal** (vLLM post-RoPE-only, no pre-RoPE key) + the policy-evict-a-span verb. |
| **SGLang RadixAttention** | NEEDS_QUALIFICATION | A unique-leaf tool result drops without touching siblings (it's a tree). The hard cases SGLang can't do: middle-span bit-exact removal, and removing a span *shared* with a sibling without CoW first. Don't credit `radixkv.EvictNode` (pruning, no re-RoPE) with the span surgery. |
| **llama.cpp / HF** | NEEDS_QUALIFICATION | llama.cpp *has* `seq_rm`+K-shift (a real third option beyond re-prefill/share-corrupt) — but K-shift composes rotations and drifts ~1e-6, **not** bit-exact. Keep the "no pre-RoPE K → not bit-exact" clause; it's what makes the claim true. |
| **Test vacuity** | NEEDS_QUALIFICATION | The `sibling B = 0` assertion is true by construction (deep Clone) — a regression guard, not a deep result. Bill it honestly; make the genuinely-non-trivial middle-span bit-identity (the HF rung) the load-bearing witness. |
| **Claim-to-evidence** | NEEDS_QUALIFICATION | The new test is a *tail*-evict + Clone-disjointness witness (its re-RoPE branch never fires); middle re-RoPE belongs to `TestKVQuarantineEqualsNeverSaw` (HF, fixture-gated). Comparator limits are implementation statements, not theorems. |

Two skeptics independently **re-ran both witnesses green** (`TestKVQuarantineEqualsNeverSaw`
token-for-token vs HF; the four `max|Δ|` lines above) and independently observed — then
confirmed *resolved* — a transient `internal/model` merge-conflict window from a peer's
in-flight `top10-arch-integration` merge (a live-fleet hazard, not a code defect; the
package compiles and the suite is green now).

## Bottom line

Regular KV caches *do* store tool results and *do* serve agent trees — that part of the
slogan is wrong. What they cannot do is **un-see** a tool result faithfully: remove it
from the middle of a kept sequence and stay bit-identical to never having seen it, and
isolate one agent branch's quarantine from its siblings provably. fak can, because it
owns the cache as a kernel data structure that keeps the pre-RoPE key (single-rotation
reposition) and forks branches by eager copy (disjoint bytes). That is the precise,
test-backed, unrefutable form of "ours natively supports this" — a structural guarantee,
honestly priced in memory, not a throughput claim.
