# vToolcall — Tool Results as a Materialized View

**Date:** 2026-06-25
**Status:** design note over a substrate that is *mostly already shipped*. The serve-from-cache-as-if-the-tool-ran mechanism exists today as the **vDSO FastPath** (`internal/vdso`, consulted at `internal/kernel/kernel.go:347` inside `Kernel.Submit`) and is reachable now through `fak_syscall` (`internal/gateway/mcp.go:235,431`) and the in-kernel agent loop. The closed-loop invalidation core — hierarchical epoch vector (`scope.go`), refutation/revoke (`revoke.go`), per-principal isolation (`principal.go`) — is implemented and tested. What is **not** built: witness-keying of the tier-2 key, a cross-process shared cache (only *revocation* crosses processes), a context-level `ProvisionalSink` for speculative un-admit, and any result-injection disposition in the PreToolUse hook protocol. Those are named honestly below and gate the speculative rungs.

> **Audit provenance (2026-06-25).** Every fak symbol named below was grounded against HEAD before writing. Five corrections from that pass are folded in and load-bearing: (1) **topology** — `internal/gateway/gateway.go:1-27`'s own package doc states `/v1/chat/completions` (and the Anthropic `/v1/messages` twin) "does NOT execute the client's tools (the client does)"; the FastPath that synthesizes results is reached only via `k.Syscall`, never the proxy path. So "hand the harness a cached result as if the API returned it" is real for `fak_syscall`/in-kernel and **structurally impossible** on the provider-forward proxy. (2) The contextq verdict set (`MaterializationKind`: HIT/FAULT/RECOMPUTE/REFUSE/ABSTAIN, `contextq.go:164-168`, **five**) and the cachemeta verdict set (`LookupKind`: hit/miss/revalidate/transform/quarantine/fault, `cachemeta.go:245-250`, **six**) are **not** byte-identical and not aliases — they overlap on HIT/FAULT and diverge elsewhere; any unified verdict needs an explicit, lossy mapping, not a rename. (3) The witness is **not** bound into the tier-2 key — `revoke.go:44-49` says so verbatim; two agents reading the same file under different git SHAs still share by `(tool,args,worldVer)`. (4) `vdso.Default` is **process-global**, not cross-process: the FILL half of a shared cache is unbuilt; only `Revoke` is on the wire (`POST /v1/fak/revoke`). (5) The context-MMU (`internal/ctxmmu`) does **not** implement `abi.ProvisionalSink`; the only registrant is `internal/spec.Sink`, which retracts a **KV-cache token span** for speculative *decode*, not an admitted tool-result page from context. Trust-law claims (`ProviderCacheVerdict` → `LookupTransform` → `CanServe()==false`, `cachemeta.go:303,320`; `materialization.go:189-191`) reproduced exactly.

---

## 0. The one-sentence idea

> Everything presented to the harness is a **materialized view** over a source of truth the kernel controls — and a tool *result* is just one more view. When the kernel already knows a tool's result (it ran a `Read` last turn, or a peer did), it can serve that result **as if the tool had just run**, skipping the round-trip — *provided* the view is still fresh by a predicate cheaper than the call itself, and *provided* the kernel is the one dispatching the tool.

vToolcall is `contextq`'s materialization machine pointed at the **tool-result wire** instead of the inbound-context wire. The scarce thing it owns — and the reason it can be sound where a provider KV cache cannot — is **closed-loop invalidation**: the kernel adjudicates every mutating call and observes every completion (`vdso.Emit`), so it *sees the write that invalidates a read* and can strand the stale entry before it is ever served. The spine of this note is that invalidation, because the failure it prevents is not a wasted token (vCache's worst case) but a **corrupted answer** (the model reasoning over a file that no longer exists on disk).

---

## 1. The polarity flip from vCache (why this note exists at all)

vCache (`docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md`) caches a **provider KV prefix**. Its Law A2 — *correctness never depends on warmth* — is true there because the model **re-derives** truth from a synthesized prefix: a stale prefix costs tokens, and any divergence self-corrects as the model reads the resent context. A wrong belief is *expensive*, never *wrong*.

A tool result is the **opposite**. The model consumes it as **ground truth** and cannot re-derive it — it has no other channel to the file's bytes. So the lethal failure inverts:

| | vCache (KV prefix) | vToolcall (tool result) |
|---|---|---|
| What is cached | a warm prefix the model re-derives from | a result the model reasons **over** |
| Worst case of a false hit | pay full price, book a phantom saving | model reasons over stale bytes → **corruption** |
| Failure class | cost / latency | **correctness** |
| Invalidation loop | **open** (provider is opaque; warmth is a belief) | **closed** (kernel sees its own writes) |
| Safe posture | "correct-and-cheaper-in-expectation, never dependent on a hit" | **sound-before-serve**: a hit must *equal* a fresh call at serve time |

This flip is the whole design. vCache *may* serve a stale prefix because the cost is bounded; vToolcall **may not**, ever — but it has the closed loop that lets it guarantee it won't. **Law T (the vToolcall soundness invariant, dual of A2):** *a served view must equal a fresh execution at the moment of serving.* `vdso.go:13-17` already states it: "a cache hit equals a fresh call." Everything below is in service of that one sentence.

---

## 2. Topology — who runs the tool decides what is even possible

This is the question that makes or breaks the concept, and the answer is **it depends on who dispatches the tool.** There are three surfaces, with structurally different powers.

### 2a. REAL + SHIPPING TODAY — `fak_syscall` / in-kernel (the kernel IS the dispatcher)

When a caller routes a tool **through** fak — an MCP client calling `fak_syscall`, or fak running its own agent loop — the path is `mcp.go:235 → k.Syscall → Kernel.Submit`, and `Submit` consults `abi.FastPaths()` at `kernel.go:347` **before** adjudication. A vDSO `Lookup` hit returns `abi.Verdict{Kind: VerdictAllow, By: "vdso"}` with the result pre-stored, carrying `served_by=vdso` and a tier tag. **This is literally "serve from cache as if the tool ran," and it is wired now.** `fak_syscall` is already advertised in `tools/list` (`mcp.go:431`).

The first shippable rung therefore has **nothing to add on the serve path** — it exists. The rung is *adoption*: a Claude Code MCP client that calls its read-only tools via fak's MCP server gets dedup + cache today. Optional polish: surface `served_by`/miss-reason (`vdso.MissReasons`) in the tool result so the client can distinguish a hit from a fresh run.

### 2b. STRUCTURALLY IMPOSSIBLE — the `/v1/messages` provider-forward proxy

On `fak guard -- claude`, fak is a proxy **downstream of the harness**. The wire cycle is strict: the model emits `tool_use` (id preserved) → **Claude Code executes the tool locally** → it sends back a `tool_result` keyed by `tool_use_id` on the next request → fak parses it via `DecodeAnthropicMessagesRequest` (`tool_use_id → ToolCallID`) and admits it via `admitInboundResults` (`gateway.go:1316`). The proxy path runs `s.adjudicate → k.Decide` only — **no FastPath, no `Lookup`, no dispatch** (confirmed: `messages.go`/`messages_stream_planner.go` call `admitInboundResults`, never `abi.FastPaths()`). The model API does not run the client's local tools; there is nothing for fak to short-circuit. **"Hand the harness a synthesized `tool_result` as if the API returned it" is the wrong direction for this surface, full stop.**

What fak *can* do here is **result-RECOGNITION, not interception**: a `tool_result` whose `(tool,args)→result` the vDSO already holds is a signal the inbound block is *redundant* and can be compacted out of the forwarded prefix — the same byte-splice discipline as `promptmmu.CompactInboundTools` / `agent.CompactAnthropicHistory`. This is Law-A2-clean (cost/latency only; the harness's real result is still the one forwarded), and it is the only legitimate cache use on the proxy wire.

### 2c. NET-NEW, BLOCKED ON A PROTOCOL — the PreToolUse hook

`tools/repo_guard.py:402` (`run_hook`) is the existing PreToolUse shape: it reads the tool call on stdin and emits `hookSpecificOutput.permissionDecision`. It **does** intercept *before* the harness executes — the only place "serve before the harness runs it" is real for an *external* harness. But the protocol it speaks carries `allow`/`deny` only; it has **no result-injection disposition**. A vToolcall hook would canonicalize `(tool,args)`, consult a fak cache, and emit a result — which requires the harness to accept a hook-supplied result. **Until that disposition exists, this rung is blocked on the harness contract** (a time-sensitive claim, dated 2026-06-25 — re-verify against the current Claude Code hook protocol before relying on the ordering). And because a hook is a *separate process* with no `EvComplete` feed, its cache runs **open-loop** and must bind an external witness to stay sound — the same machinery 2a gets for free.

**The deciding asymmetry:** 2a is not just the *easy* rung, it is the *strong* one, because invalidation is closed-loop there and open-loop everywhere else. Ship 2a first.

---

## 3. The safety taxonomy — pure / read-mutable / effectful

A tool call may be served from a view **iff serving it is observationally indistinguishable from running it fresh.** The admission floor is the **conjunction** of two independent predicates, both required:

1. **SOUND** — no false hit is ever possible. Every served entry is bound, at serve time, to an invalidation key that has *strictly advanced* if reality changed in a way that could alter the result. Serving can only ever degrade to a MISS → engine; it can never invent a stale hit. (The revoke.go lemma generalizes: every gate only turns a would-be hit into a miss.)
2. **CHEAPER-THAN-THE-CALL** — the invalidation predicate's evaluation cost is *strictly less* than executing the tool. A predicate as expensive as "re-run and diff" is sound but pointless — it deletes the speed win. This is the tool-result form of vCache's `recall.go` break-even gate (`P·r < S·U`): serving from cache is worthwhile only when the **check** is amortized-cheaper than the **call**.

Soundness alone is insufficient; that is the load-bearing addition over "just cache it."

The three classes map onto the three vDSO tiers:

| Class | Definition | Served from view? | Invalidation predicate | vDSO seam |
|---|---|---|---|---|
| **pure** | result is a total function of args (no world state) | **YES**, unconditionally | none — `args-sha256` *is* the key; re-derived not trusted | tier-1 `pure` registry (`vdso.go:351`) |
| **read-mutable** | reads world state other actors can write | **YES**, behind a sound+cheap predicate | `(tool, args-hash, principal, epoch-chain)` matches current | tier-2 `cache` + `keyLocked`; epoch bumped by `Emit` on any write-shaped completion (`vdso.go:468`) |
| **effectful** | writes / mutates / sends; or a read whose freshness is not nameable | **NEVER** | n/a — no sound cheap predicate exists, so the floor refuses | `destructive(c)` veto (`vdso.go:299`) + `resourceMisnamed` gate (`scope.go`) |

The third row's hard **NEVER** does the safety work, and the vDSO encodes it two ways that cannot drift: `IsWriteShaped` (a name heuristic over `writeShapeNeedles`: write/edit/delete/patch/exec/run/send/book/update/cancel — `vdso.go:282`) **OR** `c.Meta["destructive"]`. The heuristic deliberately **over-approximates** (a read-only `rundown_report` is excluded because its name contains "run"). That is the correct error direction: **a false "effectful" costs one cache miss; a false "pure/read" costs corruption.**

The decision procedure a hook (or `Lookup`) runs, top-to-bottom, first match wins, every deny → fresh execution:

```
admit(call c):                                          # decision, miss-reason
  1. if destructive(c):            return DENY(DESTRUCTIVE)      # effectful → never served
  2. if not (readOnlyHint and idempotentHint):
                                   return DENY(MISSING_HINTS)    # cannot prove read-shaped
  3. if classify(result(c)) != Cacheable:
                                   return DENY(SECRET)           # Law D4, fail-closed (§6)
  4. if c.Tool in pure:           out,ok = pure(args)           # tier-1: RE-DERIVE, don't trust
        if ok: return SERVE(out, tier=1, taint=servedTaint(c))
  5. if c.Tool in static:         return SERVE(static, tier=3)  # args-independent canned
  6. if granularity==Resource and resourceMisnamed(c,args):
                                   return DENY(RESOURCE_MISNAMED)
     key = keyLocked(c,args)                                     # tool:hash(+principal):epoch-chain
     if e = cache[key]:
        if revokedLocked(e.witness): evict(e); return DENY(WITNESS_REVOKED)
        return SERVE(e.ref, tier=2)                              # epoch matched ⇒ sound hit
  7. return DENY(NOT_CACHED)
```

A misclassification escape hatch must be stated honestly: a custom mutating tool named `apply` or `commit_changes` (if neither hits a needle) whose author *also* set `readOnlyHint=true` would be admitted as read-mutable and could serve a stale view of a mutated world. The effectful floor is sound for tools whose names hit `writeShapeNeedles` **or** carry `Meta["destructive"]`; for everything else it is only as sound as the hints, and a mis-hinted mutator defeats it **at runtime**. A definition-time `toollint` can catch the obvious cases, but it is not a runtime guarantee.

---

## 4. Invalidation — the spine

The user flagged cache invalidation as load-bearing. It is, and the killer insight is **closed-loop**: vCache cannot see a provider evict; the kernel *can* see its own write. fak ships the closed-loop core; the work is not a new cache but making the existing eraser **explicitly layered** and resolving the one place the loop is genuinely open.

### Four layers, ordered by trust-strength of the signal each consumes

All four key into the same tier-2 entry (`entry{key, ref, witness}`):

**LAYER 1 — GENERATION EPOCH (closed-loop, kernel-mediated writes). The ground truth and the killer advantage. Already shipped.** `Emit` (`vdso.go:468`) sees every `EvComplete`; `destructive(c)` bumps `worldVer` (root) or a `nodes[tag]` epoch via the hierarchical vector in `scope.go` (`"*" → namespace → namespace:entity`). A read binds the epoch of every node on its root→leaf chain in `keyLocked`; the key changes iff any node is bumped. Because the kernel **saw** the write, a read of a file the kernel just wrote cannot hit a stale entry — the epoch it was filed under no longer matches. A write that can name its entity strands only that subtree; an un-nameable mutation falls to root (`*`) = full flush — the sound over-approximation. *This is the file-path generation counter, generalized to a hierarchical vector.*

**LAYER 2 — STAT PROXY (the half-open loop: writes fak did NOT mediate).** This is the one place the closed loop breaks — a peer session, an external editor, the compiler. The kernel never saw the write, so no epoch bumped. The fix binds a cheap world-state **witness** = `sha(path, mtime_ns, size)` captured at fill time into `entry.witness` (the field exists; the FS witness function does not yet). Two sub-mechanisms: (a) a cheap re-`stat` on the hit path before serving — far cheaper than re-reading and re-feeding — treating a witness mismatch as a forced miss; (b) for cross-process writes, a filesystem watcher (or the peer's kernel) calls `Revoke(witness)` (`revoke.go`), which evicts every entry under that witness and broadcasts a `Revocation` so other agents are causally evicted. **Honest status: layer 2a is the gate the sound-before-serve rule needs on a shared tree, and it is unbuilt — `metaFor` (`gateway.go:1517`) threads only `readOnlyHint`/`idempotentHint`, never a witness.** Stat is a *proxy*, not proof (mtime granularity + same-size edits alias), which is why layer 3 exists.

**LAYER 3 — CONTENT-HASH, VERIFY-THEN-TRUST (ground truth, on the BACKFILL path, never as a serve gate's only basis).** The CAS digest *is* the content hash. Its role is to adjudicate the stat proxy's may-aliases **asynchronously**: when layer 2 says "unchanged" but certainty is wanted (a security-sensitive tool, a same-size edit), an async re-execute hashes the result and compares to the stored digest. Match → the served hit was correct (the common case; `FaultsAvoided`). Differ → the entry is poisoned: `Revoke(witness)` it for the *next* turn and raise the integrity high-water mark. This is verify-then-trust on *re-admission*, **not** a license to retroactively fix a serve that already happened.

**LAYER 4 — TTL BACKSTOP (the unwatchable).** For results the kernel can neither adjudicate (L1), stat (L2), nor cheaply re-hash (L3) — a network feed, a clock-dependent read, a non-deterministic `Bash`. A per-tool max-age stamped at fill; past it the entry is a forced miss regardless of epoch/witness. The only **probabilistic** layer; default **short or zero** (fail-closed, matching `resourceMisnamed`'s refusal to cache a read no write could invalidate). Never cache the genuinely-unwatchable as if it were watchable.

### The un-ring problem, resolved

**You cannot un-ring the bell.** Once the model consumes a stale `tool_result` it is in the KV/history — on the Anthropic wire the harness already POSTed it back as a content block. Therefore the predicate must be **SOUND BEFORE SERVING**, and "serve fast, then backfill-verify, correct-if-wrong" must be **rejected as the default** for tool results. The user's framing of "serve then backfill the cache/request" splits into three mechanisms with very different danger:

- **Backfill the CACHE (always safe).** Serve a *real, fresh* cached hit now; async re-execute so the *next* call is fresh. Zero new soundness surface — the served bytes were a genuine tier-2 hit, not a synthesis. "Backfill" is just an eager re-`Emit`. The pin-vs-lazy choice is the governor's `λT` crossover. **This is the legitimate, shippable form of "serve fast, reconcile later."**
- **Backfill the REQUEST/transcript (an honesty obligation, not a correctness license).** If a *synthetic* (not-yet-verified) result is ever served, it is a self-report wearing a tool result's authority — exactly what the kernel exists to distrust. It must be labeled (`verified=pending`, a `{"_synthetic":true}` wrapper dual to the existing `{"_quarantined":true}` stub at `gateway.go:1341`) and reconciled via the witness/revoke axis before any downstream agent or replay trusts it. A `[SHIPPED]` claim may never rest on a `verified==pending` result.
- **Speculative SERVE (the dangerous one — and the weakest rung).** Serve a *likely* result instantly, run the real tool async, correct on divergence. The recoverable-vs-fatal line is mechanical: a *read* that depends on an un-promoted synthetic result may proceed; a *write-shaped* dependent must be **deferred** until it promotes, because the kernel can only retract what flowed through a `ProvisionalSink` and an external side effect never did.

**Why speculative-serve is the weakest rung, stated bluntly.** The mechanism that would un-admit a synthetic context page is **not wired**: `internal/ctxmmu` does not implement `abi.ProvisionalSink`; the only registrant is `internal/spec.Sink`, whose `Rollback` evicts a **KV-cache token span** for speculative *decode* (EAGLE/Medusa-style), not an admitted tool-result page. Worse, even with that wiring, "recoverable" overclaims: once the model emits its next turn conditioned on the fake result, those tokens are causally downstream — a correction turn *appends a contradiction*, it does not un-condition the model. And worst of all, the reconciliation trigger (`WitnessTracker.Observe` re-observing the resource) requires the real tool to run anyway — so the speculative serve **pays the very cost it tried to avoid** (the trigger and the cost are the same event). The economic gate (`P·r < S·U`) independently refuses a lone speculative serve: it pays only on a sibling fan-out sharing one hot prefix. **Conclusion: speculative-serve is net-negative for almost every real tool call.** Ship backfill-the-cache; treat speculative-serve as research, gated behind the unbuilt sink.

"Correct-if-wrong" precisely defined: (a) `Revoke` the poisoned entry, (b) bump `trustEpoch` + publish a `Revocation`, (c) raise the integrity high-water mark, (d) *only if* the call was idempotent-to-redo **and** the stale result has not yet entered committed context — substitute the fresh result. It **never** means editing a `tool_result` the model already read.

### The shared-tree reality (the repo's own git model) — and where it does not pay off yet

A multi-session trunk where peers edit the same files is the read-mutable hazard at fleet scale, and it needs all three orthogonal axes:

1. **CONSISTENCY** (`worldVer` / node epochs). Safe default on a shared tree is **Global** granularity: any peer write the kernel observes does `worldVer++`, full-flush — coarse but trivially sound. Per-file `Resource` precision is opt-in and requires the **fine-writes-need-fine-reads** invariant (`scope.go:38-46`).
2. **INTEGRITY** (`trustEpoch` / witness). The git commit/blob SHA is the natural witness; `Revoke(old_sha)` evicts on a peer commit.
3. **ISOLATION** (principal, `principal.go`). A per-session principal folded into the args-hash closes the cross-agent timing oracle. Empty principal = single-tenant sharing (byte-identical to v0.1); a named principal engages isolation.

**The honest verdict for *this* repo's workload.** Three facts compound against shared-tree caching today: (i) the witness is **not in the tier-2 key** (`revoke.go:44-49`), so the only sound posture is Global = full flush per peer write; (ii) the cross-agent break-even cliff is **~1% writes** (`scope.go:8-13`) and the repo's own trajectory audit shows a write-heavy mix (Edit ≈ 99, Bash ≈ 94 vs Read ≈ 50 — far above 1%); (iii) the FS namespace is **not tagged** in `nsKeywords`, so the finer-eraser escape that would rescue the economics does not exist for `Read`/`Edit`/`Glob`. **vToolcall over the shared git tree is net-negative until witness-keying and FS namespace tagging ship.** The note refuses to claim otherwise. The single-session, own-writes-only case (Layer 1) is the sound win available now.

A second cross-process honesty note: `vdso.Default` is **process-global**. Two `fak` processes have *independent* tier-2 maps — session B's process has an empty LRU, so "peer A read it, B is served from the pool" does not happen. **Only the *revoke* half crosses processes** (`POST /v1/fak/revoke`); the FILL half (a shared/persistent pool) is unbuilt. And the revoked-witness ledger **fails closed on overflow** (>8192, `DefaultRevokedWitnessLimit`): under high commit churn the integrity axis self-disables to "refuse all witness-bearing serves" — sound, but it turns that axis off as a speed win.

---

## 5. The materialized-view architecture (the unifying frame)

fak has, **four times independently, built the same machine**: take a backing source of truth, decide *materialize or reuse* a projection, gate the decision on freshness + trust, and emit a render layout.

| View | Backing source | Existing type / file |
|---|---|---|
| **tool-DEFS** | declared tool schemas | `promptmmu.CompactInboundTools` — byte-splice rewrite, anchored on last `cache_control` breakpoint, fail-safe identity |
| **HISTORY** | prior messages | `agent.CompactAnthropicHistory` — prefix-preserving memcpy through last breakpoint, fail-safe identity (dogfooded 6203→3486B) |
| **CONTEXT pages** | CDB image pages | `contextq` — the reference implementation; `MemoryViewRecord`, `ViewCache` keyed `(step, view_type, producer)` excluding `PolicyVersion` so drift surfaces as RECOMPUTE not a silent miss |
| **tool RESULTS** | tool executions | `vdso.VDSO` — tier-2 `entry` keyed `(tool, argHash, epoch-stamp)`; `Emit` = closed-loop invalidation; `revoked[]` = integrity epoch. **This is vToolcall, already shipped as the vDSO.** |
| **KV prefix** | provider KV cache | `vcachechain` + `vcachegov` + `cachemeta.FromKVPrefix` — and the two trust laws (A2, D4) |

The unifying type names what is already there:

```
type View struct {
    Backing    SourceOfTruth        // SourceRefs + Producer + identity axes
    Policy     MaterializationPolicy // EAGER | LAZY | SYNTHETIC | SPECULATIVE
    Invalidate InvalidationPredicate // epoch-chain | witness | PolicyVersion | TTL
    Trust      TrustFloor            // Taint, Scope, Coverage, Faithfulness, CorrectnessBearing
}
```

**Two claims to make carefully, because the naive version is false.** First, the kernel-as-context-MMU framing is *literal*, not metaphor: `contextq` is demand-paging through a trust gate, `vdso` is the vDSO serving a result with no syscall, `promptmmu` is read-only-page sharing via byte-splice, and the render plan's `stable_prefix`/`volatile_tail` split is the page-residency layout. Second — and this is where the adversary was right — the **verdict vocabularies do not unify by renaming.** `contextq`'s five (HIT/FAULT/RECOMPUTE/REFUSE/ABSTAIN) and `cachemeta`'s six (hit/miss/revalidate/transform/quarantine/fault) overlap only on HIT/FAULT; REFUSE has no single cachemeta twin and `transform`/`quarantine` have no contextq twin. A single `Resolve()` would need a **tested, explicit mapping table**, not the assertion that they are "byte-identical." Treat the unification as a **facade over the existing `cachemeta.Entry`** (every machine already has a `From*` adapter into it) plus one `Resolve()` and one `RenderPlan` assembler — additive, not a rewrite of four hot paths. Ship *that* much; defer the unified verdict until the mapping is written and tested.

**New views the frame makes obvious — labeled honestly as unvalidated design consequences, not capabilities:**

1. **Redaction view** — `Policy=SYNTHETIC`, trust floor *lowered* (Faithfulness < 1.0); a new redaction rule bumps `PolicyVersion` → RECOMPUTE. The `admitInboundResults` quarantine-in-place path is already a degenerate redaction view.
2. **Counterfactual / what-if view** — `Policy=SPECULATIVE`, invalidation = *never commits* (a scratch epoch discarded), `CorrectnessBearing=false`. Lets a planner resolve a turn against a hypothetical set without mutating `worldVer`.
3. **Deterministic-replay view** — `Policy=EAGER` but *frozen*: pin every view to a fixed `(epoch, policy_version, witness)` so `Resolve` always returns the same verdict. `Emit` is the single record point; suppress write-bumps and recorded reads stay served.
4. **Multi-model shared-result view** — the model-*agnostic* dual of `KVView`. A tool-result or summary view carries no `ModelID` in its trust-relevant axes, so the same view serves Opus and a local model — a cross-model win vCache structurally cannot give (KV is model-bound). `vdso.shareable` already does the principal-axis version of this drop.

**The trap these four must respect.** The two *shipped, safe* views (`promptmmu`, `CompactAnthropicHistory`) are **byte-faithful splices with fail-safe identity** — they never put synthetic content in front of the model. Redaction and counterfactual are `SYNTHETIC` content on the model-read wire; they **cross the exact trust line those two were built never to cross**, and putting them under the same `View` type with `CorrectnessBearing=false` would be a *lie* (a synthetic view the model reads *is* correctness-bearing by construction). They need their own threat model, not inheritance from the safe splices. State them as design consequences of the frame; do not present them as falling out "for free" — that is the same modeled-vs-measured honesty trap the project corrected on its webbench number.

---

## 6. Law D4 — never warm secrets (the orthogonal refusal that runs first)

Before any epoch or economics check, the content class gates admission. `vcachegov.Warmable` permits only `Cacheable`; `Secret` (credentials, tokens, PII) and `SecretRegulated` are refused — `Secret` unconditionally, `SecretRegulated` only through a deletion-capable surface. For vToolcall: a tool whose *result* carries secrets (a credential read, a PII fetch) is **never** cached, independent of read-shape. `ClassifyPrefix` is **fail-closed**: an unknown/unestablished class → `Secret` → refused. The rationale is the membership-oracle leak — cache-hit latency reveals "has anyone recently read string X." The classifications must be **closed-vocabulary**, not runtime-determined, to avoid the timing side-channel.

---

## 7. Economics — when faking beats running

The gate from §3 restated as arithmetic: **serve from a view iff the invalidation predicate is cheaper than the tool call, amortized over reuse.** vCache's `recall.go` form is `P·r < S·U`; the vToolcall form is per-tier.

- **Tier-1 (pure)** — the predicate is *free* (the function is re-derived; `args-sha256` is the key). Net-positive whenever the function is cheaper to recompute than to round-trip. Always sound.
- **Tier-2 (read-mutable)** — the predicate is an O(1) integer epoch compare in `keyLocked`, *after* arg resolution. The "O(1)" is honest only post-resolution: `keyLocked` calls `v.bytes(ctx, c.Args)` and at Resource granularity re-parses the args JSON, and `resourceMisnamed` parses again — for a large or CAS-backed arg payload the predicate includes a blob resolve + JSON unmarshal, paid on **every** Lookup including misses. So tier-2 is net-positive when **execution cost dominates lookup+invalidation cost** — i.e. **remote/latency-bound or I/O-bound tools**, not a cheap idempotent *local* tool routed through `fak_syscall` (which pays the tier-2 machinery for nothing). The vDSO tiers gate on `readOnly+idempotent`, **not** on cost; a cheap-local-tool guard is the host's job.

**Break-even, stated as the cliff it is:** the predicate must beat the call. For a network read (tens of ms + token cost) the epoch compare wins by orders of magnitude. For a local `Read` (no network), fak's cost to *serve* is small but so is the cost to *run* — and on the proxy wire fak cannot avoid the model re-feed anyway because it isn't serving the result. The re-feed cost is the **LLM's**, not fak's; "cheaper than re-read + re-feed" overstates fak's win on the proxy surface. The honest single-process win is **execution-side latency + the cross-agent read-once-serve-N multiplier**, not the single-session token bill (which the harness's own prompt cache already largely harvests — measure with `tools/session_audit.py`, do not assume).

**The killer use cases, ranked by how sound they are *today*:**

1. **Pure / static / idempotent lookups** (tier-1/tier-3: a content hash, a canned table, `calculate`) — zero coherence concern, sound everywhere. Lead with these.
2. **Deterministic record/replay** for tests + debugging — a frozen corpus, zero live latency, witness-pinned to the recorded world-state; the one regime where you *want* the cache not to invalidate.
3. **Re-read of an unchanged file within one process where all reads AND writes route through one kernel** — Layer 1 is exact and free; the second `Read` is served from tier-2 with no engine round-trip. Sound **only** under the kernel-mediated-writes precondition (or a content witness).
4. **Fan-out of parallel sub-agents sharing one read** (read-once-serve-N via the agent-blind key or `RegisterShareable`) — the highest *leverage*, sound **only** when the sub-agents are not separate OS processes (one in-process kernel sees every write). Across separate processes, the FILL half is unbuilt (§4).
5. **Cross-session shared cache on the trunk** — the largest blast radius and **currently unbuilt** (process-local cache; witness not keyed). A named follow-on, not a shipping capability.

---

## 8. Failure modes & design rules (adversarial)

Five named laws, each the distilled output of an attempt to break one mechanism.

### Law T — a served view must equal a fresh execution at the moment of serving
The dual of vCache's A2, with polarity swapped: a tool result is consumed as ground truth and cannot be re-derived, so a false hit is *unrecoverable corruption*, not a recoverable cost. **Rule T1 — sound-before-serve.** Serve only on epoch-match **and** (on a shared tree) a cheap witness re-check; every gate may only turn a hit into a miss. **Rule T2 — backfill is telemetry, never a correctness license.** "Serve fast, reconcile later" is legitimate only as *backfill-the-cache* (real bytes, refreshed for next time); *serve-then-correct* of a *synthetic* result is banned as a default because you cannot un-feed a consumed result.

### Law A2′ — correctness never depends on a hit (inherited mechanically)
A vToolcall hit is `served_by=vdso` **speed evidence**, never proof. The trust floor is enforced in code, not prose: `ProviderCacheVerdict` returns `LookupTransform` and `CanServe()==false` (`cachemeta.go:303,320`; `materialization.go:189-191`). **The tension to surface, not paper over:** a *speculative* serve *wants* to put unverified bytes in front of the model now (that is the speed win), which is the **opposite** of `CanServe()==false`. You cannot both serve-to-the-model and mark-non-serveable. A2′ blesses backfill-the-cache (real bytes) and the *labeling* of synthetics; it **forbids** the speculative serve. Any `[SHIPPED]` claim resting on a `verified==pending` result is a fatal violation.

### Law D4 — never warm secrets (runs first, §6)
`vcachegov.Warmable` permits only `Cacheable`; fail-closed on unknown; closed-vocabulary to deny the timing oracle.

### Law C (closed-loop) — invalidation is closed-loop only within one kernel process
The differentiator over open-loop vCache holds **per process**: `Emit` observes only completions through *this* kernel. A harness-executed `Edit`, a peer's git write, a compiler — none reach `Emit`, so `worldVer` never bumps and a subsequent tier-2 read would serve stale **unless** a content witness is bound. **Rule C1 — on any non-kernel-mediated surface, the witness is mandatory, not a tightening.** **Rule C2 — cross-process coherence is revoke-only today**; a shared FILL pool is unbuilt. Scope every "closed-loop" claim to one process and say so.

### Law S (shared-tree) — the finer eraser is load-bearing and partly unbuilt
**Rule S1 — Global granularity (full flush per observed write) is the only sound default** until witness-keying lands; it net-loses past ~1% writes. **Rule S2 — fine writes need fine reads** (`scope.go:38-46`); a host wiring a new tool whose writes are entity-fine but whose reads cannot name the entity violates soundness — `resourceMisnamed` guards the tested shape, a misconfigured tagger is a foot-gun. **Rule S3 — the FS namespace is untagged** in `nsKeywords`; until it is, `Read`/`Edit`/`Glob` get root-only invalidation = full flush, so default this workload's tier-2 reads **off** rather than claim a finer-eraser win that does not exist.

### Residual risk (honest floor)
Even with every rule: on any surface where fak does not dispatch the tool, vToolcall **cannot serve at all** — it can only *recognize* a redundant result for compaction (§2b). Where it does dispatch, the witness is not yet in the key, the cross-process pool is unbuilt, and the speculative-serve sink is unwired. The genuinely sound, shipping wins are tier-1/tier-3 (always), deterministic replay (frozen), and single-process kernel-mediated reads (Layer 1). Everything else is gated on named follow-ons. **Design posture, dual to vCache's:** be *sound-before-serve and cheaper-when-you-serve* — the moment a serve outruns its invalidation predicate, one un-mediated write turns a speed win into a corrupted answer.

---

## 9. What's new vs what exists (the build boundary)

**Reuse unchanged (shipped, grounded at HEAD):**
- `vdso.VDSO` 3-tier FastPath + `Lookup` + `Submit` dispatch (`kernel.go:347`) — the serve-as-if-it-ran mechanism.
- `vdso.Emit` + `scope.go` hierarchical epoch vector + coherence bus — Layer 1 closed-loop invalidation.
- `revoke.go` (`Revoke`, `revokedLocked`, trust epoch, fail-closed ledger) — Layer 3 refutation / integrity axis.
- `principal.go` (`MetaPrincipal`, `scopeHash`, `RegisterShareable`) — the isolation axis.
- `IsWriteShaped` / `destructive` / `resourceMisnamed` / `MissReasons` — the safety taxonomy gates.
- `cachemeta.ProviderCacheVerdict` / `LookupVerdict.CanServe()` — the mechanical trust floor (Law A2′).
- `vcachegov.Warmable` / `ClassifyPrefix` — Law D4.
- `contextq` `MaterializationVerdict` / `ViewCache` / `RenderPlan` — the materialized-view reference machine.
- `promptmmu.CompactInboundTools` / `agent.CompactAnthropicHistory` — the existing safe (byte-faithful) views; the compaction discipline §2b reuses.
- `admitInboundResults` (`gateway.go:1316`) + the `{"_quarantined":true}` stub — the result-side admission seam and the dual of the `{"_synthetic":true}` label.

**Net-new (small, mostly specializations):**
1. **FS-tool wiring** into `nsKeywords` + `entityOf` (file path as entity) + a `stat` `WitnessFunc` — turns Layer 1/2 on for `Read`/`Edit`/`Glob`/`Grep`. *Prerequisite for any shared-tree win.*
2. **Witness-keying of the tier-2 key** — the named follow-on in `revoke.go:44-49`; until it lands, shared-tree caching is Global-flush only.
3. **A cross-process shared/persistent tier-2 pool** — the FILL half (only Revoke crosses today).
4. **Result-recognition compaction step** in/near `admitInboundResults` — consult `vdso.Lookup` for a matching `(tool,args)` and mark a redundant block compactible (never replace a non-matching result). The only legitimate proxy-wire use.
5. **A `verified=pending` Meta convention + `{"_synthetic":true}` transcript wrapper + a `CacheProvisional` event** — *iff* speculative serve is ever pursued (gated on #6).
6. **A context-level `ProvisionalSink`** (`internal/ctxmmu` does not implement one) — the prerequisite for speculative-serve; until built, that rung does not exist.
7. **A PreToolUse result-injection disposition** in the harness protocol — the prerequisite for rung 2c; out of fak's boundary.

### Prior art in the tree & related issues
- The vDSO unit work (`vdso.go` units 32/35/38; `scope.go` finer eraser; `revoke.go` integrity axis) is the shipped substrate — this note is its design rationale, not new machinery for the serve path.
- The history-compaction work (`agent.CompactAnthropicHistory`, the `req.Raw` splice) and `promptmmu.CompactInboundTools` are the existing *views* the §5 frame generalizes.
- vCache (`docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md`) is the *open-loop* dual; vToolcall is the *closed-loop* case and inherits its trust laws (A2, D4) verbatim.

---

## 10. Phased plan

1. **R0 — Adopt the shipped serve path.** Route a Claude Code client's read-only tools through `fak_syscall`; surface `served_by`/miss-reason in the result. **Nothing new on the serve path — the rung is adoption + observability.** Sound for tier-1/tier-3 and single-process kernel-mediated reads today.
2. **R1 — Measure the addressable fraction.** Extend `tools/session_audit.py` to key each `Read` by `argHash(path)` and count second-and-later occurrences with no intervening write — the exact vToolcall-addressable fraction, per session and machine-wide. **No headline number until measured** (the project's modeled-vs-measured law).
3. **R2 — FS-tool wiring (net-new #1).** `nsKeywords`/`entityOf` for file paths + a `stat` `WitnessFunc` (Layer 1 + Layer 2a). Single-process win first; default shared-tree tier-2 reads **off** until R3.
4. **R3 — Witness-keying (net-new #2).** Bind the git SHA / stat witness into the tier-2 key so different-witness readers stop sharing — the precondition that makes shared-tree caching net-positive. Re-measure against the ~1% cliff before claiming a fleet win.
5. **R4 — Result-recognition compaction (net-new #4).** On the `/v1/messages` proxy, compact a redundant `tool_result` the vDSO recognizes — cost/latency only, Law-A2′-clean.
6. **R5 — Cross-process pool (net-new #3).** The FILL half of a shared cache; gated on R3 (witness-keying) for soundness.
7. **Deferred — speculative serve.** Blocked on a context-level `ProvisionalSink` (#6) and an honest economic case that does not collapse to "the trigger is the cost." Research, not roadmap.

---

## 11. Open questions

- Is there a result-injection disposition in the current Claude Code PreToolUse hook protocol (re-verify; dated 2026-06-25)? If yes, rung 2c unblocks and may precede R4.
- What is the real repeat-read fraction on representative corpora (R1)? Without it, the re-read use case is intuition, not a measured win.
- Can witness-keying (R3) be made cheap enough that the per-Lookup arg-resolution + hash does not erode the win for medium-cost tools — i.e. where exactly is the tier-2 break-even on local vs remote tools?
- For the cross-process pool (R5): durable CAS + a revocation channel that survives the 8192-ledger overflow without self-disabling the integrity axis under trunk churn — is bounded-but-not-fail-closed achievable, or is fail-closed the honest ceiling?
- Does a `stat`-based Layer-2a witness close enough of the same-size/coarse-mtime aliasing gap in practice, or must security-sensitive reads always pay the Layer-3 content hash on the serve path (collapsing the speed win for that class)?
- Multi-model shared-result views: is "no `ModelID` in the trust axes" actually safe for *every* result view, or are there tool results whose interpretation is model-specific (tokenization-sensitive, format-negotiated) and must stay model-bound like `KVView`?
