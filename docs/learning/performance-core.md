---
title: "fak learning path — L400"
description: "The L400 performance stage explains cache reuse, addressable eviction, scaling laws, and the labs that verify them."
---

# L400 — The Performance Core: cache reuse, addressable eviction, and the scaling laws

**Stage 4 of the path** · prev: [L300 — The Security Core](security-core.md) · next: [L500 — Serving, Integration, and the In-Kernel Model](serving-integration.md) · back to the [overview and L100–L200](../../LEARNING-PATH.md)

**Read:** [`docs/explainers/code-linting-at-the-kernel.md`](../explainers/code-linting-at-the-kernel.md)

**Lab:**
```bash
go test ./internal/codelint/ -count=1 -timeout 120s -run 'TestGoPackReportsParseError|TestPackForKnownAndUnknown|TestParseDiagnosticsGCCStyle|TestHasErrorAndSummaryOrdersErrorsFirst' -v
```

**Checkpoint:** Explain pack-by-extension routing and why a clean file yields no opinion while a semantic (not syntactic) error is ignored by the Go pack. State why feeding errors back at the write boundary is the concrete coding-agent payoff of the FAK 310 write gate, and how it underwrites the L600 SWE-bench coding-agent material.

---

## L400 — The Performance Core: cache reuse, addressable eviction, and the scaling laws

**Theme.** Why agents stress the cache, prefill-elimination economics, the addressable/bijective KV-MMU, RadixAttention reuse, the vDSO, durable session recall, and the first-order scaling law (incl. cache legality and residency).

**Who joins here.** An ML-systems or kernel-minded reader who has the Foundations KV-cache unit and the security write-time gate. Join here if you want the speed story and how it converges with the security boundary, rather than the enforcement details. Memory/RAG engineers continue here for the scaling laws after the durability gate.

**Assumes you can already pass:** **FAK 201**, **FAK 205**, **FAK 310**.

| Course | Hard prerequisites |
|---|---|
| **FAK 401** — How Agents Stress the KV Cache | **FAK 201** |
| **FAK 402** — Prefill Elimination and the A/B/C Cost Arms | **FAK 401** |
| **FAK 403** — The 10 SOTA Serving Optimizations and the Honest Baseline | **FAK 402** |
| **FAK 404** — Addressable KV Cache: Exact Span Removal (The Second Flip) | **FAK 310**, **FAK 401** |
| **FAK 405** — RadixAttention Prefix Reuse + LRU Eviction | **FAK 401** |
| **FAK 406** — KV-MMU: Addressable, Bijective Span Eviction | **FAK 405**, **FAK 404** |
| **FAK 407** — The 3-Tier Tool vDSO (Fast-Path Cache) | **FAK 205**, **FAK 307** |
| **FAK 408** — What the Semantics-Layer Vantage Unlocks | **FAK 204**, **FAK 406** |
| **FAK 409** — recall: Session Core-Dump That Survives the Boundary | **FAK 407** |
| **FAK 410** — contextq: On-Demand Context Materialization | **FAK 409** |
| **FAK 411** — ed25519 Deletion Certificates | **FAK 317**, **FAK 406** |
| **FAK 412** — The First-Order Scaling Law of Agents | **FAK 402**, **FAK 316** |
| **FAK 413** — Cache Legality: The Next Scaling Wall | **FAK 412** |
| **FAK 414** — Three Regimes and the Agent-City Saturation Points | **FAK 413** |

### FAK 401 — How Agents Stress the KV Cache

**Prerequisites:** **FAK 201**

**You'll be able to:**
- Explain why a broken cache turns a linear loop into a quadratic one in latency and dollars
- Show why caching matters far more at 239:1 input:output (agents) than at 2:1 (chat)
- Name the failure modes (eviction during tool latency, head-mutation, injected timestamps, unstable JSON) and the zero-infra fix
- Mark why the high public cache number is just the frozen-trajectory ceiling, and the three axes that bend it toward 0% — flexibility, per-turn tool density, and cross-agent fan-out (and why fan-out is a fleet metric, not one agent's hit %)

**Read:** [`docs/explainers/kv-cache-agentic-context.md`](../explainers/kv-cache-agentic-context.md), [`docs/explainers/frozen-trajectory-cache-cliff.md`](../explainers/frozen-trajectory-cache-cliff.md), [`docs/explainers/context-tape-visuals.md`](../explainers/context-tape-visuals.md)

**Lab:**
```bash
Take a prompt with a per-request UUID at the head; move it to the tail and re-run the LCP analysis to reproduce the 0.3% -> 87% hit-rate jump described in the doc.
python tools/cache_curve.py compound   # watch the frozen 99% ceiling collapse along the flex + tool-density axes
python tools/context_tape.py trajectory <your-session>.jsonl --svg session.svg   # SEE the reused prefix dwarf the fresh tip, turn by turn, on YOUR own session (docs/explainers/context-tape-visuals.md)
```

**Checkpoint:** Explain why a changed file causes a visible cache miss (recompute) rather than a silently stale answer, and the one condition (result cache keyed on call args alone) under which staleness CAN go silent; give the fix (key on content version).

### FAK 402 — Prefill Elimination and the A/B/C Cost Arms

**Prerequisites:** **FAK 401**

**You'll be able to:**
- Distinguish arm A (naive re-send), arm B (per-agent KV, duplicated prefixes), and arm C (fak fused, one shared prefix)
- State when fak does NOT help (single-turn, zero shared context, tiny contexts)
- Read the 20-24x as vs naive, not vs a tuned baseline

**Read:** [`docs/prefill-elimination-explained.md`](../prefill-elimination-explained.md)

**Lab:**
```bash
go run ./cmd/fak swebench describe --difficulty <file>  (inspect live cost numbers); or read internal/swebench/cost.go to see how A/B/C token totals are computed.
```

**Checkpoint:** Distinguish arm B from arm C and state when fak does NOT help. Note that the 20-24x is vs naive, not vs a tuned baseline.

### FAK 403 — The 10 SOTA Serving Optimizations and the Honest Baseline

**Prerequisites:** **FAK 402**

**You'll be able to:**
- List which of the 10 optimizations fak marks IMPLEMENTED vs PARTIAL vs ENGINE-LEVEL and map each to its owning engine
- Name the three sources of the 1.5-4x-vs-tuned gain
- Name the three things the gain is explicitly NOT from (raw model speed, basic KV reuse, quantization)

**Read:** [`docs/explainers/sota-optimizations.md`](../explainers/sota-optimizations.md)

**Lab:**
```bash
From the SOTA table, list every optimization fak marks IMPLEMENTED vs PARTIAL vs NOT-FOCUSED/ENGINE-LEVEL, then map each to the engine that owns it (llama.cpp / vLLM / SGLang).
```

**Checkpoint:** When fak reports '1.5-4x vs tuned SOTA', name the three sources of the gain and the three things it is explicitly NOT from.

### FAK 404 — Addressable KV Cache: Exact Span Removal (The Second Flip)

**Prerequisites:** **FAK 310**, **FAK 401**

**You'll be able to:**
- Trace the four senses of 'addressable' (prefix / span / content / queryable-context) onto fak's status
- Explain why llama.cpp's K-shift drifts ~1e-6 while a single re-rotation from Kraw is exact
- State honestly that bit-exact span removal is proven on a synthetic model in internal/kvmmu but not yet wired into the live agent HTTP loop

**Read:** [`docs/explainers/addressable-kv-cache.md`](../explainers/addressable-kv-cache.md)

**Lab:**
```bash
Trace the four senses of 'addressable' onto fak's status; identify which test pins exact span removal (TestKVQuarantineEqualsNeverSaw, max|delta|=0).
```

**Checkpoint:** Explain why llama.cpp's K-shift drifts ~1e-6 while fak's single re-rotation from Kraw is exact, and why bit-exact span removal is proven on a synthetic model but NOT yet wired into the live fak agent HTTP loop.

### FAK 405 — RadixAttention Prefix Reuse + LRU Eviction

**Prerequisites:** **FAK 401**

**You'll be able to:**
- Explain why longest-prefix reuse + suffix prefill is bit-identical to a from-scratch prefill (logits/argmax match)
- Explain 'upward collapse': why removing a leaf can make its parent a new eviction candidate
- State the refcount-conservation invariant across a Lookup->Insert->Done cycle and why the root boundary lease is counted for a cold request

**Read:** [`docs/proofs/radixkv.md`](../proofs/radixkv.md)

**Lab:**
```bash
go test ./internal/radixkv/ -count=1 -timeout 120s -run 'TestReuseThroughSplitMatchesRecompute|TestLRUEvictsOldestRetainsHotAndLeased|TestLRUUpwardCollapse|TestRefcountConservationCycleNetsZero' -v
```

**Checkpoint:** Explain 'upward collapse' and state the refcount-conservation invariant (Sigma node.refs across a Lookup->Insert->Done cycle) and why the root boundary lease must be counted for a cold request.

### FAK 406 — KV-MMU: Addressable, Bijective Span Eviction

**Prerequisites:** **FAK 405**, **FAK 404**
  ·  **Background:** **FAK 206**

**You'll be able to:**
- State the two structural invariants (bijection over live spans; exact span addressing)
- Explain why eviction must be content/id-driven, not positional, and how RoPE re-rotation of survivors makes post-evict cache byte-identical to never-saw-it
- Identify what is explicitly SCOPED-OUT (concurrent-eviction data-race freedom, deferred to Gobra)

**Read:** [`docs/proofs/kvmmu.md`](../proofs/kvmmu.md)

**Lab:**
```bash
go test ./internal/kvmmu/ -count=1 -timeout 120s -run 'TestLedgerRenumberAfterMiddleEvict|TestWriteTimeEvictEqualsNeverSaw|TestEvictionIsContentDrivenNotPositional' -v
```

**Checkpoint:** State the two structural invariants and explain why eviction must be content/id-driven, not positional. What is explicitly SCOPED-OUT?

### FAK 407 — The 3-Tier Tool vDSO (Fast-Path Cache)

**Prerequisites:** **FAK 205**, **FAK 307**

**You'll be able to:**
- Trace the fixed lookup order (tier-1 pure recompute, tier-3 static, tier-2 cached)
- Name the four conditions that downgrade a tier-2 hit to a MISS
- Explain why the integrity epoch advances monotonically on a non-empty Revoke and is a no-op on an empty-witness Revoke

**Read:** [`docs/proofs/vdso.md`](../proofs/vdso.md), [`docs/explainers/vdso-revoke-as-comm-revoke.md`](../explainers/vdso-revoke-as-comm-revoke.md)

**Lab:**
```bash
go test -run 'Unit25|Unit26_27|Unit28|Unit29|Unit34_Miss|Scope_Soundness' ./internal/vdso/ -count=1 -timeout 120s -v
```

**Checkpoint:** Trace the fixed lookup order and name the four distinct conditions that downgrade a tier-2 hit to a MISS. Explain why the integrity (trust) epoch advances monotonically on a non-empty Revoke and is a no-op on an empty-witness Revoke.

### FAK 408 — What the Semantics-Layer Vantage Unlocks

**Prerequisites:** **FAK 204**, **FAK 406**

**You'll be able to:**
- For each of the five optimizations (us filter, exact rewind/branch, transactional turn, structure-aware eviction, per-principal audit), name the structure it depends on
- Explain why a serving engine on an anonymous token stream cannot do bit-exact middle-eviction even with zero-copy read access to fak's arena
- Distinguish 'faster at the same thing' from operations structurally impossible without identity + state machine + owned arena

**Read:** [`docs/MEMORY-LAYERS-EXPLAINER.md`](../MEMORY-LAYERS-EXPLAINER.md)

**Lab:**
```bash
For each of the five optimizations, name the one piece of structure (identity, state machine, or owned-arena+Kraw) it depends on and check its SHIPPED/SEAM-SHIPPED tag in the doc.
```

**Checkpoint:** Explain why a serving engine sitting on an anonymous token stream cannot do bit-exact middle-eviction even with zero-copy read access to fak's arena (gate 3: Kraw is a write-time decision).

### FAK 409 — recall: Session Core-Dump That Survives the Boundary

**Prerequisites:** **FAK 407**
  ·  **Background:** **FAK 205**

**You'll be able to:**
- Explain what 'same answer as replay' reduces to for a content-addressed image (per-page byte-identity + deterministic exclusion set)
- Explain why Load refuses the whole image if any blob fails to re-hash to its key
- Explain how run-to-run determinism is witnessed against Go's randomized map iteration

**Read:** [`docs/proofs/recall.md`](../proofs/recall.md)

**Lab:**
```bash
go test ./internal/recall/ -count=1 -timeout 120s -run 'TestBenignPageRoundTripsByteIdentical|TestSessionIsSelfContained|TestRecallWorkingSetExcludesPoison|TestRecallIsDeterministicAcrossRepeatedCalls' -v
```

**Checkpoint:** Explain what 'same answer as replay' reduces to for a content-addressed image. Why does Load refuse the whole image if any blob fails to re-hash to its key, and how is run-to-run determinism witnessed against Go's randomized map iteration?

### FAK 410 — contextq: On-Demand Context Materialization

**Prerequisites:** **FAK 409**

**You'll be able to:**
- Explain why the unqualified byte-identity theorem is FALSE for the summary path and how it must be restated
- State the summary path's contract (FaithfulnessProbe==1.0 extractive prefix + reported Coverage)
- Name the five MaterializationVerdicts

**Read:** [`docs/proofs/contextq.md`](../proofs/contextq.md)

**Lab:**
```bash
go test ./internal/contextq/ -count=1 -timeout 120s -run 'TestMaterializeByteIdentical|TestMaterializationDeterministic' -v
```

**Checkpoint:** Why is the unqualified byte-identity theorem FALSE for the summary path, and how must it be restated? Name the five MaterializationVerdicts.

### FAK 411 — ed25519 Deletion Certificates

**Prerequisites:** **FAK 317**, **FAK 406**

**You'll be able to:**
- List the four ordered verification rungs and what each rejects
- State the three honest non-claims (self-attesting in v1, max|delta|=0 checked only as a signed string, EvictedCount is a self-report)
- Re-derive the journal anchor row to make the receipt re-checkable, not merely asserted

**Read:** [`docs/proofs/deletioncert.md`](../proofs/deletioncert.md)

**Lab:**
```bash
go test ./internal/deletioncert/ -count=1 -timeout 120s -run 'TestMintVerifyRoundTrip|TestTamperDetected|TestNonBitExactRejected|TestAnchorAbsent|TestSubjectRelabelRejected|TestNilVerifierFailsClosed' -v
```

**Checkpoint:** List the four ordered verification rungs and explain what each rejects. State the THREE honest non-claims.

### FAK 412 — The First-Order Scaling Law of Agents

**Prerequisites:** **FAK 402**, **FAK 316**
  ·  **Background:** **FAK 203**

**You'll be able to:**
- Write the law: agents x turns x working-set x reread rate x legality checks
- Explain why reread rate is the only safe term to attack, and only when legality permits
- Explain why the measured 60.3x session result is not a '60x faster model' but a deletion of duplicate setup re-reads

**Read:** [`docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md`](../notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md)

**Lab:**
```bash
go run ./cmd/longctxbench  (compute the contention-free work floor; compare naive setup payments = agents x turns vs coherent = 1 per legal shared scope for a 5-agent x 50-turn workload)
```

**Checkpoint:** Explain why the measured 60.3x session result is NOT a '60x faster model' and which term in the scaling law it actually deletes.

### FAK 413 — Cache Legality: The Next Scaling Wall

**Prerequisites:** **FAK 412**

**You'll be able to:**
- State net reuse value = shared read hits - invalidation cost - stale-read risk, keyed on (digest, scope, world-version, taint)
- Distinguish physical (residency) coherence from semantic (legality) coherence
- Give an example where a hit passing every hardware coherence check is still the wrong answer (a git push invalidating cached git status)

**Read:** [`docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md`](../notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md)

**Lab:**
```bash
Work Scenario B from the doc on paper: a byte-coherent hot KV span after a git push — state the two distinct failures (stale fact; cross-tenant leak) and which key field (world-version / scope) the coherence kernel uses to evict exactly that span.
```

**Checkpoint:** Distinguish physical (residency) coherence from semantic (legality) coherence and give one example where a hit passing every hardware coherence check is still the wrong answer.

### FAK 414 — Three Regimes and the Agent-City Saturation Points

**Prerequisites:** **FAK 413**

**You'll be able to:**
- Distinguish single-chat / long-session / agent-city regimes by bottleneck
- Compute a Qwen2.5-7B KV geometry and show a 100k-token cache is ~143x too big for L2
