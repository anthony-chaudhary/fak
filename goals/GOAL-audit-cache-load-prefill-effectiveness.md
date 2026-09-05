---
loop: goal
goal_slug: audit-cache-load-prefill-effectiveness
witness: "go test -v ./internal/model -run TestGDNPrefixCache_ExactPrefixExtensionHits -count=1 && go test -v ./internal/radixkv -run TestReuseThroughSplitMatchesRecompute -count=1 && go test -v ./internal/session -run TestWarmSpliceWiredIntoResumeLoop -count=1"
budget: { max_iters: 15 }
lane: audit
---
# Objective
Audit how effectively cache is actually loaded back into prefill across fak inference, native model execution, radix KV tree, HAL/Metal backends, engine integration, and gateway session splicing.

# Non-Goals
- Do not make breaking architectural changes or mutate frozen ABI (`internal/abi`).
- Do not commit peer WIP or unvetted modifications.
- Do not rely on unverified assertions; trace actual execution graphs in code and tests.

# Plan
- [x] 1. Subagent 1: Audit native model prefill & KV cache loading (`internal/model/kv.go`, `internal/model/prefill_batch.go`, `internal/model/metal_prefill.go`, `internal/model/gdn_prefix_cache.go`).
- [x] 2. Subagent 2: Audit radix KV prefix discovery, snapshot tiers, and InKernel decode loop (`internal/radixkv/`, `internal/agent/inkernel_decode.go`, `internal/agent/inkernel_planner.go`).
- [x] 3. Subagent 3: Audit engine observation, session warm-splice, and cache break cost accounting (`internal/engine/`, `internal/session/warmsplice.go`, `internal/session/resume.go`, `internal/metrics/cache_break.go`).
- [x] 4. Coordinator Synthesis: Synthesize comprehensive audit report with quantitative evidence, identified bottlenecks/fallbacks, and test witnesses.

# Results and Verification Evidence

## Comprehensive Audit: Cache Loading and Prefill Effectiveness

### 1. Architectural Mechanisms of Cache Re-Use in Prefill
- **Native Model Execution (`internal/model`):**
  - Existing KV cache is cloned via `SessionFromPrefix(prefix *KVCache) *Session` (`kv.go:23-29`), establishing `base := s.Cache.Len()`.
  - Projections (Q/K/V GEMMs), embeddings, RoPE rotations, and SwiGLU activations are computed strictly on suffix tokens of length $P = \text{len}(ids) - \text{base}$ (`prefill_batch.go:43-150`, `prefill_q4k.go:65-221`). No prefix projection FLOPs are recomputed.
  - Suffix keys and values are appended to `cache.K[l]` and `cache.V[l]` (`prefill_attn.go:5-17`). Causal attention attends over $[\,j0, \text{base}+t+1\,)$, reading pre-computed prefix keys directly from the cloned cache without re-projection.
  - Suffix prefill is proven bit-identical to a full recompute (`kvreuse_test.go:11-53`).
- **Radix KV & Tiered Snapshots (`internal/radixkv`):**
  - Compressed trie walks (`radixkv.go:409-438`) discover the longest prefix.
  - Edge splits (`radixkv.go:478-498`) truncate child KV caches using `TryEvict` on the tail span, preserving exact unrotated keys (`Kraw`) without trigonometric drift.
  - Three snapshot tiers are supported: Device L1 $\to$ Host DRAM L2 (`host_l2.go`) $\to$ Remote HTTP L3 (`remote_l3.go`).
- **In-Kernel Decode Loop Prefill Orchestration (`internal/agent/inkernel_decode.go`):**
  - Matched snapshots are restored directly into backend sessions (`matchedSnapshot.Restore(s)`).
  - Exact-hit prompt matching (`matched == len(ids)`):
    - When cached logits exist: **0 prefill tokens** computed; prefill phase is completely bypassed (`inkernel_decode.go:219`). Redundant snapshot re-admission is skipped (`skipExactDeviceL1Readmission`).
    - When cached logits are missing: `inKernelRefeedLastTokenForExactHit` evicts only the last token (`matched = len(ids)-1`) and re-feeds 1 token to compute logits.
  - Multi-turn prefill savings verified: **81% prefill tokens saved** (400 vs 2160 computed, 1.28x wall time speedup in `TestInKernelReuseMultiTurnPrefillSavings`).
- **Session Warm Splicing (`internal/session`):**
  - Paused sessions reattach parked KV caches via `WarmKVStore.Splice(trace)` (`warmsplice.go:178-240`), resuming as `ResumeWarm` and bypassing prefill entirely.
  - Degrades safely to `ResumeCold` (full re-prefill) on cache miss or eviction without hanging.
- **External Engine Observation & Cache Break Accounting (`internal/engine`, `internal/metrics`):**
  - External engines (vLLM, SGLang) are observed passively via Prometheus metrics (`vllm_cache_observe.go`, `sglang_cache_observe.go`) with `CachePassiveObserve`.
  - Cache breaks (system prompt change, tool set change, altered history) are detected and priced in `CostTokens` (the warm prefix that must be re-prefilled) via `cache_break_detector.go`.

### 2. Identified Bottlenecks, Limitations, and Fallback Edges
1. **Metal Resident Prefill Fallback (`metal_prefill.go:213-218`):**
   - The resident single-command-buffer GPU pipeline (`prefillMetalResident`) only runs when `s.Cache.Len() == 0`.
   - When `s.Cache.Len() > 0`, it falls back to the hybrid path: GPU GEMMs with CPU RoPE, causal attention, and SwiGLU.
2. **Qwen3.5 GDN Hybrid Model Prefill Constraints (`kv.go:1047-1064`):**
   - Metal and Q8 hybrid prefill paths require `s.Cache.Len() == 0`. When `s.Cache.Len() > 0`, Q8 drops into serial per-token decode loop (`tokenHiddenQ`). Resident Q4_K (`q4kQwen35HybridPrefillAtPositionOK`) does support `base > 0`.
3. **Recurrent State Truncation Impossibility:**
   - GDN recurrent linear attention cannot be truncated mid-edge (`RecurrentEvictUnsupportedError`).
   - Mitigated by 64-token checkpoint snapshots (`inKernelSnapshotCheckpoint`), but arbitrary boundary edge splits force a fallback to cold re-prefill.
4. **Non-KV Architecture Refusal:**
   - Gemma4 stores token history rather than KVCache; `SessionFromPrefix` loudly panics by design (`gemma4_session.go:60`, `kv.go:24-27`).

### 3. Verification Witnesses
All subagent packages and cross-cutting integration witnesses verified green:
- `go test -v ./internal/model -run 'TestGDNPrefixCache_ExactPrefixExtensionHits|TestSessionFromPrefixRefusesGemma4' -count=1` -> PASS
- `go test -v ./internal/radixkv -run 'TestReuseThroughSplitMatchesRecompute|TestHostDRAML2StagesEvictsAndRestoresCompletePrefix|TestRemoteL3HTTPRestoresAfterAllLocalStateIsRemoved' -count=1` -> PASS
- `go test -v ./internal/agent -run 'TestInKernelReuseMatchesFullPrefill|TestInKernelReuseExactHitUsesCachedPromptLogits|TestInKernelReuseMultiTurnPrefillSavings' -count=1` -> PASS
- `go test -v ./internal/session -run 'TestWarmSpliceWiredIntoResumeLoop|TestWaitResumeWarmSplice|TestWaitResumeColdFallback' -count=1` -> PASS
- `go test -v ./internal/metrics -run 'TestCacheBreakDetectorCleanPassOnUnmutatedTurn|TestCacheBreakDetectorAttributesCauseAndPricesColdSpan' -count=1` -> PASS

# Scratch / last-refusal
