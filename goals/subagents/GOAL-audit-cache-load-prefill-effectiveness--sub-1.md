---
parent_goal: goals/GOAL-audit-cache-load-prefill-effectiveness.md
sub_step: 1_native_model_prefill_and_kv_cache
witness: "go test -v ./internal/model -run 'TestKVPrefixReuseMatchesRecompute|TestGDNPrefixCache_ExactPrefixExtensionHits' -count=1"
target_files:
  - internal/model/kv.go
  - internal/model/prefill_batch.go
  - internal/model/metal_prefill.go
  - internal/model/gdn_prefix_cache.go
---
# Sub-Goal Objective
Audit how effectively KV cache is loaded back into prefill in the native model execution engine (`internal/model`).
Specifically investigate:
1. How pre-existing KV cache is injected into a session (`SessionFromPrefix`, `Clone()`, setting `s.Cache`).
2. How the prefill math executes when `s.Cache.Len() > 0`:
   - Are embeddings, Q/K/V projections, and RoPE computed strictly for the suffix tokens (`P`), or does any part re-compute the prefix?
   - How does causal attention attend over the existing cache (cached keys/values vs suffix keys/values)?
3. What architectures or execution modes hit limitations, fallbacks, or refusals when `s.Cache.Len() > 0`?
   - Metal GPU: does `prefillMetalResident` get bypassed when `s.Cache.Len() > 0`?
   - Qwen3.5 / Qwen3.6 GDN hybrid prefill: does batched hybrid prefill require `Cache.Len() == 0`? How is recurrent state handled?
   - Architecture refusals: what architectures panic or refuse `SessionFromPrefix` (e.g. Gemma4)?
4. Run the witness test and report exact code references (files and line numbers) and quantitative mechanisms.
