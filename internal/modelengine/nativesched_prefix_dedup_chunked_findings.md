# Clause B findings: in-batch prefix dedup on the resident chunked prefill lane

Status: **ABSTAIN**. Clause B of the in-batch prefix dedup goal (fak#1914 remainder,
fak#13661) is NOT sound to implement as specified on the resident-Q4_K chunked lane.

## What Clause B asks

`NativeScheduler.admit` at `internal/modelengine/nativesched.go:402` only calls
`prefillCoalesced` when `prefillChunkTokens == 0`. For a resident-Q4_K/Qwen35-hybrid
lane `qwenPrefillChunkBudget` (`internal/modelengine/nativesched_prefill.go:316`)
returns the >=16-token budget, so admission instead takes the async chunked path
(`state = schedLanePrefilling`, `promptCursor = 0`) and `advanceQwenPrefill`
(`nativesched_prefill.go:491`) prefills chunk-by-chunk. The clause wants the chunked
lane to coalesce its shared PREFIX, with the follower adopting a prefix clone and
chunking only its divergent suffix.

## Why it is not sound

1. **The resident chunked lane's cache is a recurrent (Gated-DeltaNet) hybrid and
   cannot be truncated to a shared prefix.** The resident-Q4_K fixture is a Qwen35
   hybrid: `nativeSchedulerPrefillConfig` sets
   `LayerTypes = []string{"linear_attention", "full_attention"}` at
   `internal/modelengine/nativesched_prefill_test.go:643`, so `Config.IsQwen35Hybrid()`
   is true (`internal/model/qwen35.go:38-46`). Every such `KVCache` is constructed with
   `linear = newLinearAttnCache(cfg)` (`internal/model/kvcache.go:81`,
   `internal/model/qwen35.go:171-183`), and `Clone` deep-copies it
   (`internal/model/kvcache.go:293`). Therefore `KVCache.CanEvict()` always returns a
   typed `*RecurrentEvictUnsupportedError` for this lane
   (`internal/model/kvcache.go:143-147`): recurrent state has no per-token journal, so
   a prefix truncation is formally UNSUPPORTED.

2. **The flight group already fails that case open, so a divergent-suffix prefix reuse
   for this lane can never realize.** `radixkv.CoalesceSharedPrefixNS` refuses to
   truncate a recurrent cache and returns without a match:
   `if matched < candidate.kv.Len() && candidate.kv.CanEvict() != nil { return nil, nil, 0, false, nil }`
   (`internal/radixkv/singleflight.go:292-297`). On the resident hybrid lane the
   guard is always taken for a non-exact twin, so `prefillCoalesced`'s
   `PrefixReuses` branch (`internal/modelengine/nativesched_prefix_dedup.go:157-166`)
   is unreachable. Only an EXACT twin (`matched == len(prompt)`) could reuse. That is
   not Clause B: sharing a PREFIX with divergent suffixes is exactly what this cache
   architecture cannot do through the proven radixkv truncate handoff.

3. **The chunked lane is not driven by the synchronous blocking callback the flight
   group expects.** `prefillCoalesced` runs the leader prefill synchronously via
   `coldPrefillSync` (`nativesched_prefix_dedup.go:90-97`) inside the flight's
   `fn`. The chunked lane instead advances one bounded chunk per scheduler iteration
   from `runIteration` (`nativesched.go:760`) through `advanceQwenPrefill`
   (`nativesched_prefill.go:491`), which transfers session ownership under `s.mu` and
   runs model execution outside the lock. Coalescing would require the leader to
   publish its full-prompt cache from inside the flight while the scheduler loop (not
   the admit caller) owns it — a redesign of the async ownership path, not the
   "minimal surgical" change requested, and it would serialize every follower behind
   the leader's ENTIRE prompt rather than just the shared prefix.

## Conclusion

Implementing Clause B cannot deliver its stated witness on the actual production
resident serving lane: `InBatchPrefixDedupStats().PrefixReuses` cannot increment for
a hybrid recurrent cache, so the required "leader runs once, follower chunks only the
divergent suffix" behavior is unproven and unprovable through the radixkv truncate
seam. The existing behavior (chunked lane bypasses coalescing) remains correct and is
left unchanged; the synchronous path and eligibility predicate are untouched.

Next checkable step: if the goal requires prefix reuse on a recurrent lane, it needs a
new per-layer recurrent checkpoint/journal so a hybrid `KVCache` can discard a
leader-only suffix (the operation `RecurrentEvictUnsupportedError` currently refuses),
not a change to scheduler admission.
