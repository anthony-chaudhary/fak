---
parent_goal: goals/GOAL-audit-cache-load-prefill-effectiveness.md
sub_step: 3_engine_session_and_cache_break
witness: "go test -v ./internal/session -run 'TestWarmSplice|TestResume' -count=1 && go test -v ./internal/metrics -run 'TestCacheBreak' -count=1"
target_files:
  - internal/session/warmsplice.go
  - internal/session/resume.go
  - internal/engine/sglang_cache_observe.go
  - internal/engine/vllm_cache_observe.go
  - internal/metrics/cache_break.go
---
# Sub-Goal Objective
Audit how backend engines, session resume/splicing, and cache break accounting handle loading cached KV back into prefill or detecting re-prefill costs.

Specifically investigate:
1. Session warm splicing & resume in `internal/session/`:
   - `warmsplice.go` and `resume.go`: What determines whether a resumed turn is WARM or COLD?
   - On a warm splice, how is the parked KV reattached to skip prefill?
   - When splice fails or is absent, how does it fall back to cold re-prefill?
   - Is output bit-identical between warm splice and cold re-prefill?
2. Backend engine cache observation in `internal/engine/`:
   - In `vllm_cache_observe.go` and `sglang_cache_observe.go`: How does fak observe external engine prompt cache hits/misses?
   - Does fak drive prefix reuse directly or rely on the engine's internal prefix/radix cache?
3. Cache break cost accounting in `internal/metrics/`:
   - In `cache_break.go` and `cache_break_detector.go`: What constitutes a cache break, and how is `CostTokens` defined?
   - How does cache break accounting measure the re-prefill penalty?
4. Run the witness test and report exact file:line references and audit findings.
