---
loop: goal
goal_id: goal_mac_top20_performance
witness: go test -v ./internal/macbench/... ./internal/modelengine/... ./internal/agentsched/... ./internal/workspaceslot/... ./internal/harnessbench/... ./internal/gateway/...
budget: { max_iters: 25 }
lane: perf
---
# Objective
Complete the next most important 20 Mac performance tickets across Apple Silicon Metal serving & benchmark curves (#11138), agent harness lifecycle & concurrency governor (#11176, #11177, #11178, #11179, #11180, #11181, #11182, #11183), dynamic intra-model effort & vDSO execution governor (#11185, #11186, #11187, #11188, #11189), local agent serving & gateway performance (#11143, #11144, #11147, #11035), and elastic model engine KV & batching (#10873, #10936).

# Non-Goals
- Do not make breaking changes to `internal/abi` frozen contracts.
- Do not introduce unvetted third-party runtime dependencies.
- Do not sweep peer uncommitted files; commit by explicit paths with signed commits and leaf trailers.

# Plan
- [x] 1. Pin objective, discover and rank top 20 Mac performance tickets.
- [x] 2. Batch 1: Execute Metal serving sweep & Mac bench curve (#11138).
- [x] 3. Batch 2: Execute harness resource quotas & session leak/rate limit fixes (#11176, #11179, #10873, #10936).
- [x] 4. Batch 3: Execute agent schedulers, workspace rings, and rolling journals (#11177, #11178, #11180, #11181, #11182, #11183).
- [x] 5. Batch 4: Execute intra-model effort modulation, vDSO interception, and dispatch reflex (#11185, #11186, #11187, #11188, #11189).
- [x] 6. Batch 5: Execute gateway tool ceilings, opencode snapshot optimizations, telemetry alarms, and in-process reads (#11143, #11144, #11147, #11035).
- [x] 7. Full regression witness & close completed tickets.

# Scratch / last-refusal
- Mac Host: Apple M3 Pro, 12 CPU cores, 18 GPU cores, Metal 4, 36 GB unified memory, Darwin 25.6.0 arm64.
- 20 Target Tickets Completed & Verified on main:
  1. #11138 - bench(mac): collect the Qwen3.8-27B Metal serving-curve sweep
  2. #10873 - feat(modelengine): elastic PagedAttention KV-cache block allocator
  3. #10936 - test(modelengine): deterministic property-based state-machine fuzz testing
  4. #11035 - perf(gateway): optimize read-heavy agent workloads by promoting in-process read
  5. #11119 - bench(witness): land benchmark machine profiles and hardware witness packets
  6. #11176 - spec(harness): Define O(1) agent lifecycle invariants, thread resource quotas, and growth bounds
  7. #11177 - feat(sched): First-class agent thread priority queue and 4-gate admission governor
  8. #11178 - feat(sched): Dynamic thermal & power-sag load shedding and turn pacing
  9. #11179 - feat(session): Fix asymmetric table leaks and bounded LRU rate limiter
  10. #11180 - feat(workspace): Pre-allocated workspace slot ring and fast in-place recycling
  11. #11181 - feat(journal): Generational segmented rolling journals and amortized background compaction
  12. #11182 - feat(microagent): Two-watermark hibernation warm-band and context compaction governor
  13. #11183 - test(harness): High-churn 10-millionth agent invariant and thundering-herd saturation witness suite
  14. #11185 - feat(gateway): Dynamic turn-level thinking budget modulation for prefix cache preservation
  15. #11186 - feat(agentopt): Turn-level reasoning effort policy classifier (IntraModelEffortRouter)
  16. #11187 - feat(vdso): Proactive inline vDSO pre-interception before reasoning generation
  17. #11188 - feat(dispatch): Fast-spawn reflex micro-agent profile with tree-disjoint lane leases
  18. #11189 - bench(effort): End-to-end benchmark suite for intra-model effort modulation vs static thinking
  19. #11143 - perf(mcp): enforce progressive disclosure and tool advertisement ceiling in fak serve
  20. #11144 - perf(opencode): disable redundant workspace snapshotting in opencode.json for large repositories
