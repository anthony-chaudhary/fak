package compute

// kvresidency.go — tiered KV residency: spill cold KV across two pools instead of
// shrinking the context (#1048, child of #1045).
//
// The capacity bridge (capacity.go) can already ASK a backend "will this fit?" against
// the device pool and, separately, the host pool. But auto-sizing a KV cache on a GPU box
// only ever has ONE knob: shrink the context until the WHOLE KV fits the tight device pool.
// That leaves the roomy host RAM on the table — the exact gap llama.cpp closes with
// `--cache-ram`. When device VRAM is the binding constraint but host RAM is large, the cold
// (least-recently-attended) span of the KV cache does not need to be evicted or the context
// shrunk: it can be SPILLED to the roomy pool, with the existing eviction bridge (kvmmu /
// FAK_INKERNEL_KVMMU, model.KVCache.Evict) performing the byte movement.
//
// This file is the POLICY half: given the per-token KV cost and the headroom-adjusted bytes
// already free in each pool, it decides how many tokens stay HOT on the fast pool and how
// many spill COLD to the roomy pool, then renders that split as a classed MemoryPlan whose
// hot demand is device-scoped and cold demand host-scoped — so the existing fit planner
// (FitsMemoryPlan / RefuseMemoryPlanIfTooBig) accounts for BOTH pools, not one. It performs
// no allocation and holds no model state, the same discipline as the other compute
// estimators; the engine adapter that drives kvmmu to realize the split is the movement half.
//
// Fail-open, in the grain of the rest of capacity.go: incomplete geometry (a zero per-token
// cost) or a non-positive want yields an empty split, and a zero roomy budget degenerates to
// fast-pool-only sizing — so this can only ever keep an EQUAL-OR-LARGER effective context
// than device-only sizing, never a smaller one.

// KVResidencySplit is the planned placement of a KV cache across a fast/tight pool (device
// VRAM) and a roomy pool (host RAM). The hot span stays resident on the fast pool the decode
// reads from; the cold span is spilled to the roomy pool. The effective context is the sum
// (Tokens()), which is at least the fast-pool-only sizing and at most the requested window.
type KVResidencySplit struct {
	HotTokens  int   // tokens whose KV stays resident on the fast pool (device)
	ColdTokens int   // tokens whose KV is spilled to the roomy pool (host)
	HotBytes   int64 // device-scoped KV bytes for HotTokens
	ColdBytes  int64 // host-scoped KV bytes for ColdTokens
}

// Tokens is the effective context the split holds across both pools.
func (s KVResidencySplit) Tokens() int { return s.HotTokens + s.ColdTokens }

// Spilled reports whether any cold span was placed on the roomy pool — i.e. whether tiered
// residency bought context the fast pool alone could not hold.
func (s KVResidencySplit) Spilled() bool { return s.ColdTokens > 0 }

// PlanKVResidency splits wantTokens of KV cache across a fast/tight pool and a roomy pool. The
// hot span fills the fast pool first (the device the decode attends from); whatever does not
// fit spills cold to the roomy pool. deviceBudget and hostBudget are the headroom-adjusted
// bytes ALREADY available to KV in each pool — the caller derives them from the capacity
// probes (DeviceMemoryInfo / HostMemoryInfo or HostSystemMemoryInfo) with whatever headroom it
// reserves for weights, activations, and scratch.
//
// The result never exceeds wantTokens and never drops below what the fast pool alone holds, so
// a device-tight / host-roomy config keeps a LARGER effective context than fast-pool-only
// sizing would allow — without shrinking below it. A non-positive per-token cost (incomplete
// KV geometry) or a non-positive want yields an empty split: fail open, the same contract as
// EstimateKVStoreMemoryPlan.
func PlanKVResidency(cfg KVConfig, wantTokens int, deviceBudget, hostBudget int64) KVResidencySplit {
	if wantTokens <= 0 {
		return KVResidencySplit{}
	}
	perToken := EstimateKVStoreBytes(cfg, 1)
	if perToken <= 0 {
		return KVResidencySplit{}
	}
	hot := tokensWithinBudget(perToken, deviceBudget, wantTokens)
	cold := tokensWithinBudget(perToken, hostBudget, wantTokens-hot)
	return KVResidencySplit{
		HotTokens:  hot,
		ColdTokens: cold,
		HotBytes:   saturatingMulInt64(perToken, int64(hot)),
		ColdBytes:  saturatingMulInt64(perToken, int64(cold)),
	}
}

// KVResidencyMemoryPlan renders a residency split as a classed MemoryPlan with a device-scoped
// HOT demand and a host-scoped COLD demand, so the existing fit planner sees the hot span only
// against the device probe and the cold span only against the host probe. This is the "feed the
// residency split into the fit planner" wiring (#1048): auto-sizing that accounts for both
// pools instead of measuring the whole KV against one. An empty split yields a nil plan.
func KVResidencyMemoryPlan(split KVResidencySplit) MemoryPlan {
	var plan MemoryPlan
	if split.HotBytes > 0 {
		plan = append(plan, MemoryDemand{
			Class:  MemoryKVCache,
			Bytes:  split.HotBytes,
			Detail: "kv-residency-hot",
			Scope:  MemoryScopeDevice,
			DType:  F32.String(),
		})
	}
	if split.ColdBytes > 0 {
		plan = append(plan, MemoryDemand{
			Class:  MemoryKVCache,
			Bytes:  split.ColdBytes,
			Detail: "kv-residency-cold",
			Scope:  MemoryScopeHost,
			DType:  F32.String(),
		})
	}
	return plan
}

// tokensWithinBudget is the largest token count whose KV bytes (perToken each) fit within
// budget, capped at want. A non-positive perToken, budget, or want yields 0 — the fail-open
// floor that keeps a missing budget from ever inventing residency.
func tokensWithinBudget(perToken, budget int64, want int) int {
	if perToken <= 0 || budget <= 0 || want <= 0 {
		return 0
	}
	fit := budget / perToken
	if fit >= int64(want) {
		return want
	}
	return int(fit)
}
