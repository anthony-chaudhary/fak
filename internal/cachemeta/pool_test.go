package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// poolKVEntry builds a pooled KV-span entry with a COMPLETE materialization key and an
// explicit taint/scope, the shape PoolReuseVerdict gates on.
func poolKVEntry(model string, taint abi.TaintLabel, scope abi.ShareScope) Entry {
	e := FromKVPrefix(KVPrefix{
		TokenDigest: "deadbeef",
		Length:      4000,
		ModelID:     model,
		TokenizerID: model + "-tok",
		Owner:       "tenant-a",
	},
		WithSerializer("ser-1"),
		WithPolicyVersion("pol-1"),
	)
	e.Derivation.PositionMode = PositionPrefixAligned
	e.Labels = map[string]string{
		"position_regime":  "rope-theta-1e6",
		"admitter_version": "adj-1",
	}
	e.Security.Taint = taint
	e.Security.Scope = scope
	e.Security.AdmittedBy = "adjudicator"
	e.Security.AdmissionVerdict = AdmissionAllow
	return e
}

func wantKey(model string) MaterializationKey {
	return MaterializationKey{
		ModelID:         model,
		TokenizerID:     model + "-tok",
		SerializerID:    "ser-1",
		PositionRegime:  "rope-theta-1e6",
		PolicyVersion:   "pol-1",
		AdmitterVersion: "adj-1",
	}
}

// TestPoolProfileRegimes: only a pooled, coherent, zero-copy tier is FabricShareable;
// a host-private tier is not reachable; a reachable copy-only tier is reachable but not
// shareable.
func TestPoolProfileRegimes(t *testing.T) {
	pp := DefaultPoolProfiles()
	if pp[TierDRAM].Reachable() {
		t.Fatalf("host-private DRAM must not be reachable by another host")
	}
	if !pp[TierCXL].FabricShareable() {
		t.Fatalf("a coherent zero-copy CXL pool must be fabric-shareable")
	}
	if pp[TierRemote].FabricShareable() {
		t.Fatalf("a copy-only remote pool must NOT be fabric-shareable")
	}
	if !pp[TierRemote].Reachable() {
		t.Fatalf("a multi-host remote pool must be reachable (copy-only)")
	}
}

// TestFleetReuseCoherentPoolSavesBothAxes: a coherent CXL pool turns N prefills + N
// copies into ONE prefill + ONE copy — the headline multi-tenant economics.
func TestFleetReuseCoherentPoolSavesBothAxes(t *testing.T) {
	profiles := DefaultTierProfiles()
	pools := DefaultPoolProfiles()
	const tenants = 8
	const tokens = 4000
	const bytes = 64 << 20
	r := PlanFleetReuse(FleetReuseRequest{
		Tenants:              tenants,
		Tokens:               tokens,
		SizeBytes:            bytes,
		PerTokenPrefillNanos: 2_000_000,
		Profile:              profiles[TierCXL],
		Pool:                 pools[TierCXL],
	})
	if !r.Shareable {
		t.Fatalf("CXL pool should be fabric-shareable")
	}
	if r.PooledPrefillTokens != tokens {
		t.Fatalf("coherent pool should prefill once (%d), got %d", tokens, r.PooledPrefillTokens)
	}
	if r.PrefillTokensSaved != tenants*tokens-tokens {
		t.Fatalf("prefill saved should be (N-1)*tokens=%d, got %d", tenants*tokens-tokens, r.PrefillTokensSaved)
	}
	if r.PooledResidentBytes != bytes {
		t.Fatalf("coherent pool should hold ONE copy (%d), got %d", bytes, r.PooledResidentBytes)
	}
	if r.BytesDeduplicated != (tenants-1)*bytes {
		t.Fatalf("bytes dedup should be (N-1)*bytes=%d, got %d", (tenants-1)*bytes, r.BytesDeduplicated)
	}
	// A coherent attend stages nothing — cheaper than a re-prefill.
	if r.PerTenantAttendNanos >= r.PerTenantRecomputeNanos {
		t.Fatalf("zero-copy attend (%d) should beat recompute (%d)", r.PerTenantAttendNanos, r.PerTenantRecomputeNanos)
	}
}

// TestFleetReuseCopyOnlyPoolSavesPrefillNotMemory: a reachable but non-coherent pool
// saves the re-prefill (the owner builds it once, others copy) but still holds N copies.
func TestFleetReuseCopyOnlyPoolSavesPrefillNotMemory(t *testing.T) {
	profiles := DefaultTierProfiles()
	pools := DefaultPoolProfiles()
	const tenants = 8
	const tokens = 4000
	const bytes = 64 << 20
	r := PlanFleetReuse(FleetReuseRequest{
		Tenants:              tenants,
		Tokens:               tokens,
		SizeBytes:            bytes,
		PerTokenPrefillNanos: 2_000_000,
		Profile:              profiles[TierRemote],
		Pool:                 pools[TierRemote],
	})
	if r.Shareable {
		t.Fatalf("copy-only remote pool must not be shareable")
	}
	if !r.Reachable {
		t.Fatalf("remote pool must be reachable")
	}
	if r.PrefillTokensSaved != tenants*tokens-tokens {
		t.Fatalf("copy-only pool still saves the re-prefill, got %d", r.PrefillTokensSaved)
	}
	if r.BytesDeduplicated != 0 {
		t.Fatalf("copy-only pool holds N copies, so dedup must be 0, got %d", r.BytesDeduplicated)
	}
}

// TestFleetReuseHostPrivateSavesNothing: a host-private tier is unreachable by other
// hosts, so each tenant rebuilds its own — savings are zero on both axes (the baseline).
func TestFleetReuseHostPrivateSavesNothing(t *testing.T) {
	profiles := DefaultTierProfiles()
	pools := DefaultPoolProfiles()
	r := PlanFleetReuse(FleetReuseRequest{
		Tenants:              8,
		Tokens:               4000,
		SizeBytes:            64 << 20,
		PerTokenPrefillNanos: 2_000_000,
		Profile:              profiles[TierDRAM],
		Pool:                 pools[TierDRAM],
	})
	if r.Reachable || r.Shareable {
		t.Fatalf("host-private DRAM is neither reachable nor shareable")
	}
	if r.PrefillTokensSaved != 0 || r.BytesDeduplicated != 0 {
		t.Fatalf("host-private baseline saves nothing, got prefill=%d bytes=%d", r.PrefillTokensSaved, r.BytesDeduplicated)
	}
}

// TestPoolReuseTrustedKeyMatchHits: a trusted, fleet-scoped cell with a matching key is
// reusable across a tenant boundary.
func TestPoolReuseTrustedKeyMatchHits(t *testing.T) {
	stored := poolKVEntry("qwen3", abi.TaintTrusted, abi.ScopeFleet)
	v := PoolReuseVerdict(stored, wantKey("qwen3"))
	if !v.CanServe() {
		t.Fatalf("a trusted, fleet-scoped, key-matched cell should serve, got %s/%s", v.Kind, v.Reason)
	}
}

// TestPoolReuseModelMismatchRefused: a cell built under a different model is garbage and
// is refused even when trusted and shareable.
func TestPoolReuseModelMismatchRefused(t *testing.T) {
	stored := poolKVEntry("qwen3", abi.TaintTrusted, abi.ScopeFleet)
	v := PoolReuseVerdict(stored, wantKey("llama4"))
	if v.CanServe() || v.Reason != ReasonModelMismatch {
		t.Fatalf("model-mismatched span must be refused with model_mismatch, got %s/%s", v.Kind, v.Reason)
	}
}

// TestPoolReuseAgentScopeRefused: the fail-closed private default (ScopeAgent) refuses
// cross-tenant reuse even for a trusted, key-matched cell.
func TestPoolReuseAgentScopeRefused(t *testing.T) {
	stored := poolKVEntry("qwen3", abi.TaintTrusted, abi.ScopeAgent)
	v := PoolReuseVerdict(stored, wantKey("qwen3"))
	if v.CanServe() || v.Reason != ReasonScopeDenied {
		t.Fatalf("agent-private cell must be refused with scope_denied, got %s/%s", v.Kind, v.Reason)
	}
}

// TestPoolReuseQuarantinedNeverServed: a poisoned/quarantined cell must leave the pool —
// a non-serveable Quarantine verdict, never a hit.
func TestPoolReuseQuarantinedNeverServed(t *testing.T) {
	stored := poolKVEntry("qwen3", abi.TaintQuarantined, abi.ScopeFleet)
	v := PoolReuseVerdict(stored, wantKey("qwen3"))
	if v.CanServe() || v.Kind != LookupQuarantine {
		t.Fatalf("quarantined cell must be a non-serveable quarantine, got %s/%s", v.Kind, v.Reason)
	}
}

// TestPoolReuseTaintedRefused: an un-adjudicated (merely tainted) cell is refused — only
// proven-trusted bytes may be aliased across a tenant boundary.
func TestPoolReuseTaintedRefused(t *testing.T) {
	stored := poolKVEntry("qwen3", abi.TaintTainted, abi.ScopeFleet)
	v := PoolReuseVerdict(stored, wantKey("qwen3"))
	if v.CanServe() || v.Reason != ReasonTaintDenied {
		t.Fatalf("merely-tainted cell must be refused with taint_denied, got %s/%s", v.Kind, v.Reason)
	}
}

// TestFleetReplayAggregateGeneralizes proves the fleet win is not a single cherry-picked
// prefix: it replays a realistic mixed workload — every agent shares one warm system+tool
// prefix (a coherent CXL pool) AND carries its own private working context (host-private
// DRAM) — and tallies the aggregate. The shared prefix collapses to one copy across the
// fleet; the private contexts save nothing (correctly, they are not shared). It also
// shows the savings are HONEST: the trust gate reuses the trusted shared cell but refuses
// a poisoned variant, so a deduplicated copy is never a poisoned alias.
func TestFleetReplayAggregateGeneralizes(t *testing.T) {
	profiles := DefaultTierProfiles()
	pools := DefaultPoolProfiles()
	const agents = 6
	workload := []FleetReuseRequest{
		// shared system+tool prefix on the coherent CXL pool
		{Tenants: agents, Tokens: 4000, SizeBytes: 64 << 20, PerTokenPrefillNanos: 2_000_000, Profile: profiles[TierCXL], Pool: pools[TierCXL]},
		// each agent's private working context on host-private DRAM
		{Tenants: agents, Tokens: 2000, SizeBytes: 32 << 20, PerTokenPrefillNanos: 2_000_000, Profile: profiles[TierDRAM], Pool: pools[TierDRAM]},
	}
	var savedTokens, dedupBytes, privatePrefill, pooledPrefill int64
	for _, req := range workload {
		r := PlanFleetReuse(req)
		savedTokens += r.PrefillTokensSaved
		dedupBytes += r.BytesDeduplicated
		privatePrefill += r.PrivatePrefillTokens
		pooledPrefill += r.PooledPrefillTokens
	}
	// Only the shared prefix saves: (agents-1) re-prefills and (agents-1) copies avoided.
	if savedTokens != (agents-1)*4000 {
		t.Fatalf("fleet prefill saved = %d, want %d (private contexts must save nothing)", savedTokens, (agents-1)*4000)
	}
	if dedupBytes != (agents-1)*(64<<20) {
		t.Fatalf("fleet bytes deduplicated = %d, want %d", dedupBytes, (agents-1)*(64<<20))
	}
	// The pooled total can never exceed the per-host-private baseline — the win is real,
	// not an artifact of the single synthetic prefix.
	if pooledPrefill > privatePrefill {
		t.Fatalf("aggregate pooled prefill %d exceeds the private baseline %d", pooledPrefill, privatePrefill)
	}
	// Honest dedup: the trusted shared cell is reusable across the fleet, but a poisoned
	// variant is refused — so none of the deduplicated copies is a poisoned alias.
	if v := PoolReuseVerdict(poolKVEntry("qwen3", abi.TaintTrusted, abi.ScopeFleet), wantKey("qwen3")); !v.CanServe() {
		t.Fatalf("the trusted shared cell must be reusable across the fleet, got %s/%s", v.Kind, v.Reason)
	}
	if v := PoolReuseVerdict(poolKVEntry("qwen3", abi.TaintQuarantined, abi.ScopeFleet), wantKey("qwen3")); v.CanServe() {
		t.Fatalf("a poisoned variant of the shared cell must be refused, never aliased")
	}
}
