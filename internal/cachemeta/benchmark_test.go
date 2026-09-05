package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

var (
	benchScoreSink      PrefixStabilityScore
	benchDivergeSink    TurnDivergence
	benchPlacementSink  PlacementDecision
	benchFleetReuseSink FleetReuseResult
	benchLookupSink     LookupVerdict
	benchEntrySink      Entry
)

func BenchmarkPrefixScore(b *testing.B) {
	tr := NewPrefixStabilityTracker("5m", abi.ScopeAgent)
	tr.Tokens = 300
	tr.SizeBytes = 4096
	tr.PerTokenPrefillNanos = 2_000_000

	baseSpan := []PromptSegment{
		seg(SegStable, 128, "You are an autonomous agent kernel operating on trunk."),
		seg(SegToolSchema, 256, `{"tools":[{"name":"read_file"},{"name":"edit_file"},{"name":"bash"}]}`),
	}
	tr.Observe(baseSpan)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchScoreSink = tr.Observe(baseSpan)
	}
}

func BenchmarkPrefixScore_Mutated(b *testing.B) {
	tr := NewPrefixStabilityTracker("5m", abi.ScopeAgent)
	tr.Tokens = 300
	tr.SizeBytes = 4096
	tr.PerTokenPrefillNanos = 2_000_000

	baseSpan := []PromptSegment{
		seg(SegStable, 128, "You are an autonomous agent kernel operating on trunk."),
		seg(SegToolSchema, 256, `{"tools":[{"name":"read_file"},{"name":"edit_file"},{"name":"bash"}]}`),
	}
	mutatedSpan := []PromptSegment{
		seg(SegStable, 128, "You are an autonomous agent kernel operating on trunk."),
		seg(SegToolSchema, 280, `{"tools":[{"name":"read_file"},{"name":"edit_file"},{"name":"bash"},{"name":"glob"}]}`),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tr.Reset()
		tr.Observe(baseSpan)
		benchScoreSink = tr.Observe(mutatedSpan)
	}
}

func BenchmarkDiverge(b *testing.B) {
	prev := []PromptSegment{
		seg(SegStable, 150, "You are a coding agent. Follow project conventions rigorously."),
		seg(SegToolSchema, 300, `{"tools":[{"name":"read"},{"name":"write"},{"name":"grep"},{"name":"glob"}]}`),
		seg(SegMessage, 50, "Fix the bug in internal/cachemeta/placement.go"),
		seg(SegToolResult, 80, `{"status":"ok","content":"package cachemeta..."}`),
	}
	next := []PromptSegment{
		prev[0],
		prev[1],
		prev[2],
		prev[3],
		seg(SegMessage, 40, "Run the package tests now."),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDivergeSink = Diverge(prev, next)
	}
}

func BenchmarkPlanPlacement(b *testing.B) {
	profiles := DefaultTierProfiles()
	lc := NewLifecycle(TierHBM, 0).MarkResident(profiles, 0)
	req := PlacementRequest{
		Lifecycle:            lc,
		SizeBytes:            64 << 20,
		Tokens:               4000,
		Profiles:             profiles,
		Pressure:             TierPressure{TierHBM: 1.0, TierDRAM: 0.5},
		Policy:               LifecyclePolicy{DemoteOnExpiry: true},
		PerTokenPrefillNanos: 2_000_000,
		NowMillis:            1000,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchPlacementSink = PlanPlacement(req)
	}
}

func BenchmarkPlanFleetReuse(b *testing.B) {
	profiles := DefaultTierProfiles()
	pools := DefaultPoolProfiles()
	req := FleetReuseRequest{
		Tenants:              8,
		Tokens:               4000,
		SizeBytes:            64 << 20,
		PerTokenPrefillNanos: 2_000_000,
		Profile:              profiles[TierCXL],
		Pool:                 pools[TierCXL],
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFleetReuseSink = PlanFleetReuse(req)
	}
}

func BenchmarkPoolReuseVerdict(b *testing.B) {
	e := FromKVPrefix(KVPrefix{
		TokenDigest: "deadbeefcafebabe",
		Length:      4000,
		ModelID:     "qwen-3.8-7b",
		TokenizerID: "qwen-tok",
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
	e.Security.Taint = abi.TaintTrusted
	e.Security.Scope = abi.ScopeFleet
	e.Security.AdmittedBy = "adjudicator"
	e.Security.AdmissionVerdict = AdmissionAllow

	want := MaterializationKey{
		ModelID:         "qwen-3.8-7b",
		TokenizerID:     "qwen-tok",
		SerializerID:    "ser-1",
		PositionRegime:  "rope-theta-1e6",
		PolicyVersion:   "pol-1",
		AdmitterVersion: "adj-1",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchLookupSink = PoolReuseVerdict(e, want)
	}
}

func BenchmarkAttentionIndexLookup(b *testing.B) {
	candidate := AttentionIndex{
		PrefixDigest:      "sha256:abc123def456",
		Tokens:            []int{101, 2048, 3096, 4120},
		PrefixLength:      4,
		ModelID:           "qwen-3.8-7b",
		TokenizerID:       "qwen-tok",
		PositionMode:      PositionPrefixAligned,
		IndexerID:         "dsa-v1",
		LayerGroup:        "attn-group-0",
		Layers:            []int{0, 1, 2, 3},
		DecisionDigest:    "sha256:dec999",
		Causal:            true,
		QualityDeltaProbe: 0.005,
	}
	req := AttentionIndexRequest{
		PrefixDigest:    "sha256:abc123def456",
		Tokens:          []int{101, 2048, 3096, 4120},
		PrefixLength:    4,
		ModelID:         "qwen-3.8-7b",
		TokenizerID:     "qwen-tok",
		PositionMode:    PositionPrefixAligned,
		IndexerID:       "dsa-v1",
		LayerGroup:      "attn-group-0",
		DecisionDigest:  "sha256:dec999",
		MaxQualityDelta: 0.01,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchLookupSink = AttentionIndexLookup(req, candidate)
	}
}

func BenchmarkFromKVPrefix(b *testing.B) {
	p := KVPrefix{
		TokenDigest: "deadbeefcafebabe",
		Length:      2048,
		ModelID:     "qwen-3.8-7b",
		TokenizerID: "qwen-tok",
		Owner:       "tenant-0",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchEntrySink = FromKVPrefix(p,
			WithSerializer("msgpack"),
			WithPolicyVersion("pol-2026"),
			WithResidency(TierHBM, "tenant-0", "lease-1"),
			WithWitness("git:c0ffee"),
		)
	}
}
