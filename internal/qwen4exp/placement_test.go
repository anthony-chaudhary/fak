package qwen4exp

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestAdmitPlacementAcceptsExplicitGPUCPUAndSSDModes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		tiers [2]PlacementTier
	}{
		{name: "gpu", tiers: [2]PlacementTier{TierGPU, TierGPU}},
		{name: "cpu", tiers: [2]PlacementTier{TierCPU, TierCPU}},
		{name: "ssd", tiers: [2]PlacementTier{TierSSD, TierSSD}},
		{name: "explicit split", tiers: [2]PlacementTier{TierGPU, TierSSD}},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := validPlacementRequest(tc.tiers[0], tc.tiers[1])
			receipt, err := AdmitPlacement(req)
			if err != nil {
				t.Fatalf("AdmitPlacement() error = %v", err)
			}
			if receipt.Digest() == "" || !bytes.Contains(receipt.JSON(), []byte(`"engine":"fak-native"`)) {
				t.Fatalf("receipt missing identity: %s", receipt.JSON())
			}
			got := receipt.Components()
			if len(got) != 2 {
				t.Fatalf("component count = %d, want 2", len(got))
			}
			for _, component := range got {
				if component.Tier == "" || component.LogicalBytes == 0 || component.PhysicalBytes == 0 || component.PageBehavior == "" || component.CacheBehavior == "" || component.LoadLatency == 0 || component.LookupLatency == 0 || component.PeakMemoryBytes == 0 || !component.QualityEquivalent || component.Matched.PromptSetDigest == "" || component.RecoveryLatency == 0 {
					t.Fatalf("receipt omitted required evidence: %+v", component)
				}
			}
		})
	}
}

func TestAdmitPlacementRejectsUnsafeOrUnmeasuredRequests(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		edit func(*PlacementRequest)
		want string
	}{
		{name: "foreign engine", edit: func(r *PlacementRequest) { r.Engine = "llama.cpp" }, want: "only \"fak-native\""},
		{name: "undeclared tier", edit: func(r *PlacementRequest) { r.Capacities = r.Capacities[:2] }, want: "no declared capacity"},
		{name: "capacity", edit: func(r *PlacementRequest) { r.Capacities[2].CapacityBytes = 10 }, want: "exceed declared capacity"},
		{name: "traffic", edit: func(r *PlacementRequest) { r.Components[0].Traffic.MeasuredSustainableLookupsRate = 99 }, want: "exceeds measured sustainable rate"},
		{name: "missing traffic", edit: func(r *PlacementRequest) { r.Components[0].Traffic.Window = 0 }, want: "measured traffic evidence"},
		{name: "silent fallback", edit: func(r *PlacementRequest) { r.Components[0].CacheBehavior = "host fallback" }, want: "without fallback"},
		{name: "mmap claim", edit: func(r *PlacementRequest) { r.Components[0].PageBehavior = "mmap demand paging" }, want: "non-mmap"},
		{name: "quality regression", edit: func(r *PlacementRequest) { r.Components[0].PlacedQuality = .8 }, want: "quality delta"},
		{name: "missing comparison", edit: func(r *PlacementRequest) { r.Components[0].Matched.PromptSetDigest = "" }, want: "matched in-memory/offload"},
		{name: "missing recovery cost", edit: func(r *PlacementRequest) { r.Components[0].RecoveryLatency = 0 }, want: "failure and recovery cost"},
		{name: "missing component", edit: func(r *PlacementRequest) { r.Components = r.Components[:1] }, want: "both be declared"},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := validPlacementRequest(TierGPU, TierSSD)
			tc.edit(&req)
			_, err := AdmitPlacement(req)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("AdmitPlacement() error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestPlacementReceiptIsDeterministicAndImmutable(t *testing.T) {
	t.Parallel()

	req := validPlacementRequest(TierSSD, TierGPU)
	first, err := AdmitPlacement(req)
	if err != nil {
		t.Fatal(err)
	}
	req.Capacities[0], req.Capacities[2] = req.Capacities[2], req.Capacities[0]
	req.Components[0], req.Components[1] = req.Components[1], req.Components[0]
	second, err := AdmitPlacement(req)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest() != second.Digest() || !bytes.Equal(first.JSON(), second.JSON()) {
		t.Fatalf("equivalent inputs produced different receipts:\n%s\n%s", first.JSON(), second.JSON())
	}

	encoded := first.JSON()
	encoded[0] = '!'
	components := first.Components()
	components[0].Tier = TierCPU
	if first.JSON()[0] == '!' || first.Components()[0].Tier == TierCPU {
		t.Fatal("receipt accessors exposed mutable receipt state")
	}
}

func validPlacementRequest(pleTier, sparseTier PlacementTier) PlacementRequest {
	return PlacementRequest{
		Engine: FakNativeEngine,
		Capacities: []TierCapacity{
			{Tier: TierGPU, CapacityBytes: 8 << 30},
			{Tier: TierCPU, CapacityBytes: 32 << 30},
			{Tier: TierSSD, CapacityBytes: 128 << 30},
		},
		Components: []PlacementEvidence{
			measuredComponent(NGram3PLEEmbeddings, pleTier, 1<<30),
			measuredComponent(SparseAttentionIndex, sparseTier, 2<<30),
		},
	}
}

func measuredComponent(component ManagedComponent, tier PlacementTier, size uint64) PlacementEvidence {
	return PlacementEvidence{
		Component:         component,
		Tier:              tier,
		LogicalBytes:      size,
		PhysicalBytes:     size / 2,
		PeakMemoryBytes:   size / 4,
		PageBehavior:      "explicit 64KiB reads; faults rejected",
		CacheBehavior:     "fixed 256MiB clock cache; misses read declared tier",
		LoadLatency:       12 * time.Millisecond,
		LookupLatency:     80 * time.Microsecond,
		QualityMetric:     "matched-token-agreement",
		InMemoryQuality:   .999,
		PlacedQuality:     .999,
		QualityEquivalent: true,
		QualityTolerance:  .001,
		Traffic: TrafficEvidence{
			Window:                         time.Minute,
			ObservedLookups:                60_000,
			PeakLookupsPerSecond:           1_500,
			MeasuredSustainableLookupsRate: 2_000,
		},
		Matched: MatchedComparison{
			PromptSetDigest:       "sha256:matched-prompts",
			InMemoryTier:          TierGPU,
			InMemoryLookupLatency: 20 * time.Microsecond,
			OffloadLookupLatency:  80 * time.Microsecond,
			InMemoryPeakBytes:     size,
			OffloadPeakBytes:      size / 4,
		},
		FailureDetectionLatency: 2 * time.Millisecond,
		RecoveryLatency:         25 * time.Millisecond,
		RecoveryReadBytes:       size / 8,
	}
}
