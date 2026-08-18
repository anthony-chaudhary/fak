package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

type fakeNativePrefixSource struct {
	resident int64
	cands    []agent.KVPrefixPressureCandidate
	stages   int
	evicted  string
}

func (f *fakeNativePrefixSource) KVPrefixPressuredCandidates() (int64, []agent.KVPrefixPressureCandidate) {
	return f.resident, f.cands
}

func (f *fakeNativePrefixSource) StageKVPrefixToHost(_ context.Context, digest string) agent.KVPrefixTransfer {
	f.stages++
	return agent.KVPrefixTransfer{
		Outcome:    "ok",
		SpanDigest: digest,
		Positions:  f.cands[0].Tokens,
		BytesMoved: 8192,
	}
}

func (f *fakeNativePrefixSource) RestoreKVPrefixFromHost(_ context.Context, digest string) agent.KVPrefixTransfer {
	return agent.KVPrefixTransfer{Outcome: "miss", SpanDigest: digest}
}

func (f *fakeNativePrefixSource) EvictHotKVPrefix(digest string) int {
	if f.stages == 0 {
		return 0 // match radixkv: never drop the sole complete prefix copy.
	}
	f.evicted = digest
	return f.cands[0].Tokens
}

func TestNativePrefixPressureBridgeStagesOnlyToHostDRAM(t *testing.T) {
	source := &fakeNativePrefixSource{
		resident: 16 << 20,
		cands: []agent.KVPrefixPressureCandidate{{
			SpanDigest: "native-prefix",
			Tokens:     128,
			SizeBytes:  16 << 20,
			ModelID:    "native-model",
		}},
	}
	bridge := newInKernelPrefixPressureBridge(source)
	resident, candidates := bridge.PressuredCandidates()
	if resident != source.resident || len(candidates) != 1 || candidates[0].SpanDigest != "native-prefix" {
		t.Fatalf("provider projection resident=%d candidates=%+v", resident, candidates)
	}
	if candidates[0].PerTokenPrefillNanos <= 0 {
		t.Fatalf("native candidate retained an unknown/free recompute cost: %+v", candidates[0])
	}
	profiles := cachemeta.DefaultTierProfiles()
	decision := cachemeta.PlanPlacement(cachemeta.PlacementRequest{
		Lifecycle: cachemeta.NewLifecycle(cachemeta.TierHBM, 0).MarkResident(profiles, 0),
		SizeBytes: candidates[0].SizeBytes,
		Tokens:    int64(candidates[0].Tokens),
		Profiles:  profiles,
		Pressure: cachemeta.TierPressure{
			cachemeta.TierHBM: 1,
		},
		Policy:               cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
		PerTokenPrefillNanos: candidates[0].PerTokenPrefillNanos,
	})
	if decision.Action != cachemeta.ActionDemote || decision.ToTier != cachemeta.TierDRAM {
		t.Fatalf("native pressure candidate planned an unexecutable move %s->%s (%s)",
			decision.Action, decision.ToTier, decision.Reason)
	}
	adp := &engine.CapacityAdapter{KV: bridge}
	res, err := adp.Execute(context.Background(), engine.PlacementMove{
		Decision:   decision,
		SpanDigest: "native-prefix", N: 128,
	})
	if err != nil || !res.Applied || res.Evicted != 128 {
		t.Fatalf("DRAM demote result=%+v err=%v", res, err)
	}
	if source.stages != 1 || source.evicted != "native-prefix" {
		t.Fatalf("physical stage/evict stages=%d digest=%q", source.stages, source.evicted)
	}

	source.evicted = ""
	res, err = adp.Execute(context.Background(), engine.PlacementMove{
		Decision: cachemeta.PlacementDecision{
			Action: cachemeta.ActionSpill, FromTier: cachemeta.TierHBM, ToTier: cachemeta.TierDisk,
			Directive: cachemeta.KVOffload, EstMoveBytes: 1,
		},
		SpanDigest: "native-prefix", N: 128,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Applied || source.stages != 1 || source.evicted != "" {
		t.Fatalf("non-DRAM placement was falsely applied: result=%+v stages=%d evicted=%q", res, source.stages, source.evicted)
	}
}
