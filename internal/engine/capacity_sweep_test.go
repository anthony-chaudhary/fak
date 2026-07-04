package engine

import (
	"context"
	"errors"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

type sweepFakeKV struct {
	len        int
	stageOut   abi.KVResidencyOutcome
	stageErr   error
	stageCalls int
	evicts     []struct{ from, n int }
}

func (f *sweepFakeKV) Len() int                    { return f.len }
func (f *sweepFakeKV) Prefill(ids []int) []float32 { return nil }
func (f *sweepFakeKV) ModelID() string             { return "sweep-model" }
func (f *sweepFakeKV) Evict(from, n int) int {
	f.evicts = append(f.evicts, struct{ from, n int }{from, n})
	return n
}
func (f *sweepFakeKV) StageSpan(_ context.Context, digest string, _, n int) (abi.KVResidency, error) {
	f.stageCalls++
	if f.stageErr != nil {
		return abi.KVResidency{}, f.stageErr
	}
	return abi.KVResidency{Outcome: f.stageOut, Digest: digest, Positions: n}, nil
}
func (f *sweepFakeKV) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest}, nil
}

func TestCapacityPressureSweepDemotesUntilBelowTarget(t *testing.T) {
	const total = 100 << 20
	req := expensivePrefixRequest()
	kv := &sweepFakeKV{len: 4096, stageOut: abi.KVResidencyOK}
	rec := NewCacheEventRecorder()
	res, err := RunCapacityPressureSweep(context.Background(), CapacityPressureSweep{
		Backend:        fakeCapBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown, probe: true},
		Adapter:        &CapacityAdapter{KV: kv, Recorder: rec},
		ResidentBytes:  90 << 20,
		TargetPressure: 0.80,
		Candidates: []CapacityPressureCandidate{{
			Request: req,
			Move: PlacementMove{
				SpanDigest:   "span-pressure",
				From:         32,
				N:            4000,
				ModelID:      "sweep-model",
				TokenizerID:  "tok",
				PositionMode: cachemeta.PositionPrefixAligned,
				Owner:        "capacity-sweep",
			},
		}},
	})
	if err != nil {
		t.Fatalf("RunCapacityPressureSweep: %v", err)
	}
	if !res.Known || res.CapacityBytes != total {
		t.Fatalf("capacity not reported in result: %+v", res)
	}
	if res.AppliedMoves != 1 || res.ReclaimedBytes != req.SizeBytes {
		t.Fatalf("sweep did not apply exactly one reclaiming move: %+v", res)
	}
	if res.InitialPressure < 0.89 || res.InitialPressure > 0.91 || res.FinalPressure >= 0.80 {
		t.Fatalf("pressure not relieved as expected: initial=%v final=%v", res.InitialPressure, res.FinalPressure)
	}
	if len(res.Moves) != 1 || res.Moves[0].Decision.Action != cachemeta.ActionDemote || res.Moves[0].Decision.ToTier != cachemeta.TierDRAM {
		t.Fatalf("want HBM->DRAM demote decision, got %+v", res.Moves)
	}
	if kv.stageCalls != 1 || len(kv.evicts) != 1 || kv.evicts[0].from != 32 || kv.evicts[0].n != 4000 {
		t.Fatalf("sweep did not stage then evict the live span: stage=%d evicts=%+v", kv.stageCalls, kv.evicts)
	}
	if rows := rec.Metrics().Snapshot().Rows; len(rows) == 0 || rows[0].MemoryClass != string(compute.MemoryDDRCache) {
		t.Fatalf("demote should be visible as ddr_cache cache-event row, got %+v", rows)
	}
}

func TestCapacityPressureSweepUnknownCapacityFailsOpen(t *testing.T) {
	res, err := RunCapacityPressureSweep(context.Background(), CapacityPressureSweep{
		Backend:       fakeCapBackend{Backend: compute.Default(), total: 100 << 20, free: 0, probe: false},
		ResidentBytes: 100 << 20,
		Candidates: []CapacityPressureCandidate{{
			Request: expensivePrefixRequest(),
		}},
	})
	if err != nil {
		t.Fatalf("unknown capacity must fail open, got error %v", err)
	}
	if res.Known || len(res.Moves) != 0 || res.AppliedMoves != 0 {
		t.Fatalf("unknown capacity must not move anything: %+v", res)
	}
}

func TestCapacityPressureSweepStageFaultRetainsLiveSpan(t *testing.T) {
	const total = 100 << 20
	kv := &sweepFakeKV{len: 4096, stageOut: abi.KVResidencyOK, stageErr: errors.New("dram stage timeout")}
	rec := NewCacheEventRecorder()
	res, err := RunCapacityPressureSweep(context.Background(), CapacityPressureSweep{
		Backend:        fakeCapBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown, probe: true},
		Adapter:        &CapacityAdapter{KV: kv, Recorder: rec},
		ResidentBytes:  90 << 20,
		TargetPressure: 0.80,
		Candidates: []CapacityPressureCandidate{{
			Request: expensivePrefixRequest(),
			Move: PlacementMove{
				SpanDigest: "span-fault",
				From:       7,
				N:          9,
			},
		}},
	})
	if err != nil {
		t.Fatalf("staging fault should be a typed result, not a sweep error: %v", err)
	}
	if res.AppliedMoves != 0 || res.Faults != 1 || res.ReclaimedBytes != 0 {
		t.Fatalf("faulted stage must not reclaim live bytes: %+v", res)
	}
	if len(kv.evicts) != 0 {
		t.Fatalf("faulted stage must retain live span, evicts=%+v", kv.evicts)
	}
	if len(res.Moves) != 1 || res.Moves[0].Result.Recorded.Verdict.Kind != cachemeta.LookupFault {
		t.Fatalf("fault should be recorded as lookup fault, got %+v", res.Moves)
	}
}

// TestCapacityPressureSweepHostDRAMAwareTarget proves the #1073 host-DRAM half of the wire: the
// live sweep plans the demote TARGET against host-DRAM fullness, not just device HBM. With the
// host-DRAM probe absent (DRAMKnown=false, the fail-open default) an HBM-pressured span demotes to
// DRAM — the refute guard, identical to the device-only sweep. With the SAME span and a FULL host
// DRAM folded in (DRAMKnown, DRAMPressure 1.0) the demote target moves one tier colder (NUMA-far)
// instead of staging the span into a tier with no room, and the move STILL applies (StageSpan then
// Evict) — demoted, never dropped. This drives TestDRAMPressureFlipsDemoteTarget's flip through the
// live RunCapacityPressureSweep executor instead of the bare planner.
func TestCapacityPressureSweepHostDRAMAwareTarget(t *testing.T) {
	const total = 100 << 20
	backend := fakeCapBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown, probe: true}
	candidate := func() CapacityPressureCandidate {
		return CapacityPressureCandidate{
			Request: expensivePrefixRequest(),
			Move: PlacementMove{
				SpanDigest:   "span-dram-aware",
				From:         32,
				N:            4000,
				ModelID:      "sweep-model",
				PositionMode: cachemeta.PositionPrefixAligned,
				Owner:        "capacity-sweep",
			},
		}
	}

	// Refute guard: no host-DRAM probe (DRAMKnown=false) -> the demote target is DRAM, exactly as
	// the device-only sweep. If this ever fails the flip below proves nothing.
	kv := &sweepFakeKV{len: 4096, stageOut: abi.KVResidencyOK}
	res, err := RunCapacityPressureSweep(context.Background(), CapacityPressureSweep{
		Backend:        backend,
		Adapter:        &CapacityAdapter{KV: kv, Recorder: NewCacheEventRecorder()},
		ResidentBytes:  90 << 20,
		TargetPressure: 0.80,
		Candidates:     []CapacityPressureCandidate{candidate()},
	})
	if err != nil {
		t.Fatalf("device-only sweep: %v", err)
	}
	if res.AppliedMoves != 1 || len(res.Moves) != 1 || res.Moves[0].Decision.ToTier != cachemeta.TierDRAM {
		t.Fatalf("refute guard: without a host-DRAM probe the demote target must be DRAM, got %+v", res.Moves)
	}

	// With a FULL host DRAM folded in, the same span demotes one tier colder (NUMA-far), never into
	// the full DRAM tier — and the move still applies (staged then evicted).
	kv = &sweepFakeKV{len: 4096, stageOut: abi.KVResidencyOK}
	res, err = RunCapacityPressureSweep(context.Background(), CapacityPressureSweep{
		Backend:        backend,
		Adapter:        &CapacityAdapter{KV: kv, Recorder: NewCacheEventRecorder()},
		ResidentBytes:  90 << 20,
		TargetPressure: 0.80,
		DRAMPressure:   1.0,
		DRAMKnown:      true,
		Candidates:     []CapacityPressureCandidate{candidate()},
	})
	if err != nil {
		t.Fatalf("host-DRAM-aware sweep: %v", err)
	}
	if res.AppliedMoves != 1 || len(res.Moves) != 1 {
		t.Fatalf("a full DRAM should still demote the span one tier colder, got %+v", res)
	}
	if res.Moves[0].Decision.ToTier == cachemeta.TierDRAM {
		t.Fatal("host DRAM full did not move the demote target off DRAM — the sweep ignored HostDRAMPressure")
	}
	if res.Moves[0].Decision.ToTier != cachemeta.TierNUMAFar {
		t.Fatalf("a full DRAM should demote to the next colder attendable tier (NUMA-far), got %s", res.Moves[0].Decision.ToTier)
	}
	if kv.stageCalls != 1 || len(kv.evicts) != 1 {
		t.Fatalf("the colder-tier demote must still stage then evict (demote-not-drop), stage=%d evicts=%+v", kv.stageCalls, kv.evicts)
	}
}

// TestCapacityPressureSweepColderTierBudgetAndVictimOrder is the #1473 acceptance: under a
// pressure spike with a colder tier sized for K<N spans, the sweep demotes EXACTLY the K
// highest-value-to-relocate spans and evicts the rest, never overfilling the colder tier.
//
// The box has only HBM + DRAM, so when DRAM is full the demote ladder bottoms out and an
// overflow span evicts (recompute on demand) — the clean "K demote, rest evict" shape. DRAM
// is sized for exactly K=2 spans. Four spans are all individually cheaper to retain in DRAM
// than to recompute (each would demote on its own), but carry DISTINCT recompute values in
// ENUMERATION order low->high. A value-blind, budget-blind enumeration sweep (the prior
// behavior) would demote the first two (low value) AND — seeing an empty DRAM on every
// independent plan — demote all four, overfilling the tier. The victim order + per-tier
// budget instead demote the two HIGHEST-value spans and hold demoted bytes to the budget.
func TestCapacityPressureSweepColderTierBudgetAndVictimOrder(t *testing.T) {
	const (
		total = 1000 << 20 // 1000 MiB device
		span  = 20 << 20   // 20 MiB per span
		k     = 2          // DRAM holds exactly two spans
	)

	defaults := cachemeta.DefaultTierProfiles()
	dram := defaults[cachemeta.TierDRAM]
	dram.CapacityBytes = k * span // size the colder tier for exactly K spans
	// A box with only the top two local tiers: no NUMA-far/CXL/disk rung to fall to, so an
	// overflow demote has nowhere colder and evicts instead.
	profiles := map[cachemeta.ResidencyTier]cachemeta.TierProfile{
		cachemeta.TierHBM:  defaults[cachemeta.TierHBM],
		cachemeta.TierDRAM: dram,
	}

	mk := func(digest string, tokens int64) CapacityPressureCandidate {
		return CapacityPressureCandidate{
			Request: cachemeta.PlacementRequest{
				Lifecycle:            cachemeta.NewLifecycle(cachemeta.TierHBM, 0).MarkResident(profiles, 0),
				SizeBytes:            span,
				Tokens:               tokens,
				Profiles:             profiles,
				Pressure:             cachemeta.TierPressure{},
				Policy:               cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
				PerTokenPrefillNanos: 2_000_000,
				NowMillis:            0,
			},
			Move: PlacementMove{
				SpanDigest: digest, From: 0, N: int(tokens),
				ModelID: "sweep-model", PositionMode: cachemeta.PositionPrefixAligned,
				Owner: "capacity-sweep",
			},
		}
	}
	// Enumeration order is low->high value; the sweep must NOT reward that order.
	cands := []CapacityPressureCandidate{
		mk("low-1", 2000),
		mk("low-2", 2100),
		mk("high-1", 4000),
		mk("high-2", 4100),
	}

	kv := &sweepFakeKV{len: 1 << 20, stageOut: abi.KVResidencyOK}
	res, err := RunCapacityPressureSweep(context.Background(), CapacityPressureSweep{
		Backend:        fakeCapBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown, probe: true},
		Adapter:        &CapacityAdapter{KV: kv, Recorder: NewCacheEventRecorder()},
		ResidentBytes:  870 << 20, // pressure 0.87 -> relief needs every candidate moved
		TargetPressure: 0.80,
		Candidates:     cands,
	})
	if err != nil {
		t.Fatalf("RunCapacityPressureSweep: %v", err)
	}

	var demoted, evicted []string
	var demotedBytes int64
	for _, m := range res.Moves {
		dg := cands[m.Index].Move.SpanDigest
		switch m.Decision.Action {
		case cachemeta.ActionDemote:
			if m.Decision.ToTier != cachemeta.TierDRAM {
				t.Fatalf("demote target should be DRAM, got %s for %s", m.Decision.ToTier, dg)
			}
			demoted = append(demoted, dg)
			demotedBytes += m.Decision.EstMoveBytes
		case cachemeta.ActionEvict:
			evicted = append(evicted, dg)
		default:
			t.Fatalf("unexpected action %s for %s", m.Decision.Action, dg)
		}
	}

	// Budget: exactly K demote into DRAM and their staged bytes never exceed the tier's
	// capacity budget. The refute guard — a budget-blind sweep demotes all four (4*span >
	// budget) — is this assertion failing.
	if len(demoted) != k {
		t.Fatalf("want exactly %d demotes under the DRAM budget, got %d (%v)", k, len(demoted), demoted)
	}
	if demotedBytes > k*span {
		t.Fatalf("colder tier overfilled: demoted %d bytes > %d budget", demotedBytes, int64(k*span))
	}
	// Order: the two demoted spans are the two HIGHEST recompute-value spans, not the first
	// two in enumeration order.
	sort.Strings(demoted)
	if demoted[0] != "high-1" || demoted[1] != "high-2" {
		t.Fatalf("victim order should demote the two highest-value spans, got demoted=%v evicted=%v", demoted, evicted)
	}
	// Relief: the K demotes plus the overflow evicts together drop HBM below target — a
	// relief pass, not a stall.
	if res.AppliedMoves != len(cands) || res.FinalPressure >= 0.80 {
		t.Fatalf("sweep should move every candidate and reach target: applied=%d final=%v", res.AppliedMoves, res.FinalPressure)
	}
}

func TestPlanPlacementForDeviceAtHighWater(t *testing.T) {
	const total = 100 << 20
	dev := fakeCapBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown, probe: true}
	req := expensivePrefixRequest()
	if d := PlanPlacementForDevice(dev, 85<<20, req); d.Action != cachemeta.ActionKeep {
		t.Fatalf("raw pressure below literal full should keep, got %s", d.Action)
	}
	if d := PlanPlacementForDeviceAtHighWater(dev, 85<<20, 0.80, req); d.Action != cachemeta.ActionDemote {
		t.Fatalf("pressure above high-water should demote, got %s", d.Action)
	}
	if d := PlanPlacementForDeviceAtHighWater(dev, 70<<20, 0.80, req); d.Action != cachemeta.ActionKeep {
		t.Fatalf("pressure below high-water should keep, got %s", d.Action)
	}
}
