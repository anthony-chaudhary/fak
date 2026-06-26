package engine

import (
	"context"
	"errors"
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
