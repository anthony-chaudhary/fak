package main

// kvmmu_pressure_bridge_test.go — exercises the host half of the #1073 capacity wire: the sweeper
// closure newCapacityPressureSweeper builds + the lowerPressureCandidates lowering. It proves the
// bridge runs the REAL engine.RunCapacityPressureSweep over a fake device backend at high pressure
// and demotes the lowered candidate (StageSpan → Evict), so the cmd/fak glue the gateway injects
// is itself witnessed, not just defined.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

type bridgeFakeBackend struct {
	compute.Backend
	total, free int64
}

func (f bridgeFakeBackend) Caps() compute.Caps {
	return compute.Caps{Async: true, DeviceMemory: true, CapacityProbe: true}
}
func (f bridgeFakeBackend) DeviceMemory() (int64, int64, bool) { return f.total, f.free, true }

type bridgeFakeKV struct {
	stageCalls int
	evicts     []struct{ from, n int }
}

func (f *bridgeFakeKV) Len() int                    { return 4096 }
func (f *bridgeFakeKV) Prefill(ids []int) []float32 { return nil }
func (f *bridgeFakeKV) ModelID() string             { return "bridge-model" }
func (f *bridgeFakeKV) Evict(from, n int) int {
	f.evicts = append(f.evicts, struct{ from, n int }{from, n})
	return n
}
func (f *bridgeFakeKV) StageSpan(_ context.Context, digest string, _, n int) (abi.KVResidency, error) {
	f.stageCalls++
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: n}, nil
}
func (f *bridgeFakeKV) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest}, nil
}

func TestNewCapacityPressureSweeperDemotes(t *testing.T) {
	const total = 100 << 20
	kv := &bridgeFakeKV{}
	rec := engine.NewCacheEventRecorder()
	sweeper := newCapacityPressureSweeper(
		bridgeFakeBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown},
		&engine.CapacityAdapter{KV: kv, Recorder: rec},
	)
	relief := sweeper(context.Background(), 90<<20, 0.80, []gateway.KVPressureCandidate{{
		SpanDigest:           "span-bridge",
		From:                 16,
		N:                    4000,
		ModelID:              "bridge-model",
		SizeBytes:            64 << 20,
		Tokens:               4000,
		PerTokenPrefillNanos: 2_000_000,
	}})
	if !relief.Known || relief.AppliedMoves != 1 {
		t.Fatalf("bridge sweeper did not apply one demote: %+v", relief)
	}
	if relief.FinalPressure >= 0.80 {
		t.Fatalf("bridge sweeper did not relieve pressure below target: %+v", relief)
	}
	if kv.stageCalls != 1 || len(kv.evicts) != 1 || kv.evicts[0].from != 16 || kv.evicts[0].n != 4000 {
		t.Fatalf("bridge did not stage-then-evict the lowered span: stage=%d evicts=%+v", kv.stageCalls, kv.evicts)
	}
}

// TestNewCapacityPressureSweeperNilFailsOpen proves the bridge sweeper is fail-open: a nil backend
// or adapter, or an empty candidate list, reports no relief rather than panicking.
func TestNewCapacityPressureSweeperNilFailsOpen(t *testing.T) {
	nilBackend := newCapacityPressureSweeper(nil, &engine.CapacityAdapter{KV: &bridgeFakeKV{}})
	if r := nilBackend(context.Background(), 1, 0.8, []gateway.KVPressureCandidate{{SpanDigest: "x"}}); r.AppliedMoves != 0 || r.Known {
		t.Fatalf("nil backend should fail open, got %+v", r)
	}
	live := newCapacityPressureSweeper(
		bridgeFakeBackend{Backend: compute.Default(), total: 100 << 20, free: compute.FreeUnknown},
		&engine.CapacityAdapter{KV: &bridgeFakeKV{}},
	)
	if r := live(context.Background(), 90<<20, 0.8, nil); r.AppliedMoves != 0 {
		t.Fatalf("empty candidate list should be a no-op, got %+v", r)
	}
}
