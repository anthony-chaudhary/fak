//go:build cuda && cgo

package compute

import (
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
// The body uses synthetic durations; a full CUDA-tagged package invocation still
// runs fcuda_init. This is counter arithmetic, not CUDA or clock-overhead proof.
func TestCUDAQwen35SequenceLockProfileCounters(t *testing.T) {
	var got cudaQwen35SequenceLockCounters
	got.observe(2*time.Nanosecond, 7*time.Nanosecond)
	got.observe(0, 3*time.Nanosecond)
	if got.completed != 2 || got.waitNS != 2 || got.holdNS != 10 || got.invalid {
		t.Fatalf("completed sequence totals = %+v, want 2/2/10 valid", got)
	}
	for _, initial := range []cudaQwen35SequenceLockCounters{
		{completed: ^uint64(0)},
		{waitNS: ^uint64(0)},
		{holdNS: ^uint64(0)},
	} {
		got = initial
		got.observe(1, 1)
		if !got.invalid || got.completed != initial.completed || got.waitNS != initial.waitNS || got.holdNS != initial.holdNS {
			t.Fatalf("overflow changed totals or remained available: before=%+v after=%+v", initial, got)
		}
	}
	got = cudaQwen35SequenceLockCounters{}
	got.observe(-1, 1)
	got.observe(1, 1)
	if !got.invalid || got.completed != 0 || got.waitNS != 0 || got.holdNS != 0 {
		t.Fatalf("invalid duration must leave the observation unavailable: %+v", got)
	}
}
