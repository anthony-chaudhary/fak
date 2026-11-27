package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=200ms lane=default
func TestInkling2AsymmetricKVFitPlannerAndSharedBudget(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want Inkling2KVDType
	}{
		{"fp8k_nvfp4v", Inkling2KVAsym},
		{" k8v4 ", Inkling2KVAsym},
	} {
		got, err := Inkling2KVDTypeFrom(func(string) (string, bool) { return tc.raw, true })
		if err != nil || got != tc.want {
			t.Fatalf("Inkling2KVDTypeFrom(%q) = (%v, %v), want (%v, nil)", tc.raw, got, err, tc.want)
		}
	}
	if Inkling2KVAsym.String() != "fp8k_nvfp4v" {
		t.Fatalf("Inkling2KVAsym.String() = %q, want fp8k_nvfp4v", Inkling2KVAsym.String())
	}

	hp := inkling2MiniKVShape()
	// 8 global slots: K = 1024 B FP8 + 64 B block-16 E4M3FN scales = 1088 B/tok,
	// V = 512 B E2M1 + 64 B block-16 E4M3FN scales = 576 B/tok -> 1664 B/tok per slot.
	perGlobalToken := int64(8) * (1088 + 576)
	delta := inkling2ResidentKVBytesFor(hp, 23553, false, 0, Inkling2KVAsym) - inkling2ResidentKVBytesFor(hp, 23552, false, 0, Inkling2KVAsym)
	if want := perGlobalToken + 2*24*4; delta != want {
		t.Fatalf("marginal asym bytes per token = %d, want %d", delta, want)
	}
	if got := inkling2ResidentPerPosBytesFor(hp, Inkling2KVAsym); got != delta {
		t.Fatalf("perPos asym = %d, want %d", got, delta)
	}

	mini := hp
	mini.Hidden = 2304
	mini.ExpertsUsed = 8
	mini.ExpertWidth = 768
	const chunk = 128
	other := inkling2ResidentOtherBytes(mini, inkling2ArenaFloats, chunk, 0)
	asymKV128 := inkling2ResidentKVBytesFor(mini, 131072, true, chunk, Inkling2KVAsym)
	fp8KV128 := inkling2ResidentKVBytesFor(mini, 131072, true, chunk, Inkling2KVFP8)
	// Asymmetric saves 448 B/tok per global slot vs FP8 across 8 global slots = 3584 B/tok.
	if saved := fp8KV128 - asymKV128; saved != int64(8)*448*131072 {
		t.Fatalf("fp8-asym saved %d bytes at 128k, want %d", saved, int64(8)*448*131072)
	}

	// With ~2.65 GiB free after weights (above 2.204 GiB session total + headroom),
	// 128k fp8k_nvfp4v (chunk=128) fits 24/24 all-resident while symmetric fp8 does not.
	const freeAfterWeights128k = int64(265) * (1 << 30) / 100
	if _, err := inkling2KVFitFor(mini, 131072, true, chunk, Inkling2KVAsym, other, freeAfterWeights128k); err != nil {
		t.Fatalf("128k fp8k_nvfp4v (chunk=%d) did not fit 24/24 all-resident within 2.65 GiB free VRAM: %v", chunk, err)
	}
	if _, err := inkling2KVFitFor(mini, 131072, true, chunk, Inkling2KVFP8, other, freeAfterWeights128k); err == nil {
		t.Fatal("128k fp8 unexpectedly fit within 2.65 GiB free VRAM")
	}
	t.Logf("ctx=131072 asym_kv=%.3fGiB other_t128=%.3fGiB asym_total=%.3fGiB (vs fp8_total=%.3fGiB)",
		float64(asymKV128)/(1<<30), float64(other)/(1<<30),
		float64(asymKV128+other)/(1<<30), float64(fp8KV128+other)/(1<<30))

	// Verify shared budget helper: inkling2ResolveAutoCtxFor and inkling2KVFitFor never drift.
	be := fakeAsymMemBackend{free: freeAfterWeights128k}
	autoCtx, err := inkling2ResolveAutoCtxFor(be, mini, true, chunk, other, Inkling2KVAsym)
	if err != nil {
		t.Fatalf("inkling2ResolveAutoCtxFor: %v", err)
	}
	if autoCtx < 32768 {
		t.Fatalf("autoCtx = %d, want >= 32768", autoCtx)
	}
	if _, err := inkling2FitResidentKVFor(be, mini, autoCtx, true, chunk, Inkling2KVAsym, other); err != nil {
		t.Fatalf("autoCtx=%d drifted from inkling2FitResidentKVFor: %v", autoCtx, err)
	}
}

type fakeAsymMemBackend struct {
	compute.Backend
	free int64
}

func (fakeAsymMemBackend) Name() string { return "cuda" }
func (fakeAsymMemBackend) Caps() compute.Caps {
	return compute.Caps{CapacityProbe: true}
}
func (b fakeAsymMemBackend) DeviceMemory() (total, free int64, known bool) {
	return 16 * (1 << 30), b.free, true
}
