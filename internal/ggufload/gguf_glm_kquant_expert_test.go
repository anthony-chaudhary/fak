package ggufload

import (
	"os"
	"path/filepath"
	"testing"
)

// TestGLMMoeDsaQ6KExpertsLoadResident is the loader-routing gate for the mixed-quant lever:
// a complete glm_moe_dsa GGUF whose batched routed experts are raw-residentable must load those
// experts RESIDENT (raw bytes -> kqw, no f32 dequant round-trip), not fall through to the f32 split.
// It asserts (a) all E*3 routed-expert tensors land in the resident k-quant store, (b) the
// forward still runs to finite logits over them, and (c) the load-path breakdown tallies the
// expert type as resident — the visibility that proves the slow path was avoided.
func TestGLMMoeDsaQ6KExpertsLoadResident(t *testing.T) {
	// Expert reduction dims (H, I) must be ÷256 for the resident k-quant row layout (each row
	// is whole super-blocks), mirroring the real GLM-5.2 constraint.
	const (
		H, V                = 256, 8
		qLora, kvLora       = 32, 32
		qkNope, qkRope, vHd = 16, 16, 16
		nH                  = 2
		idxHeads, idxDim    = 2, 16
		E, I, sharedI       = 3, 256, 256
	)
	for _, tc := range []struct {
		name string
		typ  TensorType
	}{{"Q6_K", TensorQ6_K}, {"Q5_K", TensorQ5_K}, {"IQ3_XXS", TensorIQ3_XXS}, {"IQ4_XS", TensorIQ4_XS}, {"Q8_0", TensorQ8_0}} {
		t.Run(tc.name, func(t *testing.T) {
			gguf := glmMoeDsaFullGGUFTyped(H, V, qLora, kvLora, qkNope, qkRope, vHd, nH, idxHeads, idxDim, E, I, sharedI, tc.typ)
			path := filepath.Join(t.TempDir(), "glm_kq.gguf")
			if err := os.WriteFile(path, gguf, 0o644); err != nil {
				t.Fatal(err)
			}

			prof := NewLoadProfiler()
			m, err := LoadModelQ4KProfile(path, prof)
			if err != nil {
				t.Fatalf("LoadModelQ4KProfile (%s experts): %v", tc.name, err)
			}

			// (a) all E*3 routed-expert blobs (gate/up/down) loaded resident, no f32 round-trip.
			if got := m.KQuantCount(); got != E*3 {
				t.Fatalf("resident k-quant tensors = %d, want %d (E*3 routed experts held raw)", got, E*3)
			}

			// (b) the forward consumes the resident k-quant experts to finite logits.
			ids := []int{0, 1, 2, 3, 4}
			act := m.Forward(ids)
			if act == nil || len(act.Logits) != len(ids) {
				t.Fatalf("Forward returned %d logit rows, want %d", lenLogits(act), len(ids))
			}
			if p, i, bad := glmQuantNonFinite(act.Logits); bad {
				t.Fatalf("logit[%d][%d] non-finite — resident %s expert forward produced garbage", p, i, tc.name)
			}

			// (c) the load-path breakdown tallies the experts as resident (the slow path avoided).
			var row *LoadPathStat
			for i := range prof.loadPathRows() {
				r := prof.loadPathRows()[i]
				if r.QuantType == tc.name && r.Expert {
					row = &r
				}
			}
			if row == nil || row.ResidentTensors != E*3 || row.DequantTensors != 0 {
				t.Fatalf("load-path breakdown for %s experts = %+v, want resident=%d dequant=0", tc.name, row, E*3)
			}
		})
	}
}
