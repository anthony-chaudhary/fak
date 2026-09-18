package model

// v41_mhc_full_mix_failclosed_test.go — the #13258 fail-closed witness for the
// mHC mix geometry discriminator.
//
// The model has two geometries: the published FULL four-stream path (the
// flattened 4H residual projected through hc_attn_fn, logical [24, 4H] / stored
// [4H, 24]) and the reduced legacy single-stream path ([24, H]). Before this fix
// v41AdmitMHC admitted [24, H] and v41MHCWeightLayout reported flat=false for it
// REGARDLESS of the model's mode, so a FULL config carrying a [24, H] mix fell
// through to the legacy single-stream matRows path and silently mis-projected a
// sub-matrix instead of failing closed by name. v41AdmitMHC and
// v41MHCWeightLayout now key on v41ForwardGeometry(m.Cfg) -- the same source of
// truth forwardV41 uses -- so a full model refuses [24, H] at admission.

import (
	"errors"
	"testing"
)

// TestV41FullMHCShortMixFailsClosed pins the geometry-consistency of the mHC mix
// admission/layout: a FULL-geometry model whose mhc.mixes.weight is the reduced
// legacy [24, H] singleton must be REFUSED by name (v41AdmitMHC) and must not be
// reported as an admitted layout (v41MHCWeightLayout ok=false). The reduced
// fixture with the same [24, H] weight must still admit and still resolve to the
// legacy single-stream layout, and the full fixture with the flattened [24, 4H]
// weight must still admit as flat, so the guard does not over-tighten.
func TestV41FullMHCShortMixFailsClosed(t *testing.T) {
	// (a) Full geometry carrying the reduced [24, H] mix: refuse, do not
	// silently run the legacy single-stream sub-matrix. v41BuildFullModel carries
	// the geometry-consistent flattened [24, 4H], so override the manifest entry
	// to the reduced singleton this witness is about. Admission reads the declared
	// shape only, so the metadata override is sufficient.
	full := v41BuildFullModel(v41FullGeometryConfig(t), v41KVLoraRank)
	name := layerName(0, "mhc.mixes.weight")
	if meta, ok := full.manifest[name]; !ok {
		t.Fatalf("full fixture missing %s", name)
	} else {
		meta.Shape = []int{v41MHCMixWidth, full.Cfg.HiddenSize}
		full.manifest[name] = meta
	}
	if meta := full.manifest[name]; meta.Shape[0] != v41MHCMixWidth || meta.Shape[1] != full.Cfg.HiddenSize {
		t.Fatalf("full fixture %s shape = %v, want the reduced [%d %d]", name, meta.Shape, v41MHCMixWidth, full.Cfg.HiddenSize)
	}

	err := full.v41ForwardAdmitted()
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("full-geometry model with [%d %d] mHC mix admission error = %v, want ErrV41ForwardStage", v41MHCMixWidth, full.Cfg.HiddenSize, err)
	}
	if !contains(err.Error(), name) {
		t.Fatalf("full-geometry [24,H] mHC refusal = %v, want it to name %s", err, name)
	}
	if flat, _, ok := full.v41MHCWeightLayout(0); ok || flat {
		t.Fatalf("full-geometry [24,H] mHC layout = (flat=%v, ok=%v), want (false, false)", flat, ok)
	}

	// (b) Reduced geometry keeps admitting the legacy [24, H] mix unchanged and
	// still resolves it to the legacy single-stream (flat=false) layout.
	reduced := v41ReducedModel(t)
	if err := reduced.v41ForwardAdmitted(); err != nil {
		t.Fatalf("reduced legacy [24,H] mHC admission error = %v, want nil", err)
	}
	if flat, transposed, ok := reduced.v41MHCWeightLayout(0); !ok || flat || transposed {
		t.Fatalf("reduced legacy mHC layout = (flat=%v, transposed=%v, ok=%v), want (false, false, true)", flat, transposed, ok)
	}

	// (c) The full fixture carrying the flattened logical [24, 4H] mix still
	// admits as flat, so the discriminator did not over-tighten the real path.
	fullFlat := v41BuildFullModel(v41FullGeometryConfig(t), v41KVLoraRank)
	meta := fullFlat.manifest[name]
	meta.Shape = []int{v41MHCMixWidth, 4 * fullFlat.Cfg.HiddenSize}
	fullFlat.manifest[name] = meta
	if err := fullFlat.v41ForwardAdmitted(); err != nil {
		t.Fatalf("full-geometry flattened [24,4H] mHC admission error = %v, want nil", err)
	}
	if flat, transposed, ok := fullFlat.v41MHCWeightLayout(0); !ok || !flat || transposed {
		t.Fatalf("full flattened mHC layout = (flat=%v, transposed=%v, ok=%v), want (true, false, true)", flat, transposed, ok)
	}
}

// contains is a local substring helper kept private to this witness so the test
// does not pull in a strings import for a single check.
func contains(s, sub string) bool {
	if len(sub) == 0 {
		return true
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
