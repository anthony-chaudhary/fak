package model

// v41_mhc_geometry_test.go — the #13258 regression witness for the per-layer mHC
// mix projection geometry.
//
// The staged vcruz Q2_K artifact carries blk.<L>.hc_attn_fn.weight as
// [HCMult*H, 24] (parsed from the real GGUF header at H=5120, hc_mult=4; the
// loader resolves it onto the forward's mhc.mixes.weight leaf). The published
// reference (inference/model.py mHC; transcribed at v4_flash_oracle_test.go:168)
// projects the FLATTENED four-stream residual (width 4H) through that fn block,
// so the logical projection is [24, 4H] and the artifact stores its transpose
// [4H, 24]. The reduced fixture keeps the legacy [24, H] single-stream matmul.
//
// Before #13258 the admission guard keyed mhc.mixes.weight to exactly [24, H], so
// a real artifact's [4H, 24] block was refused at admission by shape even after
// the loader populated the name. This test pins that seam using ONLY the
// pre-existing public admission boundary (v41ForwardAdmitted), so it builds and
// fails RED on the parent commit and passes GREEN after the fix:
//
//   - the published [4H, 24] geometry must ADMIT (so the physical serve advances
//     to a new named refusal rather than the old admission shape refusal);
//   - the reduced [24, H] fixture must stay admitted (no regression);
//   - every other shape must still fail closed.

import (
	"errors"
	"strings"
	"testing"
)

// v41ReshapeMHCMixToPublished rewrites the reduced fixture's mhc.mixes.weight
// manifest entry from the legacy [24, H] geometry to the artifact's stored
// [4H, 24] orientation. The element count is identical (24*H == 4H*24), so the
// zero-copy raw view stays valid; only the declared [out, in] axis order changes.
// This drives the production admission boundary through the real artifact shape
// without needing the artifact itself.
func v41ReshapeMHCMixToPublished(t *testing.T, m *Model) {
	t.Helper()
	H := m.Cfg.HiddenSize
	name := layerName(0, "mhc.mixes.weight")
	meta, ok := m.manifest[name]
	if !ok {
		t.Fatalf("fixture missing %s", name)
	}
	if len(meta.Shape) != 2 || meta.Shape[0] != v41MHCMixWidth || meta.Shape[1] != H {
		t.Fatalf("fixture %s shape = %v, want [%d %d]", name, meta.Shape, v41MHCMixWidth, H)
	}
	meta.Shape = []int{4 * H, v41MHCMixWidth}
	m.manifest[name] = meta
}

// TestV41ForwardAdmitsPublishedMHCFlattenedGeometry is the positive #13258
// witness (RED on the parent commit, GREEN after): the artifact's stored [4H, 24]
// mHC mix block must pass the production admission boundary. On the parent it was
// refused by the two-axis shape guard, which keyed exactly [24, H].
func TestV41ForwardAdmitsPublishedMHCFlattenedGeometry(t *testing.T) {
	m := v41ReducedModel(t)
	v41ReshapeMHCMixToPublished(t, m)

	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("published flattened mHC geometry refused at admission: %v", err)
	}
}

// TestV41ForwardReducedMHCStaysAdmitted retains the no-regression contract: the
// reduced fixture's legacy [24, H] geometry still admits unchanged.
func TestV41ForwardReducedMHCStaysAdmitted(t *testing.T) {
	m := v41ReducedModel(t)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("reduced mHC geometry admission error = %v, want nil", err)
	}
}

// TestV41ForwardWrongMHCShapeStillRefuses retains the fail-closed contract: a
// genuinely mis-shaped mHC mix weight still refuses by name at the mHC stage.
func TestV41ForwardWrongMHCShapeStillRefuses(t *testing.T) {
	m := v41ReducedModel(t)
	name := layerName(0, "mhc.mixes.weight")
	meta := m.manifest[name]
	meta.Shape = []int{v41MHCMixWidth + 1, m.Cfg.HiddenSize}
	m.manifest[name] = meta

	err := m.v41ForwardAdmitted()
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("mis-shaped mHC mix admission error = %v, want ErrV41ForwardStage", err)
	}
	if !strings.Contains(err.Error(), name) {
		t.Fatalf("mis-shaped mHC mix error = %v, want it to name %s", err, name)
	}
}

// TestV41ForwardAbsentMHCMixStillRefuses retains the fail-closed presence
// contract: an absent mhc.mixes.weight refuses by name at the mHC stage.
func TestV41ForwardAbsentMHCMixStillRefuses(t *testing.T) {
	m := v41ReducedModel(t)
	name := layerName(0, "mhc.mixes.weight")
	delete(m.manifest, name)

	err := m.v41ForwardAdmitted()
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("absent mHC mix admission error = %v, want ErrV41ForwardStage", err)
	}
	if !strings.Contains(err.Error(), "missing tensor "+name) {
		t.Fatalf("absent mHC mix error = %v, want a named missing-tensor refusal", err)
	}
}
