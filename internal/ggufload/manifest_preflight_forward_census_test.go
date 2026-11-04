package ggufload

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

// manifest_preflight_forward_census_test.go — the layout-guard (lg-preflight-wire) fixture
// witnesses for the forward-request census and the MTP declaration check. Each test mutates
// the published V4.1 header fixture IN MEMORY into one ledger incident class and asserts the
// exact typed refusal set: without the census code the same mutation yields ZERO refusals, so
// every test here fails without the guard and passes with it. Assertions compare reason
// tokens and axes (the closed fak.new-model-refusal/1 vocabulary), never error prose.

// v41FixtureTensorIndex locates one tensor of the merged V4.1 fixture directory.
func v41FixtureTensorIndex(t *testing.T, f *File, name string) int {
	t.Helper()
	for i := range f.Tensors {
		if f.Tensors[i].Name == name {
			return i
		}
	}
	t.Fatalf("V4.1 fixture has no tensor %s", name)
	return -1
}

// TestPreflightCatchesUnmappedNames_LML01 is the LML-01 witness: a tensor whose name resolves
// to no canonical leaf and no loader route is refused TENSOR_ORPHAN off the header, the same
// seam the serial physical V4.1 re-runs paid for one dialect suffix at a time.
//
// fak-test:runtime fast est=300ms
func TestPreflightCatchesUnmappedNames_LML01(t *testing.T) {
	f := loadV41HeaderFixture(t)
	f.Tensors = append(f.Tensors, TensorInfo{Name: "blk.0.zzz_unmapped.weight", Dims: []uint64{8}, Type: TensorF32})
	p := BuildManifestPreflight(ManifestPreflightInput{File: f})
	got := preflightTokens(p)
	want := []string{string(newmodel.RefusalTensorOrphan) + " blk.0.zzz_unmapped.weight"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("refusals = %v, want %v", got, want)
	}
}

// TestPreflightCatchesShapeMismatch_LML02 is the LML-02 witness: the published-vs-converter
// layout drift (the mHC mix stored at the reduced legacy [H, 24] width on a full-geometry
// artifact) is refused TENSOR_SHAPE_MISMATCH off the header instead of moving the refusal from
// by-name to by-shape after a hardware rung.
//
// fak-test:runtime fast est=300ms
func TestPreflightCatchesShapeMismatch_LML02(t *testing.T) {
	f := loadV41HeaderFixture(t)
	i := v41FixtureTensorIndex(t, f, "blk.0.hc_attn_fn.weight")
	f.Tensors[i].Dims = []uint64{5120, 24} // legacy [24, H] outer-first on a full-geometry artifact
	p := BuildManifestPreflight(ManifestPreflightInput{File: f})
	got := preflightTokens(p)
	want := []string{string(newmodel.RefusalTensorShape) + " blk.0.hc_attn_fn.weight"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("refusals = %v, want %v", got, want)
	}
}

// TestPreflightCatchesUnconsumableAdmission_LML03 is the LML-03 witness: the routed-expert
// blob carries one fewer expert than the config declares, so the loader's 1->E split cannot
// populate the expert band the router addresses — an admitted-but-unconsumable geometry,
// refused TENSOR_SHAPE_MISMATCH off the header instead of discovered per physical run.
//
// fak-test:runtime fast est=300ms
func TestPreflightCatchesUnconsumableAdmission_LML03(t *testing.T) {
	f := loadV41HeaderFixture(t)
	i := v41FixtureTensorIndex(t, f, "blk.0.ffn_gate_exps.weight")
	f.Tensors[i].Dims = []uint64{5120, 2304, 383} // E-1: the split cannot reach expert 383
	p := BuildManifestPreflight(ManifestPreflightInput{File: f})
	got := preflightTokens(p)
	want := []string{string(newmodel.RefusalTensorShape) + " blk.0.ffn_gate_exps.weight"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("refusals = %v, want %v", got, want)
	}
}

// TestPreflightCatchesDeclaredLayerMissingTensors_LML09 is the LML-09 witness: the config
// declares three MTP layers but the directory carries zero nextn tensors — a sourcing problem
// the header-only parse names pre-dispatch with TENSOR_MISSING on the mtp axis.
//
// fak-test:runtime fast est=300ms
func TestPreflightCatchesDeclaredLayerMissingTensors_LML09(t *testing.T) {
	f := loadV41HeaderFixture(t)
	f.Metadata["deepseek41.nextn_predict_layers"] = Value{Type: TypeUint32, Value: uint32(3)}
	p := BuildManifestPreflight(ManifestPreflightInput{File: f})
	got := preflightTokens(p)
	want := []string{string(newmodel.RefusalTensorMissing) + " mtp"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("refusals = %v, want %v", got, want)
	}
}

// TestPreflightCatchesReaderGap_LML10 is the LML-10 witness: the down-projection blob stores
// its outer axes transposed ([in, E, out] innermost-first instead of [out, in, E]), so the
// 1->E split would hand the forward per-expert matrices with swapped axes — a reader gap
// refused TENSOR_SHAPE_MISMATCH off the header instead of surfacing post-first-token.
//
// fak-test:runtime fast est=300ms
func TestPreflightCatchesReaderGap_LML10(t *testing.T) {
	f := loadV41HeaderFixture(t)
	i := v41FixtureTensorIndex(t, f, "blk.0.ffn_down_exps.weight")
	f.Tensors[i].Dims = []uint64{2304, 384, 5120} // outer axes transposed vs [2304, 5120, 384]
	p := BuildManifestPreflight(ManifestPreflightInput{File: f})
	got := preflightTokens(p)
	want := []string{string(newmodel.RefusalTensorShape) + " blk.0.ffn_down_exps.weight"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("refusals = %v, want %v", got, want)
	}
}

// TestPreflightForwardCensusCleanOnPublishedV41 pins the no-false-positive side: the published
// V4.1 header passes the forward-request census with nothing missing or mis-shaped, keeping
// the aggregate report at zero refusals.
//
// fak-test:runtime fast est=300ms
func TestPreflightForwardCensusCleanOnPublishedV41(t *testing.T) {
	f := loadV41HeaderFixture(t)
	p := BuildManifestPreflight(ManifestPreflightInput{File: f})
	if p.Refused() {
		t.Fatalf("published V4.1 header refused: %v", preflightTokens(p))
	}
	if p.Tensors.Missing != 0 || p.Tensors.Shape != 0 {
		t.Fatalf("census counts missing=%d shape=%d, want 0/0", p.Tensors.Missing, p.Tensors.Shape)
	}
}

// TestPreflightMTPDeclarationIgnoresHFOntlyDeclarations pins the scope boundary: a companion
// HF config.json that declares nextn layers is NOT a refusal source — the preflight judges the
// served artifact's own metadata, so the published V4.1 header (which mirrors no such key)
// stays READY against its own reference config.
//
// fak-test:runtime fast est=300ms
func TestPreflightMTPDeclarationIgnoresHFOntlyDeclarations(t *testing.T) {
	f := loadV41HeaderFixture(t)
	p := BuildManifestPreflight(ManifestPreflightInput{
		File:     f,
		HFConfig: []byte(`{"num_nextn_predict_layers":3}`),
	})
	for _, r := range p.Refusals {
		if r.Reason == newmodel.RefusalTensorMissing && r.Axis == "mtp" {
			t.Fatalf("HF-only MTP declaration refused the artifact: %+v", r)
		}
	}
}
