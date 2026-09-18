package model

import (
	"errors"
	"strings"
	"testing"
)

// v41_grouped_wo_admission_test.go - witness for the grouped output-projection
// geometry seam of the published DeepSeek-V4.1-Flash artifact (#13264).
//
// The native V4.1 forward consumes attn.wo_a.weight through
// V41GroupedOutputProjection, which reads the weight as the group-major
// [Groups, OLoRARank, HeadsPerGroup*HeadDim] tensor the artifact's
// attn_output_a.weight actually stores. The admission row, however, declared a
// flat [OLoraRank, NumHeads*HeadDim] two-axis shape. Those two declarations
// carry the SAME element count (Groups*OLoRARank*HeadsPerGroup*HeadDim ==
// OLoRARank*NumHeads*HeadDim) but assign different numbers to the two axes, so a
// real artifact load was refused by name before the grouped projection ran:
//
//	panic: model: V4.1 forward stage attention layer=0:
//	  tensor model.layers.0.attn.wo_a.weight shape [8192 4096], want [1024 32768]
//
// The published artifact stores [8192, 4096] with OGroups=8, OLoraRank=1024,
// NumHeads=64, HeadDim=512: 8192 = OGroups*OLoraRank and 4096 = HeadsPerGroup*
// HeadDim. The reduced fixture's flat [OLoraRank, NumHeads*HeadDim] declaration
// must keep being admitted, and a genuinely inconsistent shape must still fail
// closed.
//
// This file uses only pre-existing API so the mandatory red-then-green symptom
// witness can build the parent tree and observe the refusal at runtime.

// v41GroupedWoAConfig is a full-geometry V4.1 config whose grouped axes are
// expressible: unlike v41FullGeometryConfig (NumHeads=1), NumHeads here is a
// positive multiple of OGroups so HeadsPerGroup is non-zero and the grouped wo_a
// declaration is the artifact's real [OGroups*OLoraRank, HeadsPerGroup*HeadDim].
func v41GroupedWoAConfig(t *testing.T) Config {
	t.Helper()
	cfg := v41FullGeometryConfig(t)
	// Keep HeadDim at the reduced fixture's value (== v41KVLoraRank) so
	// v41BuildFullModel's kvLatentRank-wide attn.wkv.weight stays admitted, and
	// choose the grouped axes so HeadsPerGroup is a positive integer.
	cfg.OGroups = 2
	cfg.NumHeads = 4
	cfg.OLoraRank = 4
	return cfg
}

// v41GroupedWoAPublishedShape is the artifact's stored grouped wo_a geometry for
// a grouped-consistent config: [OGroups*OLoraRank, HeadsPerGroup*HeadDim].
func v41GroupedWoAPublishedShape(cfg Config) (rows, cols int) {
	return cfg.OGroups * cfg.OLoraRank, (cfg.NumHeads / cfg.OGroups) * cfg.HeadDim
}

// v41BuildGroupedWoAModel is v41BuildFullModel with the attn.wo_a.weight row
// declared at an explicit [rows, cols] shape, so the admission arm can be
// exercised without touching any other tensor.
func v41BuildGroupedWoAModel(cfg Config, rows, cols int) *Model {
	m := v41BuildFullModel(cfg, v41KVLoraRank)
	m.manifest[layerName(0, "attn.wo_a.weight")] = tensorMeta{Dtype: "F32", Shape: []int{rows, cols}}
	return m
}

// TestV41GroupedWoAPublishedShapeAdmitted is the RED-on-parent symptom witness:
// the artifact's grouped wo_a shape [OGroups*OLoraRank, qHeadDim/OGroups] must be
// admitted by the forward admission, not refused with the named two-axis shape
// error.
func TestV41GroupedWoAPublishedShapeAdmitted(t *testing.T) {
	cfg := v41GroupedWoAConfig(t)
	rows, cols := v41GroupedWoAPublishedShape(cfg)
	m := v41BuildGroupedWoAModel(cfg, rows, cols)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("grouped published wo_a [%d %d] refused: %v", rows, cols, err)
	}
}

// TestV41GroupedWoAReducedFlatShapeStillAdmitted keeps the fixture's flat
// [OLoraRank, NumHeads*HeadDim] declaration admitted.
func TestV41GroupedWoAReducedFlatShapeStillAdmitted(t *testing.T) {
	cfg := v41GroupedWoAConfig(t)
	rows, cols := cfg.OLoraRank, cfg.NumHeads*cfg.HeadDim
	m := v41BuildGroupedWoAModel(cfg, rows, cols)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("reduced flat wo_a [%d %d] refused: %v", rows, cols, err)
	}
}

// TestV41GroupedWoAElementCountParity pins the invariant the two admissions
// share: the grouped and flat declarations describe the same weight.
func TestV41GroupedWoAElementCountParity(t *testing.T) {
	cfg := v41GroupedWoAConfig(t)
	gRows, gCols := v41GroupedWoAPublishedShape(cfg)
	if gRows*gCols != cfg.OLoraRank*(cfg.NumHeads*cfg.HeadDim) {
		t.Fatalf("grouped element count %d != flat element count %d", gRows*gCols, cfg.OLoraRank*(cfg.NumHeads*cfg.HeadDim))
	}
}

// TestV41GroupedWoAPublishedArtifactValues pins the real published axes: with
// OGroups=8, OLoraRank=1024, NumHeads=64, HeadDim=512 the artifact's wo_a is
// [8192, 4096] and the admission's flat declaration is [1024, 32768]. Both carry
// 33,554,432 elements.
func TestV41GroupedWoAPublishedArtifactValues(t *testing.T) {
	groups, rank, heads, headDim := 8, 1024, 64, 512
	gRows := groups * rank
	gCols := (heads / groups) * headDim
	flatRows := rank
	flatCols := heads * headDim
	if gRows != 8192 || gCols != 4096 {
		t.Fatalf("published grouped wo_a = [%d %d], want [8192 4096]", gRows, gCols)
	}
	if flatRows != 1024 || flatCols != 32768 {
		t.Fatalf("published flat wo_a = [%d %d], want [1024 32768]", flatRows, flatCols)
	}
	if gRows*gCols != flatRows*flatCols {
		t.Fatalf("grouped %d elements != flat %d elements", gRows*gCols, flatRows*flatCols)
	}
}

// TestV41GroupedWoAInconsistentShapeFailsClosed retains the fail-closed
// property: a shape that is neither the flat nor a consistent grouped
// declaration must still refuse with the typed named error.
func TestV41GroupedWoAInconsistentShapeFailsClosed(t *testing.T) {
	cfg := v41GroupedWoAConfig(t)
	gRows, gCols := v41GroupedWoAPublishedShape(cfg)
	cases := []struct {
		name       string
		rows, cols int
	}{
		{"rows_off_by_one", gRows - 1, gCols},
		{"cols_off_by_one", gRows, gCols - 1},
		{"zero_cols", gRows, 0},
		{"flat_rows_with_grouped_cols", cfg.OLoraRank, gCols},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := v41BuildGroupedWoAModel(cfg, tc.rows, tc.cols)
			err := m.v41ForwardAdmitted()
			if !errors.Is(err, ErrV41ForwardStage) {
				t.Fatalf("wo_a [%d %d] error = %v, want ErrV41ForwardStage", tc.rows, tc.cols, err)
			}
			if !strings.Contains(err.Error(), "attn.wo_a.weight") {
				t.Fatalf("wo_a [%d %d] error %q does not name the refused tensor", tc.rows, tc.cols, err.Error())
			}
		})
	}
}

// TestV41GroupedWoAAbsentStillRefusesByName retains the missing-tensor half of
// the fail-closed property after the grouped rung is added.
func TestV41GroupedWoAAbsentStillRefusesByName(t *testing.T) {
	cfg := v41GroupedWoAConfig(t)
	m := v41BuildFullModel(cfg, v41KVLoraRank)
	delete(m.manifest, layerName(0, "attn.wo_a.weight"))
	err := m.v41ForwardAdmitted()
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("absent wo_a error = %v, want ErrV41ForwardStage", err)
	}
	if !strings.Contains(err.Error(), "attn.wo_a.weight") {
		t.Fatalf("absent wo_a error %q does not name the missing tensor", err.Error())
	}
}
