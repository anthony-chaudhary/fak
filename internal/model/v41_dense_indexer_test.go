package model

import (
	"math"
	"testing"
)

// v41_dense_indexer_test.go -- DeepSeek V4.1 layer-0 DSA indexer semantics
// (anthony-chaudhary/fak#13227, leaf of the V4.1 native execution track
// anthony-chaudhary/fak#12640).
//
// V4.1's published schedule runs its lightning indexer on a STRIDED subset of
// layers starting at layer 2 (index_source_layer_ids = {2,8,14,20,24,28,32,36};
// compress_ratios[0:2] = 0). Layers 0/1 therefore ship NO indexer tensors AND
// have no preceding full layer to reuse. The GLM DSA band contract requires a
// band to begin on a "full"-indexer layer (internal/model/glm_dsa_session.go:171,
// partition.go:127, pipeline.go:134) and dsaIndexShare refuses a leading "shared"
// layer (dsa_index.go:250) -- a leading "shared" is UNSATISFIABLE. The correct
// role for such a prefix layer is "dense": it attends its full causal prefix,
// exactly the existing IndexNHeads==0 dense-MLA seam.
//
// Before the fix the GGUF loader classified those layers "shared", so the native
// serve failed closed with "cannot start at GLM shared-indexer layer 0" -- a
// legible error, but still no token. These tests pin the resolved semantics.

// TestV41DenseIndexerClassifiesAndBands is the pure classification/contract rung.
func TestV41DenseIndexerClassifiesAndBands(t *testing.T) {
	cfg := Config{IndexerTypes: []string{"dense", "full", "shared", "causal"}}
	cases := []struct {
		layer int
		dense bool
		full  bool
		share bool
	}{
		{0, true, false, false},
		{1, false, true, false},
		{2, false, false, true},
		{3, true, false, false}, // "causal" is a dense alias
	}
	for _, c := range cases {
		if got := glmDsaIndexerIsDense(cfg, c.layer); got != c.dense {
			t.Errorf("layer %d dense = %v, want %v", c.layer, got, c.dense)
		}
		if got := glmDsaIndexerIsFull(cfg, c.layer); got != c.full {
			t.Errorf("layer %d full = %v, want %v", c.layer, got, c.full)
		}
		if got := glmDsaIndexerIsShared(cfg, c.layer); got != c.share {
			t.Errorf("layer %d shared = %v, want %v", c.layer, got, c.share)
		}
		// A dense layer may BEGIN a band (it needs no predecessor selection); a
		// shared layer may not (it would read nil).
		wantBand := !c.share
		if got := glmDsaBandStartOK(cfg, c.layer); got != wantBand {
			t.Errorf("layer %d bandStartOK = %v, want %v", c.layer, got, wantBand)
		}
	}
}

// TestV41DenseLayerZeroDecodeBandStarts is the band-start rung: decodeBandGLMDsa
// over a schedule whose layer 0 is dense must NOT return the "cannot start at
// GLM shared-indexer layer" error. It drives the real tiny GLM-DSA fixture with
// NO indexer tensor on layer 0 (the artifact's condition), so the forward reads
// none there.
func TestV41DenseLayerZeroDecodeBandStarts(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(t, "BF16", 3,
		[]string{"dense", "full", "shared"}, false, false, true, true)
	m, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !glmDsaIndexerIsDense(m.Cfg, 0) {
		t.Fatalf("fixture layer 0 = %q, want dense", m.Cfg.IndexerTypes[0])
	}
	s := m.NewSession()
	h, err := s.decodeBandGLMDsa(3, nil, 0, 1, 0, true, false)
	if err != nil {
		t.Fatalf("decodeBandGLMDsa on a dense layer-0 band must start, got: %v", err)
	}
	for i, v := range h {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("dense layer-0 band hidden[%d] non-finite: %v", i, v)
		}
	}
}

// TestV41DenseSchedulePrefillReachesFiniteLogits is the acceptance rung: a full
// prefill over a schedule with a dense (indexer-less) layer 0 advances PAST layer
// 0 to finite logits. It is the software end of the issue's done condition: the
// strided indexer set drives tokenHiddenGLMDsa to finite output, where before the
// fix it failed closed at the band start.
func TestV41DenseSchedulePrefillReachesFiniteLogits(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(t, "BF16", 4,
		[]string{"dense", "dense", "full", "shared"}, false, false, true, true)
	m, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	s := m.NewSession()
	logits := s.Prefill([]int{3, 1, 4, 1, 5})
	if len(logits) != m.Cfg.VocabSize {
		t.Fatalf("prefill logits len = %d, want vocab %d", len(logits), m.Cfg.VocabSize)
	}
	assertFiniteGLM(t, "dense-schedule prefill logits", logits)

	step := s.Step(9)
	assertFiniteGLM(t, "dense-schedule step logits", step)

	// The step navigates the dense layer 0 again; if it had been mis-scheduled as
	// shared, the forward would have panicked/hung on a nil shared selection.
	var sum float64
	for _, v := range step {
		sum += float64(v)
	}
	if sum == 0 {
		t.Fatal("step logits are all zero -- the forward did not run")
	}
}

// TestV41DenseIndexSharePublishesNothing is the control-flow rung: dsaIndexShare
// over a dense prefix publishes an empty decision (a dense layer selects no sparse
// set) and clears the share source, so a later "shared" layer still requires a real
// preceding "full" layer.
func TestV41DenseIndexSharePublishesNothing(t *testing.T) {
	// Schedule indices: 0 dense, 1 full, 2 shared. The full layer is index 1.
	fullByLayer := map[int][][]int{1: {{0, 1}}}
	got, ok := dsaIndexShare([]string{"dense", "full", "shared"}, fullByLayer)
	if !ok {
		t.Fatal("dsaIndexShare rejected a legal dense->full->shared schedule")
	}
	if got[0] != nil {
		t.Errorf("dense layer 0 decision = %v, want nil (publishes no sparse set)", got[0])
	}
	if len(got[2]) != 1 || len(got[2][0]) != 2 {
		t.Errorf("shared layer 2 decision = %v, want the layer-1 full selection", got[2])
	}
	// A dense layer does NOT satisfy a following shared layer's predecessor need.
	if _, ok := dsaIndexShare([]string{"dense", "shared"}, nil); ok {
		t.Fatal("dsaIndexShare accepted dense->shared; a dense layer publishes no selection to share")
	}
}

// TestV41DenseUnknownIndexerStillFailsClosed retains the fail-closed half: an
// unrecognized indexer type is still refused by both the classifier and the share
// expansion, so the new "dense" kind did not open a silent fall-through.
func TestV41DenseUnknownIndexerStillFailsClosed(t *testing.T) {
	cfg := Config{IndexerTypes: []string{"bogus"}}
	if got := glmDsaIndexerKind(cfg, 0); got != "unknown" {
		t.Fatalf("unknown type classified %q, want unknown", got)
	}
	if glmDsaIndexerIsFull(cfg, 0) || glmDsaIndexerIsShared(cfg, 0) || glmDsaIndexerIsDense(cfg, 0) {
		t.Fatal("an unknown indexer type must not satisfy full/shared/dense")
	}
	if _, ok := dsaIndexShare([]string{"bogus"}, nil); ok {
		t.Fatal("dsaIndexShare accepted an unknown indexer type")
	}
}
