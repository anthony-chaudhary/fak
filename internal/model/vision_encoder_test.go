package model

import (
	"math"
	"testing"
)

// visionFixture parameterizes a tiny but complete vision tower so the tests can vary the
// two geometry facts the encoder DERIVES FROM TENSORS (temporal factor, projector depth)
// and the spatial merge, without a real mmproj.
type visionFixture struct {
	hidden        int
	heads         int
	layers        int
	patch         int
	merge         int
	temporal      int  // 1 still-image, 2 Qwen2-VL temporal-doubled
	ffn           int
	decoderHidden int  // projector out width == this
	twoLayerProj  bool // include mm.2 (GELU projector) vs single mm.0
}

// pat is a deterministic, bounded weight/pixel filler — distinct per (seed,i) so the
// forward is non-degenerate but reproducible (no rand, which the harness forbids anyway).
func pat(seed int) func(int) float32 {
	return func(i int) float32 { return float32(((i+seed)%11)-5) * 0.05 }
}

func namedTensor(name string, shape []int, fill func(int) float32) NamedTensorF32 {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = fill(i)
	}
	return NamedTensorF32{Name: name, Shape: shape, Data: data}
}

// buildVisionTower assembles every required tensor (plus a representative set of optional
// biases / position-embed / post-LN) for the fixture geometry.
func buildVisionTower(t *testing.T, f visionFixture) *VisionTower {
	t.Helper()
	H := f.hidden
	patchIn := visionChannels * f.temporal * f.patch * f.patch
	inMerge := H * f.merge * f.merge
	projMid := f.decoderHidden // single-layer projector: out == mid == decoderHidden
	tensors := []NamedTensorF32{
		namedTensor("v.patch_embd.weight", []int{H, patchIn}, pat(1)),
		namedTensor("v.patch_embd.bias", []int{H}, pat(2)),
		namedTensor("v.position_embd.weight", []int{64, H}, pat(3)), // 64 >= any seq tested
	}
	for l := 0; l < f.layers; l++ {
		p := "v.blk." + itoa(l) + "."
		tensors = append(tensors,
			namedTensor(p+"attn_q.weight", []int{H, H}, pat(10+l)),
			namedTensor(p+"attn_q.bias", []int{H}, pat(11+l)),
			namedTensor(p+"attn_k.weight", []int{H, H}, pat(12+l)),
			namedTensor(p+"attn_v.weight", []int{H, H}, pat(13+l)),
			namedTensor(p+"attn_out.weight", []int{H, H}, pat(14+l)),
			namedTensor(p+"attn_out.bias", []int{H}, pat(15+l)),
			namedTensor(p+"ln1.weight", []int{H}, pat(16+l)),
			namedTensor(p+"ln1.bias", []int{H}, pat(17+l)),
			namedTensor(p+"ln2.weight", []int{H}, pat(18+l)),
			namedTensor(p+"ffn_up.weight", []int{f.ffn, H}, pat(19+l)),
			namedTensor(p+"ffn_up.bias", []int{f.ffn}, pat(20+l)),
			namedTensor(p+"ffn_down.weight", []int{H, f.ffn}, pat(21+l)),
			namedTensor(p+"ffn_down.bias", []int{H}, pat(22+l)),
		)
	}
	if f.twoLayerProj {
		// mm.0 -> projMid (chosen 2*decoderHidden here), mm.2 -> decoderHidden.
		projMid = 2 * f.decoderHidden
		tensors = append(tensors,
			namedTensor("mm.0.weight", []int{projMid, inMerge}, pat(30)),
			namedTensor("mm.0.bias", []int{projMid}, pat(31)),
			namedTensor("mm.2.weight", []int{f.decoderHidden, projMid}, pat(32)),
			namedTensor("mm.2.bias", []int{f.decoderHidden}, pat(33)),
		)
	} else {
		tensors = append(tensors,
			namedTensor("mm.0.weight", []int{projMid, inMerge}, pat(30)),
			namedTensor("mm.0.bias", []int{projMid}, pat(31)),
		)
	}
	cfg := VisionConfig{
		HiddenSize: H, NumLayers: f.layers, NumHeads: f.heads,
		PatchSize: f.patch, MergeSize: f.merge, ImageSize: 0, LNEps: 1e-6,
	}
	tower, err := NewVisionTower(cfg, tensors)
	if err != nil {
		t.Fatalf("NewVisionTower: %v", err)
	}
	return tower
}

// makePixels builds a deterministic normalized CHW plane for a rows x cols patch grid.
func makePixels(enc *visionEncoder, gridRows, gridCols int) []float32 {
	imgH, imgW := gridRows*enc.patch, gridCols*enc.patch
	px := make([]float32, visionChannels*imgH*imgW)
	fill := pat(7)
	for i := range px {
		px[i] = fill(i)
	}
	return px
}

func finite(vecs [][]float32) bool {
	for _, v := range vecs {
		for _, x := range v {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return false
			}
		}
	}
	return true
}

// TestVisionEncoderForwardShapes runs the full patch-embed -> ViT -> projector forward
// and pins the output shape (one decoder-hidden-width vector per merged patch) plus
// finiteness. No float golden: there is no in-tree numeric oracle for the ViT tower.
func TestVisionEncoderForwardShapes(t *testing.T) {
	f := visionFixture{hidden: 4, heads: 2, layers: 2, patch: 2, merge: 1, temporal: 1, ffn: 8, decoderHidden: 6}
	enc, err := newVisionEncoder(buildVisionTower(t, f), f.decoderHidden)
	if err != nil {
		t.Fatalf("newVisionEncoder: %v", err)
	}
	if enc.temporal != 1 {
		t.Errorf("temporal = %d, want 1", enc.temporal)
	}
	if enc.projOut != f.decoderHidden {
		t.Errorf("projOut = %d, want %d", enc.projOut, f.decoderHidden)
	}
	rows, cols := 1, 3
	out, err := enc.forwardPixels(makePixels(enc, rows, cols), rows, cols)
	if err != nil {
		t.Fatalf("forwardPixels: %v", err)
	}
	if len(out) != rows*cols {
		t.Fatalf("got %d vectors, want %d", len(out), rows*cols)
	}
	for i, v := range out {
		if len(v) != f.decoderHidden {
			t.Fatalf("vector %d width %d, want %d", i, len(v), f.decoderHidden)
		}
	}
	if !finite(out) {
		t.Errorf("forward produced non-finite values")
	}
}

// TestVisionEncoderDeterministic proves the forward is a pure function of its inputs.
func TestVisionEncoderDeterministic(t *testing.T) {
	f := visionFixture{hidden: 4, heads: 2, layers: 1, patch: 2, merge: 1, temporal: 1, ffn: 8, decoderHidden: 4}
	enc, err := newVisionEncoder(buildVisionTower(t, f), f.decoderHidden)
	if err != nil {
		t.Fatalf("newVisionEncoder: %v", err)
	}
	px := makePixels(enc, 2, 2)
	a, err1 := enc.forwardPixels(px, 2, 2)
	b, err2 := enc.forwardPixels(px, 2, 2)
	if err1 != nil || err2 != nil {
		t.Fatalf("forward: %v / %v", err1, err2)
	}
	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Fatalf("nondeterministic at [%d][%d]: %v != %v", i, j, a[i][j], b[i][j])
			}
		}
	}
}

// TestVisionEncoderTemporalDerivation proves the temporal factor (Qwen2-VL's doubled
// patch_embed) is read from the tensor's in-features, not hardcoded, and the forward
// still runs under it.
func TestVisionEncoderTemporalDerivation(t *testing.T) {
	f := visionFixture{hidden: 4, heads: 2, layers: 1, patch: 2, merge: 1, temporal: 2, ffn: 8, decoderHidden: 4}
	enc, err := newVisionEncoder(buildVisionTower(t, f), f.decoderHidden)
	if err != nil {
		t.Fatalf("newVisionEncoder: %v", err)
	}
	if enc.temporal != 2 {
		t.Fatalf("derived temporal = %d, want 2", enc.temporal)
	}
	if enc.patchIn != visionChannels*2*f.patch*f.patch {
		t.Fatalf("patchIn = %d, want %d", enc.patchIn, visionChannels*2*f.patch*f.patch)
	}
	out, err := enc.forwardPixels(makePixels(enc, 1, 2), 1, 2)
	if err != nil || len(out) != 2 || !finite(out) {
		t.Fatalf("temporal forward: out=%d err=%v finite=%v", len(out), err, finite(out))
	}
}

// TestVisionEncoderTwoLayerProjector exercises the mm.0 -> GELU -> mm.2 projector path
// and its derived inner/out dims.
func TestVisionEncoderTwoLayerProjector(t *testing.T) {
	f := visionFixture{hidden: 4, heads: 2, layers: 1, patch: 2, merge: 1, temporal: 1, ffn: 8, decoderHidden: 4, twoLayerProj: true}
	enc, err := newVisionEncoder(buildVisionTower(t, f), f.decoderHidden)
	if err != nil {
		t.Fatalf("newVisionEncoder: %v", err)
	}
	if enc.projMid != 2*f.decoderHidden || enc.projOut != f.decoderHidden {
		t.Fatalf("projector dims mid=%d out=%d, want mid=%d out=%d", enc.projMid, enc.projOut, 2*f.decoderHidden, f.decoderHidden)
	}
	out, err := enc.forwardPixels(makePixels(enc, 1, 2), 1, 2)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	for _, v := range out {
		if len(v) != f.decoderHidden {
			t.Fatalf("vector width %d, want %d", len(v), f.decoderHidden)
		}
	}
}

// TestVisionEncoderMergeReducesTokens proves the spatial merge folds merge^2 patches into
// one token (a 2x2 grid at merge=2 yields exactly one vector).
func TestVisionEncoderMergeReducesTokens(t *testing.T) {
	f := visionFixture{hidden: 4, heads: 2, layers: 1, patch: 2, merge: 2, temporal: 1, ffn: 8, decoderHidden: 4}
	enc, err := newVisionEncoder(buildVisionTower(t, f), f.decoderHidden)
	if err != nil {
		t.Fatalf("newVisionEncoder: %v", err)
	}
	out, err := enc.forwardPixels(makePixels(enc, 2, 2), 2, 2)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("merge=2 on 2x2 grid gave %d tokens, want 1", len(out))
	}
	// A grid not divisible by merge is rejected, not silently truncated.
	if _, err := enc.forwardPixels(makePixels(enc, 1, 2), 1, 2); err == nil {
		t.Errorf("expected error for grid 1x2 under merge=2")
	}
}

// TestVisionEncoderTokenCounts pins the grid math EncodeImage/#4032 reconcile against.
func TestVisionEncoderTokenCounts(t *testing.T) {
	f := visionFixture{hidden: 4, heads: 2, layers: 1, patch: 2, merge: 2, temporal: 1, ffn: 8, decoderHidden: 4}
	enc, err := newVisionEncoder(buildVisionTower(t, f), f.decoderHidden)
	if err != nil {
		t.Fatalf("newVisionEncoder: %v", err)
	}
	unit := f.patch * f.merge // 4
	if got := enc.NumImageTokens(2*unit, 3*unit); got != 6 {
		t.Errorf("NumImageTokens(%d,%d) = %d, want 6", 2*unit, 3*unit, got)
	}
	if got := enc.NumImageTokens(unit+1, unit); got != 0 {
		t.Errorf("NumImageTokens on non-multiple width = %d, want 0", got)
	}
	if _, _, ok := enc.GridForImage(unit, 0); ok {
		t.Errorf("GridForImage with zero height reported ok")
	}
}

// TestVisionEncoderConstructionValidates pins the fail-fast construction contract.
func TestVisionEncoderConstructionValidates(t *testing.T) {
	f := visionFixture{hidden: 4, heads: 2, layers: 1, patch: 2, merge: 1, temporal: 1, ffn: 8, decoderHidden: 4}

	if _, err := newVisionEncoder(nil, 4); err == nil {
		t.Errorf("nil tower accepted")
	}
	// projector out (4) must equal decoder hidden; 5 must be rejected.
	if _, err := newVisionEncoder(buildVisionTower(t, f), 5); err == nil {
		t.Errorf("decoder-hidden mismatch accepted")
	}
	// heads must divide hidden.
	bad := f
	bad.heads = 3
	if _, err := newVisionEncoder(buildVisionTower(t, bad), 4); err == nil {
		t.Errorf("indivisible head count accepted")
	}
}

// TestVisionEncoderMissingTensorRejected proves a tower missing a required tensor is
// refused at construction with the resolver's precise error, not at forward time.
func TestVisionEncoderMissingTensorRejected(t *testing.T) {
	f := visionFixture{hidden: 4, heads: 2, layers: 1, patch: 2, merge: 1, temporal: 1, ffn: 8, decoderHidden: 4}
	tower := buildVisionTower(t, f)
	delete(tower.manifest, "mm.0.weight") // drop a required projector tensor
	if _, err := newVisionEncoder(tower, 4); err == nil {
		t.Fatalf("missing mm.0.weight accepted")
	}
}
