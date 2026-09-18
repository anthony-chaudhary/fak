package model

// v41_mhc_resident_read_test.go — the #13262 witness for the DeepSeek-V4.1 mHC
// mix read from a RESIDENT quantization store.
//
// The pinned vcruz Q2_K artifact carries blk.<L>.hc_attn_fn.weight (canonical
// mhc.mixes.weight) as type=10 Q2_K with GGUF dims [20480, 24], which
// modelShapeFromGGUFDims reverses to the model [24, 4H] (H=5120). Two seams then
// interact:
//
//  1. isQuantWeight now admits the .mhc.mixes.weight / .mhc.ffn_mixes.weight
//     leaves, so the loader stores the Q2_K block in the resident k-quant store
//     instead of eagerly dequantizing it to the f32 manifest. Before that arm the
//     block fell through to dequantF32Limited and was charged whole.
//  2. forwardV41 must therefore read the mix residency-completely. v41MHCMixF32
//     returns the f32 manifest view unchanged when present (the reduced fixture,
//     byte-identical) and otherwise dequantizes from whichever resident store
//     holds it (kqw/q4kw/q2w/q8w/q4w/gptqw). Before this leaf the read went
//     through m.tensor, which panics "model: missing tensor ..." on a resident-
//     only mix.
//
// These tests build models whose mhc.mixes.weight lives ONLY in a resident store
// and drive the real forwardV41. On the parent commit the m.tensor read panics;
// after the fix the forward runs and emits finite logits.

import (
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"
)

// v41Q2KConstantBlock encodes one valid 84-byte Q2_K super-block whose 256 weights
// all dequantize to val. Encoding: f16 super-scale d = val, f16 min = 0, every
// 4-bit sub-scale nibble = 1 (dl = d, ml = 0), every 2-bit code = 1 (table[1] = dl).
// val must be exactly f16-representable so the round trip is exact.
func v41Q2KConstantBlock(val float32) []byte {
	if val != 0.5 {
		panic("v41Q2KConstantBlock: only 0.5 is f16-exact for this fixture")
	}
	blk := make([]byte, 84)
	for i := 0; i < 16; i++ {
		blk[i] = 0x01 // low nibble scale = 1, high nibble min = 0
	}
	for i := 16; i < 80; i++ {
		blk[i] = 0x55 // 2-bit codes all == 1
	}
	binary.LittleEndian.PutUint16(blk[80:], 0x3800) // f16 0.5
	binary.LittleEndian.PutUint16(blk[82:], 0x0000) // f16 0.0
	return blk
}

// v41ResidentQ2KRaw builds a deterministic Q2_K payload for an [out, in] matrix
// (in a multiple of 256) whose dequantized weights are a uniform 0.5.
func v41ResidentQ2KRaw(out, in int) []byte {
	nblk := in / 256
	block := v41Q2KConstantBlock(0.5)
	raw := make([]byte, out*nblk*len(block))
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			copy(raw[o*nblk*84+b*84:o*nblk*84+(b+1)*84], block)
		}
	}
	return raw
}

// v41ResidentQ8Raw builds a deterministic valid Q8_0 payload for an [out, in]
// matrix (in a multiple of 32): each 34-byte block is f16 scale 1.0 followed by
// 32 int8 codes from a fixed pattern, so the dequantized row is that pattern.
func v41ResidentQ8Raw(out, in int) []byte {
	nblk := in / 32
	raw := make([]byte, out*nblk*34)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[o*nblk*34+b*34 : o*nblk*34+(b+1)*34]
			binary.LittleEndian.PutUint16(blk[0:], 0x3C00) // f16 1.0
			for j := 0; j < 32; j++ {
				blk[2+j] = byte(int8((o*31+b*17+j)%251 - 125))
			}
		}
	}
	return raw
}

// v41MoveMixToResidentKQuant removes the f32 manifest entry for a layer's
// mhc.mixes.weight and installs the supplied resident raw k-quant payload in
// m.kqw under the same name, so the ONLY way to read the mix is the resident
// store. shape is the model [out, in].
func v41MoveMixToResidentKQuant(m *Model, l int, shape []int, raw []byte, kind kQuantKind) {
	name := layerName(l, "mhc.mixes.weight")
	delete(m.manifest, name)
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	m.kqw[name] = quantizeKQuantFromRaw(raw, shape[0], shape[1], kind)
}

// TestV41MHCResidentRead is the #13262 acceptance witness: a model whose
// mhc.mixes.weight lives ONLY in the resident k-quant store runs the real
// reduced forward and emits finite logits. It also pins that the residency-aware
// read returns the same weights a cold dequant produces.
func TestV41MHCResidentRead(t *testing.T) {
	m := v41ReducedModel(t)
	H := m.Cfg.HiddenSize
	name := layerName(0, "mhc.mixes.weight")

	// Q8_0 is the resident kind a 64-wide reduced mix can carry (in % 32 == 0).
	raw := v41ResidentQ8Raw(v41MHCMixWidth, H)
	v41MoveMixToResidentKQuant(m, 0, []int{v41MHCMixWidth, H}, raw, kindQ8_0)

	if m.has(name) {
		t.Fatal("resident fixture still carries an f32 manifest entry for the mix; the read is not exercised")
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("resident-mix admission error = %v, want nil", err)
	}

	// The residency-aware read must agree with a cold resident dequant of the same
	// k-quant tensor.
	got, err := m.v41MHCMixF32(0)
	if err != nil {
		t.Fatalf("v41MHCMixF32 error = %v, want nil", err)
	}
	qt := m.kqw[name]
	want := make([]float32, qt.out*qt.in)
	rowBytes := qt.rowBytes()
	bb := qt.kind.blockBytes()
	bw := qt.kind.blockWeights()
	for o := 0; o < qt.out; o++ {
		row := qt.raw[o*rowBytes : (o+1)*rowBytes]
		for j := 0; j < qt.nblk; j++ {
			kQuantDequantSuperBlock(want[o*qt.in+j*bw:], row[j*bb:(j+1)*bb], qt.kind)
		}
	}
	if len(got) != len(want) {
		t.Fatalf("mix read width %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("mix read [%d] = %v, want the resident dequant %v", i, got[i], want[i])
		}
	}

	// The real forward must run instead of panicking in m.tensor.
	act, err := m.forwardV41([]int{1, 3}, nil)
	if err != nil {
		t.Fatalf("resident-mix forward error = %v, want nil", err)
	}
	if len(act.Logits) != 2 {
		t.Fatalf("forward produced %d logit rows, want 2", len(act.Logits))
	}
	for r, row := range act.Logits {
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logits[%d][%d] = %v, want finite", r, i, v)
			}
		}
	}
}

// TestV41MHCResidentReadFullFlattened drives the flattened-four-stream path with
// the mix resident as the pinned artifact's Q2_K [24, 4H] (H=64 => in=256). It
// proves the full path's v41MHCProjectFull consumption reads the resident store
// rather than panicking in m.tensor.
func TestV41MHCResidentReadFullFlattened(t *testing.T) {
	m := v41RawFullFlattenedResidentMHC(t)

	flat, transposed, ok := m.v41MHCWeightLayout(0)
	if !ok || !flat || transposed {
		t.Fatalf("resident full mHC layout = (flat=%v, transposed=%v, ok=%v), want the logical [24,4H] resident shape", flat, transposed, ok)
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("resident full-geometry admission error = %v, want nil", err)
	}
	act, err := m.forwardV41([]int{0, 1}, nil)
	if err != nil {
		t.Fatalf("resident full-geometry forward error = %v, want nil", err)
	}
	if len(act.Logits) != 2 {
		t.Fatalf("forward produced %d logit rows, want 2", len(act.Logits))
	}
	for r, row := range act.Logits {
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logits[%d][%d] = %v, want finite", r, i, v)
			}
		}
	}
}

// v41RawFullFlattenedResidentMHC builds a small full-geometry V4.1 model whose
// mhc.mixes.weight is resident as Q2_K [24, 4H] (the pinned artifact's kind and
// logical orientation). Everything else is the same synthetic layout the other
// V4.1 fixtures use.
func v41RawFullFlattenedResidentMHC(t *testing.T) *Model {
	t.Helper()
	cfg := v41FullGeometryConfig(t)
	cfg.OGroups = 1
	H := cfg.HiddenSize
	in := 4 * H
	I := cfg.MoEIntermediateSize
	qHeadDim := cfg.NumHeads * cfg.HeadDim
	oDim := cfg.OLoraRank * cfg.OGroups

	type ts = synthTensor
	tensors := []ts{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{"lm_head.weight", []int{cfg.VocabSize, H}},
		{"model.norm.weight", []int{H}},
		{layerName(0, "attn_norm.weight"), []int{H}},
		{layerName(0, "ffn_norm.weight"), []int{H}},
		{layerName(0, "mhc.mixes.weight"), []int{v41MHCMixWidth, in}},
		{layerName(0, "mhc.base"), []int{v41MHCMixWidth}},
		{layerName(0, "mhc.scale"), []int{3}},
		{layerName(0, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
		{layerName(0, "attn.wq_b.weight"), []int{qHeadDim, cfg.QLoraRank}},
		{layerName(0, "attn.wkv.weight"), []int{v41KVLoraRank, H}},
		{layerName(0, "attn.wo_a.weight"), []int{cfg.OLoraRank, qHeadDim}},
		{layerName(0, "attn.wo_b.weight"), []int{H, oDim}},
		{layerName(0, "attn.sink"), []int{cfg.NumHeads}},
		{layerName(0, "ffn.gate.weight"), []int{cfg.NumExperts, H}},
		{layerName(0, "ffn.gate.e_score_correction_bias"), []int{cfg.NumExperts}},
		{layerName(0, "ffn.shared_experts.w1.weight"), []int{I, H}},
		{layerName(0, "ffn.shared_experts.w3.weight"), []int{I, H}},
		{layerName(0, "ffn.shared_experts.w2.weight"), []int{H, I}},
	}
	for e := 0; e < cfg.NumExperts; e++ {
		stem := "ffn.experts." + itoa(e)
		tensors = append(tensors,
			ts{layerName(0, stem+".w1.weight"), []int{I, H}},
			ts{layerName(0, stem+".w3.weight"), []int{I, H}},
			ts{layerName(0, stem+".w2.weight"), []int{H, I}},
		)
	}

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") || hasSuffix(name, "ffn_norm.weight"):
			return 1.0
		case hasSuffix(name, "mhc.scale"):
			return 1.0
		case hasSuffix(name, "mhc.base"):
			return 0.0
		case hasSuffix(name, "attn.sink"):
			return 0.25 * next()
		default:
			return synthMatmulFill(name, next)
		}
	})
	m := &Model{Cfg: cfg, manifest: man, raw: raw}
	v41MoveMixToResidentKQuant(m, 0, []int{v41MHCMixWidth, in}, v41ResidentQ2KRaw(v41MHCMixWidth, in), kindQ2K)
	return m
}

// v41ScalarMHCProjectFullRef is the hand-rolled scalar transcription of the
// flattened projection used to pin v41MHCProjectFull. hcFn is the logical
// row-major [24, 4H] coefficient block.
func v41ScalarMHCProjectFullRef(xflat, hcFn []float32, eps float32) []float32 {
	flatWidth := len(xflat)
	var ss float32
	for _, v := range xflat {
		ss += v * v
	}
	rsqrt := float32(1 / math.Sqrt(float64(ss/float32(flatWidth)+eps)))
	out := make([]float32, v41MHCMixWidth)
	for m := 0; m < v41MHCMixWidth; m++ {
		row := hcFn[m*flatWidth : (m+1)*flatWidth]
		var s float32
		for i := 0; i < flatWidth; i++ {
			s += row[i] * xflat[i]
		}
		out[m] = s * rsqrt
	}
	return out
}

// TestV41MHCProjectFullMath pins the flattened four-stream projection against a
// hand-rolled scalar reference for the logical [24, 4H] orientation AND the
// stored [4H, 24] transpose, asserting both admitted orientations yield the same
// 24 outputs.
func TestV41MHCProjectFullMath(t *testing.T) {
	const H = 4
	streams := [][]float32{
		{0.25, -1.5, 2.0, 0.5},
		{-0.75, 0.125, 1.25, -2.0},
		{1.5, -0.25, 0.5, 0.75},
		{-1.25, 0.5, -0.5, 1.0},
	}
	xflat := make([]float32, 0, 4*H)
	for _, s := range streams {
		xflat = append(xflat, s...)
	}
	logical := make([]float32, v41MHCMixWidth*4*H)
	for i := range logical {
		logical[i] = float32((i%17)-8) / 9
	}
	stored := make([]float32, 4*H*v41MHCMixWidth)
	for m := 0; m < v41MHCMixWidth; m++ {
		for i := 0; i < 4*H; i++ {
			stored[i*v41MHCMixWidth+m] = logical[m*4*H+i]
		}
	}
	want := v41ScalarMHCProjectFullRef(xflat, logical, 1e-6)

	gotLogical, err := v41MHCProjectFull(logical, streams, H, 1e-6, false)
	if err != nil {
		t.Fatalf("logical projection error = %v, want nil", err)
	}
	gotStored, err := v41MHCProjectFull(stored, streams, H, 1e-6, true)
	if err != nil {
		t.Fatalf("stored projection error = %v, want nil", err)
	}
	for m := 0; m < v41MHCMixWidth; m++ {
		if d := math.Abs(float64(gotLogical[m] - want[m])); d > 1e-5 {
			t.Fatalf("logical mixes[%d] = %v, want scalar %v (diff %g)", m, gotLogical[m], want[m], d)
		}
		if gotStored[m] != gotLogical[m] {
			t.Fatalf("stored mixes[%d] = %v, want the logical %v", m, gotStored[m], gotLogical[m])
		}
	}
}

// TestV41MHCProjectFullRefusesBadGeometryResident retains the fail-closed
// contract: a residual or weight that does not carry the flattened four-stream
// geometry refuses with a typed error rather than silently projecting a wrong
// sub-matrix.
func TestV41MHCProjectFullRefusesBadGeometryResident(t *testing.T) {
	const H = 4
	good := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}
	w := make([]float32, v41MHCMixWidth*4*H)

	cases := []struct {
		name string
		w    []float32
		ss   [][]float32
	}{
		{"three streams", w, good[:3]},
		{"wrong width", w, [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}},
		{"short weight", w[:len(w)-1], good},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := v41MHCProjectFull(tc.w, tc.ss, H, 1e-6, false)
			if err == nil {
				t.Fatal("malformed flattened mHC geometry admitted, want a typed refusal")
			}
		})
	}
}

// TestV41MHCMixF32AbsentRefusesTyped pins that a mix absent from every store
// fails closed with a typed ErrV41ForwardStage naming the tensor, rather than
// panicking through m.tensor.
func TestV41MHCMixF32AbsentRefusesTyped(t *testing.T) {
	m := v41ReducedModel(t)
	name := layerName(0, "mhc.mixes.weight")
	delete(m.manifest, name)
	if _, err := m.v41MHCMixF32(0); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("absent mix read error = %v, want ErrV41ForwardStage", err)
	} else if !strings.Contains(err.Error(), name) {
		t.Fatalf("absent mix read error = %v, want it to name %s", err, name)
	}
}

// TestIsQuantWeightMHCLeaves pins the resident-residency gate: the V4.1 mHC mix
// leaves that ggufload's deepseek41MHCSuffixName emits are quant weights, and a
// sibling non-mHC name is unchanged.
func TestIsQuantWeightMHCLeaves(t *testing.T) {
	if !isQuantWeight("model.layers.0.mhc.mixes.weight") {
		t.Fatal("isQuantWeight(model.layers.0.mhc.mixes.weight) = false, want true")
	}
	if !isQuantWeight("model.layers.0.mhc.ffn_mixes.weight") {
		t.Fatal("isQuantWeight(model.layers.0.mhc.ffn_mixes.weight) = false, want true")
	}
	if isQuantWeight("model.layers.0.attn_norm.weight") {
		t.Fatal("isQuantWeight(attn_norm.weight) = true, want false (a 1-D norm is never a quant matmul weight)")
	}
	if !isQuantWeight("model.layers.0.attn.wq_a.weight") {
		t.Fatal("isQuantWeight(attn.wq_a.weight) = false, want true (existing V4.1 dense leaf unchanged)")
	}
}
