package model

// v41_attention_storage_fixture_test.go — test-only fixtures for the #13321
// V4.1 attention projection admission/storage conformance matrix.
//
// Why a dedicated config. Q2_K (kindQ2K) is a 256-wide super-block format:
// quantizeKQuantFromRaw panics unless the reduction dim in % 256 == 0. The
// reduced forward fixture (v41TestReducedConfig) narrows H=64 / QLoraRank=32 /
// oDim=32, none of which is a multiple of 256, so a Q2_K store cannot be built
// on it. This fixture instead keeps the full V4.1 identity (the parsed
// DeepSeekV41 pointer survives) but widens every axis the five attention
// projections' REDUCTION dim runs through to a legal Q2_K multiple of 256:
//
//	wq_a  [QLoraRank, H]        in = H          = 256
//	wq_b  [NumHeads*HeadDim, QLoraRank] in = QLoraRank = 256
//	wkv   [HeadDim, H]          in = H          = 256  (v41KVLoraRankReduced)
//	wo_a  [OLoraRank, NumHeads*HeadDim] (flat) in = NumHeads*HeadDim = 256
//	wo_b  [H, OGroups*OLoraRank] in = oDim      = 256
//
// It is deliberately NOT the reduced forward fixture: it exists only to drive
// the admission predicate and the projection read/contraction through a
// resident Q2_K store and a raw-f32 control, with an independent expectation.
//
// Independence discipline. The Q2_K expectation below is a LOCAL scalar
// transcription of the container's documented layout (84-byte super-block: 16
// scale bytes, 64 two-bit code bytes, f16 d, f16 min; weight = code*(d*(sc&0xf))
// - min*(sc>>4)). It reuses NONE of the production decode path
// (kQuantDequantSuperBlock / q2kDequantSuperBlock / kQuantMatRows) — those are
// the thing under test.

import (
	"encoding/binary"
	"math"
	"testing"
)

// v41StorageConfig returns a V4.1-identity Config whose five attention
// projection reduction dims are all legal Q2_K multiples of 256. It derives
// from the real parsed V4.1 config so IsDeepSeekV41 stays true and the
// DeepSeekV41 metadata pointer (attention geometry, etc.) is retained.
func v41StorageConfig(t *testing.T) Config {
	t.Helper()
	_, cfg := readDeepSeekV41Config(t)
	cfg.NumLayers = 1
	cfg.HiddenSize = 256
	cfg.NumHeads = 2
	cfg.HeadDim = 128
	// MLA split consistent with the widened head: nope 64 + rope 64 = 128.
	cfg.QKNopeHeadDim = 64
	cfg.QKRopeHeadDim = 64
	cfg.QLoraRank = 256
	cfg.OLoraRank = 128
	cfg.OGroups = 2
	cfg.MoEIntermediateSize = 256
	cfg.VocabSize = 8
	if !cfg.IsDeepSeekV41() || cfg.DeepSeekV41 == nil {
		t.Fatalf("storage config lost V4.1 identity")
	}
	return cfg
}

// v41StorageAttentionLeaf names the five attention projections the matrix
// covers, in the reference order wq_a, wq_b, wkv, wo_a, wo_b.
type v41StorageLeaf struct {
	leaf string
	// out, in are the FLAT logical [out, in] the forward consumes.
	out, in int
	// grouped is true for wo_a, whose admission runs through
	// v41AdmitGroupedWoA (the #13264 group-major/flat two-form seam).
	grouped bool
}

// v41StorageLeaves returns the five-operand matrix for cfg.
func v41StorageLeaves(cfg Config) []v41StorageLeaf {
	qHeadDim := cfg.NumHeads * cfg.HeadDim
	oDim := cfg.OGroups * cfg.OLoraRank
	return []v41StorageLeaf{
		{leaf: "attn.wq_a.weight", out: cfg.QLoraRank, in: cfg.HiddenSize},
		{leaf: "attn.wq_b.weight", out: qHeadDim, in: cfg.QLoraRank},
		{leaf: "attn.wkv.weight", out: v41KVLoraRankReduced(cfg), in: cfg.HiddenSize},
		{leaf: "attn.wo_a.weight", out: cfg.OLoraRank, in: qHeadDim, grouped: true},
		{leaf: "attn.wo_b.weight", out: cfg.HiddenSize, in: oDim},
	}
}

// v41StorageBuildModel builds a *Model carrying an f32 manifest entry for every
// attention projection (the raw-f32 control form), so the SAME model can be
// driven through the manifest arm and, after moving a projection into m.kqw,
// through the resident Q2_K arm. Only the per-layer weight tensors the matrix
// reads are materialized; no forward is run.
func v41StorageBuildModel(cfg Config) *Model {
	H := cfg.HiddenSize
	qHeadDim := cfg.NumHeads * cfg.HeadDim
	oDim := cfg.OGroups * cfg.OLoraRank
	tensors := []synthTensor{
		{layerName(0, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
		{layerName(0, "attn.wq_b.weight"), []int{qHeadDim, cfg.QLoraRank}},
		{layerName(0, "attn.wkv.weight"), []int{v41KVLoraRankReduced(cfg), H}},
		{layerName(0, "attn.wo_a.weight"), []int{cfg.OLoraRank, qHeadDim}},
		{layerName(0, "attn.wo_b.weight"), []int{H, oDim}},
	}
	man, raw := synthBuildRaw(tensors, synthMatmulFill)
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

// v41StorageAnalyticBlock builds one 84-byte Q2_K super-block by EXPLICIT
// construction, returning the block and the 256 exact f32 weights it encodes.
// It does not call any production quantizer or dequantizer.
//
// Layout (matching the Q2_K container the resident reader consumes): 16 scale
// bytes (low nibble = per-group d-scale nibble, high nibble = per-group min
// nibble), 64 code bytes (four 2-bit codes per byte, low code first; the
// 256 codes are laid out 32 per group, groups of 32 in group order), then the
// f16 super-scale d at [80:82] and the f16 super-min at [82:84].
//
// The group/scale indexing mirrors the container's scalar decode: for each of
// the 8 groups of 32 weights, group g uses scale byte scales[g]; the first 16
// of the group read code bytes q[g*8 + l] and the second 16 read
// q[g*8 + 8 + l] at bit offset 2*(g%4)... the exact scalar walk is restated in
// v41StorageScalarExpect below.
func v41StorageAnalyticBlock(d, minVal float32, scNibbles [8]byte, codeBytes [64]byte) ([]byte, []float32) {
	blk := make([]byte, q2kBlockBytes)
	for g := 0; g < 8; g++ {
		blk[g] = scNibbles[g]
	}
	copy(blk[16:16+64], codeBytes[:])
	putF16LE(blk[80:], d)
	putF16LE(blk[82:], minVal)
	want := v41StorageScalarExpect(blk)
	return blk, want
}

// v41StorageScalarExpect is the independent scalar transcription of the Q2_K
// super-block semantics. It is a deliberate re-statement of the documented
// layout, NOT a call into the production decoder.
func v41StorageScalarExpect(blk []byte) []float32 {
	scales := blk[:16]
	q := blk[16:80]
	d := f16ToF32(binary.LittleEndian.Uint16(blk[80:]))
	min := f16ToF32(binary.LittleEndian.Uint16(blk[82:]))
	out := make([]float32, 256)
	qi, is := 0, 0
	for n := 0; n < 256; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			sc := scales[is]
			is++
			dl, ml := d*float32(sc&0x0f), min*float32(sc>>4)
			tab := [4]float32{0 - ml, dl - ml, dl*2 - ml, dl*3 - ml}
			for l := 0; l < 16; l++ {
				out[n+j*32+l] = tab[(q[qi+l]>>shift)&3]
			}
			sc = scales[is]
			is++
			dl, ml = d*float32(sc&0x0f), min*float32(sc>>4)
			tab = [4]float32{0 - ml, dl - ml, dl*2 - ml, dl*3 - ml}
			for l := 0; l < 16; l++ {
				out[n+j*32+16+l] = tab[(q[qi+16+l]>>shift)&3]
			}
			shift += 2
		}
		qi += 32
	}
	return out
}

// v41StorageQ2KRaw builds a deterministic Q2_K payload for an [out, in] matrix
// (in % 256 == 0) whose every super-block is the SAME analytically constructed
// block, and returns the payload plus the [out, in] exact expected weights.
func v41StorageQ2KRaw(out, in int) ([]byte, []float32) {
	nblk := in / 256
	// Varied-but-exact nibbles/codes so a scale- or code-indexing bug cannot
	// cancel. d = 0.5 and min = 0.25 are f16-exact; scale nibbles are small
	// integers so dl/min terms stay exactly representable in f32 sums.
	var scNibbles [8]byte
	var codeBytes [64]byte
	for g := 0; g < 8; g++ {
		scNibbles[g] = byte((g%3)+1) | byte((g%2)<<4) // low: d-nibble 1..3, high: min-nibble 0/1
	}
	for i := range codeBytes {
		codeBytes[i] = byte((i*37 + 11) % 256)
	}
	blk, wantBlock := v41StorageAnalyticBlock(0.5, 0.25, scNibbles, codeBytes)

	raw := make([]byte, out*nblk*len(blk))
	want := make([]float32, out*in)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			copy(raw[(o*nblk+b)*len(blk):], blk)
			copy(want[o*in+b*256:], wantBlock)
		}
	}
	return raw, want
}

// v41StorageInstallQ2K moves a named projection out of the f32 manifest and
// into the resident k-quant store as a Q2_K tensor built from the analytic raw
// payload. The name is then resolvable ONLY through the resident store.
func v41StorageInstallQ2K(m *Model, l int, leaf string, out, in int) []float32 {
	name := layerName(l, leaf)
	delete(m.manifest, name)
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	raw, want := v41StorageQ2KRaw(out, in)
	m.kqw[name] = quantizeKQuantFromRaw(raw, out, in, kindQ2K)
	return want
}

// putF16LE writes v as a little-endian f16 at dst. Only values exactly
// representable as f16 are used by the fixtures (0.5, 0.25, 1.0, ...).
func putF16LE(dst []byte, v float32) {
	binary.LittleEndian.PutUint16(dst, f32ToF16Exact(v))
}

// f32ToF16Exact encodes a float32 that is exactly representable in f16.
func f32ToF16Exact(v float32) uint16 {
	bits := math.Float32bits(v)
	sign := uint16(bits>>16) & 0x8000
	exp := int((bits >> 23) & 0xff)
	mant := bits & 0x7fffff
	if v == 0 {
		return sign
	}
	// Bias conversion f32(127) -> f16(15).
	newexp := exp - 127 + 15
	if newexp <= 0 || newexp >= 31 || mant&0x1fff != 0 {
		panic("f32ToF16Exact: value not exactly representable in f16")
	}
	return sign | uint16(newexp<<10) | uint16(mant>>13)
}

// f16ToF32 decodes a little-endian f16 already read as uint16 to float32 using
// the package's shared bit conversion (not the production super-block decoder),
// so the expectation is independent of the Q2_K kernel under test.
func f16ToF32(h uint16) float32 {
	return math.Float32frombits(F16BitsToF32Bits(h))
}
