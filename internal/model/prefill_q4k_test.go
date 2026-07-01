package model

import (
	"math"
	"math/rand"
	"testing"
)

// fillQ4KW populates m.q4kw for every named [out,in] projection with random-but-finite
// Q4_K super-blocks whose d/min f16 scales are pinned to a SMALL magnitude so the
// dequanted weights stay in the same ~0.1 band NewSynthetic uses (unconstrained random
// blocks hit f16 exponent 30 and produce ~1e7 weights that blow the forward to Inf/NaN).
// The values are NOT derived from the f32 manifest — they are arbitrary valid Q4_K blocks.
// This is sufficient for a parity test between two paths that read the SAME q4kw store
// (batched q4kGemm vs per-token q4kMatRows): both consume identical weight bytes, so the
// comparison isolates the reduction-order / dispatch difference, not the quantization.
func fillQ4KW(t *testing.T, m *Model, names [][2]any, seed int64) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	if m.q4kw == nil {
		m.q4kw = make(map[string]*q4kTensor, len(names))
	}
	blk := make([]byte, q4kBlockBytes)
	for _, nm := range names {
		name := nm[0].(string)
		out := nm[1].(int)
		// `in` is read off the manifest shape (the f32 tensor's reduction dim), which the
		// resident q4kTensor must match exactly.
		meta, ok := m.manifest[name]
		if !ok {
			t.Fatalf("fillQ4KW: tensor %s missing from manifest", name)
		}
		in := meta.Shape[len(meta.Shape)-1]
		if in%qkK != 0 {
			t.Fatalf("fillQ4KW: %s reduction dim %d not a multiple of %d (Q4_K precondition)", name, in, qkK)
		}
		nblk := in / qkK
		raw := make([]byte, out*nblk*q4kBlockBytes)
		for i := 0; i < out*nblk; i++ {
			randQ4KBlockBounded(rng, blk, 2, 6) // d/min magnitude ~1e-4..4e-4 → weights ~0.1
			copy(raw[i*q4kBlockBytes:(i+1)*q4kBlockBytes], blk)
		}
		m.q4kw[name] = quantizeQ4KFromRaw(raw, out, in)
	}
}

// randQ4KBlockBounded is randQ4KBlock with the two d/min f16 exponents forced into
// [expMin,expMax] (biased exponent = exp+15), so the dequanted weights land in a chosen
// magnitude band instead of the full f16 range. Used only by the prefill parity test.
func randQ4KBlockBounded(rng *rand.Rand, blk []byte, expMin, expMax int) {
	for i := range blk {
		blk[i] = byte(rng.Intn(256))
	}
	if expMin < 1 {
		expMin = 1
	}
	if expMax > 30 {
		expMax = 30
	}
	for s := 0; s < 2; s++ {
		hi := blk[s*2+1]
		exp := expMin + rng.Intn(expMax-expMin+1)
		blk[s*2+1] = (hi & 0x83) | byte(exp<<2)
	}
}

// TestQ4KGemmMatchesMatRows pins the batched Q4_K GEMM to the decode GEMV: for every
// (output row o, token t), q4kGemm[t*out+o] must be BIT-IDENTICAL to q4kMatRows(qt,
// X[t])[o]. This is the load-bearing correctness contract for the resident-Q4_K prefill
// (QWEN36-NATIVE-PERF-PLAN P3) — it proves the batched path's per-super-block dequant +
// 4-accumulator dot + super-block ordering exactly matches the proven decode GEMV, so the
// only change is compute amortization (each super-block dequantized once, reused across P
// tokens), not arithmetic. A real dispatch/dequant/packing bug blows this up by orders of
// magnitude; the comparison is exact (same arithmetic, only loop nesting differs).
func TestQ4KGemmMatchesMatRows(t *testing.T) {
	// q4kGemm now dispatches int8-vs-f32 on q4kSDOTEnabled (mirroring q4kMatRowsInto); this test
	// pins the F32 reduction order vs the f32 GEMV, so force the f32 path. The int8 GEMM is held
	// to the int8 GEMV separately by TestQ4KGemmInt8MatchesMatRowsInt8.
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	const out, in, P = 32, 768, 8 // in = 3 super-blocks/row; P prompt tokens
	nblk := in / qkK
	rng := rand.New(rand.NewSource(11))
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for i := 0; i < out*nblk; i++ {
		randQ4KBlock(rng, blk)
		copy(raw[i*q4kBlockBytes:(i+1)*q4kBlockBytes], blk)
	}
	qt := quantizeQ4KFromRaw(raw, out, in)

	X := make([]float32, P*in)
	for i := range X {
		X[i] = float32(rng.NormFloat64())
	}
	Y := q4kGemm(qt, X, P)

	for tr := 0; tr < P; tr++ {
		yRef := make([]float32, out)
		// f32 scalar GEMV (the int8 SDOT decode path has its own tolerance gate; this test
		// pins q4kGemm's f32 reduction order vs the f32 serial path, so both must be f32).
		q4kMatRowsRange(qt, X[tr*in:(tr+1)*in], yRef, 0, out)
		for o := 0; o < out; o++ {
			got := Y[tr*out+o]
			if math.Float32bits(got) != math.Float32bits(yRef[o]) {
				t.Fatalf("token %d row %d: q4kGemm %v != q4kMatRows %v (NOT bit-identical)", tr, o, got, yRef[o])
			}
		}
	}
}

// TestQ4KGemmInt8ExtractOnceMatchesMatRowsInt8 pins the extract-once int8 batched GEMM to the
// per-token int8 decode GEMV within reduction-order tolerance. The new path (#60) lowers Q4_K
// nibbles to a temporary Q8-shaped tensor and runs qGemm8, whose deferred-reduction order differs
// from q4kMatRowsRangeInt8's q4kCombineRow order. A real packing, scale, or affine-min bug still
// blows this relative error up by orders of magnitude.
func TestQ4KGemmInt8ExtractOnceMatchesMatRowsInt8(t *testing.T) {
	const out, in, P = 32, 768, 8 // in = 3 super-blocks/row; P prompt tokens
	nblk := in / qkK
	rng := rand.New(rand.NewSource(13))
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for i := 0; i < out*nblk; i++ {
		randQ4KBlock(rng, blk)
		copy(raw[i*q4kBlockBytes:(i+1)*q4kBlockBytes], blk)
	}
	qt := quantizeQ4KFromRaw(raw, out, in)

	X := make([]float32, P*in)
	for i := range X {
		X[i] = float32(rng.NormFloat64())
	}
	qp := quantizeBatchPanel(X, P, in)
	Y := make([]float32, P*out)
	q4kGemmExtractOnceInt8Into(qt, qp, Y)

	yRef := make([]float32, out)
	var sumSq, maxRel float64
	for tr := 0; tr < P; tr++ {
		// int8 decode GEMV reference: quantize this token's activation exactly as q4kGemm does.
		qv := quantizeVecQ8(X[tr*in : (tr+1)*in])
		q4kMatRowsRangeInt8(qt, qv, yRef, 0, out)
		for o := 0; o < out; o++ {
			want := yRef[o]
			got := Y[tr*out+o]
			sumSq += float64(want) * float64(want)
			if rel := math.Abs(float64(got - want)); rel > maxRel {
				maxRel = rel
			}
		}
	}
	rms := math.Sqrt(sumSq / float64(P*out))
	if rms < 1e-9 {
		t.Fatalf("int8 reference RMS ~0; bad test data")
	}
	if maxRel/rms > 1e-4 {
		t.Fatalf("q4kGemm extract-once int8 max-abs/RMS %.3e > 1e-4 (abs=%.3e rms=%.3e)", maxRel/rms, maxRel, rms)
	}
	t.Logf("q4kGemm extract-once int8 max-abs/RMS = %.3e", maxRel/rms)
}

// TestPrefillBatchedQ4KMatchesSerial proves the resident-Q4_K batched prefill
// (prefillBatchedQ4K) builds a KV cache consistent with the proven per-token Q4K decode
// path (tokenHiddenQ via sessionQ4KKernel). The q4_k_m projection majority contributes
// ZERO drift — q4kGemm is bit-identical to q4kMatRows per (o,t) (TestQ4KGemmMatchesMatRows)
// — so any residual difference is the SAME attention RMSNorm/RoPE reduction-order surface
// prefillBatchedQ already shares with tokenHiddenQ (fdot batched-attention scores vs the
// per-token scalar dot), held here by cosine + relative max-abs rather than bit-identity.
// A real dispatch/wiring bug (wrong name, wrong store, q4k/q8 mismatch) diverges O(1) per
// layer and blows past these bounds by orders of magnitude.
func TestPrefillBatchedQ4KMatchesSerial(t *testing.T) {
	// Force the f32 decode path for the per-token reference, so this test compares the batched
	// q4kGemm (f32) against the f32 serial GEMV — its original dispatch/wiring intent — rather
	// than the int8 SDOT decode path's activation-quantization noise (gated separately by
	// TestQ4KInt8DotMatchesF32). The decode default stays int8 in production; this is a test
	// scoping knob, restored on cleanup.
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	// All reduction dims (H, nH*hd, I) must be multiples of qkK=256 for Q4_K. AttnSoftcap
	// != 0 forces q8FastDecodeOK=false so the per-token reference routes through blockStep +
	// sessionQ4KKernel (the Q4K decode path), matching what prefillBatchedQ4K dispatches.
	cfg := Config{
		HiddenSize: 256, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
		IntermediateSize: 256, VocabSize: 64, RMSNormEps: 1e-6, AttnSoftcap: 50.0, RopeTheta: 10000.0,
	}
	m := NewSynthetic(cfg)

	// Populate q4kw for ALL 7 projections per layer. Both the batched and per-token paths
	// read this store; the f32 manifest still feeds embeddings / norms / final-norm.
	projs := [][2]any{}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		nHhd := cfg.NumHeads * cfg.HeadDim
		nKVhd := cfg.NumKVHeads * cfg.HeadDim
		projs = append(projs,
			[2]any{p + "self_attn.q_proj.weight", nHhd},
			[2]any{p + "self_attn.k_proj.weight", nKVhd},
			[2]any{p + "self_attn.v_proj.weight", nKVhd},
			[2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize},
			[2]any{p + "mlp.gate_proj.weight", cfg.IntermediateSize},
			[2]any{p + "mlp.up_proj.weight", cfg.IntermediateSize},
			[2]any{p + "mlp.down_proj.weight", cfg.HiddenSize},
		)
	}
	fillQ4KW(t, m, projs, 99)

	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*7 + 3) % cfg.VocabSize
	}

	// Per-token reference: the proven Q4K decode loop (blockStep + sessionQ4KKernel →
	// q4kMatRows for every projection).
	ref := m.NewSession()
	ref.Q4K = true
	var refHidden []float32
	for _, id := range prompt {
		refHidden = ref.tokenHiddenQ(id, ref.Cache.Len())
	}

	// Batched: prefillBatchedQ4K (q4kGemm for every projection).
	bat := m.NewSession()
	bat.Q4K = true
	batHidden := bat.prefillBatchedQ4K(prompt)

	// Last-token hidden: cosine (the standard hidden-state metric — absolute scale is
	// dominated by a few outlier dims) plus a loose max-abs. q4k projections are
	// bit-identical, so the only drift is the attention/norm reduction order.
	if cos := cosine(refHidden, batHidden); cos < 0.999 {
		t.Fatalf("last-hidden cosine %.6f < 0.999 (batched Q4K vs per-token Q4K)", cos)
	}
	mx, _ := maxAbsDiff(refHidden, batHidden)
	if mx > 1e-2 {
		t.Fatalf("last-hidden max-abs %.4e > 1e-2 (dispatch/wiring bug suspected)", mx)
	}
	t.Logf("last-hidden cosine=%.6f max-abs=%.3e", cosine(refHidden, batHidden), mx)

	// KV cache: same length + relative max-abs per layer. K/Kraw/V must all stay close;
	// the q4k projections feeding them are bit-identical, so drift is purely the attention
	// score reduction order propagating into which V rows get weighted.
	if bat.Cache.Len() != ref.Cache.Len() {
		t.Fatalf("cache len %d != %d", bat.Cache.Len(), ref.Cache.Len())
	}
	for l := 0; l < cfg.NumLayers; l++ {
		check := func(name string, a, b []float32) {
			t.Helper()
			if len(a) != len(b) {
				t.Fatalf("layer %d %s len %d != %d", l, name, len(a), len(b))
			}
			var refSq, mx float64
			for i := range a {
				d := float64(a[i] - b[i])
				if ad := math.Abs(d); ad > mx {
					mx = ad
				}
				refSq += float64(a[i]) * float64(a[i])
			}
			rms := math.Sqrt(refSq / float64(len(a)))
			if rms < 1e-12 {
				return // a ~zero vector; max-abs is the only meaningful bound
			}
			if mx/rms > 2e-3 {
				t.Fatalf("layer %d %s: max-abs/RMS %.3e > 2e-3 (abs=%.3e rms=%.3e)", l, name, mx/rms, mx, rms)
			}
		}
		check("K", ref.Cache.K[l], bat.Cache.K[l])
		check("Kraw", ref.Cache.Kraw[l], bat.Cache.Kraw[l])
		check("V", ref.Cache.V[l], bat.Cache.V[l])
	}
	for i := range ref.Cache.pos {
		if ref.Cache.pos[i] != bat.Cache.pos[i] {
			t.Fatalf("pos[%d] %d != %d", i, ref.Cache.pos[i], bat.Cache.pos[i])
		}
	}
}

// TestPrefillBatchedQ4KDeterministic confirms the batched Q4_K prefill is reproducible:
// the same prompt must yield byte-identical KV state across runs. This catches
// non-deterministic parallel-reduction bugs (e.g. a per-worker accumulator that escapes
// its range) that a single-run parity test would miss.
func TestPrefillBatchedQ4KDeterministic(t *testing.T) {
	cfg := Config{
		HiddenSize: 256, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
		IntermediateSize: 256, VocabSize: 64, RMSNormEps: 1e-6, AttnSoftcap: 50.0, RopeTheta: 10000.0,
	}
	m := NewSynthetic(cfg)
	projs := [][2]any{}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		nHhd := cfg.NumHeads * cfg.HeadDim
		nKVhd := cfg.NumKVHeads * cfg.HeadDim
		projs = append(projs,
			[2]any{p + "self_attn.q_proj.weight", nHhd},
			[2]any{p + "self_attn.k_proj.weight", nKVhd},
			[2]any{p + "self_attn.v_proj.weight", nKVhd},
			[2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize},
			[2]any{p + "mlp.gate_proj.weight", cfg.IntermediateSize},
			[2]any{p + "mlp.up_proj.weight", cfg.IntermediateSize},
			[2]any{p + "mlp.down_proj.weight", cfg.HiddenSize},
		)
	}
	fillQ4KW(t, m, projs, 99)

	prompt := make([]int, 12)
	for i := range prompt {
		prompt[i] = (i*5 + 1) % cfg.VocabSize
	}

	s1 := m.NewSession()
	s1.Q4K = true
	h1 := s1.prefillBatchedQ4K(prompt)
	s2 := m.NewSession()
	s2.Q4K = true
	h2 := s2.prefillBatchedQ4K(prompt)

	for i := range h1 {
		if math.Float32bits(h1[i]) != math.Float32bits(h2[i]) {
			t.Fatalf("hidden[%d]: run1 %v != run2 %v (non-deterministic)", i, h1[i], h2[i])
		}
	}
	for l := 0; l < cfg.NumLayers; l++ {
		for _, pair := range [][3]any{{"K", s1.Cache.K[l], s2.Cache.K[l]}, {"V", s1.Cache.V[l], s2.Cache.V[l]}} {
			name := pair[0].(string)
			a := pair[1].([]float32)
			b := pair[2].([]float32)
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					t.Fatalf("layer %d %s[%d]: run1 != run2 (non-deterministic)", l, name, i)
				}
			}
		}
	}
}

// TestQ4KGemmGroupDispatchDeclinesWithoutMetal is the default-path-untouched guard for the batched
// prefill GEMM group: a session with MetalQ4K unset (every non-Apple-Silicon build, and the default
// on Metal) MUST get nil from q4kGemmGroupDispatch, so the prefill caller falls back to the proven
// per-weight proj. This runs on every platform (no build tag): on the pure-Go build it exercises the
// metal_q4k_off.go stub; the Metal build's on-path also short-circuits on !s.MetalQ4K. It pins the
// contract that the group batching is opt-in and never perturbs the shipping CPU prefill.
func TestQ4KGemmGroupDispatchDeclinesWithoutMetal(t *testing.T) {
	m := &Model{}
	s := m.NewSession() // MetalQ4K defaults false
	if got := s.q4kGemmGroupDispatch([]string{"a", "b"}, make([]float32, 8), 2); got != nil {
		t.Fatalf("q4kGemmGroupDispatch without MetalQ4K = %v, want nil (default prefill path must be untouched)", got)
	}
}
