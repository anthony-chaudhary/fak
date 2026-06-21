package model

import (
	"math"
	"testing"
)

func TestRefactorMatchesSerial(t *testing.T) {
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
	m := NewSynthetic(cfg)

	t.Run("f32 prefill and decode", func(t *testing.T) {
		prompt := []int{3, 17, 5, 23, 41, 2, 19}
		cur := m.NewSession()
		legacy := m.NewSession()

		curLogits := cur.Prefill(prompt)
		legacyLogits := legacyPrefillF32ForTest(legacy, prompt)
		assertFloat32BitsEqual(t, "prefill logits", legacyLogits, curLogits)
		assertKVCacheBitsEqual(t, "prefill", legacy.Cache, cur.Cache)

		for step, id := range []int{11, 29, 7} {
			curLogits = cur.Step(id)
			legacyX := legacyTokenHiddenF32ForTest(legacy, id, legacy.Cache.Len())
			legacyLogits = legacy.head(legacyX)
			assertFloat32BitsEqual(t, "decode logits", legacyLogits, curLogits)
			assertKVCacheBitsEqual(t, "decode step "+itoa(step), legacy.Cache, cur.Cache)
		}
	})

	t.Run("f32 no-cache forward (R1 oracle block)", func(t *testing.T) {
		// The cacheless full-prefill Forward path (forward.go:layer → attnSeq/mlpSeq/
		// composeSeqSublayer, the R1 oracle rung) is the one decoder-block transcription
		// the f32 prefill/decode subtests above do NOT cover: it routes its matmuls through
		// the residentKernel (matRows) instead of the cached blockStep (parMatRows). matRows
		// and parMatRows share ONE fdot reduction order by construction (forward.go:matRows
		// doc), so Forward must reproduce the legacy per-token block to the last bit. The
		// existing R2 witness of this (TestCachedDecodeMatchesPrefill, Forward==Prefill) is
		// gated on the ~538MB exported oracle and t.Skips without it; assert it here on the
		// weight-free synthetic so the SEAM-0 refactor's no-cache block stays bit-locked in
		// the fast, artifact-free path too.
		prompt := []int{3, 17, 5, 23, 41, 2, 19}
		act := m.Forward(prompt)
		legacy := m.NewSession()
		for i, id := range prompt {
			legacyX := legacyTokenHiddenF32ForTest(legacy, id, legacy.Cache.Len())
			assertFloat32BitsEqual(t, "forward logits pos "+itoa(i), legacy.head(legacyX), act.Logits[i])
		}
	})

	t.Run("q8 decode matches legacy hand-copy within quant gate", func(t *testing.T) {
		// Q8 arithmetic is lossy by design, and the optimized decode path may use a different
		// accumulation order than the old hand-copy. Keep cache positions exact, but gate Q8
		// float buffers the same way the quantized path is validated elsewhere: tight numeric
		// tolerance, cosine, and argmax.
		mq := NewSynthetic(cfg)
		mq.Quantize()
		prompt := []int{3, 17, 5, 23, 41, 2, 19}

		cur := mq.NewSession()
		cur.Quant = true
		legacy := mq.NewSession()
		legacy.Quant = true

		// prefillBatchedQ is a panel path NOT folded here, so drive the folded decode block
		// directly per token on both sides: cur via blockStep+q8Kernel, legacy via the copy.
		var curX, legX []float32
		for _, id := range prompt {
			curX = cur.tokenHiddenQ(id, cur.Cache.Len())
			legX = legacyTokenHiddenQForTest(legacy, id, legacy.Cache.Len())
		}
		assertCosineAtLeast(t, "q8 prefill hidden", legX, curX, 0.999999)
		assertKVCacheQuantClose(t, "q8 prefill", legacy.Cache, cur.Cache)
		assertQuantLogitsClose(t, "q8 prefill logits", legacy.headQ(legX), cur.headQ(curX))

		for step, id := range []int{11, 29, 7} {
			curX = cur.tokenHiddenQ(id, cur.Cache.Len())
			legX = legacyTokenHiddenQForTest(legacy, id, legacy.Cache.Len())
			assertQuantLogitsClose(t, "q8 decode logits", legacy.headQ(legX), cur.headQ(curX))
			assertKVCacheQuantClose(t, "q8 decode step "+itoa(step), legacy.Cache, cur.Cache)
		}
	})

	t.Run("multi-user prefill and decode", func(t *testing.T) {
		prompts := [][]int{
			{1, 5, 9, 13},
			{2, 4, 8, 16, 32},
			{3, 6, 12, 24, 48, 7},
		}
		bs := m.NewBatchSession(len(prompts))
		curLogits := bs.PrefillEach(prompts)
		legacy := make([]*Session, len(prompts))
		for b, prompt := range prompts {
			legacy[b] = m.NewSession()
			want := legacyPrefillF32ForTest(legacy[b], prompt)
			assertFloat32BitsEqual(t, "batch prefill logits", want, curLogits[b])
			assertKVCacheBitsEqual(t, "batch prefill user "+itoa(b), legacy[b].Cache, bs.Seqs[b].Cache)
		}

		for step, ids := range [][]int{{10, 20, 30}, {11, 21, 31}} {
			curLogits = bs.StepBatch(ids)
			for b, id := range ids {
				legacyX := legacyTokenHiddenF32ForTest(legacy[b], id, legacy[b].Cache.Len())
				want := legacy[b].head(legacyX)
				assertFloat32BitsEqual(t, "batch decode logits", want, curLogits[b])
				assertKVCacheBitsEqual(t, "batch decode step "+itoa(step)+" user "+itoa(b), legacy[b].Cache, bs.Seqs[b].Cache)
			}
		}
	})
}

func legacyPrefillF32ForTest(s *Session, ids []int) []float32 {
	var xf []float32
	for _, id := range ids {
		xf = legacyTokenHiddenF32ForTest(s, id, s.Cache.Len())
	}
	return s.head(xf)
}

// legacyTokenHiddenF32ForTest is the pre-blockStep f32 token loop, kept only as a
// regression oracle for the SEAM-0 refactor. It intentionally does not call
// blockStep, prefillBatched, or StepBatch.
func legacyTokenHiddenF32ForTest(s *Session, id, pos int) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	cos, sin := ropeRow(cfg, pos)

	embed := m.embedRows()
	x := append([]float32(nil), embed[id*H:(id+1)*H]...)

	for l := 0; l < cfg.NumLayers; l++ {
		p := func(str string) string { return layerName(l, str) }
		xn := rmsnorm(x, m.tensor(p("input_layernorm.weight")), eps)
		q := parMatRows(m.tensor(p("self_attn.q_proj.weight")), xn, nH*hd, H)
		kk := parMatRows(m.tensor(p("self_attn.k_proj.weight")), xn, w, H)
		vv := parMatRows(m.tensor(p("self_attn.v_proj.weight")), xn, w, H)
		if cfg.AttentionBias {
			addBias(q, m.tensor(p("self_attn.q_proj.bias")))
			addBias(kk, m.tensor(p("self_attn.k_proj.bias")))
			addBias(vv, m.tensor(p("self_attn.v_proj.bias")))
		}
		for h := 0; h < nH; h++ {
			applyRopeRow(q[h*hd:(h+1)*hd], cos, sin)
		}
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], kk...)
		for h := 0; h < nKV; h++ {
			applyRopeRow(kk[h*hd:(h+1)*hd], cos, sin)
		}
		s.Cache.K[l] = append(s.Cache.K[l], kk...)
		s.Cache.V[l] = append(s.Cache.V[l], vv...)

		nPos := len(s.Cache.K[l]) / w
		attnOut := make([]float32, nH*hd)
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[h*hd : (h+1)*hd]
			scores := make([]float32, nPos)
			for j := 0; j < nPos; j++ {
				kh := s.Cache.K[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				scores[j] = dot(qh, kh) * scale
			}
			softmaxInPlace(scores)
			out := attnOut[h*hd : (h+1)*hd]
			for j := 0; j < nPos; j++ {
				vh := s.Cache.V[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				wj := scores[j]
				for d := 0; d < hd; d++ {
					out[d] += wj * vh[d]
				}
			}
		}
		o := parMatRows(m.tensor(p("self_attn.o_proj.weight")), attnOut, H, nH*hd)
		for i := 0; i < H; i++ {
			x[i] += o[i]
		}
		xn2 := rmsnorm(x, m.tensor(p("post_attention_layernorm.weight")), eps)
		I := cfg.IntermediateSize
		g := parMatRows(m.tensor(p("mlp.gate_proj.weight")), xn2, I, H)
		u := parMatRows(m.tensor(p("mlp.up_proj.weight")), xn2, I, H)
		for i := 0; i < I; i++ {
			g[i] = silu(g[i]) * u[i]
		}
		down := parMatRows(m.tensor(p("mlp.down_proj.weight")), g, H, I)
		for i := 0; i < H; i++ {
			x[i] += down[i]
		}
	}
	s.Cache.pos = append(s.Cache.pos, pos)
	return rmsnorm(x, m.tensor("model.norm.weight"), eps)
}

// legacyTokenHiddenQForTest is the pre-blockStep Q8_0 decode loop, kept only as a
// regression oracle for the SEAM-0 Q8 fold. It is a verbatim copy of the hand-copied
// tokenHiddenQ body as it stood before being collapsed onto blockStep+q8Kernel, and it
// deliberately does NOT call blockStep.
func legacyTokenHiddenQForTest(s *Session, id, pos int) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	cos, sin := ropeRow(cfg, pos)

	embed := m.embedRows()
	x := append([]float32(nil), embed[id*H:(id+1)*H]...)

	for l := 0; l < cfg.NumLayers; l++ {
		p := func(str string) string { return layerName(l, str) }
		xn := quantizeVecQ8(rmsnorm(x, m.tensor(p("input_layernorm.weight")), eps))
		q := qMatRows(m.q8(p("self_attn.q_proj.weight")), xn)
		kk := qMatRows(m.q8(p("self_attn.k_proj.weight")), xn)
		vv := qMatRows(m.q8(p("self_attn.v_proj.weight")), xn)
		if cfg.AttentionBias {
			addBias(q, m.tensor(p("self_attn.q_proj.bias")))
			addBias(kk, m.tensor(p("self_attn.k_proj.bias")))
			addBias(vv, m.tensor(p("self_attn.v_proj.bias")))
		}
		for h := 0; h < nH; h++ {
			applyRopeRow(q[h*hd:(h+1)*hd], cos, sin)
		}
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], kk...)
		for h := 0; h < nKV; h++ {
			applyRopeRow(kk[h*hd:(h+1)*hd], cos, sin)
		}
		s.Cache.K[l] = append(s.Cache.K[l], kk...)
		s.Cache.V[l] = append(s.Cache.V[l], vv...)

		nPos := len(s.Cache.K[l]) / w
		attnOut := make([]float32, nH*hd)
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[h*hd : (h+1)*hd]
			scores := make([]float32, nPos)
			for j := 0; j < nPos; j++ {
				kh := s.Cache.K[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				scores[j] = dot(qh, kh) * scale
			}
			softmaxInPlace(scores)
			out := attnOut[h*hd : (h+1)*hd]
			for j := 0; j < nPos; j++ {
				vh := s.Cache.V[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				wj := scores[j]
				for d := 0; d < hd; d++ {
					out[d] += wj * vh[d]
				}
			}
		}
		o := qMatRows(m.q8(p("self_attn.o_proj.weight")), quantizeVecQ8(attnOut))
		for i := 0; i < H; i++ {
			x[i] += o[i]
		}
		xn2 := quantizeVecQ8(rmsnorm(x, m.tensor(p("post_attention_layernorm.weight")), eps))
		I := cfg.IntermediateSize
		g := qMatRows(m.q8(p("mlp.gate_proj.weight")), xn2)
		u := qMatRows(m.q8(p("mlp.up_proj.weight")), xn2)
		for i := 0; i < I; i++ {
			g[i] = silu(g[i]) * u[i]
		}
		down := qMatRows(m.q8(p("mlp.down_proj.weight")), quantizeVecQ8(g))
		for i := 0; i < H; i++ {
			x[i] += down[i]
		}
	}
	s.Cache.pos = append(s.Cache.pos, pos)
	return rmsnorm(x, m.tensor("model.norm.weight"), eps)
}

func assertKVCacheBitsEqual(t *testing.T, label string, want, got *KVCache) {
	t.Helper()
	if want.Len() != got.Len() {
		t.Fatalf("%s cache len = %d, want %d", label, got.Len(), want.Len())
	}
	for i := range want.pos {
		if want.pos[i] != got.pos[i] {
			t.Fatalf("%s pos[%d] = %d, want %d", label, i, got.pos[i], want.pos[i])
		}
	}
	for l := 0; l < want.cfg.NumLayers; l++ {
		assertFloat32BitsEqual(t, label+" K layer "+itoa(l), want.K[l], got.K[l])
		assertFloat32BitsEqual(t, label+" Kraw layer "+itoa(l), want.Kraw[l], got.Kraw[l])
		assertFloat32BitsEqual(t, label+" V layer "+itoa(l), want.V[l], got.V[l])
	}
}

func assertKVCacheQuantClose(t *testing.T, label string, want, got *KVCache) {
	t.Helper()
	assertKVCacheQuantCloseTol(t, label, want, got, 1e-5, 1e-5)
}

// assertKVCacheQuantCloseTol is assertKVCacheQuantClose with explicit K/Kraw and V
// tolerances. The two split because a quant path can legitimately carry more drift on the
// raw V store than on K/Kraw: K and Kraw pass through the per-head RMSNorm of
// applyLayerQKNorm, which is contractive and damps any propagated upstream rounding, while V
// is cached verbatim straight from the projection and so reflects the full propagated drift.
// The default helper keeps both at the strict 1e-5 f32 bound; callers that route a Q8-minority
// projection through the batched prefill GEMM (whose deferred reduction is not bit-identical
// to the decode GEMV — see parallel.go on fdot non-associativity) pass a wider V bound. See
// TestPrefillQwen35HybridQ4KMatchesTokenLoop for the worked justification.
func assertKVCacheQuantCloseTol(t *testing.T, label string, want, got *KVCache, kTol, vTol float64) {
	t.Helper()
	if want.Len() != got.Len() {
		t.Fatalf("%s cache len = %d, want %d", label, got.Len(), want.Len())
	}
	for i := range want.pos {
		if want.pos[i] != got.pos[i] {
			t.Fatalf("%s pos[%d] = %d, want %d", label, i, got.pos[i], want.pos[i])
		}
	}
	for l := 0; l < want.cfg.NumLayers; l++ {
		assertMaxAbsAtMost(t, label+" K layer "+itoa(l), want.K[l], got.K[l], kTol)
		assertMaxAbsAtMost(t, label+" Kraw layer "+itoa(l), want.Kraw[l], got.Kraw[l], kTol)
		assertMaxAbsAtMost(t, label+" V layer "+itoa(l), want.V[l], got.V[l], vTol)
	}
}

func assertQuantLogitsClose(t *testing.T, label string, want, got []float32) {
	t.Helper()
	assertCosineAtLeast(t, label, want, got, 0.999999)
	if argmax(want) != argmax(got) {
		t.Fatalf("%s argmax = %d, want %d", label, argmax(got), argmax(want))
	}
}

func assertCosineAtLeast(t *testing.T, label string, want, got []float32, min float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	cs := cosine(want, got)
	if cs < min {
		t.Fatalf("%s cosine %.9f < %.9f", label, cs, min)
	}
}

func assertMaxAbsAtMost(t *testing.T, label string, want, got []float32, max float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	d, at := maxAbsDiff(want, got)
	if d > max {
		t.Fatalf("%s max abs diff %.9g at %d > %.9g", label, d, at, max)
	}
}
