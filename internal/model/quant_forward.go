package model

// quant_forward.go — the quantized twins of tokenHidden (decode) and prefillBatched
// (prefill). They are deliberate structural copies of the f32 originals in kv.go /
// prefill_batch.go, differing in exactly one way: each weight matmul quantizes its
// activation to Q8_0 and calls the int8 kernel (qMatRows for decode / qGemm8 for the
// register-blocked prefill GEMM) against the prebuilt Q8_0 weight, instead of the f32 fdot
// kernels. Everything else — RMSNorm, RoPE,
// the GQA attention over the f32 KV cache, the residuals, the SwiGLU — is identical f32
// math. Keeping them as separate functions (rather than branching inside the hot f32
// loops) is what guarantees the proven f32 path is not perturbed by a single instruction.

import (
	"fmt"
	"os"
	"time"
)

// qprofOn enables coarse phase timing in prefillBatchedQ (FAK_QPROFILE=1), printed to
// stderr per prefill — used to locate the prefill bottleneck (GEMM vs attention vs quant).
var qprofOn = os.Getenv("FAK_QPROFILE") != ""

// headQ applies the Q8_0-quantized LM head to a post-final-norm hidden vector. The head
// (the 49,152×576 tied embedding, the single largest weight) is the biggest single
// beneficiary of quantization on the decode path.
func (s *Session) headQ(xf []float32) []float32 {
	y, t := s.headLogitsBuf()
	xq := s.quantizeVecQ8(xf)
	s.phaseEnd("q8_vector_quantize", t)
	t = s.phaseStart()
	qMatRowsInto(s.M.q8Head(), xq, y)
	logitScaleInPlace(y, s.M.Cfg) // Cohere/Gemma2; no-op for Llama
	s.phaseEnd("lm_head_q8", t)
	return y
}

// quantizeVecQ8 quantizes a decode/head-path activation through the session-owned scratch
// buffer (s.qScratch) instead of allocating a fresh q8Vec per projection. It changes only
// which buffer the quantizer writes into, never the quantized values, so it composes
// cleanly with the config-driven blockStep path used by tokenHiddenQ below.
func (s *Session) quantizeVecQ8(x []float32) q8Vec {
	return quantizeVecQ8Into(&s.qScratch, x)
}

type qDecodeBuf struct {
	X, Xn, Q, K, V, attn, O, Xn2, G, U, Down, Xnorm []float32
	cos, sin                                        []float32
	Logits                                          []float32
	scores                                          [][]float32
	caches                                          []*KVCache
}

// Reserve grows the KV cache plus quantized decode scratch for a known decode tail without
// changing the current sequence. It is the session-level counterpart to KVCache.Reserve:
// callers that know they will decode N more tokens can avoid both KV growth copies and
// per-step scratch growth in the hot loop.
func (s *Session) Reserve(extraPositions int) {
	maxPositions := extraPositions
	if s.Cache != nil {
		s.Cache.Reserve(extraPositions)
		maxPositions = s.Cache.Len() + extraPositions
	}
	if extraPositions <= 0 || !s.Quant || s.M == nil {
		return
	}
	s.reserveQDecode(maxPositions)
}

func (s *Session) reserveQDecode(maxPositions int) {
	cfg := s.M.Cfg
	if s.qDecode == nil {
		s.qDecode = &qDecodeBuf{}
	}
	db := s.qDecode
	H, hd, nH, nKV := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	w := nKV * hd
	I := cfg.IntermediateSize
	db.X = grow(db.X, H)
	db.Xn = grow(db.Xn, H)
	db.Q = grow(db.Q, nH*hd)
	db.K = grow(db.K, w)
	db.V = grow(db.V, w)
	db.attn = grow(db.attn, nH*hd)
	db.O = grow(db.O, H)
	db.Xn2 = grow(db.Xn2, H)
	db.G = grow(db.G, I)
	db.U = grow(db.U, I)
	db.Down = grow(db.Down, H)
	db.Xnorm = grow(db.Xnorm, H)
	db.Logits = grow(db.Logits, cfg.VocabSize)
	db.cos = grow(db.cos, hd/2)
	db.sin = grow(db.sin, hd/2)

	maxWidth := H
	if I > maxWidth {
		maxWidth = I
	}
	nblk := maxWidth / qBlk
	if cap(s.qScratch.q) < maxWidth {
		s.qScratch.q = make([]int8, maxWidth)
	} else {
		s.qScratch.q = s.qScratch.q[:maxWidth]
	}
	if q8PreextendVec() {
		if cap(s.qScratch.q16) < maxWidth {
			s.qScratch.q16 = make([]int16, maxWidth)
		} else {
			s.qScratch.q16 = s.qScratch.q16[:maxWidth]
		}
	} else {
		s.qScratch.q16 = s.qScratch.q16[:0]
	}
	if cap(s.qScratch.d) < nblk {
		s.qScratch.d = make([]float32, nblk)
	} else {
		s.qScratch.d = s.qScratch.d[:nblk]
	}
	s.qScratch.nblk = nblk

	grp := cfg.GroupSize()
	rows := grp
	if numWorkers > 1 {
		nw := numWorkers
		if nw > nKV {
			nw = nKV
		}
		if nw < 1 {
			nw = 1
		}
		rows = nw * grp
	}
	db.scores = grow2D(db.scores, rows, maxPositions)
	db.caches = growCaches(db.caches, 1)
}

func q8FastPreNormOK(cfg Config) bool {
	if cfg.BlockTopology != PreNorm ||
		cfg.IsMoE() ||
		cfg.DenseMLP ||
		cfg.Alibi ||
		cfg.IsQwen35Hybrid() ||
		cfg.AttnOutputGate ||
		cfg.AttnSoftcap != 0 ||
		cfg.NormGain1p ||
		cfg.LayerNorm ||
		cfg.ropeAttentionFactor() != 1 ||
		cfg.hasLayerSpecificRopeTheta() {
		return false
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.windowForLayer(l) >= 0 {
			return false
		}
	}
	return true
}

func q8FastDecodeOK(cfg Config) bool {
	return q8FastPreNormOK(cfg)
}

func q8FastDecodeSessionOK(s *Session, cfg Config) bool {
	if s != nil && (s.Q4 || s.Q4K) {
		return false
	}
	return q8FastDecodeOK(cfg)
}

// tokenHiddenQ is the Q8_0 decode path. It is now a thin shell over the shared
// single-position blockStep, selecting the Q8 kernel (q8Kernel): the block skeleton —
// RMSNorm, RoPE+Kraw stash, GQA over the f32 KV cache, residuals, SwiGLU — is the
// SAME code the f32 decode (tokenHidden) runs; only the weight matmuls are the int8
// GEMV qMatRows against the prebuilt Q8_0 weights, with the activation quantized once
// per (qkv / gate-up) group exactly as the prior hand-copy did. Appends this position's
// (f32) K/V to the kernel-owned cache, so Evict/Clone and the KV semantics are unchanged;
// returns the post-final-norm hidden (caller applies headQ).
func (s *Session) tokenHiddenQ(id, pos int) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	if !q8FastDecodeSessionOK(s, cfg) {
		mat := matKernel(sessionQ8Kernel{s})
		if s.Q4 && m.q4w != nil {
			// Resident int4 decode: the Qwen3.6 hybrid (and every non-fast-PreNorm arch)
			// runs the shared blockStep skeleton; swapping the kernel to int4 streams ~1.8×
			// fewer weight bytes/token than Q8, raising the decode ceiling. The block
			// orchestration (RMSNorm, RoPE, GQA, GDN recurrent scan, SwiGLU) is unchanged.
			mat = matKernel(sessionQ4Kernel{s})
		} else if s.Q4K && m.q4kw != nil {
			// Resident raw Q4_K decode (plan P1): same blockStep skeleton, but the q4_k_m
			// matmul majority streams at 0.5625 B/weight (raw GGUF bytes, no round-trip) and
			// the Q6_K minority (attn_qkv/ffn_down) falls back to the Q8 GEMV inside the kernel.
			mat = matKernel(sessionQ4KKernel{s})
		} else if cfg.IsMoE() {
			// MoE routing reuses the original prepared router/expert operand after nested
			// down-projection prep calls, so keep the allocation-safe kernel for that path.
			mat = q8Kernel{m}
		}
		embed := m.embedRows()
		x := append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x, cfg) // Gemma; no-op for Llama
		for l := 0; l < cfg.NumLayers; l++ {
			cos, sin := ropeRowForLayer(cfg, l, pos)
			x = s.blockStep(l, pos, x, cos, sin, mat)
		}
		s.Cache.pos = append(s.Cache.pos, pos)
		return rmsnormCfg(x, m.tensor("model.norm.weight"), eps, cfg)
	}
	w := nKV * hd
	scale := cfg.attnScale()
	if s.qDecode == nil {
		s.qDecode = &qDecodeBuf{}
	}
	db := s.qDecode
	cos := grow(db.cos, hd/2)
	sin := grow(db.sin, hd/2)
	db.cos, db.sin = cos, sin
	ropeRowInto(cos, sin, cachedInvFreq(cfg, 0), pos)

	embed := m.embedRows()
	x := grow(db.X, H)
	db.X = x
	copy(x, embed[id*H:(id+1)*H])
	scaleEmbedInPlace(x, cfg) // Gemma; no-op for Llama

	for l := 0; l < cfg.NumLayers; l++ {
		ql := m.q8Layer(l)
		// Phase brackets (q8_*) are opt-in coarse timing: phaseStart returns the zero Time
		// when no PhaseProfiler is attached and phaseEnd no-ops on it, so the proven decode
		// path pays only a nil-pointer check per phase when profiling is off, and not a single
		// instruction touches x/q/kk/etc — the Q8 decode numerics are byte-for-byte unchanged.
		// When ON, they split this fast path into the breakdown the decode-gap roofline needs
		// (qkv/attn/o/mlp), which tokenHiddenQ previously left entirely unattributed.
		tNorm := s.phaseStart()
		xnF := grow(db.Xn, H)
		db.Xn = xnF
		rmsnormInto(xnF, x, ql.inputNorm, eps)
		xn := s.quantizeVecQ8(xnF)
		s.phaseEnd("q8_norm_quant", tNorm)
		q := grow(db.Q, nH*hd)
		db.Q = q
		kk := grow(db.K, w)
		db.K = kk
		vv := grow(db.V, w)
		db.V = vv
		tQKV := s.phaseStart()
		// Grouped GEMV: q/k/v share xn under ONE parFor (k/v's 256 rows fold into q's sweep
		// instead of two poorly-parallelized tiny dispatches). Bit-identical to the three separate
		// qMatRowsInto calls (TestQMatRowsIntoManyMatchesSeparate). xn (== s.qScratch) is fully
		// consumed here before the o_proj quantize below overwrites it.
		qMatRowsIntoMany(xn, qMatTarget{ql.qProj, q}, qMatTarget{ql.kProj, kk}, qMatTarget{ql.vProj, vv})
		s.phaseEnd("q8_qkv_proj", tQKV)
		tRoPE := s.phaseStart()
		m.applyProjBias(l, q, kk, vv)
		m.applyLayerQKNorm(l, q, kk)
		s.ropeRowQK(l, q, kk, cos, sin)
		s.Cache.K[l] = append(s.Cache.K[l], kk...)
		s.Cache.V[l] = append(s.Cache.V[l], vv...)
		s.phaseEnd("q8_rope_kv", tRoPE)

		tAttn := s.phaseStart()
		attnOut := grow(db.attn, nH*hd)
		db.attn = attnOut
		clear(attnOut)
		scoreDot3 := fdot3scalar
		if attnFdot3SIMD {
			scoreDot3 = fdot3SIMD
		}
		if numWorkers <= 1 {
			db.scores = attnDecodeOne(attnOut, q, s.Cache, l, nH, hd, w, grp, scale, fdot, scoreDot3, db.scores)
		} else {
			caches := growCaches(db.caches, 1)
			db.caches = caches
			caches[0] = s.Cache
			db.scores = attnDecodeBatch(attnOut, q, caches, l, 1, nH, hd, w, grp, cfg.windowForLayer(l), scale, fdot, scoreDot3, db.scores, s.M.attnObs)
		}
		s.phaseEnd("q8_attn", tAttn)
		tO := s.phaseStart()
		o := grow(db.O, H)
		db.O = o
		qMatRowsInto(ql.oProj, s.quantizeVecQ8(attnOut), o)
		m.addBiasIfPresent(o, layerName(l, "self_attn.o_proj.bias"))
		for i := 0; i < H; i++ {
			x[i] += o[i]
		}
		s.phaseEnd("q8_o_proj", tO)
		// MLP (SwiGLU)
		tMLP := s.phaseStart()
		xn2F := grow(db.Xn2, H)
		db.Xn2 = xn2F
		rmsnormInto(xn2F, x, ql.postNorm, eps)
		xn2 := s.quantizeVecQ8(xn2F)
		I := cfg.IntermediateSize
		g := grow(db.G, I)
		db.G = g
		u := grow(db.U, I)
		db.U = u
		qMatRowsIntoMany(xn2, qMatTarget{ql.gateProj, g}, qMatTarget{ql.upProj, u})
		m.addBiasIfPresent(g, layerName(l, "mlp.gate_proj.bias"))
		m.addBiasIfPresent(u, layerName(l, "mlp.up_proj.bias"))
		for i := 0; i < I; i++ {
			g[i] = act(g[i], cfg) * u[i]
		}
		down := grow(db.Down, H)
		db.Down = down
		qMatRowsInto(ql.downProj, s.quantizeVecQ8(g), down)
		m.addBiasIfPresent(down, layerName(l, "mlp.down_proj.bias"))
		for i := 0; i < H; i++ {
			x[i] += down[i]
		}
		s.phaseEnd("q8_mlp", tMLP)
	}
	s.Cache.pos = append(s.Cache.pos, pos)
	xnorm := grow(db.Xnorm, H)
	db.Xnorm = xnorm
	rmsnormInto(xnorm, x, m.tensor("model.norm.weight"), eps)
	return xnorm
}

func attnDecodeOne(attnOut, Q []float32, cache *KVCache, layer, nH, hd, w, grp int, scale float32, scoreDot func(a, b []float32) float32, scoreDot3 func(a, b, c, x []float32) (float32, float32, float32), scoreScratch [][]float32) [][]float32 {
	nKV := nH / grp
	Kl, Vl := cache.K[layer], cache.V[layer]
	nPos := len(Kl) / w
	scoreScratch = grow2D(scoreScratch, grp, nPos)
	useSaxpy3SIMD := attnSaxpy3SIMDMinBatch <= 1 && nPos >= attnSaxpy3SIMDMinPos

	for kvh := 0; kvh < nKV; kvh++ {
		if attnGQAFuse && grp == 3 && scoreDot3 != nil {
			h0 := kvh * grp
			q0 := Q[(h0+0)*hd : (h0+1)*hd]
			q1 := Q[(h0+1)*hd : (h0+2)*hd]
			q2 := Q[(h0+2)*hd : (h0+3)*hd]
			sc0 := scoreScratch[0][:nPos]
			sc1 := scoreScratch[1][:nPos]
			sc2 := scoreScratch[2][:nPos]
			for j := 0; j < nPos; j++ {
				kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
				s0, s1, s2 := scoreDot3(q0, q1, q2, kh)
				sc0[j] = s0 * scale
				sc1[j] = s1 * scale
				sc2[j] = s2 * scale
			}
			softmaxInPlace(sc0)
			softmaxInPlace(sc1)
			softmaxInPlace(sc2)
			out0 := attnOut[(h0+0)*hd : (h0+1)*hd]
			out1 := attnOut[(h0+1)*hd : (h0+2)*hd]
			out2 := attnOut[(h0+2)*hd : (h0+3)*hd]
			for j := 0; j < nPos; j++ {
				vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
				if useSaxpy3SIMD && saxpy3Fast(out0, out1, out2, vh, sc0[j], sc1[j], sc2[j]) {
					continue
				}
				a0, a1, a2 := sc0[j], sc1[j], sc2[j]
				for d, v := range vh {
					out0[d] += a0 * v
					out1[d] += a1 * v
					out2[d] += a2 * v
				}
			}
			continue
		}

		for g := 0; g < grp; g++ {
			h := kvh*grp + g
			qh := Q[h*hd : (h+1)*hd]
			sc := scoreScratch[g][:nPos]
			for j := 0; j < nPos; j++ {
				kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
				sc[j] = scoreDot(qh, kh) * scale
			}
			softmaxInPlace(sc)
		}
		if grp == 3 {
			h0 := kvh * grp
			out0 := attnOut[(h0+0)*hd : (h0+1)*hd]
			out1 := attnOut[(h0+1)*hd : (h0+2)*hd]
			out2 := attnOut[(h0+2)*hd : (h0+3)*hd]
			sc0 := scoreScratch[0][:nPos]
			sc1 := scoreScratch[1][:nPos]
			sc2 := scoreScratch[2][:nPos]
			for j := 0; j < nPos; j++ {
				vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
				if useSaxpy3SIMD && saxpy3Fast(out0, out1, out2, vh, sc0[j], sc1[j], sc2[j]) {
					continue
				}
				a0, a1, a2 := sc0[j], sc1[j], sc2[j]
				for d, v := range vh {
					out0[d] += a0 * v
					out1[d] += a1 * v
					out2[d] += a2 * v
				}
			}
			continue
		}
		for j := 0; j < nPos; j++ {
			vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
			for d, v := range vh {
				for g := 0; g < grp; g++ {
					h := kvh*grp + g
					out := attnOut[h*hd : (h+1)*hd]
					out[d] += scoreScratch[g][j] * v
				}
			}
		}
	}
	return scoreScratch
}

// prefillBatchedQ is the Q8_0 prefill path: the structural twin of prefillBatched, with
// the projections run as quantized batched GEMMs (each weight row reused across all P
// pre-quantized activation rows). Fills the same f32 KV cache the f32 path builds.
func (s *Session) prefillBatchedQ(ids []int) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	P := len(ids)
	base := s.Cache.Len()

	// legacy reconstructs the pre-optimization prefill (legacy per-element GEMM + serial
	// SwiGLU + naive single-accumulator attention dot) so FAK_QGEMM=legacy gives a clean
	// same-environment before/after A/B of the whole prefill, not just the GEMM kernel.
	legacy := qgemmMode == qgemmModeLegacy
	scoreDot := fdot
	if legacy {
		scoreDot = dot
	}

	var tQuant, tGemm, tAttn time.Duration
	t0 := time.Now()
	tic := func() time.Time {
		if qprofOn {
			return time.Now()
		}
		return time.Time{}
	}
	toc := func(d *time.Duration, t time.Time) {
		if qprofOn {
			*d += time.Since(t)
		}
	}
	gemm := func(qt *q8Tensor, qp *q8Panel) []float32 {
		t := tic()
		r := qGemm8(qt, qp)
		toc(&tGemm, t)
		return r
	}
	// One reused scratch panel for all 4×NumLayers activation quantizations: each panel is
	// fully consumed before the next is built (q/k/v → o → gate/up → down), so a single
	// buffer is safe and avoids ~120 large allocations per prefill.
	scratch := &q8Panel{}
	qz := func(X []float32, P, width int) *q8Panel {
		t := tic()
		quantizeBatchPanelInto(scratch, X, P, width)
		toc(&tQuant, t)
		return scratch
	}

	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[t*H:(t+1)*H], cfg) // Gemma; no-op for Llama
	}

	cosP := make([][]float32, P)
	sinP := make([][]float32, P)
	for t := 0; t < P; t++ {
		cosP[t], sinP[t] = ropeRow(cfg, base+t)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		ql := m.q8Layer(l)

		Xn := make([]float32, P*H)
		parFor(P, numWorkers, func(lo, hi int) {
			wIn := m.tensor(lp("input_layernorm.weight"))
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wIn, eps, cfg))
				} else {
					rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
				}
			}
		})
		Xnq := qz(Xn, P, H)

		Q := gemm(ql.qProj, Xnq)
		K := gemm(ql.kProj, Xnq)
		V := gemm(ql.vProj, Xnq)
		for t := 0; t < P; t++ {
			m.applyProjBias(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], V[t*w:(t+1)*w])
			m.applyLayerQKNorm(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w])
		}

		// Stash raw (pre-RoPE, post-qk-norm) K straight into the cache, THEN RoPE K in place — this is the
		// same bytes the old `Kraw := append(nil, K...)` temp captured, without the extra
		// 196KB alloc+copy per layer (~5.9MB/prefill of GC churn the "rest" phase paid for).
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], K...)
		parFor(P, numWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				ropeRowQKInto(Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], cosP[t], sinP[t], hd, nH, nKV)
			}
		})

		s.Cache.K[l] = append(s.Cache.K[l], K...)
		s.Cache.V[l] = append(s.Cache.V[l], V...)
		Kl, Vl := s.Cache.K[l], s.Cache.V[l]

		attnOut := make([]float32, P*nH*hd)
		tA := tic()
		attnPrefillInto(attnOut, Q, Kl, Vl, P, base, nH, hd, w, grp, cfg.windowForLayer(l), l, scale, attnCap, scoreDot, s.M.attnObs)
		toc(&tAttn, tA)

		O := gemm(ql.oProj, qz(attnOut, P, nH*hd))
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(O[t*H:(t+1)*H], lp("self_attn.o_proj.bias"))
		}
		parFor(len(X), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += O[i]
			}
		})

		Xn2 := make([]float32, P*H)
		parFor(P, numWorkers, func(lo, hi int) {
			wPost := m.tensor(lp("post_attention_layernorm.weight"))
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn2[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wPost, eps, cfg))
				} else {
					rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
				}
			}
		})
		I := cfg.IntermediateSize
		Xn2q := qz(Xn2, P, H)
		G := gemm(ql.gateProj, Xn2q)
		U := gemm(ql.upProj, Xn2q)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(G[t*I:(t+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[t*I:(t+1)*I], lp("mlp.up_proj.bias"))
		}
		if legacy {
			for i := range G {
				G[i] = act(G[i], cfg) * U[i]
			}
		} else {
			parFor(len(G), numWorkers, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					G[i] = act(G[i], cfg) * U[i]
				}
			})
		}
		Down := gemm(ql.downProj, qz(G, P, I))
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(Down[t*H:(t+1)*H], lp("mlp.down_proj.bias"))
		}
		parFor(len(X), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += Down[i]
			}
		})
	}

	for t := 0; t < P; t++ {
		s.Cache.pos = append(s.Cache.pos, base+t)
	}
	if qprofOn {
		total := time.Since(t0)
		rest := total - tGemm - tAttn - tQuant
		ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
		fmt.Fprintf(os.Stderr, "[qprof P=%d] total=%.1f  gemm=%.1f  attn=%.1f  quant=%.1f  rest(norm/rope/resid)=%.1f ms\n",
			P, ms(total), ms(tGemm), ms(tAttn), ms(tQuant), ms(rest))
	}
	last := X[(P-1)*H : P*H]
	return rmsnormCfg(last, m.tensor("model.norm.weight"), eps, cfg)
}
