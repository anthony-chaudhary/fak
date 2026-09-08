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
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
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
	qsaGatheredK                                    []float32
	qsaGatheredV                                    []float32
	qsaBlockScores                                  []float32
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
	dispatchWorkers := currentWorkerCount()
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
	if dispatchWorkers > 1 {
		nw := dispatchWorkers
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
	if cfg.HasQSASparseAttn() {
		maxGather := compute.QSAMaxGatherTokens
		db.qsaGatheredK = grow(db.qsaGatheredK, maxGather*w)
		db.qsaGatheredV = grow(db.qsaGatheredV, maxGather*w)
		maxBlocks := (maxPositions + compute.QSABlockSize - 1) / compute.QSABlockSize
		if maxBlocks < 1 {
			maxBlocks = 1
		}
		db.qsaBlockScores = grow(db.qsaBlockScores, maxBlocks)
	}
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

// HasVectorizedDeltaNet reports whether the vectorized Gated-DeltaNet operator kernel
// is registered and available for accelerated decode execution.
func HasVectorizedDeltaNet() bool {
	if env := os.Getenv("FAK_VECTORIZED_DELTANET"); env == "0" || strings.EqualFold(env, "false") || strings.EqualFold(env, "no") || strings.EqualFold(env, "off") {
		return false
	}
	return compute.HasVectorizedDeltaNet()
}

// HasTiledChannelTranspose reports whether the tiled memory channel transpose
// kernel for DeltaNet linear attention conv concat is available.
func HasTiledChannelTranspose() bool {
	return compute.HasTiledChannelTranspose()
}

// HasQSASparseRowGather reports whether the true QSA sparse row gather
// kernel for long-context attention decode is available.
func HasQSASparseRowGather() bool {
	return compute.HasQSASparseRowGather()
}

func q8FastDecodeSessionOK(s *Session, cfg Config) bool {
	if cfg.IsHybrid() || cfg.IsQwen35Hybrid() {
		if !HasVectorizedDeltaNet() {
			return false
		}
		// Permit mixed-quantization sessions where base projection weights are Q4_K / Q8 while recurrent state remains FP32.
		return true
	}
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
func (s *Session) tokenHiddenQ(id, pos int) (out []float32) {
	finishGraph := s.beginQwen35DecodeGraph(pos)
	aborted := true
	defer func() { finishGraph(aborted) }()
	// Fail closed BY NAME for a per-layer-head_dim arch (gemma4) whose q_norm/k_norm the
	// generic block path's scalar-HeadDim qk-norm band cannot express — instead of the
	// cryptic deep panic in applyQKNormCfg on the first token (issue #4274).
	s.requireResidentQuantForwardSupported()
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	// Arm the FAK_HIDDEN_TAP decode-time hidden dump for THIS forward when it is the
	// tapped position — exactly as the f32 tokenHidden does. This is the real failing
	// path (the 27B GGUF runs s.Quant): the token-3 drift and #4273 long-context collapse
	// both live here, so without arming the tap the "decisive next witness" capture
	// (FAK_HIDDEN_TAP=… fak run --model qwen3.6-27b-q4_k_m.gguf) would dump nothing —
	// blockStep's dumpLayer / linearAttnStep's dumpOp are all gated on s.tapActive, and no
	// meta.json would be written for the probe to load. The per-op GDN taps ride the shared
	// blockStep→linearAttnStep path (the slow, resident-Q4_K/Q8 branch the hybrid always takes).
	tap := s.activeTap()
	if tap != nil && !tap.wants(pos) {
		tap = nil
	}
	prevTap := s.tapActive
	s.tapActive = tap
	defer func() { s.tapActive = prevTap }()
	if cfg.IsHybrid() || cfg.IsQwen35Hybrid() || !q8FastDecodeSessionOK(s, cfg) {
		mat := matKernel(sessionQ8Kernel{s})
		if s.Q4 && m.q4w != nil {
			// Resident int4 decode: the Qwen3.6 hybrid (and every non-fast-PreNorm arch)
			// runs the shared blockStep skeleton; swapping the kernel to int4 streams ~1.8×
			// fewer weight bytes/token than Q8, raising the decode ceiling. The block
			// orchestration (RMSNorm, RoPE, GQA, GDN recurrent scan, SwiGLU) is unchanged.
			mat = matKernel(sessionQ4Kernel{s})
		} else if s.Q4K && (m.q4kw != nil || len(m.kqw) > 0) {
			// Resident raw Q4_K decode (plan P1): same blockStep skeleton, but the q4_k_m
			// matmul majority streams at 0.5625 B/weight (raw GGUF bytes, no round-trip) and
			// the Q6_K minority (attn_qkv/ffn_down) falls back to the Q8 GEMV inside the kernel.
			mat = matKernel(sessionQ4KKernel{s})
		} else if cfg.IsMoE() {
			// MoE routing reuses the original prepared router/expert operand after nested
			// down-projection prep calls, so keep the allocation-safe kernel for that path.
			mat = q8Kernel{m}
		}
		var x []float32
		if m.Q2KEmbedding != nil {
			x = make([]float32, H)
			if err := m.Q2KEmbedding.GatherRow(id, x, cfg.embedScale()); err != nil {
				panic(err)
			}
		} else {
			embed := m.embedRows()
			x = append([]float32(nil), embed[id*H:(id+1)*H]...)
			scaleEmbedInPlace(x, cfg) // Gemma; no-op for Llama
		}
		for l := 0; l < cfg.NumLayers; l++ {
			cos, sin := ropeRowForLayer(cfg, l, pos)
			x = s.blockStep(l, pos, x, cos, sin, mat)
		}
		s.Cache.appendPosition(pos, id)
		if tap != nil {
			tap.writeMeta(cfg, H, pos)
		}
		// finalNorm, not rmsnormCfg. This branch is entered precisely BECAUSE
		// q8FastDecodeSessionOK said no, and q8FastPreNormOK (line 145) refuses every
		// cfg.LayerNorm config — so a biased-LayerNorm family (StableLM, GPT-NeoX, Falcon,
		// biased Cohere) decodes here by construction, and rmsnormCfg's hard-coded nil bias
		// threw its learned model.norm.bias away after blockStep had applied every per-layer
		// norm bias correctly.
		out = m.finalNorm(x)
		aborted = false
		return out
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

	var x []float32
	if m.Q2KEmbedding != nil {
		x = grow(db.X, H)[:H]
		db.X = x
		if err := m.Q2KEmbedding.GatherRow(id, x, cfg.embedScale()); err != nil {
			panic(err)
		}
	} else {
		embed := m.embedRows()
		x = grow(db.X, H)
		db.X = x
		copy(x, embed[id*H:(id+1)*H])
		scaleEmbedInPlace(x, cfg) // Gemma; no-op for Llama
	}

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
		if cfg.ShouldUseQSASparseGather(l, s.Cache.Len(), 1) {
			db.scores = s.attnDecodeQSA(attnOut, q, s.Cache, db, l, nH, hd, w, grp, scale, fdot, scoreDot3)
		} else if currentWorkerCount() <= 1 {
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
		// The optimized resident-Q8 loop bypasses blockStep, so carry the same
		// default-off bidirectional residual hook explicitly at its layer boundary.
		// This is outside every timed phase and costs only a nil branch when unarmed.
		if tap != nil {
			tap.applySteer(l, pos, x)
			tap.dumpLayer(l, layerKindLabel(cfg, l), x)
		}
	}
	s.Cache.appendPosition(pos, id)
	if tap != nil {
		tap.writeMeta(cfg, H, pos)
	}
	xnorm := grow(db.Xnorm, H)
	db.Xnorm = xnorm
	rmsnormInto(xnorm, x, m.tensor("model.norm.weight"), eps)
	out = xnorm
	aborted = false
	return out
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
			q0, q1, q2 := packedHead3(Q, 0, len(Q), h0, hd)
			sc0, sc1, sc2 := scoreScratchHead3(scoreScratch, 0, 1, nPos)
			fillSoftmaxAttentionScores3(sc0, sc1, sc2, q0, q1, q2, Kl, 0, nPos, w, kvh, hd, scale, scoreDot3)
		} else {
			for g := 0; g < grp; g++ {
				h := kvh*grp + g
				qh := vectorHead(Q, h, hd)
				sc := scoreScratch[g][:nPos]
				fillSoftmaxAttentionScores(sc, qh, Kl, 0, nPos, w, kvh, hd, scale, scoreDot)
			}
		}
		if grp == 3 {
			h0 := kvh * grp
			sc0, sc1, sc2 := scoreScratchHead3(scoreScratch, 0, 1, nPos)
			accumulatePackedAttentionValues3(attnOut, 0, len(attnOut), h0, hd, Vl, sc0, sc1, sc2, 0, nPos, w, kvh, useSaxpy3SIMD)
			continue
		}
		accumulateAttentionGroup(attnOut, 0, len(attnOut), kvh*grp, grp, hd, Vl, scoreScratch, 0, 0, nPos, w, kvh)
	}
	return scoreScratch
}

// attnDecodeQSA executes true QSA sparse row gather attention (Issue #465) for one decode position:
//  1. Validates dynamic gating threshold (N_kv >= 16,384 tokens).
//  2. Scores attention blocks on-device without CPU round-trips.
//  3. Uses native HIP radix top-k selection (2,048 top-k tokens + 256 local tail tokens).
//  4. Gathers selected rows into contiguous scratch memory (4.72 MB, 100% resident in Strix Halo 32 MB Infinity Cache).
//  5. Evaluates attention strictly over the gathered rows, eliminating dense masking overhead.
func (s *Session) attnDecodeQSA(
	attnOut, Q []float32,
	cache *KVCache,
	db *qDecodeBuf,
	layer, nH, hd, w, grp int,
	scale float32,
	scoreDot func(a, b []float32) float32,
	scoreDot3 func(a, b, c, x []float32) (float32, float32, float32),
) [][]float32 {
	Kl, Vl := cache.K[layer], cache.V[layer]
	totalTokens := len(Kl) / w
	blockSize := compute.QSABlockSize

	if totalTokens < compute.QSADynamicGatingThreshold {
		return attnDecodeOne(attnOut, Q, cache, layer, nH, hd, w, grp, scale, scoreDot, scoreDot3, db.scores)
	}

	totalBlocks := (totalTokens + blockSize - 1) / blockSize
	db.qsaBlockScores = grow(db.qsaBlockScores, totalBlocks)
	scores := db.qsaBlockScores[:totalBlocks]

	// Score blocks using representative query head dot product against mid-block key
	q0 := vectorHead(Q, 0, hd)
	for b := 0; b < totalBlocks; b++ {
		midToken := b*blockSize + (blockSize / 2)
		if midToken >= totalTokens {
			midToken = totalTokens - 1
		}
		kMid := packedHead(Kl, midToken, w, 0, hd)
		scores[b] = scoreDot(q0, kMid) * scale
	}

	topKBlocks := compute.QSABaseTopKTokens / blockSize
	tailBlocks := compute.QSALocalTailTokens / blockSize
	selectedBlocks, receipt, err := compute.RadixTopKBlockSelect(scores, totalBlocks, topKBlocks, tailBlocks)
	if err != nil || receipt.DynamicGatingBypassed {
		return attnDecodeOne(attnOut, Q, cache, layer, nH, hd, w, grp, scale, scoreDot, scoreDot3, db.scores)
	}

	neededGatherLen := len(selectedBlocks) * blockSize * w
	db.qsaGatheredK = grow(db.qsaGatheredK, neededGatherLen)
	db.qsaGatheredV = grow(db.qsaGatheredV, neededGatherLen)

	nGatheredElements, err := compute.SparseRowGatherKVInto(
		db.qsaGatheredK[:neededGatherLen],
		db.qsaGatheredV[:neededGatherLen],
		Kl, Vl, selectedBlocks, blockSize, nH/grp, hd, totalTokens,
	)
	if err != nil {
		return attnDecodeOne(attnOut, Q, cache, layer, nH, hd, w, grp, scale, scoreDot, scoreDot3, db.scores)
	}

	gK := db.qsaGatheredK[:nGatheredElements]
	gV := db.qsaGatheredV[:nGatheredElements]
	nGathered := nGatheredElements / w
	db.scores = grow2D(db.scores, grp, nGathered)
	useSaxpy3SIMD := attnSaxpy3SIMDMinBatch <= 1 && nGathered >= attnSaxpy3SIMDMinPos

	nKV := nH / grp
	for kvh := 0; kvh < nKV; kvh++ {
		if attnGQAFuse && grp == 3 && scoreDot3 != nil {
			h0 := kvh * grp
			q0, q1, q2 := packedHead3(Q, 0, len(Q), h0, hd)
			sc0, sc1, sc2 := scoreScratchHead3(db.scores, 0, 1, nGathered)
			fillSoftmaxAttentionScores3(sc0, sc1, sc2, q0, q1, q2, gK, 0, nGathered, w, kvh, hd, scale, scoreDot3)
		} else {
			for g := 0; g < grp; g++ {
				h := kvh*grp + g
				qh := vectorHead(Q, h, hd)
				sc := db.scores[g][:nGathered]
				fillSoftmaxAttentionScores(sc, qh, gK, 0, nGathered, w, kvh, hd, scale, scoreDot)
			}
		}
		if grp == 3 {
			h0 := kvh * grp
			sc0, sc1, sc2 := scoreScratchHead3(db.scores, 0, 1, nGathered)
			accumulatePackedAttentionValues3(attnOut, 0, len(attnOut), h0, hd, gV, sc0, sc1, sc2, 0, nGathered, w, kvh, useSaxpy3SIMD)
			continue
		}
		accumulateAttentionGroup(attnOut, 0, len(attnOut), kvh*grp, grp, hd, gV, db.scores, 0, 0, nGathered, w, kvh)
	}

	return db.scores
}

// q8PrefillOProjDstObserver is an optional test hook called before self_attn.o_proj
// GEMM in prefillBatchedQ, allowing unit tests to witness destination buffer reuse.
var q8PrefillOProjDstObserver func(layer int, dst []float32)

// prefillBatchedQ is the Q8_0 prefill path: the structural twin of prefillBatched, with
// the projections run as quantized batched GEMMs (each weight row reused across all P
// pre-quantized activation rows). Fills the same f32 KV cache the f32 path builds.
func (s *Session) prefillBatchedQ(ids []int) []float32 {
	dispatchWorkers := currentWorkerCount()
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
	gemmInto := func(qt *q8Tensor, qp *q8Panel, dst []float32) {
		t := tic()
		qGemm8Into(qt, qp, dst)
		toc(&tGemm, t)
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

	X := make([]float32, P*H)
	if m.Q2KEmbedding != nil {
		scale := cfg.embedScale()
		for t, id := range ids {
			if err := m.Q2KEmbedding.GatherRow(id, X[t*H:(t+1)*H], scale); err != nil {
				panic(err)
			}
		}
	} else {
		embed := m.embedRows()
		for t, id := range ids {
			copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
			scaleEmbedInPlace(X[t*H:(t+1)*H], cfg) // Gemma; no-op for Llama
		}
	}

	cosP := make([][]float32, P)
	sinP := make([][]float32, P)
	for t := 0; t < P; t++ {
		cosP[t], sinP[t] = ropeRow(cfg, base+t)
	}

	// Normalization consumers finish synchronously before the next norm stage.
	// Every row is overwritten, so one request-local panel serves both stages.
	normPanel := make([]float32, P*H)
	// Attention and projection consumers finish before the next layer reuses this panel.
	attnOut := make([]float32, P*nH*hd)
	// GEMM fully overwrites these separate panels; all consumers finish before reuse.
	I := cfg.IntermediateSize
	G := make([]float32, P*I)
	U := make([]float32, P*I)
	Down := make([]float32, P*H)
	// Output projection overwrites every cell; residual addition finishes before reuse.
	O := make([]float32, P*H)
	// Cache appends copy K/V; synchronous attention consumes Q before reuse.
	Q := make([]float32, P*nH*hd)
	K := make([]float32, P*w)
	V := make([]float32, P*w)
	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		ql := m.q8Layer(l)

		// q8PrefillNeedsTokenLoop (kv.go:825) has no LayerNorm term, so a quantized PreNorm
		// LayerNorm family prefills HERE while decoding through the bias-aware blockStep. The
		// learned input_layernorm.bias must therefore ride along; rmsnormCfg hard-passes nil.
		Xn := normPanel
		parFor(P, dispatchWorkers, func(lo, hi int) {
			wIn := m.tensor(lp("input_layernorm.weight"))
			bIn := m.tensorOptional(lp("input_layernorm.bias"))
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn[t*H:(t+1)*H], normCfg(X[t*H:(t+1)*H], wIn, bIn, eps, cfg))
				} else {
					rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
				}
			}
		})
		Xnq := qz(Xn, P, H)

		gemmInto(ql.qProj, Xnq, Q)
		gemmInto(ql.kProj, Xnq, K)
		gemmInto(ql.vProj, Xnq, V)
		for t := 0; t < P; t++ {
			m.applyProjBias(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], V[t*w:(t+1)*w])
			m.applyLayerQKNorm(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w])
		}

		// Stash raw (pre-RoPE, post-qk-norm) K straight into the cache, THEN RoPE K in place — this is the
		// same bytes the old `Kraw := append(nil, K...)` temp captured, without the extra
		// 196KB alloc+copy per layer (~5.9MB/prefill of GC churn the "rest" phase paid for).
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], K...)
		parFor(P, dispatchWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				ropeRowQKInto(Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], cosP[t], sinP[t], hd, nH, nKV)
			}
		})

		s.Cache.K[l] = append(s.Cache.K[l], K...)
		s.Cache.V[l] = append(s.Cache.V[l], V...)
		Kl, Vl := s.Cache.K[l], s.Cache.V[l]

		// Attention accumulates values, so discard the previous layer output.
		clear(attnOut)
		tA := tic()
		attnPrefillInto(attnOut, Q, Kl, Vl, P, base, nH, hd, w, grp, cfg.windowForLayer(l), l, scale, attnCap, scoreDot, s.M.attnObs)
		toc(&tAttn, tA)

		if q8PrefillOProjDstObserver != nil {
			q8PrefillOProjDstObserver(l, O)
		}
		gemmInto(ql.oProj, qz(attnOut, P, nH*hd), O)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(O[t*H:(t+1)*H], lp("self_attn.o_proj.bias"))
		}
		parFor(len(X), dispatchWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += O[i]
			}
		})

		Xn2 := normPanel
		parFor(P, dispatchWorkers, func(lo, hi int) {
			wPost := m.tensor(lp("post_attention_layernorm.weight"))
			bPost := m.tensorOptional(lp("post_attention_layernorm.bias"))
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn2[t*H:(t+1)*H], normCfg(X[t*H:(t+1)*H], wPost, bPost, eps, cfg))
				} else {
					rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
				}
			}
		})
		Xn2q := qz(Xn2, P, H)
		gemmInto(ql.gateProj, Xn2q, G)
		gemmInto(ql.upProj, Xn2q, U)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(G[t*I:(t+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[t*I:(t+1)*I], lp("mlp.up_proj.bias"))
		}
		if legacy {
			for i := range G {
				G[i] = act(G[i], cfg) * U[i]
			}
		} else {
			parFor(len(G), dispatchWorkers, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					G[i] = act(G[i], cfg) * U[i]
				}
			})
		}
		gemmInto(ql.downProj, qz(G, P, I), Down)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(Down[t*H:(t+1)*H], lp("mlp.down_proj.bias"))
		}
		parFor(len(X), dispatchWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += Down[i]
			}
		})
	}

	for t := 0; t < P; t++ {
		s.Cache.appendPosition(base+t, ids[t])
	}
	if qprofOn {
		total := time.Since(t0)
		rest := total - tGemm - tAttn - tQuant
		ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
		fmt.Fprintf(os.Stderr, "[qprof P=%d] total=%.1f  gemm=%.1f  attn=%.1f  quant=%.1f  rest(norm/rope/resid)=%.1f ms\n",
			P, ms(total), ms(tGemm), ms(tAttn), ms(tQuant), ms(rest))
	}
	last := X[(P-1)*H : P*H]
	// finalNorm, not a hand-rolled normCfg: it is the ONE place the final-norm weight, its
	// optional bias, and eps are bound together, so this lane cannot drift from the per-token
	// path again the way the hard-coded nil bias here did.
	return m.finalNorm(last)
}

const (
	// DefaultMALLCapacityBlocks is the exact number of 256 KiB sequence blocks that fit in 32MB MALL cache:
	// 32MB / 256 KiB = 128 blocks = 8,192 tokens.
	DefaultMALLCapacityBlocks = 128

	// DefaultBlockTokens is the standard sequence block token count (64 tokens).
	DefaultBlockTokens = 64
)

// CachePolicyHint specifies the RDNA 3.5 cache allocation directive for a token span, tree mask, or weight tensor.
type CachePolicyHint struct {
	SLC          int    `json:"slc"`           // System Level Cache: 0 = allocate/cached in MALL, 1 = bypass MALL
	GLC          int    `json:"glc"`           // Globally Coherent / L1 cache: 0 = normal cacheable, 1 = bypass
	NT           int    `json:"nt"`            // Non-Temporal flag: 0 = temporal reuse, 1 = non-temporal streaming bypass
	Temporal     bool   `json:"temporal"`      // True if cached/pinned in MALL
	Bypass       bool   `json:"bypass"`        // True if bypassing MALL to prevent cache thrashing
	Prefetch     bool   `json:"prefetch"`      // True if asynchronous prefetch (s_prefetch_data) should be emitted
	PolicyName   string `json:"policy_name"`   // "TEMPORAL_PINNED", "STREAMING_BYPASS", or "PARTITIONED"
	PinnedTokens int    `json:"pinned_tokens"` // Number of tokens in span qualifying for MALL pinning (0..8191)
	BypassTokens int    `json:"bypass_tokens"` // Number of tokens in span bypassing MALL (>8191)
}

var (
	// CacheHintTemporal directs CUs to fetch tokens with temporal caching (SLC=0, GLC=0, NT=0),
	// pinning hot KV blocks and tree masks in the 32MB MALL cache.
	CacheHintTemporal = CachePolicyHint{
		SLC:        0,
		GLC:        0,
		NT:         0,
		Temporal:   true,
		Bypass:     false,
		Prefetch:   true,
		PolicyName: "TEMPORAL_PINNED",
	}

	// CacheHintStreamingBypass directs CUs to use non-temporal streaming bypass (NT=1, SLC=1),
	// bypassing MALL to prevent cache pollution from divergent tokens or model weight streaming.
	CacheHintStreamingBypass = CachePolicyHint{
		SLC:        1,
		GLC:        0,
		NT:         1,
		Temporal:   false,
		Bypass:     true,
		Prefetch:   false,
		PolicyName: "STREAMING_BYPASS",
	}
)

// ClassifyKVBlockCacheHint assigns the RDNA 3.5 cache allocation directive for a KV cache block:
//   - Blocks 0..127 (< DefaultMALLCapacityBlocks): CacheHintTemporal (MALL-pinned).
//   - Blocks >= 128: CacheHintStreamingBypass (streamed from DRAM).
func ClassifyKVBlockCacheHint(blockID ...int) CachePolicyHint {
	bID := 0
	tokens := DefaultBlockTokens
	if len(blockID) > 0 {
		bID = blockID[0]
	}
	if len(blockID) > 1 && blockID[1] > 0 {
		tokens = blockID[1]
	}
	if bID < 0 {
		bID = 0
	}
	if bID < DefaultMALLCapacityBlocks {
		hint := CacheHintTemporal
		hint.PinnedTokens = tokens
		hint.BypassTokens = 0
		return hint
	}
	hint := CacheHintStreamingBypass
	hint.PinnedTokens = 0
	hint.BypassTokens = tokens
	return hint
}

// ClassifyTreeMaskCacheHint returns the RDNA 3.5 cache allocation directive for candidate tree
// attention masks, tagging them for temporal pinning in MALL (SLC=0, GLC=0, NT=0).
func ClassifyTreeMaskCacheHint(numNodes ...int) CachePolicyHint {
	n := 0
	if len(numNodes) > 0 {
		n = numNodes[0]
		if n < 0 {
			n = 0
		}
	}
	hint := CacheHintTemporal
	hint.PinnedTokens = n
	hint.BypassTokens = 0
	return hint
}

// ClassifyWeightTensorCacheHint returns the non-temporal streaming bypass policy (NT=1, SLC=1)
// for streaming model weights to avoid polluting the pinned KV cache in MALL.
func ClassifyWeightTensorCacheHint(tensorName ...string) CachePolicyHint {
	return CacheHintStreamingBypass
}

// AttachWeightCacheHint returns the streaming weight bypass policy hint (NT=1, SLC=1)
// to attach to weight tensor memory loads during speculative verification.
func (s *Session) AttachWeightCacheHint(tensorName ...string) CachePolicyHint {
	return ClassifyWeightTensorCacheHint(tensorName...)
}

// AttachTreeMaskCacheHint returns the temporal pinned cache policy hint (SLC=0, GLC=0, NT=0)
// to attach to candidate tree attention mask memory loads.
func (s *Session) AttachTreeMaskCacheHint(numNodes ...int) CachePolicyHint {
	return ClassifyTreeMaskCacheHint(numNodes...)
}

// AttachKVCacheHint returns the cache policy hint (temporal pinned for root context blocks < 128,
// streaming bypass beyond) to attach to KV cache block loads.
func (s *Session) AttachKVCacheHint(blockID ...int) CachePolicyHint {
	return ClassifyKVBlockCacheHint(blockID...)
}

