package model

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// SessionFromPrefix starts a session whose cache is a clone of an already-computed
// prefix, so only the suffix needs prefilling (real prefix reuse). For
// GLM-MoE-DSA the clone carries the DSA attention/index cache instead of the dense
// GQA K/V rows.
func (m *Model) SessionFromPrefix(prefix *KVCache) *Session {
	return &Session{M: m, Cache: prefix.Clone()}
}

// Session drives generation over a kernel-owned KV cache. Prefill ingests a prompt;
// Step decodes one token. Both share the exact per-token math the verified full
// forward pass uses, so cached decode is provably identical to full prefill.
// GLM-MoE-DSA uses a separate DSA attention/index cache carried inside KVCache.
//
// Naming: this is the token-decoder sense of "session" (a generator over a KV
// cache), NOT the drive-state session. The canonical "session" — run-state,
// budget, priority, pace — is internal/session.Table / session.State; the wire
// DTO of that drive state is gateway.SessionState. See the vocabulary worklist at
// docs/notes/VOCAB-DISAMBIGUATION-WORKLIST-2026-06-24.md.
type Session struct {
	M     *Model
	Cache *KVCache
	// Backend is non-nil when this session is intentionally running through the
	// internal/compute HAL instead of the legacy direct []float32 path. The legacy
	// path stays the default until the full optimized prefill/batch path is adopted.
	Backend compute.Backend
	halKV   compute.KVStore
	// halW memoizes weights staged onto Backend so a device session uploads each weight
	// to VRAM exactly once, not once per token. (On cpu-ref, Upload is identity over the
	// zero-copy host view, so caching changes nothing and the bit-equality gate holds.)
	halW map[string]compute.Tensor
	// halStep counts tokens run through the HAL; the first two warm the device buffer pool
	// + weight cache before a graph-capturing backend starts replaying captured tokens.
	halStep int

	// Quant selects the Q8_0 quantized forward path (quant_forward.go) for this session's
	// prefill and decode. The f32 path is the default and is left byte-for-byte unchanged;
	// set Quant only on a session whose Model has had Quantize() called. The KV cache it
	// builds is the same f32 object either way, so Evict/Clone and the proven KV rungs are
	// independent of this flag.
	Quant bool

	// Q4 selects the resident int4 (Q4_0-style) forward path (quant_q4.go) over the Q8/f32
	// paths. Set only on a session whose Model has had QuantizeQ4() called (the q4w resident
	// copy exists). When Q4 is set, decode routes weight matmuls through q4Kernel — int4
	// streams ~1.8× fewer bytes/token than Q8, raising the decode ceiling toward the
	// llama.cpp q4 bar (see QWEN36-NATIVE-PERF-PLAN-2026-06-19.md). Prefill still runs the
	// Q8 batched GEMM; int4 is a decode-path optimization for now.
	Q4 bool

	// Q4K selects the resident hybrid Q4_K path (quant_q4k*.go) for a model loaded via the
	// memory-lean ggufload.LoadModelQ4K loader: identity-normalized matmul weights run as
	// raw Q4_K blocks (~0.56 B/weight, the bandwidth bulk), the normalize-sensitive
	// projections as Q8_0, small tensors as f32. A resident-hybrid matKernel dispatches per
	// name, so this one flag moves both prefill and decode onto the resident mixed path.
	// This is the end-to-end-correct, low-memory route toward the q4_k_m decode bar.
	Q4K bool

	// GPTQ selects the resident AutoGPTQ/GPTQModel path loaded by LoadGPTQ. It routes
	// matmul weights through residentMatRows (GPTQ when present, f32 for small tensors)
	// using the shared per-token blockStep skeleton. It is opt-in so existing f32/Q8/Q4
	// sessions remain byte-for-byte on their prior paths.
	GPTQ bool

	// CPUOffloadExperts routes the MoE expert GEMMs (mlp.experts.* and mlp.shared_experts.*)
	// to host RAM while the dense projections + router + attention run on Backend — the
	// llama.cpp `--n-cpu-moe` hybrid. It is the path that lets a Q4_K model whose experts dwarf
	// VRAM (GLM-5.2 Q4_K_M ≈ 424 GB) serve at all: experts live in the 1007 GB host RAM, the
	// every-token dense FLOPs stay on the GPU. Only the GLM-DSA forward honors it today
	// (glmDsaMatKernel, moe_offload.go); with Backend nil it is a no-op (everything already on
	// host) and the forward stays byte-for-byte the resident path. See splitKernel.
	CPUOffloadExperts bool

	// PrecisionPolicy enables dynamic whole-token precision. When set, Prefill/Step
	// speculatively run the Q8_0 path, inspect the returned distribution, and may roll the
	// KV cache back to recompute the same token/span in f32. It is additive: nil preserves
	// the fixed f32 / fixed Quant behavior exactly.
	PrecisionPolicy *DynamicPrecisionPolicy
	PrecisionStats  PrecisionStats

	// qScratch reuses the Q8 activation vector storage for serial quantized decode/head
	// GEMVs. Each qMatRows call consumes the vector before the next quantization overwrites
	// it, so this removes hot-path allocation without changing any Q8 arithmetic.
	qScratch q8Vec
	qDecode  *qDecodeBuf

	// Metal routes PREFILL's projection GEMMs through the Metal GPU backend
	// (metal_prefill.go, built only under -tags fakmetal) to reach llama.cpp-Metal prefill
	// parity on Apple Silicon — prefill is compute-bound, where the GPU's FLOP advantage is
	// decisive. Decode is untouched (it stays the bandwidth-bound CPU Q8 path, where Metal
	// barely helps). Only set Metal on a quantized model after metalgemm.Available() is true;
	// the same f32 KV cache is built either way, so KV semantics are unchanged.
	Metal bool

	// MetalQ4K routes the resident-Q4_K hybrid PREFILL's q4_k-majority projection/MLP GEMMs
	// through the Metal q4_k dequant-GEMM (internal/metalgemm/q4k.m, built only under -tags
	// fakmetal) instead of the CPU q4kGemm. Unlike Metal (above) this needs no f16 weight set —
	// the raw q4_k blocks stay resident on the GPU (the 27B q4_k_m fits 36 GB; f16 would not),
	// and the GPU's parallel dequant clears the CPU int8 ceiling (~23 GB/s → 125 GB/s steady).
	// Opt-in (FAK_METAL): it currently keeps the CPU q4kw copy resident too, so on a memory-tight
	// box the GPU upload double-counts — the loader change that drops the CPU copy is the
	// follow-up. The CPU path is byte-faithful, so logits are unchanged within the GPU
	// float-order band (TestMetalQ4KPrefillMatchesCPU). Decode is untouched (a lone GEMV is
	// occupancy-bound; the decode bar needs the one-command-buffer forward, a tracked follow-up).
	MetalQ4K bool

	// PhaseProfiler is an opt-in coarse wall-time profiler used by modelbench to split
	// Qwen3.6 prefill/decode into real execution phases. Nil keeps the hot path free of
	// time.Now calls.
	PhaseProfiler *PhaseProfiler

	// glmDsaSharedTopK carries the current token's most recent full-indexer
	// decision across IndexShare layers while tokenHiddenGLMDsa walks the block stack.
	glmDsaSharedTopK []int

	// decodeScores reuses one attention-score buffer across heads AND decode steps. A
	// single Session decodes serially and the per-step head loop is serial, so one buffer
	// (fully overwritten each head) is bit-identical to a fresh make per head. This removes
	// the per-head/per-step `make([]float32, context)` that otherwise made f32 decode
	// allocate O(n²) score bytes over an n-token generation — pure GC-pressure relief, no
	// arithmetic change (TestDecodeStepAllocationStaysBounded guards the bound).
	decodeScores []float32
}

// NewSession starts a fresh generation session.
func (m *Model) NewSession() *Session {
	return &Session{M: m, Cache: NewKVCache(m.Cfg)}
}

// token runs one position through all layers and projects to logits. It is
// tokenHidden (the shared prefill/decode compute) followed by the LM head; kept as
// the decode path (Step) where every step's logits are actually consumed.
// requirePreNorm panics if this session's model uses a non-PreNorm block topology
// on a code path (HAL / Metal / quant-batch) that is a SEAM-0 hand-copy still
// hardcoding the Llama PreNorm wiring. Non-PreNorm topologies run only on the
// topology-aware f32 blockStep / cacheless layer() paths today (MODEL-ARCH-SEAM
// SEAM-0 collapses the remaining copies); this turns a silent wrong result into a
// loud, honest boundary.
func (s *Session) requirePreNorm(path string) {
	if t := s.M.Cfg.BlockTopology; t != PreNorm {
		panic("model: " + path + " does not yet implement BlockTopology " + t.String() + " (only PreNorm); see MODEL-ARCH-SEAM SEAM-0")
	}
	if s.M.Cfg.hasLayerSpecificRopeTheta() {
		panic("model: " + path + " does not yet implement layer-specific RoPE theta; see MODEL-ARCH-SEAM Gemma3 O3")
	}
}

func (s *Session) token(id, pos int) []float32 {
	if s.Backend != nil {
		s.requirePreNorm("HAL decode")
		return s.tokenHAL(id, pos)
	}
	if s.Q4 {
		return s.headQ4(s.tokenHiddenQ(id, pos))
	}
	if s.Q4K {
		// Resident Q4_K decode: block matmuls dispatch per name (raw q4_k majority + Q8
		// minority); the LM head is whichever resident format it loaded as, so headResident
		// picks q4k/q8/f32 rather than assuming Q8.
		return s.headResident(s.tokenHiddenQ(id, pos))
	}
	if s.GPTQ {
		return s.headResident(s.tokenHiddenGPTQ(id, pos))
	}
	if s.Quant {
		// GPU-resident decode forward (#67): run the whole token — forward + final norm + LM head —
		// in one Metal command buffer and return logits directly (metal_decode.go). Returns nil for a
		// hybrid/MoE model or when the resident path declines, so this is a cheap gate on the CPU path.
		if logits := s.metalDecodeLogitsQ8(id, pos); logits != nil {
			return logits
		}
		return s.headQ(s.tokenHiddenQ(id, pos))
	}
	return s.head(s.tokenHidden(id, pos))
}

func (s *Session) requireGLMDsaSession() {
	// #86 (partial): a compute.Backend is now PERMITTED — the GLM-MoE-DSA forward routes its
	// dense GEMMs (MoE/FFN, projections, head) through the backend (backendKernel) while the DSA
	// index-scoring + sparse-attention + KV stay host-resident (s.Cache.glm). Metal/PrecisionPolicy
	// are still unwired and fail closed.
	if s.Metal || s.PrecisionPolicy != nil {
		panic("model: GLM-MoE-DSA Session: Metal/PrecisionPolicy paths are unwired (CPU resident DSA cache; compute.Backend GEMM offload is allowed)")
	}
	if s.Cache.glm == nil {
		s.Cache.glm = newGLMDsaKVCache(s.M.Cfg)
	}
}

func (s *Session) glmDsaHead(xf []float32) []float32 {
	if s.Backend != nil {
		// #86 (partial): the vocab projection (the largest single GEMM) runs on the backend.
		// lmHeadMatHAL resolves the resident head weight (untied q8 / f32) + uploads it.
		be := s.Backend
		xt := uploadHostF32Class(be, []int{s.M.Cfg.HiddenSize}, xf, compute.MemoryActivation, "glm-dsa-lm-head-activation")
		out := be.Read(be.MatMul(s.lmHeadMatHAL(), xt))
		be.Free(xt)
		return out
	}
	if s.Quant {
		return s.headQ(xf)
	}
	return s.head(xf)
}

// head applies the (tied) LM head to a post-final-norm hidden vector. Split out from
// token so prefill can run it ONCE: Prefill returns only the last position's logits,
// so computing the 49,152×576 head at every prefill position (its weight, the tied
// embedding, is the single largest tensor at 113 MB) and discarding all but the last
// is pure waste. Skipping it is bit-identical — the head feeds neither the KV cache
// nor any hidden state, only the returned logits — so R2/R3/R14 stay oracle-green.
func (s *Session) head(xf []float32) []float32 {
	t := s.phaseStart()
	logits := parMatRows(s.M.lmHead(), xf, s.M.Cfg.VocabSize, s.M.Cfg.HiddenSize)
	logitScaleInPlace(logits, s.M.Cfg) // Cohere 0.0625 / Gemma2 logit softcap; no-op for Llama
	s.phaseEnd("lm_head_f32", t)
	return logits
}

// tokenHidden runs one position (absolute index pos, embedding-looked-up hidden x)
// through all layers against the cache, appending this position's K/V, and returns
// the post-final-norm hidden vector (NOT yet projected to logits). This is the single
// shared code path for prefill and decode; the head is applied by the caller.
func (s *Session) tokenHidden(id, pos int) []float32 {
	if s.Quant {
		return s.tokenHiddenQ(id, pos)
	}
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize

	embed := m.embedRows()
	x := append([]float32(nil), embed[id*H:(id+1)*H]...)
	scaleEmbedInPlace(x, cfg) // Gemma sqrt(hidden); no-op for Llama

	for l := 0; l < cfg.NumLayers; l++ {
		cos, sin := ropeRowForLayer(cfg, l, pos)
		x = s.blockStep(l, pos, x, cos, sin, f32Kernel{m})
	}
	s.Cache.pos = append(s.Cache.pos, pos)
	return m.finalNorm(x)
}

// blockStep is the single-position decoder block: pre-attn norm, q/k/v, RoPE, cache
// append, causal GQA, output projection, then SwiGLU MLP. The mat kernel selects f32
// (tokenHidden) vs Q8 (tokenHiddenQ); both share THIS skeleton so the block orchestration
// — the level an architecture axis lives at — exists in exactly one place. Only the
// weight-matmul arithmetic differs by kernel; the RMSNorm, RoPE, GQA, residuals, and
// SwiGLU are the identical f32 math for both, so the f32 path stays bit-exact and the
// Q8 path stays within its own argmax/cosine gate.
func (s *Session) blockStep(l, qpos int, x, cos, sin []float32, mat matKernel) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	p := func(str string) string { return layerName(l, str) }

	// MLP / FFN. The FFN sub-layer is the one architecture axis dispatched here
	// (dense SwiGLU vs MoE). Dense (NumExperts==0) lowers to the same gate/up/down
	// SwiGLU through the same mat kernel; MoE changes only this residual delta and
	// never the attention/cache path above.
	mlpBody := func(xn []float32) []float32 {
		t := s.phaseStart()
		out := m.ffnForLayer(l).apply(m, l, mat.prep(xn), mat)
		s.phaseEnd("mlp_decode", t)
		return out
	}
	runBlock := func(attnBody sublayer) []float32 {
		attnNorm := m.attentionNorms(l)
		mlpNorm := attnNorm
		if cfg.BlockTopology == ParallelResidual {
			mlpNorm = m.parallelMLPNorms(l, attnNorm)
		} else {
			mlpNorm = m.mlpNorms(l)
		}
		composeBlock(cfg.BlockTopology, x, attnNorm, mlpNorm, eps, cfg, attnBody, mlpBody)
		return x
	}
	if cfg.isLinearAttnLayer(l) {
		return runBlock(func(xn []float32) []float32 {
			return s.linearAttnStep(l, xn, mat)
		})
	}

	// attnBody runs attention on an already-normalized input and returns the raw
	// output-projection result (pre residual/post-norm). It appends THIS position's
	// K/V to the kernel-owned cache exactly as before — the cache writes are part of
	// attention and run once per block regardless of topology.
	attnBody := func(xn []float32) []float32 {
		t := s.phaseStart()
		xp := mat.prep(xn)
		qWidth := nH * hd
		var q []float32
		var gate []float32
		if cfg.AttnOutputGate {
			qf := mat.mul(p("self_attn.q_proj.weight"), xp, 2*qWidth, H)
			q = make([]float32, qWidth)
			gate = make([]float32, qWidth)
			for h := 0; h < nH; h++ {
				copy(q[h*hd:(h+1)*hd], qf[h*2*hd:h*2*hd+hd])
				copy(gate[h*hd:(h+1)*hd], qf[h*2*hd+hd:h*2*hd+2*hd])
			}
		} else {
			q = mat.mul(p("self_attn.q_proj.weight"), xp, qWidth, H)
		}
		kk := mat.mul(p("self_attn.k_proj.weight"), xp, w, H)
		vv := mat.mul(p("self_attn.v_proj.weight"), xp, w, H)
		s.phaseEnd("full_attn_qkv_proj", t)
		t = s.phaseStart()
		m.applyProjBias(l, q, kk, vv)
		// qk-norm AFTER projection, BEFORE RoPE; no-op for Llama.
		m.applyLayerQKNorm(l, q, kk)
		// RoPE q and k per head at this position, stashing the PRE-RoPE, post-qk-norm K
		// first so a later Evict can reposition this entry in a single rotation.
		if cfg.Alibi {
			s.Cache.Kraw[l] = append(s.Cache.Kraw[l], kk...)
		} else {
			s.ropeRowQK(l, q, kk, cos, sin)
		}
		// append this position's (post-RoPE) K/V to the kernel-owned cache
		s.Cache.K[l] = append(s.Cache.K[l], kk...)
		s.Cache.V[l] = append(s.Cache.V[l], vv...)
		s.phaseEnd("full_attn_qk_norm_rope", t)

		nPos := len(s.Cache.K[l]) / w
		// SWA read-time mask: query (the row just appended, at absolute position qpos)
		// attends only keys whose absolute position is >= qpos-W+1. lo=0 (full causal)
		// when W<0. Keyed off pos[] so it stays correct after an Evict compaction.
		lo := windowLoStep(s.Cache.pos, nPos, qpos, cfg.windowForLayer(l))
		attnOut := make([]float32, nH*hd)
		// One reused scores scratch for all heads this step (lo/nPos are head-independent);
		// grow() keeps amortized total allocation O(n) instead of the O(n²) a per-head make
		// would cost. Fully overwritten per head below, so reuse is bit-identical.
		s.decodeScores = grow(s.decodeScores, nPos-lo)
		t = s.phaseStart()
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[h*hd : (h+1)*hd]
			scores := s.decodeScores
			for j := lo; j < nPos; j++ {
				kh := s.Cache.K[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				scores[j-lo] = dot(qh, kh)*scale + cfg.alibiScoreBias(h, j, nPos)
			}
			softcapInPlace(scores, attnCap)
			m.softmaxAttentionScores(l, h, scores)
			if m.attnObs != nil { // #852: emit the post-softmax row (copy-out, math untouched)
				emitAttnRow(m.attnObs, l, qpos, h, lo, scores)
			}
			out := attnOut[h*hd : (h+1)*hd]
			for j := lo; j < nPos; j++ {
				vh := s.Cache.V[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				wj := scores[j-lo]
				for d := 0; d < hd; d++ {
					out[d] += wj * vh[d]
				}
			}
		}
		s.phaseEnd("full_attn_decode", t)
		if cfg.AttnOutputGate {
			t = s.phaseStart()
			for i := 0; i < qWidth; i++ {
				attnOut[i] *= sigmoidf(gate[i])
			}
			s.phaseEnd("full_attn_gate", t)
		}
		t = s.phaseStart()
		ao := mat.prep(attnOut)
		out := mat.mul(p("self_attn.o_proj.weight"), ao, H, nH*hd)
		m.addBiasIfPresent(out, p("self_attn.o_proj.bias"))
		s.phaseEnd("full_attn_o_proj", t)
		return out
	}
	return runBlock(attnBody)
}

// ropeRowQK applies RoPE to one position's q (nH heads) and k (nKV heads) in place,
// stashing the PRE-RoPE k into layer l's Kraw FIRST so KVCache.Evict can reposition a
// survivor in a single rotation. This is the single-row rotate-and-stash that every
// per-position site funnels through (decode f32/Q8, multi-user decode, profiling),
// so a RoPE-convention change lands in one place rather than ~5 hand-copies.
func (s *Session) ropeRowQK(l int, q, k, cos, sin []float32) {
	// stash PRE-RoPE k first (rotation below mutates k in place)
	s.Cache.Kraw[l] = append(s.Cache.Kraw[l], k...)
	ropeRowQKInto(q, k, cos, sin, s.M.Cfg.HeadDim, s.M.Cfg.NumHeads, s.M.Cfg.NumKVHeads)
}

// ropeRowQKInto is the operand-only form: rotate q's nH heads and k's nKV heads in
// place at one position. The Kraw stash is intentionally NOT here — the f32/Q8 decode
// paths stash k BEFORE rotation (ropeRowQK orders that correctly), while a caller that
// stashes pre-RoPE k itself (e.g. a panel path that batched the stash) calls this
// directly after its own append.
func ropeRowQKInto(q, k, cos, sin []float32, hd, nH, nKV int) {
	for h := 0; h < nH; h++ {
		applyRopeRow(q[h*hd:(h+1)*hd], cos, sin)
	}
	for h := 0; h < nKV; h++ {
		applyRopeRow(k[h*hd:(h+1)*hd], cos, sin)
	}
}

// Prefill ingests a prompt and returns the logits of its LAST token (the
// distribution over the first generated token). Each token is placed at the next
// absolute position (Cache.Len()), so a prior Evict() compaction shifts these down.
func (s *Session) Prefill(ids []int) []float32 {
	if len(ids) == 0 {
		return nil
	}
	if s.M.Cfg.isGLMMoeDsa() {
		s.requireGLMDsaSession()
		var last []float32
		for _, id := range ids {
			last = s.tokenHiddenGLMDsa(id, s.Cache.Len())
		}
		return s.glmDsaHead(last)
	}
	if s.M.Cfg.isMiniMaxSparseAttn() {
		// MiniMax-M3 MSA: the incremental cache path runs the lightning-indexer block
		// selection per position over the cached K/V (minimax_m3_session.go), so cached
		// decode/prefix-reuse agree with the cacheless Forward. It must precede the generic
		// MoE token-loop below, which would otherwise run dense GQA on the sparse layers.
		s.requireMiniMaxSession()
		var last []float32
		for _, id := range ids {
			last = s.tokenHiddenMiniMax(id, s.Cache.Len())
		}
		return s.head(last)
	}
	if s.Q4 {
		// Resident int4 prefill: the batched Q8 GEMM has no int4 twin yet, so prefill runs
		// the shared per-token blockStep with the int4 kernel. Slower than batched but uses
		// only the resident int4 weights (the lean q4-only mode freed the Q8_0 copy).
		var last []float32
		for _, id := range ids {
			last = s.tokenHiddenQ(id, s.Cache.Len())
		}
		return s.headQ4(last)
	}
	if s.Q4K {
		// Resident Q4_K prefill (plan P1/P3). For a PreNorm standard-attention model (the
		// q4_k_m regime the plan targets), run the BATCHED q4 GEMM: q4_k_m majority via
		// q4kGemm, Q6_K / normalize-sensitive minority via the proven qGemm8, KV filled in
		// one pass — each weight super-block dequantized once and reused across all P prompt
		// tokens. Architectures the batched q4 lane does not yet cover (MoE / DenseMLP /
		// Alibi / Qwen35-hybrid / non-PreNorm / layer-specific RoPE theta) fall back to the
		// per-token blockStep, exactly as the Q8 token-loop fallback does. The LM head is
		// whichever resident format it loaded as (headResident).
		if !q8PrefillNeedsTokenLoop(s.M.Cfg) {
			return s.headResident(s.prefillBatchedQ4K(ids))
		}
		// Qwen3.5/3.6 hybrid (the q8PrefillNeedsTokenLoop case the generic batched-Q4K lane
		// refuses): batch each layer's projection/MLP GEMMs over the prompt panel while keeping
		// the GDN recurrence, the resident-Q4K twin of the q8Qwen35HybridPrefillOK gate. Closes
		// QWEN36-NATIVE-PERF-PLAN P3's per-token-fallback prefill wall.
		if q4kQwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			return s.prefillQwen35HybridQ4K(ids)
		}
		var last []float32
		for _, id := range ids {
			last = s.tokenHiddenQ(id, s.Cache.Len())
		}
		return s.headResident(last)
	}
	if s.GPTQ {
		var last []float32
		for _, id := range ids {
			last = s.tokenHiddenGPTQ(id, s.Cache.Len())
		}
		return s.headResident(last)
	}
	if s.Backend != nil {
		s.requirePreNorm("HAL prefill")
		return s.prefillHAL(ids, true)
	}
	if s.Metal {
		s.requirePreNorm("Metal prefill")
		// Prefill projections on the GPU; the head stays the cheap CPU single-token GEMV.
		return s.headQ(s.prefillBatchedMetal(ids))
	}
	if s.PrecisionPolicy != nil {
		return s.prefillDynamic(ids)
	}
	if s.Quant {
		if q8Qwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			return s.prefillQwen35HybridQ(ids)
		}
		if cfg := s.M.Cfg; q8PrefillNeedsTokenLoop(cfg) {
			var last []float32
			for _, id := range ids {
				last = s.tokenHiddenQ(id, s.Cache.Len())
			}
			return s.headQ(last)
		}
		return s.headQ(s.prefillBatchedQ(ids))
	}
	if s.M.Cfg.IsMoE() || s.M.Cfg.DenseMLP || s.M.Cfg.Alibi || s.M.Cfg.IsQwen35Hybrid() || s.M.Cfg.AttnOutputGate || s.M.Cfg.BlockTopology != PreNorm || s.M.Cfg.hasLayerSpecificRopeTheta() {
		// The batched f32 GEMM is one-weight-many-rows and still hardcodes the PreNorm
		// block copy and one shared RoPE table. MoE routes each token to its own top-k
		// experts, DenseMLP removes the up-projection, ALiBi replaces RoPE with score
		// bias, non-PreNorm topology changes the residual/norm graph, and Gemma3-style
		// per-layer RoPE theta changes the rotation by layer; these axes run through
		// blockStep here, where the FFN/topology/RoPE dispatch lives.
		var last []float32
		for _, id := range ids {
			last = s.tokenHidden(id, s.Cache.Len())
		}
		return s.head(last)
	}
	// PreNorm (default): batched + parallel, one GEMM over all P tokens instead of
	// GEMV-per-token. Bit-identical to the per-token tokenHidden loop
	// (TestPrefillBatchedMatchesSerial), so the cache it builds is exactly the proven
	// one and R2/R3/R14 stay exact.
	return s.head(s.prefillBatched(ids))
}

// PrefillNoLogits ingests a prompt exactly like Prefill but discards the final-token
// distribution. It is for teacher-forced context growth where the caller already knows the
// next input token and only needs KV state advanced.
func (s *Session) PrefillNoLogits(ids []int) {
	if len(ids) == 0 {
		return
	}
	if s.M.Cfg.isGLMMoeDsa() {
		s.requireGLMDsaSession()
		for _, id := range ids {
			s.tokenHiddenGLMDsa(id, s.Cache.Len())
		}
		return
	}
	if s.M.Cfg.isMiniMaxSparseAttn() {
		s.requireMiniMaxSession()
		for _, id := range ids {
			s.tokenHiddenMiniMax(id, s.Cache.Len())
		}
		return
	}
	if s.Q4 {
		for _, id := range ids {
			s.tokenHiddenQ(id, s.Cache.Len())
		}
		return
	}
	if s.Q4K {
		if !q8PrefillNeedsTokenLoop(s.M.Cfg) {
			s.prefillBatchedQ4K(ids)
			return
		}
		if q4kQwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			s.prefillQwen35HybridQ4KNoLogits(ids)
			return
		}
		for _, id := range ids {
			s.tokenHiddenQ(id, s.Cache.Len())
		}
		return
	}
	if s.GPTQ {
		for _, id := range ids {
			s.tokenHiddenGPTQ(id, s.Cache.Len())
		}
		return
	}
	if s.Backend != nil {
		s.requirePreNorm("HAL prefill")
		s.prefillHAL(ids, false)
		return
	}
	if s.PrecisionPolicy != nil {
		s.Prefill(ids)
		return
	}
	if s.Metal {
		s.requirePreNorm("Metal prefill")
		s.prefillBatchedMetal(ids)
		return
	}
	if s.Quant {
		if q8Qwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			s.prefillQwen35HybridQNoLogits(ids)
			return
		}
		if q8PrefillNeedsTokenLoop(s.M.Cfg) {
			for _, id := range ids {
				s.tokenHiddenQ(id, s.Cache.Len())
			}
			return
		}
		s.prefillBatchedQ(ids)
		return
	}
	if s.M.Cfg.IsMoE() || s.M.Cfg.DenseMLP || s.M.Cfg.Alibi || s.M.Cfg.IsQwen35Hybrid() || s.M.Cfg.AttnOutputGate || s.M.Cfg.BlockTopology != PreNorm || s.M.Cfg.hasLayerSpecificRopeTheta() {
		for _, id := range ids {
			s.tokenHidden(id, s.Cache.Len())
		}
		return
	}
	s.prefillBatched(ids)
}

func q8PrefillNeedsTokenLoop(cfg Config) bool {
	return cfg.IsMoE() || cfg.DenseMLP || cfg.Alibi || cfg.IsQwen35Hybrid() || cfg.AttnOutputGate || cfg.BlockTopology != PreNorm || cfg.hasLayerSpecificRopeTheta()
}

// Step decodes one already-chosen token and returns the next-token logits. Quantized
// sessions reuse their logits buffer; consume or copy the returned slice before the next
// quantized Prefill/Step call on the same session.
func (s *Session) Step(id int) []float32 {
	if s.M.Cfg.isGLMMoeDsa() {
		s.requireGLMDsaSession()
		return s.glmDsaHead(s.tokenHiddenGLMDsa(id, s.Cache.Len()))
	}
	if s.M.Cfg.isMiniMaxSparseAttn() {
		s.requireMiniMaxSession()
		return s.head(s.tokenHiddenMiniMax(id, s.Cache.Len()))
	}
	if s.Backend != nil {
		return s.token(id, s.halKV.Len())
	}
	if s.PrecisionPolicy != nil {
		return s.stepDynamic(id)
	}
	return s.token(id, s.Cache.Len())
}

// Generate greedily decodes n tokens after the prompt and returns their ids.
func (s *Session) Generate(prompt []int, n int) []int {
	logits := s.Prefill(prompt)
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		next := argmaxF32(logits)
		out = append(out, next)
		if s.M.Cfg.IsEOS(next) {
			break
		}
		logits = s.Step(next)
	}
	return out
}

func argmaxF32(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}
