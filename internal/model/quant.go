package model

// quant.go — the Q8_0 quantized inference lane: close the raw-throughput gap to
// llama.cpp on its OWN terms (same quantization format), in pure Go, without spending
// a single bit of the proven f32 path.
//
// Why this is the lever. MODEL-BASELINE-RESULTS.md (Act 1/2) proved decode is
// memory-bandwidth-bound at 0.50 flop/byte: per generated token fak streams all ~537 MB
// of f32 weights, and time ≈ bytes / bandwidth. The entire residual gap to llama.cpp is
// that llama.cpp streams ~4× fewer weight bytes (Q8_0: int8 codes + a per-32-block scale
// ≈ 1.06 B/weight vs f32's 4) with vectorized integer dot products. So the parity move is
// not "write assembly" — it is "stream the same number of bytes llama.cpp does." This
// lane quantizes the weight matmuls to Q8_0 (the identical block format llama.cpp's
// SmolLM2 GGUF uses), making the head-to-head a true apples-to-apples Q8_0-vs-Q8_0
// comparison rather than f32-vs-Q8.
//
// What stays sacred. This is an OPT-IN mode (Session.Quant). The f32 path — every
// matRows/parMatRows/matMulBatch and the tokenHidden/prefillBatched loops — is left
// byte-for-byte untouched, so all the exact fak-vs-fak rungs (R2 max|Δ|=0, R14 d==0) and
// the HF oracle (argmax-exact, max|Δ|<0.05) stay green on f32. The quantized mode carries
// its OWN, honest correctness gate (greedy-continuation agreement with the HF oracle —
// exactly how llama.cpp quality is judged), because quantization is lossy by construction
// and asserting f32 bit-identity of it would be a lie.
//
// What is and isn't quantized. Only the WEIGHT matmuls (q/k/v/o, gate/up/down, the LM
// head) are quantized — that is the ~537 MB that dominates the per-token byte stream. The
// KV cache stays f32: it is the kernel-owned object the whole fusion is about (Evict /
// Clone operate on it), it is tiny at these sequence lengths (L2-resident, the profiler
// measured attn at 6.5 GB/s = latency-bound, not bandwidth-bound), and quantizing it would
// trade the security primitive's exactness for no bandwidth win. Embedding LOOKUP stays
// f32 too (it is a row gather, not a matmul); only the tied matrix's use AS the head is
// quantized.

// qBlk is the Q8_0 block size: 32 weights share one f32 scale. This is llama.cpp's
// GGML_QUANT_BLOCK for Q8_0, so the format (and thus the quantization error) matches the
// GGUF fak is benchmarked against. Every *reduction* dim a Q8_0 dot runs over in
// SmolLM2-135M is a multiple of 32 — hidden 576 (q/k/v/gate/up/o_proj over nH·hd=576, and
// the head), intermediate 1536 (down_proj) — so blocks divide evenly with no tail.
// (Output widths like the kv projection's 192 are never reduced over; quantizeQ8/quantizeVecQ8
// panic if ever handed a non-multiple-of-32 reduction dim, rather than silently dropping it.)
const qBlk = 32

// q8Tensor is a Q8_0-quantized weight matrix [out, in], in == nblk*qBlk. Row-major: row
// o is nblk blocks of 32 int8 codes (q) each with one f32 scale (d). Codes and scales are
// kept in separate slices (d is 1/32 the size); a row's scales are contiguous so they
// prefetch alongside its codes. Per-weight bytes streamed: 1 (int8) + 4/32 (scale) =
// 1.125 — vs f32's 4, the ~3.6× decode-bandwidth win.
type q8Tensor struct {
	out, in, nblk int
	q             []int8    // out*in codes, row-major
	d             []float32 // out*nblk per-block scales
}

// q8Vec is a Q8_0-quantized activation vector (len == nblk*qBlk). Activations are
// quantized too so the inner product is a pure int8×int8→int32 reduction (vectorizable,
// the path llama.cpp's vec_dot takes) rather than an int8→float convert per weight.
type q8Vec struct {
	q    []int8
	q16  []int16
	d    []float32
	nblk int
}

// q8round rounds to nearest (ties away from zero, matching C roundf / llama.cpp) and
// clamps into the int8 code range. The clamp only ever fires on FP rounding at the block
// max (which maps to ±127 by construction), never as real saturation.
//
// This is pure float32 arithmetic — NOT math.Round(float64(x)) — because it runs ~25M times
// per prefill (every weight AND activation code) and the float64 round was a measurable chunk
// of the quantization phase. It truncates toward zero and inspects the EXACT fractional part
// (x - float32(t) is exact for |x|≤127 by Sterbenz), rounding the half away from zero — so it
// is byte-for-byte identical to math.Round on this range. (The naive `int8(int32(x+0.5))`
// trick is WRONG: for x just below 0.5 the addition itself rounds up to 1.0 and yields code 1
// where math.Round gives 0; this avoids that by never rounding the +0.5.)
//
// //go:noinline is REQUIRED for determinism. Inlined into `q8round(blk[i]*inv)`, the arm64 Go
// compiler FMA-contracts `blk[i]*inv - float32(t)`, taking the fractional part from the UNROUNDED
// product — so an exact tie like 82.5 rounds to 82 (down) on arm64 while the same value gives 83
// (ties-away) on amd64 and when q8round is called standalone. Forcing a call boundary rounds the
// product to float32 first, making the rounding deterministic ties-away across arches AND letting
// the NEON quantizer (separate FMUL→FRINTA — the true IEEE result) match this reference bit-for-bit.
// Cost: a call per code on the SCALAR fallback; the hot quantizer path is NEON.
//
//go:noinline
func q8round(x float32) int8 {
	t := int32(x) // truncate toward zero
	f := x - float32(t)
	if f >= 0.5 {
		t++
	} else if f <= -0.5 {
		t--
	}
	if t > 127 {
		return 127
	}
	if t < -127 {
		return -127
	}
	return int8(t)
}

// quantizeQ8 converts a row-major f32 weight matrix [out,in] to Q8_0. Each 32-wide block
// gets d = maxabs/127 and codes q = round(w/d); a zero block (d==0) stays all-zero. Built
// once at load (Model.Quantize), parallel across rows — quantization itself is not on the
// hot path.
func quantizeQ8(w []float32, out, in int) *q8Tensor {
	if in%qBlk != 0 {
		// Without this guard a non-multiple-of-32 reduction dim would SILENTLY drop the
		// tail (in mod 32) elements from every dot — a wrong result, not a crash. Every
		// SmolLM2/Qwen reduction dim is a multiple of 32, so this only fires on a model
		// that violates the Q8_0 block precondition; fail loudly instead.
		panic("model: Q8_0 reduction dim not a multiple of 32")
	}
	nblk := in / qBlk
	qt := &q8Tensor{out: out, in: in, nblk: nblk, q: make([]int8, out*in), d: make([]float32, out*nblk)}
	parFor(out, numWorkers, func(lo, hi int) {
		for o := lo; o < hi; o++ {
			row := w[o*in : o*in+in]
			for b := 0; b < nblk; b++ {
				blk := row[b*qBlk : b*qBlk+qBlk]
				var amax float32
				for _, v := range blk {
					a := v
					if a < 0 {
						a = -a
					}
					if a > amax {
						amax = a
					}
				}
				d := amax / 127
				qt.d[o*nblk+b] = d
				base := o*in + b*qBlk
				if d == 0 {
					continue // codes already zero
				}
				inv := 1.0 / d
				for i := 0; i < qBlk; i++ {
					qt.q[base+i] = q8round(blk[i] * inv)
				}
			}
		}
	})
	return qt
}

// quantizeVecQ8 quantizes one activation vector to Q8_0 (len must be a multiple of qBlk).
func quantizeVecQ8(x []float32) q8Vec {
	var qv q8Vec
	return quantizeVecQ8Into(&qv, x)
}

func quantizeVecQ8Into(qv *q8Vec, x []float32) q8Vec {
	in := len(x)
	if in%qBlk != 0 {
		panic("model: Q8_0 activation length not a multiple of 32")
	}
	nblk := in / qBlk
	if cap(qv.q) < in {
		qv.q = make([]int8, in)
	} else {
		qv.q = qv.q[:in]
	}
	if cap(qv.d) < nblk {
		qv.d = make([]float32, nblk)
	} else {
		qv.d = qv.d[:nblk]
	}
	qv.nblk = nblk
	quantizeRowQ8(x, qv.q, qv.d, nblk)
	if q8PreextendVec() {
		if cap(qv.q16) < in {
			qv.q16 = make([]int16, in)
		} else {
			qv.q16 = qv.q16[:in]
		}
		extendQ8ToQ16(qv.q, qv.q16, nblk)
	} else {
		qv.q16 = qv.q16[:0]
	}
	return *qv
}

func extendQ8ToQ16Scalar(q []int8, q16 []int16, nblk int) {
	for i, v := range q[:nblk*qBlk] {
		q16[i] = int16(v)
	}
}

// qdot8scalar is the portable Q8_0 inner product: Σ_blocks (dw_b·dx_b)·Σ_{i∈block}(qw_i·qx_i).
// The inner 32-wide reduction is int8×int8→int32 with 4 independent accumulators, then one
// float multiply by the two block scales. The int32 per-block sum is bounded by
// 32·127·127 ≈ 5.2e5, far inside int32. Block results are summed into a float accumulator
// in fixed order (block 0,1,2,…, and (((float(isum)·dw)·dx) per block), so qdot8 is
// deterministic AND the amd64 SIMD kernel can be held BIT-IDENTICAL to it: integer
// addition is associative with no overflow here, so the SIMD int dot computes the same
// int32 isum regardless of lane order, and matching this exact float-combine order makes
// asm == scalar bit-for-bit (TestQdot8AsmMatchesScalar). qdot8 (the dispatched entry
// point) lives in quant_amd64.go / quant_noasm.go.
func qdot8scalar(qw []int8, dw []float32, qv q8Vec, nblk int) float32 {
	var acc float32
	qx := qv.q
	dx := qv.d
	for b := 0; b < nblk; b++ {
		wb := qw[b*qBlk:]
		xb := qx[b*qBlk:]
		var s0, s1, s2, s3 int32
		for i := 0; i < qBlk; i += 4 {
			s0 += int32(wb[i]) * int32(xb[i])
			s1 += int32(wb[i+1]) * int32(xb[i+1])
			s2 += int32(wb[i+2]) * int32(xb[i+2])
			s3 += int32(wb[i+3]) * int32(xb[i+3])
		}
		acc += float32((s0+s1)+(s2+s3)) * dw[b] * dx[b]
	}
	return acc
}

// qMatRows is the quantized decode GEMV: y[o] = qdot8GEMV(weight row o, qv). Row-parallel
// exactly like parMatRows — decode is memory-bound, so spreading rows across cores taps
// aggregate bandwidth; with int8 weights each core now streams ~3.6× fewer bytes.
func qMatRows(qt *q8Tensor, qv q8Vec) []float32 {
	y := make([]float32, qt.out)
	qMatRowsInto(qt, qv, y)
	return y
}

func qMatRowsInto(qt *q8Tensor, qv q8Vec, y []float32) {
	y = y[:qt.out]
	if numWorkers <= 1 || qt.out*qt.in < parThreshold {
		qMatRowsRange(qt, qv, y, 0, qt.out)
		return
	}
	body := func(lo, hi int) {
		qMatRowsRange(qt, qv, y, lo, hi)
	}
	parFor(qt.out, numWorkers, body)
}

func qMatRowsRange(qt *q8Tensor, qv q8Vec, y []float32, lo, hi int) {
	if qMatRowsRangeFast(qt, qv, y, lo, hi) {
		return
	}
	for o := lo; o < hi; o++ {
		y[o] = qdot8GEMV(qt.q[o*qt.in:o*qt.in+qt.in], qt.d[o*qt.nblk:o*qt.nblk+qt.nblk], qv, qt.nblk)
	}
}

// The batched prefill GEMM moved to quant_gemm.go (qGemm8 + the register-blocked tile
// kernel). qMatRows above stays the decode GEMV; the prefill path no longer does a
// per-element qdot8 sweep.

// Quantize builds the Q8_0 copy of every weight the quantized forward path uses (the
// per-layer projections, FFN/router/expert weights, and the LM head), once, storing it on
// the Model. Idempotent. The f32 blob is retained (the f32 path and the oracle still need
// it); the int8 copy adds ~134 MB for dense checkpoints. Call before running any Quant
// session.
func (m *Model) Quantize() {
	if m.q8w != nil {
		return
	}
	qm := make(map[string]*q8Tensor)
	add := func(name string) *q8Tensor {
		meta, ok := m.manifest[name]
		if !ok {
			panic("model: Quantize missing tensor " + name)
		}
		if len(meta.Shape) != 2 {
			panic("model: Quantize non-2D tensor " + name)
		}
		out, in := meta.Shape[0], meta.Shape[1]
		qt := quantizeQ8(m.tensor(name), out, in)
		qm[name] = qt
		return qt
	}
	addIfPresent := func(name string) {
		if m.has(name) {
			add(name)
		}
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		p := func(s string) string { return layerName(l, s) }
		if m.Cfg.isGLMMoeDsa() {
			add(p("self_attn.q_a_proj.weight"))
			add(p("self_attn.q_b_proj.weight"))
			add(p("self_attn.kv_a_proj_with_mqa.weight"))
			add(p("self_attn.kv_b_proj.weight"))
			add(p("self_attn.o_proj.weight"))
			if glmDsaIndexerIsShared(m.Cfg, l) {
				addIfPresent(p("self_attn.indexer.wq_b.weight"))
				addIfPresent(p("self_attn.indexer.wk.weight"))
				addIfPresent(p("self_attn.indexer.weights_proj.weight"))
			} else {
				add(p("self_attn.indexer.wq_b.weight"))
				add(p("self_attn.indexer.wk.weight"))
				add(p("self_attn.indexer.weights_proj.weight"))
			}
		} else if m.Cfg.isLinearAttnLayer(l) {
			add(p("linear_attn.in_proj_qkv.weight"))
			add(p("linear_attn.in_proj_z.weight"))
			add(p("linear_attn.in_proj_a.weight"))
			add(p("linear_attn.in_proj_b.weight"))
			add(p("linear_attn.out_proj.weight"))
		} else {
			add(p("self_attn.q_proj.weight"))
			add(p("self_attn.k_proj.weight"))
			add(p("self_attn.v_proj.weight"))
			add(p("self_attn.o_proj.weight"))
		}
		if m.has(routerName(l)) {
			add(routerName(l))
			for e := 0; e < m.Cfg.NumExperts; e++ {
				add(expertName(l, e, "gate_proj.weight"))
				add(expertName(l, e, "up_proj.weight"))
				add(expertName(l, e, "down_proj.weight"))
			}
			if m.Cfg.isGLMMoeDsa() {
				addIfPresent(p("mlp.shared_experts.gate_proj.weight"))
				addIfPresent(p("mlp.shared_experts.up_proj.weight"))
				addIfPresent(p("mlp.shared_experts.down_proj.weight"))
			}
		} else if m.Cfg.DenseMLP {
			add(p("mlp.gate_proj.weight"))
			add(p("mlp.down_proj.weight"))
		} else {
			add(p("mlp.gate_proj.weight"))
			add(p("mlp.up_proj.weight"))
			add(p("mlp.down_proj.weight"))
		}
	}
	add(m.headName())
	m.q8w = qm
	m.initQ8CacheIfComplete()
}

type q8Layer struct {
	qProj, kProj, vProj *q8Tensor
	oProj               *q8Tensor
	gateProj, upProj    *q8Tensor
	downProj            *q8Tensor
	inputNorm, postNorm []float32
	qBias, kBias, vBias []float32
}

// headName returns the tensor used as the LM head (tied embedding unless an explicit
// lm_head exists), as a manifest key for quantization and lookup.
func (m *Model) headName() string {
	if m.has("lm_head.weight") {
		return "lm_head.weight"
	}
	// The memory-lean quant loader drops the f32 lm_head from the manifest but keeps its Q8
	// copy in q8w (untied models — e.g. Qwen2.5-7B). Resolve to it so the quantized head path
	// finds the real head instead of falling through to the tied-embedding key (which is not
	// quantized for an untied model, and would panic in m.q8). m.has() only sees the f32
	// manifest, so this q8w check is what makes an untied model loadable leanly.
	if _, ok := m.q8w["lm_head.weight"]; ok {
		return "lm_head.weight"
	}
	return "model.embed_tokens.weight"
}

// q8 returns the prebuilt Q8_0 tensor for a name (Quantize must have run).
func (m *Model) q8(name string) *q8Tensor {
	qt, ok := m.q8w[name]
	if !ok {
		panic("model: q8 tensor not built: " + name + " (call Model.Quantize)")
	}
	return qt
}

func (m *Model) initQ8Cache() {
	get := func(name string) *q8Tensor {
		qt, ok := m.q8w[name]
		if !ok {
			panic("model: q8 tensor not built: " + name)
		}
		return qt
	}
	layers := make([]q8Layer, m.Cfg.NumLayers)
	for l := 0; l < m.Cfg.NumLayers; l++ {
		p := func(s string) string { return layerName(l, s) }
		ql := q8Layer{
			qProj:     get(p("self_attn.q_proj.weight")),
			kProj:     get(p("self_attn.k_proj.weight")),
			vProj:     get(p("self_attn.v_proj.weight")),
			oProj:     get(p("self_attn.o_proj.weight")),
			gateProj:  get(p("mlp.gate_proj.weight")),
			upProj:    get(p("mlp.up_proj.weight")),
			downProj:  get(p("mlp.down_proj.weight")),
			inputNorm: m.tensor(p("input_layernorm.weight")),
			postNorm:  m.tensor(p("post_attention_layernorm.weight")),
		}
		if m.Cfg.AttentionBias {
			ql.qBias = m.tensor(p("self_attn.q_proj.bias"))
			ql.kBias = m.tensor(p("self_attn.k_proj.bias"))
			ql.vBias = m.tensor(p("self_attn.v_proj.bias"))
		}
		layers[l] = ql
	}
	m.q8layers = layers
	m.q8head = get(m.headName())
}

func (m *Model) initQ8CacheIfComplete() bool {
	has := func(name string) bool {
		return m.q8w[name] != nil
	}
	if !has(m.headName()) {
		return false
	}
	m.q8head = m.q8w[m.headName()]
	if m.Cfg.IsQwen35Hybrid() {
		return false
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		p := func(s string) string { return layerName(l, s) }
		if !has(p("self_attn.q_proj.weight")) ||
			!has(p("self_attn.k_proj.weight")) ||
			!has(p("self_attn.v_proj.weight")) ||
			!has(p("self_attn.o_proj.weight")) ||
			!has(p("mlp.gate_proj.weight")) ||
			!has(p("mlp.up_proj.weight")) ||
			!has(p("mlp.down_proj.weight")) {
			return false
		}
	}
	m.initQ8Cache()
	return true
}

func (m *Model) q8Layer(layer int) *q8Layer {
	if m.q8layers == nil || layer < 0 || layer >= len(m.q8layers) {
		panic("model: q8 layer cache not built (call Model.Quantize)")
	}
	return &m.q8layers[layer]
}

func (m *Model) q8Head() *q8Tensor {
	if m.q8head == nil {
		if m.q8w != nil {
			m.q8head = m.q8w[m.headName()]
		}
		if m.q8head == nil {
			panic("model: q8 head not built (call Model.Quantize)")
		}
	}
	return m.q8head
}
