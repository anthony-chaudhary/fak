package model

import "math"

// Activations is the full-prefill intermediate state the oracle test compares
// against HF. Hidden[l] is the hidden state AFTER layer l-1 (Hidden[0] is the
// embedding output), flattened row-major [seq*hidden] — matching HF's
// output_hidden_states tuple of length NumLayers+1. Logits is [seq][vocab].
type Activations struct {
	Seq    int
	Hidden [][]float32 // [NumLayers+1] each len seq*hidden
	Logits [][]float32 // [seq] each len vocab
}

// rope precomputes cos/sin for every (position, freq) once per forward.
type rope struct {
	cos, sin [][]float32 // [pos][half]
}

func newRope(cfg Config, seq int) rope {
	return newRopeForLayer(cfg, 0, seq)
}

func newRopeForLayer(cfg Config, layer, seq int) rope {
	r := rope{cos: make([][]float32, seq), sin: make([][]float32, seq)}
	inv := invFreq(cfg, layer)
	scale := cfg.ropeAttentionFactor()
	for p := 0; p < seq; p++ {
		r.cos[p], r.sin[p] = ropeRowFromInvScaled(inv, p, scale)
	}
	return r
}

// apply rotates one head vector (len head_dim) in place at position p, using HF's
// non-interleaved "rotate_half" convention:
//
//	out[j]      = x[j]*cos - x[j+half]*sin
//	out[j+half] = x[j+half]*cos + x[j]*sin
func (r rope) apply(hv []float32, p int) {
	applyRopeRow(hv, r.cos[p], r.sin[p])
}

// Forward runs a full-prefill forward pass over token ids and returns every hidden
// state + per-position logits. No KV cache (that is R2); this rung proves the math.
func (m *Model) Forward(ids []int) *Activations {
	cfg := m.Cfg
	H := cfg.HiddenSize
	seq := len(ids)

	// embedding lookup -> x[t] is the working hidden vector for position t.
	embed := m.embedRows()
	x := make([][]float32, seq)
	for t, id := range ids {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], cfg) // Gemma sqrt(hidden); no-op for Llama
	}
	return m.forwardHiddenRows(x)
}

// forwardHiddenRows runs the decoder stack from already-materialized input embeddings.
// Forward builds these rows from token ids; governed multimodal callers may splice in
// externally-produced vision embeddings after admission checks.
func (m *Model) forwardHiddenRows(x [][]float32) *Activations {
	cfg := m.Cfg
	H := cfg.HiddenSize
	seq := len(x)
	act := &Activations{Seq: seq, Hidden: [][]float32{flatten(x)}}
	var glmDsaSharedTopK [][]int
	gemma4 := cfg.isGemma4()
	var gemma4RopeFreqs []float64
	if gemma4 {
		gemma4RopeFreqs = m.gemma4RopeFreqs()
	}
	for l := 0; l < cfg.NumLayers; l++ {
		switch {
		case gemma4:
			// Gemma 4 builds its own per-layer RoPE inside the heterogeneous-geometry
			// path, so the shared per-layer rope table is not used here.
			m.layerGemma4(l, x, gemma4RopeFreqs)
		case cfg.isGLMMoeDsa():
			m.layerGLMDsa(l, x, newRopeForLayer(cfg, l, seq), &glmDsaSharedTopK)
		case cfg.isMiniMaxSparseAttn():
			m.layerMiniMax(l, x, newRopeForLayer(cfg, l, seq))
		default:
			m.layer(l, x, newRopeForLayer(cfg, l, seq))
		}
		act.Hidden = append(act.Hidden, flatten(x))
	}

	// final norm + tied LM head
	mat := residentKernel{m}
	act.Logits = make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xf := m.finalNorm(x[t])
		logits := mat.mul(m.headName(), mat.prep(xf), cfg.VocabSize, H)
		logitScaleInPlace(logits, cfg) // Cohere/Gemma2; no-op for Llama
		act.Logits[t] = logits
	}
	return act
}

// layer applies one decoder block to x in place. The default (Llama, PreNorm) is
// attention + MLP, each with a pre-norm and a residual. cfg.BlockTopology selects
// the norm placement / residual wiring (arch.go); PreNorm lowers to the verbatim
// Llama instruction stream so the oracle rungs stay bit-exact.
func (m *Model) layer(l int, x [][]float32, rp rope) {
	cfg := m.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	seq := len(x)
	attnNorm := m.attentionNorms(l)
	topo := cfg.BlockTopology

	// attnSub returns the per-position attention output projections for the given
	// per-position normalized inputs (one normalized vector per position). It is the
	// whole-sequence attention body — the norm placement is owned by the topology
	// composition below, so this consumes already-normalized inputs and never norms.
	attnSub := func(xn [][]float32) [][]float32 { return m.attnSeq(l, xn, rp) }
	// mlpSub returns the per-position SwiGLU MLP outputs for normalized inputs.
	mlpSub := func(xn [][]float32) [][]float32 { return m.mlpSeq(l, xn) }

	// Qwen3.5/Qwen3-Next hybrid: a linear_attention layer swaps the attention token mixer
	// for the Gated-DeltaNet recurrent scan (qwen35.go), keeping the PreNorm + SwiGLU wiring.
	if cfg.isLinearAttnLayer(l) {
		// FAK_GDN_BATCHED routes the Gated-DeltaNet prefill through the batched-projection
		// path (issue #443); it is bit-identical to linearAttnSeq on the f32 path, certified
		// by TestQwen35LinearAttnBatchedMatchesScalar, so the opt-in is witness-gated.
		linAttnSub := func(xn [][]float32) [][]float32 {
			if gdnBatchedPrefill {
				return m.linearAttnSeqBatched(l, xn)
			}
			return m.linearAttnSeq(l, xn)
		}
		composeSeqSublayer(topo, x, attnNorm, eps, cfg, linAttnSub)
		composeSeqSublayer(topo, x, m.mlpNorms(l), eps, cfg, mlpSub)
		return
	}

	if topo == ParallelResidual {
		// Both branches read the original residual. GPT-NeoX has separate attention
		// and MLP LayerNorms; Cohere reuses the attention norm when the MLP norm is
		// absent.
		mlpNorm := m.parallelMLPNorms(l, attnNorm)
		o := attnSub(normSeq(x, attnNorm, eps, cfg))
		d := mlpSub(normSeq(x, mlpNorm, eps, cfg))
		for t := 0; t < seq; t++ {
			for i := 0; i < H; i++ {
				x[t][i] += o[t][i] + d[t][i]
			}
		}
		return
	}
	mlpNorm := m.mlpNorms(l)
	composeSeqSublayer(topo, x, attnNorm, eps, cfg, attnSub)
	composeSeqSublayer(topo, x, mlpNorm, eps, cfg, mlpSub)
}

func (m *Model) layerGLMDsa(l int, x [][]float32, rp rope, sharedTopK *[][]int) {
	cfg := m.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	seq := len(x)
	attnNorm := m.attentionNorms(l)
	topo := cfg.BlockTopology
	attnSub := func(xn [][]float32) [][]float32 {
		return m.glmDsaAttnSeqShared(l, xn, sharedTopK)
	}
	mlpSub := func(xn [][]float32) [][]float32 { return m.mlpSeq(l, xn) }

	if topo == ParallelResidual {
		mlpNorm := m.parallelMLPNorms(l, attnNorm)
		o := attnSub(normSeq(x, attnNorm, eps, cfg))
		d := mlpSub(normSeq(x, mlpNorm, eps, cfg))
		for t := 0; t < seq; t++ {
			for i := 0; i < H; i++ {
				x[t][i] += o[t][i] + d[t][i]
			}
		}
		return
	}
	mlpNorm := m.mlpNorms(l)
	composeSeqSublayer(topo, x, attnNorm, eps, cfg, attnSub)
	composeSeqSublayer(topo, x, mlpNorm, eps, cfg, mlpSub)
}

// attnSeq computes causal GQA attention over a whole sequence of already-normalized
// inputs and returns the per-position output-projection results (pre residual).
func (m *Model) attnSeq(l int, xn [][]float32, rp rope) [][]float32 {
	cfg := m.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	if cfg.isGLMMoeDsa() {
		return m.glmDsaAttnSeqShared(l, xn, nil)
	}
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	seq := len(xn)
	attnCap := float32(cfg.AttnSoftcap)
	p := func(s string) string { return layerName(l, s) }
	mat := residentKernel{m}

	// per-position q,k,v after norm + projection (+ optional bias, + optional qk-norm).
	q := make([][]float32, seq) // [seq][nH*hd]
	k := make([][]float32, seq) // [seq][nKV*hd]
	v := make([][]float32, seq) // [seq][nKV*hd]
	// Qwen3.5/Qwen3-Next full-attention layers project a doubled q_proj and split it,
	// per head, into the query and a sigmoid output gate ([query|gate] interleaved by head);
	// the gate multiplies the attention output before o_proj. Off (default) = Llama path.
	gated := cfg.AttnOutputGate
	qWidth := nH * hd
	if gated {
		qWidth = 2 * nH * hd
	}
	var gates [][]float32
	if gated {
		gates = make([][]float32, seq)
	}
	for t := 0; t < seq; t++ {
		xp := mat.prep(xn[t])
		if gated {
			qf := mat.mul(p("self_attn.q_proj.weight"), xp, qWidth, H)
			qv := make([]float32, nH*hd)
			gv := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				copy(qv[h*hd:(h+1)*hd], qf[h*2*hd:h*2*hd+hd])
				copy(gv[h*hd:(h+1)*hd], qf[h*2*hd+hd:h*2*hd+2*hd])
			}
			q[t], gates[t] = qv, gv
		} else {
			q[t] = mat.mul(p("self_attn.q_proj.weight"), xp, nH*hd, H)
		}
		k[t] = mat.mul(p("self_attn.k_proj.weight"), xp, nKV*hd, H)
		v[t] = mat.mul(p("self_attn.v_proj.weight"), xp, nKV*hd, H)
		m.applyProjBias(l, q[t], k[t], v[t])
		// qk-norm AFTER projection, BEFORE RoPE; no-op for Llama.
		m.applyLayerQKNorm(l, q[t], k[t])
		// RoPE per head on q and k, through the shared single-row builder.
		if !cfg.Alibi {
			ropeRowQKInto(q[t], k[t], rp.cos[t], rp.sin[t], hd, nH, nKV)
		}
	}

	// scaled-dot-product attention, causal, GQA. With sliding-window attention (W>=0)
	// query t attends only to keys in [lo, t], lo=max(0,t-W+1); W=-1 (the default) keeps
	// lo=0, i.e. the full causal range 0..t exactly. This is the cacheless full-prefill
	// path, so positions are 0..seq-1 (no eviction) and the index IS the absolute position.
	W := cfg.windowForLayer(l)
	scale := cfg.attnScale()
	attnOut := make([][]float32, seq) // [seq][nH*hd]
	for t := 0; t < seq; t++ {
		attnOut[t] = make([]float32, nH*hd)
		lo := 0
		if W >= 0 {
			if lo = t - W + 1; lo < 0 {
				lo = 0
			}
		}
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[t][h*hd : (h+1)*hd]
			// scores over keys lo..t (causal, optionally windowed)
			scores := make([]float32, t+1-lo)
			for j := lo; j <= t; j++ {
				kh := k[j][kvh*hd : (kvh+1)*hd]
				scores[j-lo] = dot(qh, kh)*scale + cfg.alibiScoreBias(h, j, seq)
			}
			softcapInPlace(scores, attnCap)
			m.softmaxAttentionScores(l, h, scores)
			if m.attnObs != nil { // #852: emit the post-softmax row (copy-out, math untouched)
				emitAttnRow(m.attnObs, l, t, h, lo, scores)
			}
			// weighted sum of values
			o := attnOut[t][h*hd : (h+1)*hd]
			for j := lo; j <= t; j++ {
				vh := v[j][kvh*hd : (kvh+1)*hd]
				w := scores[j-lo]
				for d := 0; d < hd; d++ {
					o[d] += w * vh[d]
				}
			}
		}
		if gated {
			gt := gates[t]
			for i := 0; i < nH*hd; i++ {
				attnOut[t][i] *= sigmoidf(gt[i])
			}
		}
		attnOut[t] = mat.mul(p("self_attn.o_proj.weight"), mat.prep(attnOut[t]), H, nH*hd)
		m.addBiasIfPresent(attnOut[t], p("self_attn.o_proj.bias"))
	}
	return attnOut
}

func (m *Model) glmDsaAttnSeqShared(l int, xn [][]float32, sharedTopK *[][]int) [][]float32 {
	cfg := m.Cfg
	seq := len(xn)
	xnFlat := flatten(xn)
	var topK [][]int
	if glmDsaIndexerIsShared(cfg, l) {
		if sharedTopK == nil || *sharedTopK == nil {
			panic("model: glm_moe_dsa shared indexer without previous full indexer")
		}
		topK = cloneIndexDecision(*sharedTopK)
	} else {
		if !glmDsaIndexerIsFull(cfg, l) {
			panic("model: glm_moe_dsa unknown indexer type")
		}
		var ok bool
		topK, ok = glmDsaTopKIndicesNormed(m, l, xnFlat, seq)
		if !ok {
			panic("model: glm_moe_dsa top-k failed")
		}
		if sharedTopK != nil {
			*sharedTopK = cloneIndexDecision(topK)
		}
	}
	out, ok := glmDsaAttentionOutputFromTopKNormed(m, l, xnFlat, seq, topK)
	if !ok {
		panic("model: glm_moe_dsa attention failed")
	}
	return splitFlatRows(out, seq, cfg.HiddenSize)
}

// mlpSeq computes the SwiGLU MLP over a whole sequence of normalized inputs and
// returns the per-position down-projection results (pre residual).
func (m *Model) mlpSeq(l int, xn [][]float32) [][]float32 {
	out := make([][]float32, len(xn))
	ffn := m.ffnForLayer(l)
	mat := residentKernel{m}
	for t := range xn {
		out[t] = ffn.apply(m, l, mat.prep(xn[t]), mat)
	}
	return out
}

// normSeq normalizes each position's vector with the supplied norm weights.
func normSeq(x [][]float32, n normWeights, eps float32, cfg Config) [][]float32 {
	out := make([][]float32, len(x))
	for t := range x {
		out[t] = normCfg(x[t], n.pre, n.preBias, eps, cfg)
	}
	return out
}

// seqSublayer is one whole-sequence residual sub-layer body: normalized per-position
// inputs in, raw per-position outputs out (pre residual/post-norm).
type seqSublayer func(xn [][]float32) [][]float32

// composeSeqSublayer applies ONE residual sub-layer (norm placement + body + add)
// across a whole sequence under topology t. PreNorm: x += body(norm(x)) — the
// verbatim Llama placement. PostNorm: x += norm(body(x)). SandwichNorm:
// x += post(body(pre(x))). Parallel is handled separately by layer (shared norm,
// two deltas into one residual).
func composeSeqSublayer(t BlockTopology, x [][]float32, n normWeights, eps float32, cfg Config, body seqSublayer) {
	H := len(x[0])
	switch t {
	case PostNorm:
		out := body(x) // sub-layer reads the RAW residual stream
		for tt := range x {
			nout := normCfg(out[tt], n.post, n.postBias, eps, cfg)
			for i := 0; i < H; i++ {
				x[tt][i] += nout[i]
			}
		}
	case SandwichNorm:
		out := body(normSeq(x, n, eps, cfg))
		for tt := range x {
			nout := normCfg(out[tt], n.post, n.postBias, eps, cfg)
			for i := 0; i < H; i++ {
				x[tt][i] += nout[i]
			}
		}
	default: // PreNorm — verbatim Llama
		out := body(normSeq(x, n, eps, cfg))
		for tt := range x {
			for i := 0; i < H; i++ {
				x[tt][i] += out[tt][i]
			}
		}
	}
}

// ---- primitive ops ---------------------------------------------------------

// rmsnorm: x / sqrt(mean(x^2)+eps) * weight (Llama convention: plain weight). The scalar
// in-order sum-of-squares is load-bearing for the f32 bit-exact rungs (R2/R14) — do not
// reorder it here; the in-place quant twin rmsnormInto is the one that may use fdot.
func rmsnorm(x, w []float32, eps float32) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
	return out
}

// rmsnormInto is the allocation-free RMSNorm used by the Q8 prefill path: it writes directly
// into dst (the caller's panel row), eliminating both the per-row heap slice rmsnorm returns
// AND the caller's copy — 2*P*NumLayers (=15360 at P=256) of each per prefill. The
// sum-of-squares uses fdot (8 accumulators, vectorized) instead of the serial reduction; the
// ~1e-6 reduction-order shift is inside the Q8 gate's tolerance (logit-cosine vs f32,
// argmax-exact vs the oracle) and never reaches the f32 bit-exact rungs, which keep rmsnorm.
func rmsnormInto(dst, x, w []float32, eps float32) {
	ss := fdot(x, x)
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i, v := range x {
		dst[i] = v * inv * w[i]
	}
}

// matRows: y[o] = sum_i W[o*in+i]*x[i], W row-major [out,in]. (HF Linear: y = x @ W^T.)
// Routes through fdot (the 8-accumulator inner product) so the serial reference, the
// row-parallel parMatRows, and the batched matMulBatch all share ONE reduction and stay
// mutually bit-identical — the invariant the exact rungs R2/R14 rely on.
func matRows(w, x []float32, out, in int) []float32 {
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		y[o] = fdot(w[o*in:o*in+in], x)
	}
	return y
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func softmaxInPlace(s []float32) {
	mx := s[0]
	for _, v := range s {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i, v := range s {
		e := float32(math.Exp(float64(v - mx)))
		s[i] = e
		sum += e
	}
	for i := range s {
		s[i] /= sum
	}
}

func silu(z float32) float32 { return z / (1 + float32(math.Exp(float64(-z)))) }

func fastExp32(x float32) float32 {
	if x <= -87 {
		return 0
	}
	if x >= 88 {
		return float32(math.Inf(1))
	}
	// Schraudolph-style exp approximation. This is only used by the Q8 decode path, never by
	// the exact f32 serial-equivalence path.
	return math.Float32frombits(uint32(12102203*x + 1064866805))
}

func fastSilu(z float32) float32 { return z / (1 + fastExp32(-z)) }

// swigluInPlaceAct applies g[i] = act(g[i]) * u[i] in place, parallelized over the
// same parFor scaffold for any element-wise gate activation act.
func swigluInPlaceAct(g, u []float32, act func(float32) float32) {
	body := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			g[i] = act(g[i]) * u[i]
		}
	}
	if len(g) < parThreshold || numWorkers <= 1 {
		body(0, len(g))
		return
	}
	parFor(len(g), numWorkers, body)
}

func swigluInPlace(g, u []float32) { swigluInPlaceAct(g, u, silu) }

func swigluFastInPlace(g, u []float32) { swigluInPlaceAct(g, u, fastSilu) }

func addBias(y, b []float32) {
	for i := range y {
		y[i] += b[i]
	}
}

func flatten(x [][]float32) []float32 {
	if len(x) == 0 {
		return nil
	}
	H := len(x[0])
	out := make([]float32, len(x)*H)
	for t := range x {
		copy(out[t*H:], x[t])
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
