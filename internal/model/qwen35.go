package model

import (
	"math"
	"strings"
)

// Qwen3.5 / Qwen3-Next hybrid support. "qwen35" here names the hybrid-Gated-DeltaNet
// arch FAMILY (HF model_type "qwen3_5_text"), NOT one model release: it is shared by both
// Qwen3.5-0.8B AND Qwen3.6-27B, because the Qwen3.6-27B checkpoint also declares
// model_type "qwen3_5_text" (config_test.go), so this one loader is the code path for both
// model versions — the "3.5" in the symbol is the arch lineage, not the release number.
// Three of every four decoder layers are a
// Gated-DeltaNet *linear-attention* token mixer — a recurrent state-space scan, not
// quadratic attention — and every fourth layer is gated full attention. This file adds
// the linear-attention mixer and the load-time tensor-name normalization. The
// full-attention output gate lives in attnSeq (forward.go) behind cfg.AttnOutputGate;
// config detection, the (1+weight) RMSNorm, the partial-rotary inv_freq denominator, and
// the per-layer token-mixer dispatch live in weights.go / kv.go / forward.go.
//
// The math is ported verbatim from transformers Qwen3_5GatedDeltaNet /
// torch_recurrent_gated_delta_rule (the witness we did not author): a per-token decay
// g=exp(-exp(A_log)*softplus(a+dt_bias)), a sigmoid selection gate beta=sigmoid(b), a
// short causal depthwise conv1d+SiLU over concat(q,k,v), q/k L2-normalization with a
// 1/sqrt(head_k_dim) query scale, the rank-1 delta-rule state update, and a per-head
// gated RMSNorm of the readout.

// IsQwen35Hybrid reports whether this checkpoint is a qwen35-family hybrid — i.e. any of
// Qwen3.5 / Qwen3.6 / Qwen3-Next, which all share HF model_type "qwen3_5_text" (any
// linear_attention layer present). It gates the (1+w) RMSNorm, the rotary-dim rope
// denominator, the tensor-name normalization, and the per-layer mixer dispatch.
func (c Config) IsQwen35Hybrid() bool {
	for _, t := range c.LayerTypes {
		if t == "linear_attention" {
			return true
		}
	}
	return false
}

// isLinearAttnLayer reports whether decoder layer l is a Gated-DeltaNet linear-attention
// layer (vs gated full attention), per config.layer_types.
func (c Config) isLinearAttnLayer(l int) bool {
	return l >= 0 && l < len(c.LayerTypes) && c.LayerTypes[l] == "linear_attention"
}

// materializeQwen35Tensors normalizes the qwen3_5 checkpoint tensor names into the
// canonical "model." scheme the loader/forward expect. The HF checkpoint nests the LM
// under "model.language_model.", and ships a vision tower ("model.visual.") plus a
// multi-token-prediction head ("mtp.") that text generation never reads; we rename the LM
// tensors and drop the rest so the standard layer-prefix lookups resolve.
func materializeQwen35Tensors(cfg Config, man map[string]tensorMeta) error {
	if !cfg.IsQwen35Hybrid() {
		return nil
	}
	const lm = "model.language_model."
	renames := make(map[string]string)
	var drop []string
	for name := range man {
		switch {
		case strings.HasPrefix(name, lm):
			renames[name] = "model." + name[len(lm):]
		case strings.HasPrefix(name, "model.visual."), strings.HasPrefix(name, "mtp."):
			drop = append(drop, name)
		}
	}
	for old, nw := range renames {
		man[nw] = man[old]
		delete(man, old)
	}
	for _, d := range drop {
		delete(man, d)
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		p := layerPrefix(l)
		renameTensorIfPresent(man, p+"linear_attn.in_proj_qkv.weight", p+"self_attn.qkv_proj.weight")
		renameTensorIfPresent(man, p+"linear_attn.in_proj_z.weight", p+"self_attn.q_gate_proj.weight")
	}
	return nil
}

func renameTensorIfPresent(man map[string]tensorMeta, dst, src string) {
	if _, exists := man[dst]; exists {
		delete(man, src)
		return
	}
	if meta, ok := man[src]; ok {
		man[dst] = meta
		delete(man, src)
	}
}

func sigmoidf(x float32) float32 { return 1.0 / (1.0 + float32(math.Exp(float64(-x)))) }

// softplus is log(1+exp(x)), numerically stable for large x.
func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

// l2normInto writes the L2-normalized x into dst: x * rsqrt(sum(x^2)+eps) (FLA's l2norm,
// eps=1e-6, sum not mean). dst and x must have equal length.
func l2normInto(dst, x []float32, eps float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss)+float64(eps)))
	for i, v := range x {
		dst[i] = v * inv
	}
}

// rmsNormGatedInPlace applies Qwen3_5RMSNormGated to x in place: x = weight *
// (x*rsqrt(mean(x^2)+eps)) * silu(gate). weight is the plain (ones-init) gated-norm
// weight — NOT the (1+w) form used by the ordinary RMSNorm layers.
func rmsNormGatedInPlace(x, weight, gate []float32, eps float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss)/float64(len(x))+float64(eps)))
	for i := range x {
		x[i] = weight[i] * (x[i] * inv) * silu(gate[i])
	}
}

type linearAttnCache struct {
	layers []linearAttnLayerState
}

type linearAttnLayerState struct {
	recurrent [][]float32 // [value_head][key_head_dim*value_head_dim]
	conv      [][]float32 // last K-1 mixed q/k/v rows, oldest first
}

func newLinearAttnCache(cfg Config) *linearAttnCache {
	if !cfg.IsQwen35Hybrid() {
		return nil
	}
	c := &linearAttnCache{layers: make([]linearAttnLayerState, cfg.NumLayers)}
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		c.layers[l] = newLinearAttnLayerState(cfg)
	}
	return c
}

func newLinearAttnLayerState(cfg Config) linearAttnLayerState {
	nV := cfg.LinearNumValueHeads
	kHd := cfg.LinearKeyHeadDim
	vHd := cfg.LinearValueHeadDim
	st := linearAttnLayerState{recurrent: make([][]float32, nV)}
	for h := range st.recurrent {
		st.recurrent[h] = make([]float32, kHd*vHd)
	}
	return st
}

// recurrentLayers returns the indices of the layers that actually hold linear-attention
// recurrent state, so the typed eviction verdict (RecurrentEvictUnsupportedError) can name
// exactly which layers block a span eviction rather than asserting "some recurrent layer".
func (c *linearAttnCache) recurrentLayers() []int {
	if c == nil {
		return nil
	}
	var out []int
	for l := range c.layers {
		if len(c.layers[l].recurrent) > 0 {
			out = append(out, l)
		}
	}
	return out
}

// itoaSlice renders []int as "[a b c]" for the typed eviction verdict's message, without
// pulling fmt into this hot package.
func itoaSlice(v []int) string {
	s := "["
	for i, n := range v {
		if i > 0 {
			s += " "
		}
		s += itoa(n)
	}
	return s + "]"
}

func (c *linearAttnCache) clone() *linearAttnCache {
	if c == nil {
		return nil
	}
	out := &linearAttnCache{layers: make([]linearAttnLayerState, len(c.layers))}
	for l := range c.layers {
		out.layers[l] = c.layers[l].clone()
	}
	return out
}

func (s linearAttnLayerState) clone() linearAttnLayerState {
	out := linearAttnLayerState{
		recurrent: make([][]float32, len(s.recurrent)),
		conv:      make([][]float32, len(s.conv)),
	}
	for h := range s.recurrent {
		out.recurrent[h] = append([]float32(nil), s.recurrent[h]...)
	}
	for i := range s.conv {
		out.conv[i] = append([]float32(nil), s.conv[i]...)
	}
	return out
}

func (c *linearAttnCache) layer(cfg Config, l int) *linearAttnLayerState {
	if c == nil || l < 0 || l >= len(c.layers) {
		panic("model: missing linear-attention recurrent cache")
	}
	if len(c.layers[l].recurrent) == 0 {
		c.layers[l] = newLinearAttnLayerState(cfg)
	}
	return &c.layers[l]
}

func (st *linearAttnLayerState) pushConvRow(row []float32, keep int) {
	if keep <= 0 {
		st.conv = st.conv[:0]
		return
	}
	if len(st.conv) < keep {
		cp := copyLinearConvRow(nil, row)
		st.conv = append(st.conv, cp)
		return
	}
	cp := st.conv[0]
	copy(st.conv, st.conv[1:])
	st.conv[keep-1] = copyLinearConvRow(cp, row)
}

func copyLinearConvRow(dst, row []float32) []float32 {
	if cap(dst) < len(row) {
		dst = make([]float32, len(row))
	} else {
		dst = dst[:len(row)]
	}
	copy(dst, row)
	return dst
}

// linearAttnSeq is the Gated-DeltaNet linear-attention token mixer over a whole sequence
// of already-(input_layernorm)-normalized inputs. It returns the per-position out_proj
// results (pre residual). The recurrent state is initialized to zero (cacheless prefill);
// the scan over positions IS the prefill, O(seq) per head, not quadratic attention.
func (m *Model) linearAttnSeq(l int, xn [][]float32) [][]float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	nK := cfg.LinearNumKeyHeads
	nV := cfg.LinearNumValueHeads
	kHd := cfg.LinearKeyHeadDim
	vHd := cfg.LinearValueHeadDim
	keyDim := nK * kHd
	valDim := nV * vHd
	convDim := 2*keyDim + valDim
	K := cfg.LinearConvKernelDim
	seq := len(xn)
	eps := float32(cfg.RMSNormEps)
	p := func(s string) string { return layerName(l, s) }
	mat := residentKernel{m}
	conv := m.tensor(p("linear_attn.conv1d.weight")) // [convDim*K] depthwise (no bias)
	aLog := m.tensor(p("linear_attn.A_log"))         // [nV]
	dtBias := m.tensor(p("linear_attn.dt_bias"))     // [nV]
	normW := m.tensor(p("linear_attn.norm.weight"))  // [vHd] gated RMSNorm weight

	// Per-position input projections + the per-head decay g and selection gate beta.
	mixed := make([][]float32, seq)  // pre-conv concat(q,k,v) [convDim]
	zAll := make([][]float32, seq)   // gate z [valDim]
	gDecay := make([][]float32, seq) // exp(g) per v-head [nV]
	beta := make([][]float32, seq)   // sigmoid(b) per v-head [nV]
	for t := range xn {
		xp := mat.prep(xn[t])
		mixed[t] = mat.mul(p("linear_attn.in_proj_qkv.weight"), xp, convDim, H)
		zAll[t] = mat.mul(p("linear_attn.in_proj_z.weight"), xp, valDim, H)
		bvec := mat.mul(p("linear_attn.in_proj_b.weight"), xp, nV, H)
		avec := mat.mul(p("linear_attn.in_proj_a.weight"), xp, nV, H)
		g := make([]float32, nV)
		bt := make([]float32, nV)
		for h := 0; h < nV; h++ {
			bt[h] = sigmoidf(bvec[h])
			a := float32(math.Exp(float64(aLog[h]))) // A = exp(A_log)
			dt := softplus(avec[h] + dtBias[h])
			g[h] = float32(math.Exp(float64(-a * dt))) // exp(g) state decay
		}
		gDecay[t] = g
		beta[t] = bt
	}

	// Causal depthwise conv1d (kernel K, no bias, left-padded) + SiLU over each channel.
	convOut := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		row := make([]float32, convDim)
		for c := 0; c < convDim; c++ {
			var acc float32
			cb := c * K
			for j := 0; j < K; j++ {
				if ti := t - (K - 1) + j; ti >= 0 {
					acc += conv[cb+j] * mixed[ti][c]
				}
			}
			row[c] = silu(acc)
		}
		convOut[t] = row
	}

	// Recurrent gated delta rule. State per v-head is [kHd*vHd] row-major st[i*vHd+d].
	scale := float32(1.0 / math.Sqrt(float64(kHd)))
	repeat := nV / nK // group-expansion factor (q,k repeated to v-head count)
	state := make([][]float32, nV)
	for h := range state {
		state[h] = make([]float32, kHd*vHd)
	}
	core := make([][]float32, seq) // [valDim] readout, pre gated-norm
	qNorm := make([]float32, keyDim)
	kNorm := make([]float32, keyDim)
	kvmem := make([]float32, vHd)
	delta := make([]float32, vHd)
	for t := 0; t < seq; t++ {
		q := convOut[t][0:keyDim]
		k := convOut[t][keyDim : 2*keyDim]
		// Qwen3.6 repeats each Q/K head across several value heads. Normalize each
		// distinct Q/K head once per token, then share those rows across the repeated
		// value heads below.
		for h := 0; h < nK; h++ {
			l2normInto(qNorm[h*kHd:(h+1)*kHd], q[h*kHd:(h+1)*kHd], 1e-6)
			l2normInto(kNorm[h*kHd:(h+1)*kHd], k[h*kHd:(h+1)*kHd], 1e-6)
			for i := h * kHd; i < (h+1)*kHd; i++ {
				qNorm[i] *= scale
			}
		}
		v := convOut[t][2*keyDim : 2*keyDim+valDim]
		out := make([]float32, valDim)
		for h := 0; h < nV; h++ {
			kh := h / repeat
			qn := qNorm[kh*kHd : (kh+1)*kHd]
			kn := kNorm[kh*kHd : (kh+1)*kHd]
			vh := v[h*vHd : (h+1)*vHd]
			g := gDecay[t][h]
			bt := beta[t][h]
			st := state[h]
			for i := range st { // decay
				st[i] *= g
			}
			for d := range kvmem { // kv_mem[d] = sum_i st[i,d]*k[i]
				kvmem[d] = 0
			}
			for i := 0; i < kHd; i++ {
				ki := kn[i]
				base := i * vHd
				for d := 0; d < vHd; d++ {
					kvmem[d] += st[base+d] * ki
				}
			}
			for d := 0; d < vHd; d++ { // delta = (v - kv_mem)*beta
				delta[d] = (vh[d] - kvmem[d]) * bt
			}
			od := out[h*vHd : (h+1)*vHd]
			for i := 0; i < kHd; i++ { // state += outer(k,delta); out = sum_i st_updated[i,:]*q[i]
				ki := kn[i]
				qi := qn[i]
				base := i * vHd
				for d := 0; d < vHd; d++ {
					st[base+d] += ki * delta[d]
					od[d] += st[base+d] * qi
				}
			}
		}
		core[t] = out
	}

	// Per-head gated RMSNorm of the readout, then output projection.
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		for h := 0; h < nV; h++ {
			rmsNormGatedInPlace(core[t][h*vHd:(h+1)*vHd], normW, zAll[t][h*vHd:(h+1)*vHd], eps)
		}
		out[t] = mat.mul(p("linear_attn.out_proj.weight"), mat.prep(core[t]), H, valDim)
	}
	return out
}

// linearAttnStep is the Session.Prefill/Step twin of linearAttnSeq: it consumes one
// already-normalized token, updates the persistent per-layer conv window and Gated-DeltaNet
// recurrent state, and returns this token's output projection.
func (s *Session) linearAttnStep(l int, xn []float32, mat matKernel) []float32 {
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	nK := cfg.LinearNumKeyHeads
	nV := cfg.LinearNumValueHeads
	kHd := cfg.LinearKeyHeadDim
	vHd := cfg.LinearValueHeadDim
	keyDim := nK * kHd
	valDim := nV * vHd
	convDim := 2*keyDim + valDim
	K := cfg.LinearConvKernelDim
	eps := float32(cfg.RMSNormEps)
	p := func(str string) string { return layerName(l, str) }
	if s.Cache.linear == nil {
		s.Cache.linear = newLinearAttnCache(cfg)
	}
	lst := s.Cache.linear.layer(cfg, l)

	t := s.phaseStart()
	xp := mat.prep(xn)
	// The four GDN in_proj weights all read the same normed input xp — group them so the
	// resident-Q4_K Metal kernel issues the q4_k-resident members (qkv, z) in a single command
	// buffer (the per-token decode lever, #67); the tiny b/a fall back per-call.
	in4 := mulGroup(mat, []string{
		p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
		p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"),
	}, xp, []int{convDim, valDim, nV, nV}, H)
	mixed, zAll, bvec, avec := in4[0], in4[1], in4[2], in4[3]
	s.phaseEnd("qwen35_linear_step_in_proj", t)
	conv := m.tensor(p("linear_attn.conv1d.weight")) // [convDim*K] depthwise
	aLog := m.tensor(p("linear_attn.A_log"))
	dtBias := m.tensor(p("linear_attn.dt_bias"))
	normW := m.tensor(p("linear_attn.norm.weight"))

	convOut := make([]float32, convDim)
	t = s.phaseStart()
	for c := 0; c < convDim; c++ {
		var acc float32
		cb := c * K
		for j := 0; j < K; j++ {
			var row []float32
			if j == K-1 {
				row = mixed
			} else {
				idx := len(lst.conv) + j - (K - 1)
				if idx < 0 {
					continue
				}
				row = lst.conv[idx]
			}
			acc += conv[cb+j] * row[c]
		}
		convOut[c] = silu(acc)
	}
	s.phaseEnd("qwen35_linear_step_conv", t)
	if tap := s.tapActive; tap != nil && tap.ops {
		tap.dumpOp(l, "convOut", convOut)
	}
	lst.pushConvRow(mixed, K-1)

	scale := float32(1.0 / math.Sqrt(float64(kHd)))
	repeat := nV / nK
	core := make([]float32, valDim)
	qNorm := make([]float32, keyDim)
	kNorm := make([]float32, keyDim)
	q := convOut[0:keyDim]
	k := convOut[keyDim : 2*keyDim]
	v := convOut[2*keyDim : 2*keyDim+valDim]
	t = s.phaseStart()
	for h := 0; h < nK; h++ {
		l2normInto(qNorm[h*kHd:(h+1)*kHd], q[h*kHd:(h+1)*kHd], 1e-6)
		l2normInto(kNorm[h*kHd:(h+1)*kHd], k[h*kHd:(h+1)*kHd], 1e-6)
		for i := h * kHd; i < (h+1)*kHd; i++ {
			qNorm[i] *= scale
		}
	}
	s.phaseEnd("qwen35_linear_step_qk_norm", t)
	if tap := s.tapActive; tap != nil && tap.ops {
		tap.dumpOp(l, "qk_norm", qNorm)
	}
	t = s.phaseStart()
	// Each value head's Gated-DeltaNet update is self-contained — it reads the shared (now
	// read-only) qNorm/kNorm and v, mutates only its OWN recurrent state lst.recurrent[h], and
	// writes the disjoint core[h*vHd:(h+1)*vHd]. The kvmem/delta scratch is the only shared
	// hazard, so each worker takes its own. The arithmetic per head is identical to the serial
	// loop, so the result is BIT-IDENTICAL regardless of worker count (pinned by the bit-exact
	// hybrid split-prefill/decode witness in qwen35_test.go) — this just spreads the nV
	// independent heads across cores, the per-token decode lever for the 75%-of-layers GDN scan.
	headStep := func(h int, kvmem, delta []float32) {
		kh := h / repeat
		qn := qNorm[kh*kHd : (kh+1)*kHd]
		kn := kNorm[kh*kHd : (kh+1)*kHd]
		vh := v[h*vHd : (h+1)*vHd]
		st := lst.recurrent[h]
		bt := sigmoidf(bvec[h])
		a := float32(math.Exp(float64(aLog[h])))
		dt := softplus(avec[h] + dtBias[h])
		g := float32(math.Exp(float64(-a * dt)))
		for i := range st {
			st[i] *= g
		}
		for d := range kvmem {
			kvmem[d] = 0
		}
		for i := 0; i < kHd; i++ {
			ki := kn[i]
			base := i * vHd
			for d := 0; d < vHd; d++ {
				kvmem[d] += st[base+d] * ki
			}
		}
		for d := 0; d < vHd; d++ {
			delta[d] = (vh[d] - kvmem[d]) * bt
		}
		od := core[h*vHd : (h+1)*vHd]
		for i := 0; i < kHd; i++ {
			ki := kn[i]
			qi := qn[i]
			base := i * vHd
			for d := 0; d < vHd; d++ {
				st[base+d] += ki * delta[d]
				od[d] += st[base+d] * qi
			}
		}
	}
	if numWorkers <= 1 || nV*kHd*vHd < parThreshold {
		kvmem := make([]float32, vHd)
		delta := make([]float32, vHd)
		for h := 0; h < nV; h++ {
			headStep(h, kvmem, delta)
		}
	} else {
		parFor(nV, numWorkers, func(lo, hi int) {
			kvmem := make([]float32, vHd) // per-worker scratch (the only cross-head hazard)
			delta := make([]float32, vHd)
			for h := lo; h < hi; h++ {
				headStep(h, kvmem, delta)
			}
		})
	}
	s.phaseEnd("qwen35_linear_step_recurrent", t)
	if tap := s.tapActive; tap != nil && tap.ops {
		tap.dumpOp(l, "recurrent", core)
	}
	t = s.phaseStart()
	for h := 0; h < nV; h++ {
		rmsNormGatedInPlace(core[h*vHd:(h+1)*vHd], normW, zAll[h*vHd:(h+1)*vHd], eps)
	}
	s.phaseEnd("qwen35_linear_step_gated_norm", t)
	if tap := s.tapActive; tap != nil && tap.ops {
		tap.dumpOp(l, "gated_norm", core)
	}
	t = s.phaseStart()
	out := mat.mul(p("linear_attn.out_proj.weight"), mat.prep(core), H, valDim)
	s.phaseEnd("qwen35_linear_step_out_proj", t)
	if tap := s.tapActive; tap != nil && tap.ops {
		tap.dumpOp(l, "out", out)
	}
	return out
}
