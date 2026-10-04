//go:build darwin && arm64 && cgo

package model

// metal_dense_decode_graph.go — the dense whole-token Q4_K Metal decode graph
// (fak#13599). For a dense PreNorm SwiGLU model (Qwen2.5 / Llama family) on the
// resident-Q4_K Metal session it encodes the WHOLE token into one ProjectionGraph:
//
//	embed → per layer: RMSNorm, q/k/v (+bias), RoPE + device-KV append + ungated
//	attention, o_proj, residual, RMSNorm, gate/up, SwiGLU, down, residual →
//	final RMSNorm → LM head → one FinishRead(logits, per-layer KRaw/KPost/V)
//
// i.e. exactly one command buffer and one completion wait per token, replacing the
// per-op GEMV command buffers (q, k, v, o and the fused MLP per layer, plus the head)
// that blockStep commits. Attention reads a session-owned device KV mirror
// (denseDecodeGraphState) so no host prefix is uploaded per token.
//
// The host KVCache stays authoritative: the token's KRaw/K/V rows are read back in
// the same terminal pack and appended exactly as blockStep appends them, so Evict,
// Truncate, Clone and prefix reuse keep working and simply invalidate the mirror.
// Every failure happens before the host cache is mutated, so the route is fully
// fail-open: the caller replays the token through blockStep. A dense model has no
// recurrent state that a failed graph could have advanced.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// denseDecodeMaxGraphFailures bounds consecutive post-Begin failures before the route
// stops trying for the session.
const denseDecodeMaxGraphFailures = 3

// tryDenseQ4KMetalDecodeGraph runs token id at pos through the dense decode graph and
// returns its (logit-scaled) logits. ok=false means the token was declined or failed
// before any host mutation and must run through the blockStep path.
func (s *Session) tryDenseQ4KMetalDecodeGraph(id, pos int) ([]float32, bool) {
	if s == nil || s.M == nil || !s.Q4K || !s.MetalQ4K {
		return nil, false
	}
	st := s.denseDecodeState()
	if reason := s.denseDecodeGraphAdmission(st, id, pos); reason != "" {
		st.decline(reason)
		return nil, false
	}
	logits, reason := s.runDenseQ4KMetalDecodeGraph(st, id)
	if reason != "" {
		st.decline(reason)
		return nil, false
	}
	return logits, true
}

// denseDecodeGraphAdmission is the pre-Begin admission. It returns "" to admit or a
// decline reason; it mutates nothing but the cached model-readiness verdict.
func (s *Session) denseDecodeGraphAdmission(st *denseDecodeGraphState, id, pos int) string {
	if !denseQ4KDecodeGraphEnabled() {
		return denseDeclineDisabled
	}
	if st.graphFailures >= denseDecodeMaxGraphFailures {
		return denseDeclineGraph
	}
	m, cfg := s.M, s.M.Cfg
	if s.Backend != nil || s.Q4 || s.GPTQ || s.F16 || s.PrecisionPolicy != nil || s.denseGPULayers() != 0 || s.Cache == nil {
		return denseDeclineSession
	}
	if pos != s.Cache.Len() {
		return denseDeclinePosition
	}
	if id < 0 || id >= cfg.VocabSize {
		return denseDeclineToken
	}
	if s.activeTap() != nil || s.tapActive != nil || s.captureTargetHidden || m.attnObs != nil || cfg.EnableResidualHook {
		return denseDeclineObserver
	}
	if !denseDecodeGraphArchitectureOK(cfg) || m.prism != nil || m.Q2KEmbedding != nil || s.residentQuantForwardUnsupported() != nil {
		return denseDeclineArchitecture
	}
	if !denseDecodeGraphGeometryOK(cfg) {
		return denseDeclineGeometry
	}
	if s.Cache.quantized() {
		return denseDeclineKVPrecision
	}
	if st.readyModel != m {
		if st.notReadyModel == m {
			return st.notReadyReason
		}
		if reason := denseDecodeGraphModelReady(m); reason != "" {
			st.notReadyModel, st.notReadyReason = m, reason
			return reason
		}
		st.readyModel = m
	}
	return ""
}

// denseDecodeGraphArchitectureOK admits exactly the dense PreNorm SwiGLU families whose
// blockStep full-attention branch the graph reproduces op-for-op: no gate, softcap,
// sliding window, ALiBi, qk-norm, MoE, recurrent/linear layers, MLA or sparse attention.
func denseDecodeGraphArchitectureOK(cfg Config) bool {
	if !q8FastPreNormOK(cfg) || !siluGatedMLP(cfg) || cfg.IsHybrid() || cfg.IsQwen35Hybrid() || cfg.IsMoE() ||
		cfg.DenseMLP || cfg.QKNorm || cfg.Alibi || cfg.HasQSASparseAttn() || cfg.usesMLAMoELayout() ||
		cfg.isMiniMaxSparseAttn() || cfg.hasKVCompressionSchedule() || cfg.HasPLEEngram() {
		return false
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			return false
		}
	}
	return true
}

// denseDecodeGraphGeometryOK mirrors the native attention entry's limits (one 256-lane
// threadgroup per head, even rotate-half RoPE) and the host RoPE table width.
func denseDecodeGraphGeometryOK(cfg Config) bool {
	hd, nH, nKV := cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	if cfg.NumLayers <= 0 || cfg.HiddenSize <= 0 || cfg.IntermediateSize <= 0 || cfg.VocabSize <= 0 ||
		hd < 2 || hd > 256 || hd%2 != 0 || nH <= 0 || nKV <= 0 || nH%nKV != 0 {
		return false
	}
	rotary := cfg.rotaryDim()
	return rotary >= 2 && rotary <= hd && rotary%2 == 0 && len(cachedInvFreq(cfg, 0)) == rotary/2
}

// denseDecodeGraphModelReady verifies, once per model, that every tensor the graph
// reads exists in the supported form: plain RMSNorm weights (no LayerNorm bias, no
// Gemma sandwich norms), no attention sinks, well-shaped optional biases, and every
// projection resolvable to a resident Metal handle (Q4_K, Q6_K or Q8) before Begin.
func denseDecodeGraphModelReady(m *Model) string {
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	qw, kvw := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	if !m.has("model.norm.weight") || m.has("model.norm.bias") || len(m.tensor("model.norm.weight")) != H {
		return denseDeclineTensors
	}
	biasOK := func(name string, width int, required bool) bool {
		if !m.has(name) {
			return !required
		}
		return len(m.tensor(name)) == width
	}
	for l := 0; l < cfg.NumLayers; l++ {
		p := func(suffix string) string { return layerName(l, suffix) }
		for _, name := range []string{p("input_layernorm.weight"), p("post_attention_layernorm.weight")} {
			if !m.has(name) || len(m.tensor(name)) != H {
				return denseDeclineTensors
			}
		}
		for _, name := range []string{p("input_layernorm.bias"), p("post_attention_layernorm.bias"),
			p("pre_feedforward_layernorm.weight"), p("post_feedforward_layernorm.weight"), p("self_attn.sinks")} {
			if m.has(name) {
				return denseDeclineTensors
			}
		}
		if !biasOK(p("self_attn.q_proj.bias"), qw, cfg.AttentionBias) || !biasOK(p("self_attn.k_proj.bias"), kvw, cfg.AttentionBias) ||
			!biasOK(p("self_attn.v_proj.bias"), kvw, cfg.AttentionBias) || !biasOK(p("self_attn.o_proj.bias"), H, false) ||
			!biasOK(p("mlp.gate_proj.bias"), I, false) || !biasOK(p("mlp.up_proj.bias"), I, false) || !biasOK(p("mlp.down_proj.bias"), H, false) {
			return denseDeclineTensors
		}
		for _, name := range denseDecodeGraphProjectionNames(l) {
			if !qwen35MetalMTPProjectionReady(m, name) {
				return denseDeclineProjection
			}
		}
	}
	if _, ok := resolveDenseQ4KMetalHead(m); !ok {
		return denseDeclineHead
	}
	return ""
}

func denseDecodeGraphProjectionNames(l int) []string {
	p := func(suffix string) string { return layerName(l, suffix) }
	return []string{
		p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight"), p("self_attn.o_proj.weight"),
		p("mlp.gate_proj.weight"), p("mlp.up_proj.weight"), p("mlp.down_proj.weight"),
	}
}

// resolveDenseQ4KMetalHead resolves the LM head in headResident's order (q4k, k-quant,
// then q2/q4/q8/GPTQ/f32) and returns a device handle only for the formats the graph
// can encode with the same weights the host head would read: Q4_K, Q6_K, or Q8. A head
// headResident would serve from another store (Q2_K, int4, GPTQ, tied f32) declines.
func resolveDenseQ4KMetalHead(m *Model) (qwen35MetalMTPHead, bool) {
	if qt := m.q4khead; qt != nil || m.q4kw[m.q4kHeadName()] != nil {
		name := m.q4kHeadName()
		if qt == nil {
			qt = m.q4kw[name]
		}
		w := m.metalQ4KWeight(name, qt)
		return qwen35MetalMTPHead{q4: w}, w != nil
	}
	if name := m.kqHeadName(); name != "" {
		qt := m.kqw[name]
		if qt == nil || qt.kind != kindQ6K {
			return qwen35MetalMTPHead{}, false
		}
		w := m.metalQ6KWeight(name, qt)
		return qwen35MetalMTPHead{q6: w}, w != nil
	}
	if m.q2w[m.residentHeadName()] != nil || m.q4head != nil {
		return qwen35MetalMTPHead{}, false
	}
	name := m.headName()
	qt := m.q8head
	if qt == nil {
		qt = m.q8w[name]
	}
	if qt == nil {
		return qwen35MetalMTPHead{}, false
	}
	w := m.metalQ8Weight(name, qt)
	return qwen35MetalMTPHead{q8: w}, w != nil
}

// denseDecodeKVLimitBytes is denseDecodeKVLimit over the live device: its working-set
// budget, the model's resident weights and every OTHER live device KV mirror (the
// caller frees its own before asking).
func denseDecodeKVLimitBytes(m *Model) int64 {
	total, ok := metalgemm.DeviceMemoryTotal()
	if !ok {
		return 0
	}
	return denseDecodeKVLimit(denseDecodeKVBudgetBytes.Load(), int64(total), m.ResidentReport().TotalResidentBytes, metalgemm.DeviceKVResidentBytes())
}

// prepare returns a device KV mirror holding exactly the host cache's Len() rows with
// room for `need` rows, reusing the current mirror when it is still valid (catching up
// rows appended outside the graph) and otherwise (re)allocating and reseeding it from
// the host cache. Seeding is one offset-addressed upload per layer and side.
func (st *denseDecodeGraphState) prepare(s *Session, need int) (*metalgemm.DeviceKV, uint64, string) {
	c, cfg := s.Cache, s.M.Cfg
	L, w := cfg.NumLayers, cfg.NumKVHeads*cfg.HeadDim
	n := c.Len()
	for l := 0; l < L; l++ {
		if len(c.K[l]) != n*w || len(c.Kraw[l]) != n*w || len(c.V[l]) != n*w {
			st.invalidate()
			return nil, 0, denseDeclineDeviceKV
		}
	}
	valid := st.kv != nil && st.cache == c && st.gen == c.mutationGeneration() && st.rows <= n
	from := st.rows
	if !valid || need > st.capacity {
		if st.kv == nil || need > st.capacity {
			// Free the old mirror first so the device room below counts only OTHER mirrors.
			st.invalidate()
			capTokens := denseDecodeKVCapacity(cfg, need, denseDecodeKVLimitBytes(s.M))
			if capTokens < need {
				return nil, 0, denseDeclineKVBudget
			}
			// Attend-only: the attention reads KPost/V; the host cache keeps KRaw (read
			// back per token), so a device KRaw side would never be read.
			kv := metalgemm.NewDeviceKVAttendOnly(L, capTokens, w)
			if kv == nil {
				return nil, 0, denseDeclineDeviceKV
			}
			st.kv, st.capacity = kv, capTokens
		}
		st.cache, st.gen, st.rows = c, c.mutationGeneration(), 0
		from = 0
		st.receipt.DeviceKVReseeds++
	}
	var seeded uint64
	if from < n {
		stride := st.kv.LayerStride()
		for l := 0; l < L; l++ {
			// Side 1 is KPost (the host's post-RoPE K), side 2 is V.
			for i, host := range [denseDecodeKVSides][]float32{c.K[l], c.V[l]} {
				if err := st.kv.UploadRegion(1+i, l*stride+from*w, host[from*w:n*w]); err != nil {
					st.invalidate()
					return nil, 0, denseDeclineDeviceKV
				}
			}
		}
		seeded = uint64(denseDecodeKVSides*L*(n-from)*w) * 4
		st.rows = n
	}
	return st.kv, seeded, ""
}

func (st *denseDecodeGraphState) onesFor(n int) []float32 {
	if len(st.ones) != n {
		st.ones = make([]float32, n)
		for i := range st.ones {
			st.ones[i] = 1
		}
	}
	return st.ones
}

// denseGraphBias adds a projection bias in place when present (or when required by
// AttentionBias, which admission already proved present), matching applyProjBias /
// addBiasIfPresent on the host.
func denseGraphBias(g *metalgemm.ProjectionGraph, m *Model, r *metalgemm.GraphResult, name string) error {
	if !m.has(name) {
		return nil
	}
	return g.AddBiasInPlace(r, m.tensor(name))
}

// runDenseQ4KMetalDecodeGraph encodes, commits and reads back one token. It returns the
// logits or a decline reason; on a decline the host cache is untouched.
func (s *Session) runDenseQ4KMetalDecodeGraph(st *denseDecodeGraphState, id int) ([]float32, string) {
	m, cfg := s.M, s.M.Cfg
	head, ok := resolveDenseQ4KMetalHead(m)
	if !ok {
		return nil, denseDeclineHead
	}
	H, hd, nH, nKV, L := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads, cfg.NumLayers
	base := s.Cache.Len()
	kv, seeded, reason := st.prepare(s, base+1)
	if reason != "" {
		return nil, reason
	}
	t := s.phaseStart()
	X := make([]float32, H)
	m.embedRowsInto(X, []int{id}, H, cfg)
	g, err := metalgemm.BeginProjectionGraph(X, nil, nil, 1, H)
	if err != nil {
		// Nothing was encoded: the mirror is still coherent, so keep it.
		return nil, denseDeclineGraph
	}
	defer g.Free()
	// Lane B owns the P=1 kernel choice; the graph default under SetGEMVDecode is used.
	g.SetGEMVDecode()
	g.SetBufferPool(8)
	if st.injectPostSubmitFailure {
		st.injectPostSubmitFailure = false
		g.InjectPostSubmitFailureForTest()
	}
	committed := false
	fail := func(err error) ([]float32, string) {
		// The graph may have appended row `base` to the device slices; drop the mirror
		// so the next token reseeds from the (untouched) host cache.
		st.invalidate()
		if committed {
			// A graph that committed and then failed still cost a command buffer.
			s.countMetalGraphCommandBuffer(1)
		}
		st.graphFailures++
		st.receipt.GraphFallbacks++
		return nil, denseDeclineGraph
	}
	x, err := g.Input(H)
	if err != nil {
		return fail(err)
	}
	quantized := make(map[*metalgemm.GraphResult]*metalgemm.QuantizedGraphResult)
	ones := st.onesFor(hd)
	eps := float32(cfg.RMSNormEps)
	scale, rotary := cfg.attnScale(), cfg.rotaryDim()
	terminal := make([]*metalgemm.GraphResult, 1, 1+3*L)
	for l := 0; l < L; l++ {
		p := func(suffix string) string { return layerName(l, suffix) }
		xn, err := g.RMSNorm(x, m.tensor(p("input_layernorm.weight")), eps, false)
		if err != nil {
			return fail(err)
		}
		qkv, err := qwen35GraphProjections(g, s, []string{p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight")}, xn, quantized)
		if err != nil {
			return fail(err)
		}
		// Bias before RoPE: the device KRaw must equal the host's post-bias Kraw stash.
		for i, suffix := range []string{"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"} {
			if err := denseGraphBias(g, m, qkv[i], p(suffix)); err != nil {
				return fail(err)
			}
		}
		cos, sin := ropeRowForLayer(cfg, l, base)
		att, err := g.FullAttentionDevice(qkv[0], qkv[1], qkv[2], nil, kv, l, ones, ones, cos, sin,
			base, nH, nKV, hd, rotary, scale, 1e-6, false, false)
		if err != nil {
			return fail(err)
		}
		o, err := qwen35GraphProjection(g, s, p("self_attn.o_proj.weight"), att.Output, quantized)
		if err != nil {
			return fail(err)
		}
		if err := denseGraphBias(g, m, o, p("self_attn.o_proj.bias")); err != nil {
			return fail(err)
		}
		// att.V aliases qkv[2]; keep it pinned through the terminal readback.
		g.Release(qkv[0])
		g.Release(qkv[1])
		g.Release(att.Output)
		g.Release(xn)
		terminal = append(terminal, att.KRaw, att.KPost, att.V)
		if err := g.AddInPlace(x, o); err != nil {
			return fail(err)
		}
		g.Release(o)
		xn2, err := g.RMSNorm(x, m.tensor(p("post_attention_layernorm.weight")), eps, false)
		if err != nil {
			return fail(err)
		}
		gu, err := qwen35GraphProjections(g, s, []string{p("mlp.gate_proj.weight"), p("mlp.up_proj.weight")}, xn2, quantized)
		if err != nil {
			return fail(err)
		}
		g.Release(xn2)
		if err := denseGraphBias(g, m, gu[0], p("mlp.gate_proj.bias")); err != nil {
			return fail(err)
		}
		if err := denseGraphBias(g, m, gu[1], p("mlp.up_proj.bias")); err != nil {
			return fail(err)
		}
		if err := g.SwiGLUInPlace(gu[0], gu[1]); err != nil {
			return fail(err)
		}
		g.Release(gu[1])
		down, err := qwen35GraphProjection(g, s, p("mlp.down_proj.weight"), gu[0], quantized)
		if err != nil {
			return fail(err)
		}
		g.Release(gu[0])
		if err := denseGraphBias(g, m, down, p("mlp.down_proj.bias")); err != nil {
			return fail(err)
		}
		if err := g.AddInPlace(x, down); err != nil {
			return fail(err)
		}
		g.Release(down)
	}
	hidden, err := g.LastRMSNorm(x, m.tensor("model.norm.weight"), eps, false)
	if err != nil {
		return fail(err)
	}
	logits, err := head.encode(g, hidden, quantized)
	if err != nil {
		return fail(err)
	}
	terminal[0] = logits
	outs, rc, err := g.FinishRead(terminal...)
	committed = rc.Committed
	if err == nil && (!rc.Committed || !rc.CompletedWait || rc.HostReadbacks != 1 || len(outs) != len(terminal) || len(outs[0]) != cfg.VocabSize) {
		err = fmt.Errorf("metalgemm: incomplete dense decode graph receipt: %+v", rc)
	}
	if err != nil {
		return fail(err)
	}
	// Host mutation starts here and cannot fail: append the token's KV rows exactly as
	// blockStep does (KRaw pre-RoPE post-bias, K post-RoPE, V post-bias), then its
	// position; the mirror already holds row `base` from the graph's device append.
	for l := 0; l < L; l++ {
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], outs[1+3*l]...)
		s.Cache.K[l] = append(s.Cache.K[l], outs[2+3*l]...)
		s.Cache.V[l] = append(s.Cache.V[l], outs[3+3*l]...)
	}
	s.Cache.appendPosition(base, id)
	st.rows = base + 1
	st.graphFailures = 0
	s.countMetalGraphCommandBuffer(1)
	y, _ := s.headLogitsBuf()
	copy(y, outs[0])
	logitScaleInPlace(y, cfg)
	s.phaseEnd("dense_decode_graph", t)
	r := &st.receipt
	r.Accepted, r.DeclineReason = true, ""
	r.CommandBuffers, r.Encoders = 1, rc.Encoders
	r.GPUMilliseconds, r.WaitMilliseconds = rc.GPUMilliseconds, rc.WaitMilliseconds
	r.HostUploadBytes, r.HostReadbackBytes = rc.HostUploadBytes, rc.HostReadbackBytes
	r.DeviceKVSeedBytes, r.DeviceKVCapacity = seeded, st.capacity
	r.AcceptedTokens++
	return y, ""
}
