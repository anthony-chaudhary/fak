//go:build darwin && arm64 && cgo

package model

// metal_prefill_hybrid.go — the Metal GPU twin of the Qwen3.6 hybrid (Gated-DeltaNet) prefill.
// Built by default on Apple Silicon with cgo. It is deliberately thin: the entire prefill body — both
// RMSNorms, the conv1d+SiLU mixer, the q/k L2-norm, the per-head delta-rule recurrent scan, the
// gated RMSNorm readout, the full-attention RoPE/GQA/output-gate, and every residual — lives in
// the backend-agnostic core prefillQwen35HybridViaMM (metal_prefill_hybrid_core.go), proven
// host-independently against the CPU template by TestQwen35HybridViaMMMatchesCPUTemplate. This
// file supplies only the one substitution that core abstracts: a GPU f16 GEMM for the projection
// /MLP matmuls. Keeping the GDN recurrence on the CPU and moving just the projections to the
// device is the measured lever (the projections are the prefill wall; the GDN scan is ~0.5%;
// #65, #977), and lifting requirePreNorm("Metal prefill") for the hybrid is what lets it use the
// Metal prefill at all (#71).
//
// Weights: like metalWeights(), the GPU holds an f16 copy of each projection, dequantized once
// from the Q8_0 store and cached per *Model. The hybrid's projection set is per-layer: every
// layer carries the three MLP matmuls, while the per-layer mixer is EITHER the five linear_attn
// projections (linear_attention layers) OR the four self_attn projections (full_attention
// layers), dispatched by isLinearAttnLayer — the same split the core's mm calls walk.

import (
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var (
	metalHybridMu           sync.Mutex
	metalHybridWt           = map[*Model]map[string]*metalgemm.Weight{} // per-Model name -> GPU f16 weight
	metalHybridMPSAvailable = metalgemm.MPSAvailable
)

type metalQwen35GDNSequenceBackend struct {
	mu                             sync.Mutex
	states                         map[Qwen35GDNAuxState]*metalgemm.GDNState
	forwardReceipt                 Qwen35MetalForwardSequenceReceipt
	forwardRan                     bool
	injectForwardPostSubmitFailure bool
}

func (b *metalQwen35GDNSequenceBackend) Qwen35MetalForwardSequenceReceipt() (Qwen35MetalForwardSequenceReceipt, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	receipt := b.forwardReceipt
	if receipt.StateIdentity != nil {
		identity := cloneQwen35MetalStateIdentityReceipt(*receipt.StateIdentity)
		receipt.StateIdentity = &identity
	}
	return receipt, b.forwardRan
}

func (b *metalQwen35GDNSequenceBackend) setForwardReceipt(r Qwen35MetalForwardSequenceReceipt) {
	b.mu.Lock()
	if r.StateIdentity != nil {
		identity := cloneQwen35MetalStateIdentityReceipt(*r.StateIdentity)
		r.StateIdentity = &identity
	}
	b.forwardReceipt = r
	b.forwardRan = true
	b.mu.Unlock()
}

func (b *metalQwen35GDNSequenceBackend) bindQwen35MetalStateIdentity(identity Qwen35MetalStateIdentityReceipt) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.forwardRan || b.forwardReceipt.StateIdentity != nil {
		return
	}
	cloned := cloneQwen35MetalStateIdentityReceipt(identity)
	b.forwardReceipt.StateIdentity = &cloned
	// FinishRead accounts the graph's one terminal pack. The GDN finalizer's
	// existing snapshot/seed transfers are outside GraphReceipt, so fold their
	// exact bytes into the model-level total only after every operation succeeds.
	b.forwardReceipt.HostReadbackBytes += identity.GDNStateD2HBytes
	b.forwardReceipt.HostUploadBytes += identity.GDNStateH2DBytes
}

func qwen35GraphProjection(g *metalgemm.ProjectionGraph, s *Session, name string, input *metalgemm.GraphResult, quantized map[*metalgemm.GraphResult]*metalgemm.QuantizedGraphResult) (*metalgemm.GraphResult, error) {
	m := s.M
	if qt := m.q4kw[name]; qt != nil {
		w := m.metalQ4KWeight(name, qt)
		if w == nil {
			return nil, fmt.Errorf("metalgemm: Q4_K graph weight unavailable: %s", name)
		}
		return g.EncodeQ4KFrom(w, input)
	}
	if qt := m.kqw[name]; qt != nil {
		w := m.metalQ6KWeight(name, qt)
		if w == nil {
			return nil, fmt.Errorf("metalgemm: Q6_K graph weight unavailable: %s", name)
		}
		return g.EncodeQ6KFrom(w, input)
	}
	qt := m.q8w[name]
	if qt == nil {
		return nil, fmt.Errorf("metalgemm: Q8 graph weight missing: %s", name)
	}
	w := m.metalQ8Weight(name, qt)
	if w == nil {
		return nil, fmt.Errorf("metalgemm: Q8 graph weight unavailable: %s", name)
	}
	q := quantized[input]
	if q == nil {
		var err error
		q, err = g.QuantizeQ8(input)
		if err != nil {
			return nil, err
		}
		quantized[input] = q
	}
	return g.EncodeQ8From(w, q)
}

func qwen35GraphProjections(g *metalgemm.ProjectionGraph, s *Session, names []string, input *metalgemm.GraphResult, quantized map[*metalgemm.GraphResult]*metalgemm.QuantizedGraphResult) ([]*metalgemm.GraphResult, error) {
	out := make([]*metalgemm.GraphResult, len(names))
	for i, name := range names {
		var err error
		out[i], err = qwen35GraphProjection(g, s, name, input, quantized)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func qwen35MetalForwardGeometryError(cfg Config) error {
	if cfg.NumExperts != 0 || cfg.AttentionBias || cfg.LayerNorm || cfg.QKNormPerHeadWeight ||
		cfg.HiddenSize <= 0 || cfg.HiddenSize%32 != 0 || cfg.IntermediateSize <= 0 || cfg.IntermediateSize%32 != 0 ||
		cfg.HeadDim < 2 || cfg.HeadDim > 256 || cfg.HeadDim%2 != 0 || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 ||
		cfg.NumHeads%cfg.NumKVHeads != 0 {
		return fmt.Errorf("metalgemm: unsupported exact P32 Qwen hybrid geometry")
	}
	return nil
}

func (b *metalQwen35GDNSequenceBackend) Qwen35MetalForwardSequence(s *Session, ids []int) ([]float32, Qwen35MetalForwardSequenceReceipt, bool, error) {
	if s == nil || s.M == nil || s.Backend != nil || !s.Q4K || !s.MetalQ4K || len(ids) != 32 || s.qwen35HAL == nil || !s.qwen35HAL.sequenceAccepted {
		return nil, Qwen35MetalForwardSequenceReceipt{}, false, nil
	}
	m, cfg := s.M, s.M.Cfg
	if err := qwen35MetalForwardGeometryError(cfg); err != nil {
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
	}
	base, H, P := s.Cache.Len(), cfg.HiddenSize, len(ids)
	if base+P > 4096 {
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, fmt.Errorf("metalgemm: P32 graph attention context %d exceeds 4096", base+P)
	}
	// Resolve every resident handle before graph construction. Once Begin succeeds,
	// any failure remains accepted and cannot replay through the host forward.
	s.prefillQwen35HybridQ4KMetalUpload()
	embed := m.embedRows()
	X := make([]float32, P*H)
	for i, id := range ids {
		copy(X[i*H:(i+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[i*H:(i+1)*H], cfg)
	}
	g, err := metalgemm.BeginProjectionGraph(X, nil, nil, P, H)
	if err != nil {
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
	}
	defer g.Free()
	if b.injectForwardPostSubmitFailure {
		g.InjectPostSubmitFailureForTest()
	}
	x, err := g.Input(H)
	if err != nil {
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
	}
	quantized := make(map[*metalgemm.GraphResult]*metalgemm.QuantizedGraphResult)
	type kvResult struct {
		layer          int
		kraw, kpost, v *metalgemm.GraphResult
	}
	var kvResults []kvResult
	eps := float32(cfg.RMSNormEps)
	for l := 0; l < cfg.NumLayers; l++ {
		p := func(suffix string) string { return layerName(l, suffix) }
		xn, runErr := g.RMSNorm(x, m.tensor(p("input_layernorm.weight")), eps, cfg.NormGain1p)
		if runErr != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
		}
		var attnOut *metalgemm.GraphResult
		if cfg.isLinearAttnLayer(l) {
			in, runErr := qwen35GraphProjections(g, s, []string{
				p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
				p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"),
			}, xn, quantized)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			state := b.state(s.qwen35HAL.sequenceLayers[l])
			if state == nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, fmt.Errorf("metalgemm: missing GDN graph owner for layer %d", l)
			}
			core, runErr := g.GDN(state, in[0], in[1], in[2], in[3], metalgemm.GDNPanel{
				Conv1D: m.tensor(p("linear_attn.conv1d.weight")), ALog: m.tensor(p("linear_attn.A_log")),
				DTBias: m.tensor(p("linear_attn.dt_bias")), Norm: m.tensor(p("linear_attn.norm.weight")), RMSNormEpsilon: eps,
			})
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			attnOut, runErr = qwen35GraphProjection(g, s, p("linear_attn.out_proj.weight"), core, quantized)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
		} else {
			qkv, runErr := qwen35GraphProjections(g, s, []string{p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight")}, xn, quantized)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			q, gate, runErr := g.SplitGatedQ(qkv[0], cfg.NumHeads*cfg.HeadDim, cfg.HeadDim)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			qnorm, knorm := make([]float32, cfg.HeadDim), make([]float32, cfg.HeadDim)
			if cfg.QKNorm {
				qnorm = m.tensor(p("self_attn.q_norm.weight"))
				knorm = m.tensor(p("self_attn.k_norm.weight"))
				qwidth, kvwidth := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
				if len(qnorm) != cfg.HeadDim && len(qnorm) != qwidth || len(knorm) != cfg.HeadDim && len(knorm) != kvwidth {
					return nil, Qwen35MetalForwardSequenceReceipt{}, true, fmt.Errorf("metalgemm: P32 Qwen graph requires shared or projection-wide per-head Q/K norm weights")
				}
			}
			rotary := cfg.rotaryDim()
			cosv, sinv := make([]float32, 0, P*(rotary/2)), make([]float32, 0, P*(rotary/2))
			for pos := 0; pos < P; pos++ {
				c, si := ropeRowForLayer(cfg, l, base+pos)
				cosv, sinv = append(cosv, c...), append(sinv, si...)
			}
			attention, runErr := g.FullAttention(q, qkv[1], qkv[2], gate, qnorm, knorm, cosv, sinv,
				s.Cache.K[l], s.Cache.V[l], base, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, rotary, cfg.attnScale(), cfg.qkNormEps(), cfg.NormGain1p, cfg.QKNorm)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			attnOut, runErr = qwen35GraphProjection(g, s, p("self_attn.o_proj.weight"), attention.Output, quantized)
			if runErr != nil {
				return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
			}
			kvResults = append(kvResults, kvResult{layer: l, kraw: attention.KRaw, kpost: attention.KPost, v: attention.V})
		}
		if err = g.AddInPlace(x, attnOut); err != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
		}
		xn2, runErr := g.RMSNorm(x, m.tensor(p("post_attention_layernorm.weight")), eps, cfg.NormGain1p)
		if runErr != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
		}
		gu, runErr := qwen35GraphProjections(g, s, []string{p("mlp.gate_proj.weight"), p("mlp.up_proj.weight")}, xn2, quantized)
		if runErr != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
		}
		if err = g.SwiGLUInPlace(gu[0], gu[1]); err != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
		}
		down, runErr := qwen35GraphProjection(g, s, p("mlp.down_proj.weight"), gu[0], quantized)
		if runErr != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, runErr
		}
		if err = g.AddInPlace(x, down); err != nil {
			return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
		}
	}
	hiddenResult, err := g.LastRMSNorm(x, m.tensor("model.norm.weight"), eps, cfg.NormGain1p)
	if err != nil {
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, err
	}
	terminal := []*metalgemm.GraphResult{hiddenResult}
	for _, kv := range kvResults {
		terminal = append(terminal, kv.kraw, kv.kpost, kv.v)
	}
	outputs, graphReceipt, err := g.FinishRead(terminal...)
	receipt := Qwen35MetalForwardSequenceReceipt{
		Path: Qwen35MetalGDNSequenceForwardPath, Available: true,
		SelectorState: Qwen35MetalSequenceSelectorOn, EvidenceState: Qwen35MetalSequenceEvidenceExecuted, Tokens: P,
		CommandBuffers: 1, Encoders: graphReceipt.Encoders, TerminalWaits: 1, TerminalReadbacks: graphReceipt.HostReadbacks,
		IntermediateWaits: graphReceipt.IntermediateWaits, IntermediateReadbacks: graphReceipt.IntermediateReadbacks,
		HostUploadBytes: graphReceipt.HostUploadBytes, HostReadbackBytes: graphReceipt.HostReadbackBytes,
		Committed: graphReceipt.Committed, CompletedWait: graphReceipt.CompletedWait, TimingAvailable: graphReceipt.TimingAvailable,
		GPUMilliseconds: graphReceipt.GPUMilliseconds, WaitMilliseconds: graphReceipt.WaitMilliseconds,
	}
	b.setForwardReceipt(receipt)
	if err != nil || !graphReceipt.Committed || !graphReceipt.CompletedWait || graphReceipt.HostReadbacks != 1 {
		if err == nil {
			err = fmt.Errorf("metalgemm: incomplete Qwen graph receipt: %+v", graphReceipt)
		}
		return nil, receipt, true, err
	}
	hidden := outputs[0]
	outIndex := 1
	for _, kv := range kvResults {
		s.Cache.Kraw[kv.layer] = append(s.Cache.Kraw[kv.layer], outputs[outIndex]...)
		s.Cache.K[kv.layer] = append(s.Cache.K[kv.layer], outputs[outIndex+1]...)
		s.Cache.V[kv.layer] = append(s.Cache.V[kv.layer], outputs[outIndex+2]...)
		outIndex += 3
	}
	for i, id := range ids {
		s.Cache.appendPosition(base+i, id)
	}
	s.q4kHybridPrefillChunks++
	s.q4kHybridPrefillLastBase = base
	return hidden, receipt, true, nil
}

func init() {
	newQwen35MetalGDNSequenceBackend = func() Qwen35GDNPreprojectedSequenceBackend {
		return &metalQwen35GDNSequenceBackend{states: make(map[Qwen35GDNAuxState]*metalgemm.GDNState)}
	}
}

func (*metalQwen35GDNSequenceBackend) Qwen35GDNPreprojectedSequencePath() string {
	return Qwen35GDNPreprojectedSequencePath
}

func metalQwen35GDNGeometry(g Qwen35GDNSequenceGeometry) metalgemm.GDNGeometry {
	return metalgemm.GDNGeometry{
		NumKeyHeads: g.NumKeyHeads, NumValueHeads: g.NumValueHeads,
		KeyHeadDim: g.KeyHeadDim, ValueHeadDim: g.ValueHeadDim, ConvKernel: g.ConvKernel,
	}
}

func (b *metalQwen35GDNSequenceBackend) NewQwen35GDNAuxState(_ int, geometry Qwen35GDNSequenceGeometry) (Qwen35GDNAuxState, error) {
	state, err := metalgemm.NewGDNState(metalQwen35GDNGeometry(geometry))
	if err != nil {
		return Qwen35GDNAuxState{}, err
	}
	conv, recurrent := state.Handles()
	handles := Qwen35GDNAuxState{Convolution: Qwen35GDNAuxHandle(conv), Recurrent: Qwen35GDNAuxHandle(recurrent)}
	b.mu.Lock()
	b.states[handles] = state
	b.mu.Unlock()
	return handles, nil
}

func (b *metalQwen35GDNSequenceBackend) state(handles Qwen35GDNAuxState) *metalgemm.GDNState {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.states[handles]
}

func (b *metalQwen35GDNSequenceBackend) Qwen35GDNPreprojectedSequence(req Qwen35GDNPreprojectedSequenceRequest) (Qwen35GDNPreprojectedSequenceResult, error) {
	state := b.state(req.State)
	if state == nil {
		return Qwen35GDNPreprojectedSequenceResult{}, &metalgemm.GDNDeclinedError{Reason: "unknown auxiliary-state owner"}
	}
	g := metalQwen35GDNGeometry(req.Geometry)
	keyDim := g.NumKeyHeads * g.KeyHeadDim
	valueDim := g.NumValueHeads * g.ValueHeadDim
	convDim := 2*keyDim + valueDim
	core := make([]float32, 0, req.Tokens*valueDim)
	for start := 0; start < req.Tokens; start += metalgemm.GDNMaxPanelTokens {
		end := min(start+metalgemm.GDNMaxPanelTokens, req.Tokens)
		panel := metalgemm.GDNPanel{
			Tokens: end - start,
			Mixed:  req.Mixed[start*convDim : end*convDim], Z: req.Z[start*valueDim : end*valueDim],
			B: req.B[start*g.NumValueHeads : end*g.NumValueHeads], A: req.A[start*g.NumValueHeads : end*g.NumValueHeads],
			Conv1D: req.Conv1D, ALog: req.ALog, DTBias: req.DTBias, Norm: req.Norm,
			RMSNormEpsilon: req.RMSNormEpsilon,
		}
		panelCore, accounting, accepted, err := state.Run(panel)
		if err != nil {
			return Qwen35GDNPreprojectedSequenceResult{}, err
		}
		if !accepted || !accounting.Committed || !accounting.CompletedWait || accounting.Encoders != 1 ||
			accounting.StateH2DTransfers != 0 || accounting.StateD2HTransfers != 0 || accounting.HostRecurrenceSteps != 0 ||
			accounting.OwnedBuffers != 2 || accounting.PrivateStateBuffers != 2 {
			return Qwen35GDNPreprojectedSequenceResult{}, fmt.Errorf("metalgemm: incomplete resident GDN observation: %+v", accounting)
		}
		core = append(core, panelCore...)
	}
	return Qwen35GDNPreprojectedSequenceResult{Core: core, State: req.State}, nil
}

func (b *metalQwen35GDNSequenceBackend) SeedQwen35GDNAuxState(handles Qwen35GDNAuxState, conv, recurrent []float32) error {
	state := b.state(handles)
	if state == nil {
		return fmt.Errorf("metalgemm: unknown GDN auxiliary-state owner")
	}
	return state.Seed(conv, recurrent)
}

func (b *metalQwen35GDNSequenceBackend) SnapshotQwen35GDNAuxState(handles Qwen35GDNAuxState) ([]float32, []float32, error) {
	state := b.state(handles)
	if state == nil {
		return nil, nil, fmt.Errorf("metalgemm: unknown GDN auxiliary-state owner")
	}
	return state.Snapshot()
}

func (b *metalQwen35GDNSequenceBackend) FreeQwen35GDNAuxState(handles Qwen35GDNAuxState) error {
	b.mu.Lock()
	state := b.states[handles]
	delete(b.states, handles)
	b.mu.Unlock()
	if state != nil {
		state.Close()
	}
	return nil
}

// metalWeightsQwen35Hybrid returns this model's GPU projection table for the hybrid prefill,
// uploading it once. It mirrors metalWeights() (same dequantQ8 -> f16 Upload, big f32 buffer
// freed after each upload) but uploads the hybrid's per-layer projection set instead of the seven
// uniform standard-attention names.
func (m *Model) metalWeightsQwen35Hybrid() map[string]*metalgemm.Weight {
	metalHybridMu.Lock()
	defer metalHybridMu.Unlock()
	if w, ok := metalHybridWt[m]; ok {
		return w
	}
	cfg := m.Cfg
	w := make(map[string]*metalgemm.Weight, 8*cfg.NumLayers)
	upload := func(name string) {
		qt := m.q8(name)
		h := metalgemm.Upload(dequantQ8(qt), qt.out, qt.in)
		if h == nil {
			panic("model: metal hybrid weight upload failed for " + name)
		}
		w[name] = h
	}
	for l := 0; l < cfg.NumLayers; l++ {
		upload(layerName(l, "mlp.gate_proj.weight"))
		upload(layerName(l, "mlp.up_proj.weight"))
		upload(layerName(l, "mlp.down_proj.weight"))
		if cfg.isLinearAttnLayer(l) {
			upload(layerName(l, "linear_attn.in_proj_qkv.weight"))
			upload(layerName(l, "linear_attn.in_proj_z.weight"))
			upload(layerName(l, "linear_attn.in_proj_b.weight"))
			upload(layerName(l, "linear_attn.in_proj_a.weight"))
			upload(layerName(l, "linear_attn.out_proj.weight"))
		} else {
			upload(layerName(l, "self_attn.q_proj.weight"))
			upload(layerName(l, "self_attn.k_proj.weight"))
			upload(layerName(l, "self_attn.v_proj.weight"))
			upload(layerName(l, "self_attn.o_proj.weight"))
		}
	}
	metalHybridWt[m] = w
	return w
}

// prefillBatchedMetalQwen35Hybrid is the Metal hybrid prefill: it feeds the backend-agnostic core
// a GPU f16 GEMM (Y[P,out] = X[P,in] * W[name]^T) for each projection and lets the core run the
// recurrence/attention/norm body on the CPU. It fills the same f32 KV + linear-attn caches the
// CPU hybrid paths build (so decode stays valid) and returns the last token's post-final-norm
// hidden (caller applies the head). Reached only for a fresh prefill via metalQwen35HybridPrefillOK.
func (s *Session) prefillBatchedMetalQwen35Hybrid(ids []int) []float32 {
	// The hybrid f16 projections require MPS in addition to the shared Metal
	// device. Decline before uploading weights or mutating prompt state when that
	// capability is absent; the established Q8 implementation remains fak-native
	// and preserves the same hidden-state contract for the caller-owned head.
	if !metalHybridMPSAvailable() {
		return s.prefillQwen35HybridQHidden(ids)
	}
	m := s.M
	P := len(ids)
	gw := m.metalWeightsQwen35Hybrid()
	// mm runs Y[P,out] = X[P,in] * W[name]^T on the GPU into a fresh buffer; `in` is implicit in
	// the uploaded weight, so the core's hybridGemmFn signature drops it.
	mm := func(name string, X []float32, out int) []float32 {
		Y := make([]float32, P*out)
		gw[name].MatMul(X, P, Y)
		return Y
	}
	return s.prefillQwen35HybridViaMM(ids, mm)
}
