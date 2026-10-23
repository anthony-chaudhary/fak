package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// promptPanelBackend is the optional device seam the dense HAL prefill uses to ingest a prompt
// as [rows, width] panels: per layer one GEMM per projection over the rows instead of one GEMV
// per token. Every op must be the per-row form of the op tokenHALOutput issues for one token,
// so a panel row is bitwise the token loop's row whatever the row count or chunking (the BitExact
// default). Row t of a panel sits at absolute position start+t. A backend that implements it
// must also run RMSNorm, AddBias, SwiGLU and AddInPlace row-wise over [rows, width] panels.
type promptPanelBackend interface {
	// PromptPanelRowExact reports whether PromptPanelMatMul over a weight of dt computes each
	// row bitwise as MatMul computes that row alone. A prompt with any other weight dtype keeps
	// the token loop.
	PromptPanelRowExact(dt compute.Dtype) bool
	// PromptPanelBegin opens a panel scratch scope; PromptPanelEnd frees every panel-sized
	// buffer the scope pooled, so no panel size survives into the decode allocator.
	// PromptPanelChunk starts each chunk of rows rows inside the scope; a backend frees the
	// previous chunk size's cached buffers when the row count changes.
	PromptPanelBegin()
	PromptPanelChunk(rows int)
	PromptPanelEnd()
	// PromptPanelMatMul is MatMul for each of the tokens rows of x.
	PromptPanelMatMul(w, x compute.Tensor, tokens int) compute.Tensor
	// PromptPanelRoPE returns a new tensor with row t rotated at position start+t.
	PromptPanelRoPE(x compute.Tensor, tokens, start, nHeads, headDim int, theta float64) compute.Tensor
	// PromptPanelAppendKV appends tokens rows of pre-RoPE keys, rotated keys and values to
	// layer, at positions start..start+tokens-1.
	PromptPanelAppendKV(kv compute.KVStore, layer int, kRaw, kRoPE, v compute.Tensor, start, tokens int)
	// PromptPanelAttention attends query row t to cache positions [0, start+t] of layer and
	// returns the [tokens, grp*nKV*headDim] context.
	PromptPanelAttention(q compute.Tensor, kv compute.KVStore, layer, tokens, start, grp int, scale float32) compute.Tensor
	// PromptPanelRow copies row of a [rows, width] panel into a new [width] tensor that
	// outlives PromptPanelEnd.
	PromptPanelRow(x compute.Tensor, row, width int) compute.Tensor
}

// promptPanelMaxTokens caps one panel chunk. A longer prompt runs as consecutive chunks that
// advance the KV start; the per-row ops make the split invisible in the bits.
const promptPanelMaxTokens = 512

// promptPanelChunkOverride, when positive, replaces the derived chunk size. Tests use it to
// prove chunking does not change bits.
var promptPanelChunkOverride int

var promptPanelProjections = [...]string{
	"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
	"self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
}

// promptPanelFor returns the backend's panel seam when this prompt can take it. The panel
// mirrors only tokenHALOutput's plain dense branch and only weight dtypes with a per-row exact
// GEMM, so every arch, capture, placement and dtype that does not qualify keeps the token loop.
func (s *Session) promptPanelFor(ids []int) (promptPanelBackend, bool) {
	if len(ids) < 2 || s.Backend == nil || s.halKV == nil {
		return nil, false
	}
	panel, ok := s.Backend.(promptPanelBackend)
	if !ok {
		return nil, false
	}
	cfg := s.M.Cfg
	if q8PrefillNeedsTokenLoop(cfg) || cfg.IsDeepSeekV4() || cfg.QKNorm || cfg.usesMLAMoELayout() ||
		cfg.isMiniMaxSparseAttn() || cfg.isGemma4() {
		return nil, false
	}
	if s.captureTargetHidden || s.qwen35TargetFeatureTxn != nil || s.qwen35TargetLayerCapture != nil {
		return nil, false
	}
	if s.M.Q2KEmbedding != nil {
		return nil, false
	}
	if _, split := s.validateDenseGPULayers(); split {
		return nil, false
	}
	for l := 0; l < cfg.NumLayers; l++ {
		for _, name := range promptPanelProjections {
			if !panel.PromptPanelRowExact(s.matWeightHAL(layerName(l, name)).Dtype) {
				return nil, false
			}
		}
	}
	return panel, true
}

// promptPanelChunkTokens sizes a chunk so its peak scratch fits the per-token transient demand
// compute.EstimateHALTransientMemoryPlan already prices, capped at promptPanelMaxTokens.
func (s *Session) promptPanelChunkTokens() int {
	if promptPanelChunkOverride > 0 {
		return promptPanelChunkOverride
	}
	c := s.M.Cfg
	return compute.PromptPanelChunkTokens(compute.TransformerScratchConfig{
		HiddenSize: c.HiddenSize, IntermediateSize: c.IntermediateSize, VocabSize: c.VocabSize,
		NumLayers: c.NumLayers, NumHeads: c.NumHeads, NumKVHeads: c.NumKVHeads, HeadDim: c.HeadDim,
		IncludeLogits: true,
	}, promptPanelMaxTokens)
}

// prefillHALPanel is the batched twin of the prefillHAL token loop for a dense PreNorm model.
// It ingests the prompt in bounded chunks; per chunk and layer it issues one RMSNorm, one
// per-row GEMM per projection, one RoPE, one KV append and one causal attention.
func (s *Session) prefillHALPanel(panel promptPanelBackend, ids []int, wantLogits bool) []float32 {
	finishLineage := s.beginHALTokenLineageWrite(ids)
	defer finishLineage()
	be := s.Backend
	H := s.M.Cfg.HiddenSize
	if be.Caps().DeviceMemory {
		compute.CountHostFallback(hostFallbackEmbedRowUpload)
	}
	s.admitHostStage(HostStageEmbeddingRowUpload)

	panel.PromptPanelBegin()
	open := true
	defer func() {
		if open {
			panel.PromptPanelEnd()
		}
	}()
	chunk := s.promptPanelChunkTokens()
	var last compute.Tensor
	for off := 0; off < len(ids); off += chunk {
		end := min(off+chunk, len(ids))
		panel.PromptPanelChunk(end - off)
		x := s.promptPanelChunk(panel, ids[off:end])
		if end == len(ids) && wantLogits {
			last = panel.PromptPanelRow(x, end-off-1, H)
		}
		be.Free(x)
	}
	panel.PromptPanelEnd()
	open = false
	s.halStep += len(ids)

	if !wantLogits {
		s.recycleHALToken()
		return nil
	}
	// halLogitsWarm stays as it was: the panel never ran the single-token transient sizes a
	// captured decode graph replays, so the first decode step must still run uncaptured.
	logits, _ := s.halFinalLogits(last, halFullLogits, false, s.useHALQ8Weights(), func() { be.Free(last) })
	s.admitHostStage(HostStageLogitsReadback)
	out := be.Read(logits)
	if out == nil {
		panic(fmt.Sprintf("model: compute backend %s returned unreadable logits", be.Name()))
	}
	s.recycleHALToken()
	return out
}

// promptPanelChunk runs every layer over one chunk of ids at the cache's current length and
// returns the chunk's residual [len(ids), hidden]; the caller frees it.
func (s *Session) promptPanelChunk(panel promptPanelBackend, ids []int) compute.Tensor {
	be := s.Backend
	m, cfg := s.M, s.M.Cfg
	P := len(ids)
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	scale := cfg.attnScale()
	start := s.halKV.Len()

	embed := m.embedRows()
	rows := make([]float32, P*H)
	for t, id := range ids {
		if id < 0 || (id+1)*H > len(embed) {
			panic(fmt.Sprintf("model: prompt token id %d outside the %d-row embedding table", id, len(embed)/H))
		}
		copy(rows[t*H:(t+1)*H], embed[id*H:(id+1)*H])
	}
	x := s.uploadHostF32([]int{P, H}, rows, compute.MemoryActivation, "hal-prompt-panel-input")
	done := false
	defer func() {
		if !done {
			be.Free(x) // a failed chunk must not strand its residual upload
		}
	}()

	for l := 0; l < cfg.NumLayers; l++ {
		p := func(str string) string { return layerName(l, str) }
		xn := be.RMSNorm(x, s.normWeightHAL(p("input_layernorm.weight")), eps)
		q := panel.PromptPanelMatMul(s.matWeightHAL(p("self_attn.q_proj.weight")), xn, P)
		kRaw := panel.PromptPanelMatMul(s.matWeightHAL(p("self_attn.k_proj.weight")), xn, P)
		v := panel.PromptPanelMatMul(s.matWeightHAL(p("self_attn.v_proj.weight")), xn, P)
		if cfg.AttentionBias {
			be.AddBias(q, s.weightHAL(p("self_attn.q_proj.bias")))
			be.AddBias(kRaw, s.weightHAL(p("self_attn.k_proj.bias")))
			be.AddBias(v, s.weightHAL(p("self_attn.v_proj.bias")))
		}
		qr := panel.PromptPanelRoPE(q, P, start, nH, hd, cfg.RopeTheta)
		kr := panel.PromptPanelRoPE(kRaw, P, start, nKV, hd, cfg.RopeTheta)
		panel.PromptPanelAppendKV(s.halKV, l, kRaw, kr, v, start, P)
		attn := panel.PromptPanelAttention(qr, s.halKV, l, P, start, grp, scale)
		o := panel.PromptPanelMatMul(s.matWeightHAL(p("self_attn.o_proj.weight")), attn, P)
		be.AddInPlace(x, o)

		xn2 := be.RMSNorm(x, s.normWeightHAL(p("post_attention_layernorm.weight")), eps)
		g := panel.PromptPanelMatMul(s.matWeightHAL(p("mlp.gate_proj.weight")), xn2, P)
		u := panel.PromptPanelMatMul(s.matWeightHAL(p("mlp.up_proj.weight")), xn2, P)
		ff := be.SwiGLU(g, u)
		down := panel.PromptPanelMatMul(s.matWeightHAL(p("mlp.down_proj.weight")), ff, P)
		be.AddInPlace(x, down)
		// Return this layer's panel transients now so the scope never holds more than one
		// layer's set plus the residual.
		for _, t := range []compute.Tensor{xn, q, kRaw, v, qr, kr, attn, o, xn2, g, u, ff, down} {
			be.Free(t)
		}
	}
	done = true
	return x
}
