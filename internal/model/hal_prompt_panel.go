package model

import (
	"fmt"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// promptPanelBackend is the optional device seam the HAL prefill uses to ingest a prompt
// as [rows, width] panels: per layer one GEMM per projection over the rows instead of one GEMV
// per token. Every op must be the per-row form of the op tokenHALOutput issues for one token,
// so a panel row is bitwise the token loop's row whatever the row count or chunking (the BitExact
// default). Row t of a panel sits at absolute position start+t. A backend that implements it
// must also run RMSNorm, AddBias, SwiGLU and AddInPlace row-wise over [rows, width] panels.
// For deepseek_v4 the routed FFN stays a per-row primitive (applyV4ExpertHAL): the panel
// extracts each residual row, runs the token loop's exact FFN on it, and writes the updated
// row back through PromptPanelSetRow.
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
	// PromptPanelSetRow copies src into row of the [rows, width] panel x, the inverse of
	// PromptPanelRow: a per-row FFN updates its residual back into the panel.
	PromptPanelSetRow(x compute.Tensor, row, width int, src compute.Tensor)
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

// promptPanelWitnessReason names one outcome axis of promptPanelFor: the positive selection
// plus every refusal axis the seam can take. The fixed set is the coverage matrix itself: a
// new refusal axis must add its reason here or TestPromptPanelSelectionMatrix cannot name it.
type promptPanelWitnessReason int

const (
	promptPanelSelected promptPanelWitnessReason = iota
	promptPanelRefusalShortPrompt
	promptPanelRefusalNoBackend
	promptPanelRefusalNoHALKV
	promptPanelRefusalBackendLacksSeam
	promptPanelRefusalArchQ8TokenLoop
	promptPanelRefusalArchDeepSeekV4NonPanel
	promptPanelRefusalArchQKNorm
	promptPanelRefusalArchMLAMoELayout
	promptPanelRefusalArchMiniMaxSparseAttn
	promptPanelRefusalArchGemma4
	promptPanelRefusalCaptureTarget
	promptPanelRefusalQ2KEmbedding
	promptPanelRefusalSplitGPULayers
	promptPanelRefusalDtypeNotRowExact
	promptPanelWitnessReasons
)

// promptPanelWitness counts every promptPanelFor outcome at the selection seam. Downstream
// fallback counters count declines-after-selection only, so they are structurally blind to
// exclusions-at-selection: without this witness a landed fast path can go uninvoked on the
// live serve path and nothing downstream names why. Fixed reasons, one atomic add per call,
// no locks on the hot path.
var promptPanelWitness [promptPanelWitnessReasons]atomic.Int64

func notePromptPanelSelection(r promptPanelWitnessReason) {
	promptPanelWitness[r].Add(1)
}

// PromptPanelSelectionCounts is a point-in-time copy of the prompt-panel selection witness:
// how many prompts took the panel and how many each refusal axis kept on the token loop.
type PromptPanelSelectionCounts struct {
	Selected               int64
	ShortPrompt            int64
	NoBackend              int64
	NoHALKV                int64
	BackendLacksSeam       int64
	ArchQ8TokenLoop        int64
	ArchDeepSeekV4NonPanel int64
	ArchQKNorm             int64
	ArchMLAMoELayout       int64
	ArchMiniMaxSparseAttn  int64
	ArchGemma4             int64
	CaptureTarget          int64
	Q2KEmbedding           int64
	SplitGPULayers         int64
	DtypeNotRowExact       int64
}

// Total is every counted promptPanelFor outcome, selected or refused.
func (c PromptPanelSelectionCounts) Total() int64 {
	return c.Selected + c.ShortPrompt + c.NoBackend + c.NoHALKV + c.BackendLacksSeam +
		c.ArchQ8TokenLoop + c.ArchDeepSeekV4NonPanel + c.ArchQKNorm + c.ArchMLAMoELayout +
		c.ArchMiniMaxSparseAttn + c.ArchGemma4 + c.CaptureTarget + c.Q2KEmbedding +
		c.SplitGPULayers + c.DtypeNotRowExact
}

// PromptPanelSelectionWitness snapshots the current witness counts. Each field is a plain
// atomic load, so the copy is race-free; fields loaded later may include calls that landed
// between loads.
func PromptPanelSelectionWitness() PromptPanelSelectionCounts {
	return PromptPanelSelectionCounts{
		Selected:               promptPanelWitness[promptPanelSelected].Load(),
		ShortPrompt:            promptPanelWitness[promptPanelRefusalShortPrompt].Load(),
		NoBackend:              promptPanelWitness[promptPanelRefusalNoBackend].Load(),
		NoHALKV:                promptPanelWitness[promptPanelRefusalNoHALKV].Load(),
		BackendLacksSeam:       promptPanelWitness[promptPanelRefusalBackendLacksSeam].Load(),
		ArchQ8TokenLoop:        promptPanelWitness[promptPanelRefusalArchQ8TokenLoop].Load(),
		ArchDeepSeekV4NonPanel: promptPanelWitness[promptPanelRefusalArchDeepSeekV4NonPanel].Load(),
		ArchQKNorm:             promptPanelWitness[promptPanelRefusalArchQKNorm].Load(),
		ArchMLAMoELayout:       promptPanelWitness[promptPanelRefusalArchMLAMoELayout].Load(),
		ArchMiniMaxSparseAttn:  promptPanelWitness[promptPanelRefusalArchMiniMaxSparseAttn].Load(),
		ArchGemma4:             promptPanelWitness[promptPanelRefusalArchGemma4].Load(),
		CaptureTarget:          promptPanelWitness[promptPanelRefusalCaptureTarget].Load(),
		Q2KEmbedding:           promptPanelWitness[promptPanelRefusalQ2KEmbedding].Load(),
		SplitGPULayers:         promptPanelWitness[promptPanelRefusalSplitGPULayers].Load(),
		DtypeNotRowExact:       promptPanelWitness[promptPanelRefusalDtypeNotRowExact].Load(),
	}
}

// promptPanelFor returns the backend's panel seam when this prompt can take it. The panel
// mirrors tokenHALOutput's plain dense branch plus deepseek_v4's per-row live-expert FFN, and
// only weight dtypes with a per-row exact GEMM, so every arch, capture, placement and dtype
// that does not qualify keeps the token loop. Every outcome notes its typed reason in the
// selection witness (notePromptPanelSelection): a refusal is attributed to the first matching
// axis in the order below, mirroring the original left-to-right OR.
func (s *Session) promptPanelFor(ids []int) (promptPanelBackend, bool) {
	switch {
	case len(ids) < 2:
		notePromptPanelSelection(promptPanelRefusalShortPrompt)
		return nil, false
	case s.Backend == nil:
		notePromptPanelSelection(promptPanelRefusalNoBackend)
		return nil, false
	case s.halKV == nil:
		notePromptPanelSelection(promptPanelRefusalNoHALKV)
		return nil, false
	}
	panel, ok := s.Backend.(promptPanelBackend)
	if !ok {
		notePromptPanelSelection(promptPanelRefusalBackendLacksSeam)
		return nil, false
	}
	cfg := s.M.Cfg
	// MoE prefill takes the panel for exactly the arch whose routed FFN is a per-row
	// primitive the panel can interleave after its batched attention sublayer:
	// deepseek_v4's live-expert route (applyV4ExpertHAL). Every other MoE axis keeps the
	// token loop, which owns those FFNs; the non-MoE refusals stay unchanged.
	panelMoE := cfg.IsDeepSeekV4() && !cfg.QKNorm && !cfg.DenseMLP && !cfg.Alibi &&
		!cfg.AttnOutputGate && cfg.BlockTopology == PreNorm && !cfg.hasLayerSpecificRopeTheta()
	if !panelMoE {
		switch {
		case q8PrefillNeedsTokenLoop(cfg):
			notePromptPanelSelection(promptPanelRefusalArchQ8TokenLoop)
			return nil, false
		case cfg.IsDeepSeekV4():
			notePromptPanelSelection(promptPanelRefusalArchDeepSeekV4NonPanel)
			return nil, false
		case cfg.QKNorm:
			notePromptPanelSelection(promptPanelRefusalArchQKNorm)
			return nil, false
		case cfg.usesMLAMoELayout():
			notePromptPanelSelection(promptPanelRefusalArchMLAMoELayout)
			return nil, false
		case cfg.isMiniMaxSparseAttn():
			notePromptPanelSelection(promptPanelRefusalArchMiniMaxSparseAttn)
			return nil, false
		case cfg.isGemma4():
			notePromptPanelSelection(promptPanelRefusalArchGemma4)
			return nil, false
		}
	}
	if s.captureTargetHidden || s.qwen35TargetFeatureTxn != nil || s.qwen35TargetLayerCapture != nil {
		notePromptPanelSelection(promptPanelRefusalCaptureTarget)
		return nil, false
	}
	if s.M.Q2KEmbedding != nil {
		notePromptPanelSelection(promptPanelRefusalQ2KEmbedding)
		return nil, false
	}
	if _, split := s.validateDenseGPULayers(); split {
		notePromptPanelSelection(promptPanelRefusalSplitGPULayers)
		return nil, false
	}
	// The panel GEMMs are the attention projections; a dense layer adds the MLP triple.
	// deepseek_v4's router and expert weights run per row through the token loop's own
	// single-row ops, so they need no row-exact panel dtype.
	projections := promptPanelProjections[:4]
	if !cfg.IsDeepSeekV4() {
		projections = promptPanelProjections[:]
	}
	for l := 0; l < cfg.NumLayers; l++ {
		for _, name := range projections {
			if !panel.PromptPanelRowExact(s.matWeightHAL(layerName(l, name)).Dtype) {
				notePromptPanelSelection(promptPanelRefusalDtypeNotRowExact)
				return nil, false
			}
		}
	}
	notePromptPanelSelection(promptPanelSelected)
	return panel, true
}

// promptPanelChunkTokens sizes a chunk so its peak scratch fits the per-token transient demand
// compute.EstimateHALTransientMemoryPlan already prices, capped at promptPanelMaxTokens.
func (s *Session) promptPanelChunkTokens() int {
	if promptPanelChunkOverride > 0 {
		return promptPanelChunkOverride
	}
	c := s.M.Cfg
	// An MoE-only width declaration (deepseek_v4) prices the row's FFN transient at the
	// expert width; the live-expert transients are hidden/router-wide, so this over-prices
	// conservatively and the estimate only ever refuses MORE rows.
	return compute.PromptPanelChunkTokens(compute.TransformerScratchConfig{
		HiddenSize: c.HiddenSize, IntermediateSize: c.ScratchFFNWidth(), VocabSize: c.VocabSize,
		NumLayers: c.NumLayers, NumHeads: c.NumHeads, NumKVHeads: c.NumKVHeads, HeadDim: c.HeadDim,
		IncludeLogits: true,
	}, promptPanelMaxTokens)
}

// prefillHALPanel is the batched twin of the prefillHAL token loop for a PreNorm model. It
// ingests the prompt in bounded chunks; per chunk and layer it issues one RMSNorm, one
// per-row GEMM per projection, one RoPE, one KV append and one causal attention. A
// deepseek_v4 layer's routed FFN stays per row: each residual row runs the token loop's exact
// live-expert FFN and is written back into the panel (PromptPanelSetRow).
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
			open = false
			// Normal completion closes explicitly below. During body panic or
			// Goexit, preserve that unwind even if checked cleanup also panics.
			defer func() { _ = recover() }()
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
	open = false
	panel.PromptPanelEnd()
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

		var xn2, g, u, ff, down compute.Tensor
		if cfg.IsDeepSeekV4() {
			// Routed FFN: the live-expert runtime is a per-token primitive, so each panel
			// row runs the token loop's exact FFN (router + top-k experts + shared add) and
			// writes its updated residual back into the panel. Row t is prompt token ids[t].
			for t := 0; t < P; t++ {
				row := panel.PromptPanelRow(x, t, H)
				if err := s.applyV4ExpertHAL(l, ids[t], row, s.normWeightHAL(p("post_attention_layernorm.weight")), eps); err != nil {
					be.Free(row)
					panic(err)
				}
				panel.PromptPanelSetRow(x, t, H, row)
				be.Free(row)
			}
		} else {
			xn2 = be.RMSNorm(x, s.normWeightHAL(p("post_attention_layernorm.weight")), eps)
			g = panel.PromptPanelMatMul(s.matWeightHAL(p("mlp.gate_proj.weight")), xn2, P)
			u = panel.PromptPanelMatMul(s.matWeightHAL(p("mlp.up_proj.weight")), xn2, P)
			ff = be.SwiGLU(g, u)
			down = panel.PromptPanelMatMul(s.matWeightHAL(p("mlp.down_proj.weight")), ff, P)
			be.AddInPlace(x, down)
		}
		// Return this layer's panel transients now so the scope never holds more than one
		// layer's set plus the residual.
		transients := []compute.Tensor{xn, q, kRaw, v, qr, kr, attn, o}
		if !cfg.IsDeepSeekV4() {
			transients = append(transients, xn2, g, u, ff, down)
		}
		for _, t := range transients {
			be.Free(t)
		}
	}
	done = true
	return x
}
