package model

// v41_forward.go assembles the DeepSeek V4.1 text forward behind new entry
// points (issue #12901, leaf of parent #12640). It composes the already-landed
// V4.1 leaves — the projected-input shared attention state (v41_attention_state.go),
// the mHC coefficient split/mix (v41_mhc.go), the routed+shared router
// (v41_router.go), the sparse sink attention contraction + grouped output
// projection (v41_sparse_attention.go) — into one ordered pass: embedding ->
// per-layer [mHC + attention + MoE] -> final norm -> head logits.
//
// Scope (gold-plating boundary from #12901): TEXT-forward assembly only. No GPU
// qualification, no GGUF/checkpoint loading, no streaming, no vision, no DSpark,
// and no Engram packed-row retrieval (that stage's rows are 264-byte packed FP8
// streams a weight-free in-memory forward cannot materialize). That omission is
// fail-closed: a config declaring an Engram layer WITHIN the decoder stack is
// refused at admission by v41EngramForwardAdmitted, so the assembly never emits
// non-Engram logits for a model it did not fully run (#13007's remaining seam is
// wiring the stage; declaring it is no longer silently dropped). The same holds
// for a declared shared-KV source layer (v41KVSourceForwardAdmitted) and a
// declared compressor/indexer layer (v41CompressIndexForwardAdmitted): the
// reduced assembly projects a per-layer attn.wkv.weight, executes neither
// compression stage, and never reuses a shared KV/index source, so an in-range
// declaration is refused rather than silently run against a per-layer cache. The assembled
// pass is exercised by a REDUCED model
// against an independent scalar oracle (v41_forward_test.go); it does not itself
// qualify the official checkpoint for generation.
//
// Fail-closed design. Every stage is gated by v41ForwardAdmitted before any math
// runs, and forwardV41 re-checks each stage as it goes, returning a typed
// *V41ForwardError. A weightless V4.1 *Model (the probe in
// v41_failclosed_test.go) has no manifest, so admission fails on the embedding
// stage with an error that WRAPS ErrV41NativeUnsupported — keeping
// errors.Is(err, ErrV41NativeUnsupported) true for that fence test — while a
// fully-populated reduced model proceeds. A loaded model missing exactly one
// stage fails with an error that wraps ErrV41ForwardStage. forwardV41 NEVER
// returns logits when a stage is missing.

import (
	"errors"
	"fmt"
)

// ErrV41ForwardStage reports that a required V4.1 forward stage could not be
// assembled (absent weight, inconsistent shape, or missing stream/state). It is
// the closed class every stage-specific V41ForwardError wraps.
var ErrV41ForwardStage = errors.New("model: DeepSeek V4.1 forward stage is not implemented")

// v41ForwardStage names the ordered assembly stage a failure occurred at.
type v41ForwardStage string

const (
	v41StageEmbedding v41ForwardStage = "embedding"
	v41StageLayer     v41ForwardStage = "layer"
	v41StageMHC       v41ForwardStage = "mhc"
	v41StageAttention v41ForwardStage = "attention"
	v41StageMoE       v41ForwardStage = "moe"
	v41StageEngram    v41ForwardStage = "engram"
	v41StageFinalNorm v41ForwardStage = "final_norm"
	v41StageHead      v41ForwardStage = "head"
	v41StageCompress  v41ForwardStage = "compress"
	v41StageIndexer   v41ForwardStage = "indexer"
)

// v41StageCompress and v41StageIndexer are the stage names for the CED/CSA2
// compressor and the lightning indexer. The reduced text forward does not execute
// them (their packed-row / compressed-stream inputs cannot be materialized
// weight-free — see the scope note at the top of this file), so they stay
// fail-closed: a config declaring an in-range compressor or indexer layer is
// refused at admission by v41CompressIndexForwardAdmitted (#13006). They become
// live when a leaf lands the compressor/indexer execution.

// V41ForwardError is the typed fail-closed error for one V4.1 assembly stage.
// Layer is the zero-based decoder layer the stage belongs to, or -1 for a
// model-global stage (embedding, final norm, head).
type V41ForwardError struct {
	Stage v41ForwardStage
	Layer int
	Err   error
}

func (e *V41ForwardError) Error() string {
	if e == nil {
		return "model: nil V4.1 forward error"
	}
	where := ""
	if e.Layer >= 0 {
		where = fmt.Sprintf(" layer=%d", e.Layer)
	}
	if e.Err != nil {
		return fmt.Sprintf("model: V4.1 forward stage %s%s: %v", e.Stage, where, e.Err)
	}
	return fmt.Sprintf("model: V4.1 forward stage %s%s failed", e.Stage, where)
}

// Unwrap exposes the wrapped cause so errors.Is reaches ErrV41ForwardStage or,
// for the weightless probe, ErrV41NativeUnsupported.
func (e *V41ForwardError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func v41StageErr(stage v41ForwardStage, layer int, err error) error {
	if err == nil {
		err = ErrV41ForwardStage
	}
	return &V41ForwardError{Stage: stage, Layer: layer, Err: err}
}

// v41ForwardState is the session-owned continuation state for the reduced V4.1
// text forward. It carries the committed token history plus a lazily-built
// V41AttentionState used to validate the per-layer projected KV window. The
// assembly is cacheless (it recomputes the full history each call, exactly like
// the dedicated gemma4 session bridge), so Prefill followed by Step is
// guaranteed consistent with a single longer Forward: the Step path simply
// re-runs the whole history.
type v41ForwardState struct {
	history []int
	attn    *V41AttentionState
}

func (st *v41ForwardState) attentionState(headDim, ratioCap int) (*V41AttentionState, error) {
	if st.attn != nil {
		return st.attn, nil
	}
	state, err := NewV41AttentionState(headDim, ratioCap)
	if err != nil {
		return nil, err
	}
	st.attn = state
	return state, nil
}

func (st *v41ForwardState) appendHistory(ids []int) {
	st.history = append(st.history, ids...)
}

// ---- reduced V4.1 geometry helpers -----------------------------------------

// v41KVLoraRankReduced is the reduced-model KV latent width. The published V4.1
// KV latent (v41KVLoraRank = 512) is a checkpoint constant the text Config does
// not carry; the reduced assembly deliberately uses HeadDim so the reduced test
// fixture stays tiny and self-consistent. It is not a claim about the official
// checkpoint geometry.
func v41KVLoraRankReduced(cfg Config) int { return cfg.HeadDim }

// v41MHCMixWidth is the mHC coefficient-vector width for hc=4: (2+hc)*hc.
const v41MHCMixWidth = 24

// v41RouterConfigFor builds the routed+shared geometry directly from Config
// rather than through v41RouterConfigFromConfig, which deliberately admits only
// the published 384/top-6/1-shared/1.5-scale envelope. The reduced fixture keeps
// the published expert/top-k envelope (v41Route.validate still enforces 384), so
// this reads the live config rather than hardcoding it.
func v41RouterConfigFor(cfg Config) (v41RouterConfig, error) {
	if cfg.NumExperts <= 0 {
		return v41RouterConfig{}, v41StageErr(v41StageMoE, -1,
			fmt.Errorf("%w: router needs a positive expert count, got %d", ErrV41ForwardStage, cfg.NumExperts))
	}
	if cfg.NumExpertsPerTok <= 0 || cfg.NumExpertsPerTok > cfg.NumExperts {
		return v41RouterConfig{}, v41StageErr(v41StageMoE, -1,
			fmt.Errorf("%w: top-k %d outside [1,%d]", ErrV41ForwardStage, cfg.NumExpertsPerTok, cfg.NumExperts))
	}
	if cfg.NSharedExperts < 1 {
		return v41RouterConfig{}, v41StageErr(v41StageMoE, -1,
			fmt.Errorf("%w: shared experts %d must be >= 1", ErrV41ForwardStage, cfg.NSharedExperts))
	}
	scale := float32(cfg.RoutedScalingFactor)
	if scale == 0 {
		scale = V41RouterRouteScale
	}
	if !finite32(scale) || scale <= 0 {
		return v41RouterConfig{}, v41StageErr(v41StageMoE, -1,
			fmt.Errorf("%w: route scale %g must be finite and positive", ErrV41ForwardStage, scale))
	}
	return v41RouterConfig{Experts: cfg.NumExperts, TopK: cfg.NumExpertsPerTok, SharedCount: cfg.NSharedExperts, RouteScale: scale}, nil
}

// ---- admission -------------------------------------------------------------

// v41EngramForwardAdmitted fails closed when the config declares an Engram layer
// that lies WITHIN the model's decoder stack. The reduced text assembly does not
// execute the Engram stage (its packed-row inputs are not materializable
// weight-free), so silently dropping a declared in-range Engram layer would emit
// logits for a model the assembly never ran. Declared Engram layers outside
// [0,NumLayers) are unreachable by this forward and stay admitted, which keeps the
// reduced oracle fixture (NumLayers=1, EngramLayerIDs [1,14]) runnable. Wiring the
// real Engram stage is #13007's remaining integration work; until then this is the
// fail-closed boundary.
func v41EngramForwardAdmitted(cfg Config) error {
	m := cfg.DeepSeekV41
	if m == nil {
		return nil
	}
	for _, layer := range m.EngramLayerIDs {
		if layer >= 0 && layer < cfg.NumLayers {
			return v41StageErr(v41StageEngram, layer,
				fmt.Errorf("%w: layer %d declares Engram but the reduced forward does not execute the Engram stage", ErrV41ForwardStage, layer))
		}
	}
	return nil
}

// v41CompressIndexForwardAdmitted fails closed when the config declares a
// CED/CSA2 compressor regime or a lightning-indexer source that lies WITHIN the
// model's decoder stack. The reduced text assembly executes neither stage (their
// packed-row / compressed-stream inputs cannot be materialized weight-free — see
// the scope note at the top of this file), so silently dropping a declared
// in-range compressor/indexer layer would emit reduced logits for a model the
// assembly never ran. Declarations that only touch out-of-range layers stay
// admitted, which keeps the reduced oracle fixture runnable: it derives from the
// published 40-layer config but narrows NumLayers to 1, so every CompressRatios
// entry above index 0 and every index source ({2,8,...}) is unreachable. Executing
// the real compressor/indexer stages is #13006's remaining integration work; until
// then this is the fail-closed boundary.
func v41CompressIndexForwardAdmitted(cfg Config) error {
	m := cfg.DeepSeekV41
	if m == nil {
		return nil
	}
	for layer := 0; layer < cfg.NumLayers && layer < len(m.CompressRatios); layer++ {
		// Ratio 0 and 1 are the uncompressed regimes; a ratio > 1 declares a
		// compressed layer the reduced forward does not execute.
		if m.CompressRatios[layer] > 1 {
			return v41StageErr(v41StageCompress, layer,
				fmt.Errorf("%w: layer %d declares compressor ratio %d but the reduced forward does not execute the CED/CSA2 compressor stage", ErrV41ForwardStage, layer, m.CompressRatios[layer]))
		}
	}
	for _, layer := range m.IndexSourceLayerIDs {
		if layer >= 0 && layer < cfg.NumLayers {
			return v41StageErr(v41StageIndexer, layer,
				fmt.Errorf("%w: layer %d declares a lightning-indexer source but the reduced forward does not execute the indexer stage", ErrV41ForwardStage, layer))
		}
	}
	return nil
}

// v41KVSourceForwardAdmitted fails closed when the config declares a shared-KV
// source layer that lies WITHIN the model's decoder stack. The reduced text
// assembly projects its own per-layer attn.wkv.weight and never consumes a KV
// source layer's shared key/value state (the reference's shared KV/index source
// schedule), so silently running an in-range KV source would emit logits from a
// per-layer KV cache where the official model reuses a source layer's state.
// Declarations that only touch out-of-range layers stay admitted, which keeps the
// reduced oracle fixture runnable: it derives from the published 40-layer config
// but narrows NumLayers to 1, so every kv source ({2,8,14,20}) is unreachable.
// Executing the real shared KV source is #12896's remaining integration work;
// until then this is the fail-closed boundary.
func v41KVSourceForwardAdmitted(cfg Config) error {
	m := cfg.DeepSeekV41
	if m == nil {
		return nil
	}
	for _, layer := range m.KVSourceLayerIDs {
		if layer >= 0 && layer < cfg.NumLayers {
			return v41StageErr(v41StageAttention, layer,
				fmt.Errorf("%w: layer %d declares a shared-KV source but the reduced forward does not execute shared KV/index state", ErrV41ForwardStage, layer))
		}
	}
	return nil
}

// v41ForwardAdmitted returns nil only when this is an admitted V4.1 config with
// every required stage's weights present and shape-consistent. It is the gate
// both Model.Forward and Session.Prefill/Step run before the assembly. A
// weightless model fails at the embedding stage with an error wrapping
// ErrV41NativeUnsupported (the #12967 fence contract); a loaded model missing a
// stage fails with an error wrapping ErrV41ForwardStage.
func (m *Model) v41ForwardAdmitted() error {
	if m == nil {
		return v41StageErr(v41StageEmbedding, -1, fmt.Errorf("%w: nil model", ErrV41NativeUnsupported))
	}
	if !m.Cfg.IsDeepSeekV41() {
		return v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: config is not DeepSeek V4.1", ErrV41ForwardStage))
	}
	if len(m.manifest) == 0 {
		return v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: reduced weights absent", ErrV41NativeUnsupported))
	}
	cfg := m.Cfg
	if err := m.v41AdmitShape("model.embed_tokens.weight", v41StageEmbedding, -1, cfg.VocabSize, cfg.HiddenSize); err != nil {
		return err
	}
	if err := m.v41AdmitShape("model.norm.weight", v41StageFinalNorm, -1, cfg.HiddenSize); err != nil {
		return err
	}
	if !m.has("lm_head.weight") && !m.has("model.embed_tokens.weight") {
		return v41StageErr(v41StageHead, -1,
			fmt.Errorf("%w: no lm_head.weight and no tied embedding", ErrV41ForwardStage))
	}
	if err := m.v41AdmitShape("lm_head.weight", v41StageHead, -1, cfg.VocabSize, cfg.HiddenSize); err != nil {
		return err
	}
	if _, err := v41RouterConfigFor(cfg); err != nil {
		return err
	}
	if err := v41EngramForwardAdmitted(cfg); err != nil {
		return err
	}
	if err := v41CompressIndexForwardAdmitted(cfg); err != nil {
		return err
	}
	if err := v41KVSourceForwardAdmitted(cfg); err != nil {
		return err
	}
	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	qHeadDim := nH * hd
	oDim := cfg.OLoraRank * cfg.OGroups
	I := cfg.MoEIntermediateSize
	for l := 0; l < cfg.NumLayers; l++ {
		if err := m.v41AdmitShape(layerName(l, "attn_norm.weight"), v41StageLayer, l, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn_norm.weight"), v41StageLayer, l, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "mhc.mixes.weight"), v41StageMHC, l, v41MHCMixWidth, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "mhc.base"), v41StageMHC, l, v41MHCMixWidth); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "mhc.scale"), v41StageMHC, l, 3); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wq_a.weight"), v41StageAttention, l, cfg.QLoraRank, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wq_b.weight"), v41StageAttention, l, qHeadDim, cfg.QLoraRank); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wkv.weight"), v41StageAttention, l, v41KVLoraRankReduced(cfg), H); err != nil {
			return err
		}
		// wo_a is the leaf's group-major [Groups, OLoRARank, HeadsPerGroup*HeadDim]
		// tensor, so its total is OLoRARank*NumHeads*HeadDim; wo_b is
		// [H, Groups*OLoRARank]. The declared shapes carry those totals.
		if err := m.v41AdmitShape(layerName(l, "attn.wo_a.weight"), v41StageAttention, l, cfg.OLoraRank, qHeadDim); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wo_b.weight"), v41StageAttention, l, H, oDim); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.sink"), v41StageAttention, l, nH); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn.gate.weight"), v41StageMoE, l, cfg.NumExperts, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn.gate.e_score_correction_bias"), v41StageMoE, l, cfg.NumExperts); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn.shared_experts.w1.weight"), v41StageMoE, l, I, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn.shared_experts.w3.weight"), v41StageMoE, l, I, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn.shared_experts.w2.weight"), v41StageMoE, l, H, I); err != nil {
			return err
		}
		for e := 0; e < cfg.NumExperts; e++ {
			stem := "ffn.experts." + itoa(e)
			if err := m.v41AdmitShape(layerName(l, stem+".w1.weight"), v41StageMoE, l, I, H); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(l, stem+".w3.weight"), v41StageMoE, l, I, H); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(l, stem+".w2.weight"), v41StageMoE, l, H, I); err != nil {
				return err
			}
		}
	}
	_ = hd
	return nil
}

// v41AdmitShape asserts a named tensor is present with the expected shape.
func (m *Model) v41AdmitShape(name string, stage v41ForwardStage, layer int, want ...int) error {
	meta, ok := m.manifest[name]
	if !ok {
		return v41StageErr(stage, layer, fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
	}
	if len(meta.Shape) != len(want) {
		return v41StageErr(stage, layer,
			fmt.Errorf("%w: tensor %s rank %d, want %d", ErrV41ForwardStage, name, len(meta.Shape), len(want)))
	}
	for i := range want {
		if meta.Shape[i] != want[i] {
			return v41StageErr(stage, layer,
				fmt.Errorf("%w: tensor %s shape %v, want %v", ErrV41ForwardStage, name, meta.Shape, want))
		}
	}
	return nil
}

// ---- the ordered assembly --------------------------------------------------

// forwardV41 runs the reduced V4.1 text forward over ids (or the state's full
// token history when st is non-nil) and returns per-position hidden states and
// logits. Every stage is checked; a missing stage returns a typed error and no
// logits. It is package-private: the public entry points are Model.Forward and
// Session.Prefill/Step.
func (m *Model) forwardV41(ids []int, st *v41ForwardState) (*Activations, error) {
	if err := m.v41ForwardAdmitted(); err != nil {
		return nil, err
	}
	cfg := m.Cfg
	// Model.Forward (st == nil) is a pure full-prefill over ids. A session folds
	// ids into its history and recomputes the whole history so Step is consistent
	// with a single longer Forward.
	seq := ids
	if st != nil {
		st.appendHistory(ids)
		seq = st.history
	}
	if len(seq) == 0 {
		return &Activations{}, nil
	}
	for _, id := range seq {
		if id < 0 || id >= cfg.VocabSize {
			return nil, v41StageErr(v41StageEmbedding, -1,
				fmt.Errorf("%w: token id %d out of range [0,%d)", ErrV41ForwardStage, id, cfg.VocabSize))
		}
	}

	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	eps := float32(cfg.RMSNormEps)

	// ---- embedding ----
	embed := m.tensor("model.embed_tokens.weight")
	if len(embed) < cfg.VocabSize*H {
		return nil, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: embedding table has %d values, want %d", ErrV41ForwardStage, len(embed), cfg.VocabSize*H))
	}
	x := make([][]float32, len(seq))
	for t, id := range seq {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], cfg)
	}

	hcIters := 1
	hcEps := float32(1e-6)
	if cfg.DeepSeekV41 != nil {
		if cfg.DeepSeekV41.HCSinkhornIters > 0 {
			hcIters = cfg.DeepSeekV41.HCSinkhornIters
		}
		if cfg.DeepSeekV41.HCEps > 0 {
			hcEps = float32(cfg.DeepSeekV41.HCEps)
		}
	}
	routeCfg, err := v41RouterConfigFor(cfg)
	if err != nil {
		return nil, err
	}

	act := &Activations{Seq: len(seq), Hidden: [][]float32{flatten(x)}}
	for l := 0; l < cfg.NumLayers; l++ {
		if err := m.v41Layer(l, x, hd, nH, H, eps, hcIters, hcEps, routeCfg, st); err != nil {
			return nil, err
		}
		act.Hidden = append(act.Hidden, flatten(x))
	}

	act.Logits = make([][]float32, len(seq))
	for t := 0; t < len(seq); t++ {
		logits, err := m.v41Head(x[t])
		if err != nil {
			return nil, err
		}
		act.Logits[t] = logits
	}
	return act, nil
}

// v41Layer applies one reduced V4.1 decoder layer to x in place.
func (m *Model) v41Layer(l int, x [][]float32, hd, nH, H int, eps float32, hcIters int, hcEps float32, routeCfg v41RouterConfig, st *v41ForwardState) error {
	cfg := m.Cfg
	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	ffnNorm := m.tensor(layerName(l, "ffn_norm.weight"))
	wMix := m.tensor(layerName(l, "mhc.mixes.weight"))
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	wQA := m.tensor(layerName(l, "attn.wq_a.weight"))
	wQB := m.tensor(layerName(l, "attn.wq_b.weight"))
	wKV := m.tensor(layerName(l, "attn.wkv.weight"))
	woA := m.tensor(layerName(l, "attn.wo_a.weight"))
	woB := m.tensor(layerName(l, "attn.wo_b.weight"))
	sink := m.tensor(layerName(l, "attn.sink"))
	wGate := m.tensor(layerName(l, "ffn.gate.weight"))
	gateBias := m.tensor(layerName(l, "ffn.gate.e_score_correction_bias"))
	sharedW1 := m.tensor(layerName(l, "ffn.shared_experts.w1.weight"))
	sharedW3 := m.tensor(layerName(l, "ffn.shared_experts.w3.weight"))
	sharedW2 := m.tensor(layerName(l, "ffn.shared_experts.w2.weight"))

	seq := len(x)
	// ---- mHC coefficient split (one split per layer) ----
	hcByPos := make([]v41MHCMix, seq)
	preByPos := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xn := rmsnormCfg(x[t], attnNorm, eps, cfg)
		mixes := matRows(wMix, xn, v41MHCMixWidth, H)
		mix, err := v41MHCSplit(mixes, mixScale, mixBase, 4, hcIters, hcEps)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		hcByPos[t] = mix
		// Reduced stand-in for the reference's hc persistent streams: four
		// identical copies of the normalized input. v41MHCPre then collapses them
		// with the learned pre coefficients.
		streams := [][]float32{xn, xn, xn, xn}
		collapsed, err := v41MHCPre(streams, mix.pre)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		preByPos[t] = collapsed
	}

	// ---- attention: projected q/kv per position, then sparse sink per position ----
	qHeads := make([][]float32, seq) // [t][nH*hd], rotated
	kvRows := make([][]float32, seq) // [t][hd], rotated (single KV head)
	for t := 0; t < seq; t++ {
		c := preByPos[t]
		qLat := matRows(wQA, c, cfg.QLoraRank, H)
		q := matRows(wQB, qLat, nH*hd, cfg.QLoraRank)
		kv := matRows(wKV, c, hd, H)
		cos, sin := ropeRowForLayer(cfg, l, t)
		for h := 0; h < nH; h++ {
			applyRopeRow(q[h*hd:(h+1)*hd], cos, sin)
		}
		applyRopeRow(kv, cos, sin)
		qHeads[t] = q
		kvRows[t] = kv
	}

	// V41AttentionState is the session-owned validation anchor for the projected
	// KV window. It runs no projection and never changes the arithmetic; it fails
	// closed on malformed geometry before the sink contraction reads the rows.
	//
	// The state's window ring is a fixed v41WindowSize rows (v41_attention_state.go).
	// It is a BOUNDED validation anchor, not the attention path: when the committed
	// history exceeds the window the assembly still contracts every causal row below
	// through the scalar sink, so it seeds the anchor only up to the window and never
	// fails a legitimate long sequence on the anchor's own bound.
	if st != nil {
		attn, err := st.attentionState(hd, 8)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		attn.Reset()
		seed := kvRows
		if len(seed) > v41WindowSize {
			seed = seed[len(seed)-v41WindowSize:]
		}
		if len(seed) > 0 {
			if err := attn.Prefill(seed, nil); err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
		}
	}

	scale := cfg.attnScale()
	attnOut := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		rows := t + 1
		idx := make([]int32, rows+1) // one trailing -1 marks an empty slot beyond the causal prefix
		for i := 0; i < rows; i++ {
			idx[i] = int32(i)
		}
		idx[rows] = -1
		flatKV := make([]float32, 0, rows*hd)
		for i := 0; i <= t; i++ {
			flatKV = append(flatKV, kvRows[i]...)
		}
		o, err := V41SparseAttentionSink(qHeads[t], flatKV, sink, idx, V41SparseAttentionSinkOptions{
			B: 1, M: 1, Heads: nH, HeadDim: hd, TopK: rows + 1, N: rows, Softmax: scale,
		})
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		projected, err := V41GroupedOutputProjection(o, woA, woB, 1, 1, nH, hd, cfg.OGroups, cfg.OLoraRank, H)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		attnOut[t] = projected
	}

	// ---- MoE: router + shared expert + routed experts ----
	for t := 0; t < seq; t++ {
		xn := rmsnormCfg(x[t], ffnNorm, eps, cfg)
		routerLogits := matRows(wGate, xn, cfg.NumExperts, H)
		picks, err := v41Route(routerLogits, gateBias, routeCfg)
		if err != nil {
			return v41StageErr(v41StageMoE, l, err)
		}
		routed := make([]float32, H)
		for _, pick := range picks {
			stem := "ffn.experts." + itoa(pick.expert)
			w1 := m.tensor(layerName(l, stem+".w1.weight"))
			w3 := m.tensor(layerName(l, stem+".w3.weight"))
			w2 := m.tensor(layerName(l, stem+".w2.weight"))
			y := v41SwiGLU(w1, w3, w2, xn, cfg.MoEIntermediateSize, H, cfg)
			for i := range routed {
				routed[i] += pick.weight * y[i]
			}
		}
		shared := v41SwiGLU(sharedW1, sharedW3, sharedW2, xn, cfg.MoEIntermediateSize, H, cfg)
		moe, err := v41SharedExpertAdd(routed, shared, routeCfg)
		if err != nil {
			return v41StageErr(v41StageMoE, l, err)
		}

		// delta = attention + MoE, then the mHC post-mix back into four streams.
		delta := make([]float32, H)
		for i := 0; i < H; i++ {
			delta[i] = attnOut[t][i] + moe[i]
		}
		streams := [][]float32{preByPos[t], preByPos[t], preByPos[t], preByPos[t]}
		next, err := v41MHCPost(delta, streams, hcByPos[t].post, hcByPos[t].comb)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		copy(x[t], next[0])
	}
	return nil
}

// v41SwiGLU is the standard SwiGLU expert/sub-layer: down(silu(w1 x) * w3 x).
func v41SwiGLU(w1, w3, w2, xn []float32, I, H int, cfg Config) []float32 {
	h1 := matRows(w1, xn, I, H)
	h3 := matRows(w3, xn, I, H)
	h := make([]float32, I)
	for i := 0; i < I; i++ {
		h[i] = act(h1[i], cfg) * h3[i]
	}
	return matRows(w2, h, H, I)
}

// v41Head runs the final norm and LM head for one hidden vector. The final norm
// itself is applied inside m.logitsFromHidden (forward.go), which is the single
// shared tail Model.Forward and the parallel twins also end at; this wrapper only
// asserts the final-norm weight is present and the hidden width is admitted, so a
// missing norm weight fails closed before the shared tail runs.
func (m *Model) v41Head(x []float32) ([]float32, error) {
	if !m.has("model.norm.weight") {
		return nil, v41StageErr(v41StageFinalNorm, -1,
			fmt.Errorf("%w: missing model.norm.weight", ErrV41ForwardStage))
	}
	if len(x) != m.Cfg.HiddenSize {
		return nil, v41StageErr(v41StageFinalNorm, -1,
			fmt.Errorf("%w: final norm width %d, want %d", ErrV41ForwardStage, len(x), m.Cfg.HiddenSize))
	}
	if !m.has("lm_head.weight") && !m.has("model.embed_tokens.weight") {
		return nil, v41StageErr(v41StageHead, -1,
			fmt.Errorf("%w: no lm_head.weight and no tied embedding", ErrV41ForwardStage))
	}
	return m.logitsFromHidden(x), nil
}

// ---- session entry points --------------------------------------------------

// prefillV41 is the DeepSeek V4.1 branch of Session.Prefill. It fails closed
// (panics) with a typed error before any math when a stage is missing, matching
// the requirePreNorm convention but preserving the #12967 weightless-probe
// errors.Is(err, ErrV41NativeUnsupported) contract.
func (s *Session) prefillV41(ids []int) []float32 {
	if err := s.M.v41ForwardAdmitted(); err != nil {
		panic(err)
	}
	act, err := s.M.forwardV41(ids, s.v41State())
	if err != nil {
		panic(err)
	}
	return lastLogits(act)
}

// stepV41 is the DeepSeek V4.1 branch of Session.Step. The assembly is
// cacheless: Step folds the token into the committed history and recomputes the
// whole history, so its logits are exactly a longer Forward's last-position
// logits (the #12901 prefill/step consistency property).
func (s *Session) stepV41(id int) []float32 {
	if err := s.M.v41ForwardAdmitted(); err != nil {
		panic(err)
	}
	act, err := s.M.forwardV41([]int{id}, s.v41State())
	if err != nil {
		panic(err)
	}
	return lastLogits(act)
}

// v41State lazily installs the session's V4.1 continuation state.
func (s *Session) v41State() *v41ForwardState {
	if s.v41Forward == nil {
		s.v41Forward = &v41ForwardState{}
	}
	return s.v41Forward
}

func lastLogits(act *Activations) []float32 {
	if act == nil || len(act.Logits) == 0 {
		return nil
	}
	return act.Logits[len(act.Logits)-1]
}
