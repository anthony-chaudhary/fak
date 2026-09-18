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
// qualification, no GGUF/checkpoint loading, no streaming, no vision, no DSpark.
// Engram packed-row retrieval is wired as of #13007 (v41_forward_engram.go):
// a declared in-range Engram layer is admitted only when the model carries a
// packed-row stage and the three mixing tensors, and the stage is injected at
// the START of the layer per the reference schedule. That is fail-closed: a
// config declaring an in-range Engram layer WITHOUT a wired row source is
// refused at admission by v41EngramForwardAdmitted with an error wrapping
// ErrV41NativeUnsupported, so the assembly never emits non-Engram logits for a
// model it did not fully run. The same holds
// for a declared shared-KV source layer (v41KVSourceForwardAdmitted) and a
// declared compressor/indexer layer (v41CompressIndexForwardAdmitted): the
// reduced assembly projects a per-layer attn.wkv.weight, executes neither
// compression stage, and never reuses a shared KV/index source, so an in-range
// declaration is refused rather than silently run against a per-layer cache. The
// blocked-candidate selection the lightning indexer reads is likewise not executed
// (v41CandidateSourceForwardAdmitted), so an in-range candidate source is refused
// rather than silently run as unblocked attention. The
// mHC hyperconnection multiplicity is likewise fixed at the published four-stream
// layout (v41HCMultForwardAdmitted), so a declared hc_mult other than 4 is refused
// rather than silently run as four streams. The assembled
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
	"math"
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
// compressor and the lightning indexer. As of #13006 the reduced text forward
// EXECUTES both stages: the compressor pools a compressed layer's projected KV
// rows (v41CompressedRows), and the lightning indexer scores a declared index
// source against those compressed keys and selects rows (v41IndexRows). A config
// declaring an in-range compressor or indexer layer without its weights is still
// refused at admission by v41CompressIndexForwardAdmitted; a malformed schedule
// fails closed there too.

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

// v41ForwardGeometry reports whether cfg declares the published full V4.1
// geometry (true) or the reduced test fixture (false). It FAILS CLOSED: when the
// parsed DeepSeekV41.Attention envelope is populated (a published/parsed config)
// and the config is not a coherently narrowed reduced fixture, the full geometry
// is REQUIRED and any mismatch is a typed ErrV41ForwardStage -- the assembly
// never silently falls back to the reduced stand-in. A lone-axis drift (a mutated
// head width while the decoder stack stays at the published envelope) is refused.
//
// The discriminator is the Attention envelope's HeadDim plus the flat layer
// count: a reduced fixture narrows both the decoder stack and the head width
// (NumLayers 40 -> 1, HeadDim 512 -> 32). A config with no DeepSeekV41 metadata or
// an empty envelope is classified by its own flat HeadDim: 512 (v41KVLoraRank)
// means full, anything else (the reduced fixture's 32) means reduced.
func v41ForwardGeometry(cfg Config) (bool, error) {
	m := cfg.DeepSeekV41
	if m == nil || m.Attention.HeadDim == 0 {
		return cfg.HeadDim == v41KVLoraRank, nil
	}
	attn := m.Attention
	// A parsed/published config derives its Attention envelope from the SAME flat
	// geometry, so a genuine published config always agrees with its envelope on
	// every decoder axis. Two situations can disagree:
	//
	//   * The reduced fixture reuses the retained published metadata pointer but
	//     narrows the flat decoder stack (NumLayers 40 -> 1) AND the head width
	//     (HeadDim 512 -> 32) -- a deliberate, coherent narrowing. That config is a
	//     fixture and is classified by its flat head width.
	//   * A corrupted full config drifts a single axis (typically the head width)
	//     while leaving the rest of the decoder stack at the published envelope.
	//     That is NOT a deliberate reduction and FAILS CLOSED: a full-intended
	//     config must never silently run the reduced stand-in geometry.
	//
	// The two are distinguished by whether the flat decoder stack itself was
	// narrowed below the envelope. Requiring BOTH a non-published head width and a
	// narrowed layer count keeps the reduced fixture admitted while refusing a
	// lone-axis drift.
	if attn.HeadDim != cfg.HeadDim {
		reducedFixture := cfg.HeadDim != v41KVLoraRank && cfg.NumLayers < attn.NumLayers
		if !reducedFixture {
			return false, v41StageErr(v41StageAttention, -1,
				fmt.Errorf("%w: published V4.1 attention envelope declares head_dim=%d but config has %d and the decoder stack is not a narrowed fixture (layers %d vs %d)",
					ErrV41ForwardStage, attn.HeadDim, cfg.HeadDim, cfg.NumLayers, attn.NumLayers))
		}
		return cfg.HeadDim == v41KVLoraRank, nil
	}
	// Authoritative (self-consistent) envelope: the full published geometry is
	// REQUIRED. Any mismatch on a decoder axis is a typed fail-closed error -- the
	// assembly never falls back to the reduced stand-in.
	type axis struct {
		name      string
		got, want int
	}
	for _, a := range []axis{
		{"head_dim", cfg.HeadDim, v41KVLoraRank},
		{"num_hidden_layers", cfg.NumLayers, attn.NumLayers},
		{"hidden_size", cfg.HiddenSize, attn.HiddenSize},
		{"num_attention_heads", cfg.NumHeads, attn.NumHeads},
		{"num_key_value_heads", cfg.NumKVHeads, attn.NumKVHeads},
	} {
		if a.got != a.want {
			return false, v41StageErr(v41StageAttention, -1,
				fmt.Errorf("%w: published V4.1 attention envelope declares %s=%d but config has %d", ErrV41ForwardStage, a.name, a.want, a.got))
		}
	}
	return true, nil
}

// v41ForwardKVLatentRank resolves the attn.wkv output width for an admitted
// config: the published v41KVLoraRank (512) on the full path, the tiny HeadDim
// stand-in on the reduced fixture. It returns v41ForwardGeometry's error rather
// than defaulting to the reduced width, so a config whose published envelope is
// inconsistent can never be admitted against a reduced stand-in shape.
func v41ForwardKVLatentRank(cfg Config) (int, error) {
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return 0, err
	}
	if full {
		return v41KVLoraRank, nil
	}
	return v41KVLoraRankReduced(cfg), nil
}

// v41MHCMixWidth is the mHC coefficient-vector width for hc=4: (2+hc)*hc.
const v41MHCMixWidth = 24

// v41RouterConfigFullGeometry is the real-path routed+shared geometry. It routes
// through v41RouterConfigFromConfig so the forward only admits the published
// 384/top-6/1-shared/1.5-scale envelope; any other axis is refused here rather
// than silently running a different MoE geometry. The router admission error is
// wrapped into the typed ErrV41ForwardStage so a config missing (or carrying a
// non-published) full-geometry axis fails closed at the forward stage boundary.
func v41RouterConfigFullGeometry(cfg Config) (v41RouterConfig, error) {
	rc, err := v41RouterConfigFromConfig(cfg)
	if err != nil {
		return v41RouterConfig{}, v41StageErr(v41StageMoE, -1,
			fmt.Errorf("%w: published router envelope rejected: %w", ErrV41ForwardStage, err))
	}
	return rc, nil
}

// ---- admission -------------------------------------------------------------

// v41EngramForwardAdmitted fails closed when the config declares an Engram layer
// that lies WITHIN the model's decoder stack but the model's Engram stage is not
// fully wired for it. As of #13007 the reduced text assembly DOES execute
// packaged-row Engram retrieval (v41_forward_engram.go): a declared in-range
// Engram layer is admitted only when it has a wired row source and the three
// mixing tensors (engram_kv.weight, engram_q_norm.weight, engram_k_norm.weight).
// Otherwise the assembly would emit logits for a model it did not fully run, so
// the layer is refused with an error wrapping ErrV41NativeUnsupported — the same
// native-forward-unavailable class the #12967 weightless fence uses, because the
// Engram stage cannot execute without its packed-row source. Declared Engram
// layers outside [0,NumLayers) are unreachable by this forward and stay admitted,
// which keeps the reduced oracle fixture (NumLayers=1, EngramLayerIDs [1,14])
// runnable.
//
// Method receiver (not a bare Config) is required because admission must see
// whether THIS model carries the packed-row stage; v41ForwardAdmitted calls it
// after the embedding/manifest checks so a weightless model still fails at the
// embedding fence with ErrV41NativeUnsupported first.
func (m *Model) v41EngramForwardAdmitted() error {
	d41 := m.Cfg.DeepSeekV41
	if d41 == nil {
		return nil
	}
	for _, layer := range d41.EngramLayerIDs {
		if layer < 0 || layer >= m.Cfg.NumLayers {
			continue
		}
		stage := m.v41EngramStageFor()
		if stage == nil || stage.cacheIndex(layer) < 0 {
			return v41StageErr(v41StageEngram, layer,
				fmt.Errorf("%w: layer %d declares Engram but no packed-row source is wired", ErrV41NativeUnsupported, layer))
		}
		cols := stage.columns
		H := m.Cfg.HiddenSize
		if err := m.v41AdmitShape(layerName(layer, "engram_kv.weight"), v41StageEngram, layer, cols*stage.headDim, (stage.hc+1)*H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "engram_q_norm.weight"), v41StageEngram, layer, stage.hc*H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "engram_k_norm.weight"), v41StageEngram, layer, stage.hc*H); err != nil {
			return err
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
//
// Two malformed-schedule arms are also refused here: a negative ratio (invalid
// geometry, never a compressed layer) and a schedule shorter than the decoder
// stack (an uncovered layer with no declared regime). Both must fail closed, not
// fall through to the uncompressed regime.
func (m *Model) v41CompressIndexForwardAdmitted() error {
	d41 := m.Cfg.DeepSeekV41
	if d41 == nil {
		return nil
	}
	cfg := m.Cfg
	// Every layer in the model's decoder stack must declare a regime. A schedule
	// shorter than the stack leaves the uncovered layers with no declared
	// compression regime, which must fail closed rather than being silently read as
	// ratio 0.
	if len(d41.CompressRatios) < cfg.NumLayers {
		return v41StageErr(v41StageCompress, len(d41.CompressRatios),
			fmt.Errorf("%w: compression schedule declares %d ratios but the model has %d layers", ErrV41ForwardStage, len(d41.CompressRatios), cfg.NumLayers))
	}
	H := cfg.HiddenSize
	for layer := 0; layer < cfg.NumLayers; layer++ {
		// Ratio 0 and 1 are the uncompressed regimes. A ratio > 1 declares a
		// compressed layer whose compressor must be wired; a negative ratio is
		// malformed geometry that must fail closed rather than being silently
		// treated as uncompressed.
		ratio := d41.CompressRatios[layer]
		switch {
		case ratio < 0:
			return v41StageErr(v41StageCompress, layer,
				fmt.Errorf("%w: layer %d declares malformed compressor ratio %d", ErrV41ForwardStage, layer, ratio))
		case ratio > 1:
			width := v41CompressorWidth(cfg)
			if err := m.v41AdmitShape(layerName(layer, "attn.compressor.wkv.weight"), v41StageCompress, layer, width, H); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(layer, "attn.compressor.wgate.weight"), v41StageCompress, layer, width, H); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(layer, "attn.compressor.norm.weight"), v41StageCompress, layer, width); err != nil {
				return err
			}
		}
	}
	indexHeads := cfg.IndexNHeads
	indexDim := cfg.IndexHeadDim
	for _, layer := range d41.IndexSourceLayerIDs {
		if layer < 0 || layer >= cfg.NumLayers {
			continue
		}
		if indexHeads <= 0 || indexDim <= 0 {
			return v41StageErr(v41StageIndexer, layer,
				fmt.Errorf("%w: layer %d declares a lightning-indexer source but the indexer geometry (nHeads=%d headDim=%d) is not declared", ErrV41ForwardStage, layer, indexHeads, indexDim))
		}
		wq := indexHeads * indexDim
		if err := m.v41AdmitShape(layerName(layer, "indexer.wq_b.weight"), v41StageIndexer, layer, wq, cfg.QLoraRank); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "indexer.wk.weight"), v41StageIndexer, layer, indexDim, v41CompressorWidth(cfg)); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "indexer.k_norm.weight"), v41StageIndexer, layer, indexDim); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "indexer.weights_proj.weight"), v41StageIndexer, layer, indexHeads, H); err != nil {
			return err
		}
	}
	return nil
}

// v41CompressorWidth is the compressor latent width. The published V4.1
// compressor pools to the KV latent width; the reduced text assembly does not
// carry that checkpoint constant on the text Config, so it pools at HeadDim,
// matching v41KVLoraRankReduced. It is not a claim about the official geometry.
func v41CompressorWidth(cfg Config) int { return cfg.HeadDim }

// v41CompressedRows pools the per-position projected KV rows of one layer
// through the CED/CSA2 compressor. It returns the emitted compressed rows in
// causal order. A non-compressed regime returns the input rows unchanged. It
// fails closed on any malformed geometry rather than emitting a partial stream.
func (m *Model) v41CompressedRows(l int, ratio int, kvRows [][]float32, inputs [][]float32) ([][]float32, error) {
	if ratio <= 1 {
		return kvRows, nil
	}
	cfg := m.Cfg
	width := v41CompressorWidth(cfg)
	H := cfg.HiddenSize
	wkv := m.tensor(layerName(l, "attn.compressor.wkv.weight"))
	wgate := m.tensor(layerName(l, "attn.compressor.wgate.weight"))
	normWeight := m.tensor(layerName(l, "attn.compressor.norm.weight"))
	eps := float32(cfg.RMSNormEps)
	pool, err := NewV41CompressorPool(ratio, width)
	if err != nil {
		return nil, v41StageErr(v41StageCompress, l, err)
	}
	var out [][]float32
	for pos := range inputs {
		var in []float32
		if pos < len(inputs) {
			in = inputs[pos]
		}
		kv := matRows(wkv, in, width, H)
		score := matRows(wgate, in, width, H)
		pooled, emitted, err := pool.PushNormalized(pos, kv, score, normWeight, eps)
		if err != nil {
			return nil, v41StageErr(v41StageCompress, l, err)
		}
		if emitted {
			out = append(out, pooled)
		}
	}
	return out, nil
}

// v41IndexRows scores a projected index query against the compressed keys and
// returns the selected compressed row IDs for one layer. It returns nil when the
// layer declares no index source. Candidate blocks are selected when the layer is
// the declared candidate source, so the selection reads a blocked pool rather
// than the full compressed set.
func (m *Model) v41IndexRows(l int, qLat []float32, hidden []float32, keys [][]float32) ([]int32, error) {
	d41 := m.Cfg.DeepSeekV41
	if d41 == nil {
		return nil, nil
	}
	isSource := false
	for _, src := range d41.IndexSourceLayerIDs {
		if src == l {
			isSource = true
			break
		}
	}
	if !isSource {
		return nil, nil
	}
	cfg := m.Cfg
	nHeads, headDim := cfg.IndexNHeads, cfg.IndexHeadDim
	if nHeads <= 0 || headDim <= 0 {
		return nil, v41StageErr(v41StageIndexer, l,
			fmt.Errorf("%w: indexer geometry nHeads=%d headDim=%d is not declared", ErrV41ForwardStage, nHeads, headDim))
	}
	wqB := m.tensor(layerName(l, "indexer.wq_b.weight"))
	wk := m.tensor(layerName(l, "indexer.wk.weight"))
	kNorm := m.tensor(layerName(l, "indexer.k_norm.weight"))
	wProj := m.tensor(layerName(l, "indexer.weights_proj.weight"))
	compressLen := len(keys)
	q := matRows(wqB, qLat, nHeads*headDim, cfg.QLoraRank)
	flatKeys := make([]float32, 0, compressLen*headDim)
	for _, row := range keys {
		projected := matRows(wk, row, headDim, len(row))
		if len(kNorm) == headDim {
			projected = rmsnormCfg(projected, kNorm, float32(cfg.RMSNormEps), cfg)
		}
		flatKeys = append(flatKeys, projected...)
	}
	weights := matRows(wProj, hidden, nHeads, cfg.HiddenSize)
	for h := 0; h < nHeads; h++ {
		weights[h] *= cfg.attnScale() * float32(1.0/math.Sqrt(float64(nHeads)))
	}
	topKBlocks, blockSize := 0, 0
	if d41.CandidateSourceLayerID == l {
		topKBlocks, blockSize = d41.CandidateTopKBlocks, d41.CandidateBlockSize
	}
	pub, err := NewV41IndexerPublication(l, q, flatKeys, weights, nHeads, headDim, compressLen, topKBlocks, blockSize, cfg.IndexTopK, 0)
	if err != nil {
		return nil, v41StageErr(v41StageIndexer, l, err)
	}
	return pub.Rows(), nil
}

// v41KVSourceForwardAdmitted fails closed when the config declares a shared-KV
// source layer that lies WITHIN the model's decoder stack but whose shared state
// the reduced assembly cannot actually publish. As of #12896 the assembly DOES
// execute the shared KV/index schedule (v41_attention.go): a declared source
// pools its projected KV input through the CED/CSA2 compressor and publishes the
// rows, and a later reader resolves them. A source is therefore admitted only
// when it declares a compressed regime (ratio > 1, so the compressor runs) and
// carries the compressor tensors its pooling reads; a source at ratio <= 1 has
// no pooled stream to publish, so it is refused rather than silently running a
// per-layer KV cache where the official model reuses a source layer's state.
// Declarations that only touch out-of-range layers stay admitted, which keeps the
// reduced oracle fixture runnable: it derives from the published 40-layer config
// but narrows NumLayers to 1, so every kv source ({2,8,14,20}) is unreachable.
func (m *Model) v41KVSourceForwardAdmitted() error {
	d41 := m.Cfg.DeepSeekV41
	if d41 == nil {
		return nil
	}
	cfg := m.Cfg
	for _, layer := range d41.KVSourceLayerIDs {
		if layer < 0 || layer >= cfg.NumLayers {
			continue
		}
		if !v41AttentionRatioImplemented(v41CompressRatioAt(cfg, layer)) {
			return v41StageErr(v41StageCompress, layer,
				fmt.Errorf("%w: layer %d declares a shared-KV source but its compress ratio %d is not an implemented variant", ErrV41ForwardStage, layer, v41CompressRatioAt(cfg, layer)))
		}
		if v41CompressRatioAt(cfg, layer) <= 1 {
			return v41StageErr(v41StageAttention, layer,
				fmt.Errorf("%w: layer %d declares a shared-KV source but has no compressed regime to publish", ErrV41ForwardStage, layer))
		}
		H := cfg.HiddenSize
		width := v41CompressorWidth(cfg)
		if err := m.v41AdmitShape(layerName(layer, "attn.compressor.wkv.weight"), v41StageCompress, layer, width, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "attn.compressor.wgate.weight"), v41StageCompress, layer, width, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "attn.compressor.norm.weight"), v41StageCompress, layer, width); err != nil {
			return err
		}
	}
	return nil
}

// v41CandidateSourceForwardAdmitted fails closed when the config declares a
// candidate-source layer that lies WITHIN the model's decoder stack. The reduced
// text assembly runs its own full per-layer attention contraction and never
// executes the CED/CSA2 blocked-candidate selection the lightning indexer reads
// (the reference's candidate_source_layer_id / candidate_topk_blocks /
// candidate_block_size schedule), so silently running an in-range candidate source
// would emit logits from an unblocked attention path where the official model
// selects a sparse candidate pool first. Declarations that only touch out-of-range
// layers stay admitted, which keeps the reduced oracle fixture runnable: it derives
// from the published 40-layer config but narrows NumLayers to 1, so the candidate
// source (20) is unreachable. Executing the real candidate selection is the
// CED/CSA2 attention leaf's remaining work; until then this is the fail-closed
// boundary.
func v41CandidateSourceForwardAdmitted(cfg Config) error {
	m := cfg.DeepSeekV41
	if m == nil {
		return nil
	}
	layer := m.CandidateSourceLayerID
	if layer >= 0 && layer < cfg.NumLayers {
		return v41StageErr(v41StageIndexer, layer,
			fmt.Errorf("%w: layer %d declares a candidate source but the reduced forward does not execute the CED/CSA2 blocked-candidate selection", ErrV41ForwardStage, layer))
	}
	return nil
}

// v41HCMultForwardAdmitted fails closed when the config declares an mHC
// hyperconnection multiplicity other than 4. The reduced text assembly hardcodes
// the four-stream geometry (v41MHCSplit called with hc=4, four identical stand-in
// streams, and the width-24 mix projection v41MHCMixWidth), so it executes only
// the published hc_mult=4 layout. A config declaring a different multiplicity
// would otherwise run the four-stream hyperconnection for a model the assembly
// never ran -- a silent omission contrary to the fail-closed invariant. The
// published artifact and every reduced fixture declare hc_mult=4, so the reduced
// oracle forward stays admitted; executing a non-4 multiplicity is the mHC
// geometry leaf's remaining work.
func v41HCMultForwardAdmitted(cfg Config) error {
	m := cfg.DeepSeekV41
	if m == nil {
		return nil
	}
	if m.HCMult != 4 {
		return v41StageErr(v41StageMHC, -1,
			fmt.Errorf("%w: config declares mHC multiplicity %d but the reduced forward only executes the four-stream hc_mult=4 geometry", ErrV41ForwardStage, m.HCMult))
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
	// Geometry discrimination fails closed: a parsed/published config whose flat
	// axes disagree with its Attention envelope is refused here rather than
	// silently assembled against the reduced stand-in geometry.
	if _, err := v41ForwardGeometry(cfg); err != nil {
		return err
	}
	// kvLatentRank is resolved through the same fail-closed discriminator, so an
	// inconsistent published envelope can never be admitted at the reduced shape.
	kvLatentRank, err := v41ForwardKVLatentRank(cfg)
	if err != nil {
		return err
	}
	if err := m.v41AdmitShape("model.embed_tokens.weight", v41StageEmbedding, -1, cfg.VocabSize, cfg.HiddenSize); err != nil {
		return err
	}
	if err := m.v41AdmitShape("model.norm.weight", v41StageFinalNorm, -1, cfg.HiddenSize); err != nil {
		return err
	}
	if !m.hasWeight("lm_head.weight") && !m.hasWeight("model.embed_tokens.weight") {
		return v41StageErr(v41StageHead, -1,
			fmt.Errorf("%w: no lm_head.weight and no tied embedding", ErrV41ForwardStage))
	}
	if err := m.v41AdmitShape("lm_head.weight", v41StageHead, -1, cfg.VocabSize, cfg.HiddenSize); err != nil {
		return err
	}
	if _, err := v41RouterConfigFullGeometry(cfg); err != nil {
		return err
	}
	if err := m.v41EngramForwardAdmitted(); err != nil {
		return err
	}
	if err := m.v41CompressIndexForwardAdmitted(); err != nil {
		return err
	}
	if err := m.v41KVSourceForwardAdmitted(); err != nil {
		return err
	}
	if err := v41CandidateSourceForwardAdmitted(cfg); err != nil {
		return err
	}
	if err := v41HCMultForwardAdmitted(cfg); err != nil {
		return err
	}
	if err := v41AttentionGeometryForwardAdmitted(cfg); err != nil {
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
		if err := m.v41AdmitMHC(l); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wq_a.weight"), v41StageAttention, l, cfg.QLoraRank, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wq_b.weight"), v41StageAttention, l, qHeadDim, cfg.QLoraRank); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wkv.weight"), v41StageAttention, l, kvLatentRank, H); err != nil {
			return err
		}
		// wo_a is the leaf's group-major [Groups, OLoRARank, HeadsPerGroup*HeadDim]
		// tensor, so its total is OLoRARank*NumHeads*HeadDim; wo_b is
		// [H, Groups*OLoRARank]. The artifact stores the group-major form while
		// the reduced fixture declares the equivalent flat form, so admit either
		// declaration (see v41AdmitGroupedWoA; the #13264 seam).
		if err := m.v41AdmitGroupedWoA(l); err != nil {
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

// v41AdmitMHC admits a layer's mHC coefficient block in either the reduced
// fixture's legacy [mixWidth, H] geometry or the published artifact's flattened
// four-stream geometry. The reference (inference/model.py mHC) projects the
// width-4H flattened residual through hc_attn_fn; the staged vcruz Q2_K artifact
// stores that projection as [4H, 24] (input-major), while the forward consumes it
// logically as [24, 4H] (coefficient-major). Both the logical [mixWidth, 4H]
// orientation and the stored [4H, mixWidth] transpose are admitted here so a real
// artifact load reaches the forward instead of refusing at admission; every other
// shape (including the reduced [mixWidth, H]) still falls through to the named
// two-axis shape guard and fails closed. The base/scale vectors are unchanged.
func (m *Model) v41AdmitMHC(l int) error {
	H := m.Cfg.HiddenSize
	name := layerName(l, "mhc.mixes.weight")
	out, in, ok := m.residentShape(name)
	if !ok {
		return v41StageErr(v41StageMHC, l, fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
	}
	switch {
	case out == v41MHCMixWidth && (in == H || in == 4*H):
		// logical [mixWidth, in]
	case in == v41MHCMixWidth && out == 4*H:
		// stored artifact transpose [4H, mixWidth]
	default:
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: tensor %s shape [%d %d], want [%d %d], [%d %d] or [%d %d]",
				ErrV41ForwardStage, name, out, in, v41MHCMixWidth, H, v41MHCMixWidth, 4*H, 4*H, v41MHCMixWidth))
	}
	if err := m.v41AdmitShape(layerName(l, "mhc.base"), v41StageMHC, l, v41MHCMixWidth); err != nil {
		return err
	}
	if err := m.v41AdmitShape(layerName(l, "mhc.scale"), v41StageMHC, l, 3); err != nil {
		return err
	}
	return nil
}

// v41AdmitGroupedWoA admits a layer's attn.wo_a.weight in either declaration of
// the same weight. The forward consumes it through V41GroupedOutputProjection,
// which reads the group-major [Groups, OLoRARank, HeadsPerGroup*HeadDim] tensor
// the published artifact's attn_output_a.weight stores. The reduced fixture
// declares the equivalent flat [OLoraRank, NumHeads*HeadDim] two-axis form.
// The two forms carry the SAME element count
// (OGroups*OLoraRank*HeadsPerGroup*HeadDim == OLoraRank*NumHeads*HeadDim) but
// assign different numbers to the two axes, so a single v41AdmitShape row cannot
// admit both: demanding the flat form refused the real artifact by name before
// the grouped projection ever ran (the #13264 seam), and demanding the grouped
// form would refuse the reduced fixture.
//
// Both declarations are admitted here. Any other shape - including a grouped
// form whose axes are individually consistent but whose total disagrees, and the
// artifact's [OGroups*OLoraRank, HeadsPerGroup*HeadDim] with a non-divisible
// head count - falls through to the named two-axis shape guard and fails closed,
// so the no-silent-mis-shape property is retained. Presence is resolved
// residency-completely by residentShape, so a resident-store wo_a is admitted at
// its real geometry on either arm.
func (m *Model) v41AdmitGroupedWoA(l int) error {
	name := layerName(l, "attn.wo_a.weight")
	cfg := m.Cfg
	out, in, ok := m.residentShape(name)
	if !ok {
		return v41StageErr(v41StageAttention, l, fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
	}
	flatRows, flatCols := cfg.OLoraRank, cfg.NumHeads*cfg.HeadDim
	groupedRows := cfg.OGroups * cfg.OLoraRank
	groupedHeadsPerGroup := 0
	if cfg.OGroups > 0 && cfg.NumHeads%cfg.OGroups == 0 {
		groupedHeadsPerGroup = cfg.NumHeads / cfg.OGroups
	}
	groupedCols := groupedHeadsPerGroup * cfg.HeadDim
	switch {
	case out == flatRows && in == flatCols:
		return nil
	case groupedHeadsPerGroup > 0 && out == groupedRows && in == groupedCols:
		return nil
	default:
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: tensor %s shape [%d %d], want [%d %d] (flat) or [%d %d] (grouped)",
				ErrV41ForwardStage, name, out, in, flatRows, flatCols, groupedRows, groupedCols))
	}
}

// v41MHCWeightLayout reports how an admitted mhc.mixes.weight is laid out:
// flat=true for the published flattened-four-stream projection (24 x 4H logical,
// or the artifact's 4H x 24 storage transpose), flat=false for the reduced
// fixture's legacy 24 x H single-stream matmul. ok=false means the weight is
// absent or holds neither admitted geometry, so the caller fails closed rather
// than reading a wrong sub-matrix.
func (m *Model) v41MHCWeightLayout(l int) (flat, transposed, ok bool) {
	out, in, present := m.residentShape(layerName(l, "mhc.mixes.weight"))
	if !present {
		return false, false, false
	}
	switch {
	case out == v41MHCMixWidth && in == 4*m.Cfg.HiddenSize:
		return true, false, true // logical [24, 4H]
	case out == 4*m.Cfg.HiddenSize && in == v41MHCMixWidth:
		return true, true, true // stored artifact transpose [4H, 24]
	case out == v41MHCMixWidth && in == m.Cfg.HiddenSize:
		return false, false, true // reduced legacy [24, H]
	default:
		return false, false, false
	}
}

// v41MHCMixF32 reads a layer's mhc.mixes.weight as a full f32 block, resolved
// residency-completely. The reduced fixture carries it in the f32 manifest, and
// that path returns the manifest view unchanged (byte-identical to the pre-#13062
// read); a real quantized serve keeps the Q2_K block in a resident store instead,
// so the f32 manifest view would panic in m.tensor. Reading the resident store
// here is what makes the flattened-four-stream forward reachable for the pinned
// artifact. The returned buffer is in the STORE'S native row-major layout, so a
// caller must consult v41MHCWeightLayout to interpret [24,4H] vs the stored
// [4H,24] transpose. A weight absent from every store fails closed with a typed
// error naming the tensor rather than panicking.
func (m *Model) v41MHCMixF32(l int) ([]float32, error) {
	name := layerName(l, "mhc.mixes.weight")
	if m.has(name) {
		return m.tensor(name), nil
	}
	if qt := m.kqw[name]; qt != nil {
		qt.ensureRawCPU("mHC mix read")
		bb := qt.kind.blockBytes()
		rowBytes := qt.rowBytes()
		w := make([]float32, qt.out*qt.in)
		for b := 0; b < qt.out; b++ {
			row := qt.raw[b*rowBytes : (b+1)*rowBytes]
			for j := 0; j < qt.nblk; j++ {
				kQuantDequantSuperBlock(w[b*qt.in+j*qt.kind.blockWeights():], row[j*bb:(j+1)*bb], qt.kind)
			}
		}
		return w, nil
	}
	if qt := m.q2w[name]; qt != nil {
		return dequantQ2Tensor(qt), nil
	}
	if qt := m.q8w[name]; qt != nil {
		return dequantQ8Tensor(qt), nil
	}
	if qt := m.q4kw[name]; qt != nil {
		raw, err := qt.materializeRaw()
		if err != nil {
			return nil, fmt.Errorf("%w: tensor %s Q4_K materialization failed: %v", ErrV41ForwardStage, name, err)
		}
		rowBytes := qt.nblk * q4kBlockBytes
		w := make([]float32, qt.out*qt.in)
		for o := 0; o < qt.out; o++ {
			row := raw[o*rowBytes : (o+1)*rowBytes]
			for j := 0; j < qt.nblk; j++ {
				q4kDequantSuperBlock(w[o*qt.in+j*qkK:], row[j*q4kBlockBytes:(j+1)*q4kBlockBytes])
			}
		}
		return w, nil
	}
	if qt := m.q4w[name]; qt != nil {
		w := make([]float32, qt.out*qt.in)
		for o := 0; o < qt.out; o++ {
			for b := 0; b < qt.nblk; b++ {
				dequantQ4Block(w[o*qt.in+b*qBlk4:], qt.d[o*qt.nblk+b], qt.q[o*qt.nblk*(qBlk4/2)+b*(qBlk4/2):])
			}
		}
		return w, nil
	}
	if qt := m.gptqw[name]; qt != nil {
		w := make([]float32, qt.out*qt.in)
		for o := 0; o < qt.out; o++ {
			gptqDequantRow(w[o*qt.in:], qt, o)
		}
		return w, nil
	}
	return nil, v41StageErr(v41StageMHC, l,
		fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
}

// v41MHCProjectFull executes the published flattened-four-stream mHC mix
// projection: the four width-H streams are laid end to end into xflat (4H), one
// shared RMS scale 1/sqrt(mean(xflat^2)+eps) is computed over the whole flattened
// residual, and the 24 mix coefficients are the linear projection of xflat scaled
// by that rsqrt. It mirrors the pinned reference oracle
// (v4_flash_oracle_test.go oracleV4FlashMHCProjection). transposed selects the
// admitted storage orientation: false reads the logical row-major [24, 4H]
// (mixes[m] = sum_i w[m*4H+i]*xflat[i]); true reads the artifact's stored
// input-major [4H, 24] transpose (mixes[m] = sum_i w[i*24+m]*xflat[i]).
//
// It validates its own inputs so a wrong-width or wrong-stream-count residual
// cannot masquerade as the flattened geometry and be silently mis-projected.
func v41MHCProjectFull(wMix []float32, streams [][]float32, H int, eps float32, transposed bool) ([]float32, error) {
	flatWidth := 4 * H
	if len(streams) != 4 || H <= 0 {
		return nil, fmt.Errorf("model: V41 full mHC projection wants 4 streams of width %d, got %d", H, len(streams))
	}
	for i, s := range streams {
		if len(s) != H {
			return nil, fmt.Errorf("model: V41 full mHC projection stream %d width %d, want %d", i, len(s), H)
		}
	}
	if len(wMix) != v41MHCMixWidth*flatWidth {
		return nil, fmt.Errorf("model: V41 full mHC projection weight has %d values, want %d", len(wMix), v41MHCMixWidth*flatWidth)
	}
	xflat := make([]float32, 0, flatWidth)
	for _, s := range streams {
		xflat = append(xflat, s...)
	}
	var ss float32
	for _, v := range xflat {
		ss += v * v
	}
	rsqrt := float32(1 / math.Sqrt(float64(ss/float32(len(xflat))+eps)))
	mixes := make([]float32, v41MHCMixWidth)
	for m := 0; m < v41MHCMixWidth; m++ {
		var s float32
		if transposed {
			for i := 0; i < flatWidth; i++ {
				s += wMix[i*v41MHCMixWidth+m] * xflat[i]
			}
		} else {
			row := wMix[m*flatWidth : (m+1)*flatWidth]
			for i := 0; i < flatWidth; i++ {
				s += row[i] * xflat[i]
			}
		}
		mixes[m] = s * rsqrt
	}
	return mixes, nil
}

// v41AdmitShape asserts a named tensor is present with the expected shape.
// Presence + shape are resolved residency-completely (residentShape), so a
// weight that a quantized serve keeps in a resident store (kqw/q4kw/q8w/...)
// rather than the f32 manifest is admitted at its real [out, in] geometry
// instead of being refused as "missing". The fail-closed property is retained:
// a tensor absent from every store still refuses by name, and a resident tensor
// whose geometry disagrees with the expected shape still refuses.
func (m *Model) v41AdmitShape(name string, stage v41ForwardStage, layer int, want ...int) error {
	if len(want) == 2 {
		out, in, ok := m.residentShape(name)
		if !ok {
			// A routed expert the R5 streamed-experts tier holds is by design
			// ABSENT from every resident store (manifest/q8w/q4w/q4kw/kqw/q2w/
			// gptqw), because the whole point of the tier is to fault one expert's
			// stride out of a fused checkpoint slab only when it is routed. So a
			// resident-store miss falls through to the tier's index, a
			// presence-only check (no IO, no fault), before refusing by name.
			//
			// The descriptor geometry is deliberately NOT re-checked here: the tier
			// entry carries the fused tensor's declared rows/cols, but the
			// per-expert shape is the loader's already-validated [out,in] declaration
			// (FusedExpertTensor.Rows/Cols), and the forward's own read of the
			// projection is the authority on shape use. A tier that carries the name
			// at a disagreeing geometry is therefore admitted at presence, and any
			// real disagreement surfaces at the forward's shape use rather than
			// being silently accepted as a different tensor.
			if m.expertCheckpoint.Has(name) {
				return nil
			}
			return v41StageErr(stage, layer, fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
		}
		if out != want[0] || in != want[1] {
			return v41StageErr(stage, layer,
				fmt.Errorf("%w: tensor %s shape [%d %d], want %v", ErrV41ForwardStage, name, out, in, want))
		}
		return nil
	}
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

	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return nil, err
	}

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

	// Persistent mHC streams, one four-stream set per position. On the full path
	// stream 0 carries the live hidden state and streams 1..3 are the reference's
	// persistent residual streams initialized to zero -- DISTINCT from stream 0,
	// not the reduced stand-in's four identical copies. Both paths carry the same
	// [][][]float32 shape; only the initialization and mixing differ, so the
	// reduced arithmetic is byte-identical to the pre-#13009 assembly.
	streams := make([][][]float32, len(seq))
	for t := range streams {
		set := make([][]float32, 4)
		set[0] = x[t]
		for h := 1; h < 4; h++ {
			set[h] = make([]float32, H)
		}
		streams[t] = set
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
	routeCfg, err := v41RouterConfigFullGeometry(cfg)
	if err != nil {
		return nil, err
	}

	act := &Activations{Seq: len(seq), Hidden: [][]float32{flatten(x)}}
	for l := 0; l < cfg.NumLayers; l++ {
		if err := m.v41Layer(l, seq, x, streams, full, hd, nH, H, eps, hcIters, hcEps, routeCfg, st); err != nil {
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

// v41Layer applies one V4.1 decoder layer to x in place, updating the persistent
// mHC streams[t] for each position. tokens carries the ids for the positions in
// x so a declared Engram layer can hash them. The reduced path reproduces the
// pre-#13009 arithmetic exactly (four identical stand-in streams derived from the
// normalized input, and stream 0 of the post-mix written back); the full path
// reads and writes all four DISTINCT persistent streams.
func (m *Model) v41Layer(l int, tokens []int, x [][]float32, streams [][][]float32, full bool, hd, nH, H int, eps float32, hcIters int, hcEps float32, routeCfg v41RouterConfig, st *v41ForwardState) error {
	cfg := m.Cfg

	// Engram injection happens at the START of the layer, into the residual,
	// before attention and before attn_norm (ds41_graph_before_attention).
	if cfg.DeepSeekV41 != nil {
		for _, eng := range cfg.DeepSeekV41.EngramLayerIDs {
			if eng == l {
				if err := m.v41EngramInject(l, x, tokens, eps); err != nil {
					return err
				}
				break
			}
		}
	}

	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	ffnNorm := m.tensor(layerName(l, "ffn_norm.weight"))
	wMix, err := m.v41MHCMixF32(l)
	if err != nil {
		return err
	}
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
	//
	// The mHC mix projection geometry is per-layer (it is a per-layer weight), so
	// resolve it once. The published artifact's hc_attn_fn is the flattened
	// four-stream projection (logical 24 x 4H, stored [4H, 24]); the reference
	// (inference/model.py mHC; v4_flash_oracle_test.go:168) computes it over the
	// four width-H streams laid end to end with a single shared flatten-RMS. The
	// reduced fixture uses the legacy 24 x H single-stream matmul. Both are
	// executed here; only the reduced path's arithmetic is held byte-identical to
	// pre-#13009.
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK {
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: mHC mix weight holds no admitted geometry", ErrV41ForwardStage))
	}
	hcByPos := make([]v41MHCMix, seq)
	preByPos := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		// The reduced path projects the normalized input through the legacy
		// single-stream [24, H] block; the full path projects the four DISTINCT
		// persistent streams through the flattened 4H residual with one shared RMS.
		// xn is retained for the reduced pre-collapse stand-in below either way.
		xn := rmsnormCfg(x[t], attnNorm, eps, cfg)
		var mixes []float32
		if mhcFlat {
			var err error
			if mixes, err = v41MHCProjectFull(wMix, streams[t], H, eps, mhcTransposed); err != nil {
				return v41StageErr(v41StageMHC, l, err)
			}
		} else {
			mixes = matRows(wMix, xn, v41MHCMixWidth, H)
		}
		mix, err := v41MHCSplit(mixes, mixScale, mixBase, 4, hcIters, hcEps)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		hcByPos[t] = mix
		// The mHC pre-collapse always reads a four-stream set collapsed by the
		// learned pre coefficients. The reduced path reconstructs the reference's
		// stand-in (four identical copies of the normalized input, byte-identical
		// to pre-#13009) rather than reading the persistent set, so its numerics
		// are unchanged. The full path reads the four DISTINCT persistent streams
		// carried into this layer (stream 0 = live hidden, 1..3 = residual).
		streams4 := streams[t]
		if !full {
			streams4 = [][]float32{xn, xn, xn, xn}
		}
		collapsed, err := v41MHCPre(streams4, mix.pre)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		preByPos[t] = collapsed
	}

	// ---- attention: projected q/kv per position, then sparse sink per position ----
	qHeads := make([][]float32, seq) // [t][nH*hd], rotated
	kvRows := make([][]float32, seq) // [t][hd], rotated (single KV head)
	qLatRows := make([][]float32, seq)
	// RoPE geometry, fail-closed. The reference rotates only the LAST
	// qk_rope_head_dim components of each head (the leading qk_nope_head_dim pass
	// through byte-identical) using the interleaved adjacent-pair convention. A
	// head width that cannot carry that slice (a non-positive, odd, or
	// head-wider rope dim) is refused with a typed error rather than silently
	// rotating a wrong sub-vector.
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	for t := 0; t < seq; t++ {
		c := preByPos[t]
		qLat := matRows(wQA, c, cfg.QLoraRank, H)
		q := matRows(wQB, qLat, nH*hd, cfg.QLoraRank)
		// KV latent seam. The full path admits and projects attn.wkv at the
		// published latent rank (v41KVLoraRank = 512); the attention contraction
		// below consumes a per-position row of width hd (head_dim), so the full
		// projection is taken at the published latent rank and sliced to the first
		// hd columns for the sink contraction. On a valid full config
		// hd == v41KVLoraRank == 512, so the slice is a no-op; an hd wider than the
		// latent rank is refused rather than silently mis-read. The reduced fixture
		// keeps its HeadDim-wide projection byte-for-byte.
		var kv []float32
		if full {
			if hd > v41KVLoraRank {
				return v41StageErr(v41StageAttention, l,
					fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
			}
			kv = matRows(wKV, c, v41KVLoraRank, H)[:hd]
		} else {
			kv = matRows(wKV, c, hd, H)
		}
		cos, sin := v41RopeTableForLayer(cfg, l, t)
		for h := 0; h < nH; h++ {
			applyRopeTailInterleaved(q[h*hd:(h+1)*hd], cos, sin, ropeDim)
		}
		applyRopeTailInterleaved(kv, cos, sin, ropeDim)
		qHeads[t] = q
		kvRows[t] = kv
		qLatRows[t] = qLat
	}

	// ---- CED/CSA2 compressor + lightning indexer stages (#13006, #12896) ----
	//
	// A layer declaring CompressRatios[l] > 1 pools its per-position projected KV
	// rows through the CED/CSA2 compressor (v41CompressedRows), and a layer
	// declaring an in-range index source scores its projected index query against
	// those compressed keys and selects rows (v41IndexRows). Both stages execute
	// here and fail closed with a typed *V41ForwardError on malformed geometry; a
	// config that declares an in-range stage without its weights is refused at
	// admission, not here.
	//
	// #12896 maps the sink contraction onto the reference's compressed KV cache:
	// a compressed layer contracts the COMPRESSED stream directly (block-causal
	// visibility, one pooled key/value row per group) instead of expanding the
	// pooled latent back onto causal positions. A declared shared-KV source layer
	// publishes its compressed rows into the session state, and a later reader
	// layer resolves its KV stream from that published source.
	plan, err := v41AttentionPlanFor(cfg, l, m.v41AttentionRolesCached())
	if err != nil {
		return err
	}
	var compressedKV [][]float32
	if plan.Ratio > 1 {
		compressed, err := m.v41CompressedRows(l, plan.Ratio, kvRows, preByPos)
		if err != nil {
			return err
		}
		compressedKV = compressed
	}
	// A reader layer whose KV source precedes it consumes the source's published
	// compressed stream; a source layer publishes its own for later readers.
	sharedKV := compressedKV
	if plan.Role == V41AttentionRoleReader && plan.KVSourceLayer >= 0 && st != nil {
		attn, err := st.attentionState(hd, 8)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		if rows, ok := attn.KVSourceRows(plan.KVSourceLayer); ok && len(rows) > 0 {
			sharedKV = rows
		}
	}
	// The layer's own lightning-index selection for the newest position. A layer
	// that is not an index source computes none (nil).
	localIdx, err := m.v41IndexRows(l, qLatRows[seq-1], preByPos[seq-1], compressedKV)
	if err != nil {
		return err
	}
	// The per-position index list the compressed contraction consumes, flattened
	// [seq][TopKWidth]. A layer that is itself the index source uses its local
	// selection replicated across the causal positions it published it for; a
	// reader layer reuses its source's published top-k selection; a layer with
	// neither passes no list and the contraction falls back to the causal mask.
	indexList, err := m.v41AttentionIndexList(plan, st, localIdx, hd, len(sharedKV), seq)
	if err != nil {
		return err
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
		// A declared source layer publishes its compressed rows and index keys so
		// later readers can resolve them within this same forward pass (the
		// V41AttentionState source-then-consumer ordering).
		updates := m.v41AttentionSourceUpdates(plan, compressedKV, qLatRows)
		if len(seed) > 0 {
			if err := attn.Prefill(seed, updates); err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
		} else if len(updates) > 0 {
			// A source layer with no projected window rows still publishes.
			if err := attn.PublishCandidates(plan.Ratio, nil); err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
		}
		// An index source publishes its own per-position top-k selection so a
		// later reader layer can reuse it without recomputing the scoring path.
		// The selection is the source's local index list, one row per published
		// query position.
		if indexSourceAt(cfg.DeepSeekV41, l) && indexList != nil && plan.TopKWidth > 0 {
			rows := make([][]int32, seq)
			for t := range rows {
				rows[t] = append([]int32(nil), indexList[t*plan.TopKWidth:(t+1)*plan.TopKWidth]...)
			}
			if err := attn.PublishTopK(plan.Ratio, rows); err != nil {
				return v41StageErr(v41StageIndexer, l, err)
			}
		}
	}

	scale := cfg.attnScale()
	attnOut := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		// A compressed/shared layer contracts the COMPRESSED KV stream directly:
		// block-causal visibility over pooled group rows, with the lightning
		// indexer's row selection when the layer published one. A per-layer layer
		// keeps the exact per-position causal sink contraction.
		if plan.Ratio > 1 || (plan.Role == V41AttentionRoleReader && len(sharedKV) > 0 && len(sharedKV) < seq) {
			opt := V41AttentionSharedKVOptions{
				Layer: l, Ratio: maxInt(plan.Ratio, 1), Groups: len(sharedKV),
				HeadDim: hd, Heads: nH, Softmax: scale, Sink: sink,
			}
			if indexList != nil && len(indexList) >= (t+1)*plan.topKWidth() {
				// The index source publishes one selection row per query position.
				opt.Idx = indexList[t*plan.topKWidth() : (t+1)*plan.topKWidth()]
				opt.IndexTopK = plan.topKWidth()
				opt.TopK = plan.topKWidth()
			}
			o, err := V41AttentionCompressedForward(qHeads[t], sharedKV, opt)
			if err != nil {
				return err
			}
			projected, err := V41GroupedOutputProjection(o, woA, woB, 1, 1, nH, hd, cfg.OGroups, cfg.OLoraRank, H)
			if err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
			attnOut[t] = projected
			continue
		}
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
		// The post-mix always reads a four-stream residual set. The reduced path
		// reconstructs the stand-in (four identical copies of the collapsed pre
		// vector) so its arithmetic is unchanged; the full path mixes the four
		// distinct persistent streams.
		residual := [][]float32{preByPos[t], preByPos[t], preByPos[t], preByPos[t]}
		if full {
			residual = streams[t]
		}
		next, err := v41MHCPost(delta, residual, hcByPos[t].post, hcByPos[t].comb)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		if full {
			// Write ALL FOUR post-mix streams back into the persistent set so the
			// next layer reads the updated state; stream 0 is the live hidden.
			for h := 0; h < 4; h++ {
				copy(streams[t][h], next[h])
			}
			copy(x[t], next[0])
		} else {
			// Reduced path: only stream 0 is propagated, exactly as before.
			copy(x[t], next[0])
		}
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
	if !m.hasWeight("lm_head.weight") && !m.hasWeight("model.embed_tokens.weight") {
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
