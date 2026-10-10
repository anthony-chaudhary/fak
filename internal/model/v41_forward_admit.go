package model

import (
	"errors"
	"fmt"
	"math"
)

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
		_, _, normCells, err := m.v41EngramProjectionDimensions(layer, stage)
		if err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "engram_q_norm.weight"), v41StageEngram, layer, normCells); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "engram_k_norm.weight"), v41StageEngram, layer, normCells); err != nil {
			return err
		}
	}
	return nil
}

// v41CompressIndexForwardAdmitted checks only tensors a layer owns: compressed
// producers need compressor weights, every index-query source needs its query
// and head-weight projections, and only index-key producers need wk/k_norm.
// Shared readers resolve already-published source rows. Ratio-one assembly stays
// closed even though its standalone compressor helper is implemented.
//
// Two malformed-schedule arms are also refused here: a negative ratio (invalid
// geometry, never a compressed layer) and a schedule shorter than the decoder
// stack (an uncovered layer with no declared regime). Both must fail closed, not
// fall through to the uncompressed regime.
func (m *Model) v41CompressIndexForwardAdmitted() error {
	d41 := m.Cfg.DeepSeekV41
	if d41 == nil {
		// The canonical GGUF identity may retain only the flat schedule before
		// optional metadata attachment. That absence cannot turn active ratio one
		// into window-only execution. Preserve all other nil-metadata behavior.
		for layer, ratio := range m.Cfg.CompressRatios {
			if layer >= m.Cfg.NumLayers {
				break
			}
			if ratio == 1 {
				return v41StageErr(v41StageCompress, layer,
					fmt.Errorf("%w: layer %d declares ratio-one compression but the ratio-one forward assembly is not implemented", ErrV41ForwardStage, layer))
			}
		}
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
	roles := v41AttentionRoles(cfg)
	for layer := 0; layer < cfg.NumLayers; layer++ {
		// Only ratio 0 is window-only. Ratio 1 requires the reference's separate
		// compressed projection/cache plus window contraction; its standalone
		// helper does not yet implement that forward assembly. Refuse it regardless
		// of source declarations or weight presence instead of executing plain KV.
		ratio := d41.CompressRatios[layer]
		switch {
		case ratio < 0:
			return v41StageErr(v41StageCompress, layer,
				fmt.Errorf("%w: layer %d declares malformed compressor ratio %d", ErrV41ForwardStage, layer, ratio))
		case ratio == 1:
			return v41StageErr(v41StageCompress, layer,
				fmt.Errorf("%w: layer %d declares ratio-one compression but the ratio-one forward assembly is not implemented", ErrV41ForwardStage, layer))
		case ratio > 1:
			if roles[layer] == V41AttentionRoleReader {
				continue // readers consume the source cache and own no compressor tensors
			}
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
		if d41.CompressRatios[layer] == 0 {
			continue // window-only layers do not execute a declared indexer
		}
		if indexHeads <= 0 || indexDim <= 0 {
			return v41StageErr(v41StageIndexer, layer,
				fmt.Errorf("%w: layer %d declares a lightning-indexer source but the indexer geometry (nHeads=%d headDim=%d) is not declared", ErrV41ForwardStage, layer, indexHeads, indexDim))
		}
		if rd := cfg.QKRopeHeadDim; rd <= 0 || rd%2 != 0 || rd > indexDim {
			return v41StageErr(v41StageIndexer, layer,
				fmt.Errorf("%w: index rotary width %d must be positive, even and <= index head width %d", ErrV41ForwardStage, rd, indexDim))
		}
		wq := indexHeads * indexDim
		if err := m.v41AdmitShape(layerName(layer, "indexer.wq_b.weight"), v41StageIndexer, layer, wq, cfg.QLoraRank); err != nil {
			return err
		}
		plan, err := v41AttentionPlanFor(cfg, layer, roles)
		if err != nil {
			return err
		}
		if plan.Role == V41AttentionRoleReader {
			if !indexSourceAt(d41, plan.KVSourceLayer) {
				return v41StageErr(v41StageIndexer, layer,
					fmt.Errorf("%w: index-only layer %d requires keys owned by KV source %d", ErrV41ForwardStage, layer, plan.KVSourceLayer))
			}
		} else {
			if err := m.v41AdmitShape(layerName(layer, "indexer.wk.weight"), v41StageIndexer, layer, indexDim, v41CompressorWidth(cfg)); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(layer, "indexer.k_norm.weight"), v41StageIndexer, layer, indexDim); err != nil {
				return err
			}
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
// through the CED/CSA2 compressor. Ratio one projects each carrier through wkv
// and the existing BF16/learned-RMSNorm tail, with no gate or pooling. Ratio zero
// returns the ordinary KV rows unchanged. This helper does not admit a shared
// ratio-one source to the forward assembly; its ownership and cache integration
// remain fenced by v41KVSourceForwardAdmitted.
func (m *Model) v41CompressedRows(l int, ratio int, kvRows [][]float32, inputs [][]float32) ([][]float32, error) {
	return m.v41CompressedRowsWithProjection(l, ratio, kvRows, inputs, nil)
}

func (m *Model) v41CompressedRowsWithProjection(l int, ratio int, kvRows [][]float32, inputs [][]float32, project v41DenseProjectionFunc) ([][]float32, error) {
	return m.v41CompressedRowsWithOperations(l, ratio, kvRows, inputs, project, nil)
}

func (m *Model) v41CompressedRowsWithOperations(l int, ratio int, kvRows [][]float32, inputs [][]float32, project v41DenseProjectionFunc, normalize v41CompressorNormFunc) ([][]float32, error) {
	if ratio <= 0 {
		return kvRows, nil
	}
	cfg := m.Cfg
	width := v41CompressorWidth(cfg)
	H := cfg.HiddenSize
	normWeight := m.tensor(layerName(l, "attn.compressor.norm.weight"))
	eps := float32(cfg.RMSNormEps)
	pool, err := NewV41CompressorPool(ratio, width)
	if err != nil {
		return nil, v41StageErr(v41StageCompress, l, err)
	}
	if normalize != nil {
		pool.normalize = func(pooled, gain []float32, eps float32) ([]float32, error) {
			return normalize(l, pooled, gain, eps)
		}
	}
	var out [][]float32
	for pos := range inputs {
		var in []float32
		if pos < len(inputs) {
			in = inputs[pos]
		}
		kv, err := m.v41ProjMatRowsWithProjection(l, "attn.compressor.wkv.weight", in, width, H, project)
		if err != nil {
			return nil, err
		}
		var score []float32
		if ratio > 1 {
			score, err = m.v41ProjMatRowsWithProjection(l, "attn.compressor.wgate.weight", in, width, H, project)
			if err != nil {
				return nil, err
			}
		}
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
func (m *Model) v41IndexRows(l, pos int, qLat []float32, hidden []float32, keys [][]float32) ([]int32, error) {
	if !indexSourceAt(m.Cfg.DeepSeekV41, l) {
		return nil, nil
	}
	projected, err := m.v41IndexKeys(l, keys, nil)
	if err != nil {
		return nil, err
	}
	ratio := v41CompressRatioAt(m.Cfg, l)
	for group, key := range projected {
		if err := m.v41IndexRoPE(l, group*ratio, key, 1); err != nil {
			return nil, err
		}
	}
	return m.v41IndexRowsProjected(l, pos, qLat, hidden, projected, nil)
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

// v41RopeForwardAdmitted validates the same per-layer base the table consumes.
// Nonzero compression ratios, including ratio 1, require their declared
// compressed base; substituting the plain base would execute a different model.
// Only active decoder layers participate, so auxiliary schedule entries and
// metadata inherited by a deliberately narrowed fixture remain untouched.
func v41RopeForwardAdmitted(cfg Config) error {
	for layer := 0; layer < cfg.NumLayers; layer++ {
		theta := v41RopeThetaForLayer(cfg, layer)
		if theta > 0 && !math.IsNaN(theta) && !math.IsInf(theta, 0) {
			continue
		}
		key := "rope_theta"
		if v41CompressRatioAt(cfg, layer) > 0 {
			key = "compress_rope_theta"
		}
		return v41StageErr(v41StageAttention, layer,
			fmt.Errorf("%w: layer %d requires finite positive %s, got %g", ErrV41ForwardStage, layer, key, theta))
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
	if err := v41RopeForwardAdmitted(cfg); err != nil {
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
	if err := m.v41AdmitShape(m.headName(), v41StageHead, -1, cfg.VocabSize, cfg.HiddenSize); err != nil {
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
	// full selects the artifact geometry over the reduced fixture. The V4.1 Q/KV
	// latent norms are an artifact-only stage, so their admission is gated on it.
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return err
	}
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
		// Q/KV latent norms (reference Attention: self.q_norm = RMSNorm(q_lora_rank)
		// and self.kv_norm = RMSNorm(head_dim)). They are admitted and applied ONLY
		// on the full path: the reduced fixture's pre-#13009 arithmetic has no such
		// stage and must stay byte-identical, so a reduced config neither requires
		// nor reads these leaves. See v41Layer's application site (#13290).
		if full {
			if err := m.v41AdmitShape(layerName(l, "attn.wq_a_norm.weight"), v41StageAttention, l, cfg.QLoraRank); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(l, "attn.kv_norm.weight"), v41StageAttention, l, kvLatentRank); err != nil {
				return err
			}
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

// v41AdmitMHC admits the attention mHC trio for reduced geometry, preserving
// its legacy [mixWidth, H] or flattened layout. Full geometry requires separate
// attention and FFN trios, each with a logical [mixWidth, 4H] coefficient matrix
// or supported stored [4H, mixWidth] transpose. Each named matrix and its own
// base/scale vectors pass exact shape admission; full geometry never falls back
// to a reduced-width or absent-FFN stand-in.
func (m *Model) v41AdmitMHC(l int) error {
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil {
		return err
	}
	if err := m.v41AdmitMHCNamed(l, "mhc.mixes.weight", "mhc.base", "mhc.scale", full); err != nil {
		return err
	}
	if full {
		return m.v41AdmitMHCNamed(l, "mhc.ffn_mixes.weight", "mhc.ffn_base", "mhc.ffn_scale", true)
	}
	return nil
}

// errV41MHCMixMissing distinguishes an absent mix from rejected geometry.
var errV41MHCMixMissing = errors.New("missing mHC mix tensor")

// v41MissingMHCMixError retains the missing name without parsing error text.
type v41MissingMHCMixError struct {
	Name string
}

func (e *v41MissingMHCMixError) Error() string { return "missing tensor " + e.Name }
func (e *v41MissingMHCMixError) Unwrap() error { return errV41MHCMixMissing }

func (m *Model) v41AdmitMHCNamed(l int, leaf, base, scale string, requireFlat bool) error {
	name := layerName(l, leaf)
	// A malformed manifest rank is still present; leave its geometry refusal
	// to the existing layout classifier. residentShape covers non-manifest stores.
	_, manifestPresent := m.manifest[name]
	_, _, residentPresent := m.residentShape(name)
	if !manifestPresent && !residentPresent {
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: %w", ErrV41ForwardStage, &v41MissingMHCMixError{Name: name}))
	}
	flat, _, ok := m.v41MHCWeightLayoutNamed(l, leaf)
	if !ok || (requireFlat && !flat) {
		return v41StageErr(v41StageMHC, l, fmt.Errorf("%w: tensor %s has no admitted mHC geometry (require flattened=%t)", ErrV41ForwardStage, name, requireFlat))
	}
	if err := m.v41AdmitShape(layerName(l, base), v41StageMHC, l, v41MHCMixWidth); err != nil {
		return err
	}
	return m.v41AdmitShape(layerName(l, scale), v41StageMHC, l, 3)
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
		if name == "model.embed_tokens.weight" {
			out, in, ok = m.v41HeadWeightShape(name)
		}
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

// v41OwnsCompressedRows keeps the legacy self-contained PerLayer fixture path
// while excluding shared readers. Only declared KV sources are producers in the
// pinned full-model topology; the no-source fallback is not reference parity.
func v41OwnsCompressedRows(plan V41AttentionPlan) bool {
	return plan.Ratio > 1 && plan.Role != V41AttentionRoleReader
}

// v41CompressedReaderRows resolves publications by KV ownership, independently
// of the most recent index-query source. Readers must never synthesize missing
// source rows, project fresh index keys, or rotate borrowed publications again.
func (m *Model) v41CompressedReaderRows(plan V41AttentionPlan, registry *V41AttentionState, positions int) ([][]float32, [][]float32, error) {
	if registry == nil || plan.KVSourceLayer < 0 || positions < 0 {
		return nil, nil, v41StageErr(v41StageAttention, plan.Layer,
			fmt.Errorf("%w: layer %d has no shared KV source state", ErrV41ForwardStage, plan.Layer))
	}
	groups := positions / plan.kvGroupSize(m.Cfg)
	rows, ok := registry.KVSourceRows(plan.KVSourceLayer)
	if len(rows) != groups || (groups > 0 && !ok) {
		return nil, nil, v41StageErr(v41StageAttention, plan.Layer,
			fmt.Errorf("%w: layer %d requires %d completed rows from KV source %d, got %d", ErrV41ForwardStage, plan.Layer, groups, plan.KVSourceLayer, len(rows)))
	}
	var keys [][]float32
	if indexSourceAt(m.Cfg.DeepSeekV41, plan.Layer) {
		keys, ok = registry.IndexKeys(plan.KVSourceLayer)
		if len(keys) != groups || (groups > 0 && !ok) {
			return nil, nil, v41StageErr(v41StageIndexer, plan.Layer,
				fmt.Errorf("%w: index-only layer %d requires %d keys from KV source %d, got %d", ErrV41ForwardStage, plan.Layer, groups, plan.KVSourceLayer, len(keys)))
		}
	}
	return rows, keys, nil
}
