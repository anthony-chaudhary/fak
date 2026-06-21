package model

import "fmt"

// partition.go — pipeline-parallel layer partitioning for the lean quant loader.
//
// A 753B GLM-5.2 checkpoint does not fit one node's host RAM even quantized
// (~830 GB at Q8); native multi-GPU serving needs each pipeline-parallel worker
// to load only ITS band of transformer layers. LayerWindow is that seam: a load
// option that keeps only the resident weights whose `model.layers.N.*` index is
// in [Lo, Hi). Weights with no layer index (embeddings, final norm, an untied
// lm_head) are layer-agnostic and always kept — a worker that owns the head or
// embedding needs them regardless of its layer band.
//
// The default window keeps every layer, so a load with no option is byte-for-byte
// the previous behavior (the no-op gate, TestLayerWindowFullIsNoOp). This is the
// loader-side foundation for GLM-5.2-NATIVE-ENGINE-GAP gaps #1 (device/
// parallelism) and #2 (bounded checkpoint load); it bounds resident memory per
// worker without touching the forward path.

// loadOptions is the resolved option set the lean loaders consult.
type loadOptions struct {
	window layerWindow
}

// layerWindow is a half-open [Lo, Hi) band of transformer layers to keep.
// hiUnset (the default) means "no upper bound" so the zero value keeps all.
type layerWindow struct {
	lo      int
	hi      int
	hiUnset bool
}

func defaultLoadOptions() loadOptions {
	return loadOptions{window: layerWindow{lo: 0, hi: 0, hiUnset: true}}
}

// keepsLayer reports whether layer index N is inside the window. Layer-agnostic
// tensors (N < 0 by convention from the caller) are always kept.
func (w layerWindow) keepsLayer(n int) bool {
	if n < 0 {
		return true
	}
	if n < w.lo {
		return false
	}
	if w.hiUnset {
		return true
	}
	return n < w.hi
}

// LoadOption configures a lean quant load. The zero set keeps all layers.
type LoadOption func(*loadOptions)

// WithLayerWindow keeps only the resident matmul weights whose layer index is in
// [lo, hi). A pipeline-parallel worker for the k-th band passes its own [lo, hi);
// layer-agnostic weights (embeddings, norm, untied lm_head) are kept regardless.
// hi <= lo would select nothing and is rejected at apply time as a programming
// error rather than silently loading an empty model.
func WithLayerWindow(lo, hi int) LoadOption {
	return func(o *loadOptions) {
		o.window = layerWindow{lo: lo, hi: hi, hiUnset: false}
	}
}

func resolveLoadOptions(opts []LoadOption) loadOptions {
	o := defaultLoadOptions()
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}
	return o
}

// tensorLayerForWindow returns the layer index a tensor belongs to, or -1 if it
// is layer-agnostic (no `model.layers.N.` prefix). Used by the loader to apply a
// layerWindow without duplicating the name-parsing rule.
func tensorLayerForWindow(name string) int {
	if layer, _, ok := parseLayerTensorSuffix(name); ok {
		return layer
	}
	return -1
}

// embedBand produces the initial hidden state for a pipeline's FIRST band: the
// token embedding lookup (with any arch embed scaling), the same as the head of
// Model.Forward. Later bands receive the hidden state from the previous worker
// instead of calling this.
func (m *Model) embedBand(ids []int) [][]float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	embed := m.embedRows()
	x := make([][]float32, len(ids))
	for t, id := range ids {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], cfg)
	}
	return x
}

// ForwardBand runs the contiguous transformer layer band [lo, hi) over the hidden
// state x IN PLACE — the execution unit a pipeline-parallel worker owns. It is
// the partition counterpart of Model.Forward: Forward is exactly
// embedBand(ids) followed by ForwardBand(x, 0, NumLayers, true). Splitting the
// model into bands and threading x between them is therefore bit-exact
// (TestForwardBandTwoWayMatchesMonolithic), because each band runs the identical
// per-layer instruction stream over the identical hidden state.
//
// When isLast is true the final norm + LM head run and the per-position logits are
// returned; otherwise logits is nil and the caller forwards the returned hidden
// state to the next band's worker. The returned [][]float32 is x (mutated in
// place) for convenience.
//
// It fails closed on a malformed band rather than producing garbage: hi must be
// in (lo, NumLayers], lo >= 0. For a GLM-MoE-DSA model a band must not begin in
// the middle of an IndexShare group (a "shared" indexer layer reuses the previous
// "full" layer's index, which a band that starts on the shared layer would not
// have computed); such a split is rejected so cross-worker DSA reuse can never be
// silently wrong.
func (m *Model) ForwardBand(x [][]float32, lo, hi int, isLast bool) ([][]float32, [][]float32, error) {
	cfg := m.Cfg
	if lo < 0 || hi <= lo || hi > cfg.NumLayers {
		return nil, nil, fmt.Errorf("model: ForwardBand range [%d,%d) invalid for %d layers", lo, hi, cfg.NumLayers)
	}
	if cfg.isGLMMoeDsa() && glmDsaIndexerIsShared(cfg, lo) {
		return nil, nil, fmt.Errorf("model: ForwardBand cannot start at GLM IndexShare shared layer %d (band must begin on a full-indexer layer)", lo)
	}
	// The IndexShare shared-top-k carries across layers WITHIN a band. A band that
	// begins on a full-indexer layer (guarded above) recomputes its own group head,
	// so a fresh per-band slice is correct.
	var glmDsaSharedTopK [][]int
	for l := lo; l < hi; l++ {
		rp := newRopeForLayer(cfg, l, len(x))
		if cfg.isGLMMoeDsa() {
			m.layerGLMDsa(l, x, rp, &glmDsaSharedTopK)
		} else {
			m.layer(l, x, rp)
		}
	}
	if !isLast {
		return x, nil, nil
	}
	mat := residentKernel{m}
	H := cfg.HiddenSize
	logits := make([][]float32, len(x))
	for t := 0; t < len(x); t++ {
		xf := m.finalNorm(x[t])
		row := mat.mul(m.headName(), mat.prep(xf), cfg.VocabSize, H)
		logitScaleInPlace(row, cfg)
		logits[t] = row
	}
	return x, logits, nil
}
