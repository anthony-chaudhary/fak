package model

// v41SharedAttentionPrepareScanProbe observes the row-KV elements the device
// preparation finite-scans. Tests set it; production leaves it nil.
var v41SharedAttentionPrepareScanProbe func(elements int)

// newV41ColdSharedAttentionContinuation returns a per-layer template that lets
// cold/full-prefill positions reuse the combined composer's original-row
// validation, so the device preparation does not re-scan the window plus the
// whole compressed bank on every position. Each position takes a fresh
// one-use copy from at; consume still binds the exact request, row slices and
// resident view. Cold prefill runs on a step-local state, so liveness pins
// that state, the committed session state and the seeded layer owner instead
// of decode's history position. Positions after the first trust the composer's
// per-layer v41ValidatedPrefix memo, which is sound because the cold bank is
// private to the layer loop: a source layer's compressed rows are rotated before
// the loop, readers use copied payload rows, the resident mirror packs its own
// staging copies, and callbacks between positions see only the output row. An
// installed clock may mutate rows between positions, so it declines, as the cold
// host contraction does.
func newV41ColdSharedAttentionContinuation(m *Model, layer int, plan V41AttentionPlan, st *v41ForwardState, owner *V41AttentionState, scratch *v41ProjScratch) *v41SharedAttentionContinuation {
	if m == nil || st == nil || owner == nil || scratch == nil || v41ColdAttentionClockInstalled(m) {
		return nil
	}
	binding := scratch.sharedAttentionBinding
	if binding == nil || binding.native.core == nil || binding.native.core.validated == nil || st.sharedAttentionBinding != binding {
		return nil
	}
	s := binding.native.owner
	if s == nil || !backendHasComparableIdentity(s.Backend) || s.M != m || binding.native.model != m || binding.native.backend != s.Backend {
		return nil
	}
	committed, scope, epoch, backend := s.v41Forward, s.backendExecutionScope, s.cacheGeometryEpoch, s.Backend
	if scope == nil || scope.owner == nil || st.layerState(layer) != owner {
		return nil
	}
	cfg, scopeOwner, ringRows := m.Cfg, scope.owner, owner.windowSize
	registry, history, nextWindowPos := st.attn, len(st.history), owner.nextWindowPos
	window := cfg.windowForLayer(layer)
	c := &v41SharedAttentionContinuation{binding: binding, layer: layer, pos: -1,
		rowOwner: owner, ringRows: ringRows,
		sourceRatio: plan.kvGroupSize(cfg), heads: cfg.NumHeads, headDim: cfg.HeadDim, scale: cfg.attnScale()}
	c.live = func() bool {
		if s.M != m || !backendHasComparableIdentity(s.Backend) || s.Backend != backend ||
			s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || s.v41Forward != committed || st.callbackOwner != s ||
			st.sharedAttentionBinding != binding || scratch.sharedAttentionBinding != binding || st.attn != registry || st.layerState(layer) != owner ||
			s.cacheGeometryEpoch != epoch || s.backendExecutionScope != scope || !scope.live || scope.owner != scopeOwner ||
			len(scope.sessions) != 1 || scope.sessions[0] != s || len(st.history) != history || owner.nextWindowPos != nextWindowPos ||
			owner.windowSize != ringRows || owner.headDim != c.headDim || v41ColdAttentionClockInstalled(m) {
			return false
		}
		return c.configLive(m, cfg, plan, window)
	}
	if !c.live() {
		return nil
	}
	return c
}

// at mints the one-use proof for one cold position. Its first key matches
// v41ColdAttentionResidentView over v41PlainWindowKeys.
func (c *v41SharedAttentionContinuation) at(pos, window int) *v41SharedAttentionContinuation {
	if c == nil {
		return nil
	}
	next := *c
	next.pos, next.used, next.firstKey = pos, false, 0
	if window > 0 {
		next.firstKey = max(0, pos-window+1)
	}
	return &next
}

func v41ColdAttentionClockInstalled(m *Model) bool {
	ledger := v41ExpertFaultLedgerOf(m, false)
	if ledger == nil {
		return false
	}
	ledger.mu.Lock()
	defer ledger.mu.Unlock()
	return ledger.nowNanos != nil
}
