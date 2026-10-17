package model

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/internal/stepobs"
)

// v41AttentionNormBinding keeps both ordinary attention-input norm edges with
// their optional canonical owner. Replacing either edge discards panel authority;
// copying a binding preserves identity without granting a new lifetime.
type v41AttentionNormBinding struct {
	normalize v41AttentionInputNormFunc
	publish   v41BF16BoundaryFunc
	native    *v41BoundAttentionNorm
}

type v41BoundAttentionNorm struct {
	owner   *Session
	model   *Model
	backend compute.Backend
}

func (s *Session) v41BindAttentionNorm() *v41AttentionNormBinding {
	if s == nil {
		return nil
	}
	model, backend := s.M, s.Backend
	normalize := s.v41AttentionInputNormFunc()
	publish, publicationModel, publicationBackend := s.v41AttentionBF16WithProvenance()
	if normalize == nil && publish == nil {
		return nil
	}
	b := &v41AttentionNormBinding{normalize: normalize, publish: publish}
	if normalize != nil && publish != nil && model != nil && backend != nil && backendHasComparableIdentity(backend) && publicationModel == model && backendHasComparableIdentity(publicationBackend) && publicationBackend == backend {
		b.native = &v41BoundAttentionNorm{owner: s, model: model, backend: backend}
	}
	return b
}

func (b *v41AttentionNormBinding) normFunc() v41AttentionInputNormFunc {
	if b == nil {
		return nil
	}
	return b.normalize
}
func (b *v41AttentionNormBinding) publishFunc() v41BF16BoundaryFunc {
	if b == nil {
		return nil
	}
	return b.publish
}
func (b *v41AttentionNormBinding) replaceNorm(f v41AttentionInputNormFunc) *v41AttentionNormBinding {
	return &v41AttentionNormBinding{normalize: f, publish: b.publishFunc()}
}
func (b *v41AttentionNormBinding) replacePublish(f v41BF16BoundaryFunc) *v41AttentionNormBinding {
	return &v41AttentionNormBinding{normalize: b.normFunc(), publish: f}
}
func (s *v41ForwardState) attentionNormFunc() v41AttentionInputNormFunc {
	return s.attentionNormBinding.normFunc()
}
func (s *v41ForwardState) attentionBF16Func() v41BF16BoundaryFunc {
	return s.attentionNormBinding.publishFunc()
}
func (s *v41ForwardState) setAttentionNorm(f v41AttentionInputNormFunc) {
	s.attentionNormBinding = s.attentionNormBinding.replaceNorm(f)
}
func (s *v41ForwardState) setAttentionBF16(f v41BF16BoundaryFunc) {
	s.attentionNormBinding = s.attentionNormBinding.replacePublish(f)
}
func (s *v41ProjScratch) attentionNormFunc() v41AttentionInputNormFunc {
	return s.attentionNormBinding.normFunc()
}
func (s *v41ProjScratch) attentionBF16Func() v41BF16BoundaryFunc {
	return s.attentionNormBinding.publishFunc()
}

// coldPanel selects only a live canonical resident cold-forward binding. Input
// refusal is decided before any effect and keeps the original scalar row order.
// Within a selected chunk, numerical checks and output publications retain that
// order; physical dispatch and cleanup timing intentionally become chunk-local.
func (b *v41AttentionNormBinding) coldPanel(m *Model, st *v41ForwardState, scratch *v41ProjScratch, resident *v41MHCResident, layer int, input [][]float32, eps float32) ([][]float32, bool, error) {
	if scratch != nil {
		scratch.attentionInputPanel = nil
	}
	if b == nil || b.native == nil || len(input) <= 1 {
		return nil, false, nil
	}
	n := b.native
	valid := func() bool {
		return n.owner != nil && n.model == m && n.owner.M == m && backendHasComparableIdentity(n.owner.Backend) && n.owner.Backend == n.backend &&
			!n.owner.halClosed && n.owner.halFailure == nil && n.owner.v41Forward != nil && n.owner.v41Forward.attentionNormBinding == b &&
			st != nil && st.attentionNormBinding == b && st.callbackOwner == n.owner && scratch != nil && scratch.attentionNormBinding == b &&
			resident != nil && resident.d != nil && resident.d.s == n.owner && resident.rows == len(input)
	}
	if !valid() || layer < 0 || layer >= m.Cfg.NumLayers || eps != float32(m.Cfg.RMSNormEps) {
		return nil, false, nil
	}
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil || !full || m.Cfg.LayerNorm || m.Cfg.NormGain1p {
		return nil, false, nil
	}
	width := m.Cfg.HiddenSize
	if width <= 0 || !finite32(eps) || eps <= 0 {
		return nil, false, nil
	}
	for _, row := range input {
		if len(row) != width || !v41BF16Widened(row) {
			return nil, false, nil
		}
	}
	// A loan is eligible only before any normalized storage exists. Canonical
	// Read returns fresh storage; canonical publication/upload neither exposes
	// nor retains it. Unknown observers, clocks and replacement callbacks retain
	// the snapshot path. Pointer equality checks identity, not ownership.
	loan := b.newInputPanel(m, st, scratch, layer, width, len(input))
	output := make([][]float32, len(input))
	for lo := 0; lo < len(input); lo += v41ProjPanelRowChunk {
		if !valid() {
			return nil, true, n.failBinding()
		}
		if loan != nil && !loan.valid(scratch) {
			loan = nil
		}
		hi := min(lo+v41ProjPanelRowChunk, len(input))
		count, ok := checkedMulInt(hi-lo, width)
		if !ok {
			return nil, true, n.failBinding()
		}
		flat := make([]float32, 0, count)
		for _, row := range input[lo:hi] {
			flat = append(flat, row...)
		}
		publicationFailed := false
		values, cause := n.owner.v41RMSNormRows(layerName(layer, "attn_norm.weight"), flat, width, hi-lo, "v41-attention-input-norm", layer, func(row []float32) ([]float32, error) {
			rounded, err := v41BF16BoundaryCopy(layer, "attn_norm.weight output", row, b.publish)
			publicationFailed = err != nil
			return rounded, err
		})
		if cause != nil {
			if !publicationFailed {
				cause = v41ProjectionOperationErr(layer, "attn_norm.weight", cause)
			}
			return nil, true, cause
		}
		if !valid() {
			return nil, true, n.failBinding()
		}
		if loan != nil {
			if loan.valid(scratch) {
				loan.chunks = append(loan.chunks, values)
			} else {
				loan = nil
			}
		}
		for row := lo; row < hi; row++ {
			start := (row - lo) * width
			output[row] = values[start : start+width : start+width]
		}
	}
	// Publish only after the complete norm panel succeeds. This scratch belongs
	// to forwardV41's stack-local continuation; neither chunks nor row headers
	// enter Session state before QKV has consumed both projections.
	if loan != nil && loan.valid(scratch) && loan.matches(layer, width, output) {
		scratch.attentionInputPanel = loan
	}
	return output, true, nil
}

func (n *v41BoundAttentionNorm) failBinding() (cause error) {
	s := n.owner
	if s.halFailure != nil {
		return s.halFailure
	}
	cause = &BackendForwardOperationError{Backend: n.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-attention-input-norm", Layer: -1, Stage: "panel binding", Cause: errV41ProjectionResult}
	s.halFailure = cause
	defer func() { s.halClosed = true; _ = recover() }()
	s.closeBackendOwned()
	return cause
}

// v41ReleaseNormPanelTensors attempts every chunk free before a publication can
// retire the owner. Deferred frees also drain on Goexit. Completion is separate
// from recover's payload, since legacy panic(nil) carries no nonnil value.
func v41ReleaseNormPanelTensors(backend compute.Backend, owned *[]compute.Tensor, cause *error, completed *bool) {
	primary := recover()
	interrupted := !*completed
	var cleanup any
	cleanupPresent := false
	defer func() {
		*owned = nil
		if primary != nil {
			panic(primary)
		}
		if interrupted {
			if *cause == nil {
				*cause = errV41ProjectionResult
			}
			return
		}
		if *cause == nil && cleanupPresent {
			if cleanup != nil {
				panic(cleanup)
			}
			*cause = errV41ProjectionResult
		}
	}()
	// Register in acquisition order so execution remains reverse acquisition.
	for _, tensor := range *owned {
		defer func(t compute.Tensor) {
			returned := false
			defer func() {
				value := recover()
				if !returned && !cleanupPresent {
					cleanup, cleanupPresent = value, true
				}
			}()
			backend.Free(t)
			returned = true
		}(tensor)
	}
}

// v41AttentionInputPanel is a private synchronous loan from coldPanel to QKV.
// It is not a generic slice-admission API: only the canonical producer mints it,
// before its fresh rows can escape. The forward remains exclusive and its
// configuration, adapters and callback bindings remain immutable until return.
type v41AttentionInputPanel struct {
	owner              *Session
	model              *Model
	backend            compute.Backend
	state              *v41ForwardState
	run                *v41AttentionInputRun
	observation        *computetrace.Observation
	norm               *v41AttentionNormBinding
	route              *v41ProjectionRouteBinding
	query              *v41QueryNormBinding
	gather             *v41DeviceGatherOps
	publication        *v41BF16PublicationBinding
	scope              *backendExecutionScope
	epoch              uint64
	layer, width, rows int
	chunks             [][]float32
}

func (b *v41AttentionNormBinding) newInputPanel(m *Model, st *v41ForwardState, scratch *v41ProjScratch, layer, width, rows int) *v41AttentionInputPanel {
	if b == nil || b.native == nil || b.native.owner == nil || scratch == nil || rows <= 1 {
		return nil
	}
	s := b.native.owner
	p := &v41AttentionInputPanel{owner: s, model: m, backend: b.native.backend, state: st, norm: b,
		route: scratch.projectionRoute, query: scratch.queryNorm, gather: scratch.gatherOps,
		observation: stepobs.CaptureHostInputObservation(),
		scope:       s.backendExecutionScope, epoch: s.cacheGeometryEpoch, layer: layer, width: width, rows: rows}
	if st != nil {
		p.run = st.attentionInputRun
	}
	if p.query != nil {
		p.publication = p.query.publication
	}
	if !p.valid(scratch) {
		return nil
	}
	p.chunks = make([][]float32, 0, 1+(rows-1)/v41ProjPanelRowChunk)
	return p
}

func (p *v41AttentionInputPanel) valid(scratch *v41ProjScratch) bool {
	return p != nil && compute.V41HostInputBorrowSafe(p.backend) && p.scopeValid(scratch)
}

// scopeValid alone grants no borrow authority; valid also requires the sealed
// exact backend. Keeping the metadata predicate separate permits CPU-only
// regression checks without forging a canonical device backend.
func (p *v41AttentionInputPanel) scopeValid(scratch *v41ProjScratch) bool {
	if p == nil || p.owner == nil || p.model == nil || scratch == nil || p.state == nil ||
		p.norm == nil || p.norm.native == nil || p.route == nil || p.query == nil || p.gather == nil || p.publication == nil ||
		scratch.activationTrace != nil || !p.observation.Unchanged() || !p.run.valid(p.owner, p.state) ||
		!backendHasComparableIdentity(p.backend) || !backendHasComparableIdentity(p.owner.Backend) {
		return false
	}
	s, m, state := p.owner, p.model, p.state
	source := p.run.source
	n, dense, query, publication := p.norm.native, p.route.nativeProjection, p.query, p.publication
	if s.M != m || s.Backend != p.backend || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld ||
		state.callbackOwner != s || s.backendExecutionScope != p.scope || s.cacheGeometryEpoch != p.epoch ||
		p.scope == nil || !p.scope.live || p.scope.owner == nil || m.lora != nil || m.prism != nil ||
		p.layer < 0 || p.layer >= m.Cfg.NumLayers || p.width != m.Cfg.HiddenSize ||
		n.owner != s || n.model != m || n.backend != p.backend || p.norm.normalize == nil || p.norm.publish == nil ||
		source.attentionNormBinding != p.norm || source.projectionRoute != p.route || source.queryNorm != query || source.gatherOps != p.gather ||
		state.attentionNormBinding != p.norm || scratch.attentionNormBinding != p.norm ||
		state.projectionRoute != p.route || scratch.projectionRoute != p.route || dense.core == nil || dense.ordinary == nil ||
		dense.owner != s || dense.model != m || dense.backend != p.backend || (p.route.deviceOnly != nil && p.route.deviceOnly != s) ||
		state.queryNorm != query || scratch.queryNorm != query || query.ordinary == nil || query.owner != s || query.model != m || query.backend != p.backend ||
		query.publication != publication || publication.core == nil || publication.ordinary == nil || publication.owner != s || publication.model != m || publication.backend != p.backend ||
		state.gatherOps != p.gather || scratch.gatherOps != p.gather || p.gather.owner != s || p.gather.bf16Publish != publication {
		return false
	}
	member := false
	for _, candidate := range p.scope.sessions {
		if candidate == s {
			member = true
			break
		}
	}
	if !member {
		return false
	}
	// The normal ledger is harmless. An injected clock is a callback and may
	// mutate captured rows, so it must keep the original snapshot semantics.
	if ledger := v41ExpertFaultLedgerOf(m, false); ledger != nil {
		ledger.mu.Lock()
		custom := ledger.nowNanos != nil
		ledger.mu.Unlock()
		if custom {
			return false
		}
	}
	return true
}

// v41AttentionInputRun binds a cold forward's temporary state to the live
// Session source that minted it. It is never snapshot or continuation data.
// Keeping both endpoints and the live token prevents copied/stale run states
// from acquiring authority merely by carrying the same callback owner.
type v41AttentionInputRun struct {
	owner       *Session
	source, run *v41ForwardState
	live        bool
}

func (source *v41ForwardState) beginAttentionInputRun(run *v41ForwardState) *v41AttentionInputRun {
	if source == nil || run == nil || source == run || source.callbackOwner == nil ||
		source.callbackOwner.v41Forward != source || source.attentionNormBinding == nil || source.attentionNormBinding.native == nil ||
		source.attentionInputRun != nil || run.attentionInputRun != nil {
		return nil
	}
	r := &v41AttentionInputRun{owner: source.callbackOwner, source: source, run: run, live: true}
	source.attentionInputRun, run.attentionInputRun = r, r
	return r
}

func (r *v41AttentionInputRun) valid(owner *Session, run *v41ForwardState) bool {
	return r != nil && r.live && r.owner == owner && r.run == run && r.source != nil && r.source != run &&
		owner != nil && owner.v41Forward == r.source && r.source.callbackOwner == owner &&
		r.source.attentionInputRun == r && run != nil && run.attentionInputRun == r
}

func (r *v41AttentionInputRun) close() {
	if r == nil {
		return
	}
	r.live = false
	if r.source.attentionInputRun == r {
		r.source.attentionInputRun = nil
	}
	if r.run.attentionInputRun == r {
		r.run.attentionInputRun = nil
	}
}

func (p *v41AttentionInputPanel) matches(layer, width int, rows [][]float32) bool {
	if p == nil || p.layer != layer || p.width != width || width <= 0 || p.rows != len(rows) || len(rows) <= 1 ||
		len(p.chunks) != 1+(len(rows)-1)/v41ProjPanelRowChunk {
		return false
	}
	for lo, chunk := range p.chunks {
		start := lo * v41ProjPanelRowChunk
		hi := min(start+v41ProjPanelRowChunk, len(rows))
		count, ok := checkedMulInt(hi-start, width)
		if !ok || len(chunk) != count {
			return false
		}
		for row := start; row < hi; row++ {
			if len(rows[row]) != width || cap(rows[row]) != width || &rows[row][0] != &chunk[(row-start)*width] {
				return false
			}
		}
	}
	return true
}

func (p *v41AttentionInputPanel) fail() error {
	s := p.owner
	if s.halFailure == nil {
		s.halFailure = &BackendForwardOperationError{Backend: p.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-attention-input-panel", Layer: p.layer, Stage: "panel binding", Cause: errV41ProjectionResult}
	}
	if !s.halClosed {
		func() { defer func() { s.halClosed = true; _ = recover() }(); s.closeBackendOwned() }()
	}
	return s.halFailure
}
