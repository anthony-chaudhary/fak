package model

import (
	"errors"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41ProjectionRouteBinding is an immutable pair of ordinary callbacks. A
// replacement creates a new pair, so state and scratch copies cannot observe
// another owner's callback replacement. Canonical native halves carry no device
// tensors and only synchronous capability. Snapshots omit the entire binding.
type v41ProjectionRouteBinding struct {
	projection       v41DenseProjectionFunc
	router           v41RouterTopKFunc
	nativeProjection v41BoundDenseProjection
	nativeRouter     v41BoundRouterTopK
	deviceOnly       *Session
}

func newV41ProjectionRouteBinding(projection v41DenseProjectionFunc, router v41RouterTopKFunc) *v41ProjectionRouteBinding {
	if projection == nil && router == nil {
		return nil
	}
	return &v41ProjectionRouteBinding{projection: projection, router: router}
}

func (b *v41ProjectionRouteBinding) projectionFunc() v41DenseProjectionFunc {
	if b == nil {
		return nil
	}
	return b.projection
}

func (b *v41ProjectionRouteBinding) routerFunc() v41RouterTopKFunc {
	if b == nil {
		return nil
	}
	return b.router
}

// Replacements retain only ordinary callbacks. Optional execution capabilities
// must never be inherited by an arbitrary wrapper around an old callback.
func (b *v41ProjectionRouteBinding) withProjection(projection v41DenseProjectionFunc) *v41ProjectionRouteBinding {
	return newV41ProjectionRouteBinding(projection, b.routerFunc())
}

func (b *v41ProjectionRouteBinding) withRouter(router v41RouterTopKFunc) *v41ProjectionRouteBinding {
	return newV41ProjectionRouteBinding(b.projectionFunc(), router)
}

func (st *v41ForwardState) denseProjection() v41DenseProjectionFunc {
	return st.projectionRoute.projectionFunc()
}

func (st *v41ForwardState) routerTopK() v41RouterTopKFunc {
	return st.projectionRoute.routerFunc()
}

func (st *v41ForwardState) setDenseProjection(projection v41DenseProjectionFunc) {
	st.projectionRoute = st.projectionRoute.withProjection(projection)
}

func (st *v41ForwardState) setRouterTopK(router v41RouterTopKFunc) {
	st.projectionRoute = st.projectionRoute.withRouter(router)
}

func (scratch *v41ProjScratch) denseProjection() v41DenseProjectionFunc {
	return scratch.projectionRoute.projectionFunc()
}

func (scratch *v41ProjScratch) routerTopK() v41RouterTopKFunc {
	return scratch.projectionRoute.routerFunc()
}

// Canonical products share one operation core with their ordinary adapters.
// Provenance is used only for native selection; ordinary closures keep their
// existing Session dereferences and the router's factory-captured operation.
type v41DenseProjectionCore func(int, string, []float32, int, int, int, *v41ProjectionContinuation) ([]float32, v41DenseProjectionOutcome, error)
type v41RouterTopKCore func(int, []float32, []float32, v41RouterConfig, bool, *v41RouterResidentInput) ([]routePick, error)

type v41BoundDenseProjection struct {
	ordinary v41DenseProjectionFunc
	core     v41DenseProjectionCore
	owner    *Session
	model    *Model
	backend  compute.Backend
}
type v41BoundRouterTopK struct {
	ordinary v41RouterTopKFunc
	core     v41RouterTopKCore
	owner    *Session
	model    *Model
	backend  compute.Backend
}

func (st *v41ForwardState) bindCanonicalDense(p v41BoundDenseProjection) {
	b := newV41ProjectionRouteBinding(p.ordinary, st.routerTopK())
	if b != nil {
		b.nativeProjection = p
	}
	st.projectionRoute = b // Never retain a previous native router half.
}
func (st *v41ForwardState) bindCanonicalRouter(r v41BoundRouterTopK) {
	old := st.projectionRoute
	b := newV41ProjectionRouteBinding(st.denseProjection(), r.ordinary)
	if b != nil && old != nil {
		b.nativeProjection = old.nativeProjection
		b.nativeRouter = r
		b.deviceOnly = old.deviceOnly
	}
	st.projectionRoute = b
}
func (st *v41ForwardState) liftDeviceOnlyDense(s *Session) {
	old := st.projectionRoute
	b := newV41ProjectionRouteBinding(v41DeviceOnlyDense(s, st.denseProjection()), st.routerTopK())
	if old != nil && old.nativeProjection.owner == s {
		b.nativeProjection, b.nativeRouter = old.nativeProjection, old.nativeRouter
		b.deviceOnly = s
	}
	st.projectionRoute = b
}

var errV41RouterProjectionScope = errors.New("model: V4.1 router projection scope changed")

type v41ProjectionContinuation struct {
	consume    func(compute.Tensor, *v41ProjectionLoan)
	valid      func() bool
	cause      error
	panicValue any
	completed  bool
}

func (c *v41ProjectionContinuation) invoke(t compute.Tensor, loan *v41ProjectionLoan) {
	defer func() { c.panicValue = recover() }()
	c.consume(t, loan)
}

// The projection remains the owner. Only its synchronous router continuation
// may retire this loan, inside the router's existing selected-operation guard.
type v41ProjectionLoan struct {
	backend   compute.Backend
	protected compute.Buffer
	owned     []compute.Tensor
	valid     func() bool
}

func (l *v41ProjectionLoan) own(t compute.Tensor) bool {
	if t.Buf() == nil || t.Backend() != l.backend || t.Buf() == l.protected {
		return false
	}
	for _, old := range l.owned {
		if old.Buf() == t.Buf() {
			return false
		}
	}
	l.owned = append(l.owned, t)
	return true
}

type v41ProjectionCleanup struct {
	value   any
	present bool
	legacy  any
}

func (l *v41ProjectionLoan) releaseCleanup() (out v41ProjectionCleanup) {
	for i := len(l.owned) - 1; i >= 0; i-- {
		func() {
			complete := false
			defer func() {
				value := recover()
				if !complete && !out.present {
					out.value, out.present = value, true
				}
				if value != nil && out.legacy == nil {
					out.legacy = value
				}
			}()
			l.backend.Free(l.owned[i])
			complete = true
		}()
	}
	l.owned = nil
	return out
}

func (l *v41ProjectionLoan) release() (first any) {
	first = l.releaseCleanup().legacy
	if first == nil && l.valid != nil && !l.valid() {
		first = errV41RouterProjectionScope
	}
	return first
}

func (l *v41ProjectionLoan) releasePresent() (any, bool) {
	out := l.releaseCleanup()
	if out.present {
		return out.value, true
	}
	if l.valid != nil && !l.valid() {
		return errV41RouterProjectionScope, true
	}
	return nil, false
}

type v41RouterResidentInput struct {
	tensor    compute.Tensor
	loan      *v41ProjectionLoan
	completed *bool
}

func (b *v41ProjectionRouteBinding) nativeOwner(m *Model, scratch *v41ProjScratch, cfg v41RouterConfig, full bool, layer int) *Session {
	if b == nil || !full || cfg != v41DefaultRouterConfig() || m.lora != nil || m.prism != nil {
		return nil
	}
	p, r := b.nativeProjection, b.nativeRouter
	s := p.owner
	if p.core == nil || r.core == nil || s == nil || r.owner != s || p.model != m || r.model != m || p.backend != r.backend ||
		s.M != m || s.Backend != p.backend || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld ||
		s.v41Forward == nil || s.v41Forward.callbackOwner != s || s.v41Forward.projectionRoute != b || scratch.projectionRoute != b {
		return nil
	}
	scope := s.backendExecutionScope
	if scope == nil || !scope.live || scope.owner == nil {
		return nil
	}
	member := false
	for _, owned := range scope.sessions {
		if owned == s {
			member = true
			break
		}
	}
	if !member {
		return nil
	}

	// Metadata only: the role-finish branch must not call tensor before its
	// projection. The held immutable raw view will be safe at its original site.
	meta, ok := m.manifest[layerName(layer, "ffn.gate.e_score_correction_bias")]
	n, valid := checkedMulInt(cfg.Experts, 4)
	if !ok || !valid || !strings.EqualFold(meta.Dtype, "F32") || len(meta.Shape) != 1 || meta.Shape[0] != cfg.Experts ||
		meta.Nbytes != n || meta.Offset < 0 || meta.Offset > len(m.raw) || n > len(m.raw)-meta.Offset {
		return nil
	}
	return s
}

// This composer is synchronous: picks cannot escape until the projection loan
// has been retired. Ordinary replacement callbacks never reach its native arm.
func (m *Model) v41RouterPicksForStep(layer int, input []float32, hidden int, cfg v41RouterConfig, full bool, scratch *v41ProjScratch, bias func() []float32) ([]routePick, error) {
	req, err := m.v41AdmitProjectionRows(layer, "ffn.gate.weight", input, cfg.Experts, hidden, 1)
	if err != nil {
		return nil, err
	}
	b := scratch.projectionRoute
	routeHost := func(logits []float32) ([]routePick, error) {
		picks, err := v41RouteForGeometryWith(layer, logits, bias(), cfg, full, scratch.routerTopK())
		if err != nil {
			return nil, v41StageErr(v41StageMoE, layer, err)
		}
		return picks, nil
	}
	s := b.nativeOwner(m, scratch, cfg, full, layer)
	if hidden != m.Cfg.HiddenSize || cfg.Experts != m.Cfg.NumExperts {
		s = nil
	}
	if s == nil {
		logits, err := m.v41ProjectionRowsAdmitted(req, input, scratch.denseProjection())
		if err != nil {
			return nil, err
		}
		return routeHost(logits)
	}
	state, scope, epoch, backend := s.v41Forward, s.backendExecutionScope, s.cacheGeometryEpoch, s.Backend
	valid := func() bool {
		return s.M == m && s.Backend == backend && s.v41Forward == state && state.projectionRoute == b && state.callbackOwner == s &&
			s.cacheGeometryEpoch == epoch && s.backendExecutionScope == scope && scope.live && m.lora == nil && m.prism == nil
	}
	var picks []routePick
	c := &v41ProjectionContinuation{valid: valid}
	c.consume = func(t compute.Tensor, loan *v41ProjectionLoan) {
		loan.valid = valid
		correction := bias() // A panic here retains its original unwrapped identity.
		if err := cfg.validate(); err != nil {
			c.cause = err
			return
		}
		view := t
		view.Shape = []int{1, cfg.Experts}
		picks, c.cause = b.nativeRouter.core(layer, nil, correction, cfg, full, &v41RouterResidentInput{tensor: view, loan: loan, completed: &c.completed})
	}
	_, outcome, cause := b.nativeProjection.core(layer, "ffn.gate.weight", input, cfg.Experts, hidden, 1, c)
	if outcome == v41ProjectionDeclined {
		if b.deviceOnly != nil {
			return nil, v41ProjectionOperationErr(layer, "ffn.gate.weight", b.deviceOnly.v41DeviceOnlyDeclined(layer, "dense-projection"))
		}
		logits, err := m.v41ProjectionHostRows(req, input)
		if err != nil {
			return nil, err
		}
		return routeHost(logits) // The projection core is never called again.
	}
	if outcome != v41ProjectionHandled || cause != nil {
		if cause == nil {
			cause = errV41ProjectionResult
		}
		return nil, v41ProjectionOperationErr(layer, "ffn.gate.weight", cause)
	}
	if c.panicValue != nil {
		panic(c.panicValue)
	}
	if c.cause != nil {
		return nil, v41StageErr(v41StageMoE, layer, c.cause)
	}
	if len(picks) != cfg.TopK {
		return nil, v41StageErr(v41StageMoE, layer, &v4RouteError{Field: "device_route", Reason: errV41DeviceRouteResult.Error()})
	}
	return picks, nil
}
