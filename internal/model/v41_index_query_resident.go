package model

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41IndexQueryProjection offers only the fixed canonical single-row projection
// to the pool-only publication continuation. The projection still owns x/y and
// finishes their cleanup before any published host result escapes.
func (m *Model) v41IndexQueryProjection(layer int, input []float32, width, in int, project v41DenseProjectionFunc, b *v41ProjectionRouteBinding, ops *v41DeviceGatherOps) (result []float32, published bool, cause error) {
	ordinary := func() ([]float32, bool, error) {
		values, err := m.v41ProjMatRowsWithProjection(layer, "indexer.wq_b.weight", input, width, in, project)
		return values, false, err
	}
	if b == nil || ops == nil || width != 8192 || m.Cfg.IndexNHeads != 64 || m.Cfg.IndexHeadDim != 128 || in != m.Cfg.QLoraRank || m.lora != nil || m.prism != nil {
		return ordinary()
	}
	p, pub := b.nativeProjection, ops.bf16Publish
	s := p.owner
	if s == nil || p.core == nil || pub == nil || pub.core == nil || p.model != m || pub.model != m || pub.owner != s || !backendHasComparableIdentity(p.backend) || pub.backend != p.backend {
		return ordinary()
	}
	state, scope, epoch, backend := s.v41Forward, s.backendExecutionScope, s.cacheGeometryEpoch, p.backend
	if _, ok := v41DeferredBF16Publisher(backend, 8192); !ok {
		return ordinary()
	}
	valid := func() bool {
		if state == nil || scope == nil || !scope.live || scope.owner == nil || s.M != m || s.Backend != backend || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || s.v41Forward != state || state.callbackOwner != s || state.projectionRoute != b || state.gatherOps != ops || ops.owner != s || ops.bf16Publish != pub || s.backendExecutionScope != scope || s.cacheGeometryEpoch != epoch || m.lora != nil || m.prism != nil {
			return false
		}
		for _, member := range scope.sessions {
			if member == s {
				return true
			}
		}
		return false
	}
	if !valid() {
		return ordinary()
	}
	req, err := m.v41AdmitProjectionRows(layer, "indexer.wq_b.weight", input, width, in, 1)
	if err != nil {
		return nil, false, err
	}
	const site = "indexer.wq_b.weight projection"
	c := &v41ProjectionContinuation{indexQuery: true, valid: valid}
	var image *compute.V41IndexQueryPublication
	closePublication := func(stage string, err error) error {
		closed := s.v41CloseBF16PublicationFailure(backend, layer, site, stage, err)
		return &V41BF16BoundaryOperationError{Layer: layer, Site: site, Cause: closed}
	}
	complete := false
	defer func() {
		if !complete {
			result = nil
			primary := cause
			if primary == nil {
				primary = c.projectionCause
			}
			if primary == nil {
				primary = c.cause
			}
			if primary == nil && c.panicPresent {
				primary, _ = c.panicValue.(error)
			}
			if primary == nil {
				primary = fmt.Errorf("incomplete resident index-query projection publication")
			}
			// This fence intentionally does not recover: typed/nil panic and Goexit
			// retain their original control flow while session retirement is scheduled.
			if c.projectionCause != nil {
				cause = s.v41DenseLeafCloseFailure(layer, "indexer.wq_b.weight", "readback", primary)
			} else {
				cause = closePublication("resident completion", primary)
			}
		}
	}()
	c.consume = func(t compute.Tensor, loan *v41ProjectionLoan) {
		loan.valid = valid
		if !valid() || len(loan.owned) != 2 || loan.owned[1].Buf() != t.Buf() {
			c.selected = true
			c.cause = errV41BF16ResidentScope
			return
		}
		image, c.selected, c.cause = compute.TryV41IndexQueryPublication(backend, t, loan.protected, loan.owned[0])
		if c.cause != nil && !c.selected {
			c.selected = true
		} // malformed optional contract is fatal
		if c.selected && c.cause == nil {
			if image == nil {
				c.cause = errV41BF16BoundaryResult
				return
			}
			c.raw = image.Raw()
			if v41HostBF16Witness.Load() {
				v41BF16PublishCalls.Add(1)
				v41BF16DeferredCalls.Add(1)
			}
		}
	}
	raw, outcome, err := p.core(layer, "indexer.wq_b.weight", input, width, in, 1, c)
	if c.panicPresent {
		panic(c.panicValue)
	}
	if err != nil || outcome == v41ProjectionError {
		if err == nil {
			err = errV41ProjectionResult
		}
		complete = true
		return nil, false, v41ProjectionOperationErr(layer, "indexer.wq_b.weight", err)
	}
	if c.cause != nil {
		cause = closePublication("grouped projection/publication transport", &V41BF16GroupedReadError{Layer: layer, Site: site, Cause: c.cause})
		complete = true
		return nil, true, cause
	}
	if c.cleanup.present {
		panic(c.cleanup.value)
	}
	if !valid() {
		cause = closePublication("resident scope", errV41BF16ResidentScope)
		complete = true
		return nil, true, cause
	}
	if outcome == v41ProjectionDeclined {
		if b.deviceOnly != nil {
			complete = true
			return nil, false, v41ProjectionOperationErr(layer, "indexer.wq_b.weight", b.deviceOnly.v41DeviceOnlyDeclined(layer, "dense-projection"))
		}
		values, err := m.v41ProjectionHostRows(req, input)
		complete = true
		return values, false, err
	}
	if outcome != v41ProjectionHandled {
		cause = closePublication("resident result", errV41ProjectionResult)
		complete = true
		return nil, true, cause
	}
	if !c.selected {
		complete = true
		return raw, false, nil
	}
	// The projection core performed its original complete raw finite scan and
	// shape checks before returning. No envelope interpretation happened earlier.
	if len(raw) != 8192 || !c.rawValidated {
		cause = closePublication("resident result", errV41ProjectionResult)
		complete = true
		return nil, true, cause
	}
	values, err := image.Published()
	if err == nil && (len(values) != 8192 || !v41BF16Widened(values)) {
		err = errV41BF16BoundaryResult
	}
	if err != nil {
		cause = closePublication("checked publication", err)
		complete = true
		return nil, true, cause
	}
	complete = true
	return values, true, nil
}
