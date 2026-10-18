package model

import (
	"fmt"
	"math"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41TryHeadScaleLoan is an opt-in, synchronous single-row Step composition.
// Both ordinary host reads remain. Projection owns every transient until the
// continuation returns; only the duplicate vocabulary-row upload disappears.
// No tensor or continuation is stored in Session state or returned to callers.
// The head timer excludes scale/read/postprocessing and scale-result retirement;
// its original deferred projection note still follows input/head retirement.
// Custom clock callbacks decline composition to preserve their call boundary.
func (m *Model) v41TryHeadScaleLoan(st *v41ForwardState, projection *v41ProjectionRouteBinding, input []float32, cfg Config) (raw, scaled []float32, selected bool, cause error) {
	if os.Getenv("FAK_V41_HEAD_SCALE_LOAN") != "1" || st == nil || st.dsparkCapture != nil || projection == nil || m == nil || m.lora != nil || m.prism != nil || v41ActivationTraceEnvPath() != "" {
		return
	}
	factor := v41LogitFactor(cfg)
	b, p := st.logitScaleBinding, projection.nativeProjection
	if b == nil || b.owner == nil || b.operation == nil || p.core == nil || p.owner != b.owner || p.model != m || b.model != m || !backendHasComparableIdentity(b.backend) || p.backend != b.backend || !finite32(factor) || factor == 1 {
		return
	}
	s := b.owner
	state, scope, epoch, backend := s.v41Forward, s.backendExecutionScope, s.cacheGeometryEpoch, b.backend
	valid := func() bool {
		if state == nil || scope == nil || !scope.live || scope.owner == nil || s.M != m || s.Backend != backend || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || s.v41Forward != state || state.callbackOwner != s || st.callbackOwner != s || state.projectionRoute != projection || st.projectionRoute != projection || state.logitScaleBinding != b || st.logitScaleBinding != b || s.backendExecutionScope != scope || s.cacheGeometryEpoch != epoch || m.lora != nil || m.prism != nil {
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
		return
	}
	// A custom clock is an observable callback at the ordinary projection
	// boundary. Preserve that boundary by declining this experimental loan.
	if ledger := v41ExpertFaultLedgerOf(m, false); ledger != nil {
		ledger.mu.Lock()
		customClock := ledger.nowNanos != nil
		ledger.mu.Unlock()
		if customClock {
			return
		}
	}
	if _, err := m.v41AdmitProjectionRows(-1, m.headName(), input, cfg.VocabSize, cfg.HiddenSize, 1); err != nil {
		return nil, nil, true, err
	}
	selected = true
	c := &v41ProjectionContinuation{headScale: true, valid: valid}
	stage := "resident input"
	closeScale := func(primary error) error {
		if s.halFailure != nil {
			return s.halFailure
		}
		failure := &BackendForwardOperationError{Backend: backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-logit-scale", Layer: -1, Stage: stage, Cause: &V41LogitScaleOperationError{Cause: primary}}
		s.halFailure = failure
		if !s.halClosed {
			func() { defer func() { s.halClosed = true; _ = recover() }(); s.closeBackendOwned() }()
		}
		return failure
	}
	complete := false
	defer func() {
		if !complete {
			raw, scaled = nil, nil
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
				primary = fmt.Errorf("incomplete retained head-scale continuation")
			}
			// Do not recover: panic(nil) and Goexit keep their original control flow.
			if c.projectionCause != nil {
				cause = s.v41HeadCloseFailure("readback", primary)
			} else {
				cause = closeScale(primary)
			}
		}
	}()
	c.consume = func(t compute.Tensor, loan *v41ProjectionLoan) {
		loan.valid = valid
		c.selected = true
		if !valid() || !c.rawValidated || len(c.raw) != cfg.VocabSize || len(loan.owned) != 2 || loan.owned[1].Buf() != t.Buf() {
			c.cause = errV41LogitScaleResult
			return
		}
		// The projection loan registers the fresh scale output immediately.
		// Detach and retire only that tail here, inside the timed continuation;
		// the projection remains sole owner of its input and head output.
		defer func() {
			if len(loan.owned) > 2 {
				tail := &v41ProjectionLoan{backend: backend, owned: loan.owned[2:]}
				loan.owned = loan.owned[:2]
				tail.releaseIndexCleanup(&c.scaleCleanup)
			}
		}()
		borrowed := func(values []float32, scale float32) ([]float32, error) {
			if !valid() || len(values) != cfg.VocabSize || scale != factor {
				return nil, errV41LogitScaleResult
			}
			stage = "multiply"
			out, err := b.operation.V41LogitScale(t, scale)
			fresh := out.Buf() != nil && out.Backend() == backend
			if fresh {
				for _, resident := range s.halW {
					if resident.Buf() == out.Buf() {
						fresh = false
						break
					}
				}
			}
			if fresh {
				fresh = loan.own(out)
			}
			if err != nil {
				return nil, err
			}
			if !fresh || !valid() || out.Dtype != compute.F32 || out.Layout != compute.RowMajor || out.Quant != nil || len(out.Shape) != 1 || out.Shape[0] != len(values) {
				return nil, errV41LogitScaleResult
			}
			stage = "multiply readback"
			read := backend.Read(out)
			if !valid() || len(read) != len(values) {
				return nil, errV41LogitScaleResult
			}
			for _, value := range read {
				if math.IsNaN(float64(value)) {
					return nil, errV41LogitScaleResult
				}
			}
			return append([]float32(nil), read...), nil
		}
		scaled, c.cause = v41ScaleHeadLogits(c.raw, cfg, borrowed, st.logitHostRefusal())
		c.completed = c.cause == nil
	}
	var outcome v41DenseProjectionOutcome
	raw, outcome, cause = p.core(-1, m.headName(), input, cfg.VocabSize, cfg.HiddenSize, 1, c)
	if c.panicPresent {
		panic(c.panicValue)
	}
	if cause != nil || outcome == v41ProjectionError {
		if cause == nil {
			cause = errV41ProjectionResult
		}
		complete = true
		return nil, nil, true, v41ProjectionOperationErr(-1, m.headName(), cause)
	}
	if c.cause != nil {
		cause = closeScale(c.cause)
		complete = true
		return nil, nil, true, cause
	}
	// Scale result retires first; preserve its earlier cleanup fault.
	if c.scaleCleanup.present {
		panic(c.scaleCleanup.value)
	}
	if c.cleanup.present {
		panic(c.cleanup.value)
	}
	if !valid() {
		cause = closeScale(errV41RouterProjectionScope)
		complete = true
		return nil, nil, true, cause
	}
	if outcome == v41ProjectionDeclined {
		complete = true
		return nil, nil, false, nil
	}
	if outcome != v41ProjectionHandled || !c.rawValidated || !c.completed || len(raw) != cfg.VocabSize || len(scaled) != len(raw) {
		cause = closeScale(errV41LogitScaleResult)
		complete = true
		return nil, nil, true, cause
	}
	complete = true
	return raw, scaled, true, nil
}
