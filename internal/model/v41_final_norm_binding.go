package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Ordinary callback replacement creates a new immutable product without native
// authority. Snapshot continuation data never owns this session-local binding.
type v41FinalNormBinding struct {
	normalize   v41FinalNormFunc
	round       v41BF16BoundaryFunc
	owner       *Session
	model       *Model
	backend     compute.Backend
	publication *v41BF16PublicationBinding
}

func (b *v41FinalNormBinding) normFunc() v41FinalNormFunc {
	if b == nil {
		return nil
	}
	return b.normalize
}
func (b *v41FinalNormBinding) roundFunc() v41BF16BoundaryFunc {
	if b == nil {
		return nil
	}
	return b.round
}
func (st *v41ForwardState) finalNorm() v41FinalNormFunc {
	return st.headNorm.normFunc()
}
func (st *v41ForwardState) bf16Boundary() v41BF16BoundaryFunc {
	return st.headNorm.roundFunc()
}
func (st *v41ForwardState) setFinalNorm(f v41FinalNormFunc) {
	st.headNorm = &v41FinalNormBinding{normalize: f, round: st.bf16Boundary()}
}
func (st *v41ForwardState) setBF16Boundary(f v41BF16BoundaryFunc) {
	st.headNorm = &v41FinalNormBinding{normalize: st.finalNorm(), round: f}
}

// Only the ordinary publication path receives composition authority. Isolated
// checked/raw BF16 witnesses retain their original callbacks and boundaries.
func (s *Session) v41BindFinalNorm(b *v41FinalNormBinding) *v41FinalNormBinding {
	if b == nil || b.round != nil || s.v41Forward == nil {
		return b
	}
	ops := s.v41Forward.gatherOps
	b = &v41FinalNormBinding{normalize: b.normalize, round: ops.bf16PublishFunc()}
	if b.normalize == nil || ops == nil {
		return b
	}
	pub := ops.bf16Publish
	if pub != nil && pub.core != nil && pub.owner == s && pub.model == s.M &&
		backendHasComparableIdentity(s.Backend) && pub.backend == s.Backend {
		b.owner, b.model, b.backend, b.publication = s, s.M, s.Backend, pub
	}
	return b
}

// A selected chain retains four transient F32 rows and two 16-byte statuses.
// The cached immutable gain is additional and must already exist. This bound
// covers the current 5120-wide model and smaller full-geometry software fixtures.
const v41FinalNormWidthLimit = 5120

func (b *v41FinalNormBinding) try(m *Model, st *v41ForwardState, input []float32) (result []float32, selected bool, cause error) {
	if b == nil || b.owner == nil || m == nil || st == nil ||
		len(input) == 0 || len(input) != m.Cfg.HiddenSize || len(input) > v41FinalNormWidthLimit ||
		m.Cfg.LayerNorm || m.Cfg.NormGain1p || m.lora != nil || m.prism != nil ||
		v41ActivationTraceEnvPath() != "" {
		return nil, false, nil
	}
	full, err := v41ForwardGeometry(m.Cfg)
	eps := float32(m.Cfg.RMSNormEps)
	if err != nil || !full || !finite32(eps) || eps <= 0 {
		return nil, false, nil
	}
	if _, bad := v41BF16PublishFault(input); bad {
		return nil, false, nil
	}
	s := b.owner
	state, scope, epoch := s.v41Forward, s.backendExecutionScope, s.cacheGeometryEpoch
	valid := func() bool {
		if b.model != m || s.M != m || !backendHasComparableIdentity(s.Backend) || s.Backend != b.backend ||
			s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || state == nil || s.v41Forward != state ||
			state.headNorm != b || st.headNorm != b || st.callbackOwner != s || state.callbackOwner != s ||
			state.gatherOps == nil || st.gatherOps != state.gatherOps || state.gatherOps.owner != s ||
			state.gatherOps.bf16Publish != b.publication || scope == nil || !scope.live || scope.owner == nil ||
			s.backendExecutionScope != scope || s.cacheGeometryEpoch != epoch {
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
		return nil, false, nil
	}
	const leaf = "model.norm.weight"
	gain, present := s.halW[leaf]
	if !present || gain.Buf() == nil || gain.Backend() != b.backend || gain.Dtype != compute.F32 ||
		gain.Layout != compute.RowMajor || gain.Quant != nil || len(gain.Shape) != 1 || gain.Shape[0] != len(input) {
		return nil, false, nil
	}
	meta, present := m.manifest[leaf]
	if !present || len(meta.Shape) != 1 || meta.Shape[0] != len(input) {
		return nil, false, nil
	}
	hostGain := m.tensor(leaf)
	if len(hostGain) != len(input) {
		return nil, false, nil
	}
	for _, v := range hostGain {
		if !finite32(v) {
			return nil, false, nil
		}
	}
	if !valid() {
		return nil, false, nil
	}

	selected = true
	stage, site := "deferred capability", leaf+" input"
	closeFailure := func(err error) error {
		if s.halFailure != nil {
			s.v41CloseSelectedBF16Failure(s.halFailure)
			return s.halFailure
		}
		if site != "" {
			closed := s.v41CloseBF16PublicationFailure(b.backend, -1, site, stage, err)
			return &V41BF16BoundaryOperationError{Layer: -1, Site: site, Cause: closed}
		}
		closed := &BackendForwardOperationError{Backend: b.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-final-norm", Layer: -1, Stage: stage, Cause: err}
		s.v41CloseSelectedBF16Failure(closed)
		return s.halFailure
	}
	complete := false
	var primary any
	defer func() {
		if !complete {
			result = nil
			if cause == nil {
				cause, _ = primary.(error)
			}
			if cause == nil {
				cause = fmt.Errorf("incomplete final normalization chain")
			}
			cause = closeFailure(cause)
		}
	}()
	run := func() (out []float32, cause error) {
		// The ordinary callback treats a capability-probe panic as a fatal
		// backend failure. Keep that probe inside the same completion boundary.
		deferred, ok := v41DeferredBF16Publisher(b.backend, len(input))
		if !ok {
			selected = false
			return nil, nil
		}
		if !valid() {
			return nil, errV41ProjectionResult
		}
		stage = "input upload"
		owned := make([]compute.Tensor, 0, 6)
		finished := false
		// Do not recover the primary unwind here: panic(nil) under the legacy
		// runtime mode and Goexit must reach the outer completion boundary.
		defer func() {
			cleanup := func() (failure backendCleanupPanic) {
				defer func() { owned = nil }()
				// Register all releases before any can fail or call Goexit.
				for _, tensor := range owned {
					defer func(t compute.Tensor) {
						returned := false
						defer func() {
							value := recover()
							if !returned && !failure.present {
								failure.value, failure.present = value, true
							}
						}()
						b.backend.Free(t)
						returned = true
					}(tensor)
				}
				return
			}()
			if finished && cause == nil && cleanup.present {
				site, stage = "", "grouped final-norm cleanup"
				panic(cleanup.value)
			}
		}()
		body := func() ([]float32, error) {
			// Track fresh allocations even when their metadata is malformed.
			own := func(t compute.Tensor) bool {
				if t.Buf() == nil || t.Backend() != b.backend || t.Buf() == gain.Buf() {
					return false
				}
				for _, old := range owned {
					if old.Buf() == t.Buf() {
						return false
					}
				}
				owned = append(owned, t)
				return true
			}
			row := func(t compute.Tensor) bool {
				return t.Dtype == compute.F32 && t.Layout == compute.RowMajor && t.Quant == nil &&
					len(t.Shape) == 1 && t.Shape[0] == len(input)
			}
			statusOK := func(st compute.V41DeferredStatus, site uint32) bool {
				t := st.Record
				return st.Kind == compute.V41DeferredBF16Publish && st.Count == len(input) &&
					st.Site == site && compute.V41DeferredRecords(st) == 1 &&
					t.Dtype == compute.F32 && t.Layout == compute.RowMajor && t.Quant == nil &&
					len(t.Shape) == 1 && t.Shape[0] == 4
			}
			x := uploadHostF32Class(b.backend, []int{len(input)}, input, compute.MemoryActivation, "V4.1 BF16 publication input")
			if !own(x) || !row(x) {
				return nil, errV41BF16BoundaryResult
			}
			stage = "checked publication"
			rounded, inputStatus, err := deferred.V41BF16PublishDeferred(x, compute.V41BF16PublishFinalNormInput)
			roundedFresh, inputStatusFresh := own(rounded), own(inputStatus.Record)
			if err != nil {
				return nil, err
			}
			if !roundedFresh || !inputStatusFresh || !row(rounded) || !statusOK(inputStatus, compute.V41BF16PublishFinalNormInput) {
				return nil, errV41BF16BoundaryResult
			}
			// Count each validated dispatch even if a later operation/read fails.
			if v41HostBF16Witness.Load() {
				v41BF16PublishCalls.Add(1)
				v41BF16DeferredCalls.Add(1)
			}
			if !valid() {
				return nil, errV41ProjectionResult
			}
			site, stage = "", "rmsnorm"
			raw := b.backend.RMSNorm(rounded, gain, eps)
			if !own(raw) || !row(raw) {
				return nil, errV41ProjectionResult
			}
			if !valid() {
				return nil, errV41ProjectionResult
			}
			site, stage = leaf+" output", "checked publication"
			published, outputStatus, err := deferred.V41BF16PublishDeferred(raw, compute.V41BF16PublishFinalNormOutput)
			publishedFresh, outputStatusFresh := own(published), own(outputStatus.Record)
			if err != nil {
				return nil, err
			}
			if !publishedFresh || !outputStatusFresh || !row(published) || !statusOK(outputStatus, compute.V41BF16PublishFinalNormOutput) {
				return nil, errV41BF16BoundaryResult
			}
			// Count each validated dispatch even if a later operation/read fails.
			if v41HostBF16Witness.Load() {
				v41BF16PublishCalls.Add(1)
				v41BF16DeferredCalls.Add(1)
			}
			if !valid() {
				return nil, errV41ProjectionResult
			}
			// Both status records are consumed by the grouped reader, including
			// transport failure. Remove them before entering that boundary.
			owned = []compute.Tensor{x, rounded, raw, published}
			site, stage = "", "grouped final-norm readback"
			verdicts, rows, err := deferred.V41ReadDeferred(
				[]compute.V41DeferredStatus{inputStatus, outputStatus},
				[]compute.Tensor{raw, published})
			if err != nil {
				return nil, err
			}
			if len(verdicts) != 2 || len(rows) != 2 || !valid() {
				return nil, errV41ProjectionResult
			}
			site, stage = leaf+" input", "checked publication"
			if verdicts[0] != nil {
				return nil, verdicts[0]
			}
			site, stage = "", "readback"
			if len(rows[0]) != len(input) {
				return nil, errV41ProjectionResult
			}
			for _, v := range rows[0] {
				if !finite32(v) {
					return nil, errV41ProjectionResult
				}
			}
			site, stage = leaf+" output", "checked publication"
			if verdicts[1] != nil {
				return nil, verdicts[1]
			}
			stage = "checked publication readback"
			if len(rows[1]) != len(input) || !v41BF16Widened(rows[1]) {
				return nil, errV41BF16BoundaryResult
			}
			if !valid() {
				return nil, errV41ProjectionResult
			}
			return append([]float32(nil), rows[1]...), nil
		}
		out, cause = body()
		finished = true
		return
	}
	returned := false
	func() {
		defer func() {
			if !returned {
				primary = recover()
			}
		}()
		result, cause = run()
		returned = true
	}()
	if !returned {
		if err, ok := primary.(error); ok {
			var prior *BackendForwardOperationError
			if errors.As(err, &prior) {
				s.v41CloseSelectedBF16Failure(err)
				cause, result, complete = s.halFailure, nil, true
				panic(primary)
			}
			var backendErr *compute.BackendError
			if errors.As(err, &backendErr) {
				result, cause, complete = nil, closeFailure(err), true
				return
			}
		}
		if err, ok := compute.ConvertCUDAPanic(primary, "", ""); ok {
			if original, ok := primary.(error); ok {
				err = original
			}
			result, cause, complete = nil, closeFailure(err), true
			return
		}
		panic(primary)
	}
	if cause != nil {
		result, cause = nil, closeFailure(cause)
	}
	complete = true
	return
}
