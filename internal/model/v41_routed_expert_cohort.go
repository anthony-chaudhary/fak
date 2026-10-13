package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const v41ExpertCohortRows = 32

// v41RoutedExpertCohort keeps the existing scalar kernels and the expert-major
// row order. Only the status/output fence is shared. The ID panel's explicit
// opt-in takes precedence; this path does not select an ID-table kernel.
func (s *Session) v41RoutedExpertCohort(layer int, stem string, inputs [][]float32, weights []float32) (result [][]float32, outcome v41RoutedExpertOutcome, cause error) {
	if s == nil || s.M == nil || s.Backend == nil || s.v41RoutedExpertFunc() == nil ||
		v41PrefillExpertPanelOptedIn() || len(inputs) < 2 || len(inputs) > v41ExpertCohortRows || len(weights) != len(inputs) {
		return nil, v41RoutedExpertDeclined, nil
	}
	H, I := s.M.Cfg.HiddenSize, s.M.Cfg.MoEIntermediateSize
	deferred, ok := v41DeferredBF16Publisher(s.Backend, H)
	publication, havePublication := s.Backend.(compute.V41BF16PublishBackend)
	if !ok || !deferred.SupportsV41DeferredStatus(I) || !havePublication || !publication.SupportsV41BF16Publish(H) {
		return nil, v41RoutedExpertDeclined, nil
	}
	// Admission may decline, never report a later row's payload fault ahead of
	// the scalar path. Every admitted input publication is already the identity.
	for r, input := range inputs {
		if len(input) != H || !v41BF16Widened(input) || !finite32(weights[r]) {
			return nil, v41RoutedExpertDeclined, nil
		}
	}
	gateName, upName := layerName(layer, stem+".w1.weight"), layerName(layer, stem+".w3.weight")
	downName := layerName(layer, stem+".w2.weight")
	if s.M.has(downName[:len(downName)-len("weight")] + "bias") {
		return nil, v41RoutedExpertDeclined, nil
	}
	s.ensureOpenBackendSession()
	stage := "gate/up"
	retire := func(err error) error {
		if s.halFailure == nil {
			s.halFailure = &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-expert", Layer: layer, Stage: stage, Cause: err}
		}
		if !s.halClosed {
			func() {
				defer func() { s.halClosed = true; _ = recover() }()
				s.closeBackendOwned()
			}()
		}
		return s.halFailure
	}
	defer func() {
		if p := recover(); p != nil {
			if err, ok := p.(error); ok {
				var backend *compute.BackendError
				var closed *BackendForwardOperationError
				if errors.As(err, &closed) {
					retire(err)
					panic(p)
				}
				if errors.As(err, &backend) {
					result, outcome, cause = nil, v41RoutedExpertError, retire(err)
					return
				}
			}
			if converted, ok := compute.ConvertCUDAPanic(p, "", ""); ok {
				if original, ok := p.(error); ok {
					converted = original
				}
				result, outcome, cause = nil, v41RoutedExpertError, retire(converted)
				return
			}
			err, ok := p.(error)
			if !ok {
				err = fmt.Errorf("unclassified backend panic: %v", p)
			}
			retire(err)
			panic(p)
		}
	}()
	gate, releaseGate, up, releaseUp, _, admitted := expertInputDeviceAdmitted(s, gateName, upName, inputs[0], H)
	if !admitted {
		return nil, v41RoutedExpertDeclined, nil
	}
	defer releaseGate()
	defer releaseUp()
	down, releaseDown, admitted := s.q4kExpertDownDeviceWeight(downName)
	if !admitted {
		return nil, v41RoutedExpertDeclined, nil
	}
	defer releaseDown()
	outcome = v41RoutedExpertError
	recordingFailed := false
	result, cause = s.v41RunExpertCohort(deferred, gate, up, down, inputs, weights, float32(s.M.Cfg.SwigluLimit), layer, &stage, &recordingFailed)
	if cause != nil {
		var numeric *V41ExpertOperationError
		if !errors.As(cause, &numeric) {
			cause = retire(cause)
		} else if recordingFailed {
			// The earlier numeric verdict remains the returned primary, but a
			// speculative selected failure still makes this session unusable.
			retire(cause)
		}
		return nil, outcome, cause
	}
	return result, v41RoutedExpertHandled, nil
}

// A row's deferred arithmetic verdict is ordered before a later row's recording
// failure. Cleanup retains the scalar function's LIFO order within each row:
// a same-row cleanup panic wins, but speculative later-row cleanup cannot hide
// an earlier row's primary. Transport failure leaves every verdict unknown, as
// specified by V41ReadDeferred; no failed operation is replayed.
func (s *Session) v41RunExpertCohort(deferred compute.V41DeferredStatusBackend, gateW, upW, downW compute.Tensor, inputs [][]float32, weights []float32, limit float32, layer int, stage *string, recordingFailed *bool) (result [][]float32, cause error) {
	H, I, count := s.M.Cfg.HiddenSize, s.M.Cfg.MoEIntermediateSize, len(inputs)
	owned := make([][]compute.Tensor, count)
	seen := map[compute.Buffer]bool{gateW.Buf(): true, upW.Buf(): true, downW.Buf(): true}
	for _, weight := range s.halW {
		if weight.Buf() != nil {
			seen[weight.Buf()] = true
		}
	}
	consumed := map[compute.Buffer]bool{}
	var statuses []compute.V41DeferredStatus
	var statusRows []int
	var published []compute.Tensor
	faultRow, current := count, 0
	var primary any
	valid := func(t compute.Tensor, n int) bool {
		return t.Buf() != nil && t.Dtype == compute.F32 && t.Layout == compute.RowMajor && t.Quant == nil && len(t.Shape) == 1 && t.Shape[0] == n
	}
	own := func(t compute.Tensor) bool {
		if t.Buf() == nil || t.Backend() != gateW.Backend() || seen[t.Buf()] {
			return false
		}
		seen[t.Buf()] = true
		owned[current] = append(owned[current], t)
		return true
	}
	// Always attempt every owned release. The last panic in a row's reverse
	// traversal is what nested Go defers in the scalar owner would propagate.
	defer func() {
		for r := range owned {
			var cleanup any
			for i := len(owned[r]) - 1; i >= 0; i-- {
				t := owned[r][i]
				if consumed[t.Buf()] {
					continue
				}
				func() {
					defer func() {
						if p := recover(); p != nil {
							cleanup = p
						}
					}()
					s.Backend.Free(t)
				}()
			}
			if cleanup != nil {
				*recordingFailed = true
				if r <= faultRow {
					faultRow, cause, primary = r, nil, cleanup
				}
			}
		}
		if primary != nil {
			panic(primary)
		}
		if cause != nil {
			result = nil
		}
	}()
	record := func(t compute.Tensor, status compute.V41DeferredStatus, n int, kind compute.V41DeferredStatusKind, site uint32, err error) error {
		fresh, recordFresh := own(t), own(status.Record)
		recordOK := recordFresh && valid(status.Record, 4) && status.Kind == kind && status.Count == n && status.Site == site && status.Records >= 0 && status.Records <= 1
		if err != nil {
			// A record returned with a selected operation error remains owned
			// for cleanup, but cannot replace that same operation's error.
			return err
		}
		if recordOK {
			statuses = append(statuses, status)
			statusRows = append(statusRows, current)
		}
		if !fresh || !valid(t, n) || !recordOK {
			return errV41ExpertResult
		}
		return nil
	}
	opened := s.M.v41NowNanos()
	func() {
		defer func() {
			if p := recover(); p != nil {
				primary, faultRow = p, current
				*recordingFailed = true
			}
		}()
		for current = 0; current < count; current++ {
			*stage = "gate/up"
			x := s.uploadHostF32([]int{H}, inputs[current], compute.MemoryActivation, "moe expert gate/up activation")
			if !own(x) || !valid(x, H) {
				cause = errV41ExpertResult
				break
			}
			g := s.Backend.MatMul(gateW, x)
			if !own(g) || !valid(g, I) {
				cause = errV41ExpertResult
				break
			}
			u := s.Backend.MatMul(upW, x)
			if !own(u) || !valid(u, I) {
				cause = errV41ExpertResult
				break
			}
			*stage = "activation"
			a, st, err := deferred.V41RoutedSwiGLUDeferred(g, u, limit, weights[current])
			if cause = record(a, st, I, compute.V41DeferredRoutedSwiGLU, 0, err); cause != nil {
				break
			}
			*stage = "down"
			d := s.Backend.MatMul(downW, a)
			if !own(d) || !valid(d, H) {
				cause = errV41ExpertResult
				break
			}
			p, st, err := deferred.V41BF16PublishDeferred(d, compute.V41BF16PublishRoutedDown)
			if cause = record(p, st, H, compute.V41DeferredBF16Publish, compute.V41BF16PublishRoutedDown, err); cause != nil {
				break
			}
			published = append(published, p)
		}
		if cause != nil {
			faultRow = current
			*recordingFailed = true
		}
	}()
	if len(statuses) == 0 {
		return nil, cause
	}
	readRows := published
	if cause != nil || primary != nil {
		readRows = nil
	}
	for _, st := range statuses {
		consumed[st.Record.Buf()] = true
	}
	var verdicts []error
	var values [][]float32
	var readErr error
	func() {
		defer func() {
			if p := recover(); p != nil {
				primary, cause, faultRow, *stage = p, nil, statusRows[0], "readback"
			}
		}()
		verdicts, values, readErr = deferred.V41ReadDeferred(statuses, readRows)
	}()
	if readErr != nil {
		cause, primary, faultRow, *stage = readErr, nil, statusRows[0], "readback"
		return nil, cause
	}
	if len(verdicts) != len(statuses) || len(values) != len(readRows) {
		if cause == nil && primary == nil {
			cause, faultRow, *stage = errV41ExpertResult, 0, "readback"
		}
		return nil, cause
	}
	for i, verdict := range verdicts {
		if verdict == nil || statusRows[i] > faultRow {
			continue
		}
		r := statusRows[i]
		*stage = "activation"
		var activation *compute.V41RoutedSwiGLUFault
		var publication *compute.V41BF16CheckedArithmeticError
		if errors.As(verdict, &activation) {
			name := "gate/up"
			if activation.Stage == compute.V41RoutedSwiGLUStageDownOperand {
				name = "down"
			}
			verdict = v41ExpertOperationErr(layer, name, errV41ExpertResult)
		} else if statuses[i].Kind == compute.V41DeferredBF16Publish {
			*stage = "down"
			if errors.As(verdict, &publication) {
				verdict = v41ExpertOperationErr(layer, "down", errV41ExpertResult)
			}
		}
		faultRow, cause, primary = r, verdict, nil
		break
	}
	if cause != nil || primary != nil {
		return nil, cause
	}
	result = make([][]float32, count)
	for r, row := range values {
		if len(row) != H || !v41BF16Widened(row) {
			faultRow, *stage = r, "readback"
			return nil, errV41ExpertResult
		}
		result[r] = append([]float32(nil), row...)
		s.M.v41NoteExpertActivation(1, 0, 16, v41FirstOpened(r, opened))
	}
	return result, nil
}
