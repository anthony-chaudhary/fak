package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const v41ExpertCohortRows = 32

// v41RoutedExpertCohort keeps the expert-major row order and shares the final
// status/output fence. An explicitly available Q2_K token tile may share only
// gate/up; scalar activation/down retain their row order. The ID panel's
// explicit opt-in takes precedence; this path selects no ID-table kernel.
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
	releaseResident := s.v41HoldResidentExpertTriple(gateName, upName, downName)
	defer releaseResident()
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
	return s.v41RunExpertCohortSelected(deferred, gateW, upW, downW, inputs, weights, limit, layer, stage, recordingFailed, nil)
}

// Only the owned decode adapter supplies options. Its prepared triples and sum
// indices are validated before selection; the cold-prefill wrapper stays nil.
type v41ExpertCohortOptions struct {
	triples   [][3]compute.Tensor
	borrowed  []compute.Tensor
	inputKeys []int
	sums      [][]int
	note      func(row int, gateUp bool)
}

func (s *Session) v41RunExpertCohortSelected(deferred compute.V41DeferredStatusBackend, gateW, upW, downW compute.Tensor, inputs [][]float32, weights []float32, limit float32, layer int, stage *string, recordingFailed *bool, options *v41ExpertCohortOptions) (result [][]float32, cause error) {
	H, I, count := s.M.Cfg.HiddenSize, s.M.Cfg.MoEIntermediateSize, len(inputs)
	owners := count
	if options != nil && options.sums != nil {
		owners++ // Sum temporaries follow every expert row in the fault ledger.
	}
	owned := make([][]compute.Tensor, owners)
	seen := map[compute.Buffer]bool{gateW.Buf(): true, upW.Buf(): true, downW.Buf(): true}
	if options != nil {
		for _, triple := range options.triples {
			for _, weight := range triple {
				seen[weight.Buf()] = true
			}
		}
		for _, weight := range options.borrowed {
			if weight.Buf() != nil {
				seen[weight.Buf()] = true
			}
		}
	}
	for _, weight := range s.halW {
		if weight.Buf() != nil {
			seen[weight.Buf()] = true
		}
	}
	consumed := map[compute.Buffer]bool{}
	var statuses []compute.V41DeferredStatus
	var statusRows []int
	var published []compute.Tensor
	var sums []compute.Tensor
	var downTile *v41ExpertDownTile
	uploads := map[int]compute.Tensor{}
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
	publishDown := func(d compute.Tensor) error {
		p, st, err := deferred.V41BF16PublishDeferred(d, compute.V41BF16PublishRoutedDown)
		if err = record(p, st, H, compute.V41DeferredBF16Publish, compute.V41BF16PublishRoutedDown, err); err != nil {
			return err
		}
		published = append(published, p)
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
		var tile *v41ExpertTile
		if options == nil {
			// Only this caller promises one common expert triple. Cross-lane
			// prepared rows may carry distinct experts and stay unchanged.
			tile = newV41ExpertTile(s.Backend, gateW, upW, count)
			downTile = newV41ExpertDownTile(s.Backend, downW, count)
		}
		var activations []compute.Tensor
		chunkStart := 0
	recordRows:
		for current = 0; current < count; current++ {
			*stage = "gate/up"
			down := downW
			g, u, tiled, err := tile.project(s, inputs, current, H, I, own, valid)
			if cause = err; cause != nil {
				break
			}
			if !tiled {
				x := compute.Tensor{}
				if options != nil {
					x = uploads[options.inputKeys[current]]
				}
				if x.Buf() == nil {
					x = s.uploadHostF32([]int{H}, inputs[current], compute.MemoryActivation, "moe expert gate/up activation")
					if !own(x) || !valid(x, H) {
						cause = errV41ExpertResult
						break
					}
					if options != nil {
						uploads[options.inputKeys[current]] = x
					}
				}
				gate, up := gateW, upW
				if options != nil {
					gate, up, down = options.triples[current][0], options.triples[current][1], options.triples[current][2]
					if options.note != nil {
						options.note(current, true)
					}
				}
				g = s.Backend.MatMul(gate, x)
				if !own(g) || !valid(g, I) {
					cause = errV41ExpertResult
					break
				}
				u = s.Backend.MatMul(up, x)
				if !own(u) || !valid(u, I) {
					cause = errV41ExpertResult
					break
				}
			}
			*stage = "activation"
			a, st, err := deferred.V41RoutedSwiGLUDeferred(g, u, limit, weights[current])
			if cause = record(a, st, I, compute.V41DeferredRoutedSwiGLU, 0, err); cause != nil {
				break
			}
			*stage = "down"
			if downTile != nil && count-chunkStart > 1 {
				activations = append(activations, a)
				if len(activations) < v41ExpertTileSize(count-chunkStart) {
					continue
				}
				last := current
				panel, err := downTile.project(activations, I, H, own, valid)
				if cause = err; cause != nil {
					break
				}
				for row := chunkStart; row <= last; row++ {
					current = row
					d, err := downTile.row(panel, row-chunkStart, H, own, valid)
					if cause = err; cause != nil {
						break recordRows
					}
					if cause = publishDown(d); cause != nil {
						break recordRows
					}
				}
				current, chunkStart, activations = last, last+1, activations[:0]
				continue
			}
			if options != nil && options.note != nil {
				options.note(current, false)
			}
			d := s.Backend.MatMul(down, a)
			if !own(d) || !valid(d, H) {
				cause = errV41ExpertResult
				break
			}
			if cause = publishDown(d); cause != nil {
				break
			}
		}
		if cause == nil && options != nil && options.sums != nil {
			*stage = "routed-sum"
			for _, order := range options.sums {
				acc := s.uploadHostF32([]int{H}, make([]float32, H), compute.MemoryActivation, "V4.1 routed-sum accumulator")
				if !own(acc) || !valid(acc, H) {
					cause = errV41ExpertResult
					break
				}
				v41DeviceAccumulate(s.Backend, acc, published, order)
				sums = append(sums, acc)
			}
		}
		if cause != nil {
			faultRow = current
			*recordingFailed = true
		}
	}()
	if len(statuses) == 0 {
		return nil, cause
	}
	if downTile != nil {
		// Tile scheduling records all activation statuses before its down
		// statuses. Decode the actually issued records in stable row/stage
		// order; never create a record for a down that was not issued.
		for i := 1; i < len(statuses); i++ {
			for j := i; j > 0 && statusRows[j] < statusRows[j-1]; j-- {
				statuses[j], statuses[j-1] = statuses[j-1], statuses[j]
				statusRows[j], statusRows[j-1] = statusRows[j-1], statusRows[j]
			}
		}
	}
	readRows := published
	if options != nil && options.sums != nil {
		readRows = sums
	}
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
		if verdict == nil || downTile == nil && statusRows[i] > faultRow {
			continue
		}
		// Under tiled scheduling every retained status was issued before a
		// recording failure, even if its row follows the failed down row.
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
	result = make([][]float32, len(values))
	for r, row := range values {
		validRow := len(row) == H
		if options != nil && options.sums != nil {
			for _, v := range row {
				validRow = validRow && finite32(v)
			}
		} else {
			validRow = validRow && v41BF16Widened(row)
		}
		if !validRow {
			faultRow, *stage = r, "readback"
			if options != nil && options.sums != nil {
				faultRow, *recordingFailed = count, true
				return nil, v41ExpertOperationErr(layer, "sum", errV41ExpertResult)
			}
			return nil, errV41ExpertResult
		}
		result[r] = append([]float32(nil), row...)
		if options == nil || options.sums == nil {
			s.M.v41NoteExpertActivation(1, 0, 16, v41FirstOpened(r, opened))
		}
	}
	if options != nil && options.sums != nil {
		for r := 0; r < count; r++ {
			s.M.v41NoteExpertActivation(1, 0, 16, v41FirstOpened(r, opened))
		}
	}
	return result, nil
}
