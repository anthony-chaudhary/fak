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
func (s *Session) v41RoutedExpertCohort(layer int, stem string, inputs [][]float32, weights []float32) ([][]float32, v41RoutedExpertOutcome, error) {
	return s.v41RoutedExpertCohortRetained(layer, stem, inputs, weights, nil)
}

// A retained call transfers only checked publications. The layer owner remains
// responsible even if a later cohort-temporary or weight-release cleanup fails.
func (s *Session) v41RoutedExpertCohortRetained(layer int, stem string, inputs [][]float32, weights []float32, handoff *v41RoutedOutputHandoff) (result [][]float32, outcome v41RoutedExpertOutcome, cause error) {
	if handoff != nil && (!handoff.owner.live() || handoff.owner.session != s || handoff.owner.layer != layer || len(handoff.rows) != len(inputs)) {
		return nil, v41RoutedExpertDeclined, nil
	}
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
		if handoff != nil {
			// Retire only after all retained rows have had a release attempt.
			// The selected compute/validation fault remains primary.
			handoff.owner.release(0, true)
		}
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
	result, cause = s.v41RunExpertCohortRetained(deferred, gate, up, down, inputs, weights, float32(s.M.Cfg.SwigluLimit), layer, &stage, &recordingFailed, nil, handoff)
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

func (s *Session) v41RunExpertCohortSelected(deferred compute.V41DeferredStatusBackend, gateW, upW, downW compute.Tensor, inputs [][]float32, weights []float32, limit float32, layer int, stage *string, recordingFailed *bool, options *v41ExpertCohortOptions) ([][]float32, error) {
	return s.v41RunExpertCohortRetained(deferred, gateW, upW, downW, inputs, weights, limit, layer, stage, recordingFailed, options, nil)
}

func (s *Session) v41RunExpertCohortRetained(deferred compute.V41DeferredStatusBackend, gateW, upW, downW compute.Tensor, inputs [][]float32, weights []float32, limit float32, layer int, stage *string, recordingFailed *bool, options *v41ExpertCohortOptions, handoff *v41RoutedOutputHandoff) (result [][]float32, cause error) {
	H, I, count := s.M.Cfg.HiddenSize, s.M.Cfg.MoEIntermediateSize, len(inputs)
	if handoff != nil && (options != nil || !handoff.owner.live() || handoff.owner.session != s ||
		handoff.owner.layer != layer || handoff.owner.width != H || handoff.owner.inner != I || handoff.owner.limit != limit || len(handoff.rows) != count) {
		return nil, errV41ExpertResult
	}
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
	usedDownTile := false
	uploads := map[int]compute.Tensor{}
	faultRow, current := count, 0
	var primary any
	valid := func(t compute.Tensor, n int) bool {
		return t.Buf() != nil && t.Dtype == compute.F32 && t.Layout == compute.RowMajor && t.Quant == nil && len(t.Shape) == 1 && t.Shape[0] == n
	}
	own := func(t compute.Tensor) bool {
		if t.Buf() == nil || t.Backend() != gateW.Backend() || seen[t.Buf()] || handoff != nil && handoff.owner.owns(t.Buf()) {
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
	coldDispatched := false
	noteDispatch := func(row int, gateUp bool) {
		if options != nil {
			if options.note != nil {
				options.note(row, gateUp)
			}
			return
		}
		coldDispatched = true
		s.M.v41NoteIncrementalDeviceDispatch(gateUp, 0)
	}
	// This host-observed envelope ends at completion/readback, before result
	// copying and cleanup. It overlaps ExpertActivationNanos, not a disjoint
	// kernel-only cost, and is charged once for the whole cold cohort.
	closeColdDispatch := func() {
		if coldDispatched {
			s.M.v41NoteIncrementalDeviceDispatchElapsed(opened)
			coldDispatched = false
		}
	}
	func() {
		defer func() {
			if p := recover(); p != nil {
				primary, faultRow = p, current
				*recordingFailed = true
			}
		}()
		var tile *v41ExpertTile
		var downTile *v41ExpertDownTile
		var activations []compute.Tensor
		groupStart, groupEnd, chunkStart := 0, 0, 0
	recordRows:
		for current = 0; current < count; current++ {
			*stage = "gate/up"
			gate, up, down := gateW, upW, downW
			if options != nil {
				gate, up, down = options.triples[current][0], options.triples[current][1], options.triples[current][2]
			}
			if current == groupEnd {
				groupStart, groupEnd, chunkStart = current, options.tileEnd(current, count), current
				tile = newV41ExpertTile(s.Backend, gate, up, groupEnd-groupStart)
				downTile = newV41ExpertDownTile(s.Backend, down, groupEnd-groupStart)
				if options == nil && downTile != nil {
					tile.qualifyColdGrid(groupEnd - groupStart)
				}
				usedDownTile = usedDownTile || downTile != nil
				if options == nil || options.note != nil {
					if tile != nil {
						tile.note = func() { noteDispatch(current, true) }
					}
					if downTile != nil {
						downTile.note = func() { noteDispatch(current, false) }
					}
				}
			}
			g, u, tiled, err := tile.project(s, inputs[groupStart:groupEnd], current-groupStart, H, I, own, valid)
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
				noteDispatch(current, true)
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
			if downTile != nil && groupEnd-chunkStart > 1 {
				activations = append(activations, a)
				if len(activations) < v41ExpertTileSize(groupEnd-chunkStart) {
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
			noteDispatch(current, false)
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
		closeColdDispatch()
		return nil, cause
	}
	if usedDownTile {
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
	closeColdDispatch()
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
		if verdict == nil || !usedDownTile && statusRows[i] > faultRow {
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
	if handoff != nil {
		n, err := handoff.adopt(published)
		if err != nil {
			faultRow, *stage, *recordingFailed = 0, "readback", true
			return nil, err
		}
		for _, row := range published[:n] {
			consumed[row.Buf()] = true // ownership moved; never freed here
		}
	}
	return result, nil
}
