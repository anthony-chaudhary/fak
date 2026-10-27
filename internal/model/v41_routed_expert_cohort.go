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
	return s.v41RoutedExpertCohortInputs(layer, stem, inputs, weights, handoff, nil)
}

func (s *Session) v41RoutedExpertCohortInputs(layer int, stem string, inputs [][]float32, weights []float32, handoff *v41RoutedOutputHandoff, input *v41RoutedInputHandoff) (result [][]float32, outcome v41RoutedExpertOutcome, cause error) {
	if input != nil && !input.valid(s, layer, inputs) {
		return nil, v41RoutedExpertDeclined, nil
	}
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
	result, cause = s.v41RunExpertCohortInputs(deferred, gate, up, down, inputs, weights, float32(s.M.Cfg.SwigluLimit), layer, &stage, &recordingFailed, nil, handoff, input)
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
	return s.v41RunExpertCohortInputs(deferred, gateW, upW, downW, inputs, weights, limit, layer, stage, recordingFailed, options, handoff, nil)
}

func (s *Session) v41RunExpertCohortInputs(deferred compute.V41DeferredStatusBackend, gateW, upW, downW compute.Tensor, inputs [][]float32, weights []float32, limit float32, layer int, stage *string, recordingFailed *bool, options *v41ExpertCohortOptions, handoff *v41RoutedOutputHandoff, input *v41RoutedInputHandoff) (result [][]float32, cause error) {
	H, I, count := s.M.Cfg.HiddenSize, s.M.Cfg.MoEIntermediateSize, len(inputs)
	if handoff != nil && (options != nil || !handoff.owner.live() || handoff.owner.session != s ||
		handoff.owner.layer != layer || handoff.owner.width != H || handoff.owner.inner != I || handoff.owner.limit != limit || len(handoff.rows) != count) {
		return nil, errV41ExpertResult
	}
	if input != nil && (options != nil || !input.valid(s, layer, inputs)) {
		return nil, errV41ExpertResult
	}
	owners := count
	if options != nil && options.sums != nil {
		owners++ // Sum temporaries follow every expert row in the fault ledger.
	}
	// One cohort-local backing replaces independent per-row append growth.
	// One-based links preserve each row's reverse acquisition order even when
	// chunk recording returns to an earlier row; zero denotes an empty row.
	type ownedTensor struct {
		tensor   compute.Tensor
		previous int
	}
	ownedHeads := make([]int, owners)
	var owned []ownedTensor
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
	usedActivationChunk := false
	usedPublicationChunk := false
	var publicationRows []int
	var panelRead compute.V41DeferredPanelReader
	panelReadChecked := false
	appendPublication := func(t compute.Tensor, rows int) {
		published = append(published, t)
		if usedPublicationChunk {
			publicationRows = append(publicationRows, rows)
		}
	}
	uploads := map[int]compute.Tensor{}
	faultRow, current := count, 0
	var primary any
	valid := func(t compute.Tensor, n int) bool {
		return t.Buf() != nil && t.Dtype == compute.F32 && t.Layout == compute.RowMajor && t.Quant == nil && len(t.Shape) == 1 && t.Shape[0] == n
	}
	own := func(t compute.Tensor) bool {
		if t.Buf() == nil || t.Backend() != gateW.Backend() || seen[t.Buf()] || s.v41RoutedInputs.owns(t.Buf()) || handoff != nil && handoff.owner.owns(t.Buf()) {
			return false
		}
		seen[t.Buf()] = true
		previous := ownedHeads[current]
		owned = append(owned, ownedTensor{tensor: t, previous: previous})
		ownedHeads[current] = len(owned)
		return true
	}
	borrow := func(t compute.Tensor) bool {
		if t.Buf() == nil || t.Backend() != gateW.Backend() || seen[t.Buf()] || handoff != nil && handoff.owner.owns(t.Buf()) {
			return false
		}
		seen[t.Buf()] = true
		return true
	}
	// Chunk output/status allocations belong to the chunk-start row. Their
	// cleanup boundary is grouped, unlike independent per-row activation frees.
	// Always attempt every owned release. The last panic in a row's reverse
	// traversal is what nested Go defers in the scalar owner would propagate.
	defer func() {
		for r, head := range ownedHeads {
			var cleanup any
			for at := head; at != 0; at = owned[at-1].previous {
				t := owned[at-1].tensor
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
	recordMany := func(t compute.Tensor, status compute.V41DeferredStatus, n, records int, kind compute.V41DeferredStatusKind, site uint32, err error) error {
		fresh, recordFresh := own(t), own(status.Record)
		recordOK := recordFresh && valid(status.Record, 4*records) && status.Kind == kind && status.Count == n && status.Site == site && (status.Records == records || records == 1 && status.Records == 0)
		if err != nil {
			// A record returned with a selected operation error remains owned
			// for cleanup, but cannot replace that same operation's error.
			return err
		}
		if recordOK {
			statuses = append(statuses, status)
			statusRows = append(statusRows, current)
		}
		if !fresh || !valid(t, n*records) || !recordOK {
			return errV41ExpertResult
		}
		return nil
	}
	record := func(t compute.Tensor, status compute.V41DeferredStatus, n int, kind compute.V41DeferredStatusKind, site uint32, err error) error {
		return recordMany(t, status, n, 1, kind, site, err)
	}
	publishDown := func(d compute.Tensor) error {
		p, st, err := deferred.V41BF16PublishDeferred(d, compute.V41BF16PublishRoutedDown)
		if err = record(p, st, H, compute.V41DeferredBF16Publish, compute.V41BF16PublishRoutedDown, err); err != nil {
			return err
		}
		appendPublication(p, 1)
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
				if tile != nil && input != nil {
					start := groupStart
					tile.input = func(row, rows int) (compute.Tensor, error) { return input.gather(start+row, rows, borrow) }
				}
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
			chunkRows := 0
			if options == nil && downTile != nil && current == chunkStart {
				chunkRows = v41ExpertTileSize(groupEnd - current)
			}
			g, u, tiled, panelRow, panelChunk, err := tile.project(s, inputs[groupStart:groupEnd], current-groupStart, H, I, options == nil, chunkRows, own, valid)
			if cause = err; cause != nil {
				break
			}
			if !tiled {
				x := compute.Tensor{}
				if options != nil {
					x = uploads[options.inputKeys[current]]
				}
				if x.Buf() == nil {
					var inputErr error
					if input != nil {
						x, inputErr = input.gather(current, 1, borrow)
					} else {
						x = s.uploadHostF32([]int{H}, inputs[current], compute.MemoryActivation, "moe expert gate/up activation")
					}
					fresh := own(x)
					if inputErr != nil {
						cause = inputErr
						break
					}
					if !fresh || !valid(x, H) {
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
			var a compute.Tensor
			var st compute.V41DeferredStatus
			if panelChunk != nil {
				usedActivationChunk = true
				a, st, err = panelChunk.V41RoutedSwiGLUPanelChunkDeferred(g, u, current-groupStart-tile.start, limit, weights[current:current+chunkRows])
			} else if panelRow != nil {
				a, st, err = panelRow.V41RoutedSwiGLUPanelRowDeferred(g, u, current-groupStart-tile.start, limit, weights[current])
			} else {
				a, st, err = deferred.V41RoutedSwiGLUDeferred(g, u, limit, weights[current])
			}
			records := 1
			if panelChunk != nil {
				records = chunkRows
			}
			if cause = recordMany(a, st, I, records, compute.V41DeferredRoutedSwiGLU, 0, err); cause != nil {
				break
			}
			if panelChunk != nil {
				current += chunkRows - 1
			}

			*stage = "down"
			if downTile != nil && groupEnd-chunkStart > 1 {
				if panelChunk == nil {
					activations = append(activations, a)
					if len(activations) < v41ExpertTileSize(groupEnd-chunkStart) {
						continue
					}
				}
				last := current
				var panel compute.Tensor
				if panelChunk != nil {
					panel, err = downTile.projectPacked(a, chunkRows, I, H, own, valid)
				} else {
					panel, err = downTile.project(activations, I, H, own, valid)
				}
				if cause = err; cause != nil {
					break
				}
				if chunks, ok := s.Backend.(compute.V41RoutedDownPanelChunkBackend); options == nil && ok &&
					(handoff == nil || handoff.canCompleteUnretained(count)) && chunks.SupportsV41RoutedDownPanelChunk(panel, 0, last-chunkStart+1) {
					if !panelReadChecked {
						panelRead = compute.BindV41DeferredPanelReader(deferred)
						panelReadChecked = true
					}
					if panelRead != nil {
						if !usedPublicationChunk {
							publicationRows = make([]int, len(published))
							for i := range publicationRows {
								publicationRows[i] = 1
							}
							usedPublicationChunk = true
						}
						current = chunkStart // Grouped publication/cleanup belongs to chunk start.
						rows := last - chunkStart + 1
						p, st, err := chunks.V41RoutedDownPanelChunkDeferred(panel, 0, rows)
						if cause = recordMany(p, st, H, rows, compute.V41DeferredBF16Publish, compute.V41BF16PublishRoutedDown, err); cause != nil {
							break recordRows
						}
						appendPublication(p, rows)
						current, chunkStart, activations = last, last+1, activations[:0]
						continue
					}
				}
				for row := chunkStart; row <= last; row++ {
					current = row
					if rows, ok := s.Backend.(compute.V41RoutedDownPanelRowBackend); options == nil && ok && rows.SupportsV41RoutedDownPanelRow(panel, row-chunkStart) {
						p, st, err := rows.V41RoutedDownPanelRowDeferred(panel, row-chunkStart)
						if cause = record(p, st, H, compute.V41DeferredBF16Publish, compute.V41BF16PublishRoutedDown, err); cause != nil {
							break recordRows
						}
						appendPublication(p, 1)
					} else {
						d, err := downTile.row(panel, row-chunkStart, H, own, valid)
						if cause = err; cause != nil {
							break recordRows
						}
						if cause = publishDown(d); cause != nil {
							break recordRows
						}
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
	// Physical descriptors remain unique owners. Only the logical verdict
	// mapping expands multi-record statuses; no repeated tensor is read or freed.
	var logicalRows []int
	var logicalKinds []compute.V41DeferredStatusKind
	var verdictOrder []int
	expectedVerdicts := len(statuses)
	if usedActivationChunk || usedPublicationChunk {
		for i, st := range statuses {
			n := st.Records
			if n == 0 {
				n = 1
			}
			for j := 0; j < n; j++ {
				logicalRows = append(logicalRows, statusRows[i]+j)
				logicalKinds = append(logicalKinds, st.Kind)
				verdictOrder = append(verdictOrder, len(verdictOrder))
			}
		}
		expectedVerdicts = len(verdictOrder)
		for i := 1; i < len(verdictOrder); i++ {
			for j := i; j > 0 && (logicalRows[verdictOrder[j]] < logicalRows[verdictOrder[j-1]] || logicalRows[verdictOrder[j]] == logicalRows[verdictOrder[j-1]] && logicalKinds[verdictOrder[j]] == compute.V41DeferredRoutedSwiGLU && logicalKinds[verdictOrder[j-1]] == compute.V41DeferredBF16Publish); j-- {
				verdictOrder[j], verdictOrder[j-1] = verdictOrder[j-1], verdictOrder[j]
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
	var readResult compute.V41DeferredReadResult
	expectedReadRows := len(readRows)
	readLayout := publicationRows
	if usedPublicationChunk {
		expectedReadRows = count
		if cause != nil || primary != nil {
			expectedReadRows = 0
			readLayout = nil
		}
	}
	ownedRead := options == nil && (usedPublicationChunk || len(readRows) >= 2)
	readCount := 0
	func() {
		defer func() {
			if p := recover(); p != nil {
				primary, cause, faultRow, *stage = p, nil, statusRows[0], "readback"
			}
		}()
		if usedPublicationChunk {
			verdicts, readResult, readErr = panelRead(statuses, readRows, readLayout, H)
			readCount = readResult.RowCount()
		} else if ownedRead {
			verdicts, readResult, readErr = compute.ReadV41DeferredResult(deferred, statuses, readRows)
			readCount = readResult.RowCount()
		} else {
			verdicts, values, readErr = deferred.V41ReadDeferred(statuses, readRows)
			readCount = len(values)
		}
	}()
	closeColdDispatch()
	if readErr != nil {
		cause, primary, faultRow, *stage = readErr, nil, statusRows[0], "readback"
		return nil, cause
	}
	if len(verdicts) != expectedVerdicts || readCount != expectedReadRows {
		if cause == nil && primary == nil {
			cause, faultRow, *stage = errV41ExpertResult, 0, "readback"
		}
		return nil, cause
	}
	for position, verdict := range verdicts {
		row, kind := 0, compute.V41DeferredRoutedSwiGLU
		if usedActivationChunk || usedPublicationChunk {
			i := verdictOrder[position]
			verdict = verdicts[i]
			row, kind = logicalRows[i], logicalKinds[i]
		} else {
			row, kind = statusRows[position], statuses[position].Kind
		}
		if verdict == nil || !usedDownTile && row > faultRow {
			continue
		}
		// Under tiled scheduling every retained status was issued before a
		// recording failure, even if its row follows the failed down row.
		r := row
		*stage = "activation"
		var activation *compute.V41RoutedSwiGLUFault
		var publication *compute.V41BF16CheckedArithmeticError
		if errors.As(verdict, &activation) {
			name := "gate/up"
			if activation.Stage == compute.V41RoutedSwiGLUStageDownOperand {
				name = "down"
			}
			verdict = v41ExpertOperationErr(layer, name, errV41ExpertResult)
		} else if kind == compute.V41DeferredBF16Publish {
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
	// Ownership is consumed only after the complete original verdict loop.
	independent := false
	if ownedRead {
		values, independent = readResult.TakeRows()
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
		result[r] = v41CohortHostRow(row, independent)
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
		if usedPublicationChunk {
			if err := handoff.completeUnretained(count); err != nil {
				faultRow, *stage, *recordingFailed = 0, "readback", true
				return nil, err
			}
			return result, nil
		}
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

// Only the canonical typed read result can authorize this transfer. Legacy
// readers and owned decode paths retain their independent defensive copy.
func v41CohortHostRow(row []float32, independent bool) []float32 {
	if independent {
		return row
	}
	return append([]float32(nil), row...)
}

// The first candidate admits one bounded cold panel. Larger prompts retain the
// historical route; no cache or device tensor survives this grouped layer.
const (
	v41RoutedInputMaxBytes = 30 << 20
	v41RoutedInputMaxRows  = 1536
)

type v41RoutedInputOwner struct {
	session      *Session
	model        *Model
	backend      compute.Backend
	scope        *backendExecutionScope
	layer, width int
	inputs       [][]float32
	packed       []float32
	copies       compute.V41RowLogBackend
	tensor       compute.Tensor
	closed       bool
}

func newV41RoutedInputOwner(s *Session, layer int, inputs [][]float32) *v41RoutedInputOwner {
	return newV41RoutedInputOwnerPacked(s, layer, inputs, nil)
}

func newV41RoutedInputOwnerPacked(s *Session, layer int, inputs [][]float32, packed []float32) *v41RoutedInputOwner {
	if s == nil || s.M == nil || s.Backend == nil || !backendHasComparableIdentity(s.Backend) || s.v41RoutedInputs != nil || s.halClosed || s.halFailure != nil || len(inputs) < 2 || len(inputs) > v41RoutedInputMaxRows || layer < 0 || layer >= s.M.Cfg.NumLayers {
		return nil
	}
	scope := s.backendExecutionScope
	if scope == nil || !scope.live || scope.owner == nil || backendExecutionOwner(s.Backend) != scope.owner {
		return nil
	}
	width := s.M.Cfg.HiddenSize
	if width <= 0 || width > v41RoutedInputMaxBytes/4/len(inputs) {
		return nil
	}
	copies, ok := s.Backend.(compute.V41RowLogBackend)
	if !ok || !copies.V41RowLogAvailable() {
		return nil
	}
	for _, row := range inputs {
		if len(row) != width || !v41BF16Widened(row) {
			return nil
		}
	}
	// The admitted row/byte bound above also bounds this multiplication. Only
	// the exact request-owned row backing is reusable, never equal-valued data.
	if len(packed) != len(inputs)*width {
		packed = nil
	} else {
		for i, row := range inputs {
			if &row[0] != &packed[i*width] {
				packed = nil
				break
			}
		}
	}
	o := &v41RoutedInputOwner{session: s, model: s.M, backend: s.Backend, scope: scope, layer: layer, width: width, inputs: inputs, packed: packed, copies: copies}
	s.v41RoutedInputs = o
	return o
}

func (o *v41RoutedInputOwner) live() bool {
	return o != nil && !o.closed && o.session.v41RoutedInputs == o && o.session.M == o.model && o.session.Backend == o.backend && o.session.backendExecutionScope == o.scope && o.scope.live && !o.session.halClosed && o.session.halFailure == nil && o.model.Cfg.HiddenSize == o.width
}
func (o *v41RoutedInputOwner) owns(buf compute.Buffer) bool {
	return o != nil && buf != nil && o.tensor.Buf() == buf
}

type v41RoutedInputHandoff struct {
	owner *v41RoutedInputOwner
	rows  []v41ExpertRow
}

func (o *v41RoutedInputOwner) handoff(rows []v41ExpertRow) *v41RoutedInputHandoff {
	if o == nil {
		return nil
	}
	return &v41RoutedInputHandoff{owner: o, rows: rows}
}
func (h *v41RoutedInputHandoff) valid(s *Session, layer int, inputs [][]float32) bool {
	if h == nil || !h.owner.live() || h.owner.session != s || h.owner.layer != layer || len(h.rows) != len(inputs) {
		return false
	}
	for i, row := range h.rows {
		if row.Token < 0 || row.Token >= len(h.owner.inputs) || len(inputs[i]) != h.owner.width || &inputs[i][0] != &h.owner.inputs[row.Token][0] {
			return false
		}
	}
	return true
}

// claim excludes weight/output aliases before ownership moves to the layer.
// The gather allocation belongs to the calling row even on a selected error.
func (h *v41RoutedInputHandoff) gather(first, count int, claim func(compute.Tensor) bool) (compute.Tensor, error) {
	o := h.owner
	if !o.live() || first < 0 || count <= 0 || first > len(h.rows)-count {
		return compute.Tensor{}, errV41ExpertResult
	}
	if o.tensor.Buf() == nil {
		packed := o.packed
		if packed == nil {
			packed = make([]float32, len(o.inputs)*o.width)
			for i, row := range o.inputs {
				copy(packed[i*o.width:(i+1)*o.width], row)
			}
		}
		t := o.session.uploadHostF32([]int{len(o.inputs), o.width}, packed, compute.MemoryActivation, "V4.1 layer routed input")
		if !claim(t) {
			return compute.Tensor{}, errV41ExpertResult
		}
		o.tensor = t
		if t.Dtype != compute.F32 || t.Layout != compute.RowMajor || t.Quant != nil || len(t.Shape) != 2 || t.Shape[0] != len(o.inputs) || t.Shape[1] != o.width {
			return compute.Tensor{}, errV41ExpertResult
		}
	}
	spans := make([]compute.V41RowSpan, count)
	for i, row := range h.rows[first : first+count] {
		spans[i] = compute.V41RowSpan{Src: o.tensor, Row: row.Token, Rows: 1}
	}
	x, err := o.copies.V41RowLogGather(spans, o.width)
	if err != nil {
		return x, err
	}
	if len(x.Shape) != 2 || x.Shape[0] != count || x.Shape[1] != o.width {
		return x, errV41ExpertResult
	}
	x.Shape = []int{count * o.width}
	return x, nil
}

// Detach before invoking backend code. A single buffer needs one release even
// when Free panics; selected retirement and normal layer finish are idempotent.
func (o *v41RoutedInputOwner) release() (cleanup v41RoutedOutputCleanup) {
	if o == nil || o.closed {
		return cleanup
	}
	o.closed = true
	if o.session.v41RoutedInputs == o {
		o.session.v41RoutedInputs = nil
	}
	t := o.tensor
	o.tensor, o.inputs, o.packed = compute.Tensor{}, nil, nil
	if t.Buf() == nil {
		return cleanup
	}
	complete := false
	defer func() {
		p := recover()
		if !complete {
			cleanup = v41RoutedOutputCleanup{value: p, present: true}
			// Goexit cannot return the cleanup receipt to finish/drain.
			// The detached owner must retire while unwinding instead.
			if p == nil {
				if o.session.halFailure == nil {
					o.session.halFailure = &BackendForwardOperationError{Backend: o.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-input", Layer: o.layer, Stage: "cleanup", Cause: errors.New("model: incomplete routed input release")}
				}
				func() { defer func() { o.session.halClosed = true; _ = recover() }(); o.session.closeBackendOwned() }()
			}
		}
	}()
	o.backend.Free(t)
	complete = true
	return cleanup
}

func (o *v41RoutedInputOwner) finish(err *error, completed *bool) {
	primary := recover()
	cleanup := o.release()
	if cleanup.present || !*completed {
		cause := *err
		known := true
		if cause == nil {
			if primary != nil {
				cause = v41RoutedCohortUnknownPanic(primary)
			} else if cleanup.present {
				cause, known = v41RoutedCohortPanicCause(cleanup.value)
				if !known {
					cause = v41RoutedCohortUnknownPanic(cleanup.value)
				}
			} else {
				cause = errors.New("model: incomplete routed input owner")
			}
		}
		if o.session.halFailure == nil {
			o.session.halFailure = &BackendForwardOperationError{Backend: o.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-input", Layer: o.layer, Stage: "cleanup", Cause: cause}
		}
		func() { defer func() { o.session.halClosed = true; _ = recover() }(); o.session.closeBackendOwned() }()
		if primary != nil {
			panic(primary)
		}
		if *err == nil {
			if cleanup.present && !known {
				panic(cleanup.value)
			}
			*err = o.session.halFailure
		}
	}
	if primary != nil {
		panic(primary)
	}
}
