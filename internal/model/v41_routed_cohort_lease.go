package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Lane is the original request index, never the compacted survivor index.
// Input is borrowed, finite widened BF16 and stable until Run returns.
type v41RoutedCohortRow struct {
	Lane    int
	Session *Session
	Input   []float32
	Weight  float32
}

// A lease stages/holds one same-expert triple without selecting arithmetic.
// The coordinator prepares every group before choosing its scheduling mode;
// any decline closes the prepared leases and retains whole-lane execution.
// Run is one-shot and has no decline outcome. Close is required on every path.
type v41RoutedCohortLease struct {
	scope          *backendExecutionScope
	rows           []v41RoutedCohortRow
	leader         *Session
	layer          int
	stage          string
	deferred       compute.V41DeferredStatusBackend
	gate, up, down compute.Tensor
	releases       []func()
	ran, closed    bool
}

func prepareV41RoutedExpertCohortOwned(scope *backendExecutionScope, layer int, stem string, rows []v41RoutedCohortRow) (lease *v41RoutedCohortLease, outcome v41RoutedExpertOutcome, faultLane int, err error) {
	faultLane = -1
	if scope == nil || !scope.live || scope.owner == nil || len(rows) < 1 || len(rows) > 4 {
		return nil, v41RoutedExpertDeclined, -1, nil
	}
	leader := rows[0].Session
	if leader == nil || leader.M == nil || leader.Backend == nil || !backendHasComparableIdentity(leader.Backend) || leader.v41RoutedExpertFunc() == nil || layer < 0 || layer >= leader.M.Cfg.NumLayers {
		return nil, v41RoutedExpertDeclined, -1, nil
	}
	H, I := leader.M.Cfg.HiddenSize, leader.M.Cfg.MoEIntermediateSize
	seen := map[*Session]bool{}
	lastLane := -1
	for _, row := range rows {
		s := row.Session
		if row.Lane <= lastLane || s == nil || seen[s] || s.M != leader.M || s.Backend != leader.Backend || s.backendExecutionScope != scope || s.halClosed || s.halFailure != nil || len(row.Input) != H || !v41BF16Widened(row.Input) || !finite32(row.Weight) {
			return nil, v41RoutedExpertDeclined, -1, nil
		}
		seen[s], lastLane = true, row.Lane
	}
	if owner := backendExecutionOwner(leader.Backend); owner != nil && owner != scope.owner {
		return nil, v41RoutedExpertDeclined, -1, nil
	}
	deferred, ok := v41DeferredBF16Publisher(leader.Backend, H)
	publisher, havePublisher := leader.Backend.(compute.V41BF16PublishBackend)
	if !ok || !deferred.SupportsV41DeferredStatus(I) || !havePublisher || !publisher.SupportsV41BF16Publish(H) {
		return nil, v41RoutedExpertDeclined, -1, nil
	}
	gateName, upName, downName := layerName(layer, stem+".w1.weight"), layerName(layer, stem+".w3.weight"), layerName(layer, stem+".w2.weight")
	if leader.M.has(downName[:len(downName)-len("weight")] + "bias") {
		return nil, v41RoutedExpertDeclined, -1, nil
	}
	l := &v41RoutedCohortLease{scope: scope, rows: append([]v41RoutedCohortRow(nil), rows...), leader: leader, layer: layer, stage: "gate/up", deferred: deferred}
	prepared := false
	defer func() {
		primary := recover()
		if !prepared {
			func() {
				defer func() {
					if p := recover(); p != nil && primary == nil {
						primary = p
					}
				}()
				l.Close()
			}()
		}
		if primary != nil {
			lease, outcome = nil, v41RoutedExpertError
			if cause, known := v41RoutedCohortPanicCause(primary); known {
				err = l.retire(cause)
				var forward *BackendForwardOperationError
				if errors.As(cause, &forward) {
					panic(primary)
				}
				return
			}
			l.retire(v41RoutedCohortUnknownPanic(primary))
			panic(primary)
		}
	}()
	leader.ensureOpenBackendSession()
	// Previously prepared groups retain their holds during this fit check.
	// A capacity decline selects no arithmetic and stages no new triple.
	if ring := leader.routedExpertRing(gateName); ring != nil {
		fits := func() bool {
			done := leader.ringEnter(ring)
			defer done()
			return leader.v41RoutedLayerRingFits(ring, [][3]string{{gateName, upName, downName}})
		}()
		if !fits {
			return nil, v41RoutedExpertDeclined, -1, nil
		}
	}
	var releaseGate, releaseUp func()
	l.gate, releaseGate, l.up, releaseUp, _, ok = expertInputDeviceAdmitted(leader, gateName, upName, rows[0].Input, H)
	if !ok {
		return nil, v41RoutedExpertDeclined, -1, nil
	}
	l.releases = append(l.releases, releaseGate, releaseUp)
	var releaseDown func()
	l.down, releaseDown, ok = leader.q4kExpertDownDeviceWeight(downName)
	if !ok {
		return nil, v41RoutedExpertDeclined, -1, nil
	}
	l.releases = append(l.releases, releaseDown)
	prepared = true
	return l, v41RoutedExpertHandled, -1, nil
}

func (l *v41RoutedCohortLease) Run() (outputs [][]float32, faultLane int, err error) {
	return runV41RoutedCohortLeases([]*v41RoutedCohortLease{l}, nil)
}

// Rows indexes the lease-major flattened expert outputs in the caller's exact
// v41RoutedSumOrder. Results use this slice's lane order, never compacted IDs.
type v41RoutedCohortSum struct {
	Lane int
	Rows []int
}

// All prepared groups form one selected recording/fence boundary. Arithmetic
// stays expert-major, while each device sum replays its original pick order.
// A nil sums list is the single-lease Run adapter, returning individual rows.
func runV41RoutedCohortLeases(leases []*v41RoutedCohortLease, sums []v41RoutedCohortSum) (outputs [][]float32, faultLane int, err error) {
	faultLane = -1 // The core's internal row ledger is not public provenance.
	if len(leases) == 0 || leases[0] == nil {
		return nil, -1, fmt.Errorf("model: empty routed cohort lease batch")
	}
	first := leases[0]
	var inputs [][]float32
	var weights []float32
	var flat []v41RoutedCohortRow
	options := &v41ExpertCohortOptions{}
	seenLeases := map[*v41RoutedCohortLease]bool{}
	seenSessions := map[*Session]int{}
	inputRows := map[int]int{}
	for _, l := range leases {
		if l == nil || l.closed || l.ran || seenLeases[l] || l.scope == nil || !l.scope.live || l.scope != first.scope || l.layer != first.layer || l.leader.M != first.leader.M || l.leader.Backend != first.leader.Backend {
			return nil, -1, fmt.Errorf("model: routed cohort lease is closed, consumed or outside its owner")
		}
		seenLeases[l] = true
		for _, tensor := range [3]compute.Tensor{l.gate, l.up, l.down} {
			if tensor.Buf() == nil || tensor.Backend() != first.gate.Backend() {
				return nil, -1, fmt.Errorf("model: routed cohort weight identity changed")
			}
		}
		for _, row := range l.rows {
			s := row.Session
			if s.backendExecutionScope != l.scope || s.halClosed || s.halFailure != nil {
				return nil, -1, fmt.Errorf("model: routed cohort participant left its prepared owner")
			}
			if lane, seen := seenSessions[s]; seen && lane != row.Lane {
				return nil, -1, fmt.Errorf("model: routed cohort session changed lane")
			} else if !seen {
				seenSessions[s] = row.Lane
				for _, weight := range s.halW {
					options.borrowed = append(options.borrowed, weight)
				}
			}
			key := len(flat)
			if prior, seen := inputRows[row.Lane]; seen {
				previous := flat[prior]
				if previous.Session != s {
					return nil, -1, fmt.Errorf("model: routed cohort lane changed session")
				}
				if len(row.Input) > 0 && len(row.Input) == len(previous.Input) && &row.Input[0] == &previous.Input[0] {
					key = prior
				}
			} else {
				inputRows[row.Lane] = key
			}
			inputs, weights, flat = append(inputs, row.Input), append(weights, row.Weight), append(flat, row)
			options.triples = append(options.triples, [3]compute.Tensor{l.gate, l.up, l.down})
			options.inputKeys = append(options.inputKeys, key)
		}
	}
	if len(flat) > v41ExpertCohortRows || sums == nil && len(leases) != 1 {
		return nil, -1, fmt.Errorf("model: routed cohort batch exceeds its bounded row plan")
	}
	if sums != nil {
		used, lanes := make([]bool, len(flat)), map[int]bool{}
		options.sums = make([][]int, len(sums))
		for i, sum := range sums {
			if lanes[sum.Lane] || len(sum.Rows) == 0 {
				return nil, -1, fmt.Errorf("model: duplicate or empty routed cohort sum")
			}
			lanes[sum.Lane] = true
			for _, row := range sum.Rows {
				if row < 0 || row >= len(flat) || used[row] || flat[row].Lane != sum.Lane {
					return nil, -1, fmt.Errorf("model: routed cohort sum row does not belong to lane")
				}
				used[row] = true
			}
			options.sums[i] = sum.Rows
		}
		for _, assigned := range used {
			if !assigned {
				return nil, -1, fmt.Errorf("model: routed cohort sum omitted a row")
			}
		}
	}
	for _, l := range leases {
		l.ran = true
	}
	stage := "gate/up"
	retire := func(cause error) error {
		var failure error
		for _, l := range leases {
			l.stage = stage
			if failure == nil {
				failure = l.retire(cause)
			} else {
				l.retire(failure)
			}
		}
		return failure
	}
	defer func() {
		if p := recover(); p != nil {
			outputs = nil
			if cause, known := v41RoutedCohortPanicCause(p); known {
				err = retire(cause)
				var forward *BackendForwardOperationError
				if errors.As(cause, &forward) {
					panic(p)
				}
				return
			}
			retire(v41RoutedCohortUnknownPanic(p))
			panic(p)
		}
	}()
	options.note = func(_ int, gateUp bool) {
		first.leader.M.v41NoteIncrementalDeviceDispatch(gateUp, 0)
	}
	selectedFailure := false
	outputs, err = first.leader.v41RunExpertCohortSelected(first.deferred, first.gate, first.up, first.down, inputs, weights, float32(first.leader.M.Cfg.SwigluLimit), first.layer, &stage, &selectedFailure, options)
	if err != nil {
		var numeric *V41ExpertOperationError
		if !errors.As(err, &numeric) {
			err = retire(err)
		} else if selectedFailure {
			retire(err)
		}
		return nil, -1, err
	}
	for range flat {
		first.leader.M.v41NoteExpertContraction()
	}
	return outputs, -1, nil
}

// Close owns only this triple's holds, never cancellation or a transaction.
// It attempts every release, is idempotent even after a panic, and preserves
// the original scalar hold-release order (down, up, gate).
func (l *v41RoutedCohortLease) Close() {
	if l == nil || l.closed {
		return
	}
	l.closed = true
	var primary any
	for i := len(l.releases) - 1; i >= 0; i-- {
		func() {
			defer func() {
				if p := recover(); p != nil {
					primary = p
				}
			}()
			l.releases[i]()
		}()
	}
	l.releases = nil
	if primary != nil {
		l.retire(v41RoutedCohortUnknownPanic(primary))
		panic(primary)
	}
}

func (l *v41RoutedCohortLease) retire(cause error) error {
	failure := cause
	var forward *BackendForwardOperationError
	if !errors.As(cause, &forward) {
		failure = &BackendForwardOperationError{Backend: l.leader.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-cohort", Layer: l.layer, Stage: l.stage, Cause: cause}
	}
	for _, row := range l.rows {
		s := row.Session
		if s.halFailure == nil {
			s.halFailure = failure
		}
		if !s.halClosed {
			func() { defer func() { s.halClosed = true; _ = recover() }(); s.closeBackendOwned() }()
		}
	}
	return failure
}

func v41RoutedCohortPanicCause(p any) (error, bool) {
	if err, ok := p.(error); ok {
		var backend *compute.BackendError
		var forward *BackendForwardOperationError
		if errors.As(err, &backend) || errors.As(err, &forward) {
			return err, true
		}
	}
	if err, ok := compute.ConvertCUDAPanic(p, "", ""); ok {
		if original, ok := p.(error); ok {
			err = original
		}
		return err, true
	}
	return nil, false
}

func v41RoutedCohortUnknownPanic(p any) error {
	if err, ok := p.(error); ok {
		return err
	}
	return fmt.Errorf("unclassified backend panic: %v", p)
}
