package model

import (
	"fmt"
	"os"
)

// v41LayerStepWork is a lane-owned continuation, never a shallow Session copy.
// Attention/router preparation may change journaled source state, but hidden
// state/window publication and the final token commit remain with this lane.
type v41LayerStepWork struct {
	layer   int
	input   []float32
	picks   []routePick
	full    bool
	scratch *v41ProjScratch
	scalar  func() ([]float32, error)
	finish  func([]float32) error
	cleanup func()
	closed  bool
}

func (w *v41LayerStepWork) close() {
	if w == nil || w.closed {
		return
	}
	w.closed = true
	if w.cleanup != nil {
		w.cleanup()
	}
}

func (w *v41LayerStepWork) finishCleanup(err *error) {
	finishModelCleanup(recover(), err, w.close)
}

func (w *v41LayerStepWork) complete(routed []float32) (err error) {
	if w == nil || w.closed {
		return fmt.Errorf("%w: layer continuation is closed", ErrV41ForwardStage)
	}
	defer w.finishCleanup(&err)
	if routed == nil {
		routed, err = w.scalar()
		if err != nil {
			return err
		}
	}
	return w.finish(routed)
}

// The existing explicit decode ID route and disabled layer batching retain precedence.
// The independent cold-prefill panel opt-in does not select a decode route.
// The new schedule is confined to the already-selected multi-session caller;
// single-lane wrappers execute their unchanged scalar/fused continuation.
func (m *Model) v41LayerCohortEnabled(lanes []v41StepCohortLane, txs []*v41StepTransaction) bool {
	if v41RoutedIDOptedIn() || os.Getenv(v41LayerExpertBatchEnv) == "0" {
		return false
	}
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil || !full {
		return false
	}
	active := 0
	for lane, tx := range txs {
		if tx == nil || tx.closed {
			continue
		}
		if lanes[lane].session.v41State().routedExpert == nil {
			return false
		}
		active++
	}
	return active >= 2 && active <= 4
}

type v41LayerExpertGroup struct {
	lease        *v41RoutedCohortLease
	lanes, slots []int
}

const v41LayerCohortMaxRows = 4 * V41RouterTopK

func closeV41LayerExpertGroups(groups []v41LayerExpertGroup) {
	var primary any
	for i := len(groups) - 1; i >= 0; i-- {
		if groups[i].lease == nil {
			continue
		}
		func() {
			defer func() {
				if r := recover(); r != nil && primary == nil {
					primary = r
				}
			}()
			groups[i].lease.Close()
		}()
	}
	if primary != nil {
		panic(primary)
	}
}

func finishV41LayerExpertGroups(groups *[]v41LayerExpertGroup, err *error, receipt *v41StepCohortReceipt) {
	primary := recover()
	owned := *groups
	*groups = nil
	if primary == nil && *err == nil {
		receipt.atLane(-1)
	}
	finishModelCleanup(primary, err, func() { closeV41LayerExpertGroups(owned) })
}

// All lanes prepare before expert selection. This intentionally changes when a
// peer preparation fault can be observed relative to another lane's experts;
// each lane's arithmetic order, journal, cancellation and commit remain private.
func (m *Model) advanceV41LayerCohort(scope *backendExecutionScope, lanes []v41StepCohortLane, txs []*v41StepTransaction, results []v41StepCohortResult, receipt *v41StepCohortReceipt, cancelled func(int) bool) (err error) {
	active := make([]int, 0, len(lanes))
	for lane, tx := range txs {
		if tx == nil || tx.closed || cancelled(lane) {
			continue
		}
		receipt.atLane(lane)
		if err = tx.prepareLayer(); err != nil {
			results[lane].err = err
			return err
		}
		active = append(active, lane)
	}
	// Cancellation during another lane's preparation withdraws only its own
	// continuation. No expert arithmetic has been selected yet.
	survivors := active[:0]
	for _, lane := range active {
		if !cancelled(lane) {
			survivors = append(survivors, lane)
		}
	}
	active = survivors
	if len(active) == 0 {
		return nil
	}

	fallback := func() error {
		for _, lane := range active {
			if cancelled(lane) {
				continue
			}
			receipt.atLane(lane)
			if cause := txs[lane].finishLayer(nil); cause != nil {
				results[lane].err = cause
				return cause
			}
		}
		return nil
	}
	if len(active) < 2 {
		return fallback()
	}
	first := txs[active[0]].pendingLayer
	if !first.full || len(first.picks) == 0 {
		return fallback()
	}
	// Build a deterministic first-seen expert union. Every selected lane keeps
	// its original slot and later routed-sum order; only cross-lane recording is
	// expert-major. The official four-lane/top-six envelope bounds both rows and
	// held expert triples before any arithmetic is selected.
	slots := make([]map[int]int, len(active))
	experts := make([]int, 0, len(active)*len(first.picks))
	seenExperts := make(map[int]bool, len(active)*len(first.picks))
	leader := lanes[active[0]].session
	if leader == nil || leader.M != m || leader.Backend == nil || leader.backendExecutionScope != scope {
		return fallback()
	}
	for row, lane := range active {
		work := txs[lane].pendingLayer
		session := lanes[lane].session
		if session == nil || session.M != leader.M || session.Backend != leader.Backend || session.backendExecutionScope != scope ||
			!work.full || work.layer != first.layer || len(work.picks) != len(first.picks) {
			return fallback()
		}
		slots[row] = make(map[int]int, len(work.picks))
		for slot, pick := range work.picks {
			if _, duplicate := slots[row][pick.expert]; duplicate {
				return fallback()
			}
			slots[row][pick.expert] = slot
			if !seenExperts[pick.expert] {
				seenExperts[pick.expert] = true
				experts = append(experts, pick.expert)
			}
		}
	}
	flatRows := len(active) * len(first.picks)
	if flatRows > v41LayerCohortMaxRows || len(experts) > v41LayerCohortMaxRows {
		return fallback()
	}
	var groups []v41LayerExpertGroup
	defer finishV41LayerExpertGroups(&groups, &err, receipt)
	for _, expert := range experts {
		rows := make([]v41RoutedCohortRow, 0, len(active))
		group := v41LayerExpertGroup{lanes: make([]int, 0, len(active)), slots: make([]int, 0, len(active))}
		for row, lane := range active {
			work := txs[lane].pendingLayer
			slot, participates := slots[row][expert]
			if !participates {
				continue
			}
			group.lanes = append(group.lanes, lane)
			group.slots = append(group.slots, slot)
			rows = append(rows, v41RoutedCohortRow{Lane: lane, Session: lanes[lane].session, Input: work.input, Weight: work.picks[slot].weight})
		}
		receipt.atLane(-1)
		lease, outcome, fault, cause := prepareV41RoutedExpertCohortOwned(scope, first.layer, "ffn.experts."+itoa(expert), rows)
		if lease != nil {
			group.lease = lease
			groups = append(groups, group)
		}
		if cause != nil {
			receipt.atLane(fault)
			if fault >= 0 && fault < len(results) {
				results[fault].err = cause
			}
			return cause
		}
		if outcome == v41RoutedExpertDeclined {
			// Failed preselection cleanup is a failure, never fallback consent.
			owned := groups
			groups = nil
			receipt.atLane(-1)
			closeV41LayerExpertGroups(owned)
			return fallback()
		}
		if outcome != v41RoutedExpertHandled || lease == nil {
			return v41ExpertOperationErr(first.layer, "cohort admission", errV41ExpertResult)
		}
	}
	// If membership changes before selection, retire every prepared lease and
	// preserve the ordinary whole-lane path for the remaining requests.
	for _, lane := range active {
		if cancelled(lane) {
			owned := groups
			groups = nil
			receipt.atLane(-1)
			closeV41LayerExpertGroups(owned)
			return fallback()
		}
	}
	leases := make([]*v41RoutedCohortLease, len(groups))
	flatSlots := make([][]int, len(lanes))
	for _, lane := range active {
		flatSlots[lane] = make([]int, len(first.picks))
		for slot := range flatSlots[lane] {
			flatSlots[lane][slot] = -1
		}
	}
	flat := 0
	for i, group := range groups {
		leases[i] = group.lease
		for row, lane := range group.lanes {
			flatSlots[lane][group.slots[row]] = flat
			flat++
		}
	}
	sums := make([]v41RoutedCohortSum, len(active))
	for row, lane := range active {
		order := v41RoutedSumOrder(txs[lane].pendingLayer.picks)
		sums[row] = v41RoutedCohortSum{Lane: lane, Rows: make([]int, len(order))}
		for i, slot := range order {
			flatRow := flatSlots[lane][slot]
			if flatRow < 0 {
				return v41ExpertOperationErr(first.layer, "cohort admission", errV41ExpertResult)
			}
			sums[row].Rows[i] = flatRow
		}
	}
	receipt.atLane(-1)
	routed, fault, cause := runV41RoutedCohortLeases(leases, sums)
	if cause != nil {
		receipt.atLane(fault)
		if fault >= 0 && fault < len(results) {
			results[fault].err = cause
		}
		return cause
	}
	if len(routed) != len(active) {
		return v41ExpertOperationErr(first.layer, "cohort sums", errV41ExpertResult)
	}
	// The selected contraction includes each lane's original device sum and a
	// single shared fence. Cancellation discards its speculative row; selected
	// experts are never replayed and no per-expert host readback is introduced.
	for row, lane := range active {
		if cancelled(lane) {
			continue
		}
		receipt.atLane(lane)
		if len(routed[row]) != m.Cfg.HiddenSize {
			return v41ExpertOperationErr(first.layer, "cohort sum", errV41ExpertResult)
		}
		if cause = txs[lane].finishLayer(routed[row]); cause != nil {
			results[lane].err = cause
			return cause
		}
	}
	return nil
}
