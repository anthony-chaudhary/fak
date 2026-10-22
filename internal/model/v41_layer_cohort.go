package model

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41LayerStepWork is a lane-owned continuation, never a shallow Session copy.
// Attention/router preparation may change journaled source state, but hidden
// state/window publication and the final token commit remain with this lane.
type v41LayerStepWork struct {
	layer              int
	input              []float32
	picks              []routePick
	full               bool
	roleAttentionState v41RoleAttentionState
	scratch            *v41ProjScratch
	scalar             func() ([]float32, error)
	finish             func([]float32) error
	cleanup            func()
	closed             bool
	// publishKV commits this position's attention window row before the MoE
	// finish, so a later position of the same session can attend to it while
	// this position's experts are still pending. finish skips the commit once
	// it ran; nil means the work cannot publish early.
	publishKV   func() error
	kvPublished bool
}

func (w *v41LayerStepWork) publishEarly() error {
	if w == nil || w.closed || w.publishKV == nil {
		return fmt.Errorf("%w: layer continuation cannot publish its window row early", ErrV41ForwardStage)
	}
	if w.kvPublished {
		return nil
	}
	if err := w.publishKV(); err != nil {
		return err
	}
	w.kvPublished = true
	return nil
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

func (w *v41LayerStepWork) finishCleanup(err *error, complete bool) {
	failed := !complete || *err != nil
	if failed {
		defer func() { _ = recover() }()
	}
	w.close()
}

func (w *v41LayerStepWork) complete(routed []float32) (err error) {
	if w == nil || w.closed {
		return fmt.Errorf("%w: layer continuation is closed", ErrV41ForwardStage)
	}
	complete := false
	defer func() { w.finishCleanup(&err, complete) }()
	err = func() error {
		if routed == nil {
			routed, err = w.scalar()
			if err != nil {
				return err
			}
		}
		return w.finish(routed)
	}()
	complete = true
	return err
}

// The existing explicit decode ID route and disabled layer batching retain precedence.
// The independent cold-prefill panel opt-in does not select a decode route.
// The new schedule is confined to the already-selected multi-session caller;
// single-lane wrappers execute their unchanged scalar/fused continuation.
func (m *Model) v41LayerCohortEnabled(lanes []v41StepCohortLane, txs []*v41StepTransaction) bool {
	if v41RoutedIDOptedIn() {
		return false
	}
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil || !full {
		return false
	}
	active := 0
	experts := os.Getenv(v41LayerExpertBatchEnv) != "0" && v41CohortExpertUnionSetting() != v41CohortExpertUnionOff
	for lane, tx := range txs {
		if tx == nil || tx.closed {
			continue
		}
		if lanes[lane].session.v41State().routedExpert == nil {
			experts = false
		}
		active++
	}
	return active >= 2 && active <= 4 && (experts || m.v41GroupedDecodeEnabled(lanes, txs))
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

func finishV41LayerExpertGroups(groups *[]v41LayerExpertGroup, err *error, receipt *v41StepCohortReceipt, complete bool) {
	failed := !complete || *err != nil
	owned := *groups
	*groups = nil
	if !failed {
		receipt.atLane(-1)
	} else {
		// Preserve the caller's unwind and lane attribution. This recovery
		// handles only secondary group-cleanup panics, never the caller.
		defer func() { _ = recover() }()
	}
	closeV41LayerExpertGroups(owned)
}

// All lanes prepare before expert selection. This intentionally changes when a
// peer preparation fault can be observed relative to another lane's experts;
// each lane's arithmetic order, journal, cancellation and commit remain private.
func (m *Model) advanceV41LayerCohort(scope *backendExecutionScope, lanes []v41StepCohortLane, txs []*v41StepTransaction, results []v41StepCohortResult, receipt *v41StepCohortReceipt, cancelled func(int) bool) (err error) {
	var attention *[4]v41LayerAttentionSuspension
	if m.v41GroupedDecodeEnabled(lanes, txs) {
		attention = new([4]v41LayerAttentionSuspension)
	}
	active := make([]int, 0, len(lanes))
	for lane, tx := range txs {
		if tx == nil || tx.closed || cancelled(lane) {
			continue
		}
		receipt.atLane(lane)
		if attention != nil {
			err = tx.prepareAttention(&attention[lane])
		} else {
			err = tx.prepareLayer()
		}
		if err != nil {
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
	if attention != nil {
		if err := m.resumeV41CohortAttention(scope, lanes, txs, active, results, receipt); err != nil {
			return err
		}
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
	// Dense attention batching does not imply routed-expert support.
	mode := v41CohortExpertUnionSetting()
	if os.Getenv(v41LayerExpertBatchEnv) == "0" || mode == v41CohortExpertUnionOff {
		return fallback()
	}
	for _, lane := range active {
		if lanes[lane].session.v41State().routedExpert == nil {
			return fallback()
		}
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
		v41CohortExpertUnionStats.declinedCapacity.Add(1)
		return fallback()
	}
	shared := flatRows - len(experts)
	observeV41CohortSharedRows(shared)
	if !mode.admitsSharedRows(shared) {
		v41CohortExpertUnionStats.declinedOverlap.Add(1)
		return fallback()
	}
	var groups []v41LayerExpertGroup
	complete := false
	defer func() { finishV41LayerExpertGroups(&groups, &err, receipt, complete) }()
	err = func() error {
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
				v41CohortExpertUnionStats.declinedAdmission.Add(1)
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
		v41CohortExpertUnionStats.selected.Add(1)
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
	}()
	complete = true
	return err
}

const v41GroupedDecodeEnv = "FAK_V41_GROUPED_DECODE"

// This opt-in selects no work by itself. The synchronous cohort owner performs
// exact model/session/geometry and per-weight admission before device selection.
func (m *Model) v41GroupedDecodeEnabled(lanes []v41StepCohortLane, txs []*v41StepTransaction) bool {
	if os.Getenv(v41GroupedDecodeEnv) != "1" || len(lanes) > 4 || m == nil || m.lora != nil || m.prism != nil {
		return false
	}
	active := 0
	for i, tx := range txs {
		if tx == nil || tx.closed {
			continue
		}
		if i >= len(lanes) || lanes[i].session == nil {
			return false
		}
		s := lanes[i].session
		if s.M != m || s.Backend == nil {
			return false
		}
		if grouped, publication := v41GroupedDecodeBindings(s); grouped == nil || publication == nil {
			return false
		}
		if _, ok := s.Backend.(compute.V41RowLogBackend); !ok {
			return false
		}
		if _, ok := s.Backend.(compute.V41BF16PublishBackend); !ok {
			return false
		}
		if _, ok := s.Backend.(compute.V41DeferredStatusBackend); !ok {
			return false
		}
		active++
	}
	return active == 2 || active == 4
}

func (m *Model) resumeV41CohortAttention(scope *backendExecutionScope, lanes []v41StepCohortLane, txs []*v41StepTransaction, active []int, results []v41StepCohortResult, receipt *v41StepCohortReceipt) error {
	var output [][]float32
	grouped := false
	if len(active) == 2 || len(active) == 4 {
		var rows [4]v41GroupedOutputCohortRow
		for row, lane := range active {
			rows[row] = v41GroupedOutputCohortRow{Lane: lane, Session: lanes[lane].session, Input: txs[lane].pendingAttention.input()}
		}
		receipt.atLane(-1)
		var groupedBindings [4]*v41GroupedOutputBinding
		var publications [4]*v41BF16PublicationBinding
		canonical := true
		for row, lane := range active {
			groupedBindings[row], publications[row] = v41GroupedDecodeBindings(lanes[lane].session)
			a := txs[lane].pendingAttention
			scratch := a.plainScratch()
			if groupedBindings[row] == nil || publications[row] == nil || scratch == nil || scratch.groupedOutput != groupedBindings[row] || scratch.gatherOps != lanes[lane].session.v41Forward.gatherOps {
				canonical = false
			}
		}
		var owner *v41GroupedOutputCohort
		outcome := v41ProjectionDeclined
		var err error
		if canonical {
			owner, outcome, err = prepareV41GroupedOutputCohort(scope, txs[active[0]].nextLayer, rows[:len(active)])
		}
		if err == nil && owner != nil {
			output, outcome, err = owner.Run()
			if err == nil {
				for row, lane := range active {
					g, p := v41GroupedDecodeBindings(lanes[lane].session)
					scratch := txs[lane].pendingAttention.plainScratch()
					if g != groupedBindings[row] || p != publications[row] || scratch.groupedOutput != g || scratch.gatherOps != lanes[lane].session.v41Forward.gatherOps {
						err = fmt.Errorf("%w: grouped decode projection provenance changed", ErrV41ForwardStage)
						if owner.selected {
							err = owner.retire(err)
						}
						output = nil
						break
					}
				}
			}
		}
		if err != nil {
			for _, lane := range active {
				results[lane].err = err
			}
			return err
		}
		grouped = outcome == v41ProjectionHandled
		if grouped && len(output) != len(active) {
			err = errV41ProjectionResult
			if owner != nil {
				err = owner.retire(err)
			}
			for _, lane := range active {
				results[lane].err = err
			}
			return err
		}
		if !grouped && outcome != v41ProjectionDeclined {
			return errV41ProjectionResult
		}
	}
	// The ordered participant set is fixed through Run and resume. Cancellation
	// is observed again by the existing expert/finish path, which rolls back
	// that lane before commit. A selected failure returned above never resumes.
	for row, lane := range active {
		receipt.atLane(lane)
		var projected []float32
		if grouped {
			projected = output[row]
		}
		if err := txs[lane].resumeAttention(projected, grouped); err != nil {
			results[lane].err = err
			return err
		}
	}
	return nil
}

// Both publications skipped by panel composition must be the canonical owner-
// bound operations. Legacy wrappers, custom callbacks and device-only wrappers
// deliberately carry no grouped authority; they retain their scalar behavior.
func v41GroupedDecodeBindings(s *Session) (*v41GroupedOutputBinding, *v41BF16PublicationBinding) {
	if s == nil || s.v41Forward == nil || s.v41Forward.callbackOwner != s {
		return nil, nil
	}
	st := s.v41Forward
	grouped, ok := st.groupedOutput.(*v41GroupedOutputBinding)
	if !ok || !grouped.canonical(s) || st.gatherOps == nil || st.gatherOps.owner != s {
		return nil, nil
	}
	publication := st.gatherOps.bf16Publish
	if publication == nil || publication.ordinary == nil || publication.core == nil || publication.owner != s || publication.model != s.M || publication.backend != s.Backend {
		return nil, nil
	}
	return grouped, publication
}
