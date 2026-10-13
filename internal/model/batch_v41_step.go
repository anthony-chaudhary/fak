package model

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var (
	// ErrV41BatchUnavailable is a preselection refusal. A caller may separately
	// choose public Session.Step, which must acquire its own execution owner.
	ErrV41BatchUnavailable = errors.New("model: V4.1 shared-head batch unavailable")
	// ErrV41BatchBusy means no domain was acquired and no forward was attempted.
	// It is a scheduling deferral, never permission to run an unowned fallback.
	ErrV41BatchBusy = errors.New("model: V4.1 backend execution owner is busy")
	// ErrV41BatchAborted identifies an uncommitted peer stopped by a batch-level
	// failure. The call's error carries the cause without blaming that peer.
	ErrV41BatchAborted = errors.New("model: V4.1 batch peer aborted")
)

// V41BatchLane identifies one chosen-token continuation. IDs must be unique in
// the call; they are returned unchanged, independent of cancellation compaction.
// A nil Context has the same meaning as context.Background(). The caller must
// retain exclusive Session use until this synchronous operation returns.
type V41BatchLane struct {
	ID      uint64
	Context context.Context
	Session *Session
	Token   int
}

// V41BatchLaneResult describes the actual transaction outcome. Committed is
// authoritative even on failure: an already-appended token must not be replayed.
// A failed cleanup can leave Committed true with nil Logits and a non-nil Err.
type V41BatchLaneResult struct {
	ID        uint64
	Logits    []float32
	Err       error
	Committed bool
}

// V41BatchResult preserves input order and an operation-local execution receipt.
// Selected forbids replay of the call even if all lanes were rolled back.
// SharedHeadPanels/Rows count completed rows>=2 projections, including rows
// canceled after projection. They exclude scalar heads and failed projections;
// they are neither an admission estimate nor a physical-throughput measurement.
type V41BatchResult struct {
	Lanes            []V41BatchLaneResult
	Selected         bool
	SharedHeadPanels int
	SharedHeadRows   int
}

// StepV41Batch attempts one shared-head continuation of two to four compatible,
// seeded V4.1 Sessions. It requires both a real backend recording-domain owner
// and explicit exact-shape shared-head capability. It never performs fallback.
// Existing StepBatch/StepBatchActive and serving admission are unchanged.
//
// Cancellation is checked under ownership and at transaction boundaries; this
// call always joins synchronous device work and rollback before releasing the
// owner. Selected operation errors become results after cleanup. Unclassified
// panics retain their original identity and never authorize replay.
func (m *Model) StepV41Batch(input []V41BatchLane) (out V41BatchResult, err error) {
	out.Lanes = make([]V41BatchLaneResult, len(input))
	lanes := make([]v41StepCohortLane, len(input))
	seenIDs := make(map[uint64]bool, len(input))
	seenSessions := make(map[*Session]bool, len(input))
	for i, lane := range input {
		out.Lanes[i].ID = lane.ID
		lanes[i] = v41StepCohortLane{session: lane.Session, id: lane.Token, ctx: lane.Context}
	}
	if m == nil || !m.Cfg.IsDeepSeekV41() || len(input) < 2 || len(input) > v41SharedHeadMaxLanes {
		return out, fmt.Errorf("%w: need two to four V4.1 sessions", ErrV41BatchUnavailable)
	}
	for _, lane := range input {
		if lane.Session == nil || lane.Session.M != m || seenIDs[lane.ID] || seenSessions[lane.Session] || lane.Token < 0 || lane.Token >= m.Cfg.VocabSize {
			return out, fmt.Errorf("%w: invalid or aliased lane", ErrV41BatchUnavailable)
		}
		seenIDs[lane.ID], seenSessions[lane.Session] = true, true
	}
	owner := backendExecutionOwner(input[0].Session.Backend)
	if owner == nil {
		return out, fmt.Errorf("%w: backend has no execution owner", ErrV41BatchUnavailable)
	}
	for _, lane := range input[1:] {
		if backendExecutionOwner(lane.Session.Backend) != owner {
			return out, fmt.Errorf("%w: backend recording domains differ", ErrV41BatchUnavailable)
		}
	}
	if !owner.TryLock() {
		return out, ErrV41BatchBusy
	}
	scope := &backendExecutionScope{owner: owner, live: true}
	defer scope.finishError(&err)
	if cause := m.v41StepCohortAdmit(lanes); cause != nil {
		return out, fmt.Errorf("%w: %w", ErrV41BatchUnavailable, cause)
	}
	if cause := m.v41SharedHeadAdmit(lanes); cause != nil {
		return out, fmt.Errorf("%w: %w", ErrV41BatchUnavailable, cause)
	}
	active := 0
	for i, lane := range input {
		if lane.Context != nil && lane.Context.Err() != nil {
			out.Lanes[i].Err = lane.Context.Err()
		} else {
			active++
		}
	}
	if active == 0 {
		return out, nil
	}
	for _, lane := range input {
		scope.bind(lane.Session)
	}
	out.Selected = true
	results := make([]v41StepCohortResult, len(input))
	receipt := &v41StepCohortReceipt{faultLane: -1}
	err = m.stepV41BatchOwned(scope, lanes, results, receipt)
	out.SharedHeadPanels, out.SharedHeadRows = receipt.sharedPanels, receipt.sharedRows
	for i, result := range results {
		lane := &out.Lanes[i]
		lane.Logits, lane.Committed, lane.Err = result.logits, result.stats.Committed, result.err
		if lane.Err == nil && !lane.Committed && err != nil {
			lane.Err = ErrV41BatchAborted
		}
	}
	return out, err
}

// Boundary duties mirror Step -> stepV41, inside the same device scope as the
// cohort. Only the faulting lane receives the selected-indexer guard; applying
// its panic to every Session would retire innocent, possibly committed peers.
func (m *Model) stepV41BatchOwned(scope *backendExecutionScope, lanes []v41StepCohortLane, results []v41StepCohortResult, receipt *v41StepCohortReceipt) (err error) {
	keepAlive := make([]func(), 0, len(lanes))
	defer v41BatchEndKeepAlive(&keepAlive, &err)
	guarded := make([]bool, len(lanes))
	defer func() {
		primary := recover()
		lane := receipt.faultLane
		// Serial stepV41 panics with returned selected-indexer errors before
		// its guard runs. Reproduce that boundary after the cohort rollback.
		var selectedScore *V41IndexerScoreOperationError
		if primary == nil && errors.As(err, &selectedScore) {
			primary = err
		}
		if primary != nil {
			if lane >= 0 && lane < len(lanes) && guarded[lane] {
				primary = v41BatchIndexerGuard(lanes[lane].session, primary)
			}
			cause, ok := primary.(error)
			if !ok || !v41BatchSelectedError(cause) {
				panic(primary)
			}
			err = cause
		}
		if err != nil && lane >= 0 && lane < len(results) {
			results[lane].err = err
		}
	}()
	for i, lane := range lanes {
		receipt.atLane(i)
		if lane.ctx != nil && lane.ctx.Err() != nil {
			results[i].err = lane.ctx.Err()
			continue
		}
		s := lane.session
		if cause := s.refuseUnimplementedV4FlashAttention(); cause != nil {
			return cause
		}
		keepAlive = append(keepAlive, s.BeginGPUKeepAlive())
		s.validateDeviceOnlyExecution("decode")
		s.ensureOpenBackendSession()
		guarded[i] = true
		if cause := m.v41ForwardAdmitted(); cause != nil {
			return cause
		}
	}
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	defer func() {
		defer m.v41SetExpertFaultPhase(V41PhaseUnknown)
		// The owned core has already drained every transaction, so this remains
		// accurate when a later lane failed after an earlier history append.
		for _, result := range results {
			if result.stats.Committed {
				m.v41NoteExpertFaultToken(1)
				if v41IncrementalStepProbe != nil {
					v41IncrementalStepProbe()
				}
			}
		}
	}()
	return m.forwardV41StepCohortOwned(scope, lanes, true, results, receipt)
}

func v41BatchIndexerGuard(s *Session, primary any) (out any) {
	defer func() { out = recover() }()
	defer s.v41IndexerScoreGuard()
	panic(primary)
}

// Keep all scopes balanced without replacing an earlier operation failure.
func v41BatchEndKeepAlive(ends *[]func(), err *error) {
	primary := recover()
	var cleanup any
	for i := len(*ends) - 1; i >= 0; i-- {
		func() {
			defer func() {
				if r := recover(); r != nil && cleanup == nil {
					cleanup = r
				}
			}()
			(*ends)[i]()
		}()
	}
	*ends = nil
	if primary != nil {
		panic(primary)
	}
	if cleanup != nil && *err == nil {
		panic(cleanup)
	}
}

func v41BatchSelectedError(err error) bool {
	var backend *BackendForwardOperationError
	var device *compute.BackendError
	var batch *v41StepBatchError
	var expert *V41ExpertOperationError
	var projection *V41ProjectionOperationError
	var rope *V41TailRoPEOperationError
	var attention *V41SharedAttentionOperationError
	var score *V41IndexerScoreOperationError
	var mhc *V41MHCOperationError
	return errors.As(err, &backend) || errors.As(err, &device) || errors.As(err, &batch) ||
		errors.As(err, &expert) || errors.As(err, &projection) || errors.As(err, &rope) ||
		errors.As(err, &attention) || errors.As(err, &score) || errors.As(err, &mhc) ||
		errors.Is(err, ErrV41BF16BoundaryOperation) || errors.Is(err, ErrV41LogitScaleOperation) ||
		errors.Is(err, ErrV41IndexWeightScaleOperation) || errors.Is(err, ErrV41QueryPrefixOperation)
}
