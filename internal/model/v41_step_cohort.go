package model

import (
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var errV41StepCohortDeclined = errors.New("model: incompatible V4.1 step cohort")

type v41StepCohortLane struct {
	session *Session
	id      int
	ctx     context.Context
	round   *v41StepRound // Optional exact retained journal; ordinary callers leave nil.
}

type v41StepCohortResult struct {
	logits []float32
	stats  v41StepStats
	err    error
}

// This operation-local receipt survives an unwind independently of return-value
// assignment. A negative faultLane identifies a shared operation, not a lane.
type v41StepCohortReceipt struct {
	faultLane                int
	sharedPanels, sharedRows int
	windowDeclined           bool
}

func (r *v41StepCohortReceipt) atLane(lane int) {
	if r != nil {
		r.faultLane = lane
	}
}

// forwardV41StepCohort is an internal continuation driver, not a serving
// admission capability or a shared-weight projection path. Each lane keeps its
// own history, retained KV, source/index registry, Engram and mHC scratch.
//
// The caller supplies the SAME device-owner mutex used by its serial forwards,
// prefill and other cohorts, and grants exclusive ownership of every Session
// for the call. The owner must be unlocked on entry: TryLock declines a busy or
// already-held owner rather than deadlocking a caller that still holds its
// serial gate. One goroutine advances all lanes in layer-major order. This does
// not make concurrent Session.Step calls safe unless those calls already use
// that owner. No package-global or model-global lock is added.
//
// Admission examines the whole cohort before binding callbacks or beginning a
// transaction. A declined backend recording window is a refusal, never consent
// to enter somebody else's window. The layer window is flushed before ANY
// head can commit; heads then run synchronously under the same device owner.
// Cancellation aborts only its lane. An execution error aborts all remaining
// uncommitted lanes, and a panic retains its original type. Neither is replayed.
func (m *Model) forwardV41StepCohort(owner *sync.Mutex, lanes []v41StepCohortLane) (results []v41StepCohortResult, err error) {
	return m.forwardV41StepCohortWithHead(owner, lanes, false)
}

// forwardV41StepCohortSharedHead uses one actual backend projection over the
// prepared surviving rows. Unsupported head dtype/layout declines before Begin.
// The caller has the same exclusive-session and shared-device-owner obligations
// as forwardV41StepCohort; this is not yet a serving admission entrypoint.
func (m *Model) forwardV41StepCohortSharedHead(owner *sync.Mutex, lanes []v41StepCohortLane) ([]v41StepCohortResult, error) {
	return m.forwardV41StepCohortWithHead(owner, lanes, true)
}

func (m *Model) forwardV41StepCohortWithHead(owner *sync.Mutex, lanes []v41StepCohortLane, sharedHead bool) (results []v41StepCohortResult, err error) {
	if owner == nil {
		return nil, fmt.Errorf("%w: missing device owner", errV41StepCohortDeclined)
	}
	if !owner.TryLock() {
		return nil, fmt.Errorf("%w: device owner is already held", errV41StepCohortDeclined)
	}
	scope := &backendExecutionScope{owner: owner, live: true}
	defer scope.finishError(&err)
	if err := m.v41StepCohortAdmit(lanes); err != nil {
		return nil, err
	}
	if sharedHead {
		if err := m.v41SharedHeadAdmit(lanes); err != nil {
			return nil, err
		}
	}
	for _, lane := range lanes {
		if domain := backendExecutionOwner(lane.session.Backend); domain != nil && domain != owner {
			return nil, fmt.Errorf("%w: caller does not own the backend recording domain", errV41StepCohortDeclined)
		}
	}
	for _, lane := range lanes {
		scope.bind(lane.session)
	}
	results = make([]v41StepCohortResult, len(lanes))
	receipt := &v41StepCohortReceipt{faultLane: -1}
	err = m.forwardV41StepCohortOwned(scope, lanes, sharedHead, results, receipt)
	if receipt.windowDeclined {
		return nil, err
	}
	return results, err
}

// forwardV41StepCohortOwned does not acquire, bind, or release a domain. Both
// callers retain that responsibility, including boundary guards and teardown.
// Results belong to the caller so a later panic cannot hide an earlier commit.
func (m *Model) forwardV41StepCohortOwned(scope *backendExecutionScope, lanes []v41StepCohortLane, sharedHead bool, results []v41StepCohortResult, receipt *v41StepCohortReceipt) (err error) {
	if scope == nil || !scope.live || len(results) != len(lanes) {
		panic("model: invalid owned V4.1 cohort scope")
	}
	for _, lane := range lanes {
		if lane.session.backendExecutionScope != scope {
			panic("model: V4.1 cohort session is not bound to its owner")
		}
	}
	txs := make([]*v41StepTransaction, len(lanes))
	scratch := make([]v41ProjScratch, len(lanes))
	complete := false
	defer func() { v41AbortStepCohort(txs, results, &err, complete) }()
	err = func() (err error) {

		cancelled := func(lane int) bool {
			receipt.atLane(lane)
			if ctx := lanes[lane].ctx; ctx != nil && ctx.Err() != nil {
				results[lane].err = ctx.Err()
				if txs[lane] != nil {
					txs[lane].abort()
				}
				return true
			}
			return false
		}
		active := 0
		for lane := range lanes {
			if !cancelled(lane) {
				active++
			}
		}
		if active == 0 {
			return nil
		}

		// Close the recording window once on success, error or panic. Mark it
		// closed BEFORE End so even a panicking End cannot cause a duplicate End.
		var end func() error
		receipt.atLane(-1)
		if backend, ok := lanes[0].session.Backend.(compute.V41StepBatchBackend); ok && os.Getenv(v41StepBatchEnv) != "0" {
			if !backend.BeginV41StepBatch() {
				if receipt != nil {
					receipt.windowDeclined = true
				}
				return fmt.Errorf("%w: backend recording window is unavailable", errV41StepCohortDeclined)
			}
			end = backend.EndV41StepBatch
		}
		finish := func() error {
			if end == nil {
				return nil
			}
			closeWindow := end
			end = nil
			if err := closeWindow(); err != nil {
				return &v41StepBatchError{err: err}
			}
			return nil
		}
		windowComplete := false
		defer func() { v41FinishStepCohort(finish, &err, windowComplete) }()
		err = func() error {

			for lane, input := range lanes {
				if cancelled(lane) {
					continue
				}
				if input.round != nil {
					txs[lane], err = input.round.beginStep(input.id, &scratch[lane])
				} else {
					txs[lane], err = m.beginV41StepAdmitted(input.id, input.session.v41State(), &scratch[lane])
				}
				if err != nil {
					results[lane].err = err
					return err
				}
			}
			for layer := 0; layer < m.Cfg.NumLayers; layer++ {
				if m.v41LayerCohortEnabled(lanes, txs) {
					if err = m.advanceV41LayerCohort(scope, lanes, txs, results, receipt, cancelled); err != nil {
						return err
					}
					continue
				}
				for lane, tx := range txs {
					if tx == nil || tx.closed || cancelled(lane) {
						continue
					}
					if err = tx.advanceLayer(); err != nil {
						results[lane].err = err
						return err
					}
				}
			}
			receipt.atLane(-1)
			if err = finish(); err != nil {
				return err
			}
			if sharedHead {
				return m.v41FinishSharedHeads(lanes, txs, results, cancelled, receipt)
			}
			for lane, tx := range txs {
				if tx == nil || tx.closed || cancelled(lane) {
					continue
				}
				results[lane].logits, err = tx.finalize()
				if err != nil {
					results[lane].err = err
					return err
				}
			}
			return nil
		}()
		windowComplete = true
		return err
	}()
	complete = true
	return err
}

// Exact Model and pointer Backend identities bind configuration and immutable
// weights, rather than trusting equal names. Only session-created/restored
// states are supported; rejecting shared state objects catches an accidental
// shallow lane clone before any ring or source publication is touched.
func (m *Model) v41StepCohortAdmit(lanes []v41StepCohortLane) error {
	return m.v41StepCohortAdmitSize(lanes, 2)
}

// A retained batch may have a singleton tail wave. Ordinary cohort entry still
// requires two lanes; the same identity, cursor and journal checks apply here.
func (m *Model) v41StepCohortAdmitSize(lanes []v41StepCohortLane, minimum int) error {
	decline := func(reason string) error { return fmt.Errorf("%w: %s", errV41StepCohortDeclined, reason) }
	if m == nil || len(lanes) < minimum || !m.Cfg.IsDeepSeekV41() {
		return decline("need at least two V4.1 sessions")
	}
	if m.v41EPRequested() || m.EPDecodeCoordinated() {
		return decline("expert-parallel sessions require coordinated scalar entrypoints")
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		return err
	}
	if lanes[0].session == nil {
		return decline("nil session")
	}
	first := lanes[0].session
	if first.Backend != nil {
		v := reflect.ValueOf(first.Backend)
		if v.Kind() != reflect.Pointer || v.IsNil() {
			return decline("backend lacks a stable pointer identity")
		}
	}
	seenSessions := make(map[*Session]bool, len(lanes))
	seenStates := make(map[*v41ForwardState]bool, len(lanes))
	seenAttention := make(map[*V41AttentionState]bool, len(lanes)*(m.Cfg.NumLayers+1))
	for _, lane := range lanes {
		s := lane.session
		if s == nil || s.M != m || seenSessions[s] || s.Backend != first.Backend || s.BackendSessionClosed() {
			return decline("session, model or backend identity differs")
		}
		if s.Quant != first.Quant || s.Q4K != first.Q4K || s.KVPrecision != first.KVPrecision || s.executionPolicy != first.executionPolicy || s.DenseGPULayers != first.DenseGPULayers || s.GPULayers != first.GPULayers {
			return decline("session execution layout differs")
		}
		if lane.id < 0 || lane.id >= m.Cfg.VocabSize || !s.v41IncrementalEligible() {
			return decline("invalid token or unseeded incremental state")
		}
		st := s.v41Forward
		if r := lane.round; r != nil {
			if r.model != m || r.state != st || st.callbackOwner != s || !r.ownsState() || r.failed || r.poisoned ||
				r.pending != nil || len(r.entries) >= r.capacity || !r.atCommittedBoundary() {
				return decline("retained round identity or boundary differs")
			}
		} else if st.activeStepRound != nil {
			return decline("active retained round requires explicit association")
		}
		if seenStates[st] {
			return decline("lanes share a continuation state")
		}
		seenSessions[s], seenStates[st] = true, true
		if err := v41CandidatePoolCovers(m.Cfg, len(st.history)+1); err != nil {
			return err
		}
		if st.attn != nil {
			if seenAttention[st.attn] {
				return decline("lanes share a source registry")
			}
			seenAttention[st.attn] = true
		}
		for layer := 0; layer < m.Cfg.NumLayers; layer++ {
			state := st.layerState(layer)
			if seenAttention[state] || state.nextWindowPos != len(st.history) {
				return decline("retained layer is shared or has a stale cursor")
			}
			seenAttention[state] = true
		}
	}
	return nil
}

// v41AbortStepCohort drains every lane even when releasing one lane's resources
// panics. It preserves an existing panic or returned error; absent either, the
// first cleanup panic escapes after all journals and stats have been processed.
func v41AbortStepCohort(txs []*v41StepTransaction, results []v41StepCohortResult, err *error, complete bool) {
	failed := !complete || *err != nil
	var cleanupPanic any
	for lane, tx := range txs {
		if tx == nil {
			continue
		}
		func() {
			defer func() {
				if r := recover(); r != nil && cleanupPanic == nil {
					cleanupPanic = r
				}
			}()
			tx.abort()
		}()
		results[lane].stats = tx.stats
	}
	if !failed && cleanupPanic != nil {
		panic(cleanupPanic)
	}
}

// v41FinishStepCohort closes the recording window once while preserving an
// earlier returned error as well as an earlier panic. A standalone End panic
// still escapes and is handled by the outer all-lane abort and owner release.
func v41FinishStepCohort(finish func() error, err *error, complete bool) {
	if !complete {
		defer func() { _ = recover() }()
		_ = finish()
		return
	}
	if *err != nil {
		defer func() { _ = recover() }()
	}
	if closeErr := finish(); closeErr != nil {
		*err = errors.Join(*err, closeErr)
	}
}
