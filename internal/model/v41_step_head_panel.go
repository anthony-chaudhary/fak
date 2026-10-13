package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41SharedHeadBackend explicitly promises an implemented BatchedMatMul for
// this exact dtype/layout and the same per-row reduction as MatMul for every
// row count in [1, maxRows], including the device's shape/grid/resource limits.
// Scalar dtype support alone is not such a promise.
// Physical backend implementations must qualify that contract separately.
const v41SharedHeadMaxLanes = 4

type v41SharedHeadBackend interface {
	V41SharedHeadAvailable(dtype compute.Dtype, out, in, maxRows int) bool
}

func (m *Model) v41SharedHeadAdmit(lanes []v41StepCohortLane) error {
	decline := func() error {
		return fmt.Errorf("%w: shared head dtype, layout or row-reduction capability is unsupported", errV41StepCohortDeclined)
	}
	if len(lanes) > v41SharedHeadMaxLanes {
		return decline()
	}
	_, inputOK := checkedMulInt(len(lanes), m.Cfg.HiddenSize)
	_, outputOK := checkedMulInt(len(lanes), m.Cfg.VocabSize)
	if !inputOK || !outputOK {
		return decline()
	}
	s := lanes[0].session
	dtype, supported := s.v41HeadProjectionDtype(m.headName(), m.Cfg.VocabSize, m.Cfg.HiddenSize)
	if !supported || (dtype != compute.F32 && dtype != compute.Q2_K && dtype != compute.Q6_K) {
		return decline()
	}
	backend, ok := s.Backend.(v41SharedHeadBackend)
	if !ok || !backend.V41SharedHeadAvailable(dtype, m.Cfg.VocabSize, m.Cfg.HiddenSize, len(lanes)) {
		return decline()
	}
	return nil
}

// v41FinishSharedHeads runs after the cohort's layer recording window flushes.
// Lane-local preparation preserves mHC collapse and the final BF16/RMSNorm
// boundaries; only the resulting independent rows share the immutable head.
func (m *Model) v41FinishSharedHeads(lanes []v41StepCohortLane, txs []*v41StepTransaction, results []v41StepCohortResult, cancelled func(int) bool, receipt *v41StepCohortReceipt) error {
	indices := make([]int, 0, len(lanes))
	panel := make([]float32, 0, len(lanes)*m.Cfg.HiddenSize)
	for lane, tx := range txs {
		if tx == nil || tx.closed || cancelled(lane) {
			continue
		}
		row, err := tx.prepareHead()
		if err != nil {
			results[lane].err = err
			return err
		}
		if cancelled(lane) {
			continue
		}
		indices = append(indices, lane)
		panel = append(panel, row...)
	}
	if len(indices) == 0 {
		return nil
	}
	receipt.atLane(-1)
	projected, err := m.v41ProjectSharedHead(lanes, txs, indices, panel)
	if err != nil {
		// Once selected, a decline/error is fatal. There is no scalar host
		// retry and none of this panel's lanes have committed a history row.
		panic(err)
	}
	// This records a completed projection, not admission or token commits.
	// Cancellation during projection or later scaling cannot erase real work.
	if receipt != nil && len(indices) > 1 {
		receipt.sharedPanels++
		receipt.sharedRows += len(indices)
	}
	for row, lane := range indices {
		if cancelled(lane) {
			continue
		}
		lo, hi := row*m.Cfg.VocabSize, (row+1)*m.Cfg.VocabSize
		var beforeCommit func() error
		if ctx := lanes[lane].ctx; ctx != nil {
			beforeCommit = ctx.Err
		}
		results[lane].logits, err = txs[lane].finishHead(projected[lo:hi:hi], beforeCommit)
		if err != nil {
			results[lane].err = err
			// A cancellation raised during this lane's scaling is observed
			// after scaling and before history append. Its peers may finish.
			if ctx := lanes[lane].ctx; ctx != nil && ctx.Err() != nil && errors.Is(err, ctx.Err()) {
				continue
			}
			return err
		}
	}
	return nil
}

func (m *Model) v41ProjectSharedHead(lanes []v41StepCohortLane, txs []*v41StepTransaction, indices []int, panel []float32) (projected []float32, err error) {
	// The selected callback closes its own Session on a backend fault. The
	// shared operation also invalidates every other participant. Roll back and
	// retire each one, retaining the primary error/panic even if teardown fails.
	retire := func(primary any) {
		cause, ok := primary.(error)
		if !ok {
			cause = fmt.Errorf("shared head panic: %v", primary)
		}
		for _, lane := range indices {
			func() {
				defer func() { _ = recover() }()
				txs[lane].abort()
			}()
			func() {
				defer func() { _ = recover() }()
				s := lanes[lane].session
				if !s.BackendSessionClosed() {
					s.v41HeadCloseFailure("shared projection", cause)
				}
			}()
		}
	}
	defer func() {
		if primary := recover(); primary != nil {
			retire(primary)
			panic(primary)
		}
		if err != nil {
			retire(err)
		}
	}()
	// Model/backend identity was admitted for all lanes. The first surviving
	// session owns this operation; its ordinary head callback stages one shared
	// weight handle, executes BatchedMatMul when rows > 1 and validates readback.
	project := lanes[indices[0]].session.v41DenseProjectionFunc()
	selected := func(layer int, leaf string, input []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		if project == nil {
			return nil, v41ProjectionError, errV41ProjectionResult
		}
		y, outcome, cause := project(layer, leaf, input, out, in, rows)
		if outcome != v41ProjectionHandled {
			if cause == nil {
				cause = errV41ProjectionResult
			}
			return nil, v41ProjectionError, cause
		}
		return y, outcome, cause
	}
	return m.v41ProjectHeadInput(panel, selected, len(indices))
}
