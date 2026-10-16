package compute

import "sync/atomic"

// V41DeferredReadResult carries the ordinary deferred read rows and, only for
// the canonical multirow reader, a transfer of their independent host storage.
// The grant is private: backend names and callback identities cannot mint it.
// A zero result has neither rows nor ownership. Copies share one consumption state.
// Concurrent takers, including copied result values, share the same atomic latch.
type V41DeferredReadResult struct {
	count int
	state *v41DeferredReadState
}

type v41DeferredReadState struct {
	taken       atomic.Bool
	rows        [][]float32
	independent bool
}

// RowCount is the immutable returned row extent, available before consumption.
// It carries no ownership authority; zero/error results report zero.
func (r V41DeferredReadResult) RowCount() int { return r.count }

// TakeRows consumes this result once. Callers must first resolve all numeric
// verdicts and validate the returned row count/geometry as on the legacy path.
// When independent is false, callers retain their defensive-copy behavior.
// When true, rows do not share backing storage with one another, device storage,
// a retained status prefix, or backend-owned scratch, and survive tensor Free.
func (r *V41DeferredReadResult) TakeRows() (rows [][]float32, independent bool) {
	if r == nil || r.state == nil || !r.state.taken.CompareAndSwap(false, true) {
		return nil, false
	}
	rows, independent = r.state.rows, r.state.independent
	r.state.rows = nil
	r.state.independent = false
	return rows, independent
}

// The unexported method seals minting to this package. An arbitrary external
// implementation of V41DeferredStatusBackend keeps the legacy copy contract.
type v41OwnedDeferredReader interface {
	v41ReadDeferredOwned([]V41DeferredStatus, []Tensor) ([]error, V41DeferredReadResult, error)
}

// ReadV41DeferredResult issues exactly one ordinary deferred read. Status
// consumption, fence behavior and numeric-verdict order remain unchanged.
// Transport errors carry no partial rows or ownership grant. Numeric verdicts
// remain distinct from transport errors and must be checked before TakeRows.
func ReadV41DeferredResult(backend V41DeferredStatusBackend, statuses []V41DeferredStatus, rows []Tensor) ([]error, V41DeferredReadResult, error) {
	if owned, ok := backend.(v41OwnedDeferredReader); ok {
		verdicts, result, err := owned.v41ReadDeferredOwned(statuses, rows)
		if err != nil {
			return nil, V41DeferredReadResult{}, err
		}
		return verdicts, result, nil
	}
	verdicts, values, err := backend.V41ReadDeferred(statuses, rows)
	if err != nil {
		return nil, V41DeferredReadResult{}, err
	}
	return verdicts, V41DeferredReadResult{count: len(values), state: &v41DeferredReadState{rows: values}}, nil
}
