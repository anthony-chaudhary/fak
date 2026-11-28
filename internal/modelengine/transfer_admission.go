package modelengine

import (
	"errors"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// Adapted from the vLLM cap on in-flight async KV loads (vllm-project/vllm,
// pinned at revision
// vllm/config/scheduler.py and
// vllm/distributed/kv_transfer/kv_connector/v1/base.py, Apache-2.0).
//
// Upstream, the scheduler may start a new async KV load (an offload restore or
// a remote fetch) even when the running batch has already reached its sequence
// cap, because the load runs on the transfer engine, not the compute batch.
// That freedom is bounded by a SEPARATE cap on the number of in-flight loads:
// without it a burst of parked requests could each start a transfer and
// saturate the transfer engine, starving the compute batch of the KV it needs.
// The two ceilings are independent by construction — a large sequence cap must
// not implicitly raise the transfer cap.
//
// Re-expressed in idiomatic Go; no upstream code copied. This is the scheduler
// component only: it makes no hardware throughput claim and performs no I/O.
// It reuses the package's existing admission verdict vocabulary
// (cachemeta.AdmissionVerdict) rather than inventing a parallel one.

// ErrTransferAdmissionConfig is returned when either budget is non-positive.
var ErrTransferAdmissionConfig = errors.New("modelengine: transfer admission budgets must be positive")

// TransferRequest is one admission attempt. NeedsTransfer marks a request whose
// KV must be fetched (offload restore or remote fetch) before it can run; such
// a request must hold an in-flight transfer slot.
type TransferRequest struct {
	ID            string
	NeedsTransfer bool
}

// TransferAdmissionDecision is the outcome of one Admit call, or of a release
// that promoted a waiter.
type TransferAdmissionDecision struct {
	// ID names the request the decision is about, so a release that promotes a
	// waiter says which one now holds the slot.
	ID string
	// Verdict is Allow when the request may proceed (it holds every slot it
	// needs) and Defer when it must wait for a slot to free.
	Verdict cachemeta.AdmissionVerdict
	// ExecutionSlot reports whether the request occupies a compute-batch slot.
	ExecutionSlot bool
	// TransferSlot reports whether the request holds an in-flight transfer slot.
	TransferSlot bool
}

// TransferAdmissionStats is the live view of both budgets.
type TransferAdmissionStats struct {
	ExecutionUsed int
	ExecutionCap  int
	TransferUsed  int
	TransferCap   int
	Waiters       int
	Admitted      uint64
	Deferred      uint64
}

// transferWaiter is one deferred request held in FIFO order until a slot frees.
type transferWaiter struct {
	req TransferRequest
}

// TransferAdmission splits admission into two independently capped budgets:
// execution slots (how many requests may occupy the compute batch) and
// in-flight KV-transfer slots (how many KV transfers may be outstanding at
// once). A request that needs a transfer must hold a transfer slot; when the
// transfer budget is exhausted admission defers rather than over-admitting.
// The two caps are independent: a large execution-slot count never raises the
// transfer cap, a full compute batch never defers a transfer start (the
// request prefetches its KV while the GPU stays saturated), and a transfer
// start never takes an execution slot, so loading requests cannot crowd a
// runnable one out of the batch.
//
// A request whose transfer finished calls ReleaseTransfer and is admitted again
// without NeedsTransfer to take its execution slot. Callers release only the
// slots their decision reported holding.
//
// All methods are safe for concurrent use.
type TransferAdmission struct {
	mu sync.Mutex

	execCap      int
	transferCap  int
	execUsed     int
	transferUsed int

	waiters  []*transferWaiter
	admitted uint64
	deferred uint64
}

// NewTransferAdmission creates an admission gate with executionSlots compute
// slots and transferSlots in-flight transfer slots. Both must be positive; a
// zero or negative budget is a configuration error, not an unbounded budget.
func NewTransferAdmission(executionSlots, transferSlots int) (*TransferAdmission, error) {
	if executionSlots <= 0 || transferSlots <= 0 {
		return nil, fmt.Errorf("%w: execution=%d transfer=%d", ErrTransferAdmissionConfig, executionSlots, transferSlots)
	}
	return &TransferAdmission{execCap: executionSlots, transferCap: transferSlots}, nil
}

// Admit attempts to admit one request. A request that needs a transfer is
// deferred when the transfer budget is exhausted; a request that does not need
// a transfer is deferred when the execution budget is exhausted. A deferred
// request consumes no slot and is queued for promotion by ReleaseExecution or
// ReleaseTransfer.
func (a *TransferAdmission) Admit(req TransferRequest) TransferAdmissionDecision {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.admitLocked(req)
}

func (a *TransferAdmission) admitLocked(req TransferRequest) TransferAdmissionDecision {
	if !a.canGrantLocked(req) {
		a.deferred++
		a.waiters = append(a.waiters, &transferWaiter{req: req})
		return TransferAdmissionDecision{ID: req.ID, Verdict: cachemeta.AdmissionDefer}
	}
	return a.grantLocked(req)
}

// canGrantLocked reports whether req can be admitted right now. The transfer
// budget gates only transfer-needing requests and the execution budget gates
// only non-transfer requests, so the two caps never widen each other.
func (a *TransferAdmission) canGrantLocked(req TransferRequest) bool {
	if req.NeedsTransfer && a.transferUsed >= a.transferCap {
		return false
	}
	if !req.NeedsTransfer && a.execUsed >= a.execCap {
		return false
	}
	return true
}

// grantLocked consumes the one slot req needs: a transfer slot for a request
// still loading its KV, an execution slot for a runnable one. Keeping a loading
// request out of the execution budget is what lets a full batch admit a
// transfer start and stops transfer starts from deferring runnable requests.
// Caller holds a.mu and has checked canGrantLocked.
func (a *TransferAdmission) grantLocked(req TransferRequest) TransferAdmissionDecision {
	d := TransferAdmissionDecision{ID: req.ID, Verdict: cachemeta.AdmissionAllow}
	if req.NeedsTransfer {
		a.transferUsed++
		d.TransferSlot = true
	} else {
		a.execUsed++
		d.ExecutionSlot = true
	}
	a.admitted++
	return d
}

// ReleaseExecution frees one compute-batch slot and promotes the first waiter
// that can now be admitted. ok is false when no waiter could be promoted.
func (a *TransferAdmission) ReleaseExecution() (TransferAdmissionDecision, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.execUsed > 0 {
		a.execUsed--
	}
	return a.promoteLocked()
}

// ReleaseTransfer frees one in-flight transfer slot and promotes the first
// waiter that can now be admitted. ok is false when no waiter could be promoted.
func (a *TransferAdmission) ReleaseTransfer() (TransferAdmissionDecision, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.transferUsed > 0 {
		a.transferUsed--
	}
	return a.promoteLocked()
}

// promoteLocked removes and grants the first waiter that can be admitted now,
// skipping (not breaking on) waiters still blocked by their own budget. Caller
// holds a.mu.
func (a *TransferAdmission) promoteLocked() (TransferAdmissionDecision, bool) {
	for i, w := range a.waiters {
		if a.canGrantLocked(w.req) {
			a.waiters = append(a.waiters[:i], a.waiters[i+1:]...)
			return a.grantLocked(w.req), true
		}
	}
	return TransferAdmissionDecision{}, false
}

// Stats returns the live view of both budgets and the waiter queue.
func (a *TransferAdmission) Stats() TransferAdmissionStats {
	a.mu.Lock()
	defer a.mu.Unlock()
	return TransferAdmissionStats{
		ExecutionUsed: a.execUsed,
		ExecutionCap:  a.execCap,
		TransferUsed:  a.transferUsed,
		TransferCap:   a.transferCap,
		Waiters:       len(a.waiters),
		Admitted:      a.admitted,
		Deferred:      a.deferred,
	}
}
