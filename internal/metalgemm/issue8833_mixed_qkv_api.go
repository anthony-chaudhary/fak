package metalgemm

import (
	"errors"
	"fmt"
	"sync/atomic"
)

// MixedQKVCallID is a process-unique opaque identity for one call-owned mixed-QKV attempt.
// It identifies observations without exposing native command-buffer addresses.
type MixedQKVCallID uint64

var mixedQKVCallSequence atomic.Uint64

func nextMixedQKVCallID() MixedQKVCallID { return MixedQKVCallID(mixedQKVCallSequence.Add(1)) }

// MixedQKVStage records whether ownership was declined before encoding or failed after submission.
type MixedQKVStage uint8

const (
	MixedQKVDeclined MixedQKVStage = iota
	MixedQKVSubmitted
)

// MixedQKVError is the typed fallback boundary. Callers may retry the established path only when
// Stage is MixedQKVDeclined; submitted work must never be encoded a second time.
type MixedQKVError struct {
	CallID MixedQKVCallID
	Stage  MixedQKVStage
	Detail string
}

func (e *MixedQKVError) Error() string {
	return fmt.Sprintf("mixed QKV call %d: %s", e.CallID, e.Detail)
}

func IsMixedQKVDecline(err error) bool {
	var e *MixedQKVError
	return errors.As(err, &e) && e.Stage == MixedQKVDeclined
}

// MixedQKVSelector names the two benchmark arms. Control intentionally preserves two owner calls;
// Candidate is reserved for the one-command-buffer native owner.
type MixedQKVSelector uint8

const (
	MixedQKVControl MixedQKVSelector = iota + 1
	MixedQKVCandidate
)

// MixedQKVResult returns all host copies together after owner completion.
type MixedQKVResult struct {
	CallID       MixedQKVCallID
	Q, K, V      []float32
	Submitted    bool
	Observation  ExecutionSnapshot
}

// ScopedExecutionEvent is a #8844 event paired with its call identity. Events are delivered only
// to the observer supplied to that call; there is no package-global observer or delta accounting.
type ScopedExecutionEvent struct {
	CallID MixedQKVCallID
	Event  ExecutionEvent
}

// ExecutionObserver receives lifecycle facts synchronously before ExecuteMixedQKV returns.
type ExecutionObserver interface {
	ObserveExecution(ScopedExecutionEvent)
}

// ExecutionObserverFunc adapts a function to ExecutionObserver.
type ExecutionObserverFunc func(ScopedExecutionEvent)

func (f ExecutionObserverFunc) ObserveExecution(e ScopedExecutionEvent) { f(e) }

// MixedQKVWeight is the portable view of an already-resident native weight handle.
type MixedQKVWeight interface { ID() int }

// MixedQKVInput contains the exact decode inputs and already-resident mixed-family weights.
type MixedQKVInput struct {
	Q, K       MixedQKVWeight
	V          MixedQKVWeight
	XQ         []int8
	XD, XF     []float32
	Hidden     int
	Observer   ExecutionObserver
	// Failure injection is intentionally available only to same-package Darwin tests.
	injectSetup, injectPost bool
}

// ExecuteMixedQKV is implemented by the Darwin owner and by a portable typed-decline stub.
func ExecuteMixedQKV(selector MixedQKVSelector, in MixedQKVInput) (MixedQKVResult, error) {
	return executeMixedQKV(selector, in)
}

// MixedQKVUnavailable creates the portable typed decline used when the native owner is not linked
// or rejects geometry before encoding. It is public so model dispatch can preserve the no-retry
// boundary without string matching.
func MixedQKVUnavailable(detail string) error {
	return &MixedQKVError{CallID: nextMixedQKVCallID(), Stage: MixedQKVDeclined, Detail: detail}
}
