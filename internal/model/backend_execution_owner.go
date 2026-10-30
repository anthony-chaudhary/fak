package model

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// backendExecutionScope is one already-acquired recording-domain lease. Only
// private owned bodies consult it. Public entrypoints acquire the owner before
// backend work; an already-detached PrefixSnapshot.Close is a terminal no-op.
// A concurrent caller can never borrow another call's scope.
// A Session remains exclusive to its executing caller, as before this seam.
type backendExecutionScope struct {
	owner    *sync.Mutex
	sessions []*Session
	after    []func()
	// drains run while the domain is still held, before any binding is invalidated; they settle
	// device work issued under this lease that may not outlive it (in-flight ring transfers).
	drains []func()
	live   bool
}

func backendExecutionOwner(be compute.Backend) *sync.Mutex {
	provider, ok := be.(compute.ForwardExecutionOwnerBackend)
	if !ok {
		return nil
	}
	owner := provider.ForwardExecutionOwner()
	if owner == nil {
		panic("model: backend advertises a nil execution owner")
	}
	return owner
}

func beginBackendExecution(be compute.Backend) *backendExecutionScope {
	owner := backendExecutionOwner(be)
	if owner == nil {
		return nil
	}
	owner.Lock()
	return &backendExecutionScope{owner: owner, live: true}
}

func (s *Session) beginBackendExecution() *backendExecutionScope {
	if s == nil {
		return nil
	}
	scope := beginBackendExecution(s.Backend)
	// If binding fails, do not strand the newly acquired domain.
	bound := false
	defer func() {
		if !bound && scope != nil {
			scope.finish()
		}
	}()
	scope.bind(s)
	bound = true
	return scope
}

func (scope *backendExecutionScope) bind(s *Session) {
	if scope == nil || s == nil {
		return
	}
	if scope.live && s.backendExecutionScope == scope {
		return // Explicit private nesting borrows this same operation's lease.
	}
	if !scope.live || s.backendExecutionScope != nil {
		panic("model: invalid or overlapping owned session scope")
	}
	if owner := backendExecutionOwner(s.Backend); owner != nil && owner != scope.owner {
		panic("model: session and execution scope domains differ")
	}
	s.backendExecutionScope = scope
	scope.sessions = append(scope.sessions, s)
}

// beforeRelease queues f to run inside this lease, ahead of release. A drain panic is collected
// like a retirement panic and never replaces the primary failure; every drain is attempted.
func (scope *backendExecutionScope) beforeRelease(f func()) {
	if scope == nil || !scope.live {
		panic("model: drain queued on expired execution scope")
	}
	scope.drains = append(scope.drains, f)
}

func (scope *backendExecutionScope) afterRelease(f func()) {
	if scope == nil {
		f()
		return
	}
	if !scope.live {
		panic("model: retirement queued on expired execution scope")
	}
	scope.after = append(scope.after, f)
}

// finish is deferred by the acquiring public boundary. It invalidates every
// binding before release, then drains model-global retirement without holding
// this domain or any session/lifecycle lock. All retirements are attempted;
// a cleanup panic never replaces the primary forward panic.
func (scope *backendExecutionScope) finish() {
	if scope == nil {
		return
	}
	scope.release(recover(), false)
}

// finishCompleted leaves the caller's unwind untouched. An explicit completion
// bit distinguishes a normal return from both legacy panic(nil) and Goexit,
// neither of which may be converted into successful session retirement.
func (scope *backendExecutionScope) finishCompleted(complete *bool) {
	if scope == nil {
		return
	}
	completed := *complete
	cleanup := scope.releaseCleanup(nil)
	if completed && cleanup.present {
		panic(cleanup.value)
	}
}

// finishError additionally preserves a primary returned error over a cleanup
// panic. The caller's error stays authoritative, just as in the forward unwind.
func (scope *backendExecutionScope) finishError(err *error) {
	if scope == nil {
		return
	}
	scope.release(recover(), err != nil && *err != nil)
}

func (scope *backendExecutionScope) release(primary any, failed bool) {
	cleanup := scope.releaseCleanup(primary)
	if cleanup.legacyValue != nil && !failed {
		panic(cleanup.legacyValue)
	}
}

// Legacy boundaries select the first nonnil payload. Completion-aware Close
// selects the first panic, including legacy panic(nil). Keep both selections:
// a nil panic followed by a nonnil panic has different established precedence.
type backendCleanupPanic struct {
	value       any
	present     bool
	legacyValue any
}

func (scope *backendExecutionScope) releaseCleanup(primary any) (cleanup backendCleanupPanic) {
	if !scope.live {
		if primary != nil {
			panic(primary)
		}
		panic("model: execution scope released twice")
	}
	drains := scope.drains
	scope.drains = nil
	for _, f := range drains {
		scope.runCleanup(&cleanup, f)
	}
	scope.live = false
	for _, s := range scope.sessions {
		s.backendExecutionScope = nil
	}
	scope.sessions = nil
	after := scope.after
	scope.after = nil
	scope.owner.Unlock()
	for _, f := range after {
		scope.runCleanup(&cleanup, f)
	}
	if primary != nil {
		panic(primary)
	}
	// Goexit does not return through the callback boundary or publish an outcome.
	return cleanup
}

// runCleanup runs one drain or retirement, recording the first panic (including panic(nil)) and the
// first nonnil payload into cleanup with the established precedence.
func (*backendExecutionScope) runCleanup(cleanup *backendCleanupPanic, f func()) {
	completed := false
	defer func() {
		r := recover()
		if !completed && !cleanup.present {
			cleanup.value, cleanup.present = r, true
		}
		if r != nil && cleanup.legacyValue == nil {
			cleanup.legacyValue = r
		}
	}()
	f()
	completed = true
}

// closeBackendOwned is used only by selected-operation error paths inside an
// already-owned forward. It never acquires or skips a public entrypoint lock.
// Legacy backends without the capability retain their existing close behavior.
func (s *Session) closeBackendOwned() {
	if s == nil {
		return
	}
	scope := s.backendExecutionScope
	if owner := backendExecutionOwner(s.Backend); owner != nil &&
		(scope == nil || !scope.live || scope.owner != owner) {
		panic("model: selected teardown has no matching execution scope")
	}
	if s.v41RoutedInputs != nil {
		o := s.v41RoutedInputs
		if cleanup := o.release(); cleanup.present && s.halFailure == nil {
			cause, known := v41RoutedCohortPanicCause(cleanup.value)
			if !known {
				cause = v41RoutedCohortUnknownPanic(cleanup.value)
			}
			s.halFailure = &BackendForwardOperationError{Backend: o.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-input", Layer: o.layer, Stage: "cleanup", Cause: cause}
			if known {
				cleanup.value = s.halFailure
			}
			defer func() { panic(cleanup.value) }()
		}
	}
	if cleanup := s.drainV41RoutedOutputs(); cleanup.present {
		// A cleanup-only failure escapes after teardown. An existing selected
		// primary stays authoritative; the drain has already attempted all rows.
		defer func() { panic(cleanup.value) }()
	}
	if s.halFailure != nil {
		// This is a selected operation's cleanup; its caller still returns or
		// panics with that primary failure after this private teardown returns.
		defer func() { s.halClosed = true; _ = recover() }()
	}
	s.closeOwned(scope)
}
