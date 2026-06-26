package safecommit

import "sync"

// Window is the adaptive concurrency window for shared-trunk writers.
//
// Non-goals, verbatim from #831: not atomicity, not a correctness gate,
// single-host signal. The hard correctness gates remain gitgate's spatial
// disjointness check, safecommit's pathspec assertion, and the git hooks. This
// window only modulates how many writers are admitted in this process at once.
type Window struct {
	mu       sync.Mutex
	limit    int
	inFlight int
}

// DefaultWindow is the process-local adaptive writer window used by Commit.
var DefaultWindow = NewWindow(1)

// NewWindow creates an AIMD writer window. initial values below 1 are clamped
// to 1 so the window can never fully close.
func NewWindow(initial int) *Window {
	if initial < 1 {
		initial = 1
	}
	return &Window{limit: initial}
}

// Limit reports the current maximum number of admitted in-flight writers.
func (w *Window) Limit() int {
	if w == nil {
		return 0
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.limit
}

// InFlight reports how many writers are currently admitted through this window.
func (w *Window) InFlight() int {
	if w == nil {
		return 0
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.inFlight
}

// TryAcquire admits one writer if the current in-flight count is below the
// window. The returned release function must be called exactly once with the
// ship Result so the AIMD law can observe the existing safecommit success/failure
// edge: success grows by one; failure halves and floors at one.
func (w *Window) TryAcquire() (func(Result), bool) {
	if w == nil {
		return func(Result) {}, true
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.limit < 1 {
		w.limit = 1
	}
	if w.inFlight >= w.limit {
		return nil, false
	}
	w.inFlight++
	released := false
	return func(res Result) {
		w.mu.Lock()
		defer w.mu.Unlock()
		if released {
			return
		}
		released = true
		if w.inFlight > 0 {
			w.inFlight--
		}
		w.observeLocked(res)
	}, true
}

// Observe updates the window from a completed ship result without changing the
// in-flight count. It is useful for a dispatcher that already admits writers by
// some other mechanism but wants the same AIMD law.
func (w *Window) Observe(res Result) {
	if w == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.observeLocked(res)
}

func (w *Window) observeLocked(res Result) {
	success, observed := shipOutcome(res)
	if !observed {
		return
	}
	if success {
		w.limit++
		return
	}
	w.limit /= 2
	if w.limit < 1 {
		w.limit = 1
	}
}

func shipOutcome(res Result) (success bool, observed bool) {
	switch res.Reason {
	case ReasonNoPath, ReasonEmptyMessage, ReasonNothingStaged, ReasonWindowFull:
		return false, false
	case "":
		return res.Verified, res.Verified
	default:
		return false, true
	}
}
