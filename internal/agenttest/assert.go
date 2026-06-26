package agenttest

import (
	"fmt"
	"strings"
)

// T is the minimal testing surface the Assert* helpers need. *testing.T satisfies it, so
// the assertions work from any Go test without this package importing "testing".
type T interface {
	Helper()
	Errorf(format string, args ...any)
}

// ---------------------------------------------------------------------------
// matchers — pure predicates over a Run; nil error means the pattern holds. They are the
// reusable core the Assert* wrappers report through, and are directly unit-testable
// without a *testing.T.
// ---------------------------------------------------------------------------

// MatchToolSequence holds iff the run called exactly want, in order. The strictest
// pattern assertion.
func (r Run) MatchToolSequence(want ...string) error {
	got := r.ToolNames()
	if !equalStrings(got, want) {
		return fmt.Errorf("tool sequence mismatch:\n  want: %v\n  got:  %v", want, got)
	}
	return nil
}

// MatchToolSubsequence holds iff want appears in order somewhere within the calls (gaps
// allowed) — useful when a workflow may interleave incidental calls.
func (r Run) MatchToolSubsequence(want ...string) error {
	got := r.ToolNames()
	i := 0
	for _, name := range got {
		if i < len(want) && name == want[i] {
			i++
		}
	}
	if i != len(want) {
		return fmt.Errorf("tool subsequence not found:\n  want (in order): %v\n  got:             %v", want, got)
	}
	return nil
}

// MatchToolOrder holds iff the first call to before precedes the first call to after, and
// both were called.
func (r Run) MatchToolOrder(before, after string) error {
	bi, ai := r.firstIndex(before), r.firstIndex(after)
	switch {
	case bi < 0:
		return fmt.Errorf("tool order: %q was never called", before)
	case ai < 0:
		return fmt.Errorf("tool order: %q was never called", after)
	case bi >= ai:
		return fmt.Errorf("tool order: %q (call #%d) did not precede %q (call #%d)", before, bi, after, ai)
	}
	return nil
}

// CountTool reports how many times tool was called.
func (r Run) CountTool(tool string) int {
	n := 0
	for _, e := range r.Tools {
		if e.Tool == tool {
			n++
		}
	}
	return n
}

// CalledWith reports whether tool was called at least once with raw arguments containing
// substr.
func (r Run) CalledWith(tool, substr string) bool {
	for _, e := range r.Tools {
		if e.Tool == tool && strings.Contains(e.Args, substr) {
			return true
		}
	}
	return false
}

// firstIndex returns the 0-based position of the first call to tool, or -1.
func (r Run) firstIndex(tool string) int {
	for i, e := range r.Tools {
		if e.Tool == tool {
			return i
		}
	}
	return -1
}

// ---------------------------------------------------------------------------
// Assert* wrappers — report a failed match through t.Errorf, the ergonomic surface a test
// calls. Each is a Helper so the failure points at the caller's line.
// ---------------------------------------------------------------------------

// AssertToolSequence fails t unless the run called exactly want, in order.
func AssertToolSequence(t T, r Run, want ...string) {
	t.Helper()
	if err := r.MatchToolSequence(want...); err != nil {
		t.Errorf("%v", err)
	}
}

// AssertToolSubsequence fails t unless want appears in order within the calls.
func AssertToolSubsequence(t T, r Run, want ...string) {
	t.Helper()
	if err := r.MatchToolSubsequence(want...); err != nil {
		t.Errorf("%v", err)
	}
}

// AssertToolOrder fails t unless before's first call precedes after's first call.
func AssertToolOrder(t T, r Run, before, after string) {
	t.Helper()
	if err := r.MatchToolOrder(before, after); err != nil {
		t.Errorf("%v", err)
	}
}

// AssertToolCalled fails t unless tool was called at least once.
func AssertToolCalled(t T, r Run, tool string) {
	t.Helper()
	if r.CountTool(tool) == 0 {
		t.Errorf("expected tool %q to be called; calls were %v", tool, r.ToolNames())
	}
}

// AssertToolNotCalled fails t if tool was called.
func AssertToolNotCalled(t T, r Run, tool string) {
	t.Helper()
	if n := r.CountTool(tool); n != 0 {
		t.Errorf("expected tool %q not to be called; it was called %d time(s)", tool, n)
	}
}

// AssertToolCount fails t unless tool was called exactly want times.
func AssertToolCount(t T, r Run, tool string, want int) {
	t.Helper()
	if got := r.CountTool(tool); got != want {
		t.Errorf("tool %q call count = %d, want %d", tool, got, want)
	}
}

// AssertCalledWith fails t unless tool was called with raw arguments containing substr.
func AssertCalledWith(t T, r Run, tool, substr string) {
	t.Helper()
	if !r.CalledWith(tool, substr) {
		t.Errorf("expected a call to %q with args containing %q; calls were %v", tool, substr, r.ToolNames())
	}
}

// AssertFinalAnswer fails t unless the run's final answer contains substr.
func AssertFinalAnswer(t T, r Run, substr string) {
	t.Helper()
	if !strings.Contains(r.FinalAnswer, substr) {
		t.Errorf("final answer %q does not contain %q", r.FinalAnswer, substr)
	}
}

// AssertAllMocked fails t if the run called any tool that had no registered mock.
func AssertAllMocked(t T, r Run) {
	t.Helper()
	if un := r.UnmockedTools(); len(un) > 0 {
		t.Errorf("run called unmocked tools: %v", un)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
