package agent

import (
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

type lookupCleanupSession struct {
	calls      int
	events     *[]string
	closePanic any
}

func (s *lookupCleanupSession) Close() {
	s.calls++
	*s.events = append(*s.events, "session")
	if s.closePanic != nil {
		panic(s.closePanic)
	}
}

type lookupCleanupPlanner struct{ v41EngramLookAhead bool }

// This model executes the source-guarded ownership statements with a session
// double. It does not execute native restore, refeed, metrics or GPU operations.
func lookupCleanupModel(p *lookupCleanupPlanner, candidate *lookupCleanupSession, owned bool, stage string, primary any, callerErr *error) (s *lookupCleanupSession, closeSession bool) {
	transferred := false
	defer finishInKernelCleanup(callerErr, func() {
		if !transferred && s != nil && (closeSession || p.v41EngramLookAhead) {
			retiring := s
			s, closeSession = nil, false
			retiring.Close()
		}
	})
	s, closeSession = candidate, owned
	if stage == "error" {
		return
	}
	if stage == "refeed" || stage == "metrics" {
		panic(primary)
	}
	if stage == "fallback" {
		retiring, owned := s, closeSession || p.v41EngramLookAhead
		s, closeSession = nil, false
		if owned {
			retiring.Close()
		}
	}
	transferred = true
	return
}

// fak-test:runtime fast est=100ms lane=default
func TestOwnedPrefixLookupCleanupFaults(t *testing.T) {
	primary, secondary := errors.New("primary"), errors.New("cleanup")
	cases := []struct {
		name, stage                                 string
		owned, lookAhead, errorReturn, cleanupPanic bool
		wantPanic                                   any
		closes                                      int
	}{
		{"refeed", "refeed", true, false, false, false, primary, 1},
		{"metrics", "metrics", true, false, false, false, primary, 1},
		{"primary-survives-close", "refeed", true, false, false, true, primary, 1},
		{"error-survives-close", "error", true, false, true, true, nil, 1},
		{"error-cleanup", "error", true, false, true, false, nil, 1},
		{"success-transfers", "success", true, false, false, false, nil, 1},
		{"fallback-retires-once", "fallback", true, false, false, false, nil, 1},
		{"fallback-close-panic-once", "fallback", true, false, false, true, secondary, 1},
		{"nonowned-cpu", "metrics", false, false, false, false, primary, 0},
		{"lookahead-cpu", "metrics", false, true, false, false, primary, 1},
		{"lookahead-success", "success", false, true, false, false, nil, 1},
		{"lookahead-fallback", "fallback", false, true, false, false, nil, 1},
		{"lookahead-fallback-close-panic-once", "fallback", false, true, false, true, secondary, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var events []string
			candidate := &lookupCleanupSession{events: &events}
			if tc.cleanupPanic {
				candidate.closePanic = secondary
			}
			var err error
			if tc.errorReturn {
				err = primary
			}
			var gotPanic any
			func() {
				defer func() { gotPanic = recover() }()
				// The page lease belongs to the caller and outlives helper rollback.
				defer func() { events = append(events, "page") }()
				s, owned := lookupCleanupModel(&lookupCleanupPlanner{tc.lookAhead}, candidate, tc.owned, tc.stage, primary, &err)
				if err != nil {
					return
				}
				if s != nil && (owned || tc.lookAhead) {
					defer finishInKernelCleanup(&err, s.Close)
				}
			}()
			if gotPanic != tc.wantPanic {
				t.Fatalf("panic=%v want=%v", gotPanic, tc.wantPanic)
			}
			if candidate.calls != tc.closes {
				t.Fatalf("close calls=%d want=%d", candidate.calls, tc.closes)
			}
			want := []string{"page"}
			if tc.closes == 1 {
				want = []string{"session", "page"}
			}
			if !reflect.DeepEqual(events, want) {
				t.Fatalf("events=%v want=%v", events, want)
			}
			if tc.errorReturn && err != primary {
				t.Fatalf("primary error changed: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestOwnedPrefixLookupCleanupSourceGuard(t *testing.T) {
	src, err := os.ReadFile("inkernel_prefix_lookup.go")
	if err != nil {
		t.Fatal(err)
	}
	norm := func(s string) string { return strings.Join(strings.Fields(s), " ") }
	s := norm(string(src))
	ownership := `transferred := false
 defer finishInKernelCleanup(callerErr, func() {
 if !transferred && s != nil && (closeSession || p.v41EngramLookAhead) {
 retiring := s
 s, closeSession = nil, false
 retiring.Close()
 }
 })`
	if !strings.Contains(s, norm(ownership)) {
		t.Fatal("rollback ownership differs from fault model")
	}
	detach := `retiring, owned := s, closeSession || p.v41EngramLookAhead
 s, matched, closeSession = nil, 0, false
 if owned { retiring.Close() }`
	if !strings.Contains(s, norm(detach)) {
		t.Fatal("fallback must detach before close")
	}
	transfer := `enginestep.Default.ObservePrefixMatched(matched)
 transferred = true
 return`
	if !strings.Contains(s, norm(transfer)) || strings.Count(s, "transferred = true") != 1 {
		t.Fatal("ownership must transfer only after final lookup observation")
	}
	if strings.Index(s, norm(ownership)) > strings.Index(s, "owner, scoped :=") {
		t.Fatal("rollback must be registered before acquisition")
	}
}
