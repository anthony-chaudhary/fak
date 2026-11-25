package witness

import (
	"context"
	"testing"
)

// candRedStateRun runs one selector against canned go test -json output.
func candRedStateRun(t *testing.T, out string, code int, name string) selectedGoTestResult {
	t.Helper()
	run := func(context.Context, string, ...string) (string, int, error) { return out, code, nil }
	return runSelectedGoTests(context.Background(), run, t.TempDir(), []string{"./pkg"}, nil, exactTestSelectors([]string{name}))
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedStateFailWinsOverPass(t *testing.T) {
	const out = `{"Action":"run","Test":"TestA"}
{"Action":"pass","Test":"TestA"}
{"Action":"fail","Test":"TestA"}
`
	got := candRedStateRun(t, out, 1, "TestA")
	if len(got.states) != 1 || got.states[0] != symptomTestFailed {
		t.Fatalf("states=%v, want [Failed] when pass and fail both appear", got.states)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedStateSubtestDoesNotSetParent(t *testing.T) {
	// A failing subtest makes go print the parent as failed, but a passing parent with a
	// failing subtest name must not be read as the parent's own verdict.
	const out = `{"Action":"run","Test":"TestA"}
{"Action":"fail","Test":"TestA/sub"}
{"Action":"pass","Test":"TestA"}
`
	got := candRedStateRun(t, out, 0, "TestA")
	if len(got.states) != 1 || got.states[0] != symptomTestPassed {
		t.Fatalf("states=%v, want [Passed]: the subtest event must not set the parent", got.states)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedStateTeardownDowngradesPass(t *testing.T) {
	// A non-zero exit with every name passing points at TestMain or teardown, so no
	// per-test pass is trusted.
	const out = `{"Action":"run","Test":"TestA"}
{"Action":"pass","Test":"TestA"}
`
	got := candRedStateRun(t, out, 1, "TestA")
	if len(got.states) != 1 || got.states[0] != symptomTestUnknown {
		t.Fatalf("states=%v, want [Unknown] for a pass under a non-zero exit", got.states)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedStateCapturesFirstDiagnostic(t *testing.T) {
	const out = `{"Action":"run","Test":"TestA"}
{"Action":"output","Test":"TestA","Output":"=== RUN   TestA\n"}
{"Action":"output","Test":"TestA","Output":"    a_test.go:3: boom\n"}
{"Action":"output","Test":"TestA","Output":"    a_test.go:4: later\n"}
{"Action":"fail","Test":"TestA"}
`
	got := candRedStateRun(t, out, 1, "TestA")
	if len(got.diag) != 1 || got.diag[0] != "a_test.go:3: boom" {
		t.Fatalf("diag=%v, want the first non-noise line", got.diag)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedStateConfirmDetailFormat(t *testing.T) {
	const base = "selected symptom test failed at parent and passed at candidate"
	if got := symptomConfirmDetail(0, symptomTolerance{}); got != base {
		t.Fatalf("no tolerance = %q, want base %q", got, base)
	}
	if got := symptomConfirmDetail(2, symptomTolerance{}); got != base+" (2 parent-unbuildable changed test file(s) excluded at parent)" {
		t.Fatalf("excluded = %q", got)
	}
	tol := symptomTolerance{preexisting: []string{"TestB", "TestA"}}
	if got := symptomConfirmDetail(0, tol); got != base+" (tolerated at candidate: pre-existing red at parent TestA, TestB)" {
		t.Fatalf("tolerated = %q", got)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedStateTolerantConfirmNotCached(t *testing.T) {
	if !symptomConfirmCacheable(symptomTolerance{}) {
		t.Fatal("a plain confirm must be cacheable")
	}
	if symptomConfirmCacheable(symptomTolerance{preexisting: []string{"TestA"}}) {
		t.Fatal("a tolerant confirm must never be stored in the proof cache")
	}
}
