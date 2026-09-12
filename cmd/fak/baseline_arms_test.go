package main

import (
	"bytes"
	"os"
	"testing"
)

// TestBaselineArms_RoutesThroughDispatcher is the red-then-green reachability
// witness: on the parent commit, "baseline-arms" is an unknown verb (the
// router falls through); after the dispatch case + verb shell land, the real
// internal/armtracking CLI runs and the usage text is served on stdout.
func TestBaselineArms_RoutesThroughDispatcher(t *testing.T) {
	var stdout, stderr bytes.Buffer
	baselineArmsStdout, baselineArmsStderr = &stdout, &stderr
	oldExit := baselineArmsExit
	exitCodes := []int{}
	baselineArmsExit = func(code int) { exitCodes = append(exitCodes, code) }
	t.Cleanup(func() {
		baselineArmsStdout, baselineArmsStderr = os.Stdout, os.Stderr
		baselineArmsExit = oldExit
	})

	if !dispatchCoreVerbA("baseline-arms", []string{"--help"}) {
		t.Fatal("fix commit must include a test that fails on parent and passes on fix: " +
			"baseline-arms is an unknown verb on parent (known-issue witness for #12871)")
	}
	if !bytes.Contains(stdout.Bytes(), []byte("usage: fak baseline-arms")) {
		t.Fatalf("dispatched help output missing usage line:\n%s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	if !dispatchCoreVerbA("baseline-arms", nil) {
		t.Fatal("router must handle baseline-arms with no args")
	}
	if len(exitCodes) != 1 || exitCodes[0] != 2 {
		t.Fatalf("no-arg baseline-arms must exit 2, got %v", exitCodes)
	}
	if !bytes.Contains(stderr.Bytes(), []byte("usage: fak baseline-arms")) {
		t.Fatalf("no-arg usage not routed to stderr:\n%s", stderr.String())
	}
}

func TestBaselineArms_RunCLIEndToEnd(t *testing.T) {
	var out, stderr bytes.Buffer
	if code := runBaselineArms(&out, &stderr, []string{"--help"}); code != 0 {
		t.Fatalf("help code=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(out.Bytes(), []byte("usage: fak baseline-arms")) {
		t.Fatalf("help output missing usage line:\n%s", out.String())
	}

	out.Reset()
	stderr.Reset()
	if code := runBaselineArms(&out, &stderr, nil); code != 2 {
		t.Fatalf("empty argv code=%d, want 2 (stdout=%s)", code, out.String())
	}
	if !bytes.Contains(stderr.Bytes(), []byte("usage: fak baseline-arms")) {
		t.Fatalf("empty argv usage not on stderr:\n%s", stderr.String())
	}
}
