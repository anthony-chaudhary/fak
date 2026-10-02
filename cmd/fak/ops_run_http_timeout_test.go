package main

import (
	"os"
	"strings"
	"testing"
)

// TestOpsRunInferenceProbeUsesBoundedClient pins the invariant behind the
// MISSING_HTTP_TIMEOUT fix: the `fak ops` inference-route preflight must issue
// its request through a client that carries its own Timeout, not through
// http.DefaultClient.
//
// The probe is bounded today only by the context deadline built one line above
// the call. That is a single point of failure: the bound lives in the caller,
// not in the request, so a future caller that drops the context turns the probe
// into an unbounded hang. Asserting the client itself is bounded keeps the
// guarantee attached to the code that depends on it, which is the same
// source-assertion idiom cmd/fak/serve_native_help_test.go uses for the
// `fak serve` native path.
func TestOpsRunInferenceProbeUsesBoundedClient(t *testing.T) {
	src, err := os.ReadFile("ops_run.go")
	if err != nil {
		t.Fatalf("read ops_run.go: %v", err)
	}
	body := string(src)
	if strings.Contains(body, "http.DefaultClient.Do(req)") {
		t.Fatalf("the inference preflight still issues through http.DefaultClient, " +
			"which carries no Timeout; build a client with Timeout: opsRunInferenceProbeTimeout")
	}
	if !strings.Contains(body, "&http.Client{Timeout: opsRunInferenceProbeTimeout}") {
		t.Fatalf("the inference preflight no longer builds a client bounded by " +
			"opsRunInferenceProbeTimeout; the probe would fall back to an unbounded request")
	}
}
