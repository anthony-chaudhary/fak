package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// TestFrontiersweEvalRunVerifierFlagParses pins the --run-verifier wiring at the
// verb boundary (#1719). The staged --reward short-circuits RunEval before any
// verifier stand-up, so the test is deterministic on any box (Docker or not) and
// never spawns a container; the run/refuse gate branches themselves are covered
// offline in internal/frontierswe/eval_verify_test.go.
func TestFrontiersweEvalRunVerifierFlagParses(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"eval", "--json", "--task", "git-to-zig", "--run-verifier",
		"--reward", "../../internal/frontierswe/testdata/frontierswe/reward/git-to-zig_full.json",
	})
	if code != 0 {
		t.Fatalf("eval exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}
	var r frontierswe.EvalResult
	if err := json.Unmarshal(stdout.Bytes(), &r); err != nil {
		t.Fatalf("stdout is not eval JSON: %v\n%s", err, stdout.String())
	}
	if !r.Available || r.Source != "existing-reward" {
		t.Fatalf("staged reward not graded: available=%t source=%q reason=%q", r.Available, r.Source, r.Reason)
	}
}
