package gateway

// subagent_witness_test.go — the witness for #2438: extend the loop-body witness to the
// SUBAGENT fold boundary. Each test maps to one acceptance bullet on the issue:
//
//   - Unwitnessed fold (taint): a child spawned via a subagent-shaped tool claims it
//     shipped/fixed but carries no commit SHA — its ResultAdmission is demoted to
//     RESIDUAL/LOOP_DONE_UNWITNESSED (the parent still sees the content, visibly unverified).
//   - Git-witnessed fold (clean): the SAME claim carrying the `git commit` success marker
//     folds clean — a plain ALLOW, no demotion row.
//   - Guards: an ordinary (non-spawn) tool result is untouched; a subagent result with no
//     completion claim is untouched.

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// newSubagentWitnessServer builds a minimal proxy server whose result-side floor is live
// (ctxmmu result admitter) so admitInboundResults produces a real per-result verdict.
// Proxy-fill stays OFF — this test cares about the admission verdict, not cache warming.
func newSubagentWitnessServer(t *testing.T) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, allowAllAdj{})
	abi.RegisterResultAdmitter(10, ctxmmu.New())

	srv, err := New(Config{EngineID: "test", Model: "test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// findAdmission returns the ResultAdmission for the given tool_call id, failing if absent.
func findAdmission(t *testing.T, adms []ResultAdmission, id string) ResultAdmission {
	t.Helper()
	for _, a := range adms {
		if a.ToolCallID == id {
			return a
		}
	}
	t.Fatalf("no ResultAdmission for tool_call %q in %+v", id, adms)
	return ResultAdmission{}
}

// TestSubagentDone_UnwitnessedFoldsTainted is the headline: a child spawned via a "Task"
// tool returns a terminal result CLAIMING it fixed and committed the change — but the
// prose carries no commit SHA / file hash. The parent must still see the content, but the
// admission must record it visibly unverified (RESIDUAL/LOOP_DONE_UNWITNESSED).
func TestSubagentDone_UnwitnessedFoldsTainted(t *testing.T) {
	srv := newSubagentWitnessServer(t)
	ctx := WithPrincipal(context.Background(), "tenantA")

	msgs := inboundTurn("child-1", "Task", `{"prompt":"fix the bug"}`,
		"I fixed the bug and committed the change. All tests pass.")
	adms, err := srv.admitInboundResults(ctx, msgs, nil, "trace-sub-1")
	if err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}

	ra := findAdmission(t, adms, "child-1")
	if ra.Verdict.Kind != "RESIDUAL" || ra.Verdict.Reason != ReasonLoopBodyUnwitnessed {
		t.Fatalf("unwitnessed subagent done-claim: verdict = %+v, want RESIDUAL/%s", ra.Verdict, ReasonLoopBodyUnwitnessed)
	}
	if ra.Verdict.By != subagentFoldBy {
		t.Fatalf("demotion By = %q, want %q", ra.Verdict.By, subagentFoldBy)
	}
	if ra.Verdict.Detail["claim"] != "ship" {
		t.Fatalf("demotion must record the unbacked claim kind, Detail = %v, want claim=ship", ra.Verdict.Detail)
	}
	// The tainted content is still forwarded — the parent sees it, only visibly unverified.
	if got := msgs[2].Content; !strings.Contains(got, "I fixed the bug") {
		t.Fatalf("tainted content must still reach the parent, got %q", got)
	}
}

// TestSubagentDone_GitWitnessedFoldsClean proves the safe path: the SAME claim carrying
// the `git commit` success marker (the artifact witness) folds clean — a plain ALLOW,
// no demotion row.
func TestSubagentDone_GitWitnessedFoldsClean(t *testing.T) {
	srv := newSubagentWitnessServer(t)
	ctx := WithPrincipal(context.Background(), "tenantA")

	msgs := inboundTurn("child-2", "Task", `{"prompt":"fix the bug"}`,
		"Fixed and committed: [main 809c5339] fix(bug): guard the nil deref")
	adms, err := srv.admitInboundResults(ctx, msgs, nil, "trace-sub-2")
	if err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}

	ra := findAdmission(t, adms, "child-2")
	if ra.Verdict.Kind != "ALLOW" {
		t.Fatalf("git-witnessed subagent done-claim must fold clean: verdict = %+v, want ALLOW", ra.Verdict)
	}
}

// TestSubagentDone_NonSpawnToolUntouched proves the gate fires only at the subagent fold
// boundary: an ordinary tool result (not a spawn) that happens to say "created" is
// admitted normally — the loop-body witness governs the parent's own prose, not this.
func TestSubagentDone_NonSpawnToolUntouched(t *testing.T) {
	srv := newSubagentWitnessServer(t)
	ctx := WithPrincipal(context.Background(), "tenantA")

	msgs := inboundTurn("c3", "get_config", `{"k":"v"}`, "created the default config entry")
	adms, err := srv.admitInboundResults(ctx, msgs, nil, "trace-sub-3")
	if err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	if ra := findAdmission(t, adms, "c3"); ra.Verdict.Kind == "RESIDUAL" {
		t.Fatalf("a non-spawn tool result must not be subagent-demoted, got %+v", ra.Verdict)
	}
}

// TestSubagentDone_NoClaimUntouched proves the gate fires only on a completion claim: a
// subagent result that returns data (no ship/create/fix self-report) folds normally.
func TestSubagentDone_NoClaimUntouched(t *testing.T) {
	srv := newSubagentWitnessServer(t)
	ctx := WithPrincipal(context.Background(), "tenantA")

	msgs := inboundTurn("c4", "Task", `{"prompt":"summarize"}`, "The three files use tabs, not spaces.")
	adms, err := srv.admitInboundResults(ctx, msgs, nil, "trace-sub-4")
	if err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	if ra := findAdmission(t, adms, "c4"); ra.Verdict.Kind == "RESIDUAL" {
		t.Fatalf("a subagent result with no done-claim must not be demoted, got %+v", ra.Verdict)
	}
}

// TestSubagentWitness_ArtifactDetection unit-tests the pure witness detector across the
// witness forms (commit marker, full SHA, sha256 file hash) and the bare-claim forms that
// carry none.
func TestSubagentWitness_ArtifactDetection(t *testing.T) {
	witnessed := []string{
		"Committed [main 809c5339] fix: guard nil",
		"file hash sha256:deadbeef",
		"the commit is 0123456789abcdef0123456789abcdef01234567",
	}
	for _, w := range witnessed {
		if !resultCarriesArtifactWitness(w) {
			t.Fatalf("resultCarriesArtifactWitness(%q) = false, want true", w)
		}
	}
	unwitnessed := []string{
		"I shipped it",
		"done, all good",
		"created the file",
		"",
	}
	for _, u := range unwitnessed {
		if resultCarriesArtifactWitness(u) {
			t.Fatalf("resultCarriesArtifactWitness(%q) = true, want false", u)
		}
	}
}

// TestSubagentWitness_ClaimKind unit-tests the deterministic completion-claim classifier.
func TestSubagentWitness_ClaimKind(t *testing.T) {
	cases := map[string]string{
		"I shipped the change":        "ship",
		"the fix was committed":       "ship",
		"I created a new file":        "create",
		"resolved the flaky test":     "fix",
		"all done here":               "done",
		"here is a summary of the go": "", // no completion verb
	}
	for content, want := range cases {
		if got := claimedDoneKind(content); got != want {
			t.Fatalf("claimedDoneKind(%q) = %q, want %q", content, got, want)
		}
	}
}
