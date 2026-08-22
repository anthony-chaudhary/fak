package devcmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuefanout"
)

func writeFanoutExistingFixture(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "existing.json")
	fixture := fmt.Sprintf(`[{"number": 42, "body": %q}]`, body)
	if err := os.WriteFile(path, []byte(fixture), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func TestIssueFanoutLiveFilesUnseenAndRerunFilesZero(t *testing.T) {
	plan, err := issuefanout.Build(issuefanout.Input{
		Title:             "fanout live test",
		Leaf:              "fanoutlivetest",
		SpineRef:          "deadbeef",
		ParentIssue:       36,
		ParentBaseline:    100,
		TargetEnvelope:    "- concurrent users: 10 users\n- sustained duration: 60 minutes",
		WitnessedEnvelope: "- concurrent users: 10 users\n- sustained duration: 60 minutes",
		Areas:             []string{"qa"},
	})
	if err != nil {
		t.Fatalf("Build fixture plan: %v", err)
	}
	if !strings.Contains(plan.Candidates[0].Key, "fanout-fanoutlivetest-spine-") {
		t.Fatalf("fixture key is not spine-qualified: %q", plan.Candidates[0].Key)
	}
	// The fixture tracker already carries the qa-edge-sweep marker key, so the
	// first live run files the other two qa candidates and skips it.
	fixture := writeFanoutExistingFixture(t, "carries "+plan.Candidates[0].Key+" already")
	var calls [][]string
	gh := func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return fmt.Sprintf("https://github.com/o/r/issues/%d", 900+len(calls)), "", true
	}
	argv := []string{
		"--title", "fanout live test", "--leaf", "fanoutlivetest", "--spine", "deadbeef",
		"--areas", "qa", "--parent-issue", "36", "--parent-baseline-points", "100", "--target-envelope", "- concurrent users: 10 users\n- sustained duration: 60 minutes", "--witnessed-envelope", "- concurrent users: 10 users\n- sustained duration: 60 minutes", "--live", "--existing-json", fixture,
	}
	var out, errOut bytes.Buffer
	if code := runIssueFanoutWith(&out, &errOut, argv, gh); code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "filed 2, skipped 1, failed 0") {
		t.Fatalf("first run output missing filed/skipped fold:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "marker key for spine deadbeef already in issue #42") {
		t.Fatalf("first run output missing matched spine:\n%s", out.String())
	}
	if len(calls) != 2 {
		t.Fatalf("gh create calls = %d, want 2", len(calls))
	}

	// Rerun against a tracker carrying every key: files zero, spams nothing.
	keys := make([]string, 0, len(plan.Candidates))
	for _, candidate := range plan.Candidates {
		keys = append(keys, candidate.Key)
	}
	all := writeFanoutExistingFixture(t, strings.Join(keys, " "))
	calls = nil
	out.Reset()
	argv[len(argv)-1] = all
	if code := runIssueFanoutWith(&out, &errOut, argv, gh); code != 0 {
		t.Fatalf("rerun exit = %d, want 0\nstderr: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "filed 0, skipped 3, failed 0") || len(calls) != 0 {
		t.Fatalf("rerun must file zero (gh calls = %d):\n%s", len(calls), out.String())
	}
}

func TestIssueFanoutLiveGhFailureExitsOne(t *testing.T) {
	fixture := writeFanoutExistingFixture(t, "no keys here")
	gh := func(args []string) (string, string, bool) { return "", "gh: boom", false }
	var out, errOut bytes.Buffer
	code := runIssueFanoutWith(&out, &errOut, []string{
		"--title", "t", "--leaf", "fanoutlivetest", "--spine", "s",
		"--areas", "qa", "--parent-issue", "36", "--parent-baseline-points", "100", "--target-envelope", "- concurrent users: 10 users\n- sustained duration: 60 minutes", "--witnessed-envelope", "- concurrent users: 10 users\n- sustained duration: 60 minutes", "--live", "--existing-json", fixture,
	}, gh)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 on gh failure\n%s", code, out.String())
	}
}

func TestIssueFanoutLiveRefusesUnboundedDedupe(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runIssueFanoutWith(&out, &errOut, []string{
		"--title", "t", "--leaf", "l", "--spine", "s", "--live", "--dedupe-cap", "0",
	}, nil)
	if code != 2 || !strings.Contains(errOut.String(), "--dedupe-cap") {
		t.Fatalf("exit = %d stderr = %q, want 2 + dedupe-cap refusal", code, errOut.String())
	}
}

func TestIssueFanoutOfflineDefaultUnchangedByLiveFlags(t *testing.T) {
	// The offline default path must not consult gh or mention live filing.
	gh := func(args []string) (string, string, bool) {
		t.Fatalf("offline run must not invoke gh, got %v", args)
		return "", "", false
	}
	var out, errOut bytes.Buffer
	if code := runIssueFanoutWith(&out, &errOut, []string{
		"--title", "t", "--leaf", "fanoutlivetest", "--spine", "s", "--areas", "qa",
	}, gh); code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr: %s", code, errOut.String())
	}
	if !strings.HasPrefix(out.String(), "fanout: 3 contract-ready follow-ons") {
		t.Fatalf("offline output changed:\n%s", out.String())
	}
}

func TestIssueFanoutJSONCarriesCanonicalProblemFramesToCohortInput(t *testing.T) {
	var out, errb bytes.Buffer
	code := runIssueFanout(&out, &errb, []string{
		"--title", "frame spine", "--leaf", "framespine", "--spine", "abc123", "--max", "3", "--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var plan issuefanout.Plan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if len(plan.Candidates) != 3 {
		t.Fatalf("candidates=%d", len(plan.Candidates))
	}
	for _, candidate := range plan.Candidates {
		if !candidate.ProblemFrame.Enforced || !candidate.ProblemFrame.Ready || len(candidate.ProblemFrame.Checks) != 4 {
			t.Fatalf("%s frame = %+v", candidate.Key, candidate.ProblemFrame)
		}
	}
}

func TestIssueFanoutCmdLeafJSONUsesRunnableCommandPackage(t *testing.T) {
	var out, errb bytes.Buffer
	code := runIssueFanout(&out, &errb, []string{
		"--title", "Scoped guard break-glass launcher",
		"--leaf", "cmd",
		"--spine", "fbe86e17acfe49673079c1583d72fa72471bea2d",
		"--paths", "cmd/fak/guard_disable.go,cmd/fak/guard_disable_test.go,docs/integrations/openai-codex.md",
		"--areas", "qa",
		"--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var plan issuefanout.Plan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	for _, candidate := range plan.Candidates {
		if !strings.Contains(candidate.Witness, "./cmd/fak") || !strings.Contains(candidate.AcceptanceGate, "./cmd/fak") {
			t.Fatalf("%s kept unrunnable cmd witness: witness=%q gate=%q", candidate.Key, candidate.Witness, candidate.AcceptanceGate)
		}
	}
}
