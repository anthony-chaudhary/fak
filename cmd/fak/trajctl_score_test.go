package main

// trajctl_score_test.go — issue #2543: the done-condition witness for the W1
// judge scorer's operator door. `fak trajctl score --method judge` must produce
// a W1 row for a docs-shaped objective from a canned gateway response, and the
// per-call token budget cap must be ENFORCED (request ceiling and fail-closed
// over-budget return), not merely documented.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// cannedJudgeGateway stubs an OpenAI-compatible gateway that always answers the
// forced emit_verdict tool call with the given progress/rationale and reports
// the given total token usage. It records each request's max_tokens so tests
// can assert the budget cap traveled into the request.
func cannedJudgeGateway(t *testing.T, progress float64, rationale string, usedTokens int) (*httptest.Server, *[]float64) {
	t.Helper()
	var maxTokensSeen []float64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("request body: %v", err)
		}
		if mt, ok := body["max_tokens"].(float64); ok {
			maxTokensSeen = append(maxTokensSeen, mt)
		}
		w.Header().Set("Content-Type", "application/json")
		args, _ := json.Marshal(trajctl.JudgeVerdict{Progress: progress, Met: false, Rationale: rationale})
		fmt.Fprintf(w, `{
		  "choices": [{"message": {"tool_calls": [
		    {"function": {"name": "emit_verdict", "arguments": %s}}
		  ]}}],
		  "usage": {"total_tokens": %d}
		}`, mustQuote(t, string(args)), usedTokens)
	}))
	t.Cleanup(srv.Close)
	return srv, &maxTokensSeen
}

func mustQuote(t *testing.T, s string) string {
	t.Helper()
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("quote: %v", err)
	}
	return string(b)
}

// declareDocsObjective appends the docs-shaped objective the issue's done
// condition names: progress that is neither commit- nor test-shaped.
func declareDocsObjective(t *testing.T, ledger string) {
	t.Helper()
	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{
		"declare", "--id", "docs-guide", "--ledger", ledger,
		"--statement", "Write the operator guide: every trajctl subcommand documented with one worked example",
	}); code != 0 {
		t.Fatalf("declare = %d, stderr=%q", code, errb.String())
	}
}

// TestTrajctlScoreJudgeEmitsW1Row is the issue's done condition end to end:
// `fak trajctl score --method judge` against a canned gateway produces a W1 row
// for a docs-shaped objective, carrying the verdict blob as evidence.
func TestTrajctlScoreJudgeEmitsW1Row(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	declareDocsObjective(t, ledger)
	srv, maxTokens := cannedJudgeGateway(t, 0.4, "guide skeleton exists; examples missing", 120)

	var out, errb bytes.Buffer
	code := runTrajctl(&out, &errb, []string{
		"score", "--objective", "docs-guide", "--method", "judge",
		"--base-url", srv.URL, "--ledger", ledger, "--json",
	})
	if code != 0 {
		t.Fatalf("score = %d, stderr=%q", code, errb.String())
	}

	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	if len(st.Scores) != 1 {
		t.Fatalf("ledger scores = %d, want exactly 1 (rows=%v)", len(st.Scores), st.Scores)
	}
	row := st.Scores[0]
	if row.ObjectiveID != "docs-guide" || row.Witness != trajctl.W1 {
		t.Errorf("row = %+v, want objective docs-guide at witness W1", row)
	}
	if row.Method != trajctl.JudgeScorerMethod || row.Version != trajctl.JudgeScorerVersion {
		t.Errorf("row method = %s v%s, want %s v%s", row.Method, row.Version, trajctl.JudgeScorerMethod, trajctl.JudgeScorerVersion)
	}
	if row.Value != 0.4 {
		t.Errorf("row value = %v, want 0.4", row.Value)
	}
	if len(row.Evidence) != 1 || row.Evidence[0].Kind != "judge-verdict" ||
		!strings.Contains(row.Evidence[0].Detail, "guide skeleton exists") {
		t.Errorf("evidence = %+v, want one judge-verdict ref carrying the rationale", row.Evidence)
	}
	// The default budget cap must have traveled into the request as max_tokens.
	if len(*maxTokens) != 1 || (*maxTokens)[0] != float64(trajctl.DefaultJudgeMaxCallTokens) {
		t.Errorf("request max_tokens = %v, want one call capped at %d", *maxTokens, trajctl.DefaultJudgeMaxCallTokens)
	}
	if !strings.Contains(out.String(), `"witness": "W1"`) {
		t.Errorf("--json output missing W1 row, got %q", out.String())
	}
}

// TestTrajctlScoreJudgeBudgetCapEnforced asserts the cap is enforced, not
// documented: the flag value rides as the request's max_tokens, and a canned
// response reporting usage over the cap yields exit 1 and NO ledger row.
func TestTrajctlScoreJudgeBudgetCapEnforced(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	declareDocsObjective(t, ledger)
	srv, maxTokens := cannedJudgeGateway(t, 0.9, "runaway generation", 5000)

	var out, errb bytes.Buffer
	code := runTrajctl(&out, &errb, []string{
		"score", "--objective", "docs-guide", "--method", "judge",
		"--base-url", srv.URL, "--ledger", ledger, "--max-call-tokens", "64",
	})
	if code != 1 {
		t.Fatalf("score with over-budget return = %d, want 1 (stderr=%q)", code, errb.String())
	}
	if len(*maxTokens) != 1 || (*maxTokens)[0] != 64 {
		t.Errorf("request max_tokens = %v, want one call capped at 64", *maxTokens)
	}
	if st := trajctl.Fold(trajctl.ReadLedgerFile(ledger)); len(st.Scores) != 0 {
		t.Errorf("over-budget call appended %d score rows, want 0 (fail-closed)", len(st.Scores))
	}
}

// TestTrajctlScoreUsageErrors pins the argument contract: missing required
// flags and unknown names refuse with the documented exit codes.
func TestTrajctlScoreUsageErrors(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	declareDocsObjective(t, ledger)
	cases := []struct {
		name string
		argv []string
		want int
	}{
		{"missing objective", []string{"score", "--method", "judge", "--base-url", "http://x", "--ledger", ledger}, 2},
		{"missing method", []string{"score", "--objective", "docs-guide", "--ledger", ledger}, 2},
		{"judge without base-url", []string{"score", "--objective", "docs-guide", "--method", "judge", "--ledger", ledger}, 2},
		{"unknown method", []string{"score", "--objective", "docs-guide", "--method", "vibes", "--base-url", "http://x", "--ledger", ledger}, 2},
		{"unknown objective", []string{"score", "--objective", "ghost", "--method", "judge", "--base-url", "http://x", "--ledger", ledger}, 1},
	}
	for _, c := range cases {
		var out, errb bytes.Buffer
		if got := runTrajctl(&out, &errb, c.argv); got != c.want {
			t.Errorf("%s: runTrajctl(%v) = %d, want %d (stderr=%q)", c.name, c.argv, got, c.want, errb.String())
		}
	}
	if st := trajctl.Fold(trajctl.ReadLedgerFile(ledger)); len(st.Scores) != 0 {
		t.Errorf("usage errors appended %d score rows, want 0", len(st.Scores))
	}
}
