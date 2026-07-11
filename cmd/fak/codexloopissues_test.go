package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

// TestBuildCodexLoopGateActionItemsFoldsClassesAndDispatchable is the core adapter
// contract: LOOP diagnoses fold into one dispatchable issue candidate per
// (tool, output-digest) class; sibling sessions on the same class fold into that
// one item (they never open a duplicate); and every emitted item passes the
// issuecontract dispatchability gate (no SkippedRow), so the class becomes a real
// issue rather than a vague one.
func TestBuildCodexLoopGateActionItemsFoldsClassesAndDispatchable(t *testing.T) {
	goalOutcome := codexRepeatedOutcome{
		Tool: "create_goal", OutputDigest: "sha256:aaaa1111bbbb2222", OutputExcerpt: "goal created",
		Count: 4, LongestRun: 4, ArgsDigestCount: 1,
	}
	execOutcome := codexRepeatedOutcome{
		Tool: "exec", OutputDigest: "sha256:cccc3333dddd4444", OutputExcerpt: "invariant failed",
		Count: 3, LongestRun: 3, ArgsDigestCount: 1,
	}
	report := codexLoopRecentReport{
		Diagnoses: []codexLoopDiagnosis{
			{SessionID: "sess-a", Verdict: "LOOP", RepeatedOutcomes: []codexRepeatedOutcome{goalOutcome}},
			// Same create_goal class from a second session: must fold, not duplicate.
			{SessionID: "sess-b", Verdict: "LOOP", RepeatedOutcomes: []codexRepeatedOutcome{goalOutcome}},
			// A distinct exec class: its own item.
			{SessionID: "sess-c", Verdict: "LOOP", RepeatedOutcomes: []codexRepeatedOutcome{execOutcome}},
			// A healthy session: never LOOP, files nothing.
			{SessionID: "sess-ok", Verdict: "OK", RepeatedOutcomes: []codexRepeatedOutcome{
				{Tool: "update_plan", OutputDigest: "sha256:eeee", Count: 6, ArgsDigestCount: 6},
			}},
		},
	}

	items := buildCodexLoopGateActionItems(report)
	if len(items) != 2 {
		t.Fatalf("want 2 folded loop classes, got %d: %+v", len(items), items)
	}

	byKey := map[string]dogfoodissues.ActionItem{}
	for _, it := range items {
		byKey[it.Key] = it
	}
	goalKey := "codex-loop-gate/create_goal/aaaa1111bbbb"
	execKey := "codex-loop-gate/exec/cccc3333dddd"
	goal, ok := byKey[goalKey]
	if !ok {
		t.Fatalf("missing folded create_goal class %q; keys=%v", goalKey, codexLoopKeysOf(items))
	}
	if _, ok := byKey[execKey]; !ok {
		t.Fatalf("missing exec class %q; keys=%v", execKey, codexLoopKeysOf(items))
	}
	// Two sessions hit the create_goal class: DebtCount folds and both are witnessed.
	if goal.DebtCount != 2 {
		t.Fatalf("create_goal DebtCount = %d, want 2 (folded sessions)", goal.DebtCount)
	}
	if len(goal.BoundaryNotes) != 2 {
		t.Fatalf("create_goal boundary notes = %d, want 2 session witnesses: %v", len(goal.BoundaryNotes), goal.BoundaryNotes)
	}
	if !strings.Contains(goal.BoundaryNotes[0], "sess-a") || !strings.Contains(goal.BoundaryNotes[1], "sess-b") {
		t.Fatalf("boundary notes did not witness both sessions: %v", goal.BoundaryNotes)
	}

	// Every emitted item must be dispatchable — the whole point of filling the scope
	// fields. A SkippedRow here means the adapter produced a vague issue.
	plan, skipped := dogfoodissues.BuildPlanWithOptions(items, nil, dogfoodissues.BuildOptions{})
	if len(skipped) != 0 {
		t.Fatalf("loop-gate items were not dispatchable: %+v", skipped)
	}
	if len(plan) != 2 {
		t.Fatalf("plan rows = %d, want 2", len(plan))
	}
	for _, row := range plan {
		if row.Action != "create" {
			t.Fatalf("row %s action = %q, want create against empty existing", row.Key, row.Action)
		}
	}
}

// TestBuildCodexLoopGateActionItemsSkipsNonLoopAndProgressOnly proves the adapter
// never files for a healthy or unactionable session: an OK verdict is skipped, and
// a (degenerate) LOOP whose only repeated traffic is forward-progress produces no
// loop-driving outcome and therefore no issue.
func TestBuildCodexLoopGateActionItemsSkipsNonLoopAndProgressOnly(t *testing.T) {
	report := codexLoopRecentReport{
		Diagnoses: []codexLoopDiagnosis{
			{SessionID: "ok", Verdict: "OK", RepeatedOutcomes: []codexRepeatedOutcome{
				{Tool: "shell_command", OutputDigest: "sha256:1", Count: 1},
			}},
			{SessionID: "progress-only", Verdict: "LOOP", RepeatedOutcomes: []codexRepeatedOutcome{
				// Forward progress (distinct args): codexTopLoopDrivingOutcome skips it,
				// so there is no concrete class to file.
				{Tool: "update_plan", OutputDigest: "sha256:2", Count: 5, ArgsDigestCount: 5},
			}},
		},
	}
	if items := buildCodexLoopGateActionItems(report); len(items) != 0 {
		t.Fatalf("want 0 issues for non-loop/progress-only sessions, got %d: %+v", len(items), items)
	}
}

// TestBuildCodexLoopGateActionItemsUpdatesExistingByMarker proves idempotency: when
// an issue already carries the class's marker key, the plan updates it in place
// instead of opening a duplicate.
func TestBuildCodexLoopGateActionItemsUpdatesExistingByMarker(t *testing.T) {
	outcome := codexRepeatedOutcome{Tool: "create_goal", OutputDigest: "sha256:feed0000cafe1111", Count: 4, LongestRun: 4}
	key := codexLoopGateActionKey(outcome)
	report := codexLoopRecentReport{Diagnoses: []codexLoopDiagnosis{
		{SessionID: "s1", Verdict: "LOOP", RepeatedOutcomes: []codexRepeatedOutcome{outcome}},
	}}
	items := buildCodexLoopGateActionItems(report)
	existing := []dogfoodissues.Issue{{
		Number: 4291, State: "OPEN", Title: "old title",
		Body: fmt.Sprintf("tracked\n<!-- fak-dogfood-action-key: %s -->\n", key),
	}}
	plan, skipped := dogfoodissues.BuildPlanWithOptions(items, existing, dogfoodissues.BuildOptions{DedupeChecked: true})
	if len(skipped) != 0 || len(plan) != 1 {
		t.Fatalf("plan=%d skipped=%d, want 1/0", len(plan), len(skipped))
	}
	row := plan[0]
	if row.Action != "update" || row.Number == nil || *row.Number != 4291 {
		t.Fatalf("existing marker did not route to update #4291: action=%q number=%v", row.Action, row.Number)
	}
}

// TestSessionsCodexLoopSyncIssuesDryRunPlan runs the wired command end-to-end over a
// real recent scan: a genuine no-progress loop becomes a dry-run create plan, the
// output is the issue Result (not the loop report), and no raw tool arguments leak.
func TestSessionsCodexLoopSyncIssuesDryRunPlan(t *testing.T) {
	home := codexHomeWithCreateGoalLoop(t)

	var stdout, stderr bytes.Buffer
	code := runSessions(&stdout, &stderr, []string{
		"codex-loop", "--recent", "--codex-home", home, "--limit", "5",
		"--sync-issues", "--json",
	})
	if code != 0 {
		t.Fatalf("sync-issues dry-run exited %d stderr=%s", code, stderr.String())
	}
	var result dogfoodissues.Result
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("issue result did not decode: %v\n%s", err, stdout.String())
	}
	if result.Mode != "dry-run" {
		t.Fatalf("mode = %q, want dry-run", result.Mode)
	}
	if len(result.Planned) != 1 {
		t.Fatalf("planned rows = %d, want 1: %+v (skipped=%+v)", len(result.Planned), result.Planned, result.Skipped)
	}
	row := result.Planned[0]
	if row.Action != "create" || !strings.HasPrefix(row.Key, "codex-loop-gate/create_goal/") {
		t.Fatalf("unexpected planned row: action=%q key=%q", row.Action, row.Key)
	}
	if len(result.Skipped) != 0 {
		t.Fatalf("loop-gate item was skipped as non-dispatchable: %+v", result.Skipped)
	}
	if len(result.Synced) != 0 {
		t.Fatalf("dry-run must not sync: %+v", result.Synced)
	}
	// The class's raw create_goal arguments must never reach a public issue body.
	if strings.Contains(stdout.String(), "secret-goal-arg") {
		t.Fatalf("issue plan leaked raw tool arguments:\n%s", stdout.String())
	}
}

// TestSessionsCodexLoopSyncIssuesLiveCreatesVerified drives the --live branch with an
// injected gh runner (no real subprocess): the plan is created and marker-verified,
// and the command reports success.
func TestSessionsCodexLoopSyncIssuesLiveCreatesVerified(t *testing.T) {
	outcome := codexRepeatedOutcome{Tool: "create_goal", OutputDigest: "sha256:abcd1234ef567890", Count: 4, LongestRun: 4}
	report := codexLoopRecentReport{Diagnoses: []codexLoopDiagnosis{
		{SessionID: "s1", Verdict: "LOOP", RepeatedOutcomes: []codexRepeatedOutcome{outcome}},
	}}
	key := codexLoopGateActionKey(outcome)

	created := 0
	runner := func(args []string) (string, string, bool) {
		switch {
		case len(args) >= 2 && args[0] == "issue" && args[1] == "create":
			created++
			return "https://github.com/o/r/issues/77\n", "", true
		case len(args) >= 2 && args[0] == "issue" && args[1] == "view":
			// marker + milestone read-back verification
			body := fmt.Sprintf(
				`{"number":77,"url":"https://github.com/o/r/issues/77","body":"<!-- fak-dogfood-action-key: %s -->","milestone":{"title":%q}}`,
				key, dogfoodissues.DefaultMilestone)
			return body, "", true
		}
		return "", "unexpected gh call: " + strings.Join(args, " "), false
	}

	var stdout, stderr bytes.Buffer
	code := runCodexLoopSyncIssues(&stdout, &stderr, report, true, codexLoopIssueOptions{
		Live:   true,
		Repo:   "o/r",
		Limit:  300,
		Runner: runner,
		// Seed existing issues via fixture so the live branch does not shell out to
		// the real gh for the existing-issue scan; only Sync uses the injected runner.
		ExistingJSON: writeExistingIssuesFixture(t, "[]"),
	})
	if code != 0 {
		t.Fatalf("live sync exited %d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if created != 1 {
		t.Fatalf("gh issue create calls = %d, want 1", created)
	}
	var result dogfoodissues.Result
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("live result did not decode: %v\n%s", err, stdout.String())
	}
	if len(result.Synced) != 1 || !result.Synced[0].OK || !result.Synced[0].Verified {
		t.Fatalf("live sync row not verified-ok: %+v", result.Synced)
	}
}

// TestSessionsCodexLoopSyncIssuesRequiresRecent guards the flag combination: the
// issue bridge is only meaningful over a recent scan.
func TestSessionsCodexLoopSyncIssuesRequiresRecent(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runSessions(&stdout, &stderr, []string{"codex-loop", "--sync-issues", "--path", "x.jsonl"})
	if code != 2 || !strings.Contains(stderr.String(), "require --recent") {
		t.Fatalf("sync-issues without --recent exited %d stderr=%s, want 2 + require --recent", code, stderr.String())
	}
}

func codexLoopKeysOf(items []dogfoodissues.ActionItem) []string {
	out := make([]string, 0, len(items))
	for _, it := range items {
		out = append(out, it.Key)
	}
	return out
}

func writeExistingIssuesFixture(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "existing.json")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// codexHomeWithCreateGoalLoop builds a Codex home holding one session that livelocks
// on create_goal: the same goal is created repeatedly with the same output, which the
// diagnosis flags LOOP with a concrete create_goal repeated outcome.
func codexHomeWithCreateGoalLoop(t *testing.T) string {
	t.Helper()
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionsDir := filepath.Join(home, "sessions", "2026", "07", "11")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	loopPath := filepath.Join(sessionsDir, "rollout-2026-07-11T10-00-00-goalloop.jsonl")
	lines := []string{
		`{"timestamp":"2026-07-11T10:00:00.000Z","type":"session_meta","payload":{"session_id":"goal-loop","originator":"codex-tui","cli_version":"0.142.5","model_provider":"openai","git":{"commit_hash":"abc1234","branch":"main"}}}`,
	}
	for i := 1; i <= 4; i++ {
		call := fmt.Sprintf(`{"timestamp":"2026-07-11T10:0%d:00.000Z","type":"response_item","payload":{"type":"function_call","name":"create_goal","arguments":"{\"goal\":\"secret-goal-arg\"}","call_id":"g%d"}}`, i, i)
		out := fmt.Sprintf(`{"timestamp":"2026-07-11T10:0%d:01.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"g%d","output":"goal already exists"}}`, i, i)
		tok := fmt.Sprintf(`{"timestamp":"2026-07-11T10:0%d:02.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":%d,"output_tokens":40,"total_tokens":%d},"last_token_usage":{"input_tokens":1000,"output_tokens":40,"total_tokens":1040}}}}`, i, i*1000, i*1000+40)
		lines = append(lines, call, out, tok)
	}
	writeCodexLoopFixture(t, loopPath, lines)
	return home
}
