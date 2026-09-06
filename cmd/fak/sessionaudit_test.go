package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func TestSessionAuditDiscoverAuditAndDeep(t *testing.T) {
	root := t.TempDir()
	sessionPath := writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "session-a.jsonl"), []map[string]any{
		sessionAuditAssistant("msg-1", 100, "Read"),
		{
			"type":      "user",
			"timestamp": "2026-06-20T00:01:00.000Z",
			"message": map[string]any{
				"content": "Run the audit",
			},
		},
	})

	var stdout, stderr bytes.Buffer
	rc := runSessionAudit(&stdout, &stderr, []string{"discover", "--root", root, "--all", "--max", "1"})
	if rc != 0 {
		t.Fatalf("discover rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "1 sessions") || !strings.Contains(stdout.String(), "C--work-fak/session-a.jsonl") {
		t.Fatalf("unexpected discover output:\n%s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	jsonOut := filepath.Join(t.TempDir(), "audit.json")
	mdOut := filepath.Join(t.TempDir(), "audit.md")
	rc = runSessionAudit(&stdout, &stderr, []string{"audit", "--root", root, "--all", "--json", jsonOut, "--md", mdOut})
	if rc != 0 {
		t.Fatalf("audit rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Session-Transcript Audit") {
		t.Fatalf("audit did not render markdown:\n%s", stdout.String())
	}
	if _, err := os.Stat(mdOut); err != nil {
		t.Fatalf("markdown output not written: %v", err)
	}
	var payload struct {
		Aggregate struct {
			NSessions int `json:"n_sessions"`
		} `json:"aggregate"`
	}
	raw, err := os.ReadFile(jsonOut)
	if err != nil {
		t.Fatalf("json output not written: %v", err)
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("bad json output: %v\n%s", err, raw)
	}
	if payload.Aggregate.NSessions != 1 {
		t.Fatalf("json sessions = %d, want 1", payload.Aggregate.NSessions)
	}

	stdout.Reset()
	stderr.Reset()
	rc = runSessionAudit(&stdout, &stderr, []string{"deep", sessionPath})
	if rc != 0 {
		t.Fatalf("deep rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "# Trajectory: session-a") || !strings.Contains(stdout.String(), "Run the audit") {
		t.Fatalf("unexpected deep output:\n%s", stdout.String())
	}
}

func TestSessionAuditWarnsWhenSubagentsExcluded(t *testing.T) {
	root := t.TempDir()
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "session-a.jsonl"), []map[string]any{
		sessionAuditAssistant("top", 100, ""),
	})
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "session-a", "subagents", "worker.jsonl"), []map[string]any{
		sessionAuditAssistant("sub", 2_000, ""),
	})
	var stdout, stderr bytes.Buffer
	rc := runSessionAudit(&stdout, &stderr, []string{"audit", "--root", root, "--all"})
	if rc != 0 {
		t.Fatalf("audit rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "NOTE: +1 subagent transcripts uncounted") {
		t.Fatalf("subagent warning missing:\n%s", stdout.String())
	}
}

func TestSessionAuditWarnsWhenMaxClipsBeforeNamespaceAudit(t *testing.T) {
	root := t.TempDir()
	older := writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "fable.jsonl"), []map[string]any{
		sessionAuditAssistant("fable", 100, ""),
	})
	newer := writeSessionAuditJSONL(t, filepath.Join(root, "C--work-job", "synthetic.jsonl"), []map[string]any{
		sessionAuditAssistant("synthetic", 10, ""),
	})
	now := time.Now()
	if err := os.Chtimes(older, now.Add(-time.Hour), now.Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(newer, now, now); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	rc := runSessionAudit(&stdout, &stderr, []string{"discover", "--root", root, "--all", "--max", "1"})
	if rc != 0 {
		t.Fatalf("discover rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "showing first 1 of 2") ||
		!strings.Contains(stdout.String(), "use --ns-prefix") ||
		strings.Contains(stdout.String(), "C--work-fak/fable.jsonl") {
		t.Fatalf("discover cap warning did not explain hidden namespace:\n%s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	rc = runSessionAudit(&stdout, &stderr, []string{"audit", "--root", root, "--all", "--max", "1"})
	if rc != 0 {
		t.Fatalf("audit rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stderr.String(), "warning: --max clipped discovery to first 1 of 2") {
		t.Fatalf("audit stderr cap warning missing:\n%s", stderr.String())
	}
	if !strings.Contains(stdout.String(), "NOTE: `--max 1` clipped this audit to the newest 1 of 2 discovered transcripts") {
		t.Fatalf("audit markdown cap warning missing:\n%s", stdout.String())
	}
	if strings.Contains(stdout.String(), "scoped audit") {
		t.Fatalf("unscoped audit should not use scoped cap wording:\n%s", stdout.String())
	}
}

func TestSessionAuditHereScopesToCurrentWorkspaceNamespace(t *testing.T) {
	root := t.TempDir()
	workspace := filepath.Join(t.TempDir(), "work", "fak")
	if err := os.MkdirAll(workspace, 0o777); err != nil {
		t.Fatal(err)
	}
	hereNS := sessionaudit.ProjectNamespace(workspace)
	herePath := writeSessionAuditJSONL(t, filepath.Join(root, hereNS, "fable.jsonl"), []map[string]any{
		sessionAuditAssistant("fable", 100, ""),
	})
	olderHerePath := writeSessionAuditJSONL(t, filepath.Join(root, hereNS, "opus.jsonl"), []map[string]any{
		sessionAuditAssistant("opus", 200, ""),
	})
	otherPath := writeSessionAuditJSONL(t, filepath.Join(root, "C--work-job", "synthetic.jsonl"), []map[string]any{
		sessionAuditAssistant("synthetic", 10, ""),
	})
	now := time.Now()
	if err := os.Chtimes(herePath, now.Add(-time.Hour), now.Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(olderHerePath, now.Add(-2*time.Hour), now.Add(-2*time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(otherPath, now, now); err != nil {
		t.Fatal(err)
	}
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(workspace); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(oldwd)
	})

	var stdout, stderr bytes.Buffer
	rc := runSessionAudit(&stdout, &stderr, []string{"discover", "--root", root, "--here", "--max", "1"})
	if rc != 0 {
		t.Fatalf("discover --here rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), hereNS+"/fable.jsonl") ||
		strings.Contains(stdout.String(), "C--work-job/synthetic.jsonl") ||
		!strings.Contains(stdout.String(), "showing first 1 of 2") {
		t.Fatalf("--here did not scope before --max:\n%s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	rc = runSessionAudit(&stdout, &stderr, []string{"audit", "--root", root, "--here", "--max", "1"})
	if rc != 0 {
		t.Fatalf("audit --here rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "namespace filter: "+hereNS) ||
		!strings.Contains(stdout.String(), "| fable |") ||
		!strings.Contains(stdout.String(), "clipped this scoped audit to the newest 1 of 2 discovered transcripts") ||
		strings.Contains(stdout.String(), "C--work-job") {
		t.Fatalf("audit --here did not report the current workspace scope:\n%s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "warning: --max clipped scoped discovery to first 1 of 2 transcripts") ||
		strings.Contains(stderr.String(), "use --ns-prefix or --here") {
		t.Fatalf("audit --here cap warning should be scoped:\n%s", stderr.String())
	}
}

func TestSessionAuditSummaryJSON(t *testing.T) {
	root := t.TempDir()
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "heavy.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("opus", 200, 0, 900_000, 50_000, "claude-opus-4-8", ""),
	})
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "fable.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("fable", 300, 0, 20_000, 1_000, "claude-fable-5", ""),
	})

	var stdout, stderr bytes.Buffer
	rc := runSessionAudit(&stdout, &stderr, []string{"summary", "--root", root, "--all", "--json"})
	if rc != 0 {
		t.Fatalf("summary --json rc=%d stderr=%s", rc, stderr.String())
	}
	var rep sessionaudit.CompactReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("bad summary json: %v\n%s", err, stdout.String())
	}
	if rep.Schema != "fak.session_audit.summary.v1" || rep.Totals.TotalContextTokens != 971_000 {
		t.Fatalf("summary report = %+v", rep)
	}
	tiers := map[string]sessionaudit.CompactTier{}
	for _, tier := range rep.Tiers {
		tiers[tier.Tier] = tier
	}
	if tiers["fable"].OutputTokens != 300 || tiers["opus"].OutputTokens != 200 {
		t.Fatalf("summary tiers = %+v", rep.Tiers)
	}
	if len(rep.TopLongContext) == 0 || rep.TopLongContext[0].Session != "heavy" ||
		rep.TopLongContext[0].TotalContextTokens != 950_000 {
		t.Fatalf("summary long-context rows = %+v", rep.TopLongContext)
	}
}

func TestSessionAuditActionsJSON(t *testing.T) {
	root := t.TempDir()
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "heavy.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("opus", 200, 0, 900_000, 50_000, "claude-opus-4-8", ""),
	})
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "fable.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("fable", 300, 0, 20_000, 1_000, "claude-fable-5", ""),
	})

	var stdout, stderr bytes.Buffer
	rc := runSessionAudit(&stdout, &stderr, []string{"actions", "--root", root, "--all", "--json"})
	if rc != 0 {
		t.Fatalf("actions --json rc=%d stderr=%s", rc, stderr.String())
	}
	var plan sessionaudit.CompactActionPlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("bad actions json: %v\n%s", err, stdout.String())
	}
	if plan.Schema != sessionaudit.CompactActionPlanSchema || plan.Counts.Total != 2 || plan.Counts.High != 2 {
		t.Fatalf("actions plan = %+v", plan)
	}
	byID := map[string]sessionaudit.CompactAction{}
	for _, action := range plan.Actions {
		byID[action.ID] = action
	}
	if byID["keep_fable_default"].Target != "model_route:fable_default" {
		t.Fatalf("fable action = %+v", byID["keep_fable_default"])
	}
	if byID["checkpoint_reset_top_long_context"].Session != "heavy" ||
		byID["checkpoint_reset_top_long_context"].Target != "session:C--work-fak/heavy" {
		t.Fatalf("long-context action = %+v", byID["checkpoint_reset_top_long_context"])
	}

	stdout.Reset()
	stderr.Reset()
	rc = runSessionAudit(&stdout, &stderr, []string{"actions", "--root", root, "--all", "--json", "--fail-on", "high"})
	if rc != 1 {
		t.Fatalf("actions --fail-on high rc=%d stderr=%s stdout=%s", rc, stderr.String(), stdout.String())
	}
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("bad gated actions json: %v\n%s", err, stdout.String())
	}
	if plan.Gate.Verdict != "refuse" || plan.Gate.Refused != 2 || plan.Gate.Threshold != "high" {
		t.Fatalf("gate = %+v, want refused high gate", plan.Gate)
	}
}

func TestSessionAuditAliasDefaultsToHereSummary(t *testing.T) {
	root := t.TempDir()
	workspace := filepath.Join(t.TempDir(), "work", "fak")
	if err := os.MkdirAll(workspace, 0o777); err != nil {
		t.Fatal(err)
	}
	hereNS := sessionaudit.ProjectNamespace(workspace)
	writeSessionAuditJSONL(t, filepath.Join(root, hereNS, "heavy.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("opus", 200, 0, 900_000, 50_000, "claude-opus-4-8", ""),
	})
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-other", "other.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("other", 300, 0, 20_000, 1_000, "claude-fable-5", ""),
	})
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(workspace); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(oldwd)
	})

	var stdout, stderr bytes.Buffer
	rc := runSession(&stdout, &stderr, []string{"audit", "--root", root, "--days", "7", "--json"})
	if rc != 0 {
		t.Fatalf("session audit alias rc=%d stderr=%s", rc, stderr.String())
	}
	var rep sessionaudit.CompactReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("bad alias summary json: %v\n%s", err, stdout.String())
	}
	if rep.Scope.NamespaceFilter != hereNS || rep.Scope.Discovered != 1 || rep.Scope.Audited != 1 {
		t.Fatalf("alias summary scope = %+v, want here-scoped single transcript", rep.Scope)
	}
	if rep.Totals.OutputTokens != 200 || rep.Totals.TotalContextTokens != 950_000 {
		t.Fatalf("alias summary totals = %+v, want only current-workspace transcript", rep.Totals)
	}

	stdout.Reset()
	stderr.Reset()
	rc = runSession(&stdout, &stderr, []string{"audit", "actions", "--root", root, "--days", "7", "--json"})
	if rc != 0 {
		t.Fatalf("session audit actions alias rc=%d stderr=%s", rc, stderr.String())
	}
	var plan sessionaudit.CompactActionPlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("bad alias actions json: %v\n%s", err, stdout.String())
	}
	if plan.Scope.NamespaceFilter != hereNS || plan.Counts.Total == 0 {
		t.Fatalf("alias actions plan = %+v, want here-scoped actions", plan)
	}
}

func TestSessionAuditBudget(t *testing.T) {
	root := t.TempDir()
	sessionPath := writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "budget.jsonl"), []map[string]any{
		sessionAuditAssistant("t1", 100, "Read"), // read tool,  context 135
		sessionAuditAssistant("t2", 50, "Edit"),  // edit tool,  context  85
		sessionAuditAssistant("t3", 30, "Bash"),  // other tool, context  65
	})

	// JSON mode: spend, fractions, and the coarse reads/edits/other breakdown
	// from real usage records.
	var stdout, stderr bytes.Buffer
	rc := runSessionAudit(&stdout, &stderr, []string{"budget", "--json", "--target-tokens", "200", "--target-turns", "5", sessionPath})
	if rc != 0 {
		t.Fatalf("budget --json rc=%d stderr=%s", rc, stderr.String())
	}
	var b sessionaudit.TaskBudget
	if err := json.Unmarshal(stdout.Bytes(), &b); err != nil {
		t.Fatalf("bad budget json: %v\n%s", err, stdout.String())
	}
	if b.TotalTokens != 285 || b.OutputTokens != 180 || b.Turns != 3 {
		t.Fatalf("budget spend = total=%d output=%d turns=%d, want 285/180/3", b.TotalTokens, b.OutputTokens, b.Turns)
	}
	if b.Breakdown.Reads != 1 || b.Breakdown.Edits != 1 || b.Breakdown.OtherTools != 1 || b.Breakdown.ToolCalls != 3 {
		t.Fatalf("budget breakdown = %+v, want reads/edits/other=1 tool_calls=3", b.Breakdown)
	}
	if b.TokenFrac == nil || !b.OverTokens {
		t.Fatalf("285 tok over a 200 target should flag OverTokens with a fraction: %+v", b)
	}
	if b.TurnFrac == nil || b.OverTurns {
		t.Fatalf("3 turns under a 5 target should not flag OverTurns: %+v", b)
	}
	if b.Breakdown.Model == "" {
		t.Fatalf("budget should name the dominating model tier: %+v", b.Breakdown)
	}

	// Default mode: one inline line the working agent can print mid-task.
	stdout.Reset()
	stderr.Reset()
	rc = runSessionAudit(&stdout, &stderr, []string{"budget", "--target-tokens", "200", sessionPath})
	if rc != 0 {
		t.Fatalf("budget text rc=%d stderr=%s", rc, stderr.String())
	}
	line := stdout.String()
	if !strings.Contains(line, "task-budget") ||
		!strings.Contains(line, "reads 1, edits 1, other 1") ||
		!strings.Contains(line, "OVER") {
		t.Fatalf("unexpected budget line:\n%s", line)
	}
}

func sessionAuditAssistant(id string, out int64, tool string) map[string]any {
	return sessionAuditAssistantDetailed(id, out, 10, 20, 5, "claude-sonnet-4-5", tool)
}

func sessionAuditAssistantDetailed(id string, out, input, cacheRead, cacheCreate int64, model, tool string) map[string]any {
	content := []any{}
	if tool != "" {
		content = append(content, map[string]any{"type": "tool_use", "name": tool, "input": map[string]any{}})
	}
	return map[string]any{
		"type":      "assistant",
		"timestamp": "2026-06-20T00:00:00.000Z",
		"message": map[string]any{
			"id":    id,
			"model": model,
			"usage": map[string]any{
				"input_tokens":                input,
				"output_tokens":               out,
				"cache_read_input_tokens":     cacheRead,
				"cache_creation_input_tokens": cacheCreate,
			},
			"content": content,
		},
	}
}

func writeSessionAuditJSONL(t *testing.T, path string, records []map[string]any) string {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	enc := json.NewEncoder(f)
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// THE LIVE DEFECT THIS PINS. `--include-subagents` is documented as including subagent /
// workflow transcripts, and it used to discover and analyze them and then aggregate the
// top-level sessions anyway. On this harness that loses the entire delegated track: every
// delegated turn is written into a spawned transcript, so the report answered UNKNOWN over
// a corpus it had already read (measured: 1,524 of 1,524 marked turns lived in spawned
// transcripts, 0 of 5,755 top-level turns were marked).
func TestSessionAuditIncludeSubagentsCountsDelegatedVolume(t *testing.T) {
	root := t.TempDir()
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "session-a.jsonl"), []map[string]any{
		sessionAuditAssistant("top", 900, "Agent"), // a spawn call: work WAS delegated
	})
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "session-a", "subagents", "worker.jsonl"), []map[string]any{
		sessionAuditAssistant("sub", 100, ""), // the delegated work itself, and unmarked
	})

	// Default scope: the spawned transcript is genuinely out of scope, so the share is
	// UNKNOWN — not a 0% this corpus never earned.
	var stdout, stderr bytes.Buffer
	if rc := runSessionAudit(&stdout, &stderr, []string{"audit", "--root", root, "--all"}); rc != 0 {
		t.Fatalf("audit rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Delegation share = UNKNOWN") {
		t.Fatalf("a scope that excludes the delegated transcripts must not name a share:\n%s", stdout.String())
	}

	// With the flag, the same corpus becomes answerable.
	stdout.Reset()
	stderr.Reset()
	if rc := runSessionAudit(&stdout, &stderr, []string{"audit", "--root", root, "--all", "--include-subagents"}); rc != 0 {
		t.Fatalf("audit rc=%d stderr=%s", rc, stderr.String())
	}
	got := stdout.String()
	if strings.Contains(got, "Delegation share = UNKNOWN") {
		t.Fatalf("--include-subagents analyzed the delegated transcript, so it must count it:\n%s", got)
	}
	if !strings.Contains(got, "Delegation share of tracked output = 10.0%") {
		t.Fatalf("want a 10%% delegated share (100 of 1,000 output tok):\n%s", got)
	}
	// And the breakout must not invite anyone to add the same tokens on twice.
	if !strings.Contains(got, "already counted in the totals above") {
		t.Fatalf("the subagent breakout must say it is already counted:\n%s", got)
	}
}

func TestSessionAuditGeminiIntegration(t *testing.T) {
	root := t.TempDir()
	geminiChatDir := filepath.Join(root, "proj-gemini", "chats")
	if err := os.MkdirAll(geminiChatDir, 0o755); err != nil {
		t.Fatal(err)
	}
	geminiPath := filepath.Join(geminiChatDir, "session-gemini.json")
	geminiData := map[string]any{
		"sessionId": "session-gemini",
		"startTime": "2026-07-10T10:00:00Z",
		"model":     "gemini-2.5-pro",
		"messages": []map[string]any{
			{
				"type":    "user",
				"content": "Read the main source file",
			},
			{
				"type":    "gemini",
				"model":   "gemini-2.5-pro",
				"content": "I will read the main file.",
				"usageMetadata": map[string]any{
					"promptTokenCount":     int64(500),
					"candidatesTokenCount": int64(100),
					"totalTokenCount":      int64(600),
				},
				"toolCalls": []map[string]any{
					{
						"id":     "tc-1",
						"name":   "read_file",
						"args":   map[string]any{"path": "main.go"},
						"status": "success",
						"result": []any{"package main"},
					},
				},
			},
		},
	}
	b, err := json.MarshalIndent(geminiData, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(geminiPath, b, 0o600); err != nil {
		t.Fatal(err)
	}

	// 1. discover
	var stdout, stderr bytes.Buffer
	rc := runSessionAudit(&stdout, &stderr, []string{"discover", "--root", root, "--all"})
	if rc != 0 {
		t.Fatalf("discover rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "1 sessions") || !strings.Contains(stdout.String(), "session-gemini.json") {
		t.Fatalf("unexpected discover output:\n%s", stdout.String())
	}

	// 2. audit
	stdout.Reset()
	stderr.Reset()
	jsonOut := filepath.Join(t.TempDir(), "audit.json")
	rc = runSessionAudit(&stdout, &stderr, []string{"audit", "--root", root, "--all", "--json", jsonOut})
	if rc != 0 {
		t.Fatalf("audit rc=%d stderr=%s", rc, stderr.String())
	}
	raw, err := os.ReadFile(jsonOut)
	if err != nil {
		t.Fatalf("failed to read audit json: %v", err)
	}
	var auditPayload struct {
		Aggregate struct {
			NSessions int `json:"n_sessions"`
			Totals    struct {
				Output int64 `json:"output"`
			} `json:"totals"`
		} `json:"aggregate"`
	}
	if err := json.Unmarshal(raw, &auditPayload); err != nil {
		t.Fatalf("unmarshal audit json: %v", err)
	}
	if auditPayload.Aggregate.NSessions != 1 {
		t.Fatalf("audit sessions = %d, want 1", auditPayload.Aggregate.NSessions)
	}
	if auditPayload.Aggregate.Totals.Output != 100 {
		t.Fatalf("audit output tokens = %d, want 100", auditPayload.Aggregate.Totals.Output)
	}

	// 3. summary
	stdout.Reset()
	stderr.Reset()
	rc = runSessionAudit(&stdout, &stderr, []string{"summary", "--root", root, "--all", "--json"})
	if rc != 0 {
		t.Fatalf("summary rc=%d stderr=%s", rc, stderr.String())
	}
	var summaryRep struct {
		Totals struct {
			OutputTokens int64 `json:"output_tokens"`
		} `json:"totals"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &summaryRep); err != nil {
		t.Fatalf("unmarshal summary json: %v", err)
	}
	if summaryRep.Totals.OutputTokens != 100 {
		t.Fatalf("summary output tokens = %d, want 100", summaryRep.Totals.OutputTokens)
	}

	// 4. deep
	stdout.Reset()
	stderr.Reset()
	rc = runSessionAudit(&stdout, &stderr, []string{"deep", geminiPath})
	if rc != 0 {
		t.Fatalf("deep rc=%d stderr=%s", rc, stderr.String())
	}
	deepOut := stdout.String()
	if !strings.Contains(deepOut, "# Trajectory: session-gemini") || !strings.Contains(deepOut, "Read the main source file") {
		t.Fatalf("unexpected deep output:\n%s", deepOut)
	}

	// 5. Dual Claude + Gemini discovery in same root
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "session-claude.jsonl"), []map[string]any{
		sessionAuditAssistant("claude-msg", 50, ""),
	})

	stdout.Reset()
	stderr.Reset()
	rc = runSessionAudit(&stdout, &stderr, []string{"discover", "--root", root, "--all"})
	if rc != 0 {
		t.Fatalf("dual discover rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "2 sessions") {
		t.Fatalf("expected 2 sessions in dual discovery:\n%s", stdout.String())
	}

	// 6. --no-gemini excludes Gemini sessions
	stdout.Reset()
	stderr.Reset()
	rc = runSessionAudit(&stdout, &stderr, []string{"discover", "--root", root, "--all", "--no-gemini"})
	if rc != 0 {
		t.Fatalf("no-gemini discover rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "1 sessions") || strings.Contains(stdout.String(), "session-gemini.json") {
		t.Fatalf("--no-gemini should only discover 1 Claude session:\n%s", stdout.String())
	}
}
