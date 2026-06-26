package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func fixtureTUIIssues() []tuiIssue {
	return []tuiIssue{
		{
			Number:    10,
			Title:     "epic(tui): native issue control pane",
			URL:       "https://example.test/issues/10",
			State:     "OPEN",
			Body:      "Umbrella for the native terminal issue spine.",
			Labels:    []tuiLabel{{Name: "priority/P0"}, {Name: "enhancement"}, {Name: "substrate"}},
			CreatedAt: "2026-05-01T00:00:00Z",
			UpdatedAt: "2026-06-01T00:00:00Z",
			Author:    &tuiUser{Login: "maintainer"},
			Assignees: []tuiUser{{Login: "owner"}},
		},
		{
			Number:    11,
			Title:     "feat(tui): rank issue work queue",
			URL:       "https://example.test/issues/11",
			State:     "OPEN",
			Body:      "Child of #10. Build the ranked issue lane view.",
			Labels:    []tuiLabel{{Name: "priority/P0"}, {Name: "bug"}, {Name: "substrate"}},
			CreatedAt: "2026-02-01T00:00:00Z",
			UpdatedAt: "2026-03-01T00:00:00Z",
			Author:    &tuiUser{Login: "agent"},
		},
		{
			Number:    12,
			Title:     "feat(model): tune loader pane",
			URL:       "https://example.test/issues/12",
			State:     "OPEN",
			Labels:    []tuiLabel{{Name: "priority/P1"}, {Name: "enhancement"}, {Name: "model"}, {Name: "in-progress"}},
			CreatedAt: "2026-06-01T00:00:00Z",
			UpdatedAt: "2026-06-20T00:00:00Z",
			Assignees: []tuiUser{{Login: "builder"}},
		},
		{
			Number:    13,
			Title:     "docs(fak): refresh old operator note",
			URL:       "https://example.test/issues/13",
			State:     "OPEN",
			CreatedAt: "2026-03-15T00:00:00Z",
			UpdatedAt: "2026-04-01T00:00:00Z",
		},
	}
}

func TestTUIIssueReportRanksAndLinksEpic(t *testing.T) {
	asOf, err := time.Parse("2006-01-02", "2026-06-25")
	if err != nil {
		t.Fatal(err)
	}
	report := buildTUIIssueReport(fixtureTUIIssues(), "fixture", asOf, 10)
	if report.Schema != tuiIssuesSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, tuiIssuesSchema)
	}
	if report.Epic == nil || report.Epic.Number != 10 {
		t.Fatalf("epic = %+v, want #10", report.Epic)
	}
	if report.Counts.Related != 2 {
		t.Fatalf("related count = %d, want 2", report.Counts.Related)
	}
	if report.Counts.Orphan != 1 {
		t.Fatalf("orphan count = %d, want 1", report.Counts.Orphan)
	}
	if len(report.Rows) == 0 || report.Rows[0].Number != 11 {
		t.Fatalf("top ranked row = %+v, want issue #11", report.Rows)
	}
	if !tuiHasTag(report.Rows[0], "orphan") || !tuiHasTag(report.Rows[0], "stale") {
		t.Fatalf("top row tags = %v, want orphan+stale", report.Rows[0].Tags)
	}
}

func TestTUIIssuesHumanOutputFromFixture(t *testing.T) {
	path := writeTUIIssuesFixture(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"issues",
		"--issues-json", path,
		"--as-of", "2026-06-25",
		"--epic", "10",
		"--top", "3",
		"--width", "100",
	})
	if code != 0 {
		t.Fatalf("runTUI code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak console issues", "Epic #10", "related loaded issues: 2", "Related", "#11", "orphan"} {
		if !strings.Contains(out, want) {
			t.Fatalf("human output missing %q:\n%s", want, out)
		}
	}
}

func TestTUIIssuesJSONOutputFromFixture(t *testing.T) {
	path := writeTUIIssuesFixture(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"issues",
		"--issues-json", path,
		"--as-of", "2026-06-25",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runTUI code=%d stderr=%s", code, stderr.String())
	}
	var report tuiIssueReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal report: %v\n%s", err, stdout.String())
	}
	if report.Schema != tuiIssuesSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, tuiIssuesSchema)
	}
	if report.Counts.Open != 4 || report.Counts.P0 != 2 || report.Counts.P1 != 1 {
		t.Fatalf("counts = %+v", report.Counts)
	}
	if len(report.Actions) == 0 {
		t.Fatalf("actions empty; report=%+v", report)
	}
}

func TestDecodeTUIIssuesAcceptsLiveCommentsArray(t *testing.T) {
	raw := []byte(`[{
		"number": 1,
		"title": "live shape",
		"state": "OPEN",
		"comments": [{"body":"first"}, {"body":"second"}],
		"labels": [{"name":"enhancement"}]
	}]`)
	issues, err := decodeTUIIssues(raw)
	if err != nil {
		t.Fatalf("decode live comments array: %v", err)
	}
	if len(issues) != 1 || int(issues[0].Comments) != 2 {
		t.Fatalf("issues = %+v, want comments count 2", issues)
	}
}

func TestTUILoopsHumanOutputFromLedger(t *testing.T) {
	path := writeTUILoopLedger(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"loops",
		"--ledger", path,
		"--at", "2026-06-25T12:10:00Z",
		"--top", "5",
		"--width", "110",
	})
	if code != 0 {
		t.Fatalf("runTUI loops code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak console loops", "running=1", "refused=1", "witness-gaps=1", "issue-dispatch/default", "needs-witness"} {
		if !strings.Contains(out, want) {
			t.Fatalf("loop output missing %q:\n%s", want, out)
		}
	}
}

func TestTUILoopsJSONOutputFromLedger(t *testing.T) {
	path := writeTUILoopLedger(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"loops",
		"--ledger", path,
		"--at", "2026-06-25T12:10:00Z",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runTUI loops code=%d stderr=%s", code, stderr.String())
	}
	var report tuiLoopReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal loop report: %v\n%s", err, stdout.String())
	}
	if report.Schema != tuiLoopsSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, tuiLoopsSchema)
	}
	if report.Counts.Loops != 3 || report.Counts.Running != 1 || report.Counts.Refused != 1 || report.Counts.WitnessGaps != 1 {
		t.Fatalf("counts = %+v", report.Counts)
	}
	if len(report.Rows) == 0 || report.Rows[0].LoopID != "issue-dispatch/default" {
		t.Fatalf("top row = %+v, want issue-dispatch/default first", report.Rows)
	}
}

func TestTUISessionsHumanOutputFromFixture(t *testing.T) {
	path := writeTUISessionsFixture(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"sessions",
		"--sessions-json", path,
		"--at", "2026-06-25T12:00:00Z",
		"--top", "5",
		"--width", "120",
	})
	if code != 0 {
		t.Fatalf("runTUI sessions code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak console sessions", "sessions=4", "paused=1", "stopped=1", "lineage=1", "sess-low", "low-turns"} {
		if !strings.Contains(out, want) {
			t.Fatalf("session output missing %q:\n%s", want, out)
		}
	}
}

func TestTUISessionsJSONOutputFromFixture(t *testing.T) {
	path := writeTUISessionsFixture(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"sessions",
		"--sessions-json", path,
		"--at", "2026-06-25T12:00:00Z",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runTUI sessions code=%d stderr=%s", code, stderr.String())
	}
	var report tuiSessionReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal session report: %v\n%s", err, stdout.String())
	}
	if report.Schema != tuiSessionsSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, tuiSessionsSchema)
	}
	if report.Counts.Sessions != 4 || report.Counts.Paused != 1 || report.Counts.Stopped != 1 || report.Counts.Budgeted != 2 || report.Counts.Lineage != 1 {
		t.Fatalf("counts = %+v", report.Counts)
	}
	if len(report.Rows) == 0 || report.Rows[0].TraceID != "sess-low" {
		t.Fatalf("top row = %+v, want sess-low first", report.Rows)
	}
}

func TestDecodeTUISessionsAcceptsArray(t *testing.T) {
	raw := []byte(`[{"trace_id":"sess-array","run":"running","budget":{"turns_left":-1,"tokens_left":-1},"rev":1}]`)
	list, err := decodeTUISessions(raw)
	if err != nil {
		t.Fatalf("decode session array: %v", err)
	}
	if list.Count != 1 || len(list.Sessions) != 1 || list.Sessions[0].TraceID != "sess-array" {
		t.Fatalf("list = %+v, want one sess-array", list)
	}
}

func TestTUIGardenHumanOutputFromFixture(t *testing.T) {
	path := writeTUIGardenFixture(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"garden",
		"--garden-json", path,
		"--check",
		"--at", "2026-06-25T12:00:00Z",
		"--width", "120",
	})
	if code != 0 {
		t.Fatalf("runTUI garden code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak console garden", "finding=garden_gate_red", "gate=1", "scorecard control pane", "red", "fresh status", "broken-loops"} {
		if !strings.Contains(out, want) {
			t.Fatalf("garden output missing %q:\n%s", want, out)
		}
	}
}

func TestTUIGardenJSONOutputFromFixture(t *testing.T) {
	path := writeTUIGardenFixture(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"garden",
		"--garden-json", path,
		"--check",
		"--at", "2026-06-25T12:00:00Z",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runTUI garden code=%d stderr=%s", code, stderr.String())
	}
	var report tuiGardenReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal garden report: %v\n%s", err, stdout.String())
	}
	if report.Schema != tuiGardenSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, tuiGardenSchema)
	}
	if report.Counts.Members != 3 || report.Counts.OK != 1 || report.Counts.Action != 1 || report.Counts.Red != 1 || report.Counts.Gating != 1 {
		t.Fatalf("counts = %+v", report.Counts)
	}
	if report.GateExit != 1 {
		t.Fatalf("gate exit = %d, want 1", report.GateExit)
	}
	if len(report.Rows) == 0 || report.Rows[0].Key != "scorecard" {
		t.Fatalf("top row = %+v, want scorecard first", report.Rows)
	}
	if !hasTUITag(report.Rows[0].Tags, "gates") || !hasTUITag(report.Rows[0].Tags, "red") {
		t.Fatalf("top row tags = %v, want red+gates", report.Rows[0].Tags)
	}
}

func TestTUIGuardHumanOutputFromFixtures(t *testing.T) {
	paths := writeTUIGuardFixtures(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"guard",
		"--guard-json", paths[0],
		"--guard-json", paths[1],
		"--at", "2026-06-25T12:00:00Z",
		"--width", "130",
	})
	if code != 0 {
		t.Fatalf("runTUI guard code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak console guard", "artifacts=2", "deny=4", "policy_block=1", "default_deny=3", "expected=1", "git_add", "Bash", "policy-block"} {
		if !strings.Contains(out, want) {
			t.Fatalf("guard output missing %q:\n%s", want, out)
		}
	}
}

func TestTUIGuardJSONOutputFromFixtures(t *testing.T) {
	paths := writeTUIGuardFixtures(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"guard",
		"--guard-json", paths[0],
		"--guard-json", paths[1],
		"--at", "2026-06-25T12:00:00Z",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runTUI guard code=%d stderr=%s", code, stderr.String())
	}
	var report tuiGuardReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal guard report: %v\n%s", err, stdout.String())
	}
	if report.Schema != tuiGuardSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, tuiGuardSchema)
	}
	if report.Status != "PASS" {
		t.Fatalf("status = %q, want PASS", report.Status)
	}
	if report.Counts.Artifacts != 2 || report.Counts.Allow != 35 || report.Counts.Deny != 4 || report.Counts.PolicyBlock != 1 || report.Counts.DefaultDeny != 3 || report.Counts.Expected != 1 {
		t.Fatalf("counts = %+v", report.Counts)
	}
	if len(report.Rows) == 0 || report.Rows[0].Tool != "Bash" {
		t.Fatalf("top row = %+v, want Bash non-allow sample first", report.Rows)
	}
	if !hasTUITag(report.Rows[0].Tags, "policy-block") || !hasTUITag(report.Rows[0].Tags, "sample") {
		t.Fatalf("top row tags = %v, want policy-block+sample", report.Rows[0].Tags)
	}
}

func TestTUIGuardCodexRecentAuditSurfacesActionability(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "codex-dos-recent-audit.json")
	payload := map[string]any{
		"schema":                    "fak-codex-dos-recent-audit/1",
		"status":                    "WARN",
		"sessions_audited":          20,
		"codex_threads_discovered":  20,
		"debug_command_body_should": "git commit -s -- private/path",
		"summary": map[string]any{
			"unknown_tree_admission_warnings": 0,
		},
		"actionability": map[string]any{
			"status": "WARN",
			"reasons": []any{
				"stop blocks or uncleared StopFailure API-wall breaker markers are present",
			},
			"residual": []any{
				"HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE",
				"HOST_SHELL_OPACITY",
			},
			"delegate_count": 0,
			"post_repair_shell_shape_counts": map[string]any{
				"shell_no_write_target_detected": 2290,
			},
			"post_repair_shell_family_counts": map[string]any{
				"git_write": 45,
				"search_rg": 281,
			},
			"post_repair_mutating_shell_family_counts": map[string]any{
				"git_write": 45,
			},
		},
		"codex_hook_fast_path": map[string]any{
			"status":              "PASS",
			"reason":              "Codex hook commands use the native launcher",
			"codex_command_modes": map[string]any{"native_launcher": 4},
		},
		"git_gate_evidence": map[string]any{
			"status":    "PASS",
			"proved_at": "2026-06-25T22:38:12Z",
			"missing":   []any{},
		},
		"workspace_stop_failures": map[string]any{
			"status":                                 "WARN",
			"markers":                                204,
			"total_failures":                         96,
			"nonzero_total_markers":                  64,
			"active_consecutive_markers":             48,
			"active_consecutive_total":               57,
			"recent_active_consecutive_markers":      3,
			"recent_active_consecutive_total":        3,
			"stale_active_consecutive_markers":       45,
			"stale_active_consecutive_total":         54,
			"origin_counts":                          map[string]any{"claude_transcript": 2, "dos_stream+claude_transcript": 1},
			"recent_active_origin_counts":            map[string]any{"claude_transcript": 1},
			"stale_active_origin_counts":             map[string]any{"dos_stream+claude_transcript": 1},
			"active_settlement_action_counts":        map[string]any{"RECENT_REVIEW": 3, "STALE_RESET_CANDIDATE": 45},
			"recent_active_settlement_action_counts": map[string]any{"RECENT_REVIEW": 3},
			"stale_active_settlement_action_counts":  map[string]any{"STALE_RESET_CANDIDATE": 45},
			"healed_nonzero_markers":                 16,
			"zero_total_markers":                     140,
			"max_consecutive":                        4,
			"top_active": []any{
				map[string]any{
					"session_id":        "b33eca05-c71e-4a51-979b-b4c8bcb43b1e",
					"mtime":             "2026-06-25T23:11:51Z",
					"total":             4,
					"consecutive":       4,
					"settlement_action": "STALE_RESET_CANDIDATE",
					"transcript": map[string]any{
						"status":  "FOUND",
						"account": ".claude",
						"project": "C--work-fak",
					},
					"transcript_summary": map[string]any{
						"evidence_tags": []any{"HOOK_OR_API_WALL_FEEDBACK"},
					},
				},
			},
			"top_recent_active": []any{
				map[string]any{
					"session_id":        "e7f31ce8-185b-4b6b-8e41-9db98bd1f4e6",
					"mtime":             "2026-06-25T23:10:51Z",
					"total":             1,
					"consecutive":       1,
					"settlement_action": "RECENT_REVIEW",
					"transcript": map[string]any{
						"status":  "FOUND",
						"account": ".claude",
						"project": "C--work-fak",
					},
					"transcript_summary": map[string]any{
						"evidence_tags": []any{"HOOK_OR_API_WALL_FEEDBACK", "HOST_PERMISSION_INTERRUPT"},
					},
				},
			},
			"recent": []any{
				map[string]any{
					"session_id":  "e7f31ce8-185b-4b6b-8e41-9db98bd1f4e6",
					"mtime":       "2026-06-25T23:10:51Z",
					"total":       1,
					"consecutive": 1,
					"transcript": map[string]any{
						"status":  "FOUND",
						"account": ".claude",
						"project": "C--work-fak",
					},
					"transcript_summary": map[string]any{
						"evidence_tags": []any{"HOOK_OR_API_WALL_FEEDBACK", "HOST_PERMISSION_INTERRUPT"},
					},
				},
				map[string]any{
					"session_id":  "2b8682ae-ced6-402b-bcc8-21180e96d5b3",
					"mtime":       "2026-06-25T23:01:02Z",
					"total":       3,
					"consecutive": 0,
					"transcript": map[string]any{
						"status":  "FOUND",
						"account": ".claude",
						"project": "C--work-fak",
					},
					"transcript_summary": map[string]any{
						"evidence_tags": []any{"SHELL_HEAVY_SESSION"},
					},
				},
			},
		},
	}
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal codex fixture: %v", err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("write codex fixture: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"guard",
		"--guard-json", path,
		"--at", "2026-06-25T12:00:00Z",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runTUI guard code=%d stderr=%s", code, stderr.String())
	}
	var report tuiGuardReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal guard report: %v\n%s", err, stdout.String())
	}
	if report.Status != "WARN" || report.Counts.Warn != 1 {
		t.Fatalf("status/counts = %s/%+v, want WARN with one warn source", report.Status, report.Counts)
	}
	encoded := string(stdout.Bytes())
	for _, want := range []string{"codex-actionability", "stopfailure-api-wall", "stopfailure-session", "codex-hook-fast-path", "codex-shell-opacity", "native_launcher", "HOST_SHELL_OPACITY", "git_write", "active_consecutive=57", "recent_consecutive=3", "RECENT_REVIEW", "HOOK_OR_API_WALL_FEEDBACK", "e7f31ce8-185b-4b6b-8e41-9db98bd1f4e6", "C--work-fak"} {
		if !strings.Contains(encoded, want) {
			t.Fatalf("codex guard JSON missing %q:\n%s", want, encoded)
		}
	}
	if strings.Contains(encoded, "git commit -s") || strings.Contains(encoded, "private/path") {
		t.Fatalf("codex guard JSON leaked command body:\n%s", encoded)
	}
	if len(report.Rows) == 0 || !hasTUITag(report.Rows[0].Tags, "warn") {
		t.Fatalf("top row = %+v, want warn-tagged Codex blocker", report.Rows)
	}

	stdout.Reset()
	stderr.Reset()
	code = runTUI(&stdout, &stderr, []string{"guard", "--guard-json", path, "--width", "150"})
	if code != 0 {
		t.Fatalf("runTUI guard human code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak console guard", "artifacts=1", "warn=1", "codex-actionability", "stopfailure-api-wall", "stopfailure-session", "warn"} {
		if !strings.Contains(out, want) {
			t.Fatalf("codex guard human missing %q:\n%s", want, out)
		}
	}
}

func TestTUIGuardStatusAuditSurfacesDefaultBlockers(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "guard-mcp-status.json")
	payload := map[string]any{
		"schema":              "fak-guard-mcp-status-audit/1",
		"status":              "PASS",
		"debug_prompt_should": "do not copy this raw prompt",
		"summary": map[string]any{
			"passed":                  13,
			"failed":                  0,
			"total":                   13,
			"default_blockers":        3,
			"active_default_blockers": 2,
		},
		"checks": []any{
			map[string]any{
				"name":   "status packet present",
				"status": "PASS",
				"detail": "status packet names the evidence and residual interpretation",
			},
		},
		"default_blockers": []any{
			map[string]any{
				"rank":         10,
				"code":         "WORKSPACE_RECENT_STOPFAILURE_API_WALL",
				"surface":      "workspace_dos",
				"status":       "ACTIVE",
				"next_action":  "Clear or rotate recent sessions.",
				"private_note": "git commit -s -- private/path",
				"evidence": map[string]any{
					"recent_active_markers":                  3,
					"recent_active_consecutive_total":        3,
					"active_consecutive_total":               57,
					"stale_active_consecutive_total":         54,
					"active_recent_threshold_hours":          6,
					"one_day_failures_total":                 96,
					"healed_nonzero_markers":                 16,
					"recent_active_origin_counts":            map[string]any{"claude_transcript": 2, "dos_stream+claude_transcript": 1},
					"active_settlement_action_counts":        map[string]any{"RECENT_REVIEW": 3, "STALE_RESET_CANDIDATE": 45},
					"recent_active_settlement_action_counts": map[string]any{"RECENT_REVIEW": 3},
					"top_recent_active_sessions_should":      []any{"raw session detail should not be copied"},
					"evidence_tag_counts": map[string]any{
						"HOOK_OR_API_WALL_FEEDBACK": 20,
						"HOST_PERMISSION_INTERRUPT": 20,
					},
				},
			},
			map[string]any{
				"rank":        20,
				"code":        "CODEX_HOST_SHELL_OPACITY",
				"surface":     "codex_hooks",
				"status":      "ACTIVE_DEBT",
				"next_action": "Prefer path-visible host tools.",
				"evidence": map[string]any{
					"shell_no_write_target_detected": 2290,
				},
			},
			map[string]any{
				"rank":        70,
				"code":        "OPENAI_AGENTS_SDK_NOT_INSTALLED",
				"surface":     "openai_hosted",
				"status":      "EXTERNAL_PREREQ",
				"next_action": "Install only when targeting hosted Agents.",
				"evidence": map[string]any{
					"blockers": []any{
						"openai-agents distribution is not installed",
						"importable agents module is not an installed OpenAI Agents SDK distribution",
					},
				},
			},
		},
	}
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal status audit fixture: %v", err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("write status audit fixture: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"guard",
		"--guard-json", path,
		"--at", "2026-06-25T12:00:00Z",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runTUI guard code=%d stderr=%s", code, stderr.String())
	}
	var report tuiGuardReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal guard report: %v\n%s", err, stdout.String())
	}
	if report.Status != "WARN" || report.Counts.Warn != 1 || report.Counts.Rows != 4 {
		t.Fatalf("status/counts = %s/%+v, want WARN with summary + 3 blocker rows", report.Status, report.Counts)
	}
	if len(report.Rows) == 0 || report.Rows[0].Reason != "WORKSPACE_RECENT_STOPFAILURE_API_WALL" {
		t.Fatalf("top row = %+v, want recent StopFailure blocker first", report.Rows)
	}
	if !hasTUITag(report.Rows[0].Tags, "blocker") || !hasTUITag(report.Rows[0].Tags, "active") {
		t.Fatalf("top row tags = %v, want blocker+active", report.Rows[0].Tags)
	}
	encoded := string(stdout.Bytes())
	for _, want := range []string{"guard-status-audit", "default-blocker", "WORKSPACE_RECENT_STOPFAILURE_API_WALL", "CODEX_HOST_SHELL_OPACITY", "OPENAI_AGENTS_SDK_NOT_INSTALLED", "recent_active_consecutive_total=3", "claude_transcript", "RECENT_REVIEW", "shell_no_write_target_detected=2290", "openai-agents distribution is not installed"} {
		if !strings.Contains(encoded, want) {
			t.Fatalf("status audit JSON missing %q:\n%s", want, encoded)
		}
	}
	for _, leak := range []string{"do not copy this raw prompt", "git commit -s", "raw session detail should not be copied"} {
		if strings.Contains(encoded, leak) {
			t.Fatalf("status audit JSON leaked %q:\n%s", leak, encoded)
		}
	}

	stdout.Reset()
	stderr.Reset()
	code = runTUI(&stdout, &stderr, []string{"guard", "--guard-json", path, "--width", "190"})
	if code != 0 {
		t.Fatalf("runTUI guard human code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak console guard", "warn=1", "default-blocker", "WORKSPACE_RECENT_STOPFAILURE_API_WALL", "CODEX_HOST_SHELL_OPACITY"} {
		if !strings.Contains(out, want) {
			t.Fatalf("status audit human missing %q:\n%s", want, out)
		}
	}
}

// writeGuardJournalFixture writes a small canonical-shaped guard decision journal:
// one ALLOW, two DENYs (POLICY_BLOCK + DEFAULT_DENY), and one QUARANTINE. Payload-free
// rows (digests/claims only), exactly as the real journal emits.
func writeGuardJournalFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "guard-audit.jsonl")
	lines := []string{
		`{"seq":1,"ts_unix_nano":1,"kind":"DECIDE","tool":"search_kb","verdict":"ALLOW","prev_hash":"","hash":"a"}`,
		`{"seq":2,"ts_unix_nano":2,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"POLICY_BLOCK","by":"floor","witness":"rm -rf /","prev_hash":"a","hash":"b"}`,
		`{"seq":3,"ts_unix_nano":3,"kind":"DENY","tool":"write_file","verdict":"DENY","reason":"DEFAULT_DENY","prev_hash":"b","hash":"c"}`,
		`{"seq":4,"ts_unix_nano":4,"kind":"QUARANTINE","tool":"fetch_url","verdict":"QUARANTINE","prev_hash":"c","hash":"d"}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatalf("write journal fixture: %v", err)
	}
	return path
}

// TestTUIGuardJournalJSON proves the #843 journal pane: tailing the canonical guard
// decision journal folds its adjudication rows through the SAME guard model — the
// denial surface (deny/policy_block/default_deny/quarantine) is surfaced and the
// highest-attention denial sorts to the top.
func TestTUIGuardJournalJSON(t *testing.T) {
	path := writeGuardJournalFixture(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"guard", "--journal", path, "--at", "2026-06-25T12:00:00Z", "--json",
	})
	if code != 0 {
		t.Fatalf("runTUI guard --journal code=%d stderr=%s", code, stderr.String())
	}
	var report tuiGuardReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal journal report: %v\n%s", err, stdout.String())
	}
	if report.Schema != tuiGuardSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, tuiGuardSchema)
	}
	c := report.Counts
	if c.Allow != 1 || c.Deny != 2 || c.PolicyBlock != 1 || c.DefaultDeny != 1 || c.Quarantine != 1 || c.Rows != 4 {
		t.Fatalf("journal counts = %+v, want allow=1 deny=2 policy_block=1 default_deny=1 quarantine=1 rows=4", c)
	}
	// Highest attention: DENY+POLICY_BLOCK (45+25) ties QUARANTINE (70); Kind asc breaks
	// the tie so "audit-deny" (Bash) sorts ahead of "audit-quarantine".
	if len(report.Rows) == 0 || report.Rows[0].Tool != "Bash" || report.Rows[0].Verdict != "DENY" {
		t.Fatalf("top row = %+v, want the Bash POLICY_BLOCK deny first", report.Rows)
	}
	if !hasTUITag(report.Rows[0].Tags, "policy-block") {
		t.Fatalf("top row tags = %v, want policy-block", report.Rows[0].Tags)
	}
}

// TestTUIGuardJournalHuman proves the denial surface renders in the human pane.
func TestTUIGuardJournalHuman(t *testing.T) {
	path := writeGuardJournalFixture(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"guard", "--journal", path, "--width", "130"})
	if code != 0 {
		t.Fatalf("runTUI guard --journal human code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak console guard", "deny=2", "policy_block=1", "default_deny=1", "quarantine=1", "Bash"} {
		if !strings.Contains(out, want) {
			t.Fatalf("journal pane missing %q:\n%s", want, out)
		}
	}
}

// TestTUIGuardJournalMissingIsEmptyPane proves a missing/empty journal yields a
// well-formed empty pane (exit 0), not an error — the acceptance for a not-yet-written
// journal.
func TestTUIGuardJournalMissingIsEmptyPane(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "never-written.jsonl")
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"guard", "--journal", missing, "--json"})
	if code != 0 {
		t.Fatalf("missing journal code=%d stderr=%s (want 0, empty pane)", code, stderr.String())
	}
	var report tuiGuardReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal empty report: %v\n%s", err, stdout.String())
	}
	if report.Status != "PASS" || report.Counts.Rows != 0 || len(report.Rows) != 0 {
		t.Fatalf("empty pane = status %q rows %d, want PASS with 0 rows", report.Status, report.Counts.Rows)
	}
}

// TestTUIGuardTailResolvesCanonical proves --tail resolves the canonical journal via
// FAK_AUDIT_JOURNAL and tolerates its absence as an empty pane.
func TestTUIGuardTailResolvesCanonical(t *testing.T) {
	path := writeGuardJournalFixture(t)
	t.Setenv("FAK_AUDIT_JOURNAL", path)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"guard", "--tail", "--json"})
	if code != 0 {
		t.Fatalf("--tail code=%d stderr=%s", code, stderr.String())
	}
	var report tuiGuardReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal --tail report: %v\n%s", err, stdout.String())
	}
	if report.Counts.Deny != 2 {
		t.Fatalf("--tail counts = %+v, want deny=2 from the canonical journal", report.Counts)
	}
}

// TestTUIGuardJournalAndArtifactConflict proves the two input modes are mutually
// exclusive (a coherent pane reads one source).
func TestTUIGuardJournalAndArtifactConflict(t *testing.T) {
	path := writeGuardJournalFixture(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"guard", "--journal", path, "--guard-json", path})
	if code != 2 {
		t.Fatalf("conflicting --journal + --guard-json code=%d, want 2\nstderr=%s", code, stderr.String())
	}
}

func TestTUIAgentDryRunDefaultsToClaudeGuardOAuth(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"agent",
		"--dry-run",
		"--at", "2026-06-25T12:00:00Z",
		"--prompt", "summarize the queue",
		"--width", "1000",
	})
	if code != 0 {
		t.Fatalf("runTUI agent dry-run code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak console agent", "backend=claude", "auth=claude-subscription-oauth", "fak", "guard", "--provider anthropic", "--anthropic-oauth", "claude -p"} {
		if !strings.Contains(out, want) {
			t.Fatalf("agent dry-run output missing %q:\n%s", want, out)
		}
	}
}

func TestTUIAgentJSONResolvesClaudeAccount(t *testing.T) {
	home := t.TempDir()
	gem8 := mkHome(t, home, ".claude-gem8-seat", "gem8@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"gem8-seat","dir":"` + jsonPath(gem8) + `","default":true},` +
		`{"name":"q","status":"tombstoned","rehome_to":"gem8-seat"}` +
		`]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"agent",
		"--json",
		"--account", "q",
		"--registry", regPath,
		"--home", home,
		"--at", "2026-06-25T12:00:00Z",
		"--",
		"--permission-mode", "bypassPermissions",
	})
	if code != 0 {
		t.Fatalf("runTUI agent json code=%d stderr=%s", code, stderr.String())
	}
	var report tuiAgentReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal agent report: %v\n%s", err, stdout.String())
	}
	if report.Schema != tuiAgentSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, tuiAgentSchema)
	}
	if report.Auth != "claude-subscription-oauth" || !hasTUIString(report.Launch, "--anthropic-oauth") {
		t.Fatalf("launch did not force Claude subscription OAuth: auth=%s launch=%v", report.Auth, report.Launch)
	}
	if report.Account != "q" || report.ResolvedAccount != "gem8-seat" || report.ClaudeConfigDir != gem8 {
		t.Fatalf("account resolution = account %q resolved %q dir %q, want q -> gem8-seat %q", report.Account, report.ResolvedAccount, report.ClaudeConfigDir, gem8)
	}
	if report.AccountIdentity != "gem8@example.test" {
		t.Fatalf("identity = %q, want gem8@example.test", report.AccountIdentity)
	}
	if !hasTUIAgentEnv(report.Env, "CLAUDE_CONFIG_DIR", gem8) {
		t.Fatalf("env = %+v, want CLAUDE_CONFIG_DIR=%s", report.Env, gem8)
	}
	if got := strings.Join(report.Command, " "); got != "claude --permission-mode bypassPermissions" {
		t.Fatalf("backend command = %q", got)
	}
	if !hasTUIString(report.Launch, "guard") || !hasTUIString(report.Launch, "--provider") || !hasTUIString(report.Launch, "anthropic") {
		t.Fatalf("launch does not route through fak guard anthropic: %v", report.Launch)
	}
}

func TestTUIAgentGatewayDryRunRedactsBearer(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	t.Setenv("API_TIMEOUT_MS", "")

	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"agent",
		"--dry-run",
		"--gateway-url", "http://node.example:8080/v1",
		"--model", "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M",
		"--prompt", "Reply with exactly: OK",
		"--width", "1000",
	})
	if code != 0 {
		t.Fatalf("runTUI agent gateway dry-run code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"provider=existing-fak-gateway", "auth=gateway-bearer", "gateway=http://node.example:8080", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "<redacted from FAK_GATEWAY_KEY>", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "API_TIMEOUT_MS", "claude -p"} {
		if !strings.Contains(out, want) {
			t.Fatalf("gateway dry-run output missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "super-secret-test-key") {
		t.Fatalf("gateway dry-run leaked bearer:\n%s", out)
	}
	if strings.Contains(out, "guard --provider") {
		t.Fatalf("gateway dry-run should launch direct agent, not local guard:\n%s", out)
	}
}

func TestTUIAgentGatewayJSONDoesNotEmbedBearer(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	t.Setenv("API_TIMEOUT_MS", "")

	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"agent",
		"--json",
		"--gateway-url", "http://node.example:8080",
		"--model", "qwen-local",
		"--prompt", "hello",
	})
	if code != 0 {
		t.Fatalf("runTUI agent gateway json code=%d stderr=%s", code, stderr.String())
	}
	if strings.Contains(stdout.String(), "super-secret-test-key") {
		t.Fatalf("gateway json leaked bearer:\n%s", stdout.String())
	}
	var report tuiAgentReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal agent gateway report: %v\n%s", err, stdout.String())
	}
	if report.Provider != "existing-fak-gateway" || report.Auth != "gateway-bearer" || report.GatewayURL != "http://node.example:8080" {
		t.Fatalf("gateway report header = provider %q auth %q url %q", report.Provider, report.Auth, report.GatewayURL)
	}
	if hasTUIString(report.Launch, "guard") {
		t.Fatalf("gateway launch should not include guard: %v", report.Launch)
	}
	if got := strings.Join(report.Command, " "); got != "claude -p hello" {
		t.Fatalf("backend command = %q", got)
	}
	if !hasTUIAgentEnvFrom(report.Env, "ANTHROPIC_API_KEY", "FAK_GATEWAY_KEY", true) {
		t.Fatalf("env = %+v, want sensitive ANTHROPIC_API_KEY from FAK_GATEWAY_KEY", report.Env)
	}
	if !hasTUIAgentEnv(report.Env, "ANTHROPIC_BASE_URL", "http://node.example:8080") {
		t.Fatalf("env = %+v, want ANTHROPIC_BASE_URL", report.Env)
	}
	if !hasTUIAgentEnv(report.Env, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1") {
		t.Fatalf("env = %+v, want CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1", report.Env)
	}
	if !hasTUIAgentEnv(report.Env, "ANTHROPIC_MODEL", "qwen-local") {
		t.Fatalf("env = %+v, want model tier override", report.Env)
	}
}

func TestTUIAgentGatewayRequiresKeyEnv(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "")

	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"agent",
		"--json",
		"--gateway-url", "http://node.example:8080",
	})
	if code != 2 {
		t.Fatalf("runTUI code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "requires FAK_GATEWAY_KEY to be set") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestTUIOverviewHumanOutputFromFixtures(t *testing.T) {
	guardPaths := writeTUIGuardFixtures(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"overview",
		"--issues-json", writeTUIIssuesFixture(t),
		"--epic", "10",
		"--ledger", writeTUILoopLedger(t),
		"--sessions-json", writeTUISessionsFixture(t),
		"--garden-json", writeTUIGardenFixture(t),
		"--guard-json", guardPaths[0],
		"--guard-json", guardPaths[1],
		"--check",
		"--as-of", "2026-06-25",
		"--at", "2026-06-25T12:10:00Z",
		"--width", "130",
	})
	if code != 0 {
		t.Fatalf("runTUI overview code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak console overview", "cards=5", "missing=0", "issues", "loops", "sessions", "garden", "guard", "garden-red"} {
		if !strings.Contains(out, want) {
			t.Fatalf("overview output missing %q:\n%s", want, out)
		}
	}
}

func TestTUIOverviewJSONOutputFromFixtures(t *testing.T) {
	guardPaths := writeTUIGuardFixtures(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{
		"overview",
		"--issues-json", writeTUIIssuesFixture(t),
		"--epic", "10",
		"--ledger", writeTUILoopLedger(t),
		"--sessions-json", writeTUISessionsFixture(t),
		"--garden-json", writeTUIGardenFixture(t),
		"--guard-json", guardPaths[0],
		"--guard-json", guardPaths[1],
		"--check",
		"--as-of", "2026-06-25",
		"--at", "2026-06-25T12:10:00Z",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runTUI overview code=%d stderr=%s", code, stderr.String())
	}
	var report tuiOverviewReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal overview report: %v\n%s", err, stdout.String())
	}
	if report.Schema != tuiOverviewSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, tuiOverviewSchema)
	}
	if report.Counts.Cards != 5 || report.Counts.Missing != 0 || report.Counts.Action != 4 || report.Counts.OK != 1 {
		t.Fatalf("counts = %+v", report.Counts)
	}
	if len(report.Cards) == 0 || report.Cards[0].Pane != "garden" {
		t.Fatalf("top card = %+v, want garden first", report.Cards)
	}
	if len(report.Actions) < 4 {
		t.Fatalf("actions = %+v, want action rows for pressure cards", report.Actions)
	}
}

func TestTUIRejectsUnknownSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"missing"})
	if code != 2 {
		t.Fatalf("runTUI code=%d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "unknown subcommand") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func hasTUITag(tags []string, want string) bool {
	for _, tag := range tags {
		if tag == want {
			return true
		}
	}
	return false
}

func hasTUIString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func hasTUIAgentEnv(env []tuiAgentEnv, name, value string) bool {
	for _, kv := range env {
		if kv.Name == name && kv.Value == value {
			return true
		}
	}
	return false
}

func hasTUIAgentEnvFrom(env []tuiAgentEnv, name, from string, sensitive bool) bool {
	for _, kv := range env {
		if kv.Name == name && kv.FromEnv == from && kv.Sensitive == sensitive {
			return true
		}
	}
	return false
}

func writeTUIIssuesFixture(t *testing.T) string {
	t.Helper()
	b, err := json.Marshal(fixtureTUIIssues())
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "issues.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func writeTUIGardenFixture(t *testing.T) string {
	t.Helper()
	payload := gardenbundle.Payload{
		OK:         false,
		Verdict:    "ACTION",
		Finding:    "garden_gate_red",
		Reason:     "garden gate RED -- scorecard control pane (score slipped)",
		NextAction: "retire the regression worst-first, then re-run go run ./cmd/fak garden --check",
		Workspace:  "C:\\work\\fak",
		Commit:     "abc1234",
		Members: []gardenbundle.MemberResult{
			{
				Key:      "scorecard",
				Label:    "scorecard control pane",
				Gates:    true,
				ExitCode: 1,
				State:    "red",
				OK:       false,
				Verdict:  "ACTION",
				Detail:   "score slipped",
			},
			{
				Key:     "fresh_status",
				Label:   "fresh status",
				State:   "ok",
				OK:      true,
				Verdict: "OK",
				Detail:  "fresh",
			},
			{
				Key:     "loop_audit",
				Label:   "fleet loop-audit",
				State:   "action",
				OK:      true,
				Verdict: "ACTION",
				Detail:  "1 loop(s) broken; 1 surfacing a condition",
				Counts:  map[string]int{"healthy": 2, "broken": 1, "action": 1},
			},
		},
		MemberCount: 3,
		Gating:      []string{"scorecard"},
	}
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal garden fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "garden.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("write garden fixture: %v", err)
	}
	return path
}

func writeTUIGuardFixtures(t *testing.T) []string {
	t.Helper()
	dir := t.TempDir()
	probe := map[string]any{
		"created_at":    "2026-06-25T10:15:36Z",
		"command_label": "git-add-deny",
		"dry_run":       true,
		"executed":      false,
		"expect_deny":   true,
		"expect_reason": "DEFAULT_DENY",
		"policy":        "examples/dev-agent-policy.json",
		"status":        "DENIED_EXPECTED",
		"tool":          "git_add",
		"preflight": map[string]any{
			"verdict":   "DENY",
			"reason":    "DEFAULT_DENY",
			"by":        "monitor",
			"exit_code": 0,
		},
	}
	historical := map[string]any{
		"schema":                     "fak-claude-historical-guard-audit/1",
		"created_at":                 "2026-06-25T14:58:24Z",
		"status":                     "PASS",
		"policy":                     "examples/dogfood-claude-policy.json",
		"sessions_audited":           6,
		"tool_calls_seen":            39,
		"unique_tool_calls_replayed": 38,
		"verdict_counts":             map[string]any{"ALLOW": 35, "DENY": 3},
		"reason_counts":              map[string]any{"DEFAULT_DENY": 2, "POLICY_BLOCK": 1},
		"transcript_shape": map[string]any{
			"summarized_sessions": 6,
			"max_result_chars":    56309,
			"evidence_tag_counts": map[string]any{
				"HOOK_OR_API_WALL_FEEDBACK": 6,
				"HOST_PERMISSION_INTERRUPT": 4,
			},
		},
		"top_friction_sessions": []any{
			map[string]any{
				"session_digest":   "abc123",
				"root_label":       ".claude/C--work-fak",
				"tool_calls":       12,
				"marker_lines":     44,
				"max_result_chars": 56309,
				"evidence_tags":    []any{"HOOK_OR_API_WALL_FEEDBACK", "HOST_PERMISSION_INTERRUPT"},
			},
		},
		"non_allow_samples": []any{
			map[string]any{
				"tool":        "TaskUpdate",
				"verdict":     "DENY",
				"reason":      "DEFAULT_DENY",
				"by":          "monitor",
				"call_digest": "c38bdf84780b2f80",
			},
			map[string]any{
				"tool":    "Bash",
				"verdict": "DENY",
				"reason":  "POLICY_BLOCK",
				"by":      "monitor",
				"claim":   "Bash.command deny_regex",
			},
		},
	}
	paths := []string{
		filepath.Join(dir, "git-add-deny.json"),
		filepath.Join(dir, "historical-guard.json"),
	}
	for i, payload := range []map[string]any{probe, historical} {
		b, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("marshal guard fixture: %v", err)
		}
		if err := os.WriteFile(paths[i], b, 0o600); err != nil {
			t.Fatalf("write guard fixture: %v", err)
		}
	}
	return paths
}

func writeTUISessionsFixture(t *testing.T) string {
	t.Helper()
	list := gateway.SessionListResponse{
		Sessions: []gateway.SessionState{
			{
				TraceID:  "sess-running",
				Run:      "running",
				Priority: 5,
				Budget:   gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1},
				Rev:      1,
			},
			{
				TraceID:  "sess-low",
				Run:      "paused",
				Priority: 0,
				Reason:   "operator-review",
				Budget:   gateway.SessionBudget{TurnsLeft: 1, TokensLeft: 750, ContextTokensLeft: 1500},
				Pace:     gateway.SessionPace{MaxTokensPerTurn: 512, MinTurnGapMs: 250},
				Rev:      3,
			},
			{
				TraceID:        "sess-child",
				Run:            "throttled",
				Priority:       2,
				Budget:         gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1, ContextTokensLeft: 64000},
				ContinuationID: "cont-1",
				ParentTrace:    "sess-parent",
				Generation:     1,
				Rev:            7,
			},
			{
				TraceID:  "sess-stopped",
				Run:      "stopped",
				Priority: 9,
				Reason:   "budget-drained",
				Budget:   gateway.SessionBudget{TurnsLeft: -1, TokensLeft: -1},
				Rev:      8,
			},
		},
		Count: 4,
	}
	b, err := json.Marshal(list)
	if err != nil {
		t.Fatalf("marshal session fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "sessions.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("write session fixture: %v", err)
	}
	return path
}

func writeTUILoopLedger(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	base := time.Date(2026, 6, 25, 12, 0, 0, 0, time.UTC)
	appendLoopFixtureEvent(t, path, base.Add(1*time.Minute), loopmgr.Event{
		LoopID: "issue-dispatch/default",
		Kind:   loopmgr.EventFire,
		Source: "schedule",
	})
	appendLoopFixtureEvent(t, path, base.Add(2*time.Minute), loopmgr.Event{
		LoopID: "issue-dispatch/default",
		RunID:  "run-1",
		Kind:   loopmgr.EventAdmit,
		Status: loopmgr.StatusRefused,
		Reason: "NO_PICKABLE_ISSUE",
	})
	appendLoopFixtureEvent(t, path, base.Add(3*time.Minute), loopmgr.Event{
		LoopID: "serve/session",
		RunID:  "run-2",
		Kind:   loopmgr.EventStart,
	})
	appendLoopFixtureEvent(t, path, base.Add(4*time.Minute), loopmgr.Event{
		LoopID:  "garden/nightly",
		RunID:   "run-3",
		Kind:    loopmgr.EventEnd,
		Status:  loopmgr.StatusClaimedDone,
		Summary: "child exited with code 0",
	})
	return path
}

func appendLoopFixtureEvent(t *testing.T, path string, at time.Time, ev loopmgr.Event) {
	t.Helper()
	if _, err := loopmgr.Append(path, ev, loopmgr.WithClock(func() time.Time { return at })); err != nil {
		t.Fatalf("append loop fixture event %+v: %v", ev, err)
	}
}
