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
