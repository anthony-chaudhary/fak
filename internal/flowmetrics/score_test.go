package flowmetrics

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// score_test.go pins the readout hop #6198 owns: the eight graded axes reach an
// operator through `fak score flow` rather than only through a Go test, and the
// payload that arrives there is the LANDED envelope rather than a re-shaped copy.

// TestRunScoreJSONEmitsLandedEnvelope is the issue's witness: the verb exits 0 and
// its stdout parses as one object whose schema is fak-flow-metrics/1 and whose kpis
// array has length 8, with `defects` and `soft` array-typed on every entry.
//
// The two KPI arrays are checked as RAW JSON, not decoded into []string, because
// decoding erases exactly the distinction that matters: both `null` and `[]`
// unmarshal into a nil slice, and the control-pane reader indexes both fields.
func TestRunScoreJSONEmitsLandedEnvelope(t *testing.T) {
	root := scoreFixtureRepo(t)
	issues := filepath.Join(root, "issues.json")
	if err := os.WriteFile(issues, []byte(scoreFixtureIssues), 0o644); err != nil {
		t.Fatalf("write issues fixture: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := RunScore(context.Background(), &stdout, &stderr, []string{
		"--json", "--issues-file", issues, "--window", "30",
	}, root)
	if code != 0 {
		t.Fatalf("RunScore exit = %d, want 0 (a readout must not exit non-zero on flow debt); stderr=%s",
			code, stderr.String())
	}

	var envelope struct {
		Schema string                     `json:"schema"`
		Corpus map[string]json.RawMessage `json:"corpus"`
		KPIs   []map[string]json.RawMessage
	}
	if err := json.Unmarshal(stdout.Bytes(), &envelope); err != nil {
		t.Fatalf("stdout is not one JSON object: %v\n%s", err, stdout.String())
	}
	if envelope.Schema != Schema {
		t.Fatalf("schema = %q, want %q", envelope.Schema, Schema)
	}
	// flow_debt is the key the control-pane card is registered on; without it the
	// pane would report the card missing rather than collect a row.
	if _, ok := envelope.Corpus["flow_debt"]; !ok {
		t.Fatalf("corpus has no flow_debt key -- the control-pane row reads it: %v", envelope.Corpus)
	}
	if len(envelope.KPIs) != 8 {
		t.Fatalf("kpis has %d entries, want 8", len(envelope.KPIs))
	}
	want := []string{
		"flow_efficiency", "queue_time", "unstarted_backlog", "aging_wip",
		"atomicity", "arrival_vs_service", "witnessed_progress", "local_wip",
	}
	for i, k := range envelope.KPIs {
		var name string
		if err := json.Unmarshal(k["kpi"], &name); err != nil {
			t.Fatalf("kpi[%d] has no name: %v", i, err)
		}
		if name != want[i] {
			t.Fatalf("kpi[%d] = %q, want %q", i, name, want[i])
		}
		for _, field := range []string{"defects", "soft"} {
			raw, ok := k[field]
			if !ok {
				t.Fatalf("kpi %q omits %q -- the control-pane reader indexes it", name, field)
			}
			if len(raw) == 0 || raw[0] != '[' {
				t.Fatalf("kpi %q has %s = %s, want an array: `null` breaks the pane reader",
					name, field, raw)
			}
		}
	}
}

// TestRunScoreHumanReadoutNamesTheAgingSet pins the default (non---json) rendering:
// it prints the corpus line, one row per graded axis, and the aging in-flight
// section. The aging section is the reason RenderAging exists -- an aging count
// names no issue to finish -- so a readout that dropped it would leave the only
// actionable axis unactionable.
func TestRunScoreHumanReadoutNamesTheAgingSet(t *testing.T) {
	root := scoreFixtureRepo(t)
	issues := filepath.Join(root, "issues.json")
	if err := os.WriteFile(issues, []byte(scoreFixtureIssues), 0o644); err != nil {
		t.Fatalf("write issues fixture: %v", err)
	}

	var stdout, stderr bytes.Buffer
	if code := RunScore(context.Background(), &stdout, &stderr, []string{
		"--issues-file", issues,
	}, root); code != 0 {
		t.Fatalf("RunScore exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"flow metrics", "corpus:", "aging in-flight", "next:"} {
		if !strings.Contains(out, want) {
			t.Fatalf("human readout omits %q:\n%s", want, out)
		}
	}
	for _, axis := range []string{"flow_efficiency", "aging_wip", "local_wip"} {
		if !strings.Contains(out, axis) {
			t.Fatalf("human readout omits the %s axis:\n%s", axis, out)
		}
	}
	// A count rendered as a percentage would make a debt of 7 read as 7%.
	if strings.Contains(out, "defect(s)%") {
		t.Fatalf("flow_debt rendered as a percentage; it is a COUNT of tripped axes:\n%s", out)
	}
}

// TestRunScoreRejectsStrayPositional keeps the verb's argv contract explicit: a
// positional is a typo'd flag, and silently ignoring it would fold a corpus nobody
// asked for while still printing an authoritative-looking reading.
func TestRunScoreRejectsStrayPositional(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := RunScore(context.Background(), &stdout, &stderr, []string{"--json", "nonsense"}, t.TempDir()); code != 2 {
		t.Fatalf("exit = %d, want 2 for a stray positional", code)
	}
}

// TestRunScoreRefusesAnUnreadableIssuesFile pins the gather failure path: a dump
// that cannot be read is a usage error (2), not an empty corpus graded as clean.
func TestRunScoreRefusesAnUnreadableIssuesFile(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunScore(context.Background(), &stdout, &stderr, []string{
		"--json", "--issues-file", filepath.Join(t.TempDir(), "absent.json"),
	}, t.TempDir())
	if code != 2 {
		t.Fatalf("exit = %d, want 2 for an unreadable issues dump", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("a refused gather still wrote a payload: %s", stdout.String())
	}
}

// scoreFixtureIssues is a saved `gh issue list --json number,title,createdAt,closedAt,labels,body`
// dump small enough to read: one closed issue landed by the fixture commit, one open
// issue nobody started, and one epic with two children so the witnessed-progress axis
// has an aggregate to grade.
const scoreFixtureIssues = `[
  {"number":1,"title":"feat(flow): seed","createdAt":"2026-01-01T00:00:00Z","closedAt":"2026-01-02T00:00:00Z","labels":[],"body":"seed"},
  {"number":2,"title":"feat(flow): unstarted","createdAt":"2026-01-03T00:00:00Z","closedAt":null,"labels":[],"body":"nobody started this"},
  {"number":3,"title":"epic(flow): aggregate","createdAt":"2026-01-04T00:00:00Z","closedAt":null,"labels":[{"name":"epic"}],"body":"- [ ] #1\n- [ ] #2"}
]`

// scoreFixtureRepo builds a throwaway git repository with one commit whose subject
// references issue #1, so the gather has a real `git log` to read without touching the
// checkout this test runs inside. hooks are pointed at an empty directory: the shared
// checkout installs commit guards globally and a fixture commit must not trip them.
func scoreFixtureRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	hooks := filepath.Join(dir, "nohooks")
	if err := os.MkdirAll(hooks, 0o755); err != nil {
		t.Fatalf("mkdir hooks: %v", err)
	}
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "core.hooksPath", hooks},
		{"config", "user.email", "flow@example.test"},
		{"config", "user.name", "flow fixture"},
		{"config", "commit.gpgsign", "false"},
		{"commit", "--allow-empty", "-q", "-m", "feat(flow): land the seed (#1) (fak flowmetrics)"},
	} {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return dir
}
