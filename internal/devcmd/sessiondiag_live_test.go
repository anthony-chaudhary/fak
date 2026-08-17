package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/devcheckpoint"
)

func TestOperatorLiveSignalsCapturedProjection(t *testing.T) {
	now := time.Date(2026, 8, 17, 16, 30, 0, 0, time.UTC)
	payload := `{"lanes":[
		{"lane":"healthy","chip":"🟢 ADVANCING","loop_ts":"2026-08-17T16:25:00Z","holder":"worker-ok","heartbeat_age_ms":20000,"liveness_reason":"alive and only 300000 ms into the run (< grace 1800000 ms)"},
		{"lane":"stale","chip":"🔴 STALLED","loop_ts":"2026-08-17T15:30:00Z","holder":"worker-stale","heartbeat_age_ms":1200000,"liveness_reason":"heartbeat old"},
		{"lane":"decision","chip":"🟢 ADVANCING","loop_ts":"2026-08-17T16:20:00Z","holder":"worker-human","heartbeat_age_ms":20000,"liveness_reason":"alive"},
		{"lane":"unknown","chip":"🟢 ADVANCING","loop_ts":"2026-08-17T16:29:00Z","holder":"worker-unknown","heartbeat_age_ms":20000,"liveness_reason":"alive"}
	]}`
	oldJoin := filepathJoin
	t.Cleanup(func() { filepathJoin = oldJoin })
	checkpointPath := filepath.Join(t.TempDir(), "dev-status.jsonl")
	filepathJoin = func(...string) string { return checkpointPath }
	writeCheckpoint := func(record devcheckpoint.Record) {
		f, err := os.OpenFile(checkpointPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if _, err := f.WriteString(mustJSON(t, record) + "\n"); err != nil {
			t.Fatal(err)
		}
	}
	writeCheckpoint(devcheckpoint.Record{Timestamp: now.Add(-2 * time.Minute), Actor: "worker-ok", State: devcheckpoint.StateProgress, Stage: &devcheckpoint.Stage{Current: 2, Total: 3, Percent: 67, Name: "testing"}, Summary: "tests running", Evidence: []string{"commit abc123"}, Next: "check when gate exits"})
	writeCheckpoint(devcheckpoint.Record{Timestamp: now.Add(-24 * time.Minute), Actor: "worker-stale", State: devcheckpoint.StateProgress, Summary: "repeating tests", Evidence: []string{"checkpoint stale"}, Next: "intervene at retry budget"})
	writeCheckpoint(devcheckpoint.Record{Timestamp: now.Add(-7 * time.Minute), Actor: "worker-human", State: devcheckpoint.StateBlocked, Blockers: []string{"operator: retry or stop decision"}, Summary: "waiting for operator", Evidence: []string{"checkpoint choice-ready"}, Next: "choose retry or stop"})

	command := func(_ context.Context, name string, args ...string) ([]byte, error) {
		if name != "dos" || strings.Join(args, " ") != "top --json" {
			t.Fatalf("unexpected command %s %v", name, args)
		}
		return []byte(payload), nil
	}
	var stdout, stderr bytes.Buffer
	if code := runOperatorLiveSignals(&stdout, &stderr, command, now, true); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{
		"ATTENTION | OUTCOME + AGE | CURRENT MOVE + AGE | NEXT CHECK",
		"needs-human | checkpoint evidence checkpoint choice-ready 7m ago | decision: waiting for operator 7m ago | choose retry or stop",
		"watch | checkpoint evidence checkpoint stale 24m ago | stale: repeating tests 24m ago | intervene at retry budget",
		"unknown | no checkpoint for 1m | unknown lease heartbeat 20s ago | emit a durable checkpoint",
		"none | checkpoint evidence commit abc123 2m ago | healthy: testing 2m ago | check when gate exits",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("captured projection missing %q:\n%s", want, got)
		}
	}
	if strings.Index(got, "needs-human") > strings.Index(got, "watch") || strings.Index(got, "watch") > strings.Index(got, "unknown") || strings.Index(got, "unknown") > strings.Index(got, "none") {
		t.Fatalf("attention order is not needs-human, watch, unknown, none:\n%s", got)
	}
	if !strings.Contains(got, "unknown | no checkpoint for 1m |") {
		t.Fatalf("a live heartbeat without a witnessed outcome must remain unknown, not healthy:\n%s", got)
	}
	for _, forbidden := range []string{"transcript", "tokens", "tool_count", "tool count"} {
		if strings.Contains(strings.ToLower(got), forbidden) {
			t.Fatalf("default projection leaked drill-down %q:\n%s", forbidden, got)
		}
	}
}

func TestOperatorLiveSignalsDefaultFoldsOnlyNonActionRows(t *testing.T) {
	now := time.Date(2026, 8, 17, 16, 30, 0, 0, time.UTC)
	payload := `{"lanes":[
		{"lane":"watch-lane","chip":"STALLED","loop_ts":"2026-08-17T15:30:00Z","holder":"watcher","heartbeat_age_ms":1200000},
		{"lane":"unknown-a","chip":"ADVANCING","loop_ts":"2026-08-17T16:29:00Z","holder":"unknown-a","heartbeat_age_ms":20000},
		{"lane":"unknown-b","chip":"ADVANCING","loop_ts":"2026-08-17T16:28:00Z","holder":"unknown-b","heartbeat_age_ms":30000},
		{"lane":"healthy-a","chip":"ADVANCING","loop_ts":"2026-08-17T16:25:00Z","holder":"healthy-a","heartbeat_age_ms":20000},
		{"lane":"healthy-b","chip":"ADVANCING","loop_ts":"2026-08-17T16:25:00Z","holder":"healthy-b","heartbeat_age_ms":20000}
	]}`
	oldJoin := filepathJoin
	t.Cleanup(func() { filepathJoin = oldJoin })
	checkpointPath := filepath.Join(t.TempDir(), "dev-status.jsonl")
	filepathJoin = func(...string) string { return checkpointPath }
	for _, actor := range []string{"healthy-a", "healthy-b"} {
		record := devcheckpoint.Record{Timestamp: now.Add(-time.Minute), Actor: actor, State: devcheckpoint.StateProgress, Summary: "testing", Evidence: []string{"commit abc"}, Next: "wait for gate"}
		f, err := os.OpenFile(checkpointPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.WriteString(mustJSON(t, record) + "\n"); err != nil {
			f.Close()
			t.Fatal(err)
		}
		f.Close()
	}
	command := func(context.Context, string, ...string) ([]byte, error) { return []byte(payload), nil }
	var stdout, stderr bytes.Buffer
	if code := runOperatorLiveSignals(&stdout, &stderr, command, now, false); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{
		"watch | no witnessed outcome since lease start 1h ago | watch-lane lease heartbeat 20m ago | inspect stalled lease now",
		"unknown x2 | no witnessed outcome | 2 live workers | emit durable checkpoints; --full lists workers",
		"none x2 | witnessed outcomes present | 2 live workers | bounded next checks; --full lists workers",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("default projection missing %q:\n%s", want, got)
		}
	}
	for _, hidden := range []string{"unknown-a lease heartbeat", "unknown-b lease heartbeat", "healthy-a: testing", "healthy-b: testing"} {
		if strings.Contains(got, hidden) {
			t.Fatalf("default projection did not fold %q:\n%s", hidden, got)
		}
	}
	if strings.Contains(got, "none x4") {
		t.Fatalf("unknown workers were conflated with healthy workers:\n%s", got)
	}
}

func TestRunSessionDiagFullRequiresLiveSignals(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runSessionDiagWith(&stdout, &stderr, []string{"--full"}, nil, nil, time.Now)
	if code != 2 || !strings.Contains(stderr.String(), "--full requires --live-signals") {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}
