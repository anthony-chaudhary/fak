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
		"unknown | no checkpoint for 1m | move unknown; unknown lease heartbeat 20s ago | check on next liveness change; worker owes checkpoint",
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
		{"lane":"healthy-b","chip":"ADVANCING","loop_ts":"2026-08-17T16:25:00Z","holder":"healthy-b","heartbeat_age_ms":20000},
		{"lane":"intent-only","chip":"ADVANCING","loop_ts":"2026-08-17T16:26:00Z","holder":"intent-only","heartbeat_age_ms":20000}
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
	intent := devcheckpoint.Record{Timestamp: now.Add(-2 * time.Minute), Actor: "intent-only", State: devcheckpoint.StateProgress, Stage: &devcheckpoint.Stage{Current: 1, Total: 3, Name: "bounded move", Percent: 33}, Summary: "work identified", Next: "inspect test result"}
	f, err := os.OpenFile(checkpointPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(mustJSON(t, intent) + "\n"); err != nil {
		f.Close()
		t.Fatal(err)
	}
	f.Close()
	command := func(context.Context, string, ...string) ([]byte, error) { return []byte(payload), nil }
	var stdout, stderr bytes.Buffer
	if code := runOperatorLiveSignals(&stdout, &stderr, command, now, false); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{
		"watch | no witnessed outcome since lease start 1h ago | move unknown; watch-lane lease heartbeat 20m ago | inspect holder watcher on watch-lane stalled lease now",
		"unknown x2 | no checkpoint | 2 live workers | check on next liveness change; workers owe checkpoints; --full lists workers",
		"unknown x1 | checkpoints without witnessed outcomes | 1 live worker | run declared next checks; --full lists workers",
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
	if strings.Contains(got, "none x3") {
		t.Fatalf("evidence-free checkpoint was conflated with a witnessed outcome:\n%s", got)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runOperatorLiveSignals(&stdout, &stderr, command, now, true); code != 0 {
		t.Fatalf("full code=%d stderr=%s", code, stderr.String())
	}
	wantIntent := "unknown | no witnessed outcome; checkpoint 2m ago | intent-only: bounded move 2m ago | inspect test result"
	if !strings.Contains(stdout.String(), wantIntent) {
		t.Fatalf("full projection lost evidence-free intent %q:\n%s", wantIntent, stdout.String())
	}
}

func TestOperatorLiveSignalsUsesGraceAsUnknownNextCheck(t *testing.T) {
	now := time.Date(2026, 8, 17, 16, 30, 0, 0, time.UTC)
	payload := `{"lanes":[
		{"lane":"young-a","chip":"ADVANCING","loop_ts":"2026-08-17T16:27:00Z","holder":"young-a","heartbeat_age_ms":180000,"liveness_reason":"alive and only 180000 ms into the run (< grace 1800000 ms)"},
		{"lane":"young-b","chip":"ADVANCING","loop_ts":"2026-08-17T16:28:00Z","holder":"young-b","heartbeat_age_ms":120000,"liveness_reason":"alive and only 120000 ms into the run (< grace 1800000 ms)"}
	]}`
	oldJoin := filepathJoin
	t.Cleanup(func() { filepathJoin = oldJoin })
	filepathJoin = func(...string) string { return filepath.Join(t.TempDir(), "missing.jsonl") }
	command := func(context.Context, string, ...string) ([]byte, error) { return []byte(payload), nil }
	var stdout, stderr bytes.Buffer
	if code := runOperatorLiveSignals(&stdout, &stderr, command, now, false); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	want := "unknown x2 | no checkpoint | 2 live workers | check at earliest grace in 27m; workers owe checkpoints; --full lists workers"
	if !strings.Contains(stdout.String(), want) {
		t.Fatalf("projection missing %q:\n%s", want, stdout.String())
	}
	if strings.Contains(stdout.String(), "emit durable checkpoints") {
		t.Fatalf("worker remediation leaked into operator next check:\n%s", stdout.String())
	}
}

func TestStalledLeaseNextIdentifiesHolderAndDegradesHonestly(t *testing.T) {
	t.Parallel()

	if got, want := stalledLeaseNext(operatorWorkerRow{Lane: "lane-a", Holder: "worker-7"}), "inspect holder worker-7 on lane-a stalled lease now"; got != want {
		t.Fatalf("holder next = %q, want %q", got, want)
	}
	if got, want := stalledLeaseNext(operatorWorkerRow{Lane: "lane-a"}), "inspect lane-a stalled lease now; holder unknown"; got != want {
		t.Fatalf("missing-holder next = %q, want %q", got, want)
	}
}

func TestOperatorLiveSignalsOrdersExceptionsByOldestEvidence(t *testing.T) {
	now := time.Date(2026, 8, 17, 16, 30, 0, 0, time.UTC)
	payload := `{"lanes":[
		{"lane":"a-watch-newer","chip":"STALLED","loop_ts":"2026-08-17T16:00:00Z","holder":"a-watch-newer","heartbeat_age_ms":1200000},
		{"lane":"z-watch-older","chip":"STALLED","loop_ts":"2026-08-17T15:00:00Z","holder":"z-watch-older","heartbeat_age_ms":3600000},
		{"lane":"a-human-newer","chip":"ADVANCING","loop_ts":"2026-08-17T16:00:00Z","holder":"a-human-newer","heartbeat_age_ms":20000},
		{"lane":"z-human-older","chip":"ADVANCING","loop_ts":"2026-08-17T15:00:00Z","holder":"z-human-older","heartbeat_age_ms":20000}
	]}`
	oldJoin := filepathJoin
	t.Cleanup(func() { filepathJoin = oldJoin })
	checkpointPath := filepath.Join(t.TempDir(), "dev-status.jsonl")
	filepathJoin = func(...string) string { return checkpointPath }
	f, err := os.OpenFile(checkpointPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	for _, record := range []devcheckpoint.Record{
		{Timestamp: now.Add(-10 * time.Minute), Actor: "a-human-newer", State: devcheckpoint.StateBlocked, Summary: "newer decision", Blockers: []string{"operator: choose"}, Next: "choose"},
		{Timestamp: now.Add(-40 * time.Minute), Actor: "z-human-older", State: devcheckpoint.StateBlocked, Summary: "older decision", Blockers: []string{"operator: choose"}, Next: "choose"},
	} {
		if _, err := f.WriteString(mustJSON(t, record) + "\n"); err != nil {
			f.Close()
			t.Fatal(err)
		}
	}
	f.Close()
	command := func(context.Context, string, ...string) ([]byte, error) { return []byte(payload), nil }
	var stdout, stderr bytes.Buffer
	if code := runOperatorLiveSignals(&stdout, &stderr, command, now, false); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	got := stdout.String()
	for _, pair := range [][2]string{
		{"z-human-older: older decision 40m ago", "a-human-newer: newer decision 10m ago"},
		{"move unknown; z-watch-older lease heartbeat 1h ago", "move unknown; a-watch-newer lease heartbeat 20m ago"},
	} {
		older := strings.Index(got, pair[0])
		newer := strings.Index(got, pair[1])
		if older < 0 || newer < 0 || older > newer {
			t.Fatalf("oldest exception %q was not before %q:\n%s", pair[0], pair[1], got)
		}
	}
}

func TestProjectLiveSignalIgnoresCheckpointBeforeCurrentLease(t *testing.T) {
	now := time.Date(2026, 8, 17, 16, 30, 0, 0, time.UTC)
	heartbeatAge := int64(20_000)
	lane := operatorWorkerRow{
		Lane:           "reused-worker",
		Chip:           "ADVANCING",
		LoopTS:         now.Add(-5 * time.Minute).Format(time.RFC3339),
		HeartbeatAgeMS: &heartbeatAge,
	}
	old := devcheckpoint.Record{
		Timestamp: now.Add(-10 * time.Minute),
		Actor:     "reused-worker",
		State:     devcheckpoint.StateDone,
		Summary:   "older run shipped",
		Evidence:  []string{"commit-old"},
	}
	row := projectLiveSignal(lane, old, now)
	if row.Attention != "unknown" || row.HasCheckpoint || row.Outcome != "no checkpoint for 5m" {
		t.Fatalf("older run checkpoint leaked into current lease: %+v", row)
	}

	current := old
	current.Timestamp = now.Add(-2 * time.Minute)
	current.Evidence = []string{"commit-current"}
	row = projectLiveSignal(lane, current, now)
	if row.Attention != "none" || !row.HasCheckpoint || !strings.Contains(row.Outcome, "commit-current") {
		t.Fatalf("current-run checkpoint was not joined: %+v", row)
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
