package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func writeLoopArchiveFixture(t *testing.T, root, sessionID string) string {
	t.Helper()
	path := filepath.Join(root, "sessions", "2026", "08", "17", "rollout-"+sessionID+".jsonl")
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	lines := []string{
		`{"timestamp":"2026-08-17T10:00:00Z","type":"session_meta","payload":{"id":"` + sessionID + `","originator":"codex_exec","cwd":"C:/work/fak","model_provider":"fak"}}`,
		`{"timestamp":"2026-08-17T10:00:01Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{}","call_id":"a"}}`,
		`{"timestamp":"2026-08-17T10:00:02Z","type":"response_item","payload":{"type":"function_call_output","call_id":"a","output":"Plan updated"}}`,
		`{"timestamp":"2026-08-17T10:00:03Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{}","call_id":"b"}}`,
		`{"timestamp":"2026-08-17T10:00:04Z","type":"response_item","payload":{"type":"function_call_output","call_id":"b","output":"Plan updated"}}`,
		`{"timestamp":"2026-08-17T10:00:05Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{}","call_id":"c"}}`,
		`{"timestamp":"2026-08-17T10:00:06Z","type":"response_item","payload":{"type":"function_call_output","call_id":"c","output":"Plan updated"}}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func withArchiveSessionRows(t *testing.T, state sessionregistry.State, sessionID string) {
	t.Helper()
	old := readSessionRows
	readSessionRows = func() ([]sessionregistry.Record, error) {
		return []sessionregistry.Record{{State: state, Identity: sessionregistry.Identity{Runtime: "codex", SessionID: sessionID}}}, nil
	}
	t.Cleanup(func() { readSessionRows = old })
}

func TestCodexLoopArchiveRefusesActiveAndAmbiguousSessions(t *testing.T) {
	for _, tc := range []struct {
		name      string
		state     sessionregistry.State
		sessionID string
		want      string
	}{
		{name: "active", state: sessionregistry.StateActive, sessionID: "active", want: "lifecycle is live"},
		{name: "ambiguous", state: sessionregistry.StateActive, sessionID: "other", want: "lifecycle is ambiguous"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			path := writeLoopArchiveFixture(t, home, tc.name)
			withArchiveSessionRows(t, tc.state, tc.sessionID)
			_, err := archiveTerminalCodexLoop(path, home, false)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want %q", err, tc.want)
			}
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("source moved on refusal: %v", err)
			}
		})
	}
}

func TestCodexLoopArchiveDryRunThenMovesWithRecoverableManifest(t *testing.T) {
	home := t.TempDir()
	path := writeLoopArchiveFixture(t, home, "done")
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	withArchiveSessionRows(t, sessionregistry.StateCompleted, "done")
	oldNow := codexLoopArchiveNow
	codexLoopArchiveNow = func() time.Time { return time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC) }
	t.Cleanup(func() { codexLoopArchiveNow = oldNow })

	dry, err := archiveTerminalCodexLoop(path, home, true)
	if err != nil {
		t.Fatal(err)
	}
	if !dry.DryRun {
		t.Fatalf("dry=%+v", dry)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("dry-run moved source: %v", err)
	}

	got, err := archiveTerminalCodexLoop(path, home, false)
	if err != nil {
		t.Fatal(err)
	}
	if got.DryRun || !strings.Contains(filepath.Base(got.Destination), got.SHA256[:12]) {
		t.Fatalf("manifest=%+v", got)
	}
	archived, err := os.ReadFile(got.Destination)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(archived, original) {
		t.Fatal("archived bytes changed")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("source still present: %v", err)
	}
	manifestPath := strings.TrimSuffix(got.Destination, ".jsonl") + ".manifest.json"
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var persisted codexLoopArchiveManifest
	if err := json.Unmarshal(raw, &persisted); err != nil || persisted.Source != path || persisted.SHA256 != got.SHA256 {
		t.Fatalf("persisted=%+v err=%v", persisted, err)
	}

	recent, err := diagnoseRecentCodexLoops(home, 24, 20)
	if err != nil {
		t.Fatal(err)
	}
	if recent.LoopCount != 0 || recent.Verdict != "OK" {
		t.Fatalf("post-archive gate report=%+v", recent)
	}

	again, err := archiveTerminalCodexLoop(path, home, false)
	if err != nil {
		t.Fatal(err)
	}
	if !again.Idempotent || again.Destination != got.Destination {
		t.Fatalf("again=%+v", again)
	}
}

func TestSessionsCodexLoopArchiveCommandEmitsJSON(t *testing.T) {
	home := t.TempDir()
	path := writeLoopArchiveFixture(t, home, "failed")
	withArchiveSessionRows(t, sessionregistry.StateFailed, "failed")
	var stdout, stderr bytes.Buffer
	if code := sessionsCodexLoop(&stdout, &stderr, []string{"archive", "--path", path, "--codex-home", home, "--dry-run", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got codexLoopArchiveManifest
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil || !got.DryRun || got.SessionID != "failed" {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}

func TestCodexLoopArchiveRefusesSourceOutsideCodexSessions(t *testing.T) {
	home := t.TempDir()
	outside := t.TempDir()
	path := writeLoopArchiveFixture(t, outside, "done")
	withArchiveSessionRows(t, sessionregistry.StateCompleted, "done")
	_, err := archiveTerminalCodexLoop(path, home, false)
	if err == nil || !strings.Contains(err.Error(), "outside") {
		t.Fatalf("err=%v", err)
	}
}
