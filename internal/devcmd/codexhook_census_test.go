package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestBuildHookCensusClassifiesCompleteLifecycle(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "profile")
	sessions := filepath.Join(home, "sessions")
	if err := os.MkdirAll(sessions, 0755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	rows := `{"timestamp":"2026-08-17T11:59:00Z","type":"session_meta","payload":{"cwd":"` + filepath.ToSlash(home) + `"}}` + "\n" + `{"timestamp":"2026-08-17T11:00:00Z","type":"response_item","payload":{"type":"function_call","call_id":"a"}}` + "\n" + `{"timestamp":"2026-08-17T11:01:00Z","type":"response_item","payload":{"type":"custom_tool_call","call_id":"b"}}` + "\n"
	if err := os.WriteFile(filepath.Join(sessions, "r.jsonl"), []byte(rows), 0644); err != nil {
		t.Fatal(err)
	}
	obs := filepath.Join(root, "observations.jsonl")
	var b bytes.Buffer
	for _, verb := range []string{"pretool", "pretool", "posttool", "posttool"} {
		_ = json.NewEncoder(&b).Encode(hookObservation{Exit: 0, Verb: verb, TS: now.Add(-time.Minute)})
	}
	if err := os.WriteFile(obs, b.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := buildHookCensus(home, home, home, "", obs, time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "HEALTHY" || got.DispatchedCalls != 2 || got.PreToolUse.Succeeded != 2 || got.PostToolUse.Unknown != 0 {
		t.Fatalf("report=%+v", got)
	}
}
func TestBuildHookCensusFailsClosedOnUnknownFailureAndProfileMismatch(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "active")
	logHome := filepath.Join(root, "other")
	if err := os.MkdirAll(filepath.Join(logHome, "sessions"), 0755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	row := `{"timestamp":"2026-08-17T11:59:00Z","type":"session_meta","payload":{"cwd":"` + filepath.ToSlash(logHome) + `"}}` + "\n" + `{"timestamp":"2026-08-17T11:00:00Z","type":"response_item","payload":{"type":"function_call","call_id":"a"}}` + "\n"
	_ = os.WriteFile(filepath.Join(logHome, "sessions", "r.jsonl"), []byte(row), 0644)
	obs := filepath.Join(root, "o.jsonl")
	o := hookObservation{Exit: 3, Verb: "posttool", TS: now.Add(-time.Minute)}
	raw, _ := json.Marshal(o)
	_ = os.WriteFile(obs, append(raw, '\n'), 0644)
	got, err := buildHookCensus(home, logHome, logHome, "", obs, time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "UNHEALTHY" || got.ProfileMatch || got.PreToolUse.Unknown != 1 || got.PostToolUse.Failed != 1 {
		t.Fatalf("report=%+v", got)
	}
}

func TestPhaseCountsTypesSkippedDisabledAndUnknown(t *testing.T) {
	got := phaseCounts(4, []hookObservation{{Exit: 0, Outcome: "passthrough"}, {Outcome: "skipped"}, {Outcome: "disabled"}}, "fixture")
	if got.Attempted != 1 || got.Succeeded != 1 || got.Skipped != 1 || got.Disabled != 1 || got.Unknown != 1 {
		t.Fatalf("counts=%+v", got)
	}
}

func TestBuildHookCensusRejectsAbsentTelemetry(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	if err := os.MkdirAll(filepath.Join(home, "sessions"), 0755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	row := `{"timestamp":"2026-08-17T11:59:00Z","type":"session_meta","payload":{"cwd":"` + filepath.ToSlash(home) + `"}}` + "\n" + `{"timestamp":"2026-08-17T11:00:00Z","type":"response_item","payload":{"type":"function_call","call_id":"a"}}` + "\n"
	_ = os.WriteFile(filepath.Join(home, "sessions", "r.jsonl"), []byte(row), 0644)
	obs := filepath.Join(root, "o.jsonl")
	_ = os.WriteFile(obs, nil, 0644)
	got, err := buildHookCensus(home, home, home, "", obs, time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "UNHEALTHY" || got.PreToolUse.Unknown != 1 || got.PostToolUse.Unknown != 1 {
		t.Fatalf("report=%+v", got)
	}
}
