package agent

import (
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestSessionEventStreamReplay(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "session_events.jsonl")
	log, err := NewSessionEventLog(logPath)
	if err != nil {
		t.Fatalf("NewSessionEventLog: %v", err)
	}

	obs := log.ModelRequestObserver()
	if err := obs(ModelRequestBoundary{
		Model: "qwen3.8",
		Turn:  1,
		Messages: []Message{
			{Role: "user", Content: "check status"},
		},
	}); err != nil {
		t.Fatalf("ModelRequestObserver turn 1: %v", err)
	}
	if _, err := log.RecordObservation(1, "call_1", "read_file", "ok", false); err != nil {
		t.Fatalf("RecordObservation: %v", err)
	}
	if err := obs(ModelRequestBoundary{
		Model: "qwen3.8",
		Turn:  2,
		Messages: []Message{
			{Role: "user", Content: "check status"},
			{Role: "tool", ToolCallID: "call_1", Content: "ok"},
		},
	}); err != nil {
		t.Fatalf("ModelRequestObserver turn 2: %v", err)
	}

	// Full replay reconstructs all 3 events.
	full, err := ReplaySessionEvents(logPath, 0)
	if err != nil {
		t.Fatalf("ReplaySessionEvents(0): %v", err)
	}
	if full.LastID != 3 || full.EventCount != 3 || full.LastTurn != 2 || full.Model != "qwen3.8" {
		t.Fatalf("unexpected full replay state: %+v", full)
	}
	if len(full.Messages) != 2 || len(full.Observations) != 1 || full.Observations[0].ToolName != "read_file" {
		t.Fatalf("unexpected messages/observations: msgs=%d obs=%+v", len(full.Messages), full.Observations)
	}

	// Cursor replay from ID=1 skips event 1 and replays only events 2 and 3.
	tail, err := ReplaySessionEvents(logPath, 1)
	if err != nil {
		t.Fatalf("ReplaySessionEvents(1): %v", err)
	}
	if tail.EventCount != 2 || tail.LastID != 3 || len(tail.Observations) != 1 {
		t.Fatalf("unexpected cursor replay state: %+v", tail)
	}

	// Reopening the log resumes monotonic IDs at 4.
	reopened, err := NewSessionEventLog(logPath)
	if err != nil {
		t.Fatalf("reopen NewSessionEventLog: %v", err)
	}
	ev4, err := reopened.RecordObservation(2, "call_2", "run_command", "done", false)
	if err != nil {
		t.Fatalf("append after reopen: %v", err)
	}
	if ev4.ID != 4 {
		t.Fatalf("expected resumed ID 4, got %d", ev4.ID)
	}
}
