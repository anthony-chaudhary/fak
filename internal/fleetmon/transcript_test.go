package fleetmon

import (
	"testing"
	"time"
)

func TestReadTranscriptMissingFileIsNotError(t *testing.T) {
	sig := ReadTranscript(t.TempDir() + "/does-not-exist.jsonl")
	if sig.Exists {
		t.Fatal("missing transcript must report Exists=false")
	}
	if sig.Error != "" {
		t.Fatalf("missing transcript must not be an error, got %q", sig.Error)
	}
}

func TestReadTranscriptFinalReport(t *testing.T) {
	now := time.Now()
	path := writeTranscript(t,
		assistantToolUse(t, tsAt(now, 5), "Read", map[string]any{"file_path": "a.go"}),
		userToolResult(t, tsAt(now, 4), "file contents", false),
		assistantText(t, tsAt(now, 1), "Done — fixed the bug in a.go and `go test` passes.", "end_turn"),
	)
	sig := ReadTranscript(path)
	if !sig.Exists {
		t.Fatal("transcript should exist")
	}
	if !sig.FinalReport {
		t.Fatal("a trailing assistant end_turn text turn must be a final report")
	}
	if sig.FinalReportText == "" {
		t.Fatal("final report text should be captured")
	}
	if !sig.HasTimestamp {
		t.Fatal("should parse the last timestamp")
	}
}

func TestReadTranscriptNoFinalReportWhenMidTool(t *testing.T) {
	now := time.Now()
	path := writeTranscript(t,
		assistantText(t, tsAt(now, 10), "Starting work.", "end_turn"),
		userToolResult(t, tsAt(now, 9), "ok", false),
		assistantToolUse(t, tsAt(now, 1), "Bash", map[string]any{"command": "go test ./..."}),
	)
	sig := ReadTranscript(path)
	if sig.FinalReport {
		t.Fatal("a transcript whose last assistant turn is a tool_use is mid-work, not a final report")
	}
}

func TestReadTranscriptChangedFilesAndWitness(t *testing.T) {
	now := time.Now()
	path := writeTranscript(t,
		assistantToolUse(t, tsAt(now, 8), "Edit", map[string]any{"file_path": "internal/x/x.go"}),
		assistantToolUse(t, tsAt(now, 7), "Write", map[string]any{"file_path": "internal/x/x_test.go"}),
		assistantToolUse(t, tsAt(now, 5), "Bash", map[string]any{"command": "go test ./internal/x/"}),
		assistantText(t, tsAt(now, 1), "Fixed and tested.", "end_turn"),
	)
	sig := ReadTranscript(path)
	if len(sig.ChangedFiles) != 2 {
		t.Fatalf("want 2 changed files, got %v", sig.ChangedFiles)
	}
	if len(sig.WitnessCommands) == 0 {
		t.Fatalf("go test should be captured as a witness command, got %v", sig.WitnessCommands)
	}
}

func TestReadTranscriptBlockerDetection(t *testing.T) {
	now := time.Now()
	cases := []struct {
		name    string
		content string
		kind    string
	}{
		{"auth", "API Error: authentication_error — please run /login", "auth"},
		{"rate", "API Error: 429 rate_limit_error, please try again later", "rate"},
		{"credit", "Your credit balance is too low to run this request", "credit"},
		{"access", "Your organization has disabled Claude subscription access", "access"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeTranscript(t,
				assistantToolUse(t, tsAt(now, 2), "Bash", map[string]any{"command": "echo hi"}),
				userToolResult(t, tsAt(now, 1), tc.content, true),
			)
			sig := ReadTranscript(path)
			if sig.BlockerKind != tc.kind {
				t.Fatalf("want blocker kind %q, got %q (blocker=%q)", tc.kind, sig.BlockerKind, sig.Blocker)
			}
		})
	}
}

func TestReadTranscriptResolvedEarlyErrorNotACurrentBlocker(t *testing.T) {
	now := time.Now()
	// An early rate-limit error, then 50 healthy turns, then a final report. The
	// bounded tail scan must not read the long-since-resolved error as current.
	lines := []string{userToolResult(t, tsAt(now, 120), "429 rate_limit_error", true)}
	for i := 0; i < 50; i++ {
		lines = append(lines, assistantToolUse(t, tsAt(now, float64(60-i)), "Read", map[string]any{"file_path": "a.go"}))
	}
	lines = append(lines, assistantText(t, tsAt(now, 1), "All good.", "end_turn"))
	sig := ReadTranscript(writeTranscript(t, lines...))
	if sig.Blocker != "" {
		t.Fatalf("a resolved early error must not read as a current blocker, got %q", sig.Blocker)
	}
	if !sig.FinalReport {
		t.Fatal("should still see the final report")
	}
}

func TestReadTranscriptLineCount(t *testing.T) {
	now := time.Now()
	sig := ReadTranscript(writeTranscript(t,
		assistantToolUse(t, tsAt(now, 3), "Read", map[string]any{"file_path": "a"}),
		userToolResult(t, tsAt(now, 2), "x", false),
		assistantText(t, tsAt(now, 1), "done", "end_turn"),
	))
	if sig.Lines != 3 {
		t.Fatalf("want 3 lines, got %d", sig.Lines)
	}
}
