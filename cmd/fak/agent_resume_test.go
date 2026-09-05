package main

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestAgentCLISessionAndResume(t *testing.T) {
	dir := t.TempDir()
	sessionID := "cli-session-test"

	outPath1 := filepath.Join(dir, "out1.json")

	// Phase 1: Start new session with 1 turn limit and session checkpointing
	runAgent([]string{
		"--native",
		"--offline",
		"--session", sessionID,
		"--session-dir", dir,
		"--max-turns", "1",
		"--out", outPath1,
		"--code-tools=false",
		"--sys-tools=false",
	})

	// Check checkpoint file was created
	cp1, err := agent.LoadSessionCheckpoint(sessionID, dir)
	if err != nil {
		t.Fatalf("failed to load phase 1 checkpoint: %v", err)
	}
	if cp1.SessionID != sessionID {
		t.Fatalf("expected session ID %q, got %q", sessionID, cp1.SessionID)
	}
	if cp1.Turn != 1 {
		t.Fatalf("expected turn 1, got %d", cp1.Turn)
	}

	outPath2 := filepath.Join(dir, "out2.json")

	// Phase 2: Resume session using --resume <session_id>
	runAgent([]string{
		"--native",
		"--offline",
		"--resume", sessionID,
		"--session-dir", dir,
		"--max-turns", "10",
		"--out", outPath2,
		"--code-tools=false",
		"--sys-tools=false",
	})

	// Check updated checkpoint file
	cp2, err := agent.LoadSessionCheckpoint(sessionID, dir)
	if err != nil {
		t.Fatalf("failed to load phase 2 checkpoint: %v", err)
	}
	if cp2.SessionID != sessionID {
		t.Fatalf("expected session ID %q, got %q", sessionID, cp2.SessionID)
	}
	if cp2.Turn <= cp1.Turn {
		t.Fatalf("expected resumed turn %d > initial turn %d", cp2.Turn, cp1.Turn)
	}
	if cp2.Status != "completed" {
		t.Fatalf("expected checkpoint status completed, got %q", cp2.Status)
	}
}

func TestAgentCLISubcommandResume(t *testing.T) {
	dir := t.TempDir()
	sessionID := "cli-subcmd-test"

	outPath1 := filepath.Join(dir, "out1.json")

	// Phase 1: Create initial checkpoint
	runAgent([]string{
		"--native",
		"--offline",
		"--session", sessionID,
		"--session-dir", dir,
		"--max-turns", "1",
		"--out", outPath1,
		"--code-tools=false",
		"--sys-tools=false",
	})

	cp1, err := agent.LoadSessionCheckpoint(sessionID, dir)
	if err != nil {
		t.Fatalf("failed to load phase 1 checkpoint: %v", err)
	}
	if cp1.Turn != 1 {
		t.Fatalf("expected turn 1, got %d", cp1.Turn)
	}

	outPath2 := filepath.Join(dir, "out2.json")

	// Phase 2: Resume session using subcommand syntax `resume <session_id>`
	runAgent([]string{
		"resume", sessionID,
		"--native",
		"--offline",
		"--session-dir", dir,
		"--max-turns", "10",
		"--out", outPath2,
		"--code-tools=false",
		"--sys-tools=false",
	})

	cp2, err := agent.LoadSessionCheckpoint(sessionID, dir)
	if err != nil {
		t.Fatalf("failed to load phase 2 checkpoint: %v", err)
	}
	if cp2.Turn <= cp1.Turn {
		t.Fatalf("expected resumed turn %d > initial turn %d", cp2.Turn, cp1.Turn)
	}
	if cp2.Status != "completed" {
		t.Fatalf("expected checkpoint status completed, got %q", cp2.Status)
	}
}

func TestAgentCLIResumeDirectPath(t *testing.T) {
	dir := t.TempDir()
	sessionID := "cli-direct-path-test"

	outPath1 := filepath.Join(dir, "out1.json")

	runAgent([]string{
		"--native",
		"--offline",
		"--session", sessionID,
		"--session-dir", dir,
		"--max-turns", "1",
		"--out", outPath1,
		"--code-tools=false",
		"--sys-tools=false",
	})

	directFilePath := filepath.Join(dir, sessionID+".json")
	outPath2 := filepath.Join(dir, "out2.json")

	// Resume using the absolute file path
	runAgent([]string{
		"--native",
		"--offline",
		"--resume", directFilePath,
		"--max-turns", "10",
		"--out", outPath2,
		"--code-tools=false",
		"--sys-tools=false",
	})

	cp, err := agent.LoadSessionCheckpoint(directFilePath, "")
	if err != nil {
		t.Fatalf("failed to load checkpoint from path %s: %v", directFilePath, err)
	}
	if cp.Status != "completed" {
		t.Fatalf("expected status completed, got %q", cp.Status)
	}
}
