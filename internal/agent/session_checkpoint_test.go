package agent

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestSaveAndLoadSessionCheckpoint(t *testing.T) {
	dir := t.TempDir()
	now := time.Now().UTC().Truncate(time.Second)

	cp := SessionCheckpoint{
		SessionID: "sess-test-123",
		CWD:       dir,
		Task:      "book flight",
		Model:     "test-model",
		Provider:  "test-provider",
		BaseURL:   "http://localhost:8080/v1",
		Messages: []Message{
			{Role: RoleSystem, Content: "system prompt"},
			{Role: RoleUser, Content: "book flight"},
			{Role: RoleAssistant, Content: "ok, booking"},
		},
		Turn:      1,
		CreatedAt: now,
		UpdatedAt: now,
		Status:    "active",
	}

	if err := SaveSessionCheckpoint(dir, cp); err != nil {
		t.Fatalf("SaveSessionCheckpoint failed: %v", err)
	}

	// 1. Load by ID with directory
	loaded, err := LoadSessionCheckpoint("sess-test-123", dir)
	if err != nil {
		t.Fatalf("LoadSessionCheckpoint by ID failed: %v", err)
	}
	if loaded.SessionID != cp.SessionID {
		t.Errorf("got SessionID %q, want %q", loaded.SessionID, cp.SessionID)
	}
	if loaded.CWD != cp.CWD {
		t.Errorf("got CWD %q, want %q", loaded.CWD, cp.CWD)
	}
	if loaded.Task != cp.Task {
		t.Errorf("got Task %q, want %q", loaded.Task, cp.Task)
	}
	if loaded.Model != cp.Model {
		t.Errorf("got Model %q, want %q", loaded.Model, cp.Model)
	}
	if loaded.Provider != cp.Provider {
		t.Errorf("got Provider %q, want %q", loaded.Provider, cp.Provider)
	}
	if loaded.BaseURL != cp.BaseURL {
		t.Errorf("got BaseURL %q, want %q", loaded.BaseURL, cp.BaseURL)
	}
	if len(loaded.Messages) != len(cp.Messages) {
		t.Errorf("got %d messages, want %d", len(loaded.Messages), len(cp.Messages))
	}
	if loaded.Turn != cp.Turn {
		t.Errorf("got Turn %d, want %d", loaded.Turn, cp.Turn)
	}
	if loaded.Status != cp.Status {
		t.Errorf("got Status %q, want %q", loaded.Status, cp.Status)
	}

	// 2. Load by ID with .json suffix
	loadedJson, err := LoadSessionCheckpoint("sess-test-123.json", dir)
	if err != nil {
		t.Fatalf("LoadSessionCheckpoint with .json failed: %v", err)
	}
	if loadedJson.SessionID != cp.SessionID {
		t.Errorf("got SessionID %q, want %q", loadedJson.SessionID, cp.SessionID)
	}

	// 3. Load by direct path
	directPath := filepath.Join(dir, "sess-test-123.json")
	loadedPath, err := LoadSessionCheckpoint(directPath, "")
	if err != nil {
		t.Fatalf("LoadSessionCheckpoint by path failed: %v", err)
	}
	if loadedPath.SessionID != cp.SessionID {
		t.Errorf("got SessionID %q, want %q", loadedPath.SessionID, cp.SessionID)
	}

	// 4. Load nonexistent fails closed
	if _, err := LoadSessionCheckpoint("nonexistent-sess", dir); err == nil {
		t.Fatal("expected error for nonexistent session checkpoint, got nil")
	}
}

func TestSessionCheckpointCrashRestartContinuation(t *testing.T) {
	dir := t.TempDir()
	sessionID := "crash-restart-session"

	ctx := context.Background()
	planner := NewMockPlanner("mock-deterministic")

	// Phase 1: Run 1 turn only (simulating an interruption/crash after turn 1)
	m1, err := RunArm(ctx, planner, DefaultTask, true, 1, nil,
		WithSessionCheckpoint(sessionID, dir),
	)
	if err != nil {
		t.Fatalf("Phase 1 RunArm failed: %v", err)
	}
	if m1.Turns != 1 {
		t.Fatalf("Phase 1 expected 1 turn, got %d", m1.Turns)
	}
	if m1.TaskCompleted {
		t.Fatal("Phase 1 should not have completed the full task in 1 turn")
	}

	// Verify checkpoint written on disk after turn 1
	cp1, err := LoadSessionCheckpoint(sessionID, dir)
	if err != nil {
		t.Fatalf("failed to load phase 1 checkpoint: %v", err)
	}
	if cp1.SessionID != sessionID {
		t.Fatalf("expected session ID %q, got %q", sessionID, cp1.SessionID)
	}
	if cp1.Turn != 1 {
		t.Fatalf("expected checkpoint turn 1, got %d", cp1.Turn)
	}
	if cp1.Status != "active" {
		t.Fatalf("expected checkpoint status active, got %q", cp1.Status)
	}
	if len(cp1.Messages) < 3 {
		t.Fatalf("expected messages to contain initial prompt and turn 1 messages, got %d", len(cp1.Messages))
	}
	initialCreatedAt := cp1.CreatedAt

	// Phase 2: Resume session continuation from the saved checkpoint
	planner2 := NewMockPlanner("mock-deterministic")
	m2, err := RunArm(ctx, planner2, cp1.Task, true, 10, nil,
		WithConversation(cp1.Messages),
		WithSessionCheckpointState(cp1, dir),
	)
	if err != nil {
		t.Fatalf("Phase 2 RunArm resume failed: %v", err)
	}
	if !m2.TaskCompleted {
		t.Fatalf("Phase 2 resumed run should have completed task, got: %+v", m2)
	}

	// Verify final checkpoint after resumed run
	cp2, err := LoadSessionCheckpoint(sessionID, dir)
	if err != nil {
		t.Fatalf("failed to load phase 2 checkpoint: %v", err)
	}
	if cp2.SessionID != sessionID {
		t.Fatalf("expected session ID %q, got %q", sessionID, cp2.SessionID)
	}
	if cp2.Turn <= cp1.Turn {
		t.Fatalf("resumed turn count %d should be greater than initial %d", cp2.Turn, cp1.Turn)
	}
	if cp2.Status != "completed" {
		t.Fatalf("expected checkpoint status completed, got %q", cp2.Status)
	}
	if !cp2.CreatedAt.Equal(initialCreatedAt) {
		t.Fatalf("resumed checkpoint should preserve initial CreatedAt %v, got %v", initialCreatedAt, cp2.CreatedAt)
	}
	if len(cp2.Messages) <= len(cp1.Messages) {
		t.Fatalf("resumed checkpoint messages %d should exceed initial %d", len(cp2.Messages), len(cp1.Messages))
	}
}
