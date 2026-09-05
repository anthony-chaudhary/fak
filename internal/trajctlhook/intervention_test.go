package trajctlhook

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func TestInterventionActions_TypesAndParse(t *testing.T) {
	// Verify closed action vocabulary constants
	if ActionSteer != "STEER" {
		t.Errorf("expected ActionSteer to be 'STEER', got %q", ActionSteer)
	}
	if ActionPause != "PAUSE" {
		t.Errorf("expected ActionPause to be 'PAUSE', got %q", ActionPause)
	}
	if ActionRollback != "ROLLBACK" {
		t.Errorf("expected ActionRollback to be 'ROLLBACK', got %q", ActionRollback)
	}
	if ActionRetry != "RETRY" {
		t.Errorf("expected ActionRetry to be 'RETRY', got %q", ActionRetry)
	}

	// Verify validity
	validActions := []Action{ActionSteer, ActionPause, ActionRollback, ActionRetry}
	for _, a := range validActions {
		if !a.IsValid() {
			t.Errorf("expected %s to be valid", a)
		}
		if a.String() != string(a) {
			t.Errorf("String() mismatch: %s != %s", a.String(), string(a))
		}
	}

	if Action("INVALID").IsValid() {
		t.Errorf("expected INVALID to be invalid")
	}
	if Action("").IsValid() {
		t.Errorf("expected empty action to be invalid")
	}

	// Verify ParseAction
	testCases := []struct {
		input   string
		want    Action
		wantErr bool
	}{
		{"STEER", ActionSteer, false},
		{"steer", ActionSteer, false},
		{"ActionSteer", ActionSteer, false},
		{"PAUSE", ActionPause, false},
		{"pause", ActionPause, false},
		{"ROLLBACK", ActionRollback, false},
		{"rollback", ActionRollback, false},
		{"RETRY", ActionRetry, false},
		{"retry", ActionRetry, false},
		{"unknown", "", true},
	}

	for _, tc := range testCases {
		got, err := ParseAction(tc.input)
		if (err != nil) != tc.wantErr {
			t.Errorf("ParseAction(%q) error = %v, wantErr = %v", tc.input, err, tc.wantErr)
		}
		if got != tc.want {
			t.Errorf("ParseAction(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestInterventionExecutor_DispatchSteer(t *testing.T) {
	deliveredText := ""
	mockSteer := func(sessionID, text string) error {
		if sessionID != "sess-123" {
			t.Errorf("unexpected sessionID: %s", sessionID)
		}
		deliveredText = text
		return nil
	}

	state := &trajctl.State{
		Objectives: make(map[string]trajctl.Objective),
	}

	exec := &InterventionExecutor{
		SessionID:    "sess-123",
		ObjectiveID:  "obj-1",
		RunID:        "run-99",
		State:        state,
		SteerDeliver: mockSteer,
	}

	c := StepClassification{
		Action:   ActionSteer,
		Reason:   "goal drift detected",
		Guidance: "re-read requirements and focus on issue #11410",
	}

	res, err := exec.Execute(context.Background(), c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res.Action != ActionSteer || !res.Executed {
		t.Errorf("unexpected result: %+v", res)
	}
	if res.Guidance != c.Guidance {
		t.Errorf("guidance mismatch: got %q, want %q", res.Guidance, c.Guidance)
	}
	if deliveredText != c.Guidance {
		t.Errorf("delivered guidance mismatch: got %q, want %q", deliveredText, c.Guidance)
	}
	if res.State != "steered" {
		t.Errorf("expected state 'steered', got %q", res.State)
	}

	if len(state.Steers) != 1 {
		t.Fatalf("expected 1 steer decision in state, got %d", len(state.Steers))
	}
	d := state.Steers[0]
	if d.ObjectiveID != "obj-1" || d.Action != trajctl.ActionNudge || !d.Delivered {
		t.Errorf("unexpected steer decision: %+v", d)
	}
}

func TestInterventionExecutor_DispatchPause(t *testing.T) {
	state := &trajctl.State{
		Objectives: map[string]trajctl.Objective{
			"obj-1": {
				ID:     "obj-1",
				Status: trajctl.StatusActive,
			},
		},
	}

	exec := &InterventionExecutor{
		ObjectiveID: "obj-1",
		State:       state,
	}

	c := StepClassification{
		Action: ActionPause,
		Reason: "potential infinite loop in worker recursion",
	}

	res, err := exec.Execute(context.Background(), c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res.Action != ActionPause || !res.Executed {
		t.Errorf("unexpected result: %+v", res)
	}
	if res.State != string(trajctl.StatusPaused) {
		t.Errorf("expected state 'paused', got %q", res.State)
	}

	obj := state.Objectives["obj-1"]
	if obj.Status != trajctl.StatusPaused {
		t.Errorf("expected objective to transition to StatusPaused, got %s", obj.Status)
	}
}

func TestInterventionExecutor_DispatchRollback(t *testing.T) {
	tempDir := t.TempDir()
	scratchDir := filepath.Join(tempDir, "_scratch")
	if err := os.MkdirAll(scratchDir, 0o755); err != nil {
		t.Fatalf("failed to create scratch dir: %v", err)
	}

	f1 := filepath.Join(scratchDir, "uncommitted_state.tmp")
	f2 := filepath.Join(scratchDir, "step.json")
	_ = os.WriteFile(f1, []byte("provisional"), 0o644)
	_ = os.WriteFile(f2, []byte("{}"), 0o644)

	exec := NewInterventionExecutor(tempDir, "_scratch")
	c := StepClassification{
		Action: ActionRollback,
		Reason: "step produced corrupted provisional artifacts",
	}

	res, err := exec.Execute(context.Background(), c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res.Action != ActionRollback || !res.Executed {
		t.Errorf("unexpected result: %+v", res)
	}
	if len(res.Reverted) < 2 {
		t.Errorf("expected at least 2 reverted files, got %d (%v)", len(res.Reverted), res.Reverted)
	}

	if _, err := os.Stat(f1); !os.IsNotExist(err) {
		t.Errorf("expected f1 to be deleted")
	}
	if _, err := os.Stat(f2); !os.IsNotExist(err) {
		t.Errorf("expected f2 to be deleted")
	}
}

func TestInterventionExecutor_DispatchRetry(t *testing.T) {
	deliveredText := ""
	mockSteer := func(_, text string) error {
		deliveredText = text
		return nil
	}

	state := &trajctl.State{
		Objectives: make(map[string]trajctl.Objective),
	}

	exec := &InterventionExecutor{
		SessionID:    "sess-retry",
		ObjectiveID:  "obj-retry",
		State:        state,
		SteerDeliver: mockSteer,
	}

	c := StepClassification{
		Action:              ActionRetry,
		Reason:              "wrong dialect for tool",
		Guidance:            "re-run using bash syntax instead of powershell",
		NegativeConstraints: []string{"do not use Get-ChildItem", "do not use alias 'ls'"},
	}

	res, err := exec.Execute(context.Background(), c)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res.Action != ActionRetry || !res.Executed {
		t.Errorf("unexpected result: %+v", res)
	}
	if res.State != "retrying" {
		t.Errorf("expected state 'retrying', got %q", res.State)
	}
	if len(res.Constraints) != 2 {
		t.Errorf("expected 2 negative constraints, got %d", len(res.Constraints))
	}
	if !strings.Contains(res.Details, "Negative constraints (DO NOT):") {
		t.Errorf("expected retry details to format negative constraints, got: %s", res.Details)
	}
	if !strings.Contains(deliveredText, "re-run using bash syntax") {
		t.Errorf("expected delivered text to contain guidance, got: %s", deliveredText)
	}
}

func TestRollback_SafetyOnScratchFiles_NonGit(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Committed progress files (outside scratch)
	committed1 := filepath.Join(tempDir, "main.go")
	committed2 := filepath.Join(tempDir, "README.md")
	_ = os.WriteFile(committed1, []byte("package main\n"), 0o644)
	_ = os.WriteFile(committed2, []byte("# Documentation\n"), 0o644)

	// 2. Provisional scratch files
	scratchDir := filepath.Join(tempDir, "_scratch")
	_ = os.MkdirAll(scratchDir, 0o755)
	scratch1 := filepath.Join(scratchDir, "temp_data.json")
	scratch2 := filepath.Join(scratchDir, "step.tmp")
	_ = os.WriteFile(scratch1, []byte(`{"scratch": true}`), 0o644)
	_ = os.WriteFile(scratch2, []byte("provisional bytes"), 0o644)

	// 3. Provisional file in root
	rootProvisional := filepath.Join(tempDir, "scratch_task.provisional")
	_ = os.WriteFile(rootProvisional, []byte("temp"), 0o644)

	reverted, err := RollbackProvisionalScratch(context.Background(), tempDir, "_scratch")
	if err != nil {
		t.Fatalf("rollback error: %v", err)
	}

	if len(reverted) != 3 {
		t.Errorf("expected 3 reverted files, got %d: %v", len(reverted), reverted)
	}

	// Assert committed progress is preserved
	if data, err := os.ReadFile(committed1); err != nil || string(data) != "package main\n" {
		t.Errorf("committed1 was altered or removed: err=%v", err)
	}
	if data, err := os.ReadFile(committed2); err != nil || string(data) != "# Documentation\n" {
		t.Errorf("committed2 was altered or removed: err=%v", err)
	}

	// Assert provisional scratch files were deleted
	if _, err := os.Stat(scratch1); !os.IsNotExist(err) {
		t.Errorf("scratch1 should have been deleted")
	}
	if _, err := os.Stat(scratch2); !os.IsNotExist(err) {
		t.Errorf("scratch2 should have been deleted")
	}
	if _, err := os.Stat(rootProvisional); !os.IsNotExist(err) {
		t.Errorf("rootProvisional should have been deleted")
	}
}

func TestRollback_SafetyOnScratchFiles_GitRepo(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available in environment")
	}

	tempDir := t.TempDir()

	// Initialize git repo
	runGit := func(args ...string) {
		cmd := exec.Command("git", append([]string{"-C", tempDir}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v failed: %v\nOutput: %s", args, err, out)
		}
	}

	runGit("init")
	runGit("config", "user.name", "Test User")
	runGit("config", "user.email", "test@example.com")

	// 1. Create and commit progress file
	committedFile := filepath.Join(tempDir, "committed.go")
	_ = os.WriteFile(committedFile, []byte("package test\n"), 0o644)

	// Create and commit a file inside scratch (e.g. baseline fixture)
	scratchDir := filepath.Join(tempDir, "_scratch")
	_ = os.MkdirAll(scratchDir, 0o755)
	committedScratch := filepath.Join(scratchDir, "baseline_spec.json")
	_ = os.WriteFile(committedScratch, []byte(`{"version": 1}`), 0o644)

	runGit("add", "committed.go", "_scratch/baseline_spec.json")
	runGit("commit", "-m", "initial commit")

	// 2. Now simulate uncommitted provisional scratch work:
	// a) Untracked scratch file
	uncommittedScratch := filepath.Join(scratchDir, "uncommitted_step.tmp")
	_ = os.WriteFile(uncommittedScratch, []byte("uncommitted"), 0o644)

	// b) Uncommitted modification to committedScratch
	_ = os.WriteFile(committedScratch, []byte(`{"version": 2, "corrupted": true}`), 0o644)

	// 3. Perform rollback
	reverted, err := RollbackProvisionalScratch(context.Background(), tempDir, "_scratch")
	if err != nil {
		t.Fatalf("rollback error: %v", err)
	}

	if len(reverted) == 0 {
		t.Errorf("expected files to be reverted, got 0")
	}

	// 4. Verify safety:
	// committed.go MUST be completely untouched
	if content, err := os.ReadFile(committedFile); err != nil || string(content) != "package test\n" {
		t.Errorf("committed.go modified or missing: %v", err)
	}

	// committedScratch MUST still exist, and must be restored to committed state {"version": 1}
	content, err := os.ReadFile(committedScratch)
	if err != nil {
		t.Fatalf("committedScratch was deleted: %v", err)
	}
	if string(content) != `{"version": 1}` {
		t.Errorf("committedScratch was not restored to committed state, got: %s", string(content))
	}

	// uncommittedScratch MUST be completely removed
	if _, err := os.Stat(uncommittedScratch); !os.IsNotExist(err) {
		t.Errorf("uncommittedScratch was not removed")
	}
}

func TestInterventionExecutor_LedgerPersistence(t *testing.T) {
	tempDir := t.TempDir()
	ledgerPath := filepath.Join(tempDir, "trajctl.jsonl")

	// Declare an active objective first
	obj := trajctl.Objective{
		ID:        "obj-persisted",
		Statement: "test objective",
		Status:    trajctl.StatusActive,
	}
	if err := trajctl.Append(ledgerPath, trajctl.ObjectiveRecord(obj)); err != nil {
		t.Fatalf("failed to seed ledger: %v", err)
	}

	state := &trajctl.State{
		Objectives: map[string]trajctl.Objective{
			"obj-persisted": obj,
		},
	}

	exec := &InterventionExecutor{
		ObjectiveID: "obj-persisted",
		LedgerPath:  ledgerPath,
		State:       state,
	}

	// 1. Execute Pause
	cPause := StepClassification{
		Action: ActionPause,
		Reason: "suspending due to classifier verdict",
	}
	if _, err := exec.Execute(context.Background(), cPause); err != nil {
		t.Fatalf("Execute(ActionPause) failed: %v", err)
	}

	// Re-read ledger from disk and fold
	rows := trajctl.ReadLedgerFile(ledgerPath)
	folded := trajctl.Fold(rows)

	if folded.Objectives["obj-persisted"].Status != trajctl.StatusPaused {
		t.Errorf("expected persisted objective to be paused, got %s", folded.Objectives["obj-persisted"].Status)
	}

	// 2. Execute Steer
	cSteer := StepClassification{
		Action:   ActionSteer,
		Reason:   "drift",
		Guidance: "focus on task",
	}
	if _, err := exec.Execute(context.Background(), cSteer); err != nil {
		t.Fatalf("Execute(ActionSteer) failed: %v", err)
	}

	rows = trajctl.ReadLedgerFile(ledgerPath)
	folded = trajctl.Fold(rows)
	if len(folded.Steers) != 1 {
		t.Fatalf("expected 1 persisted steer decision, got %d", len(folded.Steers))
	}
	if folded.Steers[0].Packet != "focus on task" {
		t.Errorf("expected persisted steer packet 'focus on task', got %q", folded.Steers[0].Packet)
	}
}

func TestInterventionExecutor_InvalidAction(t *testing.T) {
	exec := NewInterventionExecutor(".")
	_, err := exec.Execute(context.Background(), StepClassification{Action: "BOGUS"})
	if err == nil {
		t.Errorf("expected error executing invalid action")
	}
}
