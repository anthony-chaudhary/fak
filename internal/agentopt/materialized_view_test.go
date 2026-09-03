package agentopt

import (
	"strings"
	"testing"
	"time"
)

func TestMaterializedViewGenerator(t *testing.T) {
	gen := NewFoldedStateGenerator()

	// 1. Verify initial clean state.
	initialState := gen.GetFoldedState()
	if len(initialState.Files) != 0 {
		t.Fatalf("expected 0 files initially, got %d", len(initialState.Files))
	}
	if len(initialState.Tests) != 0 {
		t.Fatalf("expected 0 tests initially, got %d", len(initialState.Tests))
	}
	if len(initialState.Variables) != 0 {
		t.Fatalf("expected 0 variables initially, got %d", len(initialState.Variables))
	}
	if gen.EventCount() != 0 {
		t.Fatalf("expected 0 events initially, got %d", gen.EventCount())
	}

	// 2. File write mutation: create initial file.
	filePath := "cmd/fak/main.go"
	initialContent := "package main\n\nfunc main() {\n\tprintln(\"hello\")\n}\n"
	gen.ApplyEvent(NewFileWriteEvent(filePath, initialContent))

	state1 := gen.GetFoldedState()
	file1, ok := state1.File(filePath)
	if !ok {
		t.Fatalf("file %q not found in materialized state", filePath)
	}
	if file1.Version != 1 {
		t.Errorf("expected file version 1, got %d", file1.Version)
	}
	if file1.Content != initialContent {
		t.Errorf("content mismatch:\nexpected: %q\ngot: %q", initialContent, file1.Content)
	}
	if file1.LineCount != 5 {
		t.Errorf("expected 5 lines, got %d", file1.LineCount)
	}
	if file1.ByteSize != len(initialContent) {
		t.Errorf("expected byte size %d, got %d", len(initialContent), file1.ByteSize)
	}

	// 3. File edit mutation: replace string in existing file.
	gen.ApplyEvent(NewFileEditEvent(filePath, "\"hello\"", "\"world\"", false))

	state2 := gen.GetFoldedState()
	file2, ok := state2.File(filePath)
	if !ok {
		t.Fatalf("file %q missing after edit", filePath)
	}
	if file2.Version != 2 {
		t.Errorf("expected file version 2 after edit, got %d", file2.Version)
	}
	expectedEdited := "package main\n\nfunc main() {\n\tprintln(\"world\")\n}\n"
	if file2.Content != expectedEdited {
		t.Errorf("edited content mismatch:\nexpected: %q\ngot: %q", expectedEdited, file2.Content)
	}

	// 4. File write and delete mutation.
	scratchPath := "internal/scratch.txt"
	gen.ApplyEvent(NewFileWriteEvent(scratchPath, "temporary scratch buffer"))
	state3 := gen.GetFoldedState()
	if _, ok := state3.File(scratchPath); !ok {
		t.Fatalf("scratch file %q was not created", scratchPath)
	}
	if len(state3.ActiveFiles()) != 2 {
		t.Errorf("expected 2 active files, got %d", len(state3.ActiveFiles()))
	}

	gen.ApplyEvent(NewFileDeleteEvent(scratchPath))
	state4 := gen.GetFoldedState()
	if _, ok := state4.File(scratchPath); ok {
		t.Fatalf("scratch file %q should be absent from active files after deletion", scratchPath)
	}
	if _, deleted := state4.Deletions[scratchPath]; !deleted {
		t.Fatalf("scratch file %q should be recorded in deletions", scratchPath)
	}
	if len(state4.ActiveFiles()) != 1 {
		t.Errorf("expected 1 active file after deletion, got %d", len(state4.ActiveFiles()))
	}

	// 5. Test outcome mutations: failing test, followed by passing test run.
	testName := "TestMainExecution"
	gen.ApplyEvent(NewTestOutcomeEvent(testName, TestStatusFailed, "FAIL: unexpected token", 15*time.Millisecond))

	state5 := gen.GetFoldedState()
	test1, ok := state5.Test(testName)
	if !ok {
		t.Fatalf("test %q not found in state", testName)
	}
	if test1.Status != TestStatusFailed {
		t.Errorf("expected test status failed, got %q", test1.Status)
	}
	if test1.RunCount != 1 {
		t.Errorf("expected run count 1, got %d", test1.RunCount)
	}
	if state5.AllTestsPassed() {
		t.Errorf("expected AllTestsPassed() to be false when a test has failed")
	}

	// Re-run test and succeed.
	gen.ApplyEvent(NewTestOutcomeEvent(testName, TestStatusPassed, "PASS", 12*time.Millisecond))
	state6 := gen.GetFoldedState()
	test2, ok := state6.Test(testName)
	if !ok {
		t.Fatalf("test %q missing after re-run", testName)
	}
	if test2.Status != TestStatusPassed {
		t.Errorf("expected test status passed after re-run, got %q", test2.Status)
	}
	if test2.RunCount != 2 {
		t.Errorf("expected run count 2 after second run, got %d", test2.RunCount)
	}
	if !state6.AllTestsPassed() {
		t.Errorf("expected AllTestsPassed() to be true after successful re-run")
	}

	// Add another test.
	gen.ApplyEvent(NewTestOutcomeEvent("TestIntegration", TestStatusPassed, "PASS", 45*time.Millisecond))
	state7 := gen.GetFoldedState()
	passed, failed, skipped, total := state7.TestSummary()
	if passed != 2 || failed != 0 || skipped != 0 || total != 2 {
		t.Errorf("unexpected test summary: passed=%d failed=%d skipped=%d total=%d", passed, failed, skipped, total)
	}

	// 6. Working state variables mutation: set and delete.
	gen.ApplyEvent(NewVariableSetEvent("current_task", "ISSUE-11020"))
	gen.ApplyEvent(NewVariableSetEvent("iteration", 2))
	gen.ApplyEvent(NewVariableSetEvent("staged", true))

	state8 := gen.GetFoldedState()
	if task, ok := state8.Variable("current_task"); !ok || task != "ISSUE-11020" {
		t.Errorf("variable current_task mismatch: %v", task)
	}
	if iter, ok := state8.Variable("iteration"); !ok || iter != 2 {
		t.Errorf("variable iteration mismatch: %v", iter)
	}

	gen.ApplyEvent(NewVariableDeleteEvent("staged"))
	state9 := gen.GetFoldedState()
	if _, ok := state9.Variable("staged"); ok {
		t.Errorf("variable 'staged' should have been deleted")
	}

	// 7. Audit log verification: all events preserved in order.
	auditLog := gen.AuditLog()
	expectedEventCount := 11 // write, edit, write, delete, test fail, test pass, test2 pass, var set, var set, var set, var del = 11 events
	if gen.EventCount() != len(auditLog) {
		t.Errorf("event count %d does not match audit log length %d", gen.EventCount(), len(auditLog))
	}
	if len(auditLog) != expectedEventCount {
		t.Errorf("expected %d total events in audit log, got %d", expectedEventCount, len(auditLog))
	}

	// Audit log sequence validation.
	expectedSequence := []MutationType{
		MutationFileWrite,
		MutationFileEdit,
		MutationFileWrite,
		MutationFileDelete,
		MutationTestOutcome,
		MutationTestOutcome,
		MutationTestOutcome,
		MutationVariableSet,
		MutationVariableSet,
		MutationVariableSet,
		MutationVariableDelete,
	}
	for i, expectedType := range expectedSequence {
		if NormalizeMutationType(string(auditLog[i].Type)) != expectedType {
			t.Errorf("event %d type mismatch: expected %s, got %s", i, expectedType, auditLog[i].Type)
		}
	}

	// 8. FoldedView presentation verification.
	folded := gen.FoldedView()
	if len(folded.Files) != 1 || folded.Files[0] != filePath {
		t.Errorf("folded view files mismatch: %v", folded.Files)
	}
	if _, exists := folded.Variables["current_task"]; !exists {
		t.Errorf("folded view variables missing current_task: %v", folded.Variables)
	}
	if folded.Tests[testName] != string(TestStatusPassed) {
		t.Errorf("folded view test status mismatch for %s: %s", testName, folded.Tests[testName])
	}
	if !strings.Contains(folded.Text, "=== Current Working State ===") {
		t.Errorf("folded view text missing header banner:\n%s", folded.Text)
	}
	if !strings.Contains(folded.Text, filePath) {
		t.Errorf("folded view text missing active file %s", filePath)
	}
	if strings.Contains(folded.Text, scratchPath) && !strings.Contains(folded.Text, "Deleted Files") {
		t.Errorf("scratch path should only appear under Deleted Files section")
	}
	if !strings.Contains(folded.Summary, "1 files active") {
		t.Errorf("summary mismatch: %s", folded.Summary)
	}
}

func TestStateFolderInterface(t *testing.T) {
	var folder StateFolder = NewFoldedStateGenerator()
	folder.ApplyEvent(NewVariableSetEvent("key", "val"))

	state := folder.GetFoldedState()
	val, ok := state.Variable("key")
	if !ok || val != "val" {
		t.Fatalf("expected key=val, got %v (ok=%v)", val, ok)
	}

	fv := folder.FoldedView()
	if !strings.Contains(fv.Text, "key: val") {
		t.Fatalf("folded view text missing variable: %s", fv.Text)
	}
}

func TestFoldEventsBatch(t *testing.T) {
	events := []Event{
		NewFileWriteEvent("pkg/a.go", "package pkg\n"),
		NewFileWriteEvent("pkg/b.go", "package pkg\n"),
		NewVariableSetEvent("env", "test"),
		NewTestOutcomeEvent("TestA", TestStatusPassed, "ok", 5*time.Millisecond),
	}

	gen := NewFoldedStateGeneratorWithEvents(events)
	state := gen.GetFoldedState()

	if len(state.ActiveFiles()) != 2 {
		t.Fatalf("expected 2 active files, got %d", len(state.ActiveFiles()))
	}
	if v, ok := state.Variable("env"); !ok || v != "test" {
		t.Fatalf("expected env=test, got %v", v)
	}
	if !state.AllTestsPassed() {
		t.Fatalf("expected all tests passed")
	}
	if gen.EventCount() != 4 {
		t.Fatalf("expected 4 events, got %d", gen.EventCount())
	}
}

func TestFileEditReplaceAll(t *testing.T) {
	gen := NewFoldedStateGenerator()
	gen.ApplyEvent(NewFileWriteEvent("foo.txt", "alpha beta alpha gamma alpha"))

	// Replace all "alpha" with "delta".
	gen.ApplyEvent(NewFileEditEvent("foo.txt", "alpha", "delta", true))

	state := gen.GetFoldedState()
	file, ok := state.File("foo.txt")
	if !ok {
		t.Fatalf("file not found")
	}
	expected := "delta beta delta gamma delta"
	if file.Content != expected {
		t.Fatalf("expected %q, got %q", expected, file.Content)
	}
}

func TestFoldedStateIsolation(t *testing.T) {
	gen := NewFoldedStateGenerator()
	gen.ApplyEvent(NewFileWriteEvent("test.txt", "initial"))
	gen.ApplyEvent(NewVariableSetEvent("count", 10))

	state := gen.GetFoldedState()
	// Mutate the returned maps.
	state.Files["injected.txt"] = FileState{Path: "injected.txt"}
	state.Variables["count"] = 999

	freshState := gen.GetFoldedState()
	if _, exists := freshState.File("injected.txt"); exists {
		t.Fatalf("internal state was modified through leaked map reference")
	}
	if v, _ := freshState.Variable("count"); v != 10 {
		t.Fatalf("internal variable was modified through leaked map reference: %v", v)
	}
}

func TestFoldedStateJSON(t *testing.T) {
	gen := NewFoldedStateGenerator()
	gen.ApplyEvent(NewFileWriteEvent("test.txt", "content"))
	gen.ApplyEvent(NewTestOutcomeEvent("TestJSON", TestStatusPassed, "ok", 10*time.Millisecond))
	gen.ApplyEvent(NewVariableSetEvent("flag", true))

	state := gen.GetFoldedState()
	data, err := state.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON failed: %v", err)
	}

	decoded, err := FoldedStateFromJSON(data)
	if err != nil {
		t.Fatalf("FoldedStateFromJSON failed: %v", err)
	}

	if len(decoded.Files) != len(state.Files) {
		t.Errorf("decoded files count mismatch: %d vs %d", len(decoded.Files), len(state.Files))
	}
	if len(decoded.Tests) != len(state.Tests) {
		t.Errorf("decoded tests count mismatch: %d vs %d", len(decoded.Tests), len(state.Tests))
	}
	if len(decoded.Variables) != len(state.Variables) {
		t.Errorf("decoded variables count mismatch: %d vs %d", len(decoded.Variables), len(state.Variables))
	}
}

func TestEventMetadataResolution(t *testing.T) {
	gen := NewFoldedStateGenerator()
	gen.ApplyEvent(Event{
		Type: MutationFileWrite,
		Metadata: map[string]any{
			"path":    "meta/file.go",
			"content": "package meta",
		},
	})
	gen.ApplyEvent(Event{
		Type: MutationVariableSet,
		Metadata: map[string]any{
			"key":   "meta_var",
			"value": "resolved",
		},
	})

	state := gen.GetFoldedState()
	if _, ok := state.File("meta/file.go"); !ok {
		t.Errorf("failed to resolve path/content from metadata")
	}
	if v, ok := state.Variable("meta_var"); !ok || v != "resolved" {
		t.Errorf("failed to resolve key/value from metadata: %v", v)
	}
}

func TestReset(t *testing.T) {
	gen := NewFoldedStateGenerator()
	gen.ApplyEvent(NewFileWriteEvent("file.txt", "data"))
	gen.ApplyEvent(NewVariableSetEvent("k", "v"))

	if gen.EventCount() != 2 {
		t.Fatalf("expected 2 events before reset")
	}

	gen.Reset()

	if gen.EventCount() != 0 {
		t.Fatalf("expected 0 events after reset, got %d", gen.EventCount())
	}
	state := gen.GetFoldedState()
	if len(state.Files) != 0 || len(state.Variables) != 0 {
		t.Fatalf("state not cleared after reset: files=%d, vars=%d", len(state.Files), len(state.Variables))
	}
}
