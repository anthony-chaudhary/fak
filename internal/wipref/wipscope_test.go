package wipref

import (
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipattr"
)

func TestToolWriteAutoUpdatesWIPScope(t *testing.T) {
	sessionID := "sess-worker-101"
	tracker := NewScopeTracker()

	// 1. A mutating tool execution (write)
	writeArgs := map[string]any{
		"file_path": "internal/wipref/wipscope.go",
		"content":   "package wipref\n// auto-bound content\n",
	}
	paths, admitted := tracker.RecordToolCall(sessionID, "write", writeArgs)
	if !admitted {
		t.Fatalf("expected write tool to be admitted as mutating tool")
	}
	if len(paths) != 1 || paths[0] != "internal/wipref/wipscope.go" {
		t.Fatalf("extracted paths = %v, want [internal/wipref/wipscope.go]", paths)
	}

	// 2. Touched path is auto-bound in session scope
	scope := tracker.ActiveScope(sessionID)
	if len(scope) != 1 || scope[0] != "internal/wipref/wipscope.go" {
		t.Fatalf("active scope = %v, want [internal/wipref/wipscope.go]", scope)
	}

	// 3. Persists in wipref.Stamp
	stamp, err := tracker.RecordToolCompletion(sessionID, "write", writeArgs)
	if err != nil {
		t.Fatalf("RecordToolCompletion: %v", err)
	}
	if len(stamp.Scope) != 1 || stamp.Scope[0] != "internal/wipref/wipscope.go" {
		t.Fatalf("stamp.Scope = %v, want [internal/wipref/wipscope.go]", stamp.Scope)
	}
	if stamp.SessionID != sessionID {
		t.Fatalf("stamp.SessionID = %q, want %q", stamp.SessionID, sessionID)
	}

	// Verify EncodeStamp and DecodeStamp preserve Scope
	encoded, err := EncodeStamp(stamp)
	if err != nil {
		t.Fatalf("EncodeStamp: %v", err)
	}
	decoded, ok := DecodeStamp(encoded)
	if !ok {
		t.Fatalf("DecodeStamp failed to parse encoded stamp")
	}
	if !reflect.DeepEqual(decoded.Scope, stamp.Scope) {
		t.Fatalf("decoded scope = %v, want %v", decoded.Scope, stamp.Scope)
	}

	// 4. Resolves in wipattr as AttrOwned by that session (OWNED_BY_SELF)
	dirtyHunks := []wipattr.Hunk{
		{
			File: "internal/wipref/wipscope.go",
			Edit: []string{"+new edit line"},
		},
	}
	// Checkpoints have no hunk matching this edit signature
	checkpoints := map[string][]wipattr.Hunk{
		sessionID: {},
	}
	attrs := wipattr.Attribute(dirtyHunks, checkpoints, wipattr.WithSessionScope(wipattr.SessionScope{
		Session: sessionID,
		Scope:   stamp.Scope,
		Active:  true,
	}))
	if len(attrs) != 1 {
		t.Fatalf("expected 1 attribution, got %d", len(attrs))
	}
	if attrs[0].State != wipattr.AttrOwned {
		t.Fatalf("attribution state = %q, want %q", attrs[0].State, wipattr.AttrOwned)
	}
	if attrs[0].Owner != sessionID {
		t.Fatalf("attribution owner = %q, want %q", attrs[0].Owner, sessionID)
	}

	// 5. SweepGuard confirms it is SweepSafe ("owned by self — safe to stage")
	verdicts := wipattr.SweepGuard(attrs, sessionID, map[string]bool{sessionID: true})
	if len(verdicts) != 1 {
		t.Fatalf("expected 1 sweep verdict, got %d", len(verdicts))
	}
	if verdicts[0].Risk != wipattr.SweepSafe {
		t.Fatalf("sweep verdict risk = %q, want %q (SweepSafe / OWNED_BY_SELF)", verdicts[0].Risk, wipattr.SweepSafe)
	}

	// 6. Mutating tool execution (edit) also auto-binds
	editArgs := map[string]any{
		"filePath":  "cmd/fak/wip.go",
		"oldString": "foo",
		"newString": "bar",
	}
	editPaths, admitted := tracker.RecordToolCall(sessionID, "edit", editArgs)
	if !admitted || len(editPaths) != 1 || editPaths[0] != "cmd/fak/wip.go" {
		t.Fatalf("edit tool call failed: paths=%v, admitted=%v", editPaths, admitted)
	}
	stamp2, err := tracker.EmitCheckpoint(sessionID)
	if err != nil {
		t.Fatalf("EmitCheckpoint failed: %v", err)
	}
	wantScope := []string{"cmd/fak/wip.go", "internal/wipref/wipscope.go"}
	if !reflect.DeepEqual(stamp2.Scope, wantScope) {
		t.Fatalf("updated stamp.Scope = %v, want %v", stamp2.Scope, wantScope)
	}
}

func TestScopeTrackerDebounce(t *testing.T) {
	sessionID := "sess-debounce"
	emitted := make(chan Stamp, 5)

	tracker := NewScopeTracker(
		WithDebounce(20*time.Millisecond),
		WithEmitter(func(sess string, stamp Stamp) (string, error) {
			emitted <- stamp
			return "commit-sha", nil
		}),
	)

	tracker.RecordToolCompletion(sessionID, "write", map[string]any{"filePath": "a.go"})
	tracker.RecordToolCompletion(sessionID, "write", map[string]any{"filePath": "b.go"})
	tracker.RecordToolCompletion(sessionID, "write", map[string]any{"filePath": "c.go"})

	select {
	case stamp := <-emitted:
		want := []string{"a.go", "b.go", "c.go"}
		if !reflect.DeepEqual(stamp.Scope, want) {
			t.Errorf("debounced stamp.Scope = %v, want %v", stamp.Scope, want)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for debounced checkpoint emission")
	}

	// Check no duplicate rapid emissions
	select {
	case extra := <-emitted:
		t.Fatalf("unexpected duplicate emission: %+v", extra)
	default:
		// good
	}
}

func TestScopeTrackerMicroCheckpoint(t *testing.T) {
	sessionID := "sess-micro"
	emittedCount := 0
	tracker := NewScopeTracker(
		WithDebounce(0),
		WithEmitter(func(sess string, stamp Stamp) (string, error) {
			emittedCount++
			return "commit-sha", nil
		}),
	)

	_, err := tracker.RecordToolCompletion(sessionID, "write", map[string]any{"path": "file1.txt"})
	if err != nil {
		t.Fatalf("RecordToolCompletion: %v", err)
	}
	if emittedCount != 1 {
		t.Fatalf("emittedCount = %d, want 1", emittedCount)
	}

	_, err = tracker.RecordToolCompletion(sessionID, "edit", map[string]any{"file_path": "file2.txt"})
	if err != nil {
		t.Fatalf("RecordToolCompletion: %v", err)
	}
	if emittedCount != 2 {
		t.Fatalf("emittedCount = %d, want 2 (micro-checkpoint on each tool completion)", emittedCount)
	}
}

func TestNonMutatingToolIgnored(t *testing.T) {
	sessionID := "sess-read"
	tracker := NewScopeTracker()

	paths, admitted := tracker.RecordToolCall(sessionID, "read", map[string]any{"filePath": "foo.go"})
	if admitted || len(paths) != 0 {
		t.Fatalf("read tool should not be admitted: admitted=%v, paths=%v", admitted, paths)
	}
	if len(tracker.ActiveScope(sessionID)) != 0 {
		t.Fatalf("scope should remain empty for non-mutating tool")
	}
}
