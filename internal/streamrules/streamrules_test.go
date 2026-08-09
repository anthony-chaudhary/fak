package streamrules

import (
	"strings"
	"testing"
)

func TestDeltaReplayIsolatesSiblingToolCallsAndPinsFirstMatchIndex(t *testing.T) {
	matcher, diagnostics := Compile([]Rule{{
		Name: "secret", Pattern: `password=.+`, Scope: ScopeAnyTool,
	}})
	if len(diagnostics) != 0 {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}

	left := StreamKey{ToolCallID: "call-a", ToolName: "shell", Scope: ScopeAnyTool}
	right := StreamKey{ToolCallID: "call-b", ToolName: "shell", Scope: ScopeAnyTool}
	deltas := []struct {
		key   StreamKey
		delta string
	}{
		{left, "pass"},
		{right, "word=safe"},
		{left, "word="},
		{left, "secret"},
	}
	matchedAt := -1
	for i, step := range deltas {
		matches := matcher.CheckDelta(step.key, step.delta)
		if len(matches) != 0 {
			if matchedAt != -1 {
				t.Fatalf("unexpected repeat match at delta %d: %+v", i, matches)
			}
			matchedAt = i
			if got := matches[0].Key.ToolCallID; got != "call-a" {
				t.Fatalf("matched call = %q, want call-a", got)
			}
		}
	}
	if matchedAt != 3 {
		t.Fatalf("first match delta = %d, want 3", matchedAt)
	}
}

func TestThinkingRequiresExplicitScope(t *testing.T) {
	matcher, _ := Compile([]Rule{
		{Name: "default-text", Pattern: "needle"},
		{Name: "thinking", Pattern: "needle", Scope: ScopeThinking},
	})
	matches := matcher.CheckSnapshot(StreamKey{Scope: ScopeThinking}, "needle")
	if len(matches) != 1 || matches[0].Rule != "thinking" {
		t.Fatalf("thinking matches = %+v, want only explicit thinking rule", matches)
	}
}

func TestConsecutiveIdenticalSnapshotDoesNotRematch(t *testing.T) {
	matcher, _ := Compile([]Rule{{Name: "hit", Pattern: "needle"}})
	key := StreamKey{Scope: ScopeText}
	if got := matcher.CheckSnapshot(key, "needle"); len(got) != 1 {
		t.Fatalf("first snapshot matches = %+v, want one", got)
	}
	if got := matcher.CheckSnapshot(key, "needle"); len(got) != 0 {
		t.Fatalf("identical snapshot matches = %+v, want none", got)
	}
}

func TestInvalidPatternIsDroppedWithoutWedge(t *testing.T) {
	matcher, diagnostics := Compile([]Rule{
		{Name: "broken", Pattern: "["},
		{Name: "good", Pattern: "ok"},
	})
	if len(diagnostics) != 1 || diagnostics[0].Rule != "broken" || !strings.Contains(diagnostics[0].Error, "compile pattern") {
		t.Fatalf("diagnostics = %+v, want broken compile diagnostic", diagnostics)
	}
	matches := matcher.CheckSnapshot(StreamKey{}, "ok")
	if len(matches) != 1 || matches[0].Rule != "good" {
		t.Fatalf("remaining rule matches = %+v, want good", matches)
	}
}

func TestNamedToolScope(t *testing.T) {
	matcher, _ := Compile([]Rule{{
		Name: "shell-only", Pattern: "rm", Scope: ScopeNamedTool, Tool: "shell",
	}})
	if got := matcher.CheckSnapshot(StreamKey{ToolCallID: "read", ToolName: "read_file", Scope: ScopeAnyTool}, "rm"); len(got) != 0 {
		t.Fatalf("wrong-tool matches = %+v", got)
	}
	if got := matcher.CheckSnapshot(StreamKey{ToolCallID: "shell", ToolName: "shell", Scope: ScopeAnyTool}, "rm"); len(got) != 1 {
		t.Fatalf("named-tool matches = %+v, want one", got)
	}
}

func TestPathGlobScope(t *testing.T) {
	matcher, _ := Compile([]Rule{{
		Name: "secrets", Pattern: "token", Scope: ScopeAnyTool, PathGlob: "config/*.env",
	}})
	if got := matcher.CheckSnapshot(StreamKey{ToolCallID: "a", ToolName: "write", Path: "docs/readme.md", Scope: ScopeAnyTool}, "token"); len(got) != 0 {
		t.Fatalf("wrong-path matches = %+v", got)
	}
	if got := matcher.CheckSnapshot(StreamKey{ToolCallID: "b", ToolName: "write", Path: "config/prod.env", Scope: ScopeAnyTool}, "token"); len(got) != 1 {
		t.Fatalf("globbed-path matches = %+v, want one", got)
	}
}

func TestResetTurnClearsBuffersAndMatchLatches(t *testing.T) {
	matcher, _ := Compile([]Rule{{Name: "joined", Pattern: "ab"}})
	key := StreamKey{}
	if got := matcher.CheckDelta(key, "a"); len(got) != 0 {
		t.Fatalf("partial matches = %+v", got)
	}
	if got := matcher.CheckDelta(key, "b"); len(got) != 1 {
		t.Fatalf("completed matches = %+v", got)
	}
	matcher.ResetTurn()
	if got := matcher.CheckDelta(key, "b"); len(got) != 0 {
		t.Fatalf("old buffer survived reset: %+v", got)
	}
	if got := matcher.CheckDelta(key, "ab"); len(got) != 1 {
		t.Fatalf("match latch survived reset: %+v", got)
	}
}
