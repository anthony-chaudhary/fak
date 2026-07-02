package toolproc

import (
	"encoding/json"
	"testing"
)

func hookPre(t *testing.T, p HookPayload, env HookEnvelope, nowMS int64, existing []Event) []Event {
	t.Helper()
	ev, emit, err := HookEvent("pre", p, env, nowMS, existing)
	if err != nil {
		t.Fatalf("pre: %v", err)
	}
	if !emit {
		return existing
	}
	return append(existing, ev)
}

func hookPost(t *testing.T, p HookPayload, nowMS int64, existing []Event) []Event {
	t.Helper()
	ev, emit, err := HookEvent("post", p, HookEnvelope{}, nowMS, existing)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	if !emit {
		return existing
	}
	return append(existing, ev)
}

// TestHookLifecycleFoldsClean: pre→post from real hook payload shapes folds
// into a clean DONE proc — the whole harness seam round-trips through the
// same fail-closed fold the CLI uses.
func TestHookLifecycleFoldsClean(t *testing.T) {
	p := HookPayload{SessionID: "s1", ToolName: "Bash", ToolUseID: "toolu_01",
		ToolInput: json.RawMessage(`{"command":"sleep 5"}`)}
	var events []Event
	events = hookPre(t, p, HookEnvelope{DeadlineMS: 60_000}, 1_000, events)
	events = hookPost(t, p, 6_000, events)

	tab, err := Fold(events, 10_000, Config{})
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	if len(tab.Procs) != 1 {
		t.Fatalf("want 1 proc, got %+v", tab.Procs)
	}
	pr := tab.Procs[0]
	if pr.CallID != "toolu_01" || pr.State != StateDone || pr.ExitStatus != "ok" || pr.DeadlineMS != 60_000 {
		t.Errorf("got %+v", pr)
	}
}

// TestHookOrphanVisibleWithoutPost: a pre with no post leaves the call
// RUNNING; the stop hook's session_end marks it TOOL_ORPHANED — the crashed
// or abandoned background call is visible, not silently forgotten.
func TestHookOrphanVisibleWithoutPost(t *testing.T) {
	p := HookPayload{SessionID: "s1", ToolName: "Bash", ToolUseID: "toolu_02",
		ToolInput: json.RawMessage(`{"command":"tail -f log"}`)}
	var events []Event
	events = hookPre(t, p, HookEnvelope{}, 1_000, events)
	ev, emit, err := HookEvent("stop", HookPayload{SessionID: "s1"}, HookEnvelope{}, 9_000, events)
	if err != nil || !emit {
		t.Fatalf("stop: emit=%t err=%v", emit, err)
	}
	events = append(events, ev)

	tab, err := Fold(events, 10_000, Config{})
	if err != nil {
		t.Fatal(err)
	}
	pr := tab.Procs[0]
	if !pr.Orphaned || !hasFinding(pr, ReasonToolOrphanedName, AdviceReap) {
		t.Errorf("want orphaned + reap advice, got %+v", pr)
	}
}

// TestHookRespawnGenerations: the same identity (no tool_use_id — hash
// fallback) called twice in one session becomes two generations, never a
// duplicate-spawn violation.
func TestHookRespawnGenerations(t *testing.T) {
	p := HookPayload{SessionID: "s1", ToolName: "Bash",
		ToolInput: json.RawMessage(`{"command":"make test"}`)}
	var events []Event
	events = hookPre(t, p, HookEnvelope{}, 1_000, events)
	events = hookPost(t, p, 2_000, events)
	events = hookPre(t, p, HookEnvelope{}, 3_000, events) // identical call again
	events = hookPost(t, p, 4_000, events)

	tab, err := Fold(events, 5_000, Config{})
	if err != nil {
		t.Fatalf("Fold refused the respawn journal: %v", err)
	}
	if len(tab.Procs) != 2 {
		t.Fatalf("want 2 generations, got %+v", tab.Procs)
	}
	if tab.Procs[0].CallID == tab.Procs[1].CallID {
		t.Errorf("generations must have distinct ids: %s", tab.Procs[0].CallID)
	}
	if tab.Counts.Done != 2 {
		t.Errorf("both generations done, got %+v", tab.Counts)
	}
}

// TestHookIdempotentPreAndUnmatchedPost: a redelivered pre while running
// emits nothing; a post with no journaled spawn emits nothing (hook armed
// mid-session) — the journal stays fold-clean either way.
func TestHookIdempotentPreAndUnmatchedPost(t *testing.T) {
	p := HookPayload{SessionID: "s1", ToolName: "Read", ToolUseID: "toolu_03"}
	var events []Event
	events = hookPre(t, p, HookEnvelope{}, 1_000, events)
	if ev, emit, err := HookEvent("pre", p, HookEnvelope{}, 1_500, events); err != nil || emit {
		t.Errorf("redelivered pre must be silent, got emit=%t ev=%+v err=%v", emit, ev, err)
	}
	orphanPost := HookPayload{SessionID: "s1", ToolName: "Grep", ToolUseID: "toolu_never"}
	if ev, emit, err := HookEvent("post", orphanPost, HookEnvelope{}, 2_000, events); err != nil || emit {
		t.Errorf("post without spawn must be silent, got emit=%t ev=%+v err=%v", emit, ev, err)
	}
	if _, err := Fold(events, 3_000, Config{}); err != nil {
		t.Errorf("journal must stay fold-clean: %v", err)
	}
}

// TestHookErrorResponseAndBadKinds: is_error maps to exit error; unknown hook
// kinds and missing identities refuse.
func TestHookErrorResponseAndBadKinds(t *testing.T) {
	p := HookPayload{SessionID: "s1", ToolName: "Bash", ToolUseID: "toolu_04",
		ToolResponse: json.RawMessage(`{"is_error":true,"content":"boom"}`)}
	var events []Event
	events = hookPre(t, p, HookEnvelope{}, 1_000, events)
	ev, emit, err := HookEvent("post", p, HookEnvelope{}, 2_000, events)
	if err != nil || !emit || ev.Status != "error" {
		t.Errorf("want exit error, got emit=%t status=%q err=%v", emit, ev.Status, err)
	}

	if _, _, err := HookEvent("resume", p, HookEnvelope{}, 1_000, nil); err == nil {
		t.Error("unknown hook kind must refuse")
	}
	if _, _, err := HookEvent("pre", HookPayload{SessionID: "s1"}, HookEnvelope{}, 1_000, nil); err == nil {
		t.Error("pre without tool_name must refuse")
	}
	if _, _, err := HookEvent("stop", HookPayload{}, HookEnvelope{}, 1_000, nil); err == nil {
		t.Error("stop without session_id must refuse")
	}
}
