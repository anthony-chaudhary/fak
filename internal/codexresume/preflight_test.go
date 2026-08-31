package codexresume

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeRolloutRows(t *testing.T, path string, rows ...map[string]any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	encoder := json.NewEncoder(file)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			_ = file.Close()
			t.Fatal(err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}

func sessionMeta(threadID string) map[string]any {
	return map[string]any{
		"type": "session_meta",
		"payload": map[string]any{
			"id":             threadID,
			"model_provider": "fak",
		},
	}
}

func functionCall(itemID, callID string) map[string]any {
	return map[string]any{
		"type": "response_item",
		"payload": map[string]any{
			"type":    "function_call",
			"id":      itemID,
			"call_id": callID,
			"name":    "shell_command",
		},
	}
}

func functionCallOutput(callID string) map[string]any {
	return map[string]any{
		"type": "response_item",
		"payload": map[string]any{
			"type":    "function_call_output",
			"id":      "fco_fixture",
			"call_id": callID,
			"output":  "ok",
		},
	}
}

func TestPreflightRefusesCallItemIDAgainstFCWire(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff1ff-f812-7111-8828-039b1643778f"
	rollout := filepath.Join(home, "legacy.jsonl")
	writeRolloutRows(t, rollout,
		sessionMeta(threadID),
		functionCall("call_legacy", "call_logical"),
		functionCallOutput("call_logical"),
	)

	got := Preflight(CheckConfig{ThreadID: threadID, RolloutPath: rollout, CodexHome: home})
	if got.Verdict != VerdictIncompatibleHistory || got.Compatibility != CompatibilityIncompatible {
		t.Fatalf("preflight=%+v", got)
	}
	if got.FirstIncompatibleItemID != "call_legacy" || got.IncompatibleFunctionCallItems != 1 {
		t.Fatalf("incompatible evidence=%+v", got)
	}
	if len(got.ObservedFunctionCallItemPrefixes) != 1 || got.ObservedFunctionCallItemPrefixes[0] != "call_" {
		t.Fatalf("observed prefixes=%v", got.ObservedFunctionCallItemPrefixes)
	}
	if !strings.Contains(got.RecoveryAction, "response_item.function_call.id") ||
		!strings.Contains(got.RecoveryAction, `"fc_"`) ||
		!strings.Contains(got.RecoveryAction, "no process was spawned") {
		t.Fatalf("recovery action is not exact: %q", got.RecoveryAction)
	}
}

func TestPreflightAcceptsCorrectedFCItemIDWithCallLogicalID(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff1b0-f1e4-73c1-9b29-f3cbed22a43d"
	rollout := filepath.Join(home, "corrected.jsonl")
	writeRolloutRows(t, rollout,
		sessionMeta(threadID),
		functionCall("fc_response_item", "call_logical"),
		functionCallOutput("call_logical"),
	)

	got := Preflight(CheckConfig{ThreadID: threadID, RolloutPath: rollout, CodexHome: home})
	if got.Verdict != VerdictResumable || got.Compatibility != CompatibilityCompatible {
		t.Fatalf("preflight=%+v", got)
	}
	if got.FunctionCallItems != 1 || got.IncompatibleFunctionCallItems != 0 {
		t.Fatalf("call counts=%+v", got)
	}
	if len(got.ObservedFunctionCallItemPrefixes) != 1 || got.ObservedFunctionCallItemPrefixes[0] != "fc_" {
		t.Fatalf("observed prefixes=%v", got.ObservedFunctionCallItemPrefixes)
	}
}

func TestPreflightLiveWriterOwnerIsAlreadyActive(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff1af-9d63-7452-8109-58172f63c3eb"
	rollout := filepath.Join(home, "live-owner.jsonl")
	writeRolloutRows(t, rollout, sessionMeta(threadID))
	makeWriterLock(t, home, threadID)

	got := preflightWithProbe(CheckConfig{ThreadID: threadID, RolloutPath: rollout, CodexHome: home}, fixedOwnershipProbe{
		witness: ownershipWitness{source: "test_native_witness", conclusive: true, owners: []processOwner{{
			pid: 321, startTime: "2026-08-31T10:00:00Z", startToken: 77, image: `C:\codex.exe`,
		}}},
	})
	if got.Verdict != VerdictAlreadyActive || got.WriterOwnership.Verdict != WriterOwnershipLiveOwner {
		t.Fatalf("preflight=%+v", got)
	}
	if got.WriterOwnership.HandleReceiptID == "" || got.WriterOwnership.PID != 321 {
		t.Fatalf("ownership receipt=%+v", got.WriterOwnership)
	}
}

func TestPreflightPositiveNoOwnerWitnessAllowsCompatibleResume(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff1af-9d63-7452-8109-58172f63c3ec"
	rollout := filepath.Join(home, "stale-owner.jsonl")
	writeRolloutRows(t, rollout, sessionMeta(threadID))
	makeWriterLock(t, home, threadID)

	got := preflightWithProbe(CheckConfig{ThreadID: threadID, RolloutPath: rollout, CodexHome: home}, fixedOwnershipProbe{
		witness: ownershipWitness{source: "test_positive_no_owner", conclusive: true},
	})
	if got.Verdict != VerdictResumable || got.WriterOwnership.Verdict != WriterOwnershipStaleResidue {
		t.Fatalf("preflight=%+v", got)
	}
	if !got.StaleWriterLockSuspected {
		t.Fatalf("stale residue not surfaced: %+v", got)
	}
}

func TestPreflightUnknownWriterOwnershipFailsClosed(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff1af-9d63-7452-8109-58172f63c3ed"
	rollout := filepath.Join(home, "unknown-owner.jsonl")
	writeRolloutRows(t, rollout, sessionMeta(threadID))
	makeWriterLock(t, home, threadID)

	got := preflightWithProbe(CheckConfig{ThreadID: threadID, RolloutPath: rollout, CodexHome: home}, fixedOwnershipProbe{
		witness: ownershipWitness{source: "test_permission_probe"}, err: os.ErrPermission,
	})
	if got.Verdict != VerdictAlreadyActive || got.WriterOwnership.Verdict != WriterOwnershipUnknown {
		t.Fatalf("preflight=%+v", got)
	}
	if !strings.Contains(got.RecoveryAction, "do not delete") || !strings.Contains(got.RecoveryAction, "do not") {
		t.Fatalf("fail-closed action=%q", got.RecoveryAction)
	}
}

func TestPreflightMarksFailedWrapperAndWriterLock(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff1af-9d63-7452-8109-58172f63c3e9"
	rollout := filepath.Join(home, "failed.jsonl")
	writeRolloutRows(t, rollout,
		sessionMeta(threadID),
		functionCall("call_legacy", "call_logical"),
		map[string]any{"type": "event_msg", "payload": map[string]any{"type": "task_started", "turn_id": "turn-failed"}},
		map[string]any{
			"type": "event_msg",
			"payload": map[string]any{
				"type":    "task_complete",
				"turn_id": "turn-failed",
				"error": map[string]any{
					"message": `{"type":"error","error":{"type":"invalid_request_error","message":"Invalid input[3].id: call_legacy. Expected an ID that begins with fc."},"status":400}`,
				},
			},
		},
	)
	lock := filepath.Join(home, "thread-writer-locks", threadID+".lock")
	if err := os.MkdirAll(filepath.Dir(lock), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(lock, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	got := preflightWithProbe(CheckConfig{ThreadID: threadID, RolloutPath: rollout, CodexHome: home}, fixedOwnershipProbe{
		witness: ownershipWitness{source: "test_positive_no_owner", conclusive: true},
	})
	if got.Verdict != VerdictIncompatibleHistory || !got.FailedWrapperMarked || !got.StaleWriterLockSuspected {
		t.Fatalf("preflight=%+v", got)
	}
	if got.WriterOwnership.Verdict != WriterOwnershipStaleResidue {
		t.Fatalf("ownership=%+v", got.WriterOwnership)
	}
	if got.Compatibility != CompatibilityIncompatible || got.LatestTurnStatus != "failed" {
		t.Fatalf("compatibility/turn=%+v", got)
	}
	if got.LatestTurnError == nil || got.LatestTurnError.Status != 400 {
		t.Fatalf("latest error=%+v", got.LatestTurnError)
	}
	if strings.Contains(got.RecoveryAction, "terminate") || strings.Contains(got.RecoveryAction, "remove the writer lock") {
		t.Fatalf("unsafe recovery action=%q", got.RecoveryAction)
	}
}

func TestPreflightDiscoversRolloutBySessionMeta(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff200-0000-7000-8000-000000000001"
	rollout := filepath.Join(home, "sessions", "2026", "08", "11", "rollout-"+threadID+".jsonl")
	writeRolloutRows(t, rollout, sessionMeta(threadID))

	got := Preflight(CheckConfig{ThreadID: threadID, CodexHome: home})
	if got.Verdict != VerdictResumable {
		t.Fatalf("preflight=%+v", got)
	}
	if got.RolloutPath != rollout {
		t.Fatalf("rollout=%q want=%q", got.RolloutPath, rollout)
	}
}

func TestRecoverRefusesLegacyCallHistoryBeforeLaunch(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff200-0000-7000-8000-000000000002"
	rollout := filepath.Join(home, "legacy.jsonl")
	writeRolloutRows(t, rollout,
		sessionMeta(threadID),
		functionCall("call_would_400", "call_logical"),
		functionCallOutput("call_logical"),
	)

	got, err := Recover(context.Background(),
		CheckConfig{ThreadID: threadID, RolloutPath: rollout, CodexHome: home},
		Config{Command: []string{"this-command-must-not-run"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got.Outcome != OutcomeRefused || got.LaunchPID != 0 {
		t.Fatalf("result=%+v", got)
	}
	if got.Preflight == nil || got.Preflight.Verdict != VerdictIncompatibleHistory {
		t.Fatalf("preflight=%+v", got.Preflight)
	}
}

func TestRecoverCorrectedHistoryReachesAuthoritativeCompletion(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff200-0000-7000-8000-000000000003"
	rollout := filepath.Join(home, "corrected.jsonl")
	writeRolloutRows(t, rollout,
		sessionMeta(threadID),
		functionCall("fc_response_item", "call_logical"),
		functionCallOutput("call_logical"),
	)
	got, err := Recover(context.Background(),
		CheckConfig{ThreadID: threadID, RolloutPath: rollout, CodexHome: home},
		Config{
			Command:      helperCommand(t, "exit", rollout),
			Env:          helperEnv(),
			Deadline:     time.Second,
			PollInterval: 20 * time.Millisecond,
			Drain:        30 * time.Millisecond,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got.Outcome != OutcomeCompleted || got.LaunchPID == 0 || got.TurnStatus != "completed" || !got.TaskCompleted {
		t.Fatalf("result=%+v", got)
	}
	if got.Preflight == nil || got.Preflight.Verdict != VerdictResumable {
		t.Fatalf("preflight=%+v", got.Preflight)
	}
}

func TestPreflightEmitsTypedThreadAndResourceWithCompatibilityFields(t *testing.T) {
	home := t.TempDir()
	thread, err := NewCodexThreadIdentity(testThreadIDOne)
	if err != nil {
		t.Fatal(err)
	}
	rollout := filepath.Join(home, "sessions", "typed.jsonl")
	writeRolloutRows(t, rollout, sessionMeta(thread.ID), functionCall("fc_call", "call_logical"))
	lockPath := makeWriterLock(t, home, thread.ID)
	got := preflightWithProbe(CheckConfig{ThreadID: thread.ID, Thread: thread, RolloutPath: rollout, CodexHome: home}, fixedOwnershipProbe{
		witness: ownershipWitness{source: "test_positive", conclusive: true},
	})
	if got.Thread == nil || *got.Thread != thread {
		t.Fatalf("typed thread=%+v", got.Thread)
	}
	if got.WriterOwnership.Resource == nil || got.WriterOwnership.Resource.Thread != thread {
		t.Fatalf("typed resource=%+v", got.WriterOwnership.Resource)
	}
	if got.ThreadID != thread.ID || got.WriterLockPath != lockPath || got.WriterOwnership.LockPath != got.WriterOwnership.Resource.LockPath {
		t.Fatalf("compatibility fields result=%+v", got)
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"thread_id", "thread", "writer_lock_path", "writer_ownership", "resource", "resource_id", "lock_path"} {
		if !strings.Contains(string(data), `"`+field+`"`) {
			t.Fatalf("preflight JSON missing %s: %s", field, data)
		}
	}
}

func TestPreflightThreadIdentityMismatchFailsClosedBeforeOwnershipProbe(t *testing.T) {
	thread, _ := NewCodexThreadIdentity(testThreadIDOne)
	got := preflightWithProbe(CheckConfig{ThreadID: testThreadIDTwo, Thread: thread, CodexHome: t.TempDir()}, fixedOwnershipProbe{})
	if got.Verdict != VerdictNotFound || !strings.Contains(got.Detail, "does not match") {
		t.Fatalf("mismatched preflight=%+v", got)
	}
}
