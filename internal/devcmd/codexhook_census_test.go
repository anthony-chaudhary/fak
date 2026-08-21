package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestBuildHookCensusClassifiesCompleteLifecycle(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "profile")
	sessions := filepath.Join(home, "sessions")
	if err := os.MkdirAll(sessions, 0755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	rows := `{"timestamp":"2026-08-17T11:59:00Z","type":"session_meta","payload":{"session_id":"thread-1","cwd":"` + filepath.ToSlash(home) + `"}}` + "\n" + `{"timestamp":"2026-08-17T11:00:00Z","type":"response_item","payload":{"type":"function_call","call_id":"a","name":"Read"}}` + "\n" + `{"timestamp":"2026-08-17T11:01:00Z","type":"response_item","payload":{"type":"custom_tool_call","call_id":"b","name":"exec"}}` + "\n" + `{"timestamp":"2026-08-17T11:02:00Z","type":"response_item","payload":{"type":"function_call_output","call_id":"a","name":"Read"}}` + "\n" + `{"timestamp":"2026-08-17T11:03:00Z","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"b","name":"exec"}}` + "\n"
	if err := os.WriteFile(filepath.Join(sessions, "r.jsonl"), []byte(rows), 0644); err != nil {
		t.Fatal(err)
	}
	obs := filepath.Join(root, "observations.jsonl")
	var b bytes.Buffer
	for _, verb := range []string{"pretool", "posttool"} {
		for _, callID := range []string{"a", "b"} {
			_ = json.NewEncoder(&b).Encode(hookObservation{CallID: callID, SessionID: "thread-1", Workspace: home, Profile: home, PhaseState: "succeeded", Exit: 0, Verb: verb, TS: now.Add(-time.Minute)})
		}
	}
	if err := os.WriteFile(obs, b.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := buildHookCensus(home, home, home, "", obs, time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "HEALTHY" || got.DispatchedCalls != 2 || got.PreToolUse.Succeeded != 2 || got.PostToolUse.Unknown != 0 {
		t.Fatalf("report=%+v", got)
	}
}
func TestBuildHookCensusPostToolUseExcludesActiveCalls(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "profile")
	sessions := filepath.Join(home, "sessions")
	if err := os.MkdirAll(sessions, 0755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	rows := `{"timestamp":"2026-08-17T11:59:00Z","type":"session_meta","payload":{"session_id":"thread-1","cwd":"` + filepath.ToSlash(home) + `"}}` + "\n" +
		`{"timestamp":"2026-08-17T11:00:00Z","type":"response_item","payload":{"type":"function_call","call_id":"complete","name":"Read"}}` + "\n" +
		`{"timestamp":"2026-08-17T11:01:00Z","type":"response_item","payload":{"type":"function_call","call_id":"active","name":"Read"}}` + "\n" +
		`{"timestamp":"2026-08-17T11:02:00Z","type":"response_item","payload":{"type":"function_call_output","call_id":"complete","name":"Read"}}` + "\n"
	if err := os.WriteFile(filepath.Join(sessions, "r.jsonl"), []byte(rows), 0644); err != nil {
		t.Fatal(err)
	}
	obs := filepath.Join(root, "observations.jsonl")
	var b bytes.Buffer
	for _, o := range []hookObservation{
		{CallID: "complete", SessionID: "thread-1", Workspace: home, Profile: home, PhaseState: "succeeded", Verb: "pretool", TS: now.Add(-time.Minute)},
		{CallID: "active", SessionID: "thread-1", Workspace: home, Profile: home, PhaseState: "succeeded", Verb: "pretool", TS: now.Add(-time.Minute)},
		{CallID: "complete", SessionID: "thread-1", Workspace: home, Profile: home, PhaseState: "succeeded", Verb: "posttool", TS: now.Add(-time.Minute)},
	} {
		_ = json.NewEncoder(&b).Encode(o)
	}
	if err := os.WriteFile(obs, b.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := buildHookCensus(home, home, home, "", obs, time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	if got.DispatchedCalls != 2 || got.PreToolUse.Denominator != 2 || got.PreToolUse.Succeeded != 2 {
		t.Fatalf("pretool=%+v dispatched=%d", got.PreToolUse, got.DispatchedCalls)
	}
	if got.PostToolUse.Denominator != 1 || got.PostToolUse.Succeeded != 1 || got.PostToolUse.Unknown != 0 {
		t.Fatalf("posttool=%+v", got.PostToolUse)
	}
}
func TestBuildHookCensusExcludesToolsOutsidePostMatcher(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "profile")
	sessions := filepath.Join(home, "sessions")
	if err := os.MkdirAll(sessions, 0755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	rows := `{"timestamp":"2026-08-17T11:59:00Z","type":"session_meta","payload":{"session_id":"thread-1","cwd":"` + filepath.ToSlash(home) + `"}}` + "\n" +
		`{"timestamp":"2026-08-17T11:00:00Z","type":"response_item","payload":{"type":"function_call","call_id":"plan","name":"update_plan"}}` + "\n" +
		`{"timestamp":"2026-08-17T11:01:00Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan"}}` + "\n"
	if err := os.WriteFile(filepath.Join(sessions, "r.jsonl"), []byte(rows), 0644); err != nil {
		t.Fatal(err)
	}
	obs := filepath.Join(root, "observations.jsonl")
	o := hookObservation{CallID: "plan", SessionID: "thread-1", Workspace: home, Profile: home, PhaseState: "succeeded", Verb: "pretool", TS: now.Add(-time.Minute)}
	raw, _ := json.Marshal(o)
	if err := os.WriteFile(obs, append(raw, '\n'), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := buildHookCensus(home, home, home, "", obs, time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	if got.PreToolUse.Denominator != 1 || got.PostToolUse.Denominator != 0 || got.PostToolUse.Unknown != 0 {
		t.Fatalf("pretool=%+v posttool=%+v", got.PreToolUse, got.PostToolUse)
	}
}
func TestBuildHookCensusSettlesFreshPostToolReceipt(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "profile")
	sessions := filepath.Join(home, "sessions")
	if err := os.MkdirAll(sessions, 0755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	rows := `{"timestamp":"2026-08-17T11:59:00Z","type":"session_meta","payload":{"session_id":"thread-1","cwd":"` + filepath.ToSlash(home) + `"}}` + "\n" +
		`{"timestamp":"2026-08-17T11:59:56Z","type":"response_item","payload":{"type":"function_call","call_id":"fresh","name":"Read"}}` + "\n" +
		`{"timestamp":"2026-08-17T11:59:58Z","type":"response_item","payload":{"type":"function_call_output","call_id":"fresh","name":"Read"}}` + "\n"
	if err := os.WriteFile(filepath.Join(sessions, "r.jsonl"), []byte(rows), 0644); err != nil {
		t.Fatal(err)
	}
	obs := filepath.Join(root, "observations.jsonl")
	o := hookObservation{CallID: "fresh", SessionID: "thread-1", Workspace: home, Profile: home, PhaseState: "succeeded", Verb: "pretool", TS: now.Add(-time.Minute)}
	raw, _ := json.Marshal(o)
	if err := os.WriteFile(obs, append(raw, '\n'), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := buildHookCensus(home, home, home, "", obs, time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	if got.PostToolSettlement != "5s" || got.PostToolUse.Denominator != 0 || got.PostToolUse.Unknown != 0 {
		t.Fatalf("settlement=%q posttool=%+v", got.PostToolSettlement, got.PostToolUse)
	}
}
func TestBuildHookCensusFailsClosedOnUnknownFailureAndProfileMismatch(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "active")
	logHome := filepath.Join(root, "other")
	if err := os.MkdirAll(filepath.Join(logHome, "sessions"), 0755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	row := `{"timestamp":"2026-08-17T11:59:00Z","type":"session_meta","payload":{"session_id":"thread-1","cwd":"` + filepath.ToSlash(logHome) + `"}}` + "\n" + `{"timestamp":"2026-08-17T11:00:00Z","type":"response_item","payload":{"type":"function_call","call_id":"a","name":"Read"}}` + "\n" + `{"timestamp":"2026-08-17T11:01:00Z","type":"response_item","payload":{"type":"function_call_output","call_id":"a","name":"Read"}}` + "\n"
	_ = os.WriteFile(filepath.Join(logHome, "sessions", "r.jsonl"), []byte(row), 0644)
	obs := filepath.Join(root, "o.jsonl")
	o := hookObservation{CallID: "a", SessionID: "thread-1", Workspace: logHome, Profile: home, PhaseState: "failed", Exit: 3, Verb: "posttool", TS: now.Add(-time.Minute)}
	raw, _ := json.Marshal(o)
	_ = os.WriteFile(obs, append(raw, '\n'), 0644)
	got, err := buildHookCensus(home, logHome, logHome, "", obs, time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "UNHEALTHY" || got.ProfileMatch || got.PreToolUse.Unknown != 1 || got.PostToolUse.Failed != 1 {
		t.Fatalf("report=%+v", got)
	}
}

func TestPhaseCountsTypesSkippedDisabledAndUnknown(t *testing.T) {
	got := phaseCounts(4, []hookObservation{{Exit: 0, Outcome: "passthrough"}, {Outcome: "skipped"}, {Outcome: "disabled"}}, "fixture")
	if got.Attempted != 1 || got.Succeeded != 1 || got.Skipped != 1 || got.Disabled != 1 || got.Unknown != 1 {
		t.Fatalf("counts=%+v", got)
	}
}

func TestBuildHookCensusRejectsAbsentTelemetry(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	if err := os.MkdirAll(filepath.Join(home, "sessions"), 0755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	row := `{"timestamp":"2026-08-17T11:59:00Z","type":"session_meta","payload":{"session_id":"thread-1","cwd":"` + filepath.ToSlash(home) + `"}}` + "\n" + `{"timestamp":"2026-08-17T11:00:00Z","type":"response_item","payload":{"type":"function_call","call_id":"a","name":"Read"}}` + "\n" + `{"timestamp":"2026-08-17T11:01:00Z","type":"response_item","payload":{"type":"function_call_output","call_id":"a","name":"Read"}}` + "\n"
	_ = os.WriteFile(filepath.Join(home, "sessions", "r.jsonl"), []byte(row), 0644)
	obs := filepath.Join(root, "o.jsonl")
	_ = os.WriteFile(obs, nil, 0644)
	got, err := buildHookCensus(home, home, home, "", obs, time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "UNHEALTHY" || got.PreToolUse.Unknown != 1 || got.PostToolUse.Unknown != 1 {
		t.Fatalf("report=%+v", got)
	}
}

func TestPhaseCountsForCallsJoinsParallelAndRejectsDuplicatesAndForeignIdentity(t *testing.T) {
	root := t.TempDir()
	profile := filepath.Join(root, "profile")
	workspace := filepath.Join(root, "workspace")
	calls := []dispatchedCall{
		{CallID: "call-a", SessionID: "thread", Workspace: workspace},
		{CallID: "call-b", SessionID: "thread", Workspace: workspace},
		{CallID: "call-c", SessionID: "thread", Workspace: workspace},
		{CallID: "call-d", SessionID: "thread", Workspace: workspace},
	}
	obs := []hookObservation{
		{CallID: "call-b", SessionID: "thread", Workspace: workspace, Profile: profile, PhaseState: "skipped"},
		{CallID: "call-a", SessionID: "thread", Workspace: workspace, Profile: profile, PhaseState: "succeeded"},
		{CallID: "call-a", SessionID: "thread", Workspace: workspace, Profile: profile, PhaseState: "failed"}, // duplicate
		{CallID: "call-c", SessionID: "thread", Workspace: workspace, Profile: profile, PhaseState: "disabled"},
		{CallID: "unmatched", SessionID: "thread", Workspace: workspace, Profile: profile, PhaseState: "succeeded"},
		{CallID: "call-d", SessionID: "thread", Workspace: workspace, Profile: filepath.Join(root, "foreign"), PhaseState: "succeeded"},
	}
	got := phaseCountsForCalls(calls, obs, profile, workspace, "fixture")
	if got.Denominator != 4 || got.Attempted != 1 || got.Succeeded != 1 || got.Failed != 0 || got.Skipped != 1 || got.Disabled != 1 || got.Unknown != 1 {
		t.Fatalf("counts=%+v", got)
	}
}

func TestPhaseCountsForCallsRejectsWorkspaceMismatch(t *testing.T) {
	root := t.TempDir()
	profile := filepath.Join(root, "profile")
	workspace := filepath.Join(root, "workspace")
	calls := []dispatchedCall{{CallID: "call-a", SessionID: "thread", Workspace: workspace}}
	obs := []hookObservation{{CallID: "call-a", SessionID: "thread", Workspace: filepath.Join(root, "other"), Profile: profile, PhaseState: "succeeded"}}
	got := phaseCountsForCalls(calls, obs, profile, workspace, "fixture")
	if got.Unknown != 1 || got.Succeeded != 0 {
		t.Fatalf("counts=%+v", got)
	}
}

func TestStopLifecycleClassifiesAllStatesAndIgnoresEchoes(t *testing.T) {
	d := t.TempDir()
	path := filepath.Join(d, "hooks.jsonl")
	rows := []string{
		`{"method":"hook/started","params":{"threadId":"thread-a","turnId":"turn-1","run":{"id":"run-ok","eventName":"stop","handlerType":"command","source":"plugin","sourcePath":"C:/plugin/hooks.json","displayOrder":6,"status":"running","startedAt":1787098500000}}}`,
		`{"method":"hook/completed","params":{"threadId":"thread-a","turnId":"turn-1","run":{"id":"run-ok","eventName":"stop","handlerType":"command","source":"plugin","sourcePath":"C:/plugin/hooks.json","displayOrder":6,"status":"completed","startedAt":1787098500000,"completedAt":1787098500100}}}`,
		`{"method":"hook/completed","params":{"threadId":"thread-a","turnId":"turn-1","run":{"id":"run-block","eventName":"stop","displayOrder":7,"status":"blocked","startedAt":1787098500200,"completedAt":1787098500300}}}`,
		`{"method":"hook/completed","params":{"threadId":"thread-a","turnId":"turn-1","run":{"id":"run-fail","eventName":"stopFailure","displayOrder":8,"status":"failed","statusMessage":"exit 1","startedAt":1787098500400,"completedAt":1787098500500}}}`,
		`{"method":"hook/started","params":{"threadId":"thread-a","turnId":"turn-1","run":{"id":"run-missing","eventName":"subagentStop","displayOrder":9,"status":"running","startedAt":1787098500600}}}`,
		`{"type":"agent_message","text":"hook/completed Stop failed is only an echo"}`,
		`{"method":"hook/completed","params":{"threadId":"other","run":{"id":"ignored","eventName":"stop","status":"failed","startedAt":1787098500000}}}`,
		`{"method":"hook/completed","params":{"threadId":"thread-a","run":{"id":"old","eventName":"stop","status":"failed","startedAt":1}}}`,
		`{"method":"hook/completed","params":{"threadId":"thread-a","run":{"id":"bad","eventName":"stop"`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(rows, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	r := hookCensusReport{}
	after := time.Date(2026, 8, 18, 23, 0, 0, 0, time.UTC)
	if err := addStopLifecycle(&r, path, after, "thread-a"); err != nil {
		t.Fatal(err)
	}
	finalizeCensusVerdict(&r)
	if r.Stop.Denominator != 3 || r.Stop.Succeeded != 1 || r.Stop.Blocked != 1 || r.Stop.InvalidJSON != 1 {
		t.Fatalf("stop=%+v", r.Stop)
	}
	if r.StopFailure.Failed != 1 || r.StopFailure.Denominator != 1 {
		t.Fatalf("stopFailure=%+v", r.StopFailure)
	}
	if r.SubagentStop.Unknown != 1 || r.SubagentStop.Attempted != 1 {
		t.Fatalf("subagentStop=%+v", r.SubagentStop)
	}
	if len(r.StopRuns) != 4 {
		t.Fatalf("runs=%d %+v", len(r.StopRuns), r.StopRuns)
	}
	if r.Verdict != "UNHEALTHY" || !slices.Contains(r.Reasons, "STOP_INVALID_JSON") || !slices.Contains(r.Reasons, "STOP_FAILURE_FAILED") || !slices.Contains(r.Reasons, "SUBAGENT_STOP_UNKNOWN") {
		t.Fatalf("verdict=%s reasons=%v", r.Verdict, r.Reasons)
	}
}

func TestStopLifecycleDuplicateCompletedRunCountsOnce(t *testing.T) {
	p := filepath.Join(t.TempDir(), "hooks.jsonl")
	row := `{"method":"hook/completed","params":{"threadId":"t","run":{"id":"same","eventName":"stop","status":"completed","startedAt":1787098500000,"completedAt":1787098500100}}}`
	if err := os.WriteFile(p, []byte(row+"\n"+row+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	r := hookCensusReport{}
	if err := addStopLifecycle(&r, p, time.Time{}, "t"); err != nil {
		t.Fatal(err)
	}
	if r.Stop.Denominator != 1 || r.Stop.Succeeded != 1 {
		t.Fatalf("stop=%+v", r.Stop)
	}
}
