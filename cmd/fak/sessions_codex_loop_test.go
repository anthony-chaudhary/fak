package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCodexOutcomeProgressExemptionDiscriminates(t *testing.T) {
	// update_plan across fully distinct plans is forward progress, exempt.
	progress := codexRepeatedOutcome{Tool: "update_plan", Count: 44, ArgsDigestCount: 44}
	if !codexOutcomeIsForwardProgress(progress) {
		t.Fatalf("distinct-arg update_plan should be forward progress: %+v", progress)
	}
	// Same plan re-submitted (args_digests < count) is thrash — still a loop.
	thrash := codexRepeatedOutcome{Tool: "update_plan", Count: 44, ArgsDigestCount: 1}
	if codexOutcomeIsForwardProgress(thrash) {
		t.Fatalf("same-arg update_plan must stay a loop signal: %+v", thrash)
	}
	// A real work tool is never exempt, even with fully distinct args (this is
	// the only thing separating it from update_plan — the exec subagent fixture).
	work := codexRepeatedOutcome{Tool: "exec", Count: 3, LongestRun: 3, ArgsDigestCount: 3}
	if codexOutcomeIsForwardProgress(work) {
		t.Fatalf("exec is not a progress tool and must never be exempt: %+v", work)
	}
	// A progress tool must never mask a concurrent real loop: the work tool drives.
	top, ok := codexTopLoopDrivingOutcome([]codexRepeatedOutcome{progress, work})
	if !ok || top.Tool != "exec" {
		t.Fatalf("progress outcome masked the real loop: top=%+v ok=%v", top, ok)
	}
	// All-forward-progress → no loop-driving outcome.
	if _, ok := codexTopLoopDrivingOutcome([]codexRepeatedOutcome{progress}); ok {
		t.Fatal("all-forward-progress outcomes must not drive a loop verdict")
	}
}

func TestSessionsCodexLoopTreatsDistinctPlanTrafficAsProgress(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rollout-2026-07-11T08-00-00-progress.jsonl")
	lines := []string{
		`{"timestamp":"2026-07-11T15:00:00.000Z","type":"session_meta","payload":{"session_id":"progress-session","originator":"codex-tui","cli_version":"0.142.5","model_provider":"fak","git":{"commit_hash":"abc1234","branch":"main"}}}`,
	}
	for i, step := range []string{"one", "two", "three", "four", "five", "six"} {
		call := fmt.Sprintf("plan_%d", i+1)
		lines = append(lines,
			fmt.Sprintf(`{"timestamp":"2026-07-11T15:0%d:03.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"%s\",\"status\":\"in_progress\"}]}","call_id":"%s"}}`, i, step, call),
			fmt.Sprintf(`{"timestamp":"2026-07-11T15:0%d:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"%s","output":"Plan updated"}}`, i, call),
		)
	}
	writeCodexLoopFixture(t, path, lines)

	fh, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	d, diagErr := diagnoseCodexLoop(fh, path)
	closeErr := fh.Close()
	if diagErr != nil {
		t.Fatal(diagErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	if d.Verdict != "OK" {
		t.Fatalf("distinct-plan update_plan traffic classified %s, want OK: %+v", d.Verdict, d)
	}
	if d.Reason != "repeated_progress_tool_no_loop" {
		t.Fatalf("reason = %q, want repeated_progress_tool_no_loop: %+v", d.Reason, d)
	}
	// The traffic stays visible for token/burn observability even though it is
	// not a loop — this is where the 44-call / 6.2M-token burn is surfaced.
	if len(d.RepeatedOutcomes) != 1 || d.RepeatedOutcomes[0].Tool != "update_plan" {
		t.Fatalf("update_plan traffic should stay visible for observability: %+v", d.RepeatedOutcomes)
	}
	if got := d.RepeatedOutcomes[0]; got.Count != 6 || got.ArgsDigestCount != 6 {
		t.Fatalf("want count==args_digests==6 (fully distinct plans), got %+v", got)
	}
}

func TestSessionsCodexLoopDoesNotTreatSparseSuccessfulOutputCollisionsAsLoop(t *testing.T) {
	path := filepath.Join(t.TempDir(), "productive-session.jsonl")
	var lines []string
	lines = append(lines, `{"timestamp":"2026-08-19T07:00:00Z","type":"session_meta","payload":{"id":"productive-session","cwd":"C:/work/fak","model_provider":"fak","cli_version":"0.85.0","git":{"commit_hash":"abc123","branch":"main"}}}`)
	for i := 0; i < 12; i++ {
		callID := fmt.Sprintf("call-%d", i)
		probeID := fmt.Sprintf("probe-%d", i)
		lines = append(lines,
			fmt.Sprintf(`{"timestamp":"2026-08-19T07:00:%02dZ","type":"response_item","payload":{"type":"function_call","name":"apply_patch","arguments":{"patch":"patch-%d"},"call_id":"%s"}}`, i*4+1, i, callID),
			fmt.Sprintf(`{"timestamp":"2026-08-19T07:00:%02dZ","type":"response_item","payload":{"type":"function_call_output","call_id":"%s","output":"Success. Updated the following files:"}}`, i*4+2, callID),
			fmt.Sprintf(`{"timestamp":"2026-08-19T07:00:%02dZ","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":{"command":"echo %d"},"call_id":"%s"}}`, i*4+3, i, probeID),
			fmt.Sprintf(`{"timestamp":"2026-08-19T07:00:%02dZ","type":"response_item","payload":{"type":"function_call_output","call_id":"%s","output":"probe-%d"}}`, i*4+4, probeID, i),
		)
	}
	writeCodexLoopFixture(t, path, lines)

	var stdout, stderr bytes.Buffer
	if code := sessionsCodexLoop(&stdout, &stderr, []string{"--path", path, "--json"}); code != 0 {
		t.Fatalf("sessions codex-loop code=%d stderr=%q", code, stderr.String())
	}
	var got codexLoopDiagnosis
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode report: %v", err)
	}
	if got.Verdict != "OK" {
		t.Fatalf("verdict=%q reason=%q repeated=%+v", got.Verdict, got.Reason, got.RepeatedOutcomes)
	}
	if len(got.RepeatedOutcomes) == 0 || got.RepeatedOutcomes[0].LongestRun != 1 || got.RepeatedOutcomes[0].ArgsDigestCount != 12 {
		t.Fatalf("expected collision evidence without a loop verdict, got %+v", got.RepeatedOutcomes)
	}
}

func TestSessionsCodexLoopDiagnosesRepeatedGoalFailure(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rollout-2026-07-05T14-02-33-019f3417-0c8a-7da0-bf38-4ee9fd2354e4.jsonl")
	writeCodexLoopFixture(t, path, []string{
		`{"timestamp":"2026-07-05T21:02:38.156Z","type":"session_meta","payload":{"session_id":"019f3417-0c8a-7da0-bf38-4ee9fd2354e4","originator":"codex-tui","cli_version":"0.142.5","model_provider":"fak","git":{"commit_hash":"bf927a8","branch":"main"}}}`,
		`{"timestamp":"2026-07-05T21:02:49.000Z","type":"response_item","payload":{"type":"function_call","name":"create_goal","arguments":"{\"objective\":\"Make GLM 5.2 run through fak\"}","call_id":"call_success"}}`,
		`{"timestamp":"2026-07-05T21:02:50.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call_success","output":"{\"goal\":{\"status\":\"active\"}}"}}`,
		`{"timestamp":"2026-07-05T21:02:50.100Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":16734,"output_tokens":136,"total_tokens":16870},"last_token_usage":{"input_tokens":16734,"output_tokens":136,"total_tokens":16870}}}}`,
		`{"timestamp":"2026-07-05T21:02:57.000Z","type":"response_item","payload":{"type":"function_call","name":"create_goal","arguments":"{\"objective\":\"GLM 5.2 pure fak kernel\"}","call_id":"call_fail_1"}}`,
		`{"timestamp":"2026-07-05T21:02:58.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call_fail_1","output":"cannot create a new goal because this thread has an unfinished goal; complete the existing goal first"}}`,
		`{"timestamp":"2026-07-05T21:02:58.100Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":33468,"output_tokens":238,"total_tokens":33706},"last_token_usage":{"input_tokens":16734,"output_tokens":102,"total_tokens":16836}}}}`,
		`{"timestamp":"2026-07-05T21:03:02.000Z","type":"response_item","payload":{"type":"function_call","name":"create_goal","arguments":"{\"objective\":\"Make GLM 5.2 run through fak\"}","call_id":"call_fail_2"}}`,
		`{"timestamp":"2026-07-05T21:03:03.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call_fail_2","output":"cannot create a new goal because this thread has an unfinished goal; complete the existing goal first"}}`,
		`{"timestamp":"2026-07-05T21:03:03.100Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":50202,"output_tokens":329,"total_tokens":50531},"last_token_usage":{"input_tokens":16734,"output_tokens":91,"total_tokens":16825}}}}`,
		`{"timestamp":"2026-07-05T21:03:08.000Z","type":"response_item","payload":{"type":"function_call","name":"create_goal","arguments":"{\"objective\":\"GLM 5.2 pure fak kernel\"}","call_id":"call_fail_3"}}`,
		`{"timestamp":"2026-07-05T21:03:09.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call_fail_3","output":"cannot create a new goal because this thread has an unfinished goal; complete the existing goal first"}}`,
		`{"timestamp":"2026-07-05T21:03:09.100Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":66936,"output_tokens":435,"total_tokens":67371},"last_token_usage":{"input_tokens":16734,"output_tokens":106,"total_tokens":16840}}}}`,
		`{"timestamp":"2026-07-05T21:03:23.000Z","type":"event_msg","payload":{"type":"agent_message","message":"[fak] observed repeated admitted tool call(s): LIVELOCK_DETECTED repeat=3 repeated_call=create_goal@sha256:92347 approach=change_approach_stop_repeating_successful_call_or_summarize_result. This is advisory."}}`,
		`{"timestamp":"2026-07-05T21:04:44.675Z","type":"event_msg","payload":{"type":"turn_aborted","reason":"interrupted","duration_ms":126641}}`,
		`{"timestamp":"2026-07-05T21:04:44.687Z","type":"event_msg","payload":{"type":"thread_goal_updated","goal":{"status":"paused","tokensUsed":50501,"timeUsedSeconds":42}}}`,
	})

	var stdout, stderr bytes.Buffer
	code := runSessions(&stdout, &stderr, []string{"codex-loop", "--path", path, "--json"})
	if code != 0 {
		t.Fatalf("codex-loop --json exited %d stderr=%s", code, stderr.String())
	}
	var d codexLoopDiagnosis
	if err := json.Unmarshal(stdout.Bytes(), &d); err != nil {
		t.Fatalf("json did not decode: %v\n%s", err, stdout.String())
	}
	if d.Verdict != "LOOP" || d.Reason != "repeated_tool_output" {
		t.Fatalf("verdict = %s/%s, want LOOP/repeated_tool_output: %+v", d.Verdict, d.Reason, d)
	}
	if len(d.RepeatedOutcomes) != 1 {
		t.Fatalf("want one repeated outcome, got %+v", d.RepeatedOutcomes)
	}
	top := d.RepeatedOutcomes[0]
	if top.Tool != "create_goal" || top.Count != 3 || top.LongestRun != 3 {
		t.Fatalf("wrong repeated outcome: %+v", top)
	}
	if top.TokenTotal != 16836+16825+16840 || top.TokenEvents != 3 {
		t.Fatalf("wrong token fold for repeated failure: %+v", top)
	}
	if top.ArgsDigestCount != 2 {
		t.Fatalf("repeated output should preserve bounded arg-digest diversity, got %+v", top)
	}
	if len(d.LivelockNotices) != 1 || d.LivelockNotices[0].MaxRepeat != 3 {
		t.Fatalf("missing livelock notice: %+v", d.LivelockNotices)
	}
	if strings.Contains(stdout.String(), "Make GLM") || strings.Contains(stdout.String(), "pure fak kernel") {
		t.Fatalf("diagnosis leaked raw tool arguments:\n%s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = runSessions(&stdout, &stderr, []string{"codex-loop", "--path", path})
	if code != 0 {
		t.Fatalf("codex-loop human exited %d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{
		"verdict        : LOOP",
		"repeated tool outcomes",
		"create_goal",
		"cannot create a new goal",
		"livelock notices",
		"launch future Codex sessions through `fak codex` or `fak guard -- codex`",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("human render missing %q:\n%s", want, stdout.String())
		}
	}

	stdout.Reset()
	stderr.Reset()
	code = runSessions(&stdout, &stderr, []string{"codex-loop", "--path", path, "--json", "--fail-on", "loop"})
	if code != 1 {
		t.Fatalf("codex-loop --fail-on loop exited %d, want 1 stderr=%s", code, stderr.String())
	}
	if err := json.Unmarshal(stdout.Bytes(), &d); err != nil {
		t.Fatalf("fail-on JSON did not decode: %v\n%s", err, stdout.String())
	}
	if d.Verdict != "LOOP" || !strings.Contains(stderr.String(), "gate REFUSE fail-on=loop verdict=LOOP") {
		t.Fatalf("fail-on gate did not preserve LOOP diagnosis/stderr: diagnosis=%+v stderr=%s", d, stderr.String())
	}

	if err := writeCodexGuardWitness(dir, "019f3417-0c8a-7da0-bf38-4ee9fd2354e4"); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	code = runSessions(&stdout, &stderr, []string{"codex-loop", "--path", path, "--codex-home", dir, "--json", "--fail-on", "unguarded"})
	if code != 0 {
		t.Fatalf("guarded fak-provider session failed --fail-on unguarded: exit=%d stderr=%s", code, stderr.String())
	}
}

func TestSessionsCodexLoopPlainDiagnosisUsesGuardWitness(t *testing.T) {
	home := t.TempDir()
	sessionID := "88888888-8888-4888-8888-888888888888"
	path := filepath.Join(home, "rollout.jsonl")
	writeCodexLoopFixture(t, path, []string{
		`{"timestamp":"2026-07-14T18:00:00Z","type":"session_meta","payload":{"id":"` + sessionID + `","model_provider":"fak"}}`,
		`{"timestamp":"2026-07-14T18:00:01Z","type":"event_msg","payload":{"type":"user_message","message":"inspect"}}`,
	})

	var stdout, stderr bytes.Buffer
	if code := sessionsCodexLoop(&stdout, &stderr, []string{"--path", path, "--codex-home", home, "--json", "--fail-on", "unguarded"}); code != 1 {
		t.Fatalf("unguarded code=%d want 1 stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if err := writeCodexGuardWitness(home, sessionID); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := sessionsCodexLoop(&stdout, &stderr, []string{"--path", path, "--codex-home", home, "--json", "--fail-on", "unguarded"}); code != 0 {
		t.Fatalf("guarded code=%d want 0 stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	var after codexLoopDiagnosis
	if err := json.Unmarshal(stdout.Bytes(), &after); err != nil {
		t.Fatalf("decode guarded diagnosis: %v\n%s", err, stdout.String())
	}
	if !after.GuardWitnessed {
		t.Fatalf("guarded diagnosis omitted durable witness: %+v", after)
	}
}
func TestSessionsCodexLoopDefaultsToCurrentCodexThread(t *testing.T) {
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionsDir := filepath.Join(home, "sessions", "2026", "07", "06")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	threadID := "019f3540-52dd-7001-b559-2818dc14ede6"
	t.Setenv("CODEX_THREAD_ID", threadID)
	path := filepath.Join(sessionsDir, "rollout-2026-07-06T09-30-00-"+threadID+".jsonl")
	writeCodexLoopFixture(t, path, []string{
		`{"timestamp":"2026-07-06T16:30:00.000Z","type":"session_meta","payload":{"session_id":"019f3540-52dd-7001-b559-2818dc14ede6","originator":"codex-tui","cli_version":"0.142.5","model_provider":"openai","git":{"commit_hash":"111ff04","branch":"main"}}}`,
		`{"timestamp":"2026-07-06T16:30:02.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"{\"command\":\"git status --short\"}","call_id":"shell_1"}}`,
		`{"timestamp":"2026-07-06T16:30:03.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"shell_1","output":"## main"}}`,
	})

	var stdout, stderr bytes.Buffer
	code := runSessions(&stdout, &stderr, []string{"codex-loop", "--codex-home", home, "--json", "--fail-on", "unguarded"})
	if code != 1 {
		t.Fatalf("codex-loop default current thread exited %d, want 1 stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var d codexLoopDiagnosis
	if err := json.Unmarshal(stdout.Bytes(), &d); err != nil {
		t.Fatalf("json did not decode: %v\n%s", err, stdout.String())
	}
	if d.SessionID != threadID || filepath.Clean(d.Path) != filepath.Clean(path) {
		t.Fatalf("current-thread resolver picked wrong session: got id=%q path=%q want id=%q path=%q", d.SessionID, d.Path, threadID, path)
	}
	if d.Verdict != "OK" || d.ModelProvider != "openai" {
		t.Fatalf("diagnosis = verdict=%s provider=%s, want OK/openai: %+v", d.Verdict, d.ModelProvider, d)
	}
	if !strings.Contains(stderr.String(), "gate REFUSE fail-on=unguarded verdict=OK reason=codex_session_bypassed_fak_guard") {
		t.Fatalf("unguarded current-thread gate did not name the direct-provider reason:\n%s", stderr.String())
	}
}

func TestSessionsCodexLoopKeepsSubagentIdentityAndCustomToolTraffic(t *testing.T) {
	dir := t.TempDir()
	childID := "019f4c7b-443c-7d80-a9a5-4f15534f99cf"
	parentID := "019f4c6e-14cf-7f81-a820-9f1bb70523c7"
	path := filepath.Join(dir, "rollout-2026-07-10T07-42-54-"+childID+".jsonl")
	writeCodexLoopFixture(t, path, []string{
		`{"timestamp":"2026-07-10T14:42:56.000Z","type":"session_meta","payload":{"id":"019f4c7b-443c-7d80-a9a5-4f15534f99cf","session_id":"019f4c6e-14cf-7f81-a820-9f1bb70523c7","originator":"codex-tui","cli_version":"0.142.5","model_provider":"openai","source":{"subagent":{"thread_spawn":{}}},"git":{"commit_hash":"761899e","branch":"main"}}}`,
		`{"timestamp":"2026-07-10T14:42:56.000Z","type":"session_meta","payload":{"id":"019f4c6e-14cf-7f81-a820-9f1bb70523c7","session_id":"019f4c6e-14cf-7f81-a820-9f1bb70523c7","originator":"codex-tui","cli_version":"0.142.5","model_provider":"openai","source":"cli","git":{"commit_hash":"761899e","branch":"main"}}}`,
		`{"timestamp":"2026-07-10T14:43:05.000Z","type":"response_item","payload":{"type":"custom_tool_call","name":"exec","input":"{\"command\":\"first\"}","call_id":"exec_1"}}`,
		`{"timestamp":"2026-07-10T14:43:06.000Z","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"exec_1","output":[{"type":"input_text","text":"same result"}]}}`,
		`{"timestamp":"2026-07-10T14:43:15.000Z","type":"response_item","payload":{"type":"custom_tool_call","name":"exec","input":"{\"command\":\"second\"}","call_id":"exec_2"}}`,
		`{"timestamp":"2026-07-10T14:43:16.000Z","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"exec_2","output":[{"type":"input_text","text":"same result"}]}}`,
		`{"timestamp":"2026-07-10T14:43:25.000Z","type":"response_item","payload":{"type":"custom_tool_call","name":"exec","input":"{\"command\":\"third\"}","call_id":"exec_3"}}`,
		`{"timestamp":"2026-07-10T14:43:26.000Z","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"exec_3","output":[{"type":"input_text","text":"same result"}]}}`,
	})

	fh, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	d, diagnoseErr := diagnoseCodexLoop(fh, path)
	closeErr := fh.Close()
	if diagnoseErr != nil {
		t.Fatal(diagnoseErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	if d.SessionID != childID {
		t.Fatalf("subagent identity = %q, want rollout id %q rather than parent %q", d.SessionID, childID, parentID)
	}
	if d.ParentSessionID != parentID {
		t.Fatalf("subagent parent identity = %q, want %q", d.ParentSessionID, parentID)
	}
	if d.ToolCalls != 3 || d.ToolOutputs != 3 {
		t.Fatalf("custom tool traffic = calls=%d outputs=%d, want 3/3", d.ToolCalls, d.ToolOutputs)
	}
	if d.Verdict != "LOOP" || len(d.RepeatedOutcomes) != 1 || d.RepeatedOutcomes[0].Tool != "exec" || d.RepeatedOutcomes[0].Count != 3 {
		t.Fatalf("custom tool repeat was not diagnosed: %+v", d)
	}
}

func TestRecentCodexLoopTreatsSpoofedFakProviderAsUnguarded(t *testing.T) {
	home := t.TempDir()
	sessionID := "019f4350-52dd-7001-b559-2818dc14ede6"
	dir := filepath.Join(home, "sessions", "2026", "07", "11")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "rollout-2026-07-11T10-00-00-"+sessionID+".jsonl")
	writeCodexLoopFixture(t, path, []string{`{"timestamp":"2026-07-11T17:00:00Z","type":"session_meta","payload":{"id":"` + sessionID + `","model_provider":"fak"}}`})
	rep, err := diagnoseRecentCodexLoops(home, 0, 10)
	if err != nil {
		t.Fatal(err)
	}
	if rep.UnguardedCount != 1 || len(rep.Diagnoses) != 1 || rep.Diagnoses[0].GuardWitnessed {
		t.Fatalf("spoofed provider fold: %+v diagnoses=%+v", rep, rep.Diagnoses)
	}
	if err := writeCodexGuardWitness(home, sessionID); err != nil {
		t.Fatal(err)
	}
	rep, err = diagnoseRecentCodexLoops(home, 0, 10)
	if err != nil {
		t.Fatal(err)
	}
	if rep.UnguardedCount != 0 || !rep.Diagnoses[0].GuardWitnessed {
		t.Fatalf("witnessed provider fold: %+v diagnoses=%+v", rep, rep.Diagnoses)
	}
}

func TestSessionsCodexLoopRecentScansCodexHome(t *testing.T) {
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionsDir := filepath.Join(home, "sessions", "2026", "07", "05")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	loopPath := filepath.Join(sessionsDir, "rollout-2026-07-05T19-24-43-loop.jsonl")
	writeCodexLoopFixture(t, loopPath, []string{
		`{"timestamp":"2026-07-06T02:24:43.315Z","type":"session_meta","payload":{"session_id":"loop-session","originator":"codex-tui","cli_version":"0.142.5","model_provider":"openai","git":{"commit_hash":"4926739","branch":"main"}}}`,
		// Same plan re-submitted each turn (ArgsDigestCount==1 < Count): a
		// genuine no-progress loop the recent scan must still flag LOOP.
		`{"timestamp":"2026-07-06T02:25:03.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_1"}}`,
		`{"timestamp":"2026-07-06T02:25:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_1","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:04.100Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":1000,"output_tokens":20,"total_tokens":1020},"last_token_usage":{"input_tokens":1000,"output_tokens":20,"total_tokens":1020}}}}`,
		`{"timestamp":"2026-07-06T02:25:15.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_2"}}`,
		`{"timestamp":"2026-07-06T02:25:16.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_2","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:16.100Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":2000,"output_tokens":30,"total_tokens":2030},"last_token_usage":{"input_tokens":1000,"output_tokens":10,"total_tokens":1010}}}}`,
		`{"timestamp":"2026-07-06T02:25:27.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_3"}}`,
		`{"timestamp":"2026-07-06T02:25:28.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_3","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:28.100Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":3000,"output_tokens":40,"total_tokens":3040},"last_token_usage":{"input_tokens":1000,"output_tokens":10,"total_tokens":1010}}}}`,
	})
	okPath := filepath.Join(sessionsDir, "rollout-2026-07-05T19-25-43-ok.jsonl")
	writeCodexLoopFixture(t, okPath, []string{
		`{"timestamp":"2026-07-06T02:25:43.315Z","type":"session_meta","payload":{"session_id":"ok-session","originator":"codex-tui","cli_version":"0.142.5","model_provider":"openai","git":{"commit_hash":"4926739","branch":"main"}}}`,
		`{"timestamp":"2026-07-06T02:25:44.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"{\"command\":\"git status --short\"}","call_id":"shell_1"}}`,
		`{"timestamp":"2026-07-06T02:25:45.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"shell_1","output":"## main"}}`,
	})

	var stdout, stderr bytes.Buffer
	code := runSessions(&stdout, &stderr, []string{"codex-loop", "--recent", "--codex-home", home, "--limit", "2", "--json"})
	if code != 0 {
		t.Fatalf("codex-loop --recent --json exited %d stderr=%s", code, stderr.String())
	}
	var r codexLoopRecentReport
	if err := json.Unmarshal(stdout.Bytes(), &r); err != nil {
		t.Fatalf("json did not decode: %v\n%s", err, stdout.String())
	}
	if r.Verdict != "LOOP" || r.LoopCount != 1 || r.OKCount != 1 || r.Scanned != 2 {
		t.Fatalf("wrong recent report: %+v", r)
	}
	if r.ProviderCounts["openai"] != 2 || r.UnguardedCount != 2 {
		t.Fatalf("provider fold = counts=%v unguarded=%d, want openai=2 unguarded=2", r.ProviderCounts, r.UnguardedCount)
	}
	if r.GuardedLoopCount != 0 || r.UnguardedLoopCount != 1 || r.UnknownLoopCount != 0 {
		t.Fatalf("loop routes = guarded=%d direct=%d unknown=%d, want 0/1/0", r.GuardedLoopCount, r.UnguardedLoopCount, r.UnknownLoopCount)
	}
	if len(r.TopRepeated) != 1 || r.TopRepeated[0].Tool != "update_plan" || r.TopRepeated[0].Count != 3 {
		t.Fatalf("wrong top repeated fold: %+v", r.TopRepeated)
	}
	if len(r.Diagnoses) == 0 || !strings.Contains(r.Diagnoses[0].NextAction, "fak codex") {
		t.Fatalf("direct-provider diagnosis did not point at guarded launch path: %+v", r.Diagnoses)
	}
	if !strings.Contains(r.NextAction, "fak codex") {
		t.Fatalf("recent next_action = %q, want guarded launch guidance", r.NextAction)
	}
	if strings.Contains(stdout.String(), "step") {
		t.Fatalf("recent report leaked raw tool arguments:\n%s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = runSessions(&stdout, &stderr, []string{"codex-loop", "--recent", "--codex-home", home, "--limit", "2"})
	if code != 0 {
		t.Fatalf("codex-loop --recent human exited %d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{
		"fak sessions codex-loop --recent",
		"verdict        : LOOP",
		"LOOP=1 ACTION=0 OK=1",
		"providers      : openai=2 unguarded=2",
		"loop routes    : guarded=0 direct=1 unknown=0",
		"next action    : launch future Codex sessions through `fak codex`",
		"update_plan",
		"loop-session verdict=LOOP",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("human render missing %q:\n%s", want, stdout.String())
		}
	}

	stdout.Reset()
	stderr.Reset()
	code = runSessions(&stdout, &stderr, []string{"codex-loop", "--recent", "--codex-home", home, "--limit", "2", "--fail-on", "loop"})
	if code != 1 {
		t.Fatalf("codex-loop --recent --fail-on loop exited %d, want 1 stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "verdict        : LOOP") || !strings.Contains(stderr.String(), "gate REFUSE fail-on=loop verdict=LOOP") {
		t.Fatalf("recent fail-on did not emit diagnosis and gate refusal:\nstdout=%s\nstderr=%s", stdout.String(), stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = runSessions(&stdout, &stderr, []string{"codex-loop", "--recent", "--codex-home", home, "--limit", "2", "--fail-on", "unguarded"})
	if code != 1 {
		t.Fatalf("codex-loop --recent --fail-on unguarded exited %d, want 1 stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "providers      : openai=2 unguarded=2") ||
		!strings.Contains(stderr.String(), "gate REFUSE fail-on=unguarded verdict=LOOP reason=recent_codex_sessions_include_2_unguarded_provider_session(s)") {
		t.Fatalf("recent unguarded gate did not emit provider evidence:\nstdout=%s\nstderr=%s", stdout.String(), stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = runSessions(&stdout, &stderr, []string{"codex-loop", "--recent", "--codex-home", home, "--fail-on", "urgent"})
	if code != 2 || !strings.Contains(stderr.String(), "invalid --fail-on") {
		t.Fatalf("invalid fail-on exited %d stderr=%s", code, stderr.String())
	}
}

// A whitespace- or CR-polluted CODEX_HOME (setx / CRLF .env / trailing space) must
// still resolve like the sibling resolveCodexHome: whitespace-only falls back to
// ~/.codex, and a real dir with trailing space resolves to the trimmed dir — never a
// bogus relative "sessions" root that fails the live continuation hook open.
func TestResolvedCodexLoopHomeTrimsPollutedEnv(t *testing.T) {
	userHome := t.TempDir()
	t.Setenv("HOME", userHome)        // POSIX os.UserHomeDir
	t.Setenv("USERPROFILE", userHome) // Windows os.UserHomeDir
	wantFallback := filepath.Clean(filepath.Join(userHome, ".codex"))

	t.Run("whitespace-only env falls back to ~/.codex", func(t *testing.T) {
		t.Setenv("CODEX_HOME", "   ")
		got, err := resolvedCodexLoopHome("")
		if err != nil {
			t.Fatalf("resolvedCodexLoopHome: %v", err)
		}
		if got != wantFallback {
			t.Fatalf("polluted CODEX_HOME resolved to %q, want fallback %q", got, wantFallback)
		}
		if strings.TrimSpace(got) == "" || !filepath.IsAbs(got) {
			t.Fatalf("resolved home %q is not a usable absolute path", got)
		}
	})

	t.Run("real dir with trailing space resolves to the trimmed dir", func(t *testing.T) {
		real := filepath.Join(t.TempDir(), "codex-home")
		t.Setenv("CODEX_HOME", real+"  ")
		got, err := resolvedCodexLoopHome("")
		if err != nil {
			t.Fatalf("resolvedCodexLoopHome: %v", err)
		}
		if got != filepath.Clean(real) {
			t.Fatalf("padded CODEX_HOME resolved to %q, want %q", got, filepath.Clean(real))
		}
	})
}

func writeCodexLoopFixture(t *testing.T, path string, lines []string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}
