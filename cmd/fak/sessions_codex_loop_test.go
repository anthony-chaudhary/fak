package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

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
		"hard-fuse repeated unfinished-goal failures",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("human render missing %q:\n%s", want, stdout.String())
		}
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
		`{"timestamp":"2026-07-06T02:25:03.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_1"}}`,
		`{"timestamp":"2026-07-06T02:25:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_1","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:04.100Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":1000,"output_tokens":20,"total_tokens":1020},"last_token_usage":{"input_tokens":1000,"output_tokens":20,"total_tokens":1020}}}}`,
		`{"timestamp":"2026-07-06T02:25:15.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"two\",\"status\":\"in_progress\"}]}","call_id":"plan_2"}}`,
		`{"timestamp":"2026-07-06T02:25:16.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_2","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:16.100Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":2000,"output_tokens":30,"total_tokens":2030},"last_token_usage":{"input_tokens":1000,"output_tokens":10,"total_tokens":1010}}}}`,
		`{"timestamp":"2026-07-06T02:25:27.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"three\",\"status\":\"in_progress\"}]}","call_id":"plan_3"}}`,
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
	if len(r.TopRepeated) != 1 || r.TopRepeated[0].Tool != "update_plan" || r.TopRepeated[0].Count != 3 {
		t.Fatalf("wrong top repeated fold: %+v", r.TopRepeated)
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
		"update_plan",
		"loop-session verdict=LOOP",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("human render missing %q:\n%s", want, stdout.String())
		}
	}
}

func writeCodexLoopFixture(t *testing.T, path string, lines []string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}
