package codexlifecycle

// #4767 fixtures: expected MERGE_HEAD absence, wait control exit, genuine command
// failure, timeout, missing output, interrupted task, and a completed multi-hour
// task with idle gaps. Every fixture is shape-matched to the live rollout corpus
// (function_call / function_call_output / token_count / task lifecycle records).

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// callLine renders a real shell function_call record.
func callLine(ts, callID, tool, command string) string {
	args, _ := json.Marshal(map[string]string{"command": command})
	payload, _ := json.Marshal(map[string]any{
		"type": "function_call", "name": tool, "call_id": callID, "arguments": string(args),
	})
	return `{"timestamp":"` + ts + `","type":"response_item","payload":` + string(payload) + `}`
}

// patchCallLine renders an apply_patch call carrying file targets in its input.
func patchCallLine(ts, callID, file string) string {
	patch := "*** Begin Patch\n*** Update File: " + file + "\n+x\n*** End Patch"
	args, _ := json.Marshal(map[string]string{"input": patch})
	payload, _ := json.Marshal(map[string]any{
		"type": "function_call", "name": "apply_patch", "call_id": callID, "arguments": string(args),
	})
	return `{"timestamp":"` + ts + `","type":"response_item","payload":` + string(payload) + `}`
}

// outLine renders the harness text envelope observed in the live store.
func outLine(ts, callID string, exit int, wallS float64, body string) string {
	env := "Exit code: " + itoa(exit) + "\nWall time: " + ftoa(wallS) + " seconds\nOutput:\n" + body
	payload, _ := json.Marshal(map[string]any{
		"type": "function_call_output", "call_id": callID, "output": env,
	})
	return `{"timestamp":"` + ts + `","type":"response_item","payload":` + string(payload) + `}`
}

func rawOutLine(ts, callID, output string) string {
	payload, _ := json.Marshal(map[string]any{
		"type": "function_call_output", "call_id": callID, "output": output,
	})
	return `{"timestamp":"` + ts + `","type":"response_item","payload":` + string(payload) + `}`
}

func tokensLine(ts string) string {
	return `{"timestamp":"` + ts + `","type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":100,"output_tokens":10}}}}`
}

// completeWithDur renders a task_complete carrying the producer-recorded duration —
// the "recorded task duration" the issue's distribution is built from.
func completeWithDur(ts, id string, durMS int64) string {
	return `{"timestamp":"` + ts + `","type":"event_msg","payload":{"type":"task_complete","turn_id":"` + id + `","last_agent_message":null,"duration_ms":` + itoa(int(durMS)) + `}}`
}

func itoa(n int) string     { b, _ := json.Marshal(n); return string(b) }
func ftoa(f float64) string { b, _ := json.Marshal(f); return string(b) }

func analyzeLines(t *testing.T, fresh bool, lines ...string) RolloutAnalytics {
	t.Helper()
	meta, recs, err := ReadAnalyticsRollout(strings.NewReader(strings.Join(lines, "\n") + "\n"))
	if err != nil {
		t.Fatalf("ReadAnalyticsRollout: %v", err)
	}
	return AnalyzeRollout(meta, recs, fresh)
}

func onlyOutcome(t *testing.T, ra RolloutAnalytics) CallOutcome {
	t.Helper()
	if len(ra.Outcomes) != 1 {
		t.Fatalf("outcomes = %d, want 1: %+v", len(ra.Outcomes), ra.Outcomes)
	}
	return ra.Outcomes[0]
}

// --- Envelope decoding: structured first, text second, malformed typed, opaque assumed ---

func TestDecodeEnvelope(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   string
		want Envelope
	}{
		{"text_ok", "Exit code: 0\nWall time: 0.6 seconds\nOutput:\nfine",
			Envelope{Kind: EnvelopeText, HasExit: true, ExitCode: 0, WallS: 0.6}},
		{"text_fail", "Exit code: 1\nWall time: 0.8 seconds\nOutput:\nboom",
			Envelope{Kind: EnvelopeText, HasExit: true, ExitCode: 1, WallS: 0.8}},
		{"text_timeout_124", "Exit code: 124\nWall time: 30.2 seconds\nOutput:\ncommand timed out after 30228 milliseconds\nrest",
			Envelope{Kind: EnvelopeText, HasExit: true, ExitCode: 124, WallS: 30.2, TimedOut: true}},
		{"structured_metadata", `{"output":"done","metadata":{"exit_code":2,"duration_seconds":1.5}}`,
			Envelope{Kind: EnvelopeStructured, HasExit: true, ExitCode: 2, WallS: 1.5}},
		{"structured_no_exit", `{"goal":{"status":"active"}}`,
			Envelope{Kind: EnvelopeStructured}},
		{"malformed_json", `{"oh no`,
			Envelope{Kind: EnvelopeMalformed}},
		{"malformed_exit_header", "Exit code: broken\nrest",
			Envelope{Kind: EnvelopeMalformed}},
		{"opaque", "Plan updated",
			Envelope{Kind: EnvelopeOpaque}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := DecodeEnvelope(tc.in); got != tc.want {
				t.Errorf("DecodeEnvelope = %+v, want %+v", got, tc.want)
			}
		})
	}
}

// Result CONTENT mentioning "timed out" must not fake a timeout: only the first
// body line of the harness envelope is consulted.
func TestDecodeEnvelope_ContentCannotFakeTimeout(t *testing.T) {
	env := DecodeEnvelope("Exit code: 0\nWall time: 1.9 seconds\nOutput:\n#150 apply: orchestrator timeout burns 41 hours\nsecond line timed out after everything")
	if env.TimedOut {
		t.Error("body content below the first line must not mark the call timed out")
	}
}

// --- FIXTURE: expected MERGE_HEAD absence is an answer, not a failure ---

func TestClassify_MergeHeadProbeIsExpectedNegative(t *testing.T) {
	ra := analyzeLines(t, false,
		meta("s", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T10:00:00.000Z", "A"),
		callLine("2026-07-16T10:00:01.000Z", "c1", "shell_command", "git rev-parse -q --verify MERGE_HEAD"),
		outLine("2026-07-16T10:00:02.000Z", "c1", 1, 0.3, ""),
		complete("2026-07-16T10:00:03.000Z", "A"))
	o := onlyOutcome(t, ra)
	if o.Class != ToolExpectedNegative || o.Reason != "merge_head_probe" {
		t.Errorf("class/reason = %s/%s, want expected_negative/merge_head_probe", o.Class, o.Reason)
	}
	if o.Class.CountsAsFailure() {
		t.Error("an expected negative must never count as a failure")
	}
	if o.Confidence != ConfidenceObserved {
		t.Errorf("confidence = %s, want observed (the envelope was decoded)", o.Confidence)
	}
}

// grep-family probes are EXIT-GATED: exit 1 is the documented "no matches"
// answer, exit 2 is a genuine grep error and stays a failure.
func TestClassify_GrepNoMatchIsExitGated(t *testing.T) {
	env1 := Envelope{Kind: EnvelopeText, HasExit: true, ExitCode: 1}
	if class, reason, _ := ClassifyOutcome(`rg -n "needle" internal/`, env1); class != ToolExpectedNegative || reason != "grep_no_match" {
		t.Errorf("rg exit 1 = %s/%s, want expected_negative/grep_no_match", class, reason)
	}
	if class, _, _ := ClassifyOutcome(`git grep -n needle`, env1); class != ToolExpectedNegative {
		t.Errorf("git grep exit 1 = %s, want expected_negative", class)
	}
	env2 := Envelope{Kind: EnvelopeText, HasExit: true, ExitCode: 2}
	if class, reason, _ := ClassifyOutcome(`rg -n "needle" missing-dir/`, env2); class != ToolFailure || reason != "exit_2" {
		t.Errorf("rg exit 2 = %s/%s, want failure/exit_2 (a real grep error)", class, reason)
	}
	// Piping THROUGH a matcher does not qualify: the probe is segment-start only.
	if class, _, _ := ClassifyOutcome(`fak sweep | grep -c done`, env1); class != ToolFailure {
		t.Errorf("pipe-through grep exit 1 = %s, want failure (not a registered probe)", class)
	}
}

// --- FIXTURE: `wait` exit-1 is a control outcome, not a failure ---

func TestClassify_WaitControlExit(t *testing.T) {
	ra := analyzeLines(t, false,
		meta("s", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T10:00:00.000Z", "A"),
		callLine("2026-07-16T10:00:01.000Z", "c1", "shell_command", "wait 4242"),
		outLine("2026-07-16T10:03:00.000Z", "c1", 1, 178.0, ""),
		complete("2026-07-16T10:03:01.000Z", "A"))
	o := onlyOutcome(t, ra)
	if o.Class != ToolControlExit || o.Reason != "wait_control_exit" {
		t.Errorf("class/reason = %s/%s, want control_exit/wait_control_exit", o.Class, o.Reason)
	}
	if o.Class.CountsAsFailure() {
		t.Error("a wait control exit must never count as a failure")
	}
	// A blocking wait's span lands in the WAIT bucket, not tool work.
	task := ra.Tasks[0]
	if task.WaitMS == 0 || task.ToolMS != 0 {
		t.Errorf("wait/tool = %d/%d ms, want the span booked as wait only", task.WaitMS, task.ToolMS)
	}
}

// The control probe is a REGISTERED shape, not a substring sweep: a wait builtin
// at a segment start is control flow; a command merely containing "wait" is not.
func TestControlExitProbe_SegmentStartOnly(t *testing.T) {
	for _, tc := range []struct {
		cmd  string
		want bool
	}{
		{"wait 4242", true},
		{"wait", true},
		{"sleep 5; wait 99", true},
		{"kick_off.sh && wait", true},
		{".\\dgxbridge.exe -probe -probe-wait 90s doctor", false},
		{"python watch.py --max-wait-s 0", false},
		{"Wait-Process -Id 41540 -Timeout 30", false},
		{`rg -n "asyncio.wait_for" agents/run.py`, false},
	} {
		got := controlExitProbes[0].Re.MatchString(tc.cmd)
		if got != tc.want {
			t.Errorf("wait probe on %q = %v, want %v", tc.cmd, got, tc.want)
		}
	}
}

// --- FIXTURE: a genuine command failure stays a failure, with a stable reason ---

func TestClassify_GenuineFailure(t *testing.T) {
	ra := analyzeLines(t, false,
		meta("s", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T10:00:00.000Z", "A"),
		callLine("2026-07-16T10:00:01.000Z", "c1", "shell_command", "go test ./internal/broken"),
		outLine("2026-07-16T10:00:09.000Z", "c1", 1, 7.5, "FAIL"),
		complete("2026-07-16T10:00:10.000Z", "A"))
	o := onlyOutcome(t, ra)
	if o.Class != ToolFailure || o.Reason != "exit_1" {
		t.Errorf("class/reason = %s/%s, want failure/exit_1", o.Class, o.Reason)
	}
	if !o.Class.CountsAsFailure() {
		t.Error("a genuine failure must count as a failure")
	}
}

// --- FIXTURE: harness timeout with partial state ---

func TestClassify_Timeout(t *testing.T) {
	ra := analyzeLines(t, false,
		meta("s", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T10:00:00.000Z", "A"),
		callLine("2026-07-16T10:00:01.000Z", "c1", "shell_command", "python slow.py"),
		outLine("2026-07-16T10:00:32.000Z", "c1", 124, 30.2, "command timed out after 30228 milliseconds\npartial output"),
		complete("2026-07-16T10:00:33.000Z", "A"))
	o := onlyOutcome(t, ra)
	if o.Class != ToolTimeout || o.Reason != "timeout" {
		t.Errorf("class/reason = %s/%s, want timeout/timeout", o.Class, o.Reason)
	}
	if ra.Behavior.TimeoutKills != 1 {
		t.Errorf("timeout_kills = %d, want 1", ra.Behavior.TimeoutKills)
	}
}

// --- FIXTURE: missing output, typed by the reconciled task boundary ---

func TestMissingOutput_TypedByTaskBoundary(t *testing.T) {
	const (
		t0 = "2026-07-16T10:00:00.000Z"
		t1 = "2026-07-16T10:00:01.000Z"
		t2 = "2026-07-16T10:00:02.000Z"
		t3 = "2026-07-16T10:00:03.000Z"
		t4 = "2026-07-16T10:00:04.000Z"
	)
	t.Run("later_output_proves_missing", func(t *testing.T) {
		ra := analyzeLines(t, false,
			meta("s", "fak", "0.144.4", `C:\work\fak`),
			started(t0, "A"),
			callLine(t1, "c1", "shell_command", "echo one"), // output never arrives
			callLine(t2, "c2", "shell_command", "echo two"),
			outLine(t3, "c2", 0, 0.1, "two"),
			complete(t4, "A"))
		if got := ra.Outcomes[0]; got.Class != ToolMissingResult || got.Reason != "later_output_in_task" {
			t.Errorf("c1 = %s/%s, want missing_result/later_output_in_task", got.Class, got.Reason)
		}
	})
	t.Run("completed_task_without_result", func(t *testing.T) {
		ra := analyzeLines(t, false,
			meta("s", "fak", "0.144.4", `C:\work\fak`),
			started(t0, "A"),
			callLine(t1, "c1", "shell_command", "echo one"),
			complete(t2, "A"))
		o := onlyOutcome(t, ra)
		if o.Class != ToolMissingResult || o.Reason != "task_completed_without_result" {
			t.Errorf("class/reason = %s/%s, want missing_result/task_completed_without_result", o.Class, o.Reason)
		}
	})
	t.Run("interrupted_task_boundary", func(t *testing.T) {
		ra := analyzeLines(t, false,
			meta("s", "fak", "0.144.4", `C:\work\fak`),
			started(t0, "A"),
			callLine(t1, "c1", "shell_command", "echo one"),
			aborted(t2, "A"))
		o := onlyOutcome(t, ra)
		if o.Class != ToolInterrupted || o.Confidence != ConfidenceInferred {
			t.Errorf("class/conf = %s/%s, want interrupted/inferred", o.Class, o.Confidence)
		}
	})
	t.Run("superseded_task_boundary", func(t *testing.T) {
		ra := analyzeLines(t, false,
			meta("s", "fak", "0.144.4", `C:\work\fak`),
			started(t0, "A"),
			callLine(t1, "c1", "shell_command", "echo one"),
			started(t2, "B"), // supersedes A with c1 still open
			complete(t3, "B"))
		if got := ra.Outcomes[0]; got.Class != ToolInterrupted || got.Reason != "task_boundary_superseded" {
			t.Errorf("c1 = %s/%s, want interrupted/task_boundary_superseded", got.Class, got.Reason)
		}
	})
	t.Run("live_tail_stays_unknown", func(t *testing.T) {
		ra := analyzeLines(t, true, // FRESH rollout: the call may genuinely still run
			meta("s", "fak", "0.144.4", `C:\work\fak`),
			started(t0, "A"),
			callLine(t1, "c1", "shell_command", "go test ./..."))
		o := onlyOutcome(t, ra)
		if o.Class != ToolLiveTail || o.Confidence != ConfidenceInferred {
			t.Errorf("class/conf = %s/%s, want live_tail/inferred — never inferred success", o.Class, o.Confidence)
		}
	})
	t.Run("dead_tail_is_interrupted", func(t *testing.T) {
		ra := analyzeLines(t, false, // STALE rollout: the writer died
			meta("s", "fak", "0.144.4", `C:\work\fak`),
			started(t0, "A"),
			callLine(t1, "c1", "shell_command", "go test ./..."))
		o := onlyOutcome(t, ra)
		if o.Class != ToolInterrupted || o.Reason != "task_boundary_process_death" {
			t.Errorf("class/reason = %s/%s, want interrupted/task_boundary_process_death", o.Class, o.Reason)
		}
	})
}

// --- FIXTURE: a malformed envelope is typed, not counted as failure or success ---

func TestClassify_MalformedEnvelope(t *testing.T) {
	ra := analyzeLines(t, false,
		meta("s", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T10:00:00.000Z", "A"),
		callLine("2026-07-16T10:00:01.000Z", "c1", "shell_command", "echo hi"),
		rawOutLine("2026-07-16T10:00:02.000Z", "c1", `{"torn":"env`),
		complete("2026-07-16T10:00:03.000Z", "A"))
	o := onlyOutcome(t, ra)
	if o.Class != ToolMalformedEnvelope {
		t.Errorf("class = %s, want malformed_envelope", o.Class)
	}
	if o.Class.CountsAsFailure() {
		t.Error("a malformed envelope is an ingestion defect, not a command failure")
	}
}

// --- FIXTURE: completed multi-hour task with idle gaps ---

func TestMultiHourTask_IdleGapsAttributed(t *testing.T) {
	ra := analyzeLines(t, false,
		meta("s", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T10:00:00.000Z", "A"),
		tokensLine("2026-07-16T10:00:05.000Z"), // first token: TTFT = 5s
		callLine("2026-07-16T10:00:10.000Z", "c1", "shell_command", "make build"),
		outLine("2026-07-16T10:00:20.000Z", "c1", 0, 9.8, "ok"),
		// One hour of NOTHING, then the next call: 300s model cap + 3300s idle.
		callLine("2026-07-16T11:00:20.000Z", "c2", "shell_command", "make test"),
		outLine("2026-07-16T11:00:25.000Z", "c2", 0, 4.9, "ok"),
		// Ninety more minutes of nothing before the terminal.
		complete("2026-07-16T12:30:25.000Z", "A"))
	if len(ra.Tasks) != 1 {
		t.Fatalf("tasks = %d, want 1", len(ra.Tasks))
	}
	task := ra.Tasks[0]
	if task.Outcome != Complete {
		t.Fatalf("outcome = %s, want complete", task.Outcome)
	}
	if task.WallMS != int64((2*time.Hour + 30*time.Minute + 25*time.Second).Milliseconds()) {
		t.Errorf("wall_ms = %d, want the full 2h30m25s", task.WallMS)
	}
	if task.TTFTMS != 5000 {
		t.Errorf("ttft_ms = %d, want 5000", task.TTFTMS)
	}
	if task.ToolMS != 15000 { // 10s + 5s call spans
		t.Errorf("tool_ms = %d, want 15000", task.ToolMS)
	}
	if task.IdleGaps != 2 {
		t.Errorf("idle_gaps = %d, want 2 (the hour and the ninety minutes)", task.IdleGaps)
	}
	wantIdle := int64((time.Hour - 5*time.Minute).Milliseconds()) + int64((90*time.Minute - 5*time.Minute).Milliseconds())
	if task.IdleMS != wantIdle {
		t.Errorf("idle_ms = %d, want %d (gap minus the 300s model cap, twice)", task.IdleMS, wantIdle)
	}
	if len(task.Critical) == 0 || task.Critical[0].Category != "idle" {
		t.Errorf("critical = %+v, want idle ranked first", task.Critical)
	}
	// The buckets must not exceed the wall: the decomposition is conservative.
	if sum := task.ToolMS + task.WaitMS + task.ModelMS + task.IdleMS; sum > task.WallMS {
		t.Errorf("bucket sum %d exceeds wall %d", sum, task.WallMS)
	}
}

// --- Detectors ported from #2365 to Codex event shape ---

func TestDetectors_RepeatFailureAndSuccessLoop(t *testing.T) {
	lines := []string{
		meta("s", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T10:00:00.000Z", "A"),
	}
	ts := func(i int) string {
		return time.Date(2026, 7, 16, 10, 0, 1+i, 0, time.UTC).Format("2006-01-02T15:04:05.000Z")
	}
	n := 0
	for i := 0; i < 3; i++ { // identical failing call x3 => repeat-failure row
		lines = append(lines,
			callLine(ts(n), itoa(1000+n), "shell_command", "go test ./stuck"),
			outLine(ts(n+1), itoa(1000+n), 1, 0.5, "FAIL"))
		n += 2
	}
	for i := 0; i < 8; i++ { // identical successful call x8 => success-loop row
		lines = append(lines,
			callLine(ts(n), itoa(1000+n), "shell_command", "git status --porcelain"),
			outLine(ts(n+1), itoa(1000+n), 0, 0.2, ""))
		n += 2
	}
	lines = append(lines, complete(ts(n), "A"))
	ra := analyzeLines(t, false, lines...)
	if len(ra.Behavior.RepeatFailures) != 1 || ra.Behavior.RepeatFailures[0].Count != 3 {
		t.Errorf("repeat_failures = %+v, want one row of 3", ra.Behavior.RepeatFailures)
	}
	if len(ra.Behavior.SuccessLoops) != 1 || ra.Behavior.SuccessLoops[0].Count != 8 {
		t.Errorf("success_loops = %+v, want one row of 8", ra.Behavior.SuccessLoops)
	}
	// The rows are scrubbed: hashed signatures, never command text.
	if sig := ra.Behavior.RepeatFailures[0].Sig; strings.Contains(sig, "go test") || len(sig) != 8 {
		t.Errorf("sig = %q, want an 8-hex hash, never the command", sig)
	}
}

func TestDetectors_SleepPollsAndEditChurn(t *testing.T) {
	lines := []string{
		meta("s", "fak", "0.144.4", `C:\work\fak`),
		started("2026-07-16T10:00:00.000Z", "A"),
	}
	ts := func(i int) string {
		return time.Date(2026, 7, 16, 10, 0, 1+i, 0, time.UTC).Format("2006-01-02T15:04:05.000Z")
	}
	n := 0
	for i := 0; i < 2; i++ {
		lines = append(lines,
			callLine(ts(n), itoa(2000+n), "shell_command", "Start-Sleep 30"),
			outLine(ts(n+1), itoa(2000+n), 0, 30.0, ""))
		n += 2
	}
	for i := 0; i < 5; i++ { // same file patched 5x => churn row
		lines = append(lines,
			patchCallLine(ts(n), itoa(2000+n), "internal/foo/foo.go"),
			rawOutLine(ts(n+1), itoa(2000+n), "Done"))
		n += 2
	}
	lines = append(lines, complete(ts(n), "A"))
	ra := analyzeLines(t, false, lines...)
	if ra.Behavior.SleepPolls != 2 {
		t.Errorf("sleep_polls = %d, want 2", ra.Behavior.SleepPolls)
	}
	if len(ra.Behavior.EditChurn) != 1 || ra.Behavior.EditChurn[0].File != "internal/foo/foo.go" || ra.Behavior.EditChurn[0].Count != 5 {
		t.Errorf("edit_churn = %+v, want internal/foo/foo.go x5", ra.Behavior.EditChurn)
	}
}

func TestPercentiles_NearestRank(t *testing.T) {
	vals := make([]float64, 100)
	for i := range vals {
		vals[i] = float64(i + 1) // 1..100
	}
	p := percentiles(vals)
	if p.P50 != 50 || p.P90 != 90 || p.P95 != 95 || p.P99 != 99 || p.Max != 100 || p.N != 100 {
		t.Errorf("percentiles = %+v, want 50/90/95/99/100 over n=100", p)
	}
	if z := percentiles(nil); z.N != 0 || z.Max != 0 {
		t.Errorf("empty percentiles = %+v, want zeros", z)
	}
}
