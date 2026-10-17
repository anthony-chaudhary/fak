package codexresume

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms
func TestChildPWDProcess(t *testing.T) {
	args := os.Args
	if len(args) < 3 || args[len(args)-2] != "--child-pwd" {
		return
	}
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	_, hasOld := os.LookupEnv("OLDPWD")
	if err := json.NewEncoder(os.Stdout).Encode(map[string]any{
		"cwd": cwd, "pwd": os.Getenv("PWD"), "oldpwd": os.Getenv("OLDPWD"), "has_oldpwd": hasOld,
		"inherited": os.Getenv("CHILD_PWD_INHERITED"), "custom": os.Getenv("CHILD_PWD_CUSTOM"),
	}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	f, err := os.OpenFile(args[len(args)-1], os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	// Run requires a post-launch start witness before accepting completion.
	_, err = fmt.Fprintln(f, `{"type":"event_msg","payload":{"type":"task_started","turn_id":"pwd-helper"}}`+"\n"+
		`{"type":"event_msg","payload":{"type":"task_complete","turn_id":"pwd-helper"}}`)
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		fmt.Fprintln(os.Stderr, err, closeErr)
		os.Exit(2)
	}
	os.Exit(0)
}

// fak-test:runtime medium est=5s
func TestChildPWDRun(t *testing.T) {
	t.Setenv("PWD", "stale-launcher")
	t.Setenv("OLDPWD", "older-launcher")
	t.Setenv("CHILD_PWD_INHERITED", "parent")
	parent, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	child := t.TempDir()
	relative, err := filepath.Rel(parent, child)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, dir string
		env       []string
	}{
		{"inherited", child, nil},
		{"relative-custom", relative, []string{"PWD=wrong", "OLDPWD=wrong-old", "CHILD_PWD_CUSTOM=value"}},
		{"empty", child, []string{}},
		{"no-directory", "", []string{"PWD=untouched", "OLDPWD=untouched-old", "CHILD_PWD_CUSTOM=value"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rollout := filepath.Join(t.TempDir(), "rollout.jsonl")
			if err := os.WriteFile(rollout, nil, 0600); err != nil {
				t.Fatal(err)
			}
			var stdout, stderr bytes.Buffer
			r, err := Run(context.Background(), Config{
				Command: []string{os.Args[0], "-test.run=^TestChildPWDProcess$", "--", "--child-pwd", rollout},
				Dir:     tc.dir, Env: tc.env, RolloutPath: rollout, Deadline: 10 * time.Second,
				PollInterval: 10 * time.Millisecond, Drain: 2 * time.Second, Stdout: &stdout, Stderr: &stderr,
			})
			if err != nil || r.Outcome != OutcomeCompleted || !r.ProcessExit || r.ExitCode != 0 || r.ForcedReclaim || !r.TaskStarted || !r.TaskCompleted {
				t.Fatalf("result=%+v err=%v stderr=%s", r, err, stderr.String())
			}
			var got struct {
				CWD, PWD, OLDPWD, Inherited, Custom string
				HasOld                              bool `json:"has_oldpwd"`
			}
			if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
				t.Fatalf("child report: %v: %s", err, stdout.String())
			}
			wantCWD, wantPWD, wantOld := child, child, ""
			if tc.dir == "" {
				wantCWD, wantPWD, wantOld = parent, "untouched", "untouched-old"
			}
			actualInfo, actualErr := os.Stat(got.CWD)
			wantInfo, wantErr := os.Stat(wantCWD)
			if actualErr != nil || wantErr != nil || !os.SameFile(actualInfo, wantInfo) {
				t.Fatalf("actual cwd=%q want=%q errors=%v/%v", got.CWD, wantCWD, actualErr, wantErr)
			}
			if got.PWD != wantPWD || got.OLDPWD != wantOld || got.HasOld != (tc.dir == "") {
				t.Fatalf("child directory environment=%+v want PWD=%q OLDPWD=%q", got, wantPWD, wantOld)
			}
			wantInherited, wantCustom := "", ""
			if tc.env == nil {
				wantInherited = "parent"
			} else if tc.name != "empty" {
				wantCustom = "value"
			}
			if got.Inherited != wantInherited || got.Custom != wantCustom {
				t.Fatalf("child environment isolation=%+v", got)
			}
		})
	}
}

func helperCommand(t *testing.T, mode, rollout string) []string {
	t.Helper()
	return []string{os.Args[0], "-test.run=TestResumeHelperProcess", "--", mode, rollout}
}

// The race runtime otherwise holds the helper open for one second after main
// exits, which tests instrumentation shutdown rather than Run's drain boundary.
func helperEnv() []string {
	return append(os.Environ(), "GO_WANT_CODEXRESUME_HELPER=1", "GORACE=atexit_sleep_ms=0")
}

func TestResumeHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_CODEXRESUME_HELPER") != "1" {
		return
	}
	args := os.Args
	sep := 0
	for i, a := range args {
		if a == "--" {
			sep = i
			break
		}
	}
	mode, path := args[sep+1], args[sep+2]
	appendPayload := func(typ string, payload map[string]any) {
		f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
		defer f.Close()
		_ = json.NewEncoder(f).Encode(map[string]any{"type": typ, "payload": payload})
	}
	appendRow := func(typ, sub, reason string) {
		appendPayload(typ, map[string]any{"type": sub, "reason": reason, "turn_id": "turn-helper"})
	}
	appendRow("event_msg", "task_started", "")
	switch mode {
	case "exit":
		appendRow("event_msg", "task_complete", "")
		os.Exit(0)
	case "completed-hung":
		appendRow("response_item", "custom_tool_call_output", "")
		appendRow("event_msg", "task_complete", "")
		time.Sleep(time.Hour)
	case "interrupted":
		appendRow("event_msg", "turn_aborted", "interrupted")
		time.Sleep(time.Hour)
	case "failed-hung":
		appendPayload("event_msg", map[string]any{
			"type":    "task_complete",
			"turn_id": "turn-helper",
			"error": map[string]any{
				"message": `{"type":"error","error":{"type":"invalid_request_error","message":"Invalid input[3].id: call_legacy. Expected an ID that begins with fc."},"status":400}`,
			},
		})
		time.Sleep(time.Hour)
	case "stalled":
		time.Sleep(time.Hour)
	}
	os.Exit(3)
}

func runHelper(t *testing.T, mode string, deadline time.Duration) Result {
	t.Helper()
	dir := t.TempDir()
	rollout := filepath.Join(dir, "rollout.jsonl")
	if err := os.WriteFile(rollout, []byte("seed\n"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := Config{Command: helperCommand(t, mode, rollout), RolloutPath: rollout, Deadline: deadline, PollInterval: 20 * time.Millisecond, Drain: 30 * time.Millisecond, Env: helperEnv()}
	r, err := Run(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestRunNormalExit(t *testing.T) {
	r := runHelper(t, "exit", time.Second)
	if r.Outcome != OutcomeCompleted || !r.ProcessExit || !r.TaskCompleted {
		t.Fatalf("result=%+v", r)
	}
}
func TestRunReclaimsCompletedHungProcess(t *testing.T) {
	r := runHelper(t, "completed-hung", 10*time.Second)
	if r.Outcome != OutcomeCompletedReclaimed || !r.ForcedReclaim || !r.UsefulWork || !r.TaskCompleted {
		t.Fatalf("result=%+v", r)
	}
}
func TestRunTypesUpstreamInterrupted(t *testing.T) {
	r := runHelper(t, "interrupted", 10*time.Second)
	if r.Outcome != OutcomeUpstreamInterrupted || !r.Interrupted || !r.ForcedReclaim {
		t.Fatalf("result=%+v", r)
	}
}
func TestRunTypesPreTerminalStall(t *testing.T) {
	r := runHelper(t, "stalled", 10*time.Second)
	if r.Outcome != OutcomeStalledBeforeTerminal || !r.ForcedReclaim || !r.TaskStarted {
		t.Fatalf("result=%+v", r)
	}
}

func TestRunDoesNotTreatFailedTaskCompleteAsSuccess(t *testing.T) {
	r := runHelper(t, "failed-hung", 10*time.Second)
	if r.Outcome != OutcomeTurnFailedReclaimed || !r.ForcedReclaim || r.TaskCompleted || r.TurnStatus != "failed" {
		t.Fatalf("result=%+v", r)
	}
	if r.TurnError == nil || r.TurnError.Status != 400 || r.TurnError.Type != "invalid_request_error" {
		t.Fatalf("turn error=%+v", r.TurnError)
	}
	const exact = "Invalid input[3].id: call_legacy. Expected an ID that begins with fc."
	if r.TurnError.Message != exact {
		t.Fatalf("message=%q want=%q", r.TurnError.Message, exact)
	}
}

func TestRunRequiresInputs(t *testing.T) {
	_, e := Run(context.Background(), Config{})
	if e == nil {
		t.Fatal("expected error")
	}
	_ = fmt.Sprint(e)
}

// Completion without a start in the observed suffix is not a successful task.
func TestScanRolloutRejectsOrphanCompletion(t *testing.T) {
	const start = `{"type":"event_msg","payload":{"type":"task_started","turn_id":"orphan-helper"}}` + "\n"
	const complete = `{"type":"event_msg","payload":{"type":"task_complete","turn_id":"orphan-helper"}}` + "\n"
	for _, tc := range []struct {
		name, contents string
		offset         int64
	}{
		{"no-start", complete, 0},
		{"start-before-baseline", start + complete, int64(len(start))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "rollout.jsonl")
			if err := os.WriteFile(path, []byte(tc.contents), 0600); err != nil {
				t.Fatal(err)
			}
			state, err := scanRollout(path, tc.offset)
			if err != nil {
				t.Fatal(err)
			}
			if state.started || state.completed || state.terminal {
				t.Fatalf("orphan completion accepted: %+v", state)
			}
			if got := outcomeFromState(state, false); got != OutcomeExited {
				t.Fatalf("outcome=%q want %q", got, OutcomeExited)
			}
		})
	}
}
