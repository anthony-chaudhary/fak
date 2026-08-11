package codexresume

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func helperCommand(t *testing.T, mode, rollout string) []string {
	t.Helper()
	return []string{os.Args[0], "-test.run=TestResumeHelperProcess", "--", mode, rollout}
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
	appendRow := func(typ, sub, reason string) {
		f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
		defer f.Close()
		_ = json.NewEncoder(f).Encode(map[string]any{"type": typ, "payload": map[string]any{"type": sub, "reason": reason}})
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
	cfg := Config{Command: helperCommand(t, mode, rollout), RolloutPath: rollout, Deadline: deadline, PollInterval: 20 * time.Millisecond, Drain: 30 * time.Millisecond, Env: append(os.Environ(), "GO_WANT_CODEXRESUME_HELPER=1")}
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
	r := runHelper(t, "completed-hung", time.Second)
	if r.Outcome != OutcomeCompletedReclaimed || !r.ForcedReclaim || !r.UsefulWork || !r.TaskCompleted {
		t.Fatalf("result=%+v", r)
	}
}
func TestRunTypesUpstreamInterrupted(t *testing.T) {
	r := runHelper(t, "interrupted", 120*time.Millisecond)
	if r.Outcome != OutcomeStalledBeforeTerminal || !r.Interrupted || !r.ForcedReclaim {
		t.Fatalf("result=%+v", r)
	}
}
func TestRunTypesPreTerminalStall(t *testing.T) {
	r := runHelper(t, "stalled", 120*time.Millisecond)
	if r.Outcome != OutcomeStalledBeforeTerminal || !r.ForcedReclaim || !r.TaskStarted {
		t.Fatalf("result=%+v", r)
	}
}
func TestRunRequiresInputs(t *testing.T) {
	_, e := Run(context.Background(), Config{})
	if e == nil {
		t.Fatal("expected error")
	}
	_ = fmt.Sprint(e)
}
