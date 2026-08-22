package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/codexresume"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

func TestWatchdogPlanRowCodexCoordinatesRoundTrip(t *testing.T) {
	want := resume.WatchdogPlanRow{Session: "thread-id", Harness: "codex", CWD: t.TempDir(), Rollout: "rollout.jsonl", GoalFile: "goal.txt", ResultFile: "result.json"}
	raw, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got resume.WatchdogPlanRow
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got=%+v want=%+v", got, want)
	}
}

func TestResumeBrokerSelectsCodexWithoutLeakingGoal(t *testing.T) {
	p := resume.WatchdogPlanRow{Session: "thread-id", Harness: "codex", CWD: t.TempDir(), Rollout: "rollout.jsonl", GoalFile: "goal.txt", ResultFile: "result.json"}
	got := rwResumeBrokerAttempt("fak-bin", "claude-bin", p, "claude-config", nil)
	if got.Backend != "codex" {
		t.Fatalf("backend=%q", got.Backend)
	}
	want := []string{"fak-bin", "m", "--", "fak-bin", "codex-resume", "--json", "--rollout", "rollout.jsonl", "--cwd", p.CWD, "--prompt-file", "goal.txt", "--result-file", "result.json", "thread-id"}
	if !reflect.DeepEqual(got.Argv, want) {
		t.Fatalf("argv=%q want=%q", got.Argv, want)
	}
	if _, ok := got.Env["ANTHROPIC_BASE_URL"]; ok {
		t.Fatalf("Codex child inherited Anthropic gateway wiring")
	}
	if strings.Contains(strings.Join(got.Argv, " "), "continue working toward") {
		t.Fatalf("goal prose leaked into argv: %q", got.Argv)
	}
}

func TestResumeBrokerKeepsLegacyClaudeDefault(t *testing.T) {
	p := resume.WatchdogPlanRow{Session: "claude-session", CWD: t.TempDir()}
	got := rwResumeBrokerAttempt("fak-bin", "claude-bin", p, "claude-config", nil)
	if got.Backend != "claude" {
		t.Fatalf("backend=%q", got.Backend)
	}
	if len(got.Argv) < 7 || got.Argv[0] != "fak-bin" || got.Argv[1] != "m" || got.Argv[2] != "--" || got.Argv[3] != "claude-bin" || got.Argv[4] != "--resume" || got.Argv[5] != p.Session {
		t.Fatalf("argv=%q", got.Argv)
	}
}

func TestLoadCodexCompletionsTypesReclaimedSuccess(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "result.json")
	want := codexresume.Result{Outcome: codexresume.OutcomeCompletedReclaimed, UsefulWork: true, TaskCompleted: true, ForcedReclaim: true}
	b, _ := json.Marshal(want)
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	got := rwLoadCodexCompletions([]resume.WatchdogPlanRow{{Session: "thread", Harness: "codex", ResultFile: path}, {Session: "claude", ResultFile: path}})
	if len(got) != 1 || got[0].Session != "thread" || got[0].Result != want {
		t.Fatalf("got=%+v", got)
	}
}

func TestCodexWatchdogBypassesClaudeSeatRoster(t *testing.T) {
	p := resume.WatchdogPlanRow{Session: "thread", Harness: "codex"}
	guards := resume.WatchdogGuards{WorkerAccounts: map[string]bool{".claude-worker": true}, MaxAttempts: 2}
	effective := guards
	if rwHarness(p) == "codex" {
		effective.WorkerAccounts = nil
	}
	d := resume.DecideWatchdogRow(p, effective, nil, resume.OutcomeUnknown)
	if d.Action != resume.WatchdogLaunch {
		t.Fatalf("decision=%+v", d)
	}
}

func TestCodexPlanRequiresTypedResultSink(t *testing.T) {
	err := validateCodexResumeCoordinates(resume.WatchdogPlanRow{Harness: "codex", Session: "s", Rollout: "r", GoalFile: "g"})
	if err == nil || !strings.Contains(err.Error(), "result_file") {
		t.Fatalf("err=%v", err)
	}
}

func TestCodexWatchdogFreshProcessRecordsCompletionWithoutKillingUnrelatedChild(t *testing.T) {
	dir := t.TempDir()
	rollout := filepath.Join(dir, "rollout.jsonl")
	goal := filepath.Join(dir, "goal.txt")
	result := filepath.Join(dir, "result.json")
	if err := os.WriteFile(rollout, []byte("seed\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(goal, []byte("continue"), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	unrelated := exec.CommandContext(ctx, os.Args[0], "-test.run=TestWatchdogCodexHelper", "--", "unrelated", rollout)
	unrelated.Env = append(os.Environ(), "GO_WANT_WATCHDOG_CODEX_HELPER=1")
	if err := unrelated.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { cancel(); _ = unrelated.Wait() }()
	p := resume.WatchdogPlanRow{Session: "thread-id", Harness: "codex", CWD: dir, Rollout: rollout, GoalFile: goal, ResultFile: result}
	grant := launchBrokerGrant{Argv: []string{os.Args[0], "-test.run=TestWatchdogCodexHelper", "--", "watchdog-runner", rollout, result}, Env: envMap(append(os.Environ(), "GO_WANT_WATCHDOG_CODEX_HELPER=1")), CWD: dir}
	if _, err := rwSpawnResume("", p, "", dir, grant); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		raw, err := os.ReadFile(result)
		if err == nil {
			var got codexresume.Result
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatal(err)
			}
			if got.Outcome != codexresume.OutcomeCompletedReclaimed || !got.TaskCompleted || !got.ForcedReclaim {
				t.Fatalf("result=%+v", got)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("result not written: %v", err)
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err := unrelated.Process.Signal(os.Signal(syscall.Signal(0))); err != nil {
		t.Fatalf("unrelated child was killed: %v", err)
	}
}

func TestWatchdogCodexHelper(t *testing.T) {
	if os.Getenv("GO_WANT_WATCHDOG_CODEX_HELPER") != "1" {
		return
	}
	args := os.Args
	sep := slices.Index(args, "--")
	mode, rollout := args[sep+1], args[sep+2]
	if mode == "unrelated" {
		time.Sleep(time.Hour)
		return
	}
	if mode == "completed-hung" {
		f, _ := os.OpenFile(rollout, os.O_APPEND|os.O_WRONLY, 0)
		enc := json.NewEncoder(f)
		_ = enc.Encode(map[string]any{"type": "event_msg", "payload": map[string]any{"type": "task_started"}})
		_ = enc.Encode(map[string]any{"type": "response_item", "payload": map[string]any{"type": "custom_tool_call_output"}})
		_ = enc.Encode(map[string]any{"type": "event_msg", "payload": map[string]any{"type": "task_complete"}})
		_ = f.Close()
		time.Sleep(time.Hour)
		return
	}
	resultPath := args[sep+3]
	command := []string{os.Args[0], "-test.run=TestWatchdogCodexHelper", "--", "completed-hung", rollout}
	r, err := codexresume.Run(context.Background(), codexresume.Config{Command: command, RolloutPath: rollout, Deadline: 2 * time.Second, PollInterval: 10 * time.Millisecond, Drain: 20 * time.Millisecond, Env: append(os.Environ(), "GO_WANT_WATCHDOG_CODEX_HELPER=1")})
	if err != nil {
		os.Exit(4)
	}
	b, _ := json.Marshal(r)
	_ = os.WriteFile(resultPath, b, 0o600)
}
