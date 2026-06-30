package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopdrive"
	"github.com/anthony-chaudhary/fak/internal/loopgate"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestLoopDriveReadsGoalFreshEachTurn(t *testing.T) {
	oldNewCommand := loopDriveNewCommand
	oldWitness := loopDriveRunWitness
	defer func() {
		loopDriveNewCommand = oldNewCommand
		loopDriveRunWitness = oldWitness
	}()

	goal := filepath.Join(t.TempDir(), "GOAL.md")
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	writeLoopDriveGoal(t, goal, false, false)

	var nextItems []string
	witnessCalls := 0
	loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
		next := loopDriveEnvValue(env, "FAK_GOAL_NEXT")
		nextItems = append(nextItems, next)
		return &loopDriveFakeCommand{wait: func() error {
			switch next {
			case "first step":
				writeLoopDriveGoal(t, goal, true, false)
			case "second step":
				writeLoopDriveGoal(t, goal, true, true)
			default:
				t.Fatalf("unexpected next item %q", next)
			}
			return nil
		}}
	}
	loopDriveRunWitness = func(spec loopdrive.Spec, headBefore, headAfter string) loopDriveWitnessResult {
		witnessCalls++
		if witnessCalls == 1 {
			return loopDriveWitnessResult{Status: loopmgr.StatusWitnessRefused, Reason: "NOT_YET", Summary: "first turn not done", ExitCode: 1}
		}
		return loopDriveWitnessResult{Status: loopmgr.StatusWitnessedDone, Reason: "WITNESS_OK", Summary: "done", ExitCode: 0}
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--goal", goal, "--ledger", ledger, "--max-iters", "3", "--", "worker"})
	if code != 0 {
		t.Fatalf("drive code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	want := []string{"first step", "second step"}
	if !reflect.DeepEqual(nextItems, want) {
		t.Fatalf("next items = %v, want %v", nextItems, want)
	}
	if !strings.Contains(stdout.String(), "loop drive witnessed done") {
		t.Fatalf("stdout missing witnessed done line: %s", stdout.String())
	}
	var statusOut, statusErr bytes.Buffer
	if statusCode := runLoop(&statusOut, &statusErr, []string{"status", "--ledger", ledger}); statusCode != 0 {
		t.Fatalf("status code=%d stderr=%s", statusCode, statusErr.String())
	}
	for _, want := range []string{"issue-1175", "fires=2", "witnessed=1", "last_run=", "witnessed_done"} {
		if !strings.Contains(statusOut.String(), want) {
			t.Fatalf("status missing %q:\n%s", want, statusOut.String())
		}
	}
}

func TestLoopDriveBudgetExhaustionRecordsStructuredReason(t *testing.T) {
	oldNewCommand := loopDriveNewCommand
	oldWitness := loopDriveRunWitness
	defer func() {
		loopDriveNewCommand = oldNewCommand
		loopDriveRunWitness = oldWitness
	}()

	goal := filepath.Join(t.TempDir(), "GOAL.md")
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	writeLoopDriveGoal(t, goal, false, false)
	loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
		return &loopDriveFakeCommand{wait: func() error { return nil }}
	}
	loopDriveRunWitness = func(spec loopdrive.Spec, headBefore, headAfter string) loopDriveWitnessResult {
		return loopDriveWitnessResult{Status: loopmgr.StatusWitnessRefused, Reason: "NOT_YET", Summary: "still open", ExitCode: 1}
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--goal", goal, "--ledger", ledger, "--max-iters", "1", "--", "worker"})
	if code != 3 {
		t.Fatalf("drive code=%d, want 3 stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatal(err)
	}
	foundBudget := false
	for _, ev := range events {
		if ev.Kind == loopmgr.EventAdmit && ev.Status == loopmgr.StatusRefused && ev.Reason == loopdrive.ReasonBudgetSpent {
			foundBudget = true
		}
	}
	if !foundBudget {
		t.Fatalf("ledger missing refused admit with %s: %+v", loopdrive.ReasonBudgetSpent, events)
	}
	raw, err := os.ReadFile(goal)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), loopdrive.ReasonBudgetSpent) {
		t.Fatalf("goal scratch missing %s:\n%s", loopdrive.ReasonBudgetSpent, raw)
	}
}

func TestLoopDriveAppendsScratchOnFailure(t *testing.T) {
	oldNewCommand := loopDriveNewCommand
	defer func() { loopDriveNewCommand = oldNewCommand }()

	goal := filepath.Join(t.TempDir(), "GOAL.md")
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	writeLoopDriveGoal(t, goal, false, false)
	loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
		return &loopDriveFakeCommand{wait: func() error { return errors.New("not yet") }}
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--goal", goal, "--ledger", ledger, "--", "worker"})
	if code != 1 {
		t.Fatalf("drive code=%d, want 1 stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	raw, err := os.ReadFile(goal)
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, want := range []string{"# Scratch / last-refusal", "NOT_YET", "first step", "not yet"} {
		if !strings.Contains(text, want) {
			t.Fatalf("goal scratch missing %q:\n%s", want, text)
		}
	}
}

func TestLoopDriveReviewModelExportsCommitReviewEnv(t *testing.T) {
	oldNewCommand := loopDriveNewCommand
	oldWitness := loopDriveRunWitness
	defer func() {
		loopDriveNewCommand = oldNewCommand
		loopDriveRunWitness = oldWitness
	}()

	goal := filepath.Join(t.TempDir(), "GOAL.md")
	writeLoopDriveGoal(t, goal, false, true)
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")

	var sawEnv []string
	loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
		sawEnv = append([]string(nil), env...)
		return &loopDriveFakeCommand{wait: func() error {
			writeLoopDriveGoal(t, goal, true, true)
			return nil
		}}
	}
	loopDriveRunWitness = func(spec loopdrive.Spec, headBefore, headAfter string) loopDriveWitnessResult {
		return loopDriveWitnessResult{Status: loopmgr.StatusWitnessedDone, Reason: "WITNESS_OK", Summary: "done", ExitCode: 0}
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"drive",
		"--goal", goal,
		"--review-model", "cheap-scout",
		"--review-endpoint", "http://reviewer/v1",
		"--review-api-key-env", "SCOUT_KEY",
		"--ledger", ledger,
		"--",
		"worker",
	})
	if code != 0 {
		t.Fatalf("drive code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	checks := map[string]string{
		"FAK_REVIEW_MODEL":       "cheap-scout",
		"FAK_REVIEW_OBJECTIVE":   "Ship the loop driver.",
		"FAK_REVIEW_ENDPOINT":    "http://reviewer/v1",
		"FAK_REVIEW_API_KEY_ENV": "SCOUT_KEY",
		"FAK_LOOP_LEDGER":        ledger,
		"FAK_GOAL_RUN":           "issue-1175-turn-1",
	}
	for key, want := range checks {
		if got := loopDriveEnvValue(sawEnv, key); got != want {
			t.Fatalf("%s = %q, want %q", key, got, want)
		}
	}
}

func TestLoopDriveTemplate(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--template", "--loop", "issue-1175"})
	if code != 0 {
		t.Fatalf("template code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"loop: issue-1175", "witness: commit-audit", "# Objective", "# Plan"} {
		if !strings.Contains(out, want) {
			t.Fatalf("template missing %q:\n%s", want, out)
		}
	}
}

func TestLoopDriveParsesDOSCitationResolveCriterion(t *testing.T) {
	c, err := loopDriveGateCriterion("dos citation-resolve 999 F.999 1")
	if err != nil {
		t.Fatal(err)
	}
	if c.Kind != loopgate.CriterionCitationResolve || c.Subject != "999 F.999 1" {
		t.Fatalf("criterion = %+v, want citation-resolve subject", c)
	}
}

type loopDriveFakeCommand struct {
	wait func() error
}

func (c *loopDriveFakeCommand) Start() error { return nil }
func (c *loopDriveFakeCommand) Wait() error {
	if c.wait != nil {
		return c.wait()
	}
	return nil
}
func (c *loopDriveFakeCommand) PID() int    { return 1234 }
func (c *loopDriveFakeCommand) Kill() error { return nil }

func writeLoopDriveGoal(t *testing.T, path string, firstDone, secondDone bool) {
	t.Helper()
	mark := func(done bool) string {
		if done {
			return "x"
		}
		return " "
	}
	body := fmt.Sprintf(`---
loop: issue-1175
witness: commit-audit
budget: { max_iters: 5 }
---
# Objective
Ship the loop driver.

# Plan
- [%s] first step
- [%s] second step

# Scratch / last-refusal
`, mark(firstDone), mark(secondDone))
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func loopDriveEnvValue(env []string, key string) string {
	prefix := key + "="
	for _, kv := range env {
		if strings.HasPrefix(kv, prefix) {
			return strings.TrimPrefix(kv, prefix)
		}
	}
	return ""
}

// TestLoopDriveGatesWitnessedDoneWithoutHandoffNextStep proves the wiring: a
// turn that the exit gate witnesses done is still blocked when the agent wrote a
// task-handoff record with no next step and no no-next-step reason. The handoff
// file is exposed to the child via FAK_TASK_HANDOFF_FILE.
func TestLoopDriveGatesWitnessedDoneWithoutHandoffNextStep(t *testing.T) {
	oldNewCommand := loopDriveNewCommand
	oldWitness := loopDriveRunWitness
	defer func() {
		loopDriveNewCommand = oldNewCommand
		loopDriveRunWitness = oldWitness
	}()

	goal := filepath.Join(t.TempDir(), "GOAL.md")
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	handoff := filepath.Join(t.TempDir(), "task-handoff.json")
	writeLoopDriveGoal(t, goal, true, true)

	loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
		// The child sees the handoff path and writes a verified-done record that
		// pushes neither a next step nor an explicit no-next-step reason.
		if got := loopDriveEnvValue(env, "FAK_TASK_HANDOFF_FILE"); got != handoff {
			t.Fatalf("FAK_TASK_HANDOFF_FILE=%q, want %q", got, handoff)
		}
		return &loopDriveFakeCommand{wait: func() error {
			rec := `{"schema":"fak.task-handoff.v1","task":{"task_id":"t1","state":"done","witness":{"verified_state":"verified_done"}},"current_state":"shipped"}`
			return os.WriteFile(handoff, []byte(rec), 0o644)
		}}
	}
	loopDriveRunWitness = func(spec loopdrive.Spec, headBefore, headAfter string) loopDriveWitnessResult {
		return loopDriveWitnessResult{Status: loopmgr.StatusWitnessedDone, Reason: "WITNESS_OK", Summary: "done", ExitCode: 0}
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--goal", goal, "--ledger", ledger, "--task-handoff-file", handoff, "--max-iters", "1", "--", "worker"})
	if code != 3 {
		t.Fatalf("drive code=%d, want 3 (handoff-gated) stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if strings.Contains(stdout.String(), "loop drive witnessed done") {
		t.Fatalf("witnessed-done completion should be gated, not announced: %s", stdout.String())
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatal(err)
	}
	foundRefusal := false
	for _, ev := range events {
		if ev.Kind == loopmgr.EventAdmit && ev.Status == loopmgr.StatusRefused && ev.Reason == loopdrive.ReasonHandoffRefused {
			foundRefusal = true
		}
	}
	if !foundRefusal {
		t.Fatalf("ledger missing handoff refusal admit (%s): %+v", loopdrive.ReasonHandoffRefused, events)
	}
}

// TestLoopDrivePassesWitnessedDoneWithHandoffReason proves the pass path: a
// verified-done handoff with an explicit no-next-step reason lets the completion
// through.
func TestLoopDrivePassesWitnessedDoneWithHandoffReason(t *testing.T) {
	oldNewCommand := loopDriveNewCommand
	oldWitness := loopDriveRunWitness
	defer func() {
		loopDriveNewCommand = oldNewCommand
		loopDriveRunWitness = oldWitness
	}()

	goal := filepath.Join(t.TempDir(), "GOAL.md")
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	handoff := filepath.Join(t.TempDir(), "task-handoff.json")
	writeLoopDriveGoal(t, goal, true, true)

	loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
		return &loopDriveFakeCommand{wait: func() error {
			rec := `{"schema":"fak.task-handoff.v1","task":{"task_id":"t1","state":"done","witness":{"verified_state":"verified_done"}},"current_state":"shipped","no_next_step_reason":"feature fully shipped; nothing reasonable remains"}`
			return os.WriteFile(handoff, []byte(rec), 0o644)
		}}
	}
	loopDriveRunWitness = func(spec loopdrive.Spec, headBefore, headAfter string) loopDriveWitnessResult {
		return loopDriveWitnessResult{Status: loopmgr.StatusWitnessedDone, Reason: "WITNESS_OK", Summary: "done", ExitCode: 0}
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--goal", goal, "--ledger", ledger, "--task-handoff-file", handoff, "--max-iters", "1", "--", "worker"})
	if code != 0 {
		t.Fatalf("drive code=%d, want 0 stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "loop drive witnessed done") {
		t.Fatalf("stdout missing witnessed-done line: %s", stdout.String())
	}
}
