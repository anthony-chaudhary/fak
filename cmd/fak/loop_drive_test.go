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
)

func TestLoopDriveReadsGoalFreshEachTurn(t *testing.T) {
	oldNewCommand := loopDriveNewCommand
	defer func() { loopDriveNewCommand = oldNewCommand }()

	goal := filepath.Join(t.TempDir(), "GOAL.md")
	writeLoopDriveGoal(t, goal, false, false)

	var nextItems []string
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

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--goal", goal, "--max-iters", "3", "--", "worker"})
	if code != 0 {
		t.Fatalf("drive code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	want := []string{"first step", "second step"}
	if !reflect.DeepEqual(nextItems, want) {
		t.Fatalf("next items = %v, want %v", nextItems, want)
	}
	if !strings.Contains(stdout.String(), "loop drive done") {
		t.Fatalf("stdout missing done line: %s", stdout.String())
	}
}

func TestLoopDriveAppendsScratchOnFailure(t *testing.T) {
	oldNewCommand := loopDriveNewCommand
	defer func() { loopDriveNewCommand = oldNewCommand }()

	goal := filepath.Join(t.TempDir(), "GOAL.md")
	writeLoopDriveGoal(t, goal, false, false)
	loopDriveNewCommand = func(argv []string, env []string, stdout, stderr io.Writer) loopCommand {
		return &loopDriveFakeCommand{wait: func() error { return errors.New("not yet") }}
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--goal", goal, "--", "worker"})
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
	defer func() { loopDriveNewCommand = oldNewCommand }()

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
		"FAK_GOAL_RUN":           "issue-1176-turn-1",
	}
	for key, want := range checks {
		if got := loopDriveEnvValue(sawEnv, key); got != want {
			t.Fatalf("%s = %q, want %q", key, got, want)
		}
	}
}

func TestLoopDriveTemplate(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"drive", "--template", "--loop", "issue-1176"})
	if code != 0 {
		t.Fatalf("template code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"loop: issue-1176", "witness: commit-audit", "# Objective", "# Plan"} {
		if !strings.Contains(out, want) {
			t.Fatalf("template missing %q:\n%s", want, out)
		}
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
loop: issue-1176
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
