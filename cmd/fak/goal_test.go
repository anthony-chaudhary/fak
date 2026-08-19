package main

import (
	"bytes"
	"encoding/json"
	"io"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/goalregistry"
)

func TestGoalCLIEndToEndCreateBindShow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "goals.json")
	var out, errb bytes.Buffer
	code := runGoal(&out, &errb, []string{"create", "--registry", path, "--title", "Observe turns", "--summary", "Safe summary", "--actor", "operator", "--authority", "operator-declared"})
	if code != 0 {
		t.Fatalf("create code=%d stderr=%s", code, errb.String())
	}
	var g goalregistry.Goal
	if err := json.Unmarshal(out.Bytes(), &g); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	code = runGoal(&out, &errb, []string{"bind", "--registry", path, "--id", g.GoalID, "--namespace", "codex:goal", "--external-id", "thread-7", "--actor", "harness", "--authority", "harness-report"})
	if code != 0 {
		t.Fatalf("bind code=%d stderr=%s", code, errb.String())
	}
	out.Reset()
	errb.Reset()
	code = runGoal(&out, &errb, []string{"show", "--registry", path, "--id", g.GoalID})
	if code != 0 || !bytes.Contains(out.Bytes(), []byte(`"namespace": "codex:goal"`)) {
		t.Fatalf("show code=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
}

func TestGoalResolveEmitsCanonicalLaunchEnvironment(t *testing.T) {
	path := filepath.Join(t.TempDir(), "goals.json")
	var create bytes.Buffer
	if code := runGoal(&create, io.Discard, []string{"create", "--registry", path, "--title", "Cross harness", "--actor", "operator", "--authority", "user"}); code != 0 {
		t.Fatalf("create=%d", code)
	}
	var goal struct {
		ID string `json:"goal_id"`
	}
	if err := json.Unmarshal(create.Bytes(), &goal); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"bind", "--registry", path, "--id", goal.ID, "--namespace", "claude:goal", "--external-id", "cg-1", "--actor", "claude", "--authority", "harness"},
		{"bind", "--registry", path, "--id", goal.ID, "--namespace", "codex:goal", "--external-id", "xg-1", "--actor", "codex", "--authority", "harness"},
	} {
		if code := runGoal(io.Discard, io.Discard, args); code != 0 {
			t.Fatalf("bind=%d", code)
		}
	}
	for _, pair := range [][2]string{{"claude:goal", "cg-1"}, {"codex:goal", "xg-1"}} {
		var out bytes.Buffer
		if code := runGoal(&out, io.Discard, []string{"resolve", "--registry", path, "--namespace", pair[0], "--external-id", pair[1]}); code != 0 {
			t.Fatalf("resolve=%d", code)
		}
		var got struct {
			Schema string            `json:"schema"`
			GoalID string            `json:"goal_id"`
			Env    map[string]string `json:"env"`
		}
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if got.Schema != "fak-goal-resolution/1" || got.GoalID != goal.ID || got.Env["FAK_GOAL_ID"] != goal.ID {
			t.Fatalf("resolution = %#v", got)
		}
	}
	var stderr bytes.Buffer
	if code := runGoal(io.Discard, &stderr, []string{"resolve", "--registry", path, "--namespace", "codex:goal", "--external-id", "missing"}); code != 1 || !strings.Contains(stderr.String(), "binding not found") {
		t.Fatalf("missing code=%d stderr=%q", code, stderr.String())
	}
}
