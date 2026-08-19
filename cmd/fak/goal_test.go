package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
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
