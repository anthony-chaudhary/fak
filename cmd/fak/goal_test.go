package main

import (
	"bytes"
	"encoding/json"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
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

func TestGoalBackfillRootRequiresWitnessAndAppliesExactRoot(t *testing.T) {
	dir := t.TempDir()
	goals, sessions := filepath.Join(dir, "goals.json"), filepath.Join(dir, "sessions.jsonl")
	var create bytes.Buffer
	if code := runGoal(&create, io.Discard, []string{"create", "--registry", goals, "--title", "Observe", "--actor", "operator", "--authority", "user"}); code != 0 {
		t.Fatalf("create=%d", code)
	}
	var goal struct {
		GoalID string `json:"goal_id"`
	}
	if err := json.Unmarshal(create.Bytes(), &goal); err != nil {
		t.Fatal(err)
	}
	store := sessionregistry.Store{Path: sessions}
	now := time.Unix(1700000000, 0).UTC()
	for _, row := range []sessionregistry.Record{
		{Schema: sessionregistry.Schema, RegistrationID: "root-a", RootRegistrationID: "root-a", TaskID: "same", AttemptID: "a", LaunchKind: "guarded_tui", Identity: sessionregistry.Identity{Runtime: "claude"}, State: sessionregistry.StateRegistered, CreatedAt: now},
		{Schema: sessionregistry.Schema, RegistrationID: "root-b", RootRegistrationID: "root-b", TaskID: "same", AttemptID: "b", LaunchKind: "guarded_tui", Identity: sessionregistry.Identity{Runtime: "codex"}, State: sessionregistry.StateRegistered, CreatedAt: now.Add(time.Second)},
	} {
		if err := store.Register(row); err != nil {
			t.Fatal(err)
		}
	}
	base := []string{"backfill-root", "--registry", goals, "--session-registry", sessions, "--id", goal.GoalID, "--root-registration-id", "root-a"}
	if code := runGoal(io.Discard, io.Discard, base); code != 1 {
		t.Fatalf("missing witness=%d", code)
	}
	var preview bytes.Buffer
	if code := runGoal(&preview, io.Discard, append(base, "--witness", "operator-ledger:17")); code != 0 {
		t.Fatalf("preview=%d", code)
	}
	var result struct {
		Schema          string   `json:"schema"`
		Applied         bool     `json:"applied"`
		RegistrationIDs []string `json:"registration_ids"`
	}
	if err := json.Unmarshal(preview.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Schema != "fak-goal-root-backfill/1" || result.Applied || len(result.RegistrationIDs) != 1 {
		t.Fatalf("preview=%#v", result)
	}
	if code := runGoal(io.Discard, io.Discard, append(base, "--witness", "operator-ledger:17", "--actor", "operator", "--authority", "operator-declared", "--apply")); code != 0 {
		t.Fatalf("apply=%d", code)
	}
	rows, err := store.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].GoalID != goal.GoalID || rows[1].GoalID != "" {
		t.Fatalf("rows=%#v", rows)
	}
}
