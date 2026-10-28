package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"os/exec"
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

func TestGoalTopologyCLIExposesExecutionRoots(t *testing.T) {
	dir := t.TempDir()
	goals, sessions := filepath.Join(dir, "goals.json"), filepath.Join(dir, "sessions.jsonl")
	var create bytes.Buffer
	if code := runGoal(&create, io.Discard, []string{"create", "--registry", goals, "--title", "Topology", "--actor", "operator", "--authority", "user"}); code != 0 {
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
		{Schema: sessionregistry.Schema, RegistrationID: "claude-root", RootRegistrationID: "claude-root", GoalID: goal.GoalID, AttemptID: "a", LaunchKind: "guarded_tui", Identity: sessionregistry.Identity{Runtime: "claude", SessionID: "cs"}, State: sessionregistry.StateRegistered, CreatedAt: now},
		{Schema: sessionregistry.Schema, RegistrationID: "codex-root", RootRegistrationID: "codex-root", GoalID: goal.GoalID, AttemptID: "b", LaunchKind: "guarded_tui", Identity: sessionregistry.Identity{Runtime: "codex", ThreadID: "ct"}, State: sessionregistry.StateRegistered, CreatedAt: now},
		{Schema: sessionregistry.Schema, RegistrationID: "unbound", RootRegistrationID: "unbound", TaskID: "Topology", AttemptID: "c", LaunchKind: "guarded_tui", Identity: sessionregistry.Identity{Runtime: "codex"}, State: sessionregistry.StateRegistered, CreatedAt: now},
	} {
		if err := store.Register(row); err != nil {
			t.Fatal(err)
		}
	}
	var out bytes.Buffer
	if code := runGoal(&out, io.Discard, []string{"topology", "--registry", goals, "--session-registry", sessions, "--id", goal.GoalID}); code != 0 {
		t.Fatalf("topology=%d", code)
	}
	var got struct {
		Schema         string `json:"schema"`
		ExecutionRoots []struct {
			RootRegistrationID string                   `json:"root_registration_id"`
			Registrations      []sessionregistry.Record `json:"registrations"`
		} `json:"execution_roots"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-goal-topology/1" || len(got.ExecutionRoots) != 2 || got.ExecutionRoots[0].RootRegistrationID != "claude-root" || got.ExecutionRoots[1].Registrations[0].Identity.ThreadID != "ct" {
		t.Fatalf("topology=%#v", got)
	}
}

// fak-test:runtime fast est=100ms
func TestGoalCLIShowsWitnessedLifecycleEvidence(t *testing.T) {
	repo := t.TempDir()
	t.Chdir(repo)
	// TestMain clears repository-locating variables; also keep caller hooks,
	// signing settings, and templates out of this new fixture repository.
	t.Setenv("GIT_CONFIG_GLOBAL", os.DevNull)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
	t.Setenv("GIT_CONFIG_COUNT", "0")
	t.Setenv("GIT_CONFIG_PARAMETERS", "")
	t.Setenv("GIT_TEMPLATE_DIR", t.TempDir())
	git := func(args ...string) string {
		t.Helper()
		out, err := exec.Command("git", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "--quiet")
	git("-c", "user.name=Goal Fixture", "-c", "user.email=goal-fixture@example.invalid", "commit", "--allow-empty", "-m", "goal witness")
	commitRef := "commit:" + git("rev-parse", "HEAD")
	path := filepath.Join(repo, "goals.json")
	var create, stderr bytes.Buffer
	if code := runGoal(&create, &stderr, []string{"create", "--registry", path, "--title", "Witness", "--actor", "operator", "--authority", "user"}); code != 0 {
		t.Fatalf("create=%d stderr=%s", code, stderr.String())
	}
	var goal struct {
		GoalID string `json:"goal_id"`
	}
	if err := json.Unmarshal(create.Bytes(), &goal); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	base := []string{"transition", "--registry", path, "--id", goal.GoalID, "--lifecycle", "achieved", "--evidence-author", "judge"}
	for _, refusal := range []struct{ class, reference, diagnostic string }{
		{"agent_assertion", commitRef, "requires independent_witness"},
		{"independent_witness", "commit:" + strings.Repeat("0", 40), "does not resolve"},
	} {
		stderr.Reset()
		args := append(base, "--evidence-class", refusal.class, "--evidence-ref", refusal.reference)
		if code := runGoal(io.Discard, &stderr, args); code != 1 || !strings.Contains(stderr.String(), refusal.diagnostic) {
			t.Fatalf("%s %s: code=%d stderr=%s", refusal.class, refusal.reference, code, stderr.String())
		}
		after, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(before, after) {
			t.Fatalf("refused %s %s mutated registry", refusal.class, refusal.reference)
		}
	}
	stderr.Reset()
	if code := runGoal(io.Discard, &stderr, append(base, "--evidence-class", "independent_witness", "--evidence-ref", commitRef)); code != 0 {
		t.Fatalf("witness=%d stderr=%s", code, stderr.String())
	}
	var out bytes.Buffer
	stderr.Reset()
	if code := runGoal(&out, &stderr, []string{"show", "--registry", path, "--id", goal.GoalID}); code != 0 {
		t.Fatalf("show=%d stderr=%s", code, stderr.String())
	}
	var got struct {
		Goal            goalregistry.Goal              `json:"goal"`
		OutcomeEvidence []goalregistry.OutcomeEvidence `json:"outcome_evidence"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Goal.Lifecycle != goalregistry.Achieved || len(got.OutcomeEvidence) != 1 || got.OutcomeEvidence[0].Reference != commitRef {
		t.Fatalf("show=%#v", got)
	}
	stderr.Reset()
	if code := runGoal(io.Discard, &stderr, []string{"reopen", "--registry", path, "--id", goal.GoalID, "--evidence-author", "operator", "--evidence-ref", "decision:retry"}); code != 0 {
		t.Fatalf("reopen=%d stderr=%s", code, stderr.String())
	}
}
