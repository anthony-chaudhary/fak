package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/codexresume"
)

func TestCodexResumeUsage(t *testing.T) {
	var out, err bytes.Buffer
	if c := runCodexResume(&out, &err, nil); c != 2 {
		t.Fatalf("code=%d", c)
	}
	if !strings.Contains(err.String(), "usage: fak codex-resume") {
		t.Fatalf("stderr=%q", err.String())
	}
}

func TestWriteCodexResumeResultRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "result.json")
	want := codexresume.Result{Outcome: codexresume.OutcomeCompletedReclaimed, UsefulWork: true, TaskCompleted: true, ForcedReclaim: true}
	if err := writeCodexResumeResult(path, want); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got codexresume.Result
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("got=%+v want=%+v", got, want)
	}
}

func writeCodexResumeFixture(t *testing.T, home, threadID, functionCallID string) string {
	t.Helper()
	path := filepath.Join(home, "sessions", "2026", "08", "11", "rollout-"+threadID+".jsonl")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	rows := []map[string]any{
		{"type": "session_meta", "payload": map[string]any{"id": threadID, "model_provider": "fak"}},
		{"type": "response_item", "payload": map[string]any{"type": "function_call", "id": functionCallID, "call_id": "call_logical", "name": "shell_command"}},
		{"type": "response_item", "payload": map[string]any{"type": "function_call_output", "id": "fco_fixture", "call_id": "call_logical", "output": "ok"}},
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	encoder := json.NewEncoder(file)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			_ = file.Close()
			t.Fatal(err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCodexResumePreflightOnlyReportsCompatibilityWithoutPrompt(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff200-1000-7000-8000-000000000001"
	rollout := writeCodexResumeFixture(t, home, threadID, "call_legacy")
	var out, errOut bytes.Buffer
	code := runCodexResume(&out, &errOut, []string{
		"--json",
		"--preflight-only",
		"--codex-home", home,
		"--rollout", rollout,
		threadID,
	})
	if code != 1 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
	var got codexresume.Result
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Outcome != codexresume.OutcomeRefused || got.Preflight == nil ||
		got.Preflight.Verdict != codexresume.VerdictIncompatibleHistory ||
		got.Preflight.LookupState != codexresume.LookupFound ||
		got.Preflight.CodexHome != home || got.Preflight.SessionRoot != filepath.Join(home, "sessions") ||
		got.LaunchState != codexresume.LaunchNotAttempted || got.LaunchPID != 0 {
		t.Fatalf("result=%+v", got)
	}
}

func TestCodexResumeBulkPreflightEmitsPerThreadResultsAndAggregateExit(t *testing.T) {
	home := t.TempDir()
	compatible := "019ff200-1000-7000-8000-000000000002"
	incompatible := "019ff200-1000-7000-8000-000000000003"
	writeCodexResumeFixture(t, home, compatible, "fc_response_item")
	writeCodexResumeFixture(t, home, incompatible, "call_legacy")

	var out, errOut bytes.Buffer
	code := runCodexResume(&out, &errOut, []string{
		"--json",
		"--preflight-only",
		"--codex-home", home,
		"--thread", compatible,
		"--thread", incompatible,
	})
	if code != 1 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
	var got codexResumeBatchResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-codex-resume-batch/1" || got.Succeeded != 1 || got.Failed != 1 ||
		got.ExitCode != 1 || len(got.Results) != 2 {
		t.Fatalf("batch=%+v", got)
	}
	if got.Results[0].Preflight == nil || got.Results[0].Preflight.Verdict != codexresume.VerdictResumable {
		t.Fatalf("compatible result=%+v", got.Results[0])
	}
	if got.Results[1].Preflight == nil || got.Results[1].Preflight.Verdict != codexresume.VerdictIncompatibleHistory {
		t.Fatalf("incompatible result=%+v", got.Results[1])
	}
}

func readyCodexResumeHome(name, dir string) accounts.Home {
	return accounts.Home{
		Name: name,
		Dir:  dir,
		Identity: accounts.Identity{
			Exists:   true,
			HasCreds: true,
		},
	}
}

func writeCodexResumeAuth(t *testing.T, home, accountID string) {
	t.Helper()
	if err := os.MkdirAll(home, 0o700); err != nil {
		t.Fatal(err)
	}
	payload, err := json.Marshal(map[string]any{"account_id": accountID})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, "auth.json"), payload, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestPlanCodexResumeAccountCopiesSelectedRolloutAndUsesExactUUID(t *testing.T) {
	sourceHome := t.TempDir()
	targetHome := t.TempDir()
	threadID := "019ff200-1000-7000-8000-000000000101"
	sourceRollout := writeCodexResumeFixture(t, sourceHome, threadID, "fc_response_item")
	writeCodexResumeAuth(t, sourceHome, "source-account-secret")
	writeCodexResumeAuth(t, targetHome, "target-account-secret")
	before, err := os.ReadFile(sourceRollout)
	if err != nil {
		t.Fatal(err)
	}

	plan, err := planCodexResumeAccount(
		accounts.Registry{},
		[]accounts.Home{readyCodexResumeHome("source", sourceHome), readyCodexResumeHome("target", targetHome)},
		filepath.Join(t.TempDir(), "bindings"),
		"target",
		threadID[:12],
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Account != "target" || plan.TargetHome != targetHome || plan.SourceHome != sourceHome {
		t.Fatalf("plan selected wrong account/home: %+v", plan)
	}
	if plan.ThreadID != threadID {
		t.Fatalf("thread id=%q want exact UUID %q", plan.ThreadID, threadID)
	}
	if plan.Transition != codexresume.RehomeUnbound {
		t.Fatalf("transition=%q want %q", plan.Transition, codexresume.RehomeUnbound)
	}
	targetCanonical, err := canonicalCodexResumeHome(targetHome)
	if err != nil {
		t.Fatal(err)
	}
	rolloutCanonical, err := canonicalCodexResumeHome(filepath.Dir(plan.RolloutPath))
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Clean(plan.RolloutPath) == filepath.Clean(sourceRollout) || !strings.HasPrefix(rolloutCanonical, targetCanonical+string(filepath.Separator)) {
		t.Fatalf("rollout path=%q is not beneath selected target home", plan.RolloutPath)
	}
	copied, err := os.ReadFile(plan.RolloutPath)
	if err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(sourceRollout)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(copied, before) || !bytes.Equal(after, before) {
		t.Fatal("selected rollout was not copied exactly or source changed")
	}
	if strings.Contains(plan.Binding.AccountKeyDigest, "target-account-secret") {
		t.Fatalf("binding leaked raw account identity: %+v", plan.Binding)
	}
}

func TestPlanCodexResumeAccountRefusesFallbackCollisionAndAmbiguousOwnership(t *testing.T) {
	firstHome := t.TempDir()
	secondHome := t.TempDir()
	threadID := "019ff200-1000-7000-8000-000000000102"
	writeCodexResumeFixture(t, firstHome, threadID, "fc_first")
	writeCodexResumeFixture(t, secondHome, threadID, "fc_second")
	writeCodexResumeAuth(t, firstHome, "first-account-secret")
	writeCodexResumeAuth(t, secondHome, "second-account-secret")
	discovered := []accounts.Home{
		readyCodexResumeHome("first", firstHome),
		readyCodexResumeHome("second", secondHome),
	}
	stateRoot := filepath.Join(t.TempDir(), "bindings")

	if _, err := planCodexResumeAccount(accounts.Registry{}, discovered, stateRoot, "missing", threadID, false); err == nil {
		t.Fatal("missing exact account unexpectedly fell back")
	}
	duplicateName := append(discovered, readyCodexResumeHome("second", t.TempDir()))
	if _, err := planCodexResumeAccount(accounts.Registry{}, duplicateName, stateRoot, "second", threadID, false); err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("duplicate discovered account name error=%v", err)
	}
	collidingRegistry := accounts.Registry{Homes: []accounts.Home{{Name: "second", Dir: t.TempDir()}}}
	if _, err := planCodexResumeAccount(collidingRegistry, discovered, stateRoot, "second", threadID, false); err == nil || !strings.Contains(err.Error(), "collides") {
		t.Fatalf("persisted-name collision error=%v", err)
	}
	if _, err := planCodexResumeAccount(accounts.Registry{}, discovered, stateRoot, "second", threadID, false); err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("ambiguous source ownership error=%v", err)
	}
}

func TestPlanCodexResumeAccountClassifiesSameHomeOAuthReplacementWithoutCopy(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff200-1000-7000-8000-000000000103"
	rollout := writeCodexResumeFixture(t, home, threadID, "fc_response_item")
	writeCodexResumeAuth(t, home, "new-account-secret")
	store := codexresume.BindingStore{Root: filepath.Join(t.TempDir(), "bindings")}
	oldBinding, err := codexresume.NewThreadBinding(threadID, home, "old-account-digest", rollout, time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Save(oldBinding); err != nil {
		t.Fatal(err)
	}

	plan, err := planCodexResumeAccount(
		accounts.Registry{},
		[]accounts.Home{readyCodexResumeHome("seat", home)},
		store.Root,
		"seat",
		threadID,
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Transition != codexresume.RehomeSameHomeDifferentAccount {
		t.Fatalf("transition=%q want %q", plan.Transition, codexresume.RehomeSameHomeDifferentAccount)
	}
	wantRollout, err := filepath.EvalSymlinks(rollout)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Clean(plan.RolloutPath) != filepath.Clean(wantRollout) {
		t.Fatalf("same-home replacement copied rollout: got %q want %q", plan.RolloutPath, wantRollout)
	}
}

func TestCodexResumeAccountLaunchEnvAndBindingCommitGate(t *testing.T) {
	threadID := "019ff200-1000-7000-8000-000000000104"
	cmd, env := buildCodexResumeLaunch("codex-custom", threadID, "continue", []string{
		"PATH=/bin",
		"CODEX_HOME=/wrong",
		"CODEX_SESSION_ID=stale",
		"FAK_ATTEMPT_ID=stale",
	}, "/selected")
	if len(cmd) != 7 || cmd[0] != "codex-custom" || cmd[5] != threadID || cmd[6] != "continue" { //boundarylint:ignore CHANGE_DETECTOR_TEST the resume launch argv contract specifies exactly seven elements
		t.Fatalf("launch command did not use exact thread UUID: %v", cmd)
	}
	var codexHomes int
	for _, kv := range env {
		if strings.HasPrefix(kv, "CODEX_HOME=") {
			codexHomes++
			if kv != "CODEX_HOME=/selected" {
				t.Fatalf("selected env=%q", kv)
			}
		}
		if strings.HasPrefix(kv, "CODEX_SESSION_ID=") || strings.HasPrefix(kv, "FAK_ATTEMPT_ID=") {
			t.Fatalf("stale child identity survived: %q", kv)
		}
	}
	if codexHomes != 1 {
		t.Fatalf("CODEX_HOME count=%d env=%v", codexHomes, env)
	}

	home := t.TempDir()
	rollout := writeCodexResumeFixture(t, home, threadID, "fc_response_item")
	binding, err := codexresume.NewThreadBinding(threadID, home, "account-key", rollout, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	store := codexresume.BindingStore{Root: filepath.Join(t.TempDir(), "bindings")}
	if err := saveCodexResumeBindingOnSuccess(store, binding, codexresume.OutcomeExited); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load(binding.ThreadID); !errors.Is(err, codexresume.ErrBindingNotFound) {
		t.Fatalf("failed outcome changed binding: %v", err)
	}
	if err := saveCodexResumeBindingOnSuccess(store, binding, codexresume.OutcomeCompletedReclaimed); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load(binding.ThreadID); err != nil {
		t.Fatalf("successful outcome did not save binding: %v", err)
	}
}

func TestCodexResumeAccountRejectsBatch(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runCodexResume(&out, &errOut, []string{
		"--account", "seat",
		"--preflight-only",
		"--thread", "019ff200-1000-7000-8000-000000000105",
		"--thread", "019ff200-1000-7000-8000-000000000106",
	})
	if code != 2 || !strings.Contains(errOut.String(), "exactly one candidate") {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, out.String(), errOut.String())
	}
}

func TestCodexResumePreflightOnlyMissingShowsSelectedRoot(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff200-1000-7000-8000-000000000904"
	var out, errOut bytes.Buffer
	code := runCodexResume(&out, &errOut, []string{"--json", "--preflight-only", "--codex-home", home, threadID})
	if code != 1 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
	var got codexresume.Result
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Preflight == nil || got.Preflight.LookupState != codexresume.LookupNotFound || got.Preflight.SessionRoot != filepath.Join(home, "sessions") || got.LaunchState != codexresume.LaunchNotAttempted {
		t.Fatalf("result=%+v", got)
	}
}

func TestCodexResumeHumanOutputExplainsFreshProcess(t *testing.T) {
	old := codexResumeRecover
	defer func() { codexResumeRecover = old }()
	codexResumeRecover = func(_ context.Context, check codexresume.CheckConfig, _ codexresume.Config) (codexresume.Result, error) {
		preflight := codexresume.PreflightResult{ThreadID: check.ThreadID, CodexHome: check.CodexHome, SessionRoot: filepath.Join(check.CodexHome, "sessions"), LookupState: codexresume.LookupFound, Verdict: codexresume.VerdictResumable}
		return codexresume.Result{ThreadID: check.ThreadID, Preflight: &preflight, Outcome: codexresume.OutcomeExited, LaunchState: codexresume.LaunchStarted, LaunchPID: 4321}, nil
	}
	home := t.TempDir()
	threadID := "019ff200-1000-7000-8000-000000000905"
	var out, errOut bytes.Buffer
	_ = runCodexResume(&out, &errOut, []string{"--codex-home", home, threadID, "continue"})
	text := out.String()
	if !strings.Contains(text, "fresh Codex process started") || !strings.Contains(text, "pid=4321") || !strings.Contains(text, "lookup=found") || (!strings.Contains(text, home) && !strings.Contains(text, strconv.Quote(home))) {
		t.Fatalf("stdout=%q stderr=%q", text, errOut.String())
	}
}
