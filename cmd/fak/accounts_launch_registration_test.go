package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func TestRegisteredAccountsChildPersistsBeforeAgentRunnerAndTerminalizes(t *testing.T) {
	old := accountsLaunchRun
	defer func() { accountsLaunchRun = old }()
	path := filepath.Join(t.TempDir(), "r.jsonl")
	called := false
	accountsLaunchRun = func(_, _ io.Writer, argv, env []string) launchRunResult {
		called = true
		rows, err := (sessionregistry.Store{Path: path}).ReadAll()
		if err != nil || len(rows) != 1 || rows[0].State != sessionregistry.StateRegistered {
			t.Fatalf("prelaunch=%+v err=%v", rows, err)
		}
		m := envMapFromStrings(env)
		if m["FAK_REGISTRATION_ID"] == "" || m["FAK_ATTEMPT_ID"] == "" || m["FAK_SESSION_REGISTRY"] != path {
			t.Fatalf("env=%+v", m)
		}
		return launchRunResult{Code: 0}
	}
	env := []string{"FAK_SESSION_REGISTRY=" + path, "FAK_ROOT_ISSUE=6469", "FAK_TASK_ID=issue-6469", "FAK_WITNESS_REF=commit:test"}
	res := runRegisteredAccountsChild(&bytes.Buffer{}, &bytes.Buffer{}, []string{"claude", "-p", "task"}, env, "")
	if !called || res.Code != 0 || res.RegistrationID == "" || res.AttemptID == "" {
		t.Fatalf("res=%+v called=%v", res, called)
	}
	rows, err := (sessionregistry.Store{Path: path}).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].State != sessionregistry.StateCompleted || rows[0].WitnessRef != "commit:test" || rows[0].RootIssue != "6469" {
		t.Fatalf("row=%+v", rows[0])
	}
}

func TestRegisteredAccountsChildFailureAndFallbackAttemptsAreDistinct(t *testing.T) {
	old := accountsLaunchRun
	defer func() { accountsLaunchRun = old }()
	path := filepath.Join(t.TempDir(), "r.jsonl")
	calls := 0
	accountsLaunchRun = func(_, _ io.Writer, _, _ []string) launchRunResult { calls++; return launchRunResult{Code: 7} }
	env := []string{"FAK_SESSION_REGISTRY=" + path}
	first := runRegisteredAccountsChild(io.Discard, io.Discard, []string{"codex", "exec"}, env, "")
	env2 := append(env, "FAK_REGISTRATION_ID="+first.RegistrationID, "FAK_ATTEMPT_ID="+first.AttemptID, "FAK_ROOT_REGISTRATION_ID="+first.RootRegistrationID)
	second := runRegisteredAccountsChild(io.Discard, io.Discard, []string{"codex", "exec"}, env2, first.AttemptID)
	rows, err := (sessionregistry.Store{Path: path}).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || len(rows) != 2 || first.AttemptID == second.AttemptID || rows[1].ResumeOfAttemptID != first.AttemptID || rows[1].ParentRegistrationID != first.RegistrationID || rows[1].RootRegistrationID != first.RootRegistrationID || rows[0].State != sessionregistry.StateFailed || rows[1].State != sessionregistry.StateFailed {
		t.Fatalf("first=%+v second=%+v rows=%+v", first, second, rows)
	}
}

func TestAccountsNonAgentCommandPreservesRunnerWithoutRegistration(t *testing.T) {
	old := accountsLaunchRun
	defer func() { accountsLaunchRun = old }()
	path := filepath.Join(t.TempDir(), "r.jsonl")
	calls := 0
	accountsLaunchRun = func(_, _ io.Writer, _, _ []string) launchRunResult { calls++; return launchRunResult{Code: 3} }
	res := runRegisteredAccountsChild(io.Discard, io.Discard, []string{"git", "status"}, []string{"FAK_SESSION_REGISTRY=" + path}, "")
	if calls != 1 || res.Code != 3 {
		t.Fatalf("res=%+v calls=%d", res, calls)
	}
	if _, err := (sessionregistry.Store{Path: path}).ReadAll(); err == nil {
		t.Fatal("non-agent created registry")
	}
}

func TestAccountsAgentRuntimeRecognizesWrappedAndDirect(t *testing.T) {
	for _, tc := range []struct {
		argv []string
		want string
		ok   bool
	}{{[]string{"claude"}, "claude", true}, {[]string{"fak", "guard", "--", "codex", "exec"}, "codex", true}, {[]string{"opencode", "run"}, "opencode", true}, {[]string{"git", "status"}, "", false}} {
		got, ok := accountsAgentRuntime(tc.argv)
		if got != tc.want || ok != tc.ok {
			t.Fatalf("argv=%v got=%q,%v", tc.argv, got, ok)
		}
	}
}

func TestExecLaunchChildReadsBackPIDAndStart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "r.jsonl")
	s := sessionregistry.Store{Path: path}
	r, _ := sessionregistry.New(sessionregistry.NewInput{RegistrationID: "r", AttemptID: "a", LaunchKind: "external_account_launch", Runtime: "codex"})
	if err := s.Register(r); err != nil {
		t.Fatal(err)
	}
	env := append(os.Environ(), "FAK_SESSION_REGISTRY="+path, "FAK_REGISTRATION_ID=r", "GO_WANT_ACCOUNTS_REG_HELPER=1")
	res := execLaunchChild(io.Discard, io.Discard, []string{os.Args[0], "-test.run=^TestAccountsRegistrationHelper$"}, env)
	if res.Code != 0 {
		t.Fatalf("res=%+v", res)
	}
	rows, err := s.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].State != sessionregistry.StateActive || rows[0].Identity.PID == 0 || rows[0].StartedAt.IsZero() {
		t.Fatalf("row=%+v", rows[0])
	}
}
func TestAccountsRegistrationHelper(t *testing.T) {
	if os.Getenv("GO_WANT_ACCOUNTS_REG_HELPER") != "1" {
		return
	}
	os.Exit(0)
}
