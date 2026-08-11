package main

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

func TestGuardChildRegistrationPersistsBeforeLauncherAndTerminalizes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.jsonl")
	meta := guardChildSpawnMetadata{AgentRunID: "run", ToolCallID: "guard-child:run", Backend: "codex", PolicyDigest: "sha256:test", Envelope: toolprocgate.CapabilityEnvelope{Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn}}, RegistryPath: path}
	broker := toolprocgate.NewSpawnBroker()
	launched := false
	launcher := func(g toolprocgate.SpawnGrant) (*exec.Cmd, error) {
		launched = true
		rows, err := (sessionregistry.Store{Path: path}).ReadAll()
		if err != nil || len(rows) != 1 || rows[0].State != sessionregistry.StateRegistered {
			t.Fatalf("prelaunch rows=%+v err=%v", rows, err)
		}
		env := envMapFromGrant(g.Env)
		if env["FAK_REGISTRATION_ID"] == "" || env["FAK_ATTEMPT_ID"] == "" {
			t.Fatalf("lineage env=%+v", env)
		}
		if env["FAK_ATTEMPT_ID"] == env["FAK_PARENT_ATTEMPT_ID"] && env["FAK_PARENT_ATTEMPT_ID"] != "" {
			t.Fatalf("child reused parent attempt: %+v", env)
		}
		cmd := exec.Command(os.Args[0], "-test.run=^TestGuardRegistrationHelper$")
		cmd.Env = append(os.Environ(), "GO_WANT_GUARD_REG_HELPER=1")
		return cmd, nil
	}
	_, child, err := launchGuardChildWithBroker([]string{"agent"}, nil, false, meta, broker, launcher)
	if err != nil {
		t.Fatal(err)
	}
	if !launched {
		t.Fatal("launcher not called")
	}
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	if err := startBoundGuardRegistration(child); err != nil {
		t.Fatal(err)
	}
	runErr := child.Wait()
	terminalGuardChild(child, runErr, "")
	rows, err := (sessionregistry.Store{Path: path}).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].State != sessionregistry.StateCompleted || rows[0].Identity.PID == 0 || rows[0].StartedAt.IsZero() {
		t.Fatalf("rows=%+v", rows)
	}
}

func TestGuardChildRegistrationFailurePreventsLaunch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.jsonl")
	store := sessionregistry.Store{Path: path}
	parent, _ := sessionregistry.New(sessionregistry.NewInput{RegistrationID: "other-parent", AttemptID: "p", LaunchKind: "guarded_tui", Runtime: "codex"})
	if err := store.Register(parent); err != nil {
		t.Fatal(err)
	}
	meta := guardChildSpawnMetadata{AgentRunID: "run", ToolCallID: "guard-child:run", Backend: "codex", PolicyDigest: "sha256:test", Envelope: toolprocgate.CapabilityEnvelope{Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn}}, RegistryPath: path}
	called := false
	// A child naming an absent parent must fail before launcher invocation.
	t.Setenv("FAK_REGISTRATION_ID", "missing-parent")
	t.Setenv("FAK_ROOT_REGISTRATION_ID", "missing-parent")
	_, _, err := launchGuardChildWithBroker([]string{"agent"}, nil, false, meta, toolprocgate.NewSpawnBroker(), func(toolprocgate.SpawnGrant) (*exec.Cmd, error) {
		called = true
		return nil, errors.New("must not run")
	})
	if err == nil || called || !strings.Contains(err.Error(), "parent registration") {
		t.Fatalf("err=%v called=%v", err, called)
	}
}

func TestGuardRestartEnvCreatesDistinctResumeAttempt(t *testing.T) {
	ev := guardBudgetRestartEvent{FromTraceID: "attempt-1", ToTraceID: "attempt-2", Reason: "budget"}
	m := map[string]string{}
	for _, kv := range guardRestartEnv(ev) {
		m[kv[0]] = kv[1]
	}
	if m["FAK_CHILD_ATTEMPT_ID"] != "attempt-2" || m["FAK_RESUME_OF_ATTEMPT_ID"] != "attempt-1" || m["FAK_CHILD_ATTEMPT_ID"] == m["FAK_RESUME_OF_ATTEMPT_ID"] {
		t.Fatalf("env=%+v", m)
	}
}

func TestTerminalGuardChildRestartIsCancelled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "r.jsonl")
	s := sessionregistry.Store{Path: path}
	r, _ := sessionregistry.New(sessionregistry.NewInput{RegistrationID: "r", AttemptID: "a", LaunchKind: "guarded_tui", Runtime: "codex", Now: time.Now()})
	if err := s.Register(r); err != nil {
		t.Fatal(err)
	}
	cmd := &exec.Cmd{}
	bindGuardRegistration(cmd, guardRegistration{Store: s, Record: r})
	markGuardChildTerminalIntent(cmd, "restart")
	terminalGuardChild(cmd, errors.New("killed"), "")
	rows, _ := s.ReadAll()
	if rows[0].State != sessionregistry.StateCancelled || rows[0].Reason != "restart" {
		t.Fatalf("row=%+v", rows[0])
	}
}

func TestGuardRegistrationHelper(t *testing.T) {
	if os.Getenv("GO_WANT_GUARD_REG_HELPER") != "1" {
		return
	}
	os.Exit(0)
}

func TestGuardLauncherFailureLeavesTypedTerminalRegistration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "r.jsonl")
	meta := guardChildSpawnMetadata{AgentRunID: "run", ToolCallID: "guard-child:run", Backend: "codex", PolicyDigest: "sha256:test", Envelope: toolprocgate.CapabilityEnvelope{Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn}}, RegistryPath: path}
	_, _, err := launchGuardChildWithBroker([]string{"agent"}, nil, false, meta, toolprocgate.NewSpawnBroker(), func(toolprocgate.SpawnGrant) (*exec.Cmd, error) { return nil, errors.New("boom") })
	if err == nil {
		t.Fatal("launch admitted")
	}
	rows, readErr := (sessionregistry.Store{Path: path}).ReadAll()
	if readErr != nil || len(rows) != 1 || rows[0].State != sessionregistry.StateFailed || rows[0].Reason != "launch_prepare_failed" {
		t.Fatalf("rows=%+v err=%v", rows, readErr)
	}
}
