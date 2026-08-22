package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

func TestOrchestrationPersistsActivationBeforeSpawn(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	t.Setenv("CODEX_THREAD_ID", "session-activation")
	old := orchestrationWorkerLauncher
	launches := 0
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		launches++
		persisted, ok := readCodexOrchestrationLaunchReceipt(home, "session-activation")
		if !ok || len(persisted.Activations) != launches {
			t.Fatalf("pre-spawn receipt missing at launch %d: ok=%v receipt=%+v", launches, ok, persisted)
		}
		activation := persisted.Activations[launches-1]
		if activation.ChildID != req.Role.ID || activation.State() != ultracodebench.ActivationUnknown || !activation.Injected {
			t.Fatalf("activation=%+v request=%+v", activation, req)
		}
		raw, _ := json.Marshal(activation)
		for _, forbidden := range []string{"path", "prompt", "account", "host", "settings", "argv"} {
			if strings.Contains(string(raw), forbidden) {
				t.Fatalf("activation retained %q: %s", forbidden, raw)
			}
		}
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 100 + launches, Status: "started"}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = old })

	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--profile", "ultracode", "--task-text", "implement and verify the activation receipt", "--codex-home", home, "--launch", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if launches < 2 {
		t.Fatalf("launches=%d", launches)
	}
}

func TestUltracodeStatusReportsActivationCoverageWithoutPrivatePaths(t *testing.T) {
	home := t.TempDir()
	active, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{RunID: "orch-status", ChildID: "active", Harness: "codex", Requested: ultracodebench.SettingOn, Resolved: ultracodebench.SettingOn, Injected: true})
	if err != nil {
		t.Fatal(err)
	}
	active, err = ultracodebench.Acknowledge(active, ultracodebench.ObservableActive, ultracodebench.SourceExplicitAcknowledgement)
	if err != nil {
		t.Fatal(err)
	}
	degraded, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{RunID: "orch-status", ChildID: "degraded", Harness: "codex", Requested: ultracodebench.SettingOn, Resolved: ultracodebench.SettingOn, Degradations: []string{"harness_cannot_inject"}})
	if err != nil {
		t.Fatal(err)
	}
	unknown, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{RunID: "orch-status", ChildID: "unknown", Harness: "codex", Requested: ultracodebench.SettingOn, Resolved: ultracodebench.SettingOn, Injected: true})
	if err != nil {
		t.Fatal(err)
	}
	inactive, err := ultracodebench.BeforeSpawn(ultracodebench.BeforeSpawnInput{RunID: "orch-status", ChildID: "inactive", Harness: "codex", Requested: ultracodebench.SettingOff, Resolved: ultracodebench.SettingOff})
	if err != nil {
		t.Fatal(err)
	}
	receipt := codexOrchestrationLaunchReceipt{
		Schema: codexOrchestrationLaunchSchema, SessionID: "activation-status", RunID: "orch-status",
		RequestedProfile: "auto", ResolvedProfile: "ultracode", Status: "launched",
		Workers: []codexOrchestrationWorkerLaunch{
			{RoleID: "active", Status: "started", LogPath: `C:\private\active.jsonl`},
			{RoleID: "degraded", Status: "started", LogPath: `C:\private\degraded.jsonl`},
			{RoleID: "unknown", Status: "started", LogPath: `C:\private\unknown.jsonl`},
			{RoleID: "inactive", Status: "started", LogPath: `C:\private\inactive.jsonl`},
		},
		Activations: []ultracodebench.ActivationReceipt{active, degraded, unknown, inactive},
	}
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if code := runUltracodeStatus(&out, &stderr, []string{"--home", home, "--session", "activation-status", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got ultracodeStatus
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != ultracodeStatusSchema || got.Activation.Active != 1 || got.Activation.Degraded != 1 || got.Activation.Unknown != 1 || got.Activation.Inactive != 1 {
		t.Fatalf("status=%+v", got)
	}
	for _, forbidden := range []string{"log_path", "C:\\\\private", "prompt", "account", "host", "raw_settings", "argv"} {
		if strings.Contains(out.String(), forbidden) {
			t.Fatalf("status retained %q:\n%s", forbidden, out.String())
		}
	}
}

func TestUltracodeStatusMarksLegacyChildrenUnknown(t *testing.T) {
	home := t.TempDir()
	receipt := codexOrchestrationLaunchReceipt{
		Schema: codexOrchestrationLaunchSchema, SessionID: "legacy", RunID: "orch-legacy",
		Workers: []codexOrchestrationWorkerLaunch{{RoleID: "worker-1", Status: "started"}},
	}
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		t.Fatal(err)
	}
	got, err := projectUltracodeStatus(home, receipt)
	if err != nil {
		t.Fatal(err)
	}
	if got.Activation.Total != 1 || got.Activation.Unknown != 1 || got.Workers[0].Activation != ultracodebench.ActivationUnknown {
		t.Fatalf("legacy status=%+v", got)
	}
}
