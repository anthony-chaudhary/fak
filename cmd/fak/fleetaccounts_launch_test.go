package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accountprobe"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

func TestFleetLaunchLedgerCapturesDecisionWithoutSecrets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "launches.jsonl")
	d := fleetaccounts.LaunchDecision{OK: true, Account: "codex-tier1", Product: "codex", ConfiguredModel: "gpt-5-codex", InvokedModel: "gpt-5-codex", EndpointClass: "subscription", TaskTier: 1, Env: map[string]string{"CODEX_HOME": "/secret/path"}}
	if err := appendFleetLaunchLedger(path, d); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, want := range []string{`"resolved_account":"codex-tier1"`, `"product":"codex"`, `"configured_model":"gpt-5-codex"`, `"invoked_model":"gpt-5-codex"`, `"endpoint_class":"subscription"`, `"task_tier":1`} {
		if !strings.Contains(text, want) {
			t.Errorf("ledger %s missing %s", text, want)
		}
	}
	if strings.Contains(text, "/secret/path") || strings.Contains(text, "CODEX_HOME") {
		t.Fatalf("ledger leaked launch environment: %s", text)
	}
	var row map[string]any
	if err := json.Unmarshal(raw, &row); err != nil {
		t.Fatal(err)
	}
}

func TestExecuteFleetLaunchRejectsPromptHookFalseSuccess(t *testing.T) {
	old := fleetLaunchExecCommand
	fleetLaunchExecCommand = fleetLaunchHelperCommand
	t.Cleanup(func() { fleetLaunchExecCommand = old })
	t.Setenv("GO_WANT_FLEET_LAUNCH_HELPER", "blocked")

	var stdout, stderr bytes.Buffer
	d := fleetaccounts.LaunchDecision{OK: true, Product: "codex", Argv: []string{"fak", "guarded-codex"}}
	if code := executeFleetLaunch(d, nil, &stdout, &stderr, []string{"GO_WANT_FLEET_LAUNCH_HELPER=blocked"}); code != 70 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "PROMPT_HOOK_BLOCK") {
		t.Fatalf("missing typed failure: %q", stderr.String())
	}
}

func TestExecuteFleetLaunchAcceptsCompletedAssistantAndOverlaysAccountHome(t *testing.T) {
	old := fleetLaunchExecCommand
	fleetLaunchExecCommand = fleetLaunchHelperCommand
	t.Cleanup(func() { fleetLaunchExecCommand = old })
	t.Setenv("GO_WANT_FLEET_LAUNCH_HELPER", "success")

	var stdout, stderr bytes.Buffer
	d := fleetaccounts.LaunchDecision{
		OK: true, Product: "codex", Argv: []string{"fak", "guarded-codex"},
		Env: map[string]string{"CODEX_HOME": "/accounts/two"},
	}
	base := []string{"PATH=/bin", "CODEX_HOME=/accounts/one", "GO_WANT_FLEET_LAUNCH_HELPER=success"}
	if code := executeFleetLaunch(d, nil, &stdout, &stderr, base); code != 0 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if strings.Count(stdout.String(), `"type":"item.completed"`) != 1 {
		t.Fatalf("captured event not preserved: %q", stdout.String())
	}
}

func fleetLaunchHelperCommand(_ string, _ ...string) *exec.Cmd {
	return exec.Command(os.Args[0], "-test.run=^TestFleetLaunchHelperProcess$")
}

func TestFleetLaunchHelperProcess(t *testing.T) {
	mode := os.Getenv("GO_WANT_FLEET_LAUNCH_HELPER")
	if mode == "" {
		return
	}
	if got := os.Getenv("CODEX_HOME"); mode == "success" && got != "/accounts/two" {
		_, _ = os.Stderr.WriteString("CODEX_HOME=" + got)
		os.Exit(9)
	}
	if mode == "auth" {
		_, _ = os.Stdout.WriteString(`{"type":"turn.failed","error":{"message":"401 Unauthorized: upstream rejected the credential"}}` + "\n")
		os.Exit(0)
	}
	if mode == "blocked" {
		_, _ = os.Stdout.WriteString("UserPromptSubmit Blocked\n" + `{"type":"turn.completed"}` + "\n")
		os.Exit(0)
	}
	_, _ = os.Stdout.WriteString(`{"type":"item.completed","item":{"type":"agent_message","text":"FAK_OK"}}` + "\n")
	os.Exit(0)
	t.Fatal("helper process returned after os.Exit")
}

func TestOverlayFleetLaunchEnvIsCaseInsensitive(t *testing.T) {
	key := "CODEX_HOME"
	if runtime.GOOS == "windows" {
		key = "Codex_Home"
	}
	got := overlayFleetLaunchEnv([]string{key + "=old", "PATH=/bin"}, map[string]string{"CODEX_HOME": "new"})
	joined := strings.Join(got, "\n")
	if strings.Count(strings.ToUpper(joined), "CODEX_HOME=") != 1 || !strings.Contains(joined, "CODEX_HOME=new") {
		t.Fatalf("overlay=%v", got)
	}
}

func TestExecuteFleetLaunchRecordsGuardedAuthFailure(t *testing.T) {
	old := fleetLaunchExecCommand
	fleetLaunchExecCommand = fleetLaunchHelperCommand
	t.Cleanup(func() { fleetLaunchExecCommand = old })
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)

	var stdout, stderr bytes.Buffer
	d := fleetaccounts.LaunchDecision{OK: true, Account: ".codex-four", Product: "codex", Argv: []string{"fak"}}
	env := []string{"GO_WANT_FLEET_LAUNCH_HELPER=auth"}
	if code := executeFleetLaunch(d, nil, &stdout, &stderr, env); code != 70 {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	entry := accountprobe.LastProbeByAccount(regDir)[".codex-four"]
	if entry.Status != "AUTH" || entry.BlockReason != "UPSTREAM_AUTH" {
		t.Fatalf("probe entry = %+v", entry)
	}
}
