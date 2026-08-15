package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestManageParityPacketCoversPopularHarnessesAndBoundaries(t *testing.T) {
	packet := buildComparisonReport("/tmp/fak-manage-parity")
	if packet.Verdict != "PASS" {
		t.Fatalf("parity verdict = %s: %+v", packet.Verdict, packet)
	}
	if len(packet.Cases) != 3 {
		t.Fatalf("cases = %d, want 3", len(packet.Cases))
	}
	seen := map[string]bool{}
	seenPlatform := map[string]bool{}
	seenSeparator := map[bool]bool{}
	seenAlias := false
	for _, row := range packet.Cases {
		seen[row.Manage.Harness] = true
		seenPlatform[row.Manage.Platform] = true
		seenSeparator[row.Manage.Separator] = true
		seenAlias = seenAlias || row.Manage.Invocation == "m"
		if row.Verdict != "PASS" || !sameLaunchContract(row.Manage, row.Legacy) {
			t.Fatalf("case %s diverged: %+v", row.Name, row)
		}
		if row.Manage.Provider == "" || row.Manage.BaseURL == "" || row.Manage.Policy == "" || len(row.Manage.ChildArgv) == 0 {
			t.Fatalf("case %s omitted launch contract: %+v", row.Name, row.Manage)
		}
	}
	for _, harness := range []string{"claude", "codex", "gemini"} {
		if !seen[harness] {
			t.Errorf("missing harness %s", harness)
		}
	}
	if !seenPlatform["windows"] || !seenPlatform["posix"] || !seenSeparator[true] || !seenSeparator[false] || !seenAlias {
		t.Fatalf("argv/alias coverage incomplete: platform=%v separator=%v alias=%v", seenPlatform, seenSeparator, seenAlias)
	}
	if packet.Cases[0].Manage.Hooks.Settings.Status != "installed" {
		t.Fatal("Claude native hook posture was not captured")
	}
	for _, row := range packet.Cases {
		if row.Manage.Hooks.Settings.Status == "" {
			t.Fatalf("%s has untyped settings posture", row.Name)
		}
	}
	if !packet.OperatorProbe.Routed || packet.OperatorProbe.ListenerMade || packet.OperatorProbe.Verdict != "PASS" {
		t.Fatalf("operator probe failed: %+v", packet.OperatorProbe)
	}
	if packet.ExternalModel {
		t.Fatal("dry-run parity packet claimed external model traffic")
	}
	if _, err := json.Marshal(packet); err != nil {
		t.Fatalf("packet is not machine readable: %v", err)
	}
}

func TestManageParityDetectsSemanticDrift(t *testing.T) {
	packet := buildComparisonReport("/tmp/fak-manage-parity")
	managed, legacy := packet.Cases[0].Manage, packet.Cases[0].Legacy
	managed.Provider = "openai"
	if sameLaunchContract(managed, legacy) {
		t.Fatal("provider drift was accepted")
	}
}

func TestInstallManagedNativeHooksUsesSupportedSeams(t *testing.T) {
	codex, restore, err := installManagedNativeHooks([]string{"codex", "exec", "ok"})
	if err != nil {
		t.Fatal(err)
	}
	restore()
	joined := strings.Join(codex, " ")
	for _, event := range []string{"Stop", "PreCompact", "PreToolUse", "PostToolUse"} {
		if !strings.Contains(joined, event) {
			t.Errorf("Codex adapter missing %s: %s", event, joined)
		}
	}
	old, had := os.LookupEnv("GEMINI_CLI_SYSTEM_SETTINGS_PATH")
	defer func() {
		if had {
			_ = os.Setenv("GEMINI_CLI_SYSTEM_SETTINGS_PATH", old)
		} else {
			_ = os.Unsetenv("GEMINI_CLI_SYSTEM_SETTINGS_PATH")
		}
	}()
	_, restore, err = installManagedNativeHooks([]string{"gemini", "-p", "ok"})
	if err != nil {
		t.Fatal(err)
	}
	path := os.Getenv("GEMINI_CLI_SYSTEM_SETTINGS_PATH")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(path) != "settings.json" || !bytes.Contains(data, []byte(`"BeforeTool"`)) || !bytes.Contains(data, []byte(`"AfterAgent"`)) {
		t.Fatalf("Gemini settings = %s", data)
	}
	restore()
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("temporary Gemini settings survived restore: %v", err)
	}
}

func TestManageNativeHookFailsClosedOnMalformedInput(t *testing.T) {
	var out bytes.Buffer
	cmdManageNativeHook([]string{"--harness", "gemini", "--event", "BeforeTool"}, strings.NewReader(`{"hook_event_name":"wrong"}`), &out)
	if !strings.Contains(out.String(), `"decision":"deny"`) {
		t.Fatalf("malformed input did not deny: %s", out.String())
	}
	out.Reset()
	cmdManageNativeHook([]string{"--harness", "gemini", "--event", "BeforeTool"}, strings.NewReader(`{"hook_event_name":"BeforeTool"}`), &out)
	if !strings.Contains(out.String(), `"decision":"allow"`) {
		t.Fatalf("valid input did not allow: %s", out.String())
	}
}
