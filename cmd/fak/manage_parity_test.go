package main

import (
	"bytes"
	"encoding/json"
	"fmt"
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
	old, had := os.LookupEnv(geminiSystemSettingsEnv)
	defer func() {
		if had {
			_ = os.Setenv(geminiSystemSettingsEnv, old)
		} else {
			_ = os.Unsetenv(geminiSystemSettingsEnv)
		}
	}()
	source := filepath.Join(t.TempDir(), "system-settings.json")
	sourceData := []byte(`{
  // Existing system policy remains effective in the launch overlay.
  "security": {"auth": {"selectedType": "gemini-api-key"}},
  "hooks": {"SessionStart": [{"matcher": "startup", "hooks": [{"type": "command", "command": "existing-hook"}]}]}
}`)
	if err := os.WriteFile(source, sourceData, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Setenv(geminiSystemSettingsEnv, source); err != nil {
		t.Fatal(err)
	}
	_, restore, err = installManagedNativeHooks([]string{"gemini", "-p", "ok"})
	if err != nil {
		t.Fatal(err)
	}
	path := os.Getenv(geminiSystemSettingsEnv)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(path) != "settings.json" || !bytes.Contains(data, []byte(`"BeforeTool"`)) || !bytes.Contains(data, []byte(`"AfterAgent"`)) {
		t.Fatalf("Gemini settings = %s", data)
	}
	var settings struct {
		Security struct {
			Auth struct {
				SelectedType string `json:"selectedType"`
			} `json:"auth"`
		} `json:"security"`
		Hooks map[string][]geminiHookGroup `json:"hooks"`
	}
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("parse Gemini settings: %v", err)
	}
	if settings.Security.Auth.SelectedType != "gemini-api-key" {
		t.Fatalf("Gemini overlay dropped existing auth policy: %+v", settings.Security)
	}
	start := settings.Hooks["SessionStart"]
	if len(start) != 2 || start[0].Matcher != "startup" || start[1].Matcher != "clear" || len(start[1].Hooks) != 1 {
		t.Fatalf("Gemini SessionStart hook = %+v", start)
	}
	if command := start[1].Hooks[0].Command; !strings.Contains(command, "guard-sessionstart") || !strings.Contains(command, "--provider") || !strings.Contains(command, "gemini") {
		t.Fatalf("Gemini SessionStart command = %q", command)
	}
	restore()
	if got := os.Getenv(geminiSystemSettingsEnv); got != source {
		t.Fatalf("Gemini settings env restored to %q, want %q", got, source)
	}
	if after, err := os.ReadFile(source); err != nil || !bytes.Equal(after, sourceData) {
		t.Fatalf("persistent Gemini settings changed: err=%v after=%s", err, after)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("temporary Gemini settings survived restore: %v", err)
	}
}

func TestManageNativeHookCodexSuccessJSON(t *testing.T) {
	for _, event := range []string{"Stop", "PreCompact", "PreToolUse", "PostToolUse"} {
		t.Run(event, func(t *testing.T) {
			var out bytes.Buffer
			input := fmt.Sprintf(`{"hook_event_name":%q}`, event)
			cmdManageNativeHook([]string{"--harness", "codex", "--event", event}, strings.NewReader(input), &out)
			if got, want := out.String(), "{}\n"; got != want {
				t.Fatalf("Codex %s success JSON = %q, want %q", event, got, want)
			}
		})
	}
}

func TestManageNativeHookCodexMalformedInputJSON(t *testing.T) {
	tests := []struct {
		event string
		want  string
	}{
		{"Stop", `{"decision":"block","reason":"fak rejected malformed native hook input"}` + "\n"},
		{"PreToolUse", `{"decision":"block","reason":"fak rejected malformed native hook input"}` + "\n"},
		{"PostToolUse", `{"decision":"block","reason":"fak rejected malformed native hook input"}` + "\n"},
		{"PreCompact", `{"continue":false,"stopReason":"fak rejected malformed native hook input"}` + "\n"},
	}
	for _, test := range tests {
		t.Run(test.event, func(t *testing.T) {
			var out bytes.Buffer
			cmdManageNativeHook([]string{"--harness", "codex", "--event", test.event}, strings.NewReader(`{"hook_event_name":"wrong"}`), &out)
			if got := out.String(); got != test.want {
				t.Fatalf("Codex %s malformed-input JSON = %q, want %q", test.event, got, test.want)
			}
		})
	}
}

func TestManageNativeHookPreservesGeminiResponses(t *testing.T) {
	var out bytes.Buffer
	cmdManageNativeHook([]string{"--harness", "gemini", "--event", "BeforeTool"}, strings.NewReader(`{"hook_event_name":"wrong"}`), &out)
	want := `{"decision":"deny","reason":"fak rejected malformed native hook input"}` + "\n"
	if got := out.String(); got != want {
		t.Fatalf("Gemini malformed-input JSON = %q, want %q", got, want)
	}
	out.Reset()
	cmdManageNativeHook([]string{"--harness", "gemini", "--event", "BeforeTool"}, strings.NewReader(`{"hook_event_name":"BeforeTool"}`), &out)
	want = `{"decision":"allow"}` + "\n"
	if got := out.String(); got != want {
		t.Fatalf("Gemini valid-input JSON = %q, want %q", got, want)
	}
}
