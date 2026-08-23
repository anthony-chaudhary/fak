package devcmd

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestClassifyEffectiveHookStates(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "hooks", "hooks.json")
	if err := os.MkdirAll(filepath.Join(root, "bin"), 0755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(root, "bin", "dos-hook"), []byte("x"), 0755)
	_ = os.WriteFile(filepath.Join(root, "bin", "dos-hook-"+runtimeGOOS()+"-"+runtimeGOARCH()+exeSuffix()), []byte("x"), 0755)
	tests := []struct {
		name        string
		enabled     bool
		trust, want string
	}{{"ok", true, "trusted", "effective"}, {"disabled", false, "trusted", "disabled"}, {"changed", true, "modified", "stale_hash"}, {"untrusted", true, "untrusted", "untrusted"}, {"unknown", true, "mystery", "unknown"}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := effectiveHook{Enabled: tt.enabled, TrustStatus: tt.trust, PluginID: "p", SourcePath: source, Command: "$root/bin/dos-hook"}
			classifyEffectiveHook(&h, root)
			if h.State != tt.want {
				t.Fatalf("state=%s want %s", h.State, tt.want)
			}
		})
	}
}
func runtimeGOOS() string   { return runtime.GOOS }
func runtimeGOARCH() string { return runtime.GOARCH }
func exeSuffix() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}
func TestNormalizeHookEvent(t *testing.T) {
	for in, want := range map[string]string{"PreToolUse": "pre_tool_use", "post_tool_use": "post_tool_use", "SubagentStop": "subagent_stop", "StopFailure": "stop_failure"} {
		if got := normalizeHookEvent(in); got != want {
			t.Fatalf("%s => %s", in, got)
		}
	}
}

func TestObservedHookEventsIncludeStopFamily(t *testing.T) {
	for _, event := range []string{"pre_tool_use", "post_tool_use", "stop", "subagent_stop", "stop_failure"} {
		if !isObservedHookEvent(event) {
			t.Fatalf("%s was not observed", event)
		}
	}
	if isObservedHookEvent("session_start") {
		t.Fatal("unrelated lifecycle event entered tool/Stop profile")
	}
}

func TestHookCommandPlatformCompatibility(t *testing.T) {
	tests := []struct {
		name, command, goos string
		want                bool
	}{
		{
			name:    "windows rejects selected POSIX stop command",
			command: `root="${CODEX_PLUGIN_ROOT:-}"; command -p sh "$root/bin/dos-hook" stop 2>/dev/null || python3 -m dos.cli hook stop`,
			goos:    "windows",
			want:    true,
		},
		{
			name:    "windows accepts native adapter",
			command: `& (Join-Path $env:PLUGIN_ROOT 'bin\dos-hook-codex.ps1') stop --workspace .; exit $LASTEXITCODE`,
			goos:    "windows",
		},
		{
			name:    "linux accepts POSIX command",
			command: `root="${CODEX_PLUGIN_ROOT:-}"; command -p sh "$root/bin/dos-hook" stop`,
			goos:    "linux",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := hookCommandPlatformIncompatible(tt.command, tt.goos); got != tt.want {
				t.Fatalf("incompatible=%v want %v", got, tt.want)
			}
		})
	}
}

func TestClassifyEffectiveHookRejectsIncompatibleWindowsCommand(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("classification follows the current host platform")
	}
	h := effectiveHook{
		Enabled:     true,
		TrustStatus: "trusted",
		Command:     `root="${CODEX_PLUGIN_ROOT:-}"; command -p sh "$root/bin/dos-hook" stop`,
	}
	classifyEffectiveHook(&h, t.TempDir())
	if h.State != "platform_incompatible" {
		t.Fatalf("state=%s want platform_incompatible", h.State)
	}
	if h.Remediation == "" {
		t.Fatal("platform incompatibility needs actionable remediation")
	}
}
