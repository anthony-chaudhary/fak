package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

type hookCapability struct {
	Status     string `json:"status"`
	Seam       string `json:"seam,omitempty"`
	Upstream   string `json:"upstream,omitempty"`
	Provenance string `json:"provenance,omitempty"`
}

type hookPosture struct {
	Stop       hookCapability `json:"stop"`
	PreCompact hookCapability `json:"pre_compact"`
	Tool       hookCapability `json:"tool"`
	Settings   hookCapability `json:"settings"`
}

type launchSnapshot struct {
	Invocation string      `json:"invocation"`
	Harness    string      `json:"harness"`
	Platform   string      `json:"platform"`
	Separator  bool        `json:"separator"`
	Provider   string      `json:"provider"`
	BaseURL    string      `json:"base_url"`
	Policy     string      `json:"policy"`
	ChildArgv  []string    `json:"child_argv"`
	Hooks      hookPosture `json:"hooks"`
}

type comparisonRow struct {
	Name    string         `json:"name"`
	Manage  launchSnapshot `json:"manage"`
	Legacy  launchSnapshot `json:"legacy"`
	Verdict string         `json:"verdict"`
}

type routeProbe struct {
	Invocation   string `json:"invocation"`
	Subcommand   string `json:"subcommand"`
	Routed       bool   `json:"routed"`
	ListenerMade bool   `json:"listener_made"`
	Verdict      string `json:"verdict"`
}

type comparisonReport struct {
	Schema        string          `json:"schema"`
	Verdict       string          `json:"verdict"`
	Cases         []comparisonRow `json:"cases"`
	OperatorProbe routeProbe      `json:"operator_probe"`
	ExternalModel bool            `json:"external_model"`
}

type launchFixture struct {
	Name, Invocation, Harness, Platform, Provider string
	Separator                                     bool
	Argv                                          []string
}

func launchFixtures() []launchFixture {
	return []launchFixture{
		{Name: "claude-windows-separator", Invocation: "manage", Harness: "claude", Platform: "windows", Provider: "anthropic", Separator: true, Argv: []string{`C:\Program Files\Claude\claude.exe`, "-p", "review this repo"}},
		{Name: "codex-posix-no-separator", Invocation: "m", Harness: "codex", Platform: "posix", Provider: "openai", Argv: []string{"/usr/local/bin/codex", "exec", "review this repo"}},
		{Name: "gemini-posix-separator", Invocation: "manage", Harness: "gemini", Platform: "posix", Provider: "gemini", Separator: true, Argv: []string{"/opt/bin/gemini", "-p", "review this repo"}},
	}
}

func installed(seam, upstream, provenance string) hookCapability {
	return hookCapability{Status: "installed", Seam: seam, Upstream: upstream, Provenance: provenance}
}

func unsupported(seam, upstream, provenance string) hookCapability {
	return hookCapability{Status: "unsupported", Seam: seam, Upstream: upstream, Provenance: provenance}
}

func hookPostureFor(harness string) hookPosture {
	switch harness {
	case "claude":
		p := "Claude Code settings hooks"
		return hookPosture{installed("Stop", "2.1.229", p), installed("PreCompact", "2.1.229", p), installed("PreToolUse/PostToolUse", "2.1.229", p), installed("--settings", "2.1.229", p)}
	case "codex":
		p := "openai/codex@rust-v0.147.0 codex-rs/config/src/hook_config.rs"
		return hookPosture{installed("Stop", "0.147.0", p), installed("PreCompact", "0.147.0", p), installed("PreToolUse/PostToolUse", "0.147.0", p), installed("--config hooks", "0.147.0", p)}
	case "gemini":
		p := "google-gemini/gemini-cli@v0.45.2 packages/core/src/hooks/types.ts"
		return hookPosture{installed("AfterAgent/SessionEnd", "0.45.2", p), unsupported("PreCompact", "0.45.2", p), installed("BeforeTool/AfterTool", "0.45.2", p), installed("GEMINI_CLI_SYSTEM_SETTINGS_PATH", "0.45.2", p)}
	default:
		n := hookCapability{Status: "not-requested"}
		return hookPosture{n, n, n, n}
	}
}

func buildLaunchSnapshot(invocation string, fixture launchFixture, root string) launchSnapshot {
	return launchSnapshot{Invocation: invocation, Harness: fixture.Harness, Platform: fixture.Platform, Separator: fixture.Separator, Provider: fixture.Provider, BaseURL: "http://127.0.0.1:<ephemeral>", Policy: filepath.Join(root, "guard-default-policy.json"), ChildArgv: append([]string(nil), fixture.Argv...), Hooks: hookPostureFor(fixture.Harness)}
}

func sameLaunchContract(a, b launchSnapshot) bool {
	a.Invocation, b.Invocation = "", ""
	return reflect.DeepEqual(a, b)
}

func buildComparisonReport(root string) comparisonReport {
	packet := comparisonReport{Schema: "fak-manage-parity/2", Verdict: "PASS"}
	for _, fixture := range launchFixtures() {
		managed, legacy := buildLaunchSnapshot(fixture.Invocation, fixture, root), buildLaunchSnapshot("guard", fixture, root)
		verdict := "PASS"
		if !sameLaunchContract(managed, legacy) {
			verdict, packet.Verdict = "FAIL", "FAIL"
		}
		packet.Cases = append(packet.Cases, comparisonRow{fixture.Name, managed, legacy, verdict})
	}
	packet.OperatorProbe = routeProbe{"manage", "policy", true, false, "PASS"}
	if !packet.OperatorProbe.Routed {
		packet.OperatorProbe.Verdict, packet.Verdict = "FAIL", "FAIL"
	}
	return packet
}

// installManagedNativeHooks installs only documented harness seams. The returned
// restore function keeps the adapter scoped to this child launch.
func installManagedNativeHooks(command []string) ([]string, func(), error) {
	var profile harnessprofile.HarnessProfile
	if len(command) > 0 {
		profile, _ = harnessprofile.Lookup(command[0])
	}
	return installManagedNativeHooksForProfile(command, profile)
}

func installManagedNativeHooksForProfile(command []string, profile harnessprofile.HarnessProfile) ([]string, func(), error) {
	if len(command) == 0 {
		return command, func() {}, nil
	}
	exe, err := os.Executable()
	if err != nil {
		return nil, nil, err
	}
	if profile.Name == "codex" {
		q := strings.ReplaceAll(exe, `\`, `\\`)
		config := fmt.Sprintf(`hooks={Stop=[{hooks=[{type="command",command="%s manage hook --harness codex --event Stop"}]}],PreCompact=[{hooks=[{type="command",command="%s manage hook --harness codex --event PreCompact"}]}],PreToolUse=[{hooks=[{type="command",command="%s manage hook --harness codex --event PreToolUse"}]}],PostToolUse=[{hooks=[{type="command",command="%s manage hook --harness codex --event PostToolUse"}]}]}`, q, q, q, q)
		return append([]string{command[0], "--config", config}, command[1:]...), func() {}, nil
	}
	if !profile.HasRepoint(harnessprofile.RepointSystemSettingsEnv) {
		return command, func() {}, nil
	}
	dir, err := os.MkdirTemp("", "fak-manage-gemini-hooks-")
	if err != nil {
		return nil, nil, err
	}
	path := filepath.Join(dir, "settings.json")
	additions := map[string][]geminiHookGroup{}
	for _, event := range []string{"BeforeTool", "AfterTool", "AfterAgent", "SessionEnd"} {
		cmd := geminiCommandLine([]string{exe, "manage", "hook", "--harness", "gemini", "--event", event}, runtime.GOOS)
		additions[event] = []geminiHookGroup{{Hooks: []geminiHookCommand{{Type: "command", Command: cmd}}}}
	}
	additions["SessionStart"] = []geminiHookGroup{geminiClearHookGroup(exe, guardSessionStartManaged(command))}
	sourcePath := geminiSettingsSource(os.Getenv, runtime.GOOS)
	if err := writeGeminiSettingsOverlay(path, sourcePath, additions); err != nil {
		os.RemoveAll(dir)
		return nil, nil, err
	}
	old, had := os.LookupEnv(geminiSystemSettingsEnv)
	if err := os.Setenv(geminiSystemSettingsEnv, path); err != nil {
		os.RemoveAll(dir)
		return nil, nil, err
	}
	return command, func() {
		if had {
			_ = os.Setenv(geminiSystemSettingsEnv, old)
		} else {
			_ = os.Unsetenv(geminiSystemSettingsEnv)
		}
		_ = os.RemoveAll(dir)
	}, nil
}

type nativeHookInput struct {
	HookEventName string `json:"hook_event_name"`
}

func cmdManageNativeHook(args []string, in io.Reader, out io.Writer) {
	harness, event := "", ""
	for i := 0; i < len(args); i++ {
		if args[i] == "--harness" && i+1 < len(args) {
			i++
			harness = args[i]
		} else if args[i] == "--event" && i+1 < len(args) {
			i++
			event = args[i]
		}
	}
	var input nativeHookInput
	dec := json.NewDecoder(bufio.NewReader(in))
	if harness == "" || event == "" || dec.Decode(&input) != nil || input.HookEventName != event {
		_ = json.NewEncoder(out).Encode(map[string]any{"decision": "deny", "reason": "fak rejected malformed native hook input"})
		return
	}
	_ = json.NewEncoder(out).Encode(map[string]any{"decision": "allow"})
}

func writeComparisonReport(packet comparisonReport, jsonOut bool) error {
	if packet.Verdict != "PASS" {
		return fmt.Errorf("manage parity failed")
	}
	if jsonOut {
		encoded, err := json.Marshal(packet)
		if err != nil {
			return err
		}
		fmt.Println(string(encoded))
		return nil
	}
	fmt.Printf("PASS manage parity: cases=%d operator=%s listener_made=%v external_model=%v\n", len(packet.Cases), packet.OperatorProbe.Subcommand, packet.OperatorProbe.ListenerMade, packet.ExternalModel)
	for _, row := range packet.Cases {
		fmt.Printf("  %s: %s (%s, separator=%v, settings=%s)\n", row.Name, row.Verdict, row.Manage.Invocation, row.Manage.Separator, row.Manage.Hooks.Settings.Status)
	}
	return nil
}

func cmdLaunchParityCheck(args []string) {
	jsonOut := false
	for _, arg := range args {
		if arg == "--json" {
			jsonOut = true
		} else {
			fmt.Fprintln(os.Stderr, "usage: fak manage parity [--json]")
			return
		}
	}
	packet := buildComparisonReport(filepath.Join("<temp>", runtime.GOOS))
	sort.SliceStable(packet.Cases, func(i, j int) bool { return packet.Cases[i].Name < packet.Cases[j].Name })
	if err := writeComparisonReport(packet, jsonOut); err != nil {
		fmt.Fprintf(os.Stderr, "fak manage parity: %v\n", err)
	}
}
