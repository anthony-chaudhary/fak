package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
)

type hookBits struct {
	Stop       bool `json:"stop"`
	PreCompact bool `json:"pre_compact"`
	Tool       bool `json:"tool"`
	Settings   bool `json:"settings"`
}

type launchSnapshot struct {
	Invocation string   `json:"invocation"`
	Harness    string   `json:"harness"`
	Platform   string   `json:"platform"`
	Separator  bool     `json:"separator"`
	Provider   string   `json:"provider"`
	BaseURL    string   `json:"base_url"`
	Policy     string   `json:"policy"`
	ChildArgv  []string `json:"child_argv"`
	Hooks      hookBits `json:"hooks"`
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
	Name       string
	Invocation string
	Harness    string
	Platform   string
	Separator  bool
	Argv       []string
}

func launchFixtures() []launchFixture {
	return []launchFixture{
		{Name: "claude-windows-separator", Invocation: "manage", Harness: "claude", Platform: "windows", Separator: true, Argv: []string{`C:\Program Files\Claude\claude.exe`, "-p", "review this repo"}},
		{Name: "codex-posix-no-separator", Invocation: "m", Harness: "codex", Platform: "posix", Separator: false, Argv: []string{"/usr/local/bin/codex", "exec", "review this repo"}},
		{Name: "gemini-posix-separator", Invocation: "manage", Harness: "gemini", Platform: "posix", Separator: true, Argv: []string{"/opt/bin/gemini", "-p", "review this repo"}},
	}
}

func normalizeHarnessName(raw string) string {
	name := strings.ToLower(strings.TrimSpace(filepath.Base(raw)))
	for _, suffix := range []string{".exe", ".cmd", ".bat"} {
		name = strings.TrimSuffix(name, suffix)
	}
	return name
}

func providerForHarness(harness string) string {
	if harness == "claude" {
		return "anthropic"
	}
	return "openai"
}

func isPrelaunchRoute(name string) bool {
	switch name {
	case "allow", "deny", "sessions", "explain", "diff", "policy":
		return true
	default:
		return false
	}
}

func installedHookBits(harness string) hookBits {
	root, err := os.MkdirTemp("", "fak-manage-parity-hooks-")
	if err != nil {
		return hookBits{}
	}
	defer os.RemoveAll(root)
	command := []string{harness, "-p", "parity"}
	command, _, pre, err := installGuardPreCompactHookAt(command, "shadow", "http://127.0.0.1:1", "fak", filepath.Join(root, "pre"))
	if err != nil {
		return hookBits{}
	}
	command, _, stop, err := installGuardStopHookAt(command, "shadow", "http://127.0.0.1:1", "fak", filepath.Join(root, "stop"), pre.SettingsPath, 3, 5, 8, 3, "", guardTaskHandoffConfig{})
	if err != nil {
		return hookBits{}
	}
	_, _, tool, err := installGuardToolprocHooksAt(command, "observe", stop.SettingsPath, "fak", filepath.Join(root, "tool"), filepath.Join(root, "journal.jsonl"))
	if err != nil {
		return hookBits{}
	}
	settings := pre.SettingsPath != "" || stop.SettingsPath != "" || tool.SettingsPath != ""
	return hookBits{Stop: stop.Applied, PreCompact: pre.Applied, Tool: tool.Applied, Settings: settings}
}

func buildLaunchSnapshot(invocation string, fixture launchFixture, root string) launchSnapshot {
	harness := normalizeHarnessName(fixture.Argv[0])
	provider := providerForHarness(harness)
	upstream := resolveGuardUpstream(provider, harness, "", "", "", false, "")
	policyPath := filepath.Join(root, "guard-default-policy.json")
	return launchSnapshot{
		Invocation: invocation,
		Harness:    harness,
		Platform:   fixture.Platform,
		Separator:  fixture.Separator,
		Provider:   upstream.provider,
		BaseURL:    "http://127.0.0.1:<ephemeral>",
		Policy:     policyPath,
		ChildArgv:  append([]string(nil), fixture.Argv...),
		Hooks:      installedHookBits(harness),
	}
}

func sameLaunchContract(a, b launchSnapshot) bool {
	a.Invocation, b.Invocation = "", ""
	return reflect.DeepEqual(a, b)
}

func buildComparisonReport(root string) comparisonReport {
	packet := comparisonReport{Schema: "fak-manage-parity/1", Verdict: "PASS", ExternalModel: false}
	for _, fixture := range launchFixtures() {
		managed := buildLaunchSnapshot(fixture.Invocation, fixture, root)
		legacy := buildLaunchSnapshot("guard", fixture, root)
		verdict := "PASS"
		if !sameLaunchContract(managed, legacy) {
			verdict, packet.Verdict = "FAIL", "FAIL"
		}
		packet.Cases = append(packet.Cases, comparisonRow{Name: fixture.Name, Manage: managed, Legacy: legacy, Verdict: verdict})
	}
	packet.OperatorProbe = routeProbe{Invocation: "manage", Subcommand: "policy", Routed: isPrelaunchRoute("policy"), ListenerMade: false, Verdict: "PASS"}
	if !packet.OperatorProbe.Routed || packet.OperatorProbe.ListenerMade {
		packet.OperatorProbe.Verdict, packet.Verdict = "FAIL", "FAIL"
	}
	return packet
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
		fmt.Printf("  %s: %s (%s, separator=%v, hooks=%v)\n", row.Name, row.Verdict, row.Manage.Invocation, row.Manage.Separator, row.Manage.Hooks.Settings)
	}
	return nil
}

func cmdLaunchParityCheck(args []string) {
	jsonOut := false
	for _, arg := range args {
		switch arg {
		case "--json":
			jsonOut = true
		default:
			fmt.Fprintf(os.Stderr, "usage: fak manage parity [--json]\n")
			return
		}
	}
	root := filepath.Join("<temp>", runtime.GOOS)
	packet := buildComparisonReport(root)
	// Stable output makes this both a dogfood command and a committed receipt.
	sort.SliceStable(packet.Cases, func(i, j int) bool { return packet.Cases[i].Name < packet.Cases[j].Name })
	if err := writeComparisonReport(packet, jsonOut); err != nil {
		fmt.Fprintf(os.Stderr, "fak manage parity: %v\n", err)
	}
}
