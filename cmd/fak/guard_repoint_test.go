package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

// TestGuardRepointDrivenByProfile pins that repoint selection reads profile.Repoint: the
// built-in agents get exactly the mechanisms they get today (codex -> cli-config, claude ->
// settings-file, opencode -> neither, and env for all), and — the C3 payoff — a SYNTHETIC
// profile declaring a mechanism is selected with NO new `if guardIsX`.
func TestGuardRepointDrivenByProfile(t *testing.T) {
	// Built-in agents, resolved through the command-facing wrapper.
	builtins := []struct {
		command      string
		wantEnv      bool
		wantCLI      bool
		wantSettings bool
	}{
		{"claude", true, false, true},
		{"claude-code", true, false, true},
		{"codex", true, true, false},
		{"opencode", true, false, false},
		{"aider", true, false, false},
		{"vim", true, false, false},             // unrecognized -> env only
		{"", true, false, false},                // empty -> env only (guardInjectedEnv still fires)
		{`C:\bin\codex.exe`, true, true, false}, // normalization matches Lookup
	}
	for _, tc := range builtins {
		if got := guardProfileHasRepoint(tc.command, harnessprofile.RepointEnv); got != tc.wantEnv {
			t.Errorf("guardProfileHasRepoint(%q, env) = %v, want %v", tc.command, got, tc.wantEnv)
		}
		if got := guardProfileHasRepoint(tc.command, harnessprofile.RepointCLIConfig); got != tc.wantCLI {
			t.Errorf("guardProfileHasRepoint(%q, cli-config) = %v, want %v", tc.command, got, tc.wantCLI)
		}
		if got := guardProfileHasRepoint(tc.command, harnessprofile.RepointSettingsFile); got != tc.wantSettings {
			t.Errorf("guardProfileHasRepoint(%q, settings-file) = %v, want %v", tc.command, got, tc.wantSettings)
		}
	}

	// The C3 fence: a brand-new harness that declares settings-file + env is selected purely
	// from its Repoint set — no code names it. If this required a new `if guardIsX`, the
	// dispatcher would not be data-driven.
	synthetic := harnessprofile.HarnessProfile{
		Name:    "acme-cli",
		Names:   []string{"acme-cli"},
		Wire:    harnessprofile.WireAnthropic,
		Repoint: []harnessprofile.RepointMechanism{harnessprofile.RepointEnv, harnessprofile.RepointSettingsFile},
	}
	if !guardRepointWants(synthetic, true, harnessprofile.RepointSettingsFile) {
		t.Error("a synthetic profile declaring settings-file must be selected with no name check")
	}
	if guardRepointWants(synthetic, true, harnessprofile.RepointCLIConfig) {
		t.Error("the synthetic profile does not declare cli-config; it must not be selected")
	}
	if !guardRepointWants(synthetic, true, harnessprofile.RepointEnv) {
		t.Error("the synthetic profile declares env; it must be selected")
	}
}

// TestGuardIsCodexStillGatesCodexOnly re-asserts the load-bearing invariant guardIsCodex
// guarded before C3: it is true for codex (the only cli-config profile) and false for every
// other agent, so the codex `-c` overrides never land on a foreign argv — now proven to hold
// when the gate is driven by profile.Repoint instead of a hardcoded base-name check.
func TestGuardIsCodexStillGatesCodexOnly(t *testing.T) {
	cases := map[string]bool{
		"codex":     true,
		"Codex":     true,
		"codex.exe": true,
		"opencode":  false, // contains "codex" as a substring but is NOT codex
		"claude":    false,
		"aider":     false,
		"":          false,
	}
	for command, want := range cases {
		if got := guardIsCodex(command); got != want {
			t.Errorf("guardIsCodex(%q) = %v, want %v", command, got, want)
		}
	}
}
