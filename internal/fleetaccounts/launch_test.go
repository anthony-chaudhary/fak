package fleetaccounts

import (
	"reflect"
	"testing"
)

func resolved(product, account, model, dir string, tier int) Resolved {
	return Resolved{OK: true, Account: account, Product: product, Model: model, ModelTier: &tier, ConfigDir: dir}
}

func TestDecideLaunchClaudeSpeedPosture(t *testing.T) {
	seat := resolved("claude", "claude-tier1", "opus", "/accounts/claude", 1)
	tests := []struct {
		name, speed string
		wantFast    bool
	}{
		{name: "fast", speed: "fast", wantFast: true},
		{name: "standard", speed: "standard", wantFast: false},
		{name: "auto interactive", speed: "auto", wantFast: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := DecideLaunch(LaunchRequest{Account: seat, TaskTier: 1, Prompt: "interactive task", Speed: tt.speed})
			if !d.OK {
				t.Fatalf("decision=%+v", d)
			}
			hasFast := false
			for i := 0; i+1 < len(d.Argv); i++ {
				if d.Argv[i] == "--settings" && d.Argv[i+1] == `{"fastMode":true}` {
					hasFast = true
				}
			}
			if hasFast != tt.wantFast || d.Speed != map[bool]string{true: "fast", false: "standard"}[tt.wantFast] {
				t.Fatalf("argv=%v speed=%q", d.Argv, d.Speed)
			}
		})
	}
}

func TestDecideLaunchSpeedIgnoredForNonClaude(t *testing.T) {
	d := DecideLaunch(LaunchRequest{Account: resolved("codex", "codex-tier1", "gpt", "/accounts/codex", 1), TaskTier: 1, Prompt: "task", Speed: "fast"})
	if d.Speed != "" {
		t.Fatalf("codex speed=%q, want empty", d.Speed)
	}
}
func TestDecideLaunchAllProductsCaptureArgvEnvAndMetadata(t *testing.T) {
	tests := []struct{ product, envKey, binary string }{
		{"claude", "CLAUDE_CONFIG_DIR", "claude"},
		{"codex", "CODEX_HOME", "codex"},
		{"opencode", "XDG_CONFIG_HOME", "opencode"},
	}
	for _, tt := range tests {
		t.Run(tt.product, func(t *testing.T) {
			d := DecideLaunch(LaunchRequest{Account: resolved(tt.product, tt.product+"-seat", "configured", "/accounts/"+tt.product, 1), TaskTier: 1, InvokedModel: "invoked", Prompt: "fix #6534"})
			if !d.OK || !launchesBinary(d.Argv, tt.binary) || d.Env[tt.envKey] == "" {
				t.Fatalf("decision = %#v", d)
			}
			if d.Account != tt.product+"-seat" || d.ConfiguredModel != "configured" || d.InvokedModel != "invoked" || d.TaskTier != 1 {
				t.Fatalf("metadata = %#v", d)
			}
			if reflect.DeepEqual(d.Env, map[string]string{}) {
				t.Fatal("missing account environment")
			}
		})
	}
}

func TestHardEngineeringNeverLaunchesRestrictedTier3OpenCode(t *testing.T) {
	restricted := resolved("opencode", "nemo-tier3", "nvidia/nemotron", "/accounts/nemo/opencode", 3)
	for _, override := range []bool{false, true} {
		d := DecideLaunch(LaunchRequest{Account: restricted, TaskTier: 1, Prompt: "hard engineering", Tier3Override: override})
		if d.OK {
			t.Fatalf("override=%v launched restricted seat: %#v", override, d)
		}
	}
	codex := DecideLaunch(LaunchRequest{Account: resolved("codex", "codex-tier1", "gpt-5-codex", "/accounts/codex", 1), TaskTier: 1, Prompt: "hard engineering"})
	if !codex.OK || codex.Product != "codex" {
		t.Fatalf("codex decision = %#v", codex)
	}
}

func TestTier3OverrideIsExplicitAndNarrow(t *testing.T) {
	seat := resolved("opencode", "nemo-tier3", "nvidia/nemotron", "/accounts/nemo/opencode", 3)
	if got := DecideLaunch(LaunchRequest{Account: seat, TaskTier: 3, Prompt: "narrow docs"}); got.OK {
		t.Fatalf("unoverridden tier3 launched: %#v", got)
	}
	got := DecideLaunch(LaunchRequest{Account: seat, TaskTier: 3, Prompt: "narrow docs", Tier3Override: true})
	if !got.OK || !got.OperatorOverride {
		t.Fatalf("override decision = %#v", got)
	}
}

func TestDecideLaunchCodexUsesGuardedHookCompatibleExec(t *testing.T) {
	seat := resolved("codex", "codex-four", "gpt-5-codex", "/accounts/codex-four", 1)
	d := DecideLaunch(LaunchRequest{Account: seat, TaskTier: 1, Prompt: "return ok"})
	want := []string{
		"fak", "guard", "--provider", "openai-responses",
		"--codex-home", "/accounts/codex-four", "--codex-loop-gate", "off",
		"--", "codex", "exec", "--dangerously-bypass-hook-trust", "--json",
		"--model", "gpt-5-codex", "return ok",
	}
	if !d.OK || !reflect.DeepEqual(d.Argv, want) {
		t.Fatalf("decision argv = %#v, want %#v", d.Argv, want)
	}
	if d.Env["CODEX_HOME"] != "/accounts/codex-four" {
		t.Fatalf("CODEX_HOME = %q", d.Env["CODEX_HOME"])
	}
}

func TestDecideLaunchCodexAccountsRemainProcessIsolated(t *testing.T) {
	one := DecideLaunch(LaunchRequest{Account: resolved("codex", "one", "gpt", "/accounts/one", 1), TaskTier: 1, Prompt: "one"})
	two := DecideLaunch(LaunchRequest{Account: resolved("codex", "two", "gpt", "/accounts/two", 1), TaskTier: 1, Prompt: "two"})
	if one.Env["CODEX_HOME"] == two.Env["CODEX_HOME"] || one.Argv[5] == two.Argv[5] {
		t.Fatalf("account launches cross-wired: one=%#v two=%#v", one, two)
	}
	if one.Env["CODEX_HOME"] != "/accounts/one" || two.Env["CODEX_HOME"] != "/accounts/two" {
		t.Fatalf("isolated homes lost: one=%q two=%q", one.Env["CODEX_HOME"], two.Env["CODEX_HOME"])
	}
}

func launchesBinary(argv []string, binary string) bool {
	for _, arg := range argv {
		if arg == binary {
			return true
		}
	}
	return false
}
