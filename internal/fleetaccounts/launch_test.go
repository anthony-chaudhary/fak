package fleetaccounts

import (
	"reflect"
	"testing"
)

func resolved(product, account, model, dir string, tier int) Resolved {
	return Resolved{OK: true, Account: account, Product: product, Model: model, ModelTier: &tier, ConfigDir: dir}
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
			if !d.OK || d.Argv[0] != tt.binary || d.Env[tt.envKey] == "" {
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
