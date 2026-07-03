package main

import (
	"strings"
	"testing"
)

// TestNewLaunchBrokerAttemptStripsInheritedSecrets is the #2360 wiring witness
// for the dispatch/account/resume spawn surface: it proves the always-on #2358
// secret floor runs inside newLaunchBrokerAttempt, so a credential present in a
// worker's inherited environment never reaches the brokered SpawnAttempt (and
// therefore never reaches the launched child), while the non-secret config the
// worker needs survives and the held-out names are recorded for the audit.
func TestNewLaunchBrokerAttemptStripsInheritedSecrets(t *testing.T) {
	env := map[string]string{
		"PATH":                    "/usr/bin:/bin",
		"DISPATCH_LANE":           "core",
		"FLEET_RESOLVE_ISSUE":     "2358",
		"CLAUDE_CONFIG_DIR":       "/home/agent/.claude",
		"ANTHROPIC_API_KEY":       "sk-ant-live-api-billing-survives",
		"GITHUB_TOKEN":            "ghp_redteam_canary",
		"FAK_CANARY_SECRET":       "sk-proj-redteam-canary",
		"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-redteam-canary",
	}
	a := newLaunchBrokerAttempt("dispatch_tick", "claude", []string{"fak", "guard", "--", "claude"}, env, t.TempDir())

	stripped := []string{"GITHUB_TOKEN", "FAK_CANARY_SECRET", "CLAUDE_CODE_OAUTH_TOKEN"}
	for _, name := range stripped {
		if _, ok := a.Env[name]; ok {
			t.Fatalf("brokered attempt env kept credential %s — inherited-secret leak", name)
		}
	}
	kept := []string{"PATH", "DISPATCH_LANE", "FLEET_RESOLVE_ISSUE", "CLAUDE_CONFIG_DIR", "ANTHROPIC_API_KEY"}
	for _, name := range kept {
		if _, ok := a.Env[name]; !ok {
			t.Fatalf("brokered attempt env dropped legitimate config %s", name)
		}
	}

	// The SpawnAttempt env (exactly what a launcher hands the child) must carry no
	// canary value on any channel.
	for _, kv := range a.Spawn.Env {
		if strings.Contains(kv.Value, "redteam-canary") || strings.Contains(kv.Value, "redteam_canary") {
			t.Fatalf("SpawnAttempt.Env leaked a canary value via %s", kv.Name)
		}
	}

	// The audit metadata records which names the boundary held out — the epic's
	// "operators can inspect which agent boundary denied or quarantined" line.
	recorded := map[string]bool{}
	for _, name := range a.Metadata.StrippedSecretEnv {
		recorded[name] = true
	}
	for _, name := range stripped {
		if !recorded[name] {
			t.Fatalf("audit did not record stripped secret %s: %v", name, a.Metadata.StrippedSecretEnv)
		}
	}
}
