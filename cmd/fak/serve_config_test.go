package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
)

func parseEffectiveServeConfig(t *testing.T, manifestText string, argv ...string) effectiveServeConfigReport {
	t.Helper()
	fs, sf := newServeFlagSet()
	var m deploymanifest.Manifest
	hasManifest := false
	if manifestText != "" {
		path := filepath.Join(t.TempDir(), "fak.toml")
		if err := os.WriteFile(path, []byte(manifestText), 0o600); err != nil {
			t.Fatal(err)
		}
		var err error
		m, err = deploymanifest.Load(path)
		if err != nil {
			t.Fatal(err)
		}
		applyServeManifestDefaults(sf, m)
		hasManifest = true
		argv = append([]string{"--config", path}, argv...)
	}
	if err := fs.Parse(argv); err != nil {
		t.Fatal(err)
	}
	return effectiveServeConfig(sf, m, hasManifest, explicitFlagNames(fs))
}

func TestServeEffectiveConfigPreservesBuiltInDefaults(t *testing.T) {
	rep := parseEffectiveServeConfig(t, "")
	for name, wantSource := range map[string]string{
		"addr": "built-in", "policy": "built-in", "require_key_env": "built-in", "context_budget_tokens": "built-in",
	} {
		if got := rep.Values[name].Source; got != wantSource {
			t.Fatalf("%s source = %q, want %q", name, got, wantSource)
		}
	}
	if got := rep.Values["addr"].Value; got != "127.0.0.1:8080" {
		t.Fatalf("addr = %v, want unchanged zero-config default", got)
	}
}

func TestServeManifestDefaultsAndExplicitFlagsWin(t *testing.T) {
	manifest := `[policy]
floor = "team-policy.json"
[auth]
require_key_env = "TEAM_GATEWAY_KEY"
[budgets]
default_tokens = 12000
[observability]
bind = "127.0.0.1:9191"
`
	rep := parseEffectiveServeConfig(t, manifest,
		"--policy", "personal-policy.json",
		"--context-budget-tokens", "0", // Explicitly choosing the built-in value must still win.
	)
	checks := map[string]effectiveConfigValue{
		"addr":                  {Value: "127.0.0.1:9191", Source: "manifest"},
		"policy":                {Value: "personal-policy.json", Source: "flag"},
		"require_key_env":       {Value: "TEAM_GATEWAY_KEY", Source: "manifest"},
		"context_budget_tokens": {Value: 0, Source: "flag"},
	}
	for name, want := range checks {
		got := rep.Values[name]
		// Normalize JSON-number-shaped interface values for a useful structural assertion.
		gb, _ := json.Marshal(got)
		wb, _ := json.Marshal(want)
		if string(gb) != string(wb) {
			t.Errorf("%s = %s, want %s", name, gb, wb)
		}
	}
}

func TestServeConfigPathIsExplicitAndUnambiguous(t *testing.T) {
	if got, err := serveConfigPath([]string{"--model", "x"}); err != nil || got != "" {
		t.Fatalf("ambient path = %q, %v; want empty", got, err)
	}
	if got, err := serveConfigPath([]string{"--config=team.toml"}); err != nil || got != "team.toml" {
		t.Fatalf("equals path = %q, %v", got, err)
	}
	if _, err := serveConfigPath([]string{"--config", "a", "--config", "b"}); err == nil {
		t.Fatal("duplicate --config accepted")
	}
}

func TestServeConfigRejectsUnknownKeyWithNamedReason(t *testing.T) {
	_, err := deploymanifest.Parse([]byte("[auth]\nrequre_key_env = \"TYPO\"\n"))
	loadErr, ok := err.(*deploymanifest.LoadError)
	if !ok || loadErr.Reason != deploymanifest.ReasonUnknownKey {
		t.Fatalf("error = %#v, want UNKNOWN_KEY LoadError", err)
	}
}
