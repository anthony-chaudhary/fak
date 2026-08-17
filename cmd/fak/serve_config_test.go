package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
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

func TestServeManifestDispositionCoversClosedVocabulary(t *testing.T) {
	keys := deploymanifest.KnownKeys()
	if len(serveManifestSpecs) > len(keys) {
		t.Fatalf("disposition specs = %d exceeds known keys = %d", len(serveManifestSpecs), len(keys))
	}
	for _, key := range keys {
		dotted := key.Dotted()
		spec, ok := serveManifestSpecs[dotted]
		if !ok {
			t.Errorf("%s has no serve disposition", dotted)
			continue
		}
		if spec.reason == "" {
			t.Errorf("%s has no disposition reason", dotted)
		}
		// This also proves the typed value projector covers the same vocabulary without panic.
		_ = deploymanifest.Defaults().Value(key)
	}
}

func TestServeManifestOpinionsReportAppliedReservedAndRefused(t *testing.T) {
	m, err := deploymanifest.Parse([]byte(`[runtimes]
gateway = true
agent_runtime = false
model = "upstream"
[policy]
floor = "policy.json"
[observability]
metrics = false
bind = "127.0.0.1:9191"
`))
	if err != nil {
		t.Fatal(err)
	}
	opinions := serveManifestOpinions(m)
	checks := map[string]string{
		"policy.floor":           "applied",
		"observability.bind":     "applied",
		"runtimes.gateway":       "applied",  // invoking serve realizes this topology opinion
		"runtimes.model":         "reserved", // same default opinion is harmless
		"runtimes.agent_runtime": "refused",  // a changed orchestration opinion cannot disappear
		"observability.metrics":  "refused",  // serve cannot disable its metrics surface yet
	}
	for dotted, want := range checks {
		if got := opinions[dotted].Disposition; got != want {
			t.Errorf("%s disposition = %q, want %q", dotted, got, want)
		}
	}
	if err := validateServeManifestOpinions(m); err == nil || !strings.Contains(err.Error(), "CONFIG_OPINION_UNSUPPORTED") || !strings.Contains(err.Error(), "runtimes.agent_runtime=false") {
		t.Fatalf("validation error = %v, want named refusal with field/value", err)
	}
}

func TestServeMinimalManifestIsSafeAndFullyAccounted(t *testing.T) {
	m, err := deploymanifest.Parse(deploymanifest.Minimal())
	if err != nil {
		t.Fatal(err)
	}
	if err := validateServeManifestOpinions(m); err != nil {
		t.Fatalf("fak init output must remain serve-safe: %v", err)
	}
	opinions := serveManifestOpinions(m)
	if len(opinions) != len(deploymanifest.KnownKeys()) {
		t.Fatalf("reported opinions = %d, known keys = %d", len(opinions), len(deploymanifest.KnownKeys()))
	}
	for dotted, opinion := range opinions {
		if opinion.Disposition == "" || opinion.Reason == "" {
			t.Errorf("%s disappeared into an empty disposition: %+v", dotted, opinion)
		}
	}
}

func TestCompileToolPluginConfigPinnedAndMonotone(t *testing.T) {
	profiles := toolplugin.BuiltinProfiles()
	var audit toolplugin.Profile
	for _, p := range profiles {
		if p.ID == "builtin.audit" {
			audit = p
		}
	}
	text := fmt.Sprintf("[tool_plugins]\nplugins = [{ id = %q, version = %q, digest = %q }]\n[tool_plugins.organization]\nrequire_witness = true\ndisclosure = \"org\"\n[tool_plugins.user]\nrequire_witness = false\nwait_mode = \"local\"\n", audit.ID, audit.Version, audit.Digest)
	m, err := deploymanifest.Parse([]byte(text))
	if err != nil {
		t.Fatal(err)
	}
	plugins, layers, err := compileToolPluginConfig(m)
	if err != nil {
		t.Fatal(err)
	}
	if len(plugins) != 1 || plugins[0].Profile().ID != "builtin.audit" {
		t.Fatalf("plugins=%+v", plugins)
	}
	resolved := toolplugin.ResolvePreferences(layers)
	if !resolved.RequireWitness || resolved.WaitMode != "local" || resolved.Disclosure != "org" {
		t.Fatalf("resolved=%+v", resolved)
	}
	if resolved.Sources["require_witness"] != "organization" {
		t.Fatalf("sources=%+v", resolved.Sources)
	}
}

func TestCompileToolPluginConfigFailsClosed(t *testing.T) {
	for _, text := range []string{
		"[tool_plugins]\nplugins = [{ id = \"builtin.audit\", version = \"1\", digest = \"\" }]\n",
		"[tool_plugins]\nplugins = [{ id = \"unknown\", version = \"1\", digest = \"sha256:x\" }]\n",
		"[tool_plugins]\nplugins = [{ id = \"builtin.audit\", version = \"1\", digest = \"sha256:x\" }, { id = \"builtin.audit\", version = \"1\", digest = \"sha256:x\" }]\n",
	} {
		m, err := deploymanifest.Parse([]byte(text))
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err = compileToolPluginConfig(m); err == nil {
			t.Fatalf("accepted %s", text)
		}
	}
}

func TestCompileToolPluginConfigZeroValuePreservesLegacyPath(t *testing.T) {
	plugins, layers, err := compileToolPluginConfig(deploymanifest.Manifest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(plugins) != 0 {
		t.Fatalf("plugins=%d", len(plugins))
	}
	if got := toolplugin.ResolvePreferences(layers); got.RequireWitness || got.WaitMode != "" || got.TransformMode != "" || got.Disclosure != "" {
		t.Fatalf("resolved=%+v", got)
	}
}

func TestApplyToolPluginConfigArmsGatewayConfig(t *testing.T) {
	profiles := toolplugin.BuiltinProfiles()
	p := profiles[0]
	m, err := deploymanifest.Parse([]byte(fmt.Sprintf("[tool_plugins]\nplugins = [{ id = %q, version = %q, digest = %q }]\n[tool_plugins.project]\ntransform_mode = \"preview\"\n", p.ID, p.Version, p.Digest)))
	if err != nil {
		t.Fatal(err)
	}
	var cfg gateway.Config
	if err := applyToolPluginConfig(&cfg, m); err != nil {
		t.Fatal(err)
	}
	if len(cfg.ToolPlugins) != 1 || toolplugin.ResolvePreferences(cfg.ToolPreferences).TransformMode != "preview" {
		t.Fatalf("cfg plugins=%d prefs=%+v", len(cfg.ToolPlugins), cfg.ToolPreferences)
	}
}
