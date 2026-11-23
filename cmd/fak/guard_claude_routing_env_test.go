package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

const (
	routingEnvTestParentURL = "http://127.0.0.1:18102"
	routingEnvTestGuardURL  = "http://127.0.0.1:49152"
)

func readRoutingSettingsEnv(t *testing.T, path string) map[string]string {
	t.Helper()
	s := readGuardHookSettings(t, path)
	env := map[string]string{}
	if err := json.Unmarshal(s.Extra[guardClaudeSettingsEnvKey], &env); err != nil {
		t.Fatalf("parse env block of %s: %v", path, err)
	}
	return env
}

// The guarded Claude child must resolve guard's gateway, not the endpoint its launching
// session was pinned to: the settings file guard hands it carries guard's own base URL,
// and the process env carries exactly one value for it.
func TestGuardClaudeRoutingEnvDoesNotCarryParentBaseURL(t *testing.T) {
	t.Setenv("ANTHROPIC_BASE_URL", routingEnvTestParentURL)
	injected := guardInjectedEnv("anthropic", "", routingEnvTestGuardURL)

	command, path, err := installGuardClaudeRoutingEnvAt([]string{"claude", "-p", "hi"}, injected, "", t.TempDir())
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if path == "" || !slices.Contains(command, path) || !slices.Contains(command, guardClaudeSettingsFlag) {
		t.Fatalf("child argv %v does not carry the routing settings file %q", command, path)
	}
	env := readRoutingSettingsEnv(t, path)
	if got := env["ANTHROPIC_BASE_URL"]; got != routingEnvTestGuardURL {
		t.Fatalf("settings ANTHROPIC_BASE_URL = %q, want guard gateway %q", got, routingEnvTestGuardURL)
	}
	if len(env) != 1 {
		t.Fatalf("settings env = %v, want only the endpoint", env)
	}

	_, childEnv := guardChildPlanCommandEnv(newGuardLaunchPlan(command), injected, false)
	vars, err := toolprocgate.EnvFromStrings(childEnv)
	if err != nil {
		t.Fatalf("normalize child env: %v", err)
	}
	grantEnv := map[string]string{}
	for _, v := range vars {
		grantEnv[v.Name] = v.Value
	}
	if got := grantEnv["ANTHROPIC_BASE_URL"]; got != routingEnvTestGuardURL {
		t.Fatalf("child process ANTHROPIC_BASE_URL = %q, want guard gateway %q (parent was %q)", got, routingEnvTestGuardURL, routingEnvTestParentURL)
	}
	if grantEnv[guardActiveEnv] != "1" {
		t.Fatalf("child env lost %s", guardActiveEnv)
	}
}

// Merging into the hook installers' file keeps their hooks and the caller's other env
// entries, and overrides only a caller-supplied endpoint.
func TestGuardClaudeRoutingEnvMergesIntoExistingSettings(t *testing.T) {
	path := filepath.Join(t.TempDir(), "claude-settings.json")
	existing := `{"hooks":{"Stop":[{"hooks":[{"type":"command","command":"fak","args":["guard-stop"]}]}]},` +
		`"ultracode":true,"env":{"ANTHROPIC_BASE_URL":"` + routingEnvTestParentURL + `","ANTHROPIC_MODEL":"keep-me"}}`
	if err := os.WriteFile(path, []byte(existing), 0o600); err != nil {
		t.Fatal(err)
	}
	in := []string{"claude", guardClaudeSettingsFlag, path, "-p", "hi"}
	injected := [][2]string{{"ANTHROPIC_BASE_URL", routingEnvTestGuardURL}, {guardEnableToolSearchEnv, guardEnableToolSearchValue}}

	command, got, err := installGuardClaudeRoutingEnvAt(in, injected, path, "")
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if got != path || !slices.Equal(command, in) {
		t.Fatalf("merge changed argv or path: %v %q", command, got)
	}
	env := readRoutingSettingsEnv(t, path)
	if env["ANTHROPIC_BASE_URL"] != routingEnvTestGuardURL || env["ANTHROPIC_MODEL"] != "keep-me" || len(env) != 2 {
		t.Fatalf("merged env = %v", env)
	}
	s := readGuardHookSettings(t, path)
	if len(s.Hooks["Stop"]) != 1 || string(s.Extra["ultracode"]) != "true" {
		t.Fatalf("merge dropped existing settings: hooks=%v extra=%v", s.Hooks, s.Extra)
	}
}

func TestGuardClaudeRoutingEnvSkipsWhenNothingToPin(t *testing.T) {
	dir := t.TempDir()
	for name, tc := range map[string]struct {
		command  []string
		injected [][2]string
	}{
		"non-claude child":      {[]string{"opencode", "run"}, [][2]string{{"ANTHROPIC_BASE_URL", routingEnvTestGuardURL}}},
		"explicit env override": {[]string{"claude"}, guardInjectedEnv("anthropic", "MY_BASE_URL", routingEnvTestGuardURL)},
	} {
		command, path, err := installGuardClaudeRoutingEnvAt(tc.command, tc.injected, "", dir)
		if err != nil || path != "" || !slices.Equal(command, tc.command) {
			t.Fatalf("%s: got command=%v path=%q err=%v, want untouched", name, command, path, err)
		}
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Fatalf("skip wrote files: %v", entries)
	}
}
