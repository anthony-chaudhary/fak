package main

import (
	"errors"
	"flag"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
)

// serveConfigPath finds the bootstrap flag that must be known before the full
// serve FlagSet is parsed. Configuration is deliberately explicit: an ambient
// fak.toml never changes a command merely because its working directory moved.
func serveConfigPath(argv []string) (string, error) {
	var path string
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "--config" || arg == "-config":
			if i+1 >= len(argv) || strings.HasPrefix(argv[i+1], "-") {
				return "", errors.New("--config requires a file path")
			}
			i++
			if path != "" {
				return "", errors.New("--config may be specified only once")
			}
			path = argv[i]
		case strings.HasPrefix(arg, "--config=") || strings.HasPrefix(arg, "-config="):
			value := strings.TrimSpace(strings.SplitN(arg, "=", 2)[1])
			if value == "" {
				return "", errors.New("--config requires a file path")
			}
			if path != "" {
				return "", errors.New("--config may be specified only once")
			}
			path = value
		}
	}
	return path, nil
}

// applyServeManifestDefaults wires the deployment fields that have an exact
// serve counterpart. Pointer assignment changes the parse defaults; the normal
// flag parser then naturally gives every explicitly supplied flag final say.
func applyServeManifestDefaults(sf *serveFlags, m deploymanifest.Manifest) {
	if m.Present("policy", "floor") {
		*sf.policyPath = m.Policy.Floor
	}
	if m.Present("auth", "require_key_env") {
		*sf.requireKeyEnv = m.Auth.RequireKeyEnv
	}
	if m.Present("budgets", "default_tokens") {
		*sf.contextBudgetTokens = m.Budgets.DefaultTokens
	}
	if m.Present("observability", "bind") {
		*sf.addr = m.Observability.Bind
	}
}

func explicitFlagNames(fs *flag.FlagSet) map[string]bool {
	names := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) { names[f.Name] = true })
	return names
}

type effectiveConfigValue struct {
	Value  any    `json:"value"`
	Source string `json:"source"`
}

type effectiveServeConfigReport struct {
	Schema string                          `json:"schema"`
	Config string                          `json:"config,omitempty"`
	Values map[string]effectiveConfigValue `json:"values"`
}

func effectiveServeConfig(sf *serveFlags, m deploymanifest.Manifest, hasManifest bool, explicit map[string]bool) effectiveServeConfigReport {
	source := func(flagName, section, key string) string {
		if explicit[flagName] {
			return "flag"
		}
		if hasManifest && m.Present(section, key) {
			return "manifest"
		}
		return "built-in"
	}
	return effectiveServeConfigReport{
		Schema: "fak-serve-effective-config/1",
		Config: *sf.configPath,
		Values: map[string]effectiveConfigValue{
			"addr":                  {Value: *sf.addr, Source: source("addr", "observability", "bind")},
			"policy":                {Value: *sf.policyPath, Source: source("policy", "policy", "floor")},
			"require_key_env":       {Value: *sf.requireKeyEnv, Source: source("require-key-env", "auth", "require_key_env")},
			"context_budget_tokens": {Value: *sf.contextBudgetTokens, Source: source("context-budget-tokens", "budgets", "default_tokens")},
		},
	}
}
