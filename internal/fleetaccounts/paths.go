package fleetaccounts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
)

// Paths resolves the discovery roots + policy/registry file locations the way
// fleet_accounts.py does (honoring the same env overrides), so the Go surface reads the
// SAME single sources of truth — never a second contract.
type Paths struct {
	Home         string // user home (Claude config root)
	ConfigHome   string // XDG config home (opencode config root)
	PolicyPath   string // accounts_policy.json (operator config)
	ExamplePath  string // accounts_policy.example.json (committed template)
	RegistryDir  string // runtime registry dir
	RegistryPath string // sessions.json
}

// ResolvePaths mirrors the module-level path resolution in fleet_accounts.py.
// repoToolsDir is the tools/ dir of the repo (where the committed example + default
// _registry/ live); pass "" to skip the repo-relative defaults.
func ResolvePaths(repoToolsDir string) Paths {
	home := getenvOr("FLEET_USER_HOME", userHome())
	configHome := os.Getenv("FLEET_CONFIG_HOME")
	if configHome == "" {
		configHome = os.Getenv("XDG_CONFIG_HOME")
	}
	if configHome == "" {
		configHome = filepath.Join(home, ".config")
	}
	regDir := os.Getenv("FLEET_REG_DIR")
	if regDir == "" && repoToolsDir != "" {
		regDir = filepath.Join(repoToolsDir, "_registry")
	}
	policyDir := os.Getenv("FLEET_POLICY_DIR")
	if policyDir == "" && repoToolsDir != "" {
		policyDir = filepath.Join(repoToolsDir, "_registry")
	}
	policyPath := os.Getenv("FLEET_POLICY_PATH")
	if policyPath == "" && policyDir != "" {
		policyPath = filepath.Join(policyDir, "accounts_policy.json")
	}
	example := ""
	if repoToolsDir != "" {
		example = filepath.Join(repoToolsDir, "accounts_policy.example.json")
	}
	registryPath := ""
	if regDir != "" {
		registryPath = filepath.Join(regDir, "sessions.json")
	}
	return Paths{
		Home: home, ConfigHome: configHome, PolicyPath: policyPath, ExamplePath: example,
		RegistryDir: regDir, RegistryPath: registryPath,
	}
}

func getenvOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func userHome() string {
	if h, err := os.UserHomeDir(); err == nil {
		return h
	}
	if runtime.GOOS == "windows" {
		return os.Getenv("USERPROFILE")
	}
	return os.Getenv("HOME")
}

// LoadPolicy loads the operator policy from disk, falling back to the committed example,
// then to DefaultPolicy. A malformed or absent file yields defaults — policy must never
// crash discovery. Mirrors fleet_accounts.load_policy resolution order: requested path,
// then (only when it is the canonical PolicyPath) the example, then built-in defaults;
// built-in defaults always backstop missing keys.
func LoadPolicy(p Paths) Policy {
	pol := DefaultPolicy()
	src := p.PolicyPath
	if !fileExists(src) && p.ExamplePath != "" && fileExists(p.ExamplePath) {
		src = p.ExamplePath
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return pol
	}
	var user map[string]json.RawMessage
	if json.Unmarshal(data, &user) != nil {
		return pol
	}
	if raw, ok := user["exclude"]; ok {
		var v []string
		if json.Unmarshal(raw, &v) == nil {
			pol.Exclude = v
		}
	}
	if raw, ok := user["include_only"]; ok {
		var v []string
		if json.Unmarshal(raw, &v) == nil {
			pol.IncludeOnly = v
		}
	}
	mergeMapOverride(user, "notes", pol.Notes)
	mergeMapOverride(user, "account_profiles", pol.AccountProfiles)
	mergeMapOverride(user, "route_weights", pol.RouteWeights)
	mergeMapOverride(user, "lane_models", pol.LaneModels)
	if raw, ok := user["routing"]; ok {
		var v Routing
		if json.Unmarshal(raw, &v) == nil {
			if v.LightConfidence != 0 {
				pol.Routing.LightConfidence = v.LightConfidence
			}
			if v.HardTier1Fallback != "" {
				pol.Routing.HardTier1Fallback = v.HardTier1Fallback
			}
		}
	}
	return pol
}

// mergeMapOverride folds one user-policy map field into the built-in defaults, entry by
// entry so an override of one key never drops the rest. The policy file is advisory:
// an absent key, or one whose JSON does not parse as this map's shape, leaves the
// defaults exactly as they were rather than failing the load.
func mergeMapOverride[V any](user map[string]json.RawMessage, key string, into map[string]V) {
	raw, ok := user[key]
	if !ok {
		return
	}
	var v map[string]V
	if json.Unmarshal(raw, &v) != nil {
		return
	}
	for k, val := range v {
		into[k] = val
	}
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}
