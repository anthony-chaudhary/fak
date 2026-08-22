package main

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// discoveredCodexHomes projects the canonical fleet census into the general account
// switcher's credential-safe shape. It does not persist or mutate the Claude registry.
func discoveredCodexHomes() []accounts.Home {
	cwd, _ := os.Getwd()
	paths := fleetaccounts.ResolvePaths(filepath.Join(findRepoRoot(cwd), "tools"))
	rows := fleetaccounts.AnnotateWithProbes(
		fleetaccounts.Discover(paths.Home, paths.ConfigHome, fleetaccounts.LoadPolicy(paths)),
		fleetaccounts.LoadRegistry(paths.RegistryPath),
		paths.RegistryDir,
	)
	out := make([]accounts.Home, 0, len(rows))
	for _, row := range rows {
		if row.Product != "codex" || row.Kind != fleetaccounts.KindWorker {
			continue
		}
		h := accounts.Home{
			Name: row.Tag,
			Dir:  row.Dir,
			Identity: accounts.Identity{
				Exists:      true,
				HasCreds:    row.CanServe != nil && *row.CanServe,
				AccountUUID: codexDerefString(row.AccountUUID),
			},
		}
		if row.Available != nil && !*row.Available {
			disabled := false
			h.Enabled = &disabled
			h.Note = codexDerefString(row.BlockReason)
		}
		out = append(out, h)
	}
	return out
}

func appendDiscoveredCodexHomes(reg accounts.Registry) accounts.Registry {
	seen := make(map[string]bool, len(reg.Homes))
	for _, h := range reg.Homes {
		seen[h.Name] = true
	}
	for _, h := range discoveredCodexHomes() {
		if !seen[h.Name] {
			reg.Homes = append(reg.Homes, h)
		}
	}
	return reg
}

func resolveNamedCodexHome(name string) (accounts.Home, bool) {
	name = strings.TrimSpace(name)
	for _, h := range discoveredCodexHomes() {
		if h.Name == name {
			return h, true
		}
	}
	return accounts.Home{}, false
}

func codexDerefString(v *string) string {
	if v == nil {
		return ""
	}
	return *v
}

func codexHomeForCommand(command string, home accounts.Home) string {
	if guardAgentBaseName(command) != "codex" {
		return ""
	}
	return home.Dir
}

func launchCodexEnv(base []string, home string) []string {
	out := make([]string, 0, len(base)+1)
	for _, kv := range base {
		name, _, _ := strings.Cut(kv, "=")
		if strings.EqualFold(name, "CODEX_HOME") {
			continue
		}
		out = append(out, kv)
	}
	return append(out, "CODEX_HOME="+home)
}
