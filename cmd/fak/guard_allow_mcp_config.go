package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func guardAllowMCPConfigPath(positional []string) (string, error) {
	if len(positional) > 1 {
		return "", fmt.Errorf("--from-mcp-config accepts at most one path")
	}
	if len(positional) == 1 {
		return positional[0], nil
	}
	return filepath.Join(findRepoRoot("."), ".mcp.json"), nil
}

func loadGuardAllowMCPPrefixes(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read MCP config %s: %w", path, err)
	}
	var cfg guardMCPClientConfig
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("parse MCP config %s: %w", path, err)
	}
	prefixes := make([]string, 0, len(cfg.MCPServers))
	for raw := range cfg.MCPServers {
		name := strings.TrimSpace(raw)
		if name == "" {
			continue
		}
		prefixes = append(prefixes, "mcp__"+name+"__")
	}
	sort.Strings(prefixes)
	return guardAllowNormalize(prefixes), nil
}

func runGuardAllowFromMCPConfig(stdout, stderr io.Writer, overlayPath string, ov *guardAllowOverlay, positional []string, addAll bool) int {
	path, err := guardAllowMCPConfigPath(positional)
	if err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 2
	}
	prefixes, err := loadGuardAllowMCPPrefixes(path)
	if err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	if len(prefixes) == 0 {
		fmt.Fprintf(stdout, "fak guard allow: no mcpServers found in %s � nothing to import.\n", path)
		return 0
	}
	fmt.Fprintf(stdout, "MCP server inventory -> operator allow prefixes (from %s):\n", path)
	fmt.Fprintln(stdout, "  COARSE GRANT: each server prefix admits ALL tools exposed by that server.")
	for _, prefix := range prefixes {
		fmt.Fprintf(stdout, "  %-28s fak guard allow --prefix %s\n", prefix, prefix)
	}
	if !addAll {
		fmt.Fprintln(stdout, "  (or add every prefix: fak guard allow --from-mcp-config --add-all)")
		return 0
	}
	ov.AllowPrefix = append(ov.AllowPrefix, prefixes...)
	if err := saveGuardAllowOverlay(overlayPath, *ov); err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	reloaded, err := loadGuardAllowOverlay(overlayPath)
	if err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	*ov = reloaded
	fmt.Fprintf(stdout, "\nAdded %d server prefix(es) to %s.\n", len(prefixes), overlayPath)
	return 0
}
