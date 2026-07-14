package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// guard_allow_claude_settings.go implements `fak guard allow --from-claude-settings`
// — the import path from a Claude Code allowlist into the operator allow overlay
// (see guard_allow.go for the overlay contract). An operator adopting `fak guard --
// claude` has usually already curated `.claude/settings.json` /
// `settings.local.json` `permissions.allow`; that list has no effect on the gateway
// floor, so without this verb every already-blessed tool must be re-declared by hand,
// reactively, one DEFAULT_DENY at a time. fak already parses the settings.json schema
// for its `hooks` key (guardPreCompactClaudeSettings); this reads the sibling
// `permissions` key it never modeled before.
//
// The import is NAME-LEVEL only, because the overlay is deliberately widen-only
// (allow / allow_prefix; no denies, no arg-rules). So:
//   - a bare tool name (WebFetch, mcp__github__create_issue) -> allow
//   - a bare MCP server (mcp__github)                         -> allow_prefix mcp__github__
//   - an arg-scoped rule (Bash(gh issue *), Read(//p/**))     -> reported unmappable
//     (the overlay cannot express an arg-conditional allow)
//   - permissions.deny                                        -> never imported
//
// It mirrors the `--from-journal` print-then-apply shape: it prints the plan by
// default and only writes the overlay under `--add-all`. Malformed settings fail loud
// (the same discipline loadGuardAllowOverlay uses), so an operator who believes they
// imported their allowlist is never silently left on the bare floor.

// guardAllowClaudeSettings is the SUBSET of the Claude Code settings.json schema this
// import reads: only permissions.allow / permissions.deny. It is decoded WITHOUT
// DisallowUnknownFields on purpose — a real settings.json carries many keys (hooks,
// model, env, ...) this verb has no business modeling, and ignoring them is correct;
// only genuinely malformed JSON should fail the read.
type guardAllowClaudeSettings struct {
	Permissions struct {
		Allow []string `json:"allow"`
		Deny  []string `json:"deny"`
	} `json:"permissions"`
}

// guardAllowClaudeMapping maps one permissions.allow entry onto the overlay schema.
// kind is "allow", "allow_prefix", or "" (unmappable); value is what would be recorded
// on the overlay; reason explains an unmappable entry in one line.
type guardAllowClaudeMapping struct {
	entry  string
	kind   string
	value  string
	reason string
}

// mapClaudeSettingsAllowEntry maps a single permissions.allow string onto the
// widen-only overlay. An arg-scoped rule (any `Tool(pattern)` form) is unmappable
// because the overlay has no way to express an arg-conditional allow; an MCP server
// entry (mcp__server, no third `__` segment) becomes an allow_prefix so every tool the
// server exposes is admitted; everything else — a bare tool name or a specific
// mcp__server__tool — is an exact allow.
func mapClaudeSettingsAllowEntry(entry string) guardAllowClaudeMapping {
	e := strings.TrimSpace(entry)
	m := guardAllowClaudeMapping{entry: e}
	switch {
	case e == "":
		m.reason = "empty entry"
	case strings.Contains(e, "("):
		m.reason = "arg-scoped rule; the widen-only overlay cannot express arg-conditional allows"
	default:
		if rest, ok := strings.CutPrefix(e, "mcp__"); ok {
			switch {
			case rest == "":
				m.reason = "malformed mcp entry (no server name)"
			case strings.Contains(rest, "__"):
				// mcp__server__tool — a specific MCP tool: an exact allow.
				m.kind, m.value = "allow", e
			default:
				// mcp__server — the whole server: an allow_prefix so every
				// mcp__server__* tool the server exposes is admitted.
				m.kind, m.value = "allow_prefix", e+"__"
			}
		} else {
			// A bare, non-MCP tool name (WebFetch, Task, ...).
			m.kind, m.value = "allow", e
		}
	}
	return m
}

// guardAllowClaudeSettingsPaths resolves the settings files to read: an explicit
// positional path (or paths) wins, else the default project pair
// .claude/settings.json + .claude/settings.local.json, merged in that order.
func guardAllowClaudeSettingsPaths(positional []string) []string {
	var paths []string
	for _, p := range positional {
		if s := strings.TrimSpace(p); s != "" {
			paths = append(paths, s)
		}
	}
	if len(paths) > 0 {
		return paths
	}
	dir := filepath.Join(findRepoRoot("."), ".claude")
	return []string{
		filepath.Join(dir, "settings.json"),
		filepath.Join(dir, "settings.local.json"),
	}
}

// loadGuardAllowClaudeSettings reads permissions.allow/deny from each path in order and
// merges them. Malformed JSON fails loud. A missing file is skipped when it came from
// DEFAULT discovery (requireExist=false) — a project often has settings.json but no
// settings.local.json — but a missing EXPLICITLY-named path fails loud, so an operator
// who points at a file that is not there hears about the typo instead of a silent no-op.
// Returns the merged allow + deny lists and the paths that actually contributed.
func loadGuardAllowClaudeSettings(paths []string, requireExist bool) (allow, deny, read []string, err error) {
	for _, p := range paths {
		b, rerr := os.ReadFile(p)
		if rerr != nil {
			if errors.Is(rerr, os.ErrNotExist) && !requireExist {
				continue
			}
			return nil, nil, nil, fmt.Errorf("claude settings %s: %w", p, rerr)
		}
		var s guardAllowClaudeSettings
		if jerr := json.Unmarshal(b, &s); jerr != nil {
			return nil, nil, nil, fmt.Errorf("claude settings %s: invalid: %w", p, jerr)
		}
		allow = append(allow, s.Permissions.Allow...)
		deny = append(deny, s.Permissions.Deny...)
		read = append(read, p)
	}
	return allow, deny, read, nil
}

// runGuardAllowFromClaudeSettings imports a Claude settings.json allowlist into the
// overlay. It prints the mapping plan (what would import as allow / allow_prefix, what
// is unmappable, how many deny entries were ignored) and, only with addAll, writes the
// mappable entries to the overlay. positional is the optional path override from
// fs.Args(); an empty slice uses the default project settings pair.
func runGuardAllowFromClaudeSettings(stdout, stderr io.Writer, overlayPath string, ov *guardAllowOverlay, positional []string, addAll bool) int {
	paths := guardAllowClaudeSettingsPaths(positional)
	requireExist := len(positional) > 0
	allow, deny, read, err := loadGuardAllowClaudeSettings(paths, requireExist)
	if err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	if len(read) == 0 {
		fmt.Fprintf(stdout, "fak guard allow: no Claude settings found (looked in %s) — nothing to import.\n", strings.Join(paths, ", "))
		return 0
	}

	// Map every permissions.allow entry, deduping by entry so a name that appears in
	// both settings.json and settings.local.json is planned once.
	var allowVals, prefixVals []string
	var unmappable []guardAllowClaudeMapping
	seen := map[string]bool{}
	for _, e := range allow {
		m := mapClaudeSettingsAllowEntry(e)
		if m.entry == "" || seen[m.entry] {
			continue
		}
		seen[m.entry] = true
		switch m.kind {
		case "allow":
			allowVals = append(allowVals, m.value)
		case "allow_prefix":
			prefixVals = append(prefixVals, m.value)
		default:
			unmappable = append(unmappable, m)
		}
	}
	allowVals = guardAllowNormalize(allowVals)
	prefixVals = guardAllowNormalize(prefixVals)

	fmt.Fprintf(stdout, "Claude settings permissions.allow -> operator allow overlay (from %s):\n", strings.Join(read, ", "))
	if len(allowVals) > 0 {
		fmt.Fprintf(stdout, "  allow (exact) : %s\n", strings.Join(allowVals, ", "))
	}
	if len(prefixVals) > 0 {
		fmt.Fprintf(stdout, "  allow (prefix): %s\n", strings.Join(prefixVals, ", "))
	}
	if len(unmappable) > 0 {
		fmt.Fprintf(stdout, "  unmappable (%d) — left for the operator to model by hand:\n", len(unmappable))
		for _, m := range unmappable {
			fmt.Fprintf(stdout, "    %-28s %s\n", m.entry, m.reason)
		}
	}
	if len(deny) > 0 {
		fmt.Fprintf(stdout, "  permissions.deny ignored (widen-only contract): %d entr(ies)\n", len(deny))
	}
	if len(allowVals) == 0 && len(prefixVals) == 0 {
		fmt.Fprintln(stdout, "  (nothing mappable to import)")
		return 0
	}

	if addAll {
		beforeA, beforeP := len(ov.Allow), len(ov.AllowPrefix)
		ov.Allow = append(ov.Allow, allowVals...)
		ov.AllowPrefix = append(ov.AllowPrefix, prefixVals...)
		if err := saveGuardAllowOverlay(overlayPath, *ov); err != nil {
			fmt.Fprintln(stderr, "fak guard allow:", err)
			return 1
		}
		// Reload so *ov reflects the normalized (deduped/sorted) union and the added
		// counts below are computed against the same on-disk state the operator sees.
		reloaded, err := loadGuardAllowOverlay(overlayPath)
		if err != nil {
			fmt.Fprintln(stderr, "fak guard allow:", err)
			return 1
		}
		*ov = reloaded
		fmt.Fprintf(stdout, "\nAdded %d allow + %d allow_prefix entr(ies) to the operator allow overlay: %s\n",
			len(ov.Allow)-beforeA, len(ov.AllowPrefix)-beforeP, overlayPath)
		fmt.Fprintln(stdout, "  Takes effect on the next `fak guard` launch.")
		printGuardAllowOverlay(stdout, overlayPath, *ov)
		return 0
	}

	fmt.Fprintln(stdout, "\nTo import the mappable entries into the overlay (operator, out-of-band from the agent):")
	applyCmd := "fak guard allow --from-claude-settings --add-all"
	if len(positional) > 0 {
		applyCmd += " " + strings.Join(positional, " ")
	}
	fmt.Fprintf(stdout, "  %s\n", applyCmd)
	return 0
}
