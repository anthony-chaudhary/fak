package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestMapClaudeSettingsAllowEntry locks the name-level mapping rules: a bare tool name
// and a specific mcp__server__tool are exact allows, an mcp__server entry widens to an
// allow_prefix, and any arg-scoped Tool(pattern) rule is unmappable with a reason.
func TestMapClaudeSettingsAllowEntry(t *testing.T) {
	cases := []struct {
		entry, kind, value string
	}{
		{"WebFetch", "allow", "WebFetch"},
		{"mcp__github__create_issue", "allow", "mcp__github__create_issue"},
		{"mcp__github", "allow_prefix", "mcp__github__"},
		{"  Edit  ", "allow", "Edit"}, // trimmed
		{"Bash(gh issue *)", "", ""},  // arg-scoped -> unmappable
		{"Read(//tmp/**)", "", ""},    // arg-scoped -> unmappable
		{"mcp__", "", ""},             // malformed -> unmappable
	}
	for _, c := range cases {
		m := mapClaudeSettingsAllowEntry(c.entry)
		if m.kind != c.kind || m.value != c.value {
			t.Errorf("map(%q) = {kind:%q value:%q}, want {kind:%q value:%q} (reason=%q)",
				c.entry, m.kind, m.value, c.kind, c.value, m.reason)
		}
		if c.kind == "" && strings.TrimSpace(m.reason) == "" {
			t.Errorf("map(%q) is unmappable but carries no reason", c.entry)
		}
	}
}

// TestRunGuardAllowFromClaudeSettings is the acceptance-criteria test: a settings.json
// with one bare name, one MCP-server entry, one Bash(...) arg-rule, and a deny list —
// the bare name imports to allow, the server to allow_prefix mcp__<server>__, the Bash
// entry is reported unmappable, and the deny list is never imported.
func TestRunGuardAllowFromClaudeSettings(t *testing.T) {
	dir := t.TempDir()
	settingsPath := filepath.Join(dir, "settings.json")
	// Extra keys (model) exercise that unknown settings keys are ignored, not fatal.
	const settings = `{
	  "model": "claude-opus",
	  "permissions": {
	    "allow": ["WebFetch", "mcp__github", "Bash(gh issue *)"],
	    "deny": ["Read(./secrets/**)"]
	  }
	}`
	if err := os.WriteFile(settingsPath, []byte(settings), 0o644); err != nil {
		t.Fatal(err)
	}
	overlayPath := filepath.Join(dir, "allow.json")

	// Plan mode (no --add-all): prints the mapping and writes nothing.
	ov, err := loadGuardAllowOverlay(overlayPath)
	if err != nil {
		t.Fatalf("load overlay: %v", err)
	}
	var out, errb bytes.Buffer
	if code := runGuardAllowFromClaudeSettings(&out, &errb, overlayPath, &ov, []string{settingsPath}, false); code != 0 {
		t.Fatalf("plan exit=%d stderr=%s", code, errb.String())
	}
	plan := out.String()
	if !strings.Contains(plan, "WebFetch") {
		t.Errorf("plan missing bare-name allow WebFetch:\n%s", plan)
	}
	if !strings.Contains(plan, "mcp__github__") {
		t.Errorf("plan missing server allow_prefix mcp__github__:\n%s", plan)
	}
	if !strings.Contains(plan, "Bash(gh issue *)") || !strings.Contains(plan, "unmappable") {
		t.Errorf("plan did not report Bash(...) as unmappable:\n%s", plan)
	}
	if !strings.Contains(plan, "deny ignored") {
		t.Errorf("plan did not report the ignored deny entry:\n%s", plan)
	}
	if _, err := os.Stat(overlayPath); !os.IsNotExist(err) {
		t.Errorf("plan mode must not write the overlay, but %s exists", overlayPath)
	}

	// Apply mode (--add-all): records the name + prefix, never the deny or the arg-rule.
	ov, _ = loadGuardAllowOverlay(overlayPath)
	out.Reset()
	errb.Reset()
	if code := runGuardAllowFromClaudeSettings(&out, &errb, overlayPath, &ov, []string{settingsPath}, true); code != 0 {
		t.Fatalf("add-all exit=%d stderr=%s", code, errb.String())
	}
	saved, err := loadGuardAllowOverlay(overlayPath)
	if err != nil {
		t.Fatalf("reload overlay: %v", err)
	}
	if len(saved.Allow) != 1 || saved.Allow[0] != "WebFetch" {
		t.Errorf("overlay allow = %v, want [WebFetch]", saved.Allow)
	}
	if len(saved.AllowPrefix) != 1 || saved.AllowPrefix[0] != "mcp__github__" {
		t.Errorf("overlay allow_prefix = %v, want [mcp__github__]", saved.AllowPrefix)
	}
	for _, v := range append(append([]string{}, saved.Allow...), saved.AllowPrefix...) {
		if strings.Contains(v, "secrets") || strings.Contains(v, "Bash") || strings.Contains(v, "(") {
			t.Errorf("overlay imported a forbidden entry %q — deny/arg-rules must never import", v)
		}
	}
}

// TestRunGuardAllowFromClaudeSettingsMerges confirms the default-source behavior: two
// settings files (settings.json + settings.local.json) are merged into one plan.
func TestRunGuardAllowFromClaudeSettingsMerges(t *testing.T) {
	dir := t.TempDir()
	p1 := filepath.Join(dir, "settings.json")
	p2 := filepath.Join(dir, "settings.local.json")
	if err := os.WriteFile(p1, []byte(`{"permissions":{"allow":["WebFetch"]}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p2, []byte(`{"permissions":{"allow":["mcp__github"]}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	overlayPath := filepath.Join(dir, "allow.json")
	ov, _ := loadGuardAllowOverlay(overlayPath)
	var out, errb bytes.Buffer
	if code := runGuardAllowFromClaudeSettings(&out, &errb, overlayPath, &ov, []string{p1, p2}, true); code != 0 {
		t.Fatalf("merge exit=%d stderr=%s", code, errb.String())
	}
	saved, _ := loadGuardAllowOverlay(overlayPath)
	if len(saved.Allow) != 1 || saved.Allow[0] != "WebFetch" {
		t.Errorf("merged allow = %v, want [WebFetch]", saved.Allow)
	}
	if len(saved.AllowPrefix) != 1 || saved.AllowPrefix[0] != "mcp__github__" {
		t.Errorf("merged allow_prefix = %v, want [mcp__github__]", saved.AllowPrefix)
	}
}

// TestRunGuardAllowFromClaudeSettingsFailsLoud asserts malformed settings and a missing
// explicitly-named path both fail loud (nonzero) rather than silently importing nothing.
func TestRunGuardAllowFromClaudeSettingsFailsLoud(t *testing.T) {
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	ov, _ := loadGuardAllowOverlay(overlayPath)

	bad := filepath.Join(dir, "settings.json")
	if err := os.WriteFile(bad, []byte(`{"permissions": {`), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runGuardAllowFromClaudeSettings(&out, &errb, overlayPath, &ov, []string{bad}, true); code == 0 {
		t.Errorf("malformed settings must fail loud, got exit 0:\n%s", out.String())
	}

	out.Reset()
	errb.Reset()
	missing := filepath.Join(dir, "does-not-exist.json")
	if code := runGuardAllowFromClaudeSettings(&out, &errb, overlayPath, &ov, []string{missing}, true); code == 0 {
		t.Errorf("a missing explicitly-named settings path must fail loud, got exit 0:\n%s", out.String())
	}
}
