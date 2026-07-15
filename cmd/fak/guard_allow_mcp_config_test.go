package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func TestRunGuardAllowFromMCPConfigListsAndAddsPrefixes(t *testing.T) {
	dir := t.TempDir()
	config := filepath.Join(dir, ".mcp.json")
	if err := os.WriteFile(config, []byte(`{"mcpServers":{"b":{"type":"http","url":"http://b"},"a":{"type":"stdio"}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	overlayPath := filepath.Join(dir, "allow.json")
	ov := guardAllowOverlay{}
	var out, errb bytes.Buffer
	if code := runGuardAllowFromMCPConfig(&out, &errb, overlayPath, &ov, []string{config}, false); code != 0 {
		t.Fatalf("list exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"mcp__a__", "mcp__b__", "fak guard allow --prefix mcp__a__", "COARSE GRANT", "ALL tools"} {
		if !strings.Contains(got, want) {
			t.Errorf("list missing %q: %s", want, got)
		}
	}
	if _, err := os.Stat(overlayPath); !os.IsNotExist(err) {
		t.Fatal("list mode wrote overlay")
	}

	out.Reset()
	errb.Reset()
	if code := runGuardAllowFromMCPConfig(&out, &errb, overlayPath, &ov, []string{config}, true); code != 0 {
		t.Fatalf("add exit=%d stderr=%s", code, errb.String())
	}
	loaded, err := loadGuardAllowOverlay(overlayPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(loaded.AllowPrefix, ",") != "mcp__a__,mcp__b__" {
		t.Fatalf("prefixes = %v", loaded.AllowPrefix)
	}
}

func TestRunGuardAllowFromMCPConfigMalformedFails(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".mcp.json")
	if err := os.WriteFile(path, []byte(`{"mcpServers":`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runGuardAllowFromMCPConfig(&out, &errb, filepath.Join(t.TempDir(), "allow.json"), &guardAllowOverlay{}, []string{path}, false); code == 0 {
		t.Fatalf("malformed exit=0 output=%s", out.String())
	}
	if !strings.Contains(errb.String(), "parse MCP config") {
		t.Fatalf("stderr=%s", errb.String())
	}
}

func TestGuardAllowMCPImportDoesNotAlterDenyOrArgRules(t *testing.T) {
	dir := t.TempDir()
	config := filepath.Join(dir, ".mcp.json")
	if err := os.WriteFile(config, []byte(`{"mcpServers":{"a":{}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	ov := guardAllowOverlay{}
	if code := runGuardAllowFromMCPConfig(&bytes.Buffer{}, &bytes.Buffer{}, filepath.Join(dir, "allow.json"), &ov, []string{config}, true); code != 0 {
		t.Fatalf("import exit=%d", code)
	}
	rt := policy.Runtime{Adjudicator: adjudicator.Policy{Deny: map[string]abi.ReasonCode{"danger": abi.ReasonPolicyBlock}, ArgPredicates: []adjudicator.ArgPredicate{{Tool: "Bash", Arg: "command"}}}}
	beforeDeny, beforeRules := len(rt.Adjudicator.Deny), len(rt.Adjudicator.ArgPredicates)
	guardApplyAllowOverlay(&rt, ov)
	if len(rt.Adjudicator.Deny) != beforeDeny || len(rt.Adjudicator.ArgPredicates) != beforeRules {
		t.Fatalf("import changed deny/rules: deny %d->%d rules %d->%d", beforeDeny, len(rt.Adjudicator.Deny), beforeRules, len(rt.Adjudicator.ArgPredicates))
	}
}
