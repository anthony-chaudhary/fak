package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeDumpedRoster dumps the built-in account roster to a temp file and returns its
// path - the same starter `fak route --accounts-dump` emits for a user to edit.
func writeDumpedRoster(t *testing.T) string {
	t.Helper()
	code, out, _ := runRT("--accounts-dump")
	if code != 0 || !strings.Contains(out, "fak-accounts/v1") {
		t.Fatalf("--accounts-dump failed: code=%d out=%s", code, out)
	}
	path := filepath.Join(t.TempDir(), "roster.json")
	if err := os.WriteFile(path, []byte(out), 0o600); err != nil {
		t.Fatalf("write roster: %v", err)
	}
	return path
}

// --accounts-dump emits a valid roster, and --accounts-check accepts it and prints the
// account + binding surface.
func TestRouteAccountsDumpAndCheck(t *testing.T) {
	path := writeDumpedRoster(t)
	code, out, _ := runRT("--accounts-check", path)
	if code != 0 {
		t.Fatalf("--accounts-check exit=%d out=%s", code, out)
	}
	for _, want := range []string{"roster valid", "accounts:", "bindings", "residency"} {
		if !strings.Contains(out, want) {
			t.Fatalf("check surface missing %q:\n%s", want, out)
		}
	}
}

// Routing a write-shaped tool call through the roster resolves BOTH guard-ensemble
// members to their accounts - and they land on DIFFERENT accounts (the mix-and-match).
func TestRouteAccountsBindsEnsembleAcrossAccounts(t *testing.T) {
	path := writeDumpedRoster(t)
	code, out, _ := runRT("--aspect", "tool_call", "--tool", "write_file", "--accounts", path, "--json")
	if code != 0 {
		t.Fatalf("exit=%d out=%s", code, out)
	}
	var rep struct {
		Binding struct {
			Members []struct {
				Model       string `json:"model"`
				Account     string `json:"account"`
				Kind        string `json:"kind"`
				Local       bool   `json:"local"`
				EngineRoute string `json:"engine_route"`
				CredEnv     string `json:"cred_env"`
			} `json:"members"`
		} `json:"binding"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if len(rep.Binding.Members) != 2 {
		t.Fatalf("want 2 bound members, got %d:\n%s", len(rep.Binding.Members), out)
	}
	if rep.Binding.Members[0].Account == rep.Binding.Members[1].Account {
		t.Fatalf("guard ensemble should span two accounts, both = %q", rep.Binding.Members[0].Account)
	}
	// Each remote member's engine route is residency-honest (carries its kind, not local:).
	for _, m := range rep.Binding.Members {
		if !m.Local && strings.HasPrefix(m.EngineRoute, "local:") {
			t.Fatalf("a remote member must not carry a local: route: %+v", m)
		}
	}
}

// The CLI output (human and JSON) carries credential ENV-VAR NAMES, never secrets,
// even when the env var holds a key.
func TestRouteAccountsNeverPrintsSecret(t *testing.T) {
	t.Setenv("OPENAI_WORK_API_KEY", "sk-must-not-print")
	path := writeDumpedRoster(t)
	for _, args := range [][]string{
		{"--accounts-check", path},
		{"--aspect", "tool_call", "--tool", "write_file", "--accounts", path},
		{"--aspect", "tool_call", "--tool", "write_file", "--accounts", path, "--json"},
	} {
		_, out, _ := runRT(args...)
		if strings.Contains(out, "sk-must-not-print") {
			t.Fatalf("CLI leaked a secret for args %v:\n%s", args, out)
		}
		if !strings.Contains(out, "OPENAI_WORK_API_KEY") {
			t.Fatalf("CLI should show the env-var NAME for args %v:\n%s", args, out)
		}
	}
}

// A roster that fails validation (here, a local account with a remote base_url) is a
// fail-loud error at --accounts-check, exit 1.
func TestRouteAccountsCheckFailsLoud(t *testing.T) {
	bad := `{"version":"fak-accounts/v1","accounts":[{"id":"l","kind":"local","base_url":"https://api.openai.com/v1"}]}`
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte(bad), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	code, _, errs := runRT("--accounts-check", path)
	if code != 1 {
		t.Fatalf("a residency-bypass roster must exit 1, got %d (stderr=%s)", code, errs)
	}
}
