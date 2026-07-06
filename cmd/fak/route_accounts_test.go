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
	for _, want := range []string{"roster valid", "accounts:", "credential readiness", "bindings", "residency"} {
		if !strings.Contains(out, want) {
			t.Fatalf("check surface missing %q:\n%s", want, out)
		}
	}
}

func TestRouteAccountsStatusReportsEnvReadiness(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-must-not-print")
	t.Setenv("OPENAI_WORK_API_KEY", "")
	path := writeDumpedRoster(t)

	code, out, errs := runRT("--accounts-status", path)
	if code != 0 {
		t.Fatalf("--accounts-status exit=%d stderr=%s out=%s", code, errs, out)
	}
	for _, want := range []string{"fak route accounts status", "needs_credential", "not_required", "OPENAI_WORK_API_KEY"} {
		if !strings.Contains(out, want) {
			t.Fatalf("accounts status missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "sk-must-not-print") {
		t.Fatalf("accounts status leaked a secret:\n%s", out)
	}

	code, out, errs = runRT("--accounts-status", path, "--json")
	if code != 0 {
		t.Fatalf("--accounts-status --json exit=%d stderr=%s out=%s", code, errs, out)
	}
	var rep struct {
		Schema  string `json:"schema"`
		Summary struct {
			Total           int `json:"total"`
			NeedsCredential int `json:"needs_credential"`
		} `json:"summary"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if rep.Schema != "fak.modelroute.accounts.v1" || rep.Summary.Total == 0 || rep.Summary.NeedsCredential == 0 {
		t.Fatalf("unexpected readiness json: %+v\n%s", rep, out)
	}
	if strings.Contains(out, "sk-must-not-print") {
		t.Fatalf("accounts status json leaked a secret:\n%s", out)
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

func TestRouteAccountsBindsDeepSeekProfile(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "sk-must-not-print")
	roster := writeAPIHostDeepSeekModelAccountsRoster(t, "https://api.deepseek.com", "https://api.deepseek.com/anthropic")
	manifest := filepath.Join(t.TempDir(), "deepseek-route.json")
	body := `{
  "version": "fak-route/v1",
  "default": {
    "members": [{"model": "deepseek-pro", "role": "primary"}],
    "reason": "DeepSeek V4 Pro profile witness"
  }
}`
	if err := os.WriteFile(manifest, []byte(body), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	code, out, errs := runRT("--manifest", manifest, "--accounts", roster, "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s out=%s", code, errs, out)
	}
	var rep struct {
		Binding struct {
			Members []struct {
				Model           string `json:"model"`
				Account         string `json:"account"`
				Kind            string `json:"kind"`
				BaseURL         string `json:"base_url"`
				CredEnv         string `json:"cred_env"`
				UpstreamModel   string `json:"upstream_model"`
				Local           bool   `json:"local"`
				EngineRoute     string `json:"engine_route"`
				ContextTokens   int    `json:"context_tokens"`
				MaxOutputTokens int    `json:"max_output_tokens"`
			} `json:"members"`
		} `json:"binding"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if len(rep.Binding.Members) != 1 {
		t.Fatalf("want one bound DeepSeek member, got %d:\n%s", len(rep.Binding.Members), out)
	}
	m := rep.Binding.Members[0]
	if m.Account != "deepseek" || m.Kind != "deepseek" || m.BaseURL != "https://api.deepseek.com" ||
		m.CredEnv != "DEEPSEEK_API_KEY" || m.UpstreamModel != "deepseek-v4-pro" ||
		m.Local || m.EngineRoute != "deepseek:deepseek/deepseek-v4-pro" ||
		m.ContextTokens != 1000000 || m.MaxOutputTokens != 384000 {
		t.Fatalf("DeepSeek binding wrong: %+v\n%s", m, out)
	}
	if strings.Contains(out, "sk-must-not-print") {
		t.Fatalf("route output leaked a secret:\n%s", out)
	}
}

func TestRouteAccountsBindsGroqQwen36Profile(t *testing.T) {
	t.Setenv("FAK_GROQ_API_KEY", "fake-groq-secret-value")
	roster := writeDumpedRoster(t)
	manifest := filepath.Join(t.TempDir(), "groq-route.json")
	body := `{
  "version": "fak-route/v1",
  "default": {
    "members": [{"model": "qwen36-groq", "role": "primary"}],
    "reason": "Groq Qwen3.6 27B profile witness"
  }
}`
	if err := os.WriteFile(manifest, []byte(body), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	code, out, errs := runRT("--manifest", manifest, "--accounts", roster, "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s out=%s", code, errs, out)
	}
	var rep struct {
		Binding struct {
			Members []struct {
				Model             string `json:"model"`
				Account           string `json:"account"`
				Kind              string `json:"kind"`
				BaseURL           string `json:"base_url"`
				CredEnv           string `json:"cred_env"`
				UpstreamModel     string `json:"upstream_model"`
				Local             bool   `json:"local"`
				EngineRoute       string `json:"engine_route"`
				RequestsPerMinute int    `json:"requests_per_minute"`
				RequestsPerDay    int    `json:"requests_per_day"`
				TokensPerMinute   int    `json:"tokens_per_minute"`
				TokensPerDay      int    `json:"tokens_per_day"`
			} `json:"members"`
		} `json:"binding"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if len(rep.Binding.Members) != 1 {
		t.Fatalf("want one bound Groq member, got %d:\n%s", len(rep.Binding.Members), out)
	}
	m := rep.Binding.Members[0]
	if m.Account != "july6netra_groq" || m.Kind != "openai" || m.BaseURL != "https://api.groq.com/openai/v1" ||
		m.CredEnv != "FAK_GROQ_API_KEY" || m.UpstreamModel != "qwen/qwen3.6-27b" ||
		m.Local || m.EngineRoute != "openai:july6netra_groq/qwen/qwen3.6-27b" ||
		m.RequestsPerMinute != 30 || m.RequestsPerDay != 1000 ||
		m.TokensPerMinute != 8000 || m.TokensPerDay != 200000 {
		t.Fatalf("Groq Qwen3.6 binding wrong: %+v\n%s", m, out)
	}
	if strings.Contains(out, "fake-groq-secret-value") {
		t.Fatalf("route output leaked a Groq key value:\n%s", out)
	}
}

func TestRouteAccountsBindsGroqCompoundProfile(t *testing.T) {
	t.Setenv("FAK_GROQ_API_KEY", "fake-groq-secret-value")
	roster := writeDumpedRoster(t)
	manifest := filepath.Join(t.TempDir(), "groq-compound-route.json")
	body := `{
  "version": "fak-route/v1",
  "default": {
    "members": [{"model": "groq-compound", "role": "primary"}],
    "reason": "Groq Compound lower-quality tier witness"
  }
}`
	if err := os.WriteFile(manifest, []byte(body), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	code, out, errs := runRT("--manifest", manifest, "--accounts", roster, "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s out=%s", code, errs, out)
	}
	var rep struct {
		Binding struct {
			Members []struct {
				Model             string `json:"model"`
				Account           string `json:"account"`
				Kind              string `json:"kind"`
				BaseURL           string `json:"base_url"`
				CredEnv           string `json:"cred_env"`
				UpstreamModel     string `json:"upstream_model"`
				Local             bool   `json:"local"`
				EngineRoute       string `json:"engine_route"`
				RequestsPerMinute int    `json:"requests_per_minute"`
				RequestsPerDay    int    `json:"requests_per_day"`
				TokensPerMinute   int    `json:"tokens_per_minute"`
				TokensPerDay      int    `json:"tokens_per_day"`
			} `json:"members"`
		} `json:"binding"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if len(rep.Binding.Members) != 1 {
		t.Fatalf("want one bound Groq Compound member, got %d:\n%s", len(rep.Binding.Members), out)
	}
	m := rep.Binding.Members[0]
	if m.Account != "july6netra_groq_compound" || m.Kind != "openai" || m.BaseURL != "https://api.groq.com/openai/v1" ||
		m.CredEnv != "FAK_GROQ_API_KEY" || m.UpstreamModel != "groq/compound" ||
		m.Local || m.EngineRoute != "openai:july6netra_groq_compound/groq/compound" ||
		m.RequestsPerMinute != 30 || m.RequestsPerDay != 250 ||
		m.TokensPerMinute != 0 || m.TokensPerDay != 0 {
		t.Fatalf("Groq Compound binding wrong: %+v\n%s", m, out)
	}
	if strings.Contains(out, "fake-groq-secret-value") {
		t.Fatalf("route output leaked a Groq key value:\n%s", out)
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
