package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func apiHostTestServer() *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/ok/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"m1"}]}`))
	})
	return httptest.NewServer(mux)
}

func TestAPIHostReadinessCommandWritesReports(t *testing.T) {
	server := apiHostTestServer()
	defer server.Close()
	root := t.TempDir()
	jsonPath := filepath.Join(root, "readiness.json")
	mdPath := filepath.Join(root, "readiness.md")
	var stdout, stderr bytes.Buffer

	rc := runAPIHost(&stdout, &stderr, []string{
		"readiness",
		"--target", "ok|" + server.URL + "/ok",
		"--out", jsonPath,
		"--markdown", mdPath,
	})
	if rc != 0 {
		t.Fatalf("runAPIHost readiness rc=%d stderr=%q", rc, stderr.String())
	}
	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read JSON report: %v", err)
	}
	var report struct {
		Schema  string `json:"schema"`
		Summary struct {
			ModelsConfirmed int  `json:"models_confirmed"`
			ReadinessGate   bool `json:"readiness_gate"`
		} `json:"summary"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatalf("unmarshal JSON report: %v", err)
	}
	if report.Schema != "fak.api-host-readiness.v1" || report.Summary.ModelsConfirmed != 1 || !report.Summary.ReadinessGate {
		t.Fatalf("unexpected readiness report: %+v", report)
	}
	md, err := os.ReadFile(mdPath)
	if err != nil {
		t.Fatalf("read markdown report: %v", err)
	}
	if !strings.Contains(string(md), "API-Host Readiness Probe") {
		t.Fatalf("markdown missing title: %s", md)
	}
}

func TestAPIHostAcceptanceCommandWritesReports(t *testing.T) {
	server := apiHostTestServer()
	defer server.Close()
	root := t.TempDir()
	jsonPath := filepath.Join(root, "acceptance.json")
	mdPath := filepath.Join(root, "acceptance.md")
	var stdout, stderr bytes.Buffer

	rc := runAPIHost(&stdout, &stderr, []string{
		"acceptance",
		"--target", "ok|openai-compatible|" + server.URL + "/ok||m1",
		"--root", root,
		"--out", jsonPath,
		"--markdown", mdPath,
	})
	if rc != 0 {
		t.Fatalf("runAPIHost acceptance rc=%d stderr=%q", rc, stderr.String())
	}
	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read JSON report: %v", err)
	}
	var report struct {
		Schema  string `json:"schema"`
		Summary struct {
			ReadyForLiveBridgeRun int  `json:"ready_for_live_bridge_run"`
			AcceptanceGate        bool `json:"acceptance_gate"`
		} `json:"summary"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatalf("unmarshal JSON report: %v", err)
	}
	if report.Schema != "fak.api-host-acceptance.v1" || report.Summary.ReadyForLiveBridgeRun != 1 || !report.Summary.AcceptanceGate {
		t.Fatalf("unexpected acceptance report: %+v", report)
	}
	md, err := os.ReadFile(mdPath)
	if err != nil {
		t.Fatalf("read markdown report: %v", err)
	}
	if !strings.Contains(string(md), "API-Host Acceptance Probe") {
		t.Fatalf("markdown missing title: %s", md)
	}
}

func writeAPIHostModelAccountsRoster(t *testing.T, baseURL string) string {
	t.Helper()
	body := fmt.Sprintf(`{
  "version": "fak-accounts/v1",
  "accounts": [
    {"id":"local-probe","kind":"local","base_url":%q},
    {"id":"claude-sub","kind":"anthropic","cred_env":"CLAUDE_CODE_OAUTH_TOKEN"}
  ],
  "default": "local-probe",
  "bindings": [
    {"model":"small","account":"local-probe","upstream_model":"m1"},
    {"model":"large","account":"claude-sub","upstream_model":"claude-opus-4-6"}
  ]
}`, baseURL)
	path := filepath.Join(t.TempDir(), "model-accounts.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write model account roster: %v", err)
	}
	return path
}

func writeAPIHostDeepSeekModelAccountsRoster(t *testing.T, openAIBaseURL, anthropicBaseURL string) string {
	t.Helper()
	body := fmt.Sprintf(`{
  "version": "fak-accounts/v1",
  "accounts": [
    {
      "id": "deepseek",
      "kind": "deepseek",
      "base_url": %q,
      "cred_env": "DEEPSEEK_API_KEY",
      "context_tokens": 1000000,
      "max_output_tokens": 384000
    },
    {
      "id": "deepseek-anthropic",
      "kind": "anthropic",
      "base_url": %q,
      "cred_env": "DEEPSEEK_API_KEY",
      "context_tokens": 1000000,
      "max_output_tokens": 384000
    }
  ],
  "default": "deepseek",
  "bindings": [
    {"model":"deepseek-pro","account":"deepseek","upstream_model":"deepseek-v4-pro"},
    {"model":"deepseek-flash","account":"deepseek","upstream_model":"deepseek-v4-flash"},
    {"model":"deepseek-pro-anthropic","account":"deepseek-anthropic","upstream_model":"deepseek-v4-pro"},
    {
      "model": "deepseek-chat-compat",
      "account": "deepseek",
      "upstream_model": "deepseek-chat",
      "compatibility_only": true,
      "deprecated_after_utc": "2026-07-24 15:59 UTC",
      "deprecated_alias_for": "deepseek-v4-flash non-thinking mode"
    }
  ]
}`, openAIBaseURL, anthropicBaseURL)
	path := filepath.Join(t.TempDir(), "deepseek-model-accounts.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write DeepSeek model account roster: %v", err)
	}
	return path
}

func TestAPIHostReadinessFromModelAccountsRoster(t *testing.T) {
	server := apiHostTestServer()
	defer server.Close()
	roster := writeAPIHostModelAccountsRoster(t, server.URL+"/ok")
	var stdout, stderr bytes.Buffer

	rc := runAPIHost(&stdout, &stderr, []string{
		"readiness",
		"--from-model-accounts", roster,
	})
	if rc != 0 {
		t.Fatalf("runAPIHost readiness rc=%d stderr=%q stdout=%s", rc, stderr.String(), stdout.String())
	}
	var report struct {
		Summary struct {
			Targets         int  `json:"targets"`
			ModelsConfirmed int  `json:"models_confirmed"`
			ReadinessGate   bool `json:"readiness_gate"`
		} `json:"summary"`
		Probes []struct {
			Name      string `json:"name"`
			ModelHint string `json:"model_hint"`
		} `json:"probes"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal JSON report: %v\n%s", err, stdout.String())
	}
	if report.Summary.Targets != 1 || report.Summary.ModelsConfirmed != 1 || !report.Summary.ReadinessGate {
		t.Fatalf("unexpected readiness summary: %+v\n%s", report.Summary, stdout.String())
	}
	if len(report.Probes) != 1 || report.Probes[0].Name != "local-probe" || report.Probes[0].ModelHint != "m1" {
		t.Fatalf("unexpected readiness probes: %+v\n%s", report.Probes, stdout.String())
	}
}

func TestAPIHostReadinessFromModelAccountsIncludesDeepSeekOpenAIOnly(t *testing.T) {
	server := apiHostTestServer()
	defer server.Close()
	t.Setenv("DEEPSEEK_API_KEY", "sk-must-not-print")
	roster := writeAPIHostDeepSeekModelAccountsRoster(t, server.URL+"/ok", server.URL+"/anthropic")
	var stdout, stderr bytes.Buffer

	rc := runAPIHost(&stdout, &stderr, []string{
		"readiness",
		"--from-model-accounts", roster,
	})
	if rc != 0 {
		t.Fatalf("runAPIHost readiness rc=%d stderr=%q stdout=%s", rc, stderr.String(), stdout.String())
	}
	var report struct {
		Summary struct {
			Targets         int  `json:"targets"`
			ModelsConfirmed int  `json:"models_confirmed"`
			ReadinessGate   bool `json:"readiness_gate"`
		} `json:"summary"`
		Probes []struct {
			Name      string   `json:"name"`
			ModelHint string   `json:"model_hint"`
			Models    []string `json:"models"`
		} `json:"probes"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal JSON report: %v\n%s", err, stdout.String())
	}
	if report.Summary.Targets != 1 || report.Summary.ModelsConfirmed != 1 || !report.Summary.ReadinessGate {
		t.Fatalf("unexpected readiness summary: %+v\n%s", report.Summary, stdout.String())
	}
	if len(report.Probes) != 1 || report.Probes[0].Name != "deepseek" || report.Probes[0].ModelHint == "" {
		t.Fatalf("unexpected DeepSeek readiness probes: %+v\n%s", report.Probes, stdout.String())
	}
	if strings.Contains(stdout.String(), "sk-must-not-print") {
		t.Fatalf("readiness output leaked the credential value:\n%s", stdout.String())
	}
}

func TestAPIHostAcceptanceFromModelAccountsRosterKeepsNativeAccounts(t *testing.T) {
	server := apiHostTestServer()
	defer server.Close()
	roster := writeAPIHostModelAccountsRoster(t, server.URL+"/ok")
	var stdout, stderr bytes.Buffer

	rc := runAPIHost(&stdout, &stderr, []string{
		"acceptance",
		"--from-model-accounts", roster,
		"--root", t.TempDir(),
	})
	if rc != 0 {
		t.Fatalf("runAPIHost acceptance rc=%d stderr=%q stdout=%s", rc, stderr.String(), stdout.String())
	}
	var report struct {
		Summary struct {
			Targets               int  `json:"targets"`
			ReadyForLiveBridgeRun int  `json:"ready_for_live_bridge_run"`
			WireSupportedUnprobed int  `json:"wire_supported_unprobed"`
			AcceptanceGate        bool `json:"acceptance_gate"`
		} `json:"summary"`
		Targets []struct {
			Name     string `json:"name"`
			Provider string `json:"provider"`
			Status   string `json:"status"`
		} `json:"targets"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal JSON report: %v\n%s", err, stdout.String())
	}
	if report.Summary.Targets != 2 || report.Summary.ReadyForLiveBridgeRun != 1 ||
		report.Summary.WireSupportedUnprobed != 1 || !report.Summary.AcceptanceGate {
		t.Fatalf("unexpected acceptance summary: %+v\n%s", report.Summary, stdout.String())
	}
	statuses := map[string]string{}
	providers := map[string]string{}
	for _, row := range report.Targets {
		statuses[row.Name] = row.Status
		providers[row.Name] = row.Provider
	}
	if statuses["local-probe"] != "READY_FOR_LIVE_BRIDGE_RUN" || providers["local-probe"] != "openai-compatible" {
		t.Fatalf("local account row wrong: statuses=%v providers=%v", statuses, providers)
	}
	if statuses["claude-sub"] != "WIRE_SUPPORTED_UNPROBED" || providers["claude-sub"] != "anthropic" {
		t.Fatalf("native account row wrong: statuses=%v providers=%v", statuses, providers)
	}
}

func TestAPIHostAcceptanceFromModelAccountsKeepsDeepSeekAnthropicUnprobed(t *testing.T) {
	server := apiHostTestServer()
	defer server.Close()
	t.Setenv("DEEPSEEK_API_KEY", "sk-must-not-print")
	roster := writeAPIHostDeepSeekModelAccountsRoster(t, server.URL+"/ok", server.URL+"/anthropic")
	var stdout, stderr bytes.Buffer

	rc := runAPIHost(&stdout, &stderr, []string{
		"acceptance",
		"--from-model-accounts", roster,
		"--root", t.TempDir(),
	})
	if rc != 0 {
		t.Fatalf("runAPIHost acceptance rc=%d stderr=%q stdout=%s", rc, stderr.String(), stdout.String())
	}
	var report struct {
		Summary struct {
			Targets               int  `json:"targets"`
			ReadyForLiveBridgeRun int  `json:"ready_for_live_bridge_run"`
			WireSupportedUnprobed int  `json:"wire_supported_unprobed"`
			AcceptanceGate        bool `json:"acceptance_gate"`
		} `json:"summary"`
		Targets []struct {
			Name            string `json:"name"`
			Provider        string `json:"provider"`
			Status          string `json:"status"`
			ReadinessStatus string `json:"readiness_status"`
		} `json:"targets"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal JSON report: %v\n%s", err, stdout.String())
	}
	if report.Summary.Targets != 2 || report.Summary.ReadyForLiveBridgeRun != 1 ||
		report.Summary.WireSupportedUnprobed != 1 || !report.Summary.AcceptanceGate {
		t.Fatalf("unexpected acceptance summary: %+v\n%s", report.Summary, stdout.String())
	}
	statuses := map[string]string{}
	providers := map[string]string{}
	readiness := map[string]string{}
	for _, row := range report.Targets {
		statuses[row.Name] = row.Status
		providers[row.Name] = row.Provider
		readiness[row.Name] = row.ReadinessStatus
	}
	if statuses["deepseek"] != "READY_FOR_LIVE_BRIDGE_RUN" || providers["deepseek"] != "deepseek" || readiness["deepseek"] != "MODELS_CONFIRMED" {
		t.Fatalf("DeepSeek OpenAI-compatible row wrong: statuses=%v providers=%v readiness=%v", statuses, providers, readiness)
	}
	if statuses["deepseek-anthropic"] != "WIRE_SUPPORTED_UNPROBED" || providers["deepseek-anthropic"] != "anthropic" || readiness["deepseek-anthropic"] != "NOT_PROBED" {
		t.Fatalf("DeepSeek Anthropic row wrong: statuses=%v providers=%v readiness=%v", statuses, providers, readiness)
	}
	if strings.Contains(stdout.String(), "sk-must-not-print") {
		t.Fatalf("acceptance output leaked the credential value:\n%s", stdout.String())
	}
}

func TestAPIHostRejectsMultipleSources(t *testing.T) {
	cases := [][]string{
		{"readiness", "--target", "ok|http://example.invalid", "--from-roster", "roster.json"},
		{"readiness", "--target", "ok|http://example.invalid", "--from-model-accounts", "accounts.json"},
		{"acceptance", "--from-roster", "roster.json", "--from-model-accounts", "accounts.json"},
	}
	for _, argv := range cases {
		var stdout, stderr bytes.Buffer
		rc := runAPIHost(&stdout, &stderr, argv)
		if rc != 2 {
			t.Fatalf("%v rc=%d, want 2", argv, rc)
		}
		if !strings.Contains(stderr.String(), "mutually exclusive") {
			t.Fatalf("%v stderr = %q", argv, stderr.String())
		}
	}
}
