package apihostprobe

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func testServer() *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/ok/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"m1"},{"id":"m2"}]}`))
	})
	mux.HandleFunc("/auth/models", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"Authentication required"}`))
	})
	mux.HandleFunc("/billing/models", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = w.Write([]byte(`{"error":{"code":"no_payment_method"}}`))
	})
	mux.HandleFunc("/edge/models", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"title":"Error 1010: Access denied","error_name":"browser_signature_banned"}`))
	})
	mux.HandleFunc("/weird/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":["m1"]}`))
	})
	return httptest.NewServer(mux)
}

func TestReadinessParseTarget(t *testing.T) {
	got, err := ParseReadinessTarget("n|http://x|KEY|m")
	if err != nil {
		t.Fatalf("ParseReadinessTarget returned error: %v", err)
	}
	want := ReadinessTarget{Name: "n", BaseURL: "http://x", APIKeyEnv: "KEY", ModelHint: "m"}
	if got != want {
		t.Fatalf("ParseReadinessTarget = %+v, want %+v", got, want)
	}
	for _, spec := range []string{"only-name", "n|", "n|/relative"} {
		if _, err := ParseReadinessTarget(spec); err == nil {
			t.Fatalf("ParseReadinessTarget(%q) returned nil error", spec)
		}
	}
}

func TestProbeReadinessTargetModelsConfirmed(t *testing.T) {
	server := testServer()
	defer server.Close()

	row := ProbeReadinessTarget(context.Background(), ReadinessTarget{Name: "ok", BaseURL: server.URL + "/ok"}, ReadinessOptions{Timeout: 2 * time.Second})
	if row.Status != "MODELS_CONFIRMED" {
		t.Fatalf("status = %s, want MODELS_CONFIRMED", row.Status)
	}
	if got := strings.Join(row.Models, ","); got != "m1,m2" {
		t.Fatalf("models = %v, want [m1 m2]", row.Models)
	}
}

func TestProbeReadinessTargetTypedHTTPStates(t *testing.T) {
	server := testServer()
	defer server.Close()

	tests := []struct {
		name string
		path string
		want string
	}{
		{name: "auth", path: "/auth", want: "AUTH_REQUIRED"},
		{name: "billing", path: "/billing", want: "BILLING_REQUIRED"},
		{name: "edge", path: "/edge", want: "ACCESS_DENIED"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			row := ProbeReadinessTarget(context.Background(), ReadinessTarget{Name: tc.name, BaseURL: server.URL + tc.path}, ReadinessOptions{Timeout: 2 * time.Second})
			if row.Status != tc.want {
				t.Fatalf("status = %s, want %s", row.Status, tc.want)
			}
		})
	}
}

func TestProbeReadinessTargetMissingEnvSkipsNetwork(t *testing.T) {
	server := testServer()
	defer server.Close()
	t.Setenv("NO_SUCH_API_KEY_FOR_TEST", "")

	row := ProbeReadinessTarget(context.Background(), ReadinessTarget{
		Name:      "missing",
		BaseURL:   server.URL + "/ok",
		APIKeyEnv: "NO_SUCH_API_KEY_FOR_TEST",
	}, ReadinessOptions{Timeout: 2 * time.Second})
	if row.Status != "AUTH_ENV_MISSING" {
		t.Fatalf("status = %s, want AUTH_ENV_MISSING", row.Status)
	}
	if row.HTTPStatus != nil {
		t.Fatalf("http status = %v, want nil", *row.HTTPStatus)
	}
}

func TestReadinessInvalidTargetAndReportGate(t *testing.T) {
	row := ProbeReadinessTarget(context.Background(), ReadinessTarget{Name: "bad"}, ReadinessOptions{})
	if row.Status != "INVALID_TARGET" {
		t.Fatalf("status = %s, want INVALID_TARGET", row.Status)
	}
	if !strings.Contains(row.Error, "base_url") {
		t.Fatalf("error = %q, want base_url detail", row.Error)
	}

	report := BuildReadinessReport(context.Background(), []ReadinessTarget{{Name: "bad"}}, ReadinessOptions{})
	if report.Summary.ReadinessGate {
		t.Fatal("readiness gate = true, want false")
	}
	if report.Summary.InvalidTargets != 1 {
		t.Fatalf("invalid targets = %d, want 1", report.Summary.InvalidTargets)
	}
}

func TestReadinessReportAndMarkdown(t *testing.T) {
	server := testServer()
	defer server.Close()

	report := BuildReadinessReport(context.Background(), []ReadinessTarget{
		{Name: "ok", BaseURL: server.URL + "/ok"},
		{Name: "auth", BaseURL: server.URL + "/auth"},
	}, ReadinessOptions{Timeout: 2 * time.Second})
	if !report.Summary.ReadinessGate {
		t.Fatal("readiness gate = false, want true")
	}
	if report.Summary.ModelsConfirmed != 1 {
		t.Fatalf("models confirmed = %d, want 1", report.Summary.ModelsConfirmed)
	}
	raw, err := MarshalReport(report)
	if err != nil {
		t.Fatalf("marshal report: %v", err)
	}
	var decoded ReadinessReport
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("report JSON did not unmarshal: %v", err)
	}
	if decoded.Schema != ReadinessSchema {
		t.Fatalf("schema = %q, want %q", decoded.Schema, ReadinessSchema)
	}
	if !strings.Contains(ReadinessMarkdown(report), "API-Host Readiness Probe") {
		t.Fatal("markdown missing title")
	}
}

func TestLoadReadinessRosterTargetsFiltersOpenAICompatible(t *testing.T) {
	server := testServer()
	defer server.Close()
	root := t.TempDir()
	roster := filepath.Join(root, "roster.json")
	body := map[string]any{
		"targets": []map[string]any{
			{
				"name":           "ok",
				"provider":       "openai-compatible",
				"contract_class": "openai_compatible_upstream",
				"base_url":       server.URL + "/ok",
				"api_key_env":    "",
				"model_hint":     "m1",
				"status":         "SUPPORTED_TEMPLATE",
			},
			{
				"name":           "native",
				"provider":       "anthropic",
				"contract_class": "native_provider_transcript_adapters",
				"base_url":       "https://example.invalid",
				"api_key_env":    "",
				"model_hint":     "",
				"status":         "SUPPORTED_TEMPLATE",
			},
			{
				"name":           "bad",
				"provider":       "openai-compatible",
				"contract_class": "openai_compatible_upstream",
				"base_url":       "https://example.invalid",
				"status":         "INVALID_TARGET",
			},
		},
	}
	raw, _ := json.Marshal(body)
	if err := os.WriteFile(roster, raw, 0o644); err != nil {
		t.Fatalf("write roster: %v", err)
	}

	targets, err := LoadReadinessRosterTargets(roster)
	if err != nil {
		t.Fatalf("LoadReadinessRosterTargets: %v", err)
	}
	if len(targets) != 1 || targets[0].Name != "ok" {
		t.Fatalf("targets = %+v, want only ok", targets)
	}
}

func TestAcceptanceParseTargetAndProviderAliases(t *testing.T) {
	got, err := ParseAcceptanceTarget("n|grok|http://x|KEY|m")
	if err != nil {
		t.Fatalf("ParseAcceptanceTarget returned error: %v", err)
	}
	want := AcceptanceTarget{Name: "n", Provider: "xai", BaseURL: "http://x", APIKeyEnv: "KEY", ModelHint: "m"}
	if got != want {
		t.Fatalf("ParseAcceptanceTarget = %+v, want %+v", got, want)
	}
	for _, spec := range []string{"n|openai-compatible", "n|openai-compatible|"} {
		if _, err := ParseAcceptanceTarget(spec); err == nil {
			t.Fatalf("ParseAcceptanceTarget(%q) returned nil error", spec)
		}
	}
}

func TestAcceptanceOpenAICompatibleReadyForLiveBridgeRun(t *testing.T) {
	server := testServer()
	defer server.Close()

	row := ClassifyAcceptanceTarget(context.Background(), AcceptanceTarget{
		Name: "ok", Provider: "openai-compatible", BaseURL: server.URL + "/ok", ModelHint: "m1",
	}, AcceptanceOptions{Timeout: 2 * time.Second}, nil)
	if row.Status != "READY_FOR_LIVE_BRIDGE_RUN" {
		t.Fatalf("status = %s, want READY_FOR_LIVE_BRIDGE_RUN", row.Status)
	}
	if row.ContractClass != "openai_compatible_upstream" {
		t.Fatalf("contract class = %s", row.ContractClass)
	}
	if !strings.Contains(row.NextLiveCommand, "run_transcript_adapter_sweep.ps1") || strings.Contains(row.NextLiveCommand, "-Provider") {
		t.Fatalf("next live command = %q", row.NextLiveCommand)
	}
}

func TestAcceptanceTypedExternalBlockerAndShapeMismatch(t *testing.T) {
	server := testServer()
	defer server.Close()

	auth := ClassifyAcceptanceTarget(context.Background(), AcceptanceTarget{Name: "auth", Provider: "openai-compatible", BaseURL: server.URL + "/auth"}, AcceptanceOptions{Timeout: 2 * time.Second}, nil)
	weird := ClassifyAcceptanceTarget(context.Background(), AcceptanceTarget{Name: "weird", Provider: "openai-compatible", BaseURL: server.URL + "/weird"}, AcceptanceOptions{Timeout: 2 * time.Second}, nil)
	if auth.Status != "AUTH_REQUIRED" {
		t.Fatalf("auth status = %s, want AUTH_REQUIRED", auth.Status)
	}
	if weird.Status != "MODELS_SHAPE_MISMATCH" {
		t.Fatalf("weird status = %s, want MODELS_SHAPE_MISMATCH", weird.Status)
	}
}

func TestAcceptanceMissingAPIKeyEnvIsTypedWithoutNetwork(t *testing.T) {
	server := testServer()
	defer server.Close()
	t.Setenv("NO_SUCH_API_KEY_FOR_ACCEPTANCE_TEST", "")

	row := ClassifyAcceptanceTarget(context.Background(), AcceptanceTarget{
		Name:      "missing",
		Provider:  "openai-compatible",
		BaseURL:   server.URL + "/ok",
		APIKeyEnv: "NO_SUCH_API_KEY_FOR_ACCEPTANCE_TEST",
	}, AcceptanceOptions{Timeout: 2 * time.Second}, nil)
	if row.Status != "NEEDS_AUTH_ENV" {
		t.Fatalf("status = %s, want NEEDS_AUTH_ENV", row.Status)
	}
	if row.ReadinessStatus != "AUTH_ENV_MISSING" {
		t.Fatalf("readiness status = %s, want AUTH_ENV_MISSING", row.ReadinessStatus)
	}
}

func TestAcceptanceNativeAndDirectWiresSupportedButUnprobed(t *testing.T) {
	tests := []struct {
		provider string
		class    string
	}{
		{"anthropic", "native_provider_transcript_adapters"},
		{"gemini", "native_provider_transcript_adapters"},
		{"direct-http", "direct_kernel_http_syscall"},
		{"direct-mcp", "direct_kernel_mcp_syscall"},
	}
	for _, tc := range tests {
		row := ClassifyAcceptanceTarget(context.Background(), AcceptanceTarget{Name: tc.provider, Provider: tc.provider, BaseURL: "http://example.invalid"}, AcceptanceOptions{Timeout: 2 * time.Second}, nil)
		if row.Status != "WIRE_SUPPORTED_UNPROBED" {
			t.Fatalf("%s status = %s, want WIRE_SUPPORTED_UNPROBED", tc.provider, row.Status)
		}
		if row.ContractClass != tc.class {
			t.Fatalf("%s class = %s, want %s", tc.provider, row.ContractClass, tc.class)
		}
	}
}

func TestAcceptanceUnsupportedAndInvalidTargetsFailGate(t *testing.T) {
	server := testServer()
	defer server.Close()
	root := t.TempDir()

	unsupported := BuildAcceptanceReport(context.Background(), []AcceptanceTarget{
		{Name: "ok", Provider: "openai-compatible", BaseURL: server.URL + "/ok"},
		{Name: "bad", Provider: "unknown-provider", BaseURL: "http://example.invalid"},
	}, AcceptanceOptions{Timeout: 2 * time.Second, Root: root})
	if unsupported.Summary.AcceptanceGate {
		t.Fatal("unsupported report gate = true, want false")
	}
	if unsupported.Summary.UnsupportedWire != 1 {
		t.Fatalf("unsupported wire = %d, want 1", unsupported.Summary.UnsupportedWire)
	}

	invalid := BuildAcceptanceReport(context.Background(), []AcceptanceTarget{
		{Name: "bad", Provider: "openai-compatible"},
	}, AcceptanceOptions{Timeout: 2 * time.Second, Root: root})
	if invalid.Summary.AcceptanceGate {
		t.Fatal("invalid report gate = true, want false")
	}
	if invalid.Summary.InvalidTargets != 1 || invalid.Targets[0].Probe != nil {
		t.Fatalf("invalid report = %+v", invalid)
	}
}

func TestAcceptanceSweepArtifactErrorsFailGate(t *testing.T) {
	root := t.TempDir()
	bad := filepath.Join(root, "fak", "experiments", "agent-live", "transcript-adapter-sweep-bad", "sweep-summary.json")
	if err := os.MkdirAll(filepath.Dir(bad), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(bad, []byte("{bad json"), 0o644); err != nil {
		t.Fatalf("write bad summary: %v", err)
	}

	report := BuildAcceptanceReport(context.Background(), []AcceptanceTarget{
		{Name: "direct", Provider: "direct-http", BaseURL: "http://example.invalid"},
	}, AcceptanceOptions{Timeout: 2 * time.Second, Root: root})
	if report.Summary.AcceptanceGate {
		t.Fatal("acceptance gate = true, want false")
	}
	if report.Summary.SweepArtifactErrors != 1 {
		t.Fatalf("sweep artifact errors = %d, want 1", report.Summary.SweepArtifactErrors)
	}
	if report.ArtifactErrors[0].Path != "fak/experiments/agent-live/transcript-adapter-sweep-bad/sweep-summary.json" {
		t.Fatalf("artifact path = %q", report.ArtifactErrors[0].Path)
	}
	if !strings.Contains(report.ArtifactErrors[0].Error, "invalid JSON") {
		t.Fatalf("artifact error = %q", report.ArtifactErrors[0].Error)
	}
}

func TestAcceptanceNonObjectSweepRowsFailGate(t *testing.T) {
	root := t.TempDir()
	bad := filepath.Join(root, "fak", "experiments", "agent-live", "transcript-adapter-sweep-bad", "sweep-summary.json")
	if err := os.MkdirAll(filepath.Dir(bad), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(bad, []byte(`[[]]`), 0o644); err != nil {
		t.Fatalf("write bad summary: %v", err)
	}

	report := BuildAcceptanceReport(context.Background(), []AcceptanceTarget{
		{Name: "direct", Provider: "direct-http", BaseURL: "http://example.invalid"},
	}, AcceptanceOptions{Timeout: 2 * time.Second, Root: root})
	if report.Summary.AcceptanceGate {
		t.Fatal("acceptance gate = true, want false")
	}
	if report.Summary.SweepArtifactErrors != 1 {
		t.Fatalf("sweep artifact errors = %d, want 1", report.Summary.SweepArtifactErrors)
	}
	if report.ArtifactErrors[0].RowIndex == nil || *report.ArtifactErrors[0].RowIndex != 0 {
		t.Fatalf("row index = %v, want 0", report.ArtifactErrors[0].RowIndex)
	}
}

func TestAcceptanceLiveSweepOverridesReadiness(t *testing.T) {
	server := testServer()
	defer server.Close()

	failed := ClassifyAcceptanceTarget(context.Background(), AcceptanceTarget{
		Name: "ok", Provider: "openai-compatible", BaseURL: server.URL + "/ok", ModelHint: "m1",
	}, AcceptanceOptions{Timeout: 2 * time.Second}, []map[string]any{{
		"generated_at": "2026-06-18T00:00:00-07:00",
		"kind":         "api",
		"base_url":     server.URL + "/ok",
		"model":        "m1",
		"status":       "failed",
		"error":        `fak: planner: HTTP 402: {"error":{"code":"no_payment_method"}}`,
	}})
	if failed.ReadinessStatus != "MODELS_CONFIRMED" {
		t.Fatalf("readiness status = %s, want MODELS_CONFIRMED", failed.ReadinessStatus)
	}
	if failed.Status != "BILLING_REQUIRED" {
		t.Fatalf("status = %s, want BILLING_REQUIRED", failed.Status)
	}

	confirmed := ClassifyAcceptanceTarget(context.Background(), AcceptanceTarget{
		Name: "ok", Provider: "openai-compatible", BaseURL: server.URL + "/ok", ModelHint: "m1",
	}, AcceptanceOptions{Timeout: 2 * time.Second}, []map[string]any{{
		"generated_at":   "2026-06-18T00:00:00-07:00",
		"kind":           "api",
		"base_url":       server.URL + "/ok",
		"model":          "m1",
		"status":         "ok",
		"live":           true,
		"transcript_sha": "abc",
	}})
	if confirmed.Status != "LIVE_BRIDGE_CONFIRMED" {
		t.Fatalf("status = %s, want LIVE_BRIDGE_CONFIRMED", confirmed.Status)
	}
}

func TestLoadAcceptanceRosterTargetsClassifiesSupportedWires(t *testing.T) {
	server := testServer()
	defer server.Close()
	root := t.TempDir()
	roster := filepath.Join(root, "roster.json")
	body := map[string]any{
		"targets": []map[string]any{
			{
				"name":           "ok",
				"provider":       "openai-compatible",
				"contract_class": "openai_compatible_upstream",
				"base_url":       server.URL + "/ok",
				"api_key_env":    "",
				"model_hint":     "m1",
				"status":         "SUPPORTED_TEMPLATE",
			},
			{
				"name":           "native",
				"provider":       "anthropic",
				"contract_class": "native_provider_transcript_adapters",
				"base_url":       "https://example.invalid",
				"api_key_env":    "",
				"model_hint":     "",
				"status":         "SUPPORTED_TEMPLATE",
			},
		},
	}
	raw, _ := json.Marshal(body)
	if err := os.WriteFile(roster, raw, 0o644); err != nil {
		t.Fatalf("write roster: %v", err)
	}

	targets, err := LoadAcceptanceRosterTargets(roster)
	if err != nil {
		t.Fatalf("LoadAcceptanceRosterTargets: %v", err)
	}
	report := BuildAcceptanceReport(context.Background(), targets, AcceptanceOptions{Timeout: 2 * time.Second, Root: root})
	statuses := map[string]string{}
	for _, row := range report.Targets {
		statuses[row.Name] = row.Status
	}
	if statuses["ok"] != "READY_FOR_LIVE_BRIDGE_RUN" {
		t.Fatalf("ok status = %s", statuses["ok"])
	}
	if statuses["native"] != "WIRE_SUPPORTED_UNPROBED" {
		t.Fatalf("native status = %s", statuses["native"])
	}
}

func TestAppVersionHonorsEnvVersionFileAndConflictMarkers(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "VERSION"), []byte("1.2.3\n"), 0o644); err != nil {
		t.Fatalf("write version: %v", err)
	}
	if got := AppVersion(root); got != "1.2.3" {
		t.Fatalf("AppVersion(root) = %q, want 1.2.3", got)
	}
	t.Setenv("FAK_APP_VERSION", "9.9.9")
	if got := AppVersion(root); got != "9.9.9" {
		t.Fatalf("AppVersion with env = %q, want 9.9.9", got)
	}
	t.Setenv("FAK_APP_VERSION", "")
	if err := os.WriteFile(filepath.Join(root, "VERSION"), []byte("<<<<<<< HEAD\n"), 0o644); err != nil {
		t.Fatalf("write conflict version: %v", err)
	}
	if got := AppVersion(root); got != "dev" {
		t.Fatalf("AppVersion conflict = %q, want dev", got)
	}
}
