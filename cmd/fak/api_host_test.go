package main

import (
	"bytes"
	"encoding/json"
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

func TestAPIHostRejectsTargetAndRosterTogether(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runAPIHost(&stdout, &stderr, []string{
		"readiness",
		"--target", "ok|http://example.invalid",
		"--from-roster", "roster.json",
	})
	if rc != 2 {
		t.Fatalf("rc=%d, want 2", rc)
	}
	if !strings.Contains(stderr.String(), "mutually exclusive") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}
