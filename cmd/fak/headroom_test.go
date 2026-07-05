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

func TestRunHeadroomListShowsPlugins(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"list"}); code != 0 {
		t.Fatalf("list exit=%d", code)
	}
	for _, want := range []string{"noop", "native", "headroom"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("list missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunHeadroomStatus(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"status"}); code != 0 {
		t.Fatalf("status exit=%d", code)
	}
	if !strings.Contains(out.String(), "selected:") || !strings.Contains(out.String(), "headroom url:") {
		t.Fatalf("status output unexpected:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "plugin health:") || !strings.Contains(out.String(), "fidelity") {
		t.Fatalf("status output missing plugin health table:\n%s", out.String())
	}
}

func TestRunHeadroomStatusJSONReportsPluginHealth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/stats" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	t.Setenv("FAK_HEADROOM_URL", srv.URL)

	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"status", "--json"}); code != 0 {
		t.Fatalf("status --json exit=%d stderr=%s", code, errb.String())
	}
	var rep struct {
		Selected          string `json:"selected"`
		HeadroomReachable bool   `json:"headroom_reachable"`
		Plugins           []struct {
			Name         string `json:"name"`
			Owner        string `json:"owner"`
			Dependency   string `json:"dependency"`
			Fidelity     string `json:"fidelity"`
			Status       string `json:"status"`
			Reachability string `json:"reachability"`
		} `json:"plugins"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if !rep.HeadroomReachable {
		t.Fatalf("headroom sidecar should be reachable through stub: %+v", rep)
	}
	var sawNative, sawHeadroom bool
	for _, p := range rep.Plugins {
		switch p.Name {
		case "native":
			sawNative = p.Owner == "fak" && p.Dependency == "in_process" && p.Fidelity == "recoverable"
		case "headroom":
			sawHeadroom = p.Owner == "external" && p.Dependency == "external_http_sidecar" &&
				p.Status == "available" && p.Reachability == "reachable"
		}
	}
	if !sawNative || !sawHeadroom {
		t.Fatalf("plugin health missing native/headroom descriptors: %+v", rep.Plugins)
	}
}

func TestRunHeadroomCompressFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "blob.json")
	pretty := "{\n    \"a\": 1,\n    \"b\": [\n        1,\n        2,\n        3\n    ]\n}\n"
	if err := os.WriteFile(path, []byte(pretty), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"compress", "--via", "native", path}); code != 0 {
		t.Fatalf("compress exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "compressed: true") || !strings.Contains(s, "status:     saved") || !strings.Contains(s, "json-min") {
		t.Fatalf("compress did not report a JSON saving:\n%s", s)
	}
}

func TestRunHeadroomCompressEmit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "blob.json")
	if err := os.WriteFile(path, []byte("{\n  \"x\": 1\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"compress", "--via", "native", "--emit", path}); code != 0 {
		t.Fatalf("emit exit=%d", code)
	}
	if out.String() != `{"x":1}` {
		t.Fatalf("emit should write minified bytes, got %q", out.String())
	}
}

func TestRunHeadroomUnknownSubcommand(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"frobnicate"}); code != 2 {
		t.Fatalf("unknown subcommand exit=%d, want 2", code)
	}
}

func TestRunHeadroomCompressUnknownVia(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "blob.txt")
	if err := os.WriteFile(path, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"compress", "--via", "nope", path}); code != 2 {
		t.Fatalf("unknown --via exit=%d, want 2", code)
	}
}

func TestRunHeadroomBenchJSONReportsOutcomeAttribution(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runHeadroom(&out, &errb, []string{"bench", "--via", "native", "--json"}); code != 0 {
		t.Fatalf("bench --json exit=%d stderr=%s", code, errb.String())
	}
	var rep struct {
		Compressor string `json:"compressor"`
		Owner      string `json:"owner"`
		Dependency string `json:"dependency"`
		Fidelity   string `json:"fidelity"`
		Evidence   string `json:"evidence"`
		Status     string `json:"status"`
		Samples    []struct {
			Name   string `json:"name"`
			Status string `json:"status"`
			Reason string `json:"reason"`
		} `json:"samples"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if rep.Compressor != "native" || rep.Owner != "fak" || rep.Dependency != "in_process" ||
		rep.Fidelity != "recoverable" || rep.Evidence != "witnessed" || rep.Status != "measured" {
		t.Fatalf("bench attribution/status = %+v", rep)
	}
	var sawSaved, sawNoEffect bool
	for _, sample := range rep.Samples {
		if sample.Status == "saved" && sample.Reason != "" {
			sawSaved = true
		}
		if sample.Status == "no_effect" {
			sawNoEffect = true
		}
	}
	if !sawSaved || !sawNoEffect {
		t.Fatalf("bench samples missing saved/no_effect outcomes: %+v", rep.Samples)
	}
}
