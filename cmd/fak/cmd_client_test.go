package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRunClient_Help(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runClient(&stdout, &stderr, []string{"help"})
	if code != 0 {
		t.Fatalf("expected code 0, got %d", code)
	}
	if !strings.Contains(stdout.String(), "fak client health") {
		t.Errorf("expected health usage, got:\n%s", stdout.String())
	}
}

func TestRunClient_HealthAndPing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	for _, subcmd := range []string{"health", "ping"} {
		var stdout, stderr bytes.Buffer
		code := runClient(&stdout, &stderr, []string{subcmd, "--url", server.URL})
		if code != 0 {
			t.Fatalf("%s: expected exit code 0, got %d; stderr: %s", subcmd, code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "OK") {
			t.Errorf("%s: expected stdout to contain OK, got %q", subcmd, stdout.String())
		}
	}
}

func TestRunClient_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	code := runClient(&stdout, &stderr, []string{"health", "--url", server.URL})
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}
	if !strings.Contains(stderr.String(), "error:") {
		t.Errorf("expected stderr to contain error:, got %q", stderr.String())
	}
}
