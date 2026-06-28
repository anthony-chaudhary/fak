package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/bgloop"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestBgloopDemoWitnesses is the host-free witness at the CLI layer: the offline demo
// runs a heartbeat and a panicking loop and reports that the heartbeat progressed
// while the panic was contained — exit 0 and a WITNESS line.
func TestBgloopDemoWitnesses(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runBgloop(&out, &errb, []string{"demo", "--duration", "300ms", "--interval", "20ms"})
	if rc != 0 {
		t.Fatalf("demo rc=%d (stderr=%q)", rc, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "WITNESS:") {
		t.Errorf("demo output missing WITNESS line:\n%s", s)
	}
	if !strings.Contains(s, "heartbeat") || !strings.Contains(s, "flaky") {
		t.Errorf("demo output should list both loops:\n%s", s)
	}
}

// TestBgloopStatusTable renders a live-server snapshot fetched over HTTP.
func TestBgloopStatusTable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/fak/loops" {
			http.NotFound(w, r)
			return
		}
		resp := gateway.BgloopsResponse{
			Schema: "fak.bgloops.v1",
			Loops: []bgloop.Status{
				{Name: "heartbeat", State: bgloop.StateIdle, Interval: "30s", Ticks: 7},
				{Name: "sweeper", State: bgloop.StateBackoff, Interval: "1m", Errors: 2, Restarts: 2, LastErr: "disk full"},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	var out, errb bytes.Buffer
	rc := runBgloop(&out, &errb, []string{"status", "--addr", srv.URL})
	if rc != 0 {
		t.Fatalf("status rc=%d (stderr=%q)", rc, errb.String())
	}
	s := out.String()
	for _, want := range []string{"heartbeat", "sweeper", "disk full", "2 loop(s)"} {
		if !strings.Contains(s, want) {
			t.Errorf("status table missing %q:\n%s", want, s)
		}
	}
}

// TestBgloopStatusJSON passes the raw body through with --json.
func TestBgloopStatusJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(gateway.BgloopsResponse{Schema: "fak.bgloops.v1"})
	}))
	defer srv.Close()

	var out, errb bytes.Buffer
	rc := runBgloop(&out, &errb, []string{"status", "--addr", srv.URL, "--json"})
	if rc != 0 {
		t.Fatalf("status --json rc=%d (stderr=%q)", rc, errb.String())
	}
	if !strings.Contains(out.String(), "fak.bgloops.v1") {
		t.Errorf("--json output missing schema:\n%s", out.String())
	}
}

func TestBgloopUnknownSubcommand(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runBgloop(&out, &errb, []string{"frobnicate"}); rc != 2 {
		t.Errorf("unknown subcommand rc=%d want 2", rc)
	}
}
