package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunFixtureSmokeJSONClaimDisallowed(t *testing.T) {
	fixture := filepath.Join("..", "..", "internal", "livecodebench", "testdata", "fixture.json")
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	code := run([]string{"--fixture", fixture, "--check", "--json"})
	_ = w.Close()
	os.Stdout = oldStdout
	var out bytes.Buffer
	_, _ = out.ReadFrom(r)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stdout=%s", code, out.String())
	}
	if !strings.Contains(out.String(), `"result_claim_allowed": false`) {
		t.Fatalf("json did not pin result_claim_allowed=false:\n%s", out.String())
	}
}

// TestRunPreflightNeverProbesNetworkByDefault pins #2111: --preflight must
// never emit a benchmark number, and without --probe-dataset/--probe-gateway
// it must not attempt network I/O -- the dataset and gateway gates report
// "not probed" instead of blocking forever on an unreachable host.
func TestRunPreflightNeverProbesNetworkByDefault(t *testing.T) {
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	code := run([]string{"--preflight", "--json"})
	_ = w.Close()
	os.Stdout = oldStdout
	var out bytes.Buffer
	_, _ = out.ReadFrom(r)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stdout=%s", code, out.String())
	}
	var report struct {
		Schema             string `json:"schema"`
		Status             string `json:"status"`
		ResultClaimAllowed bool   `json:"result_claim_allowed"`
		Gates              []struct {
			Name   string `json:"name"`
			OK     bool   `json:"ok"`
			Detail string `json:"detail"`
		} `json:"gates"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("preflight output not valid JSON: %v\n%s", err, out.String())
	}
	if report.Schema != "fak.livecodebench-preflight.v1" {
		t.Fatalf("schema = %q", report.Schema)
	}
	if report.ResultClaimAllowed {
		t.Fatal("preflight must never allow a result claim")
	}
	if report.Status != "BLOCKED_PREFLIGHT" && report.Status != "READY" {
		t.Fatalf("status = %q, want BLOCKED_PREFLIGHT or READY", report.Status)
	}
	byName := map[string]bool{}
	for _, g := range report.Gates {
		byName[g.Name] = true
		if g.Name == "hf_dataset_reachable" || g.Name == "fak_gateway_reachable" {
			if g.OK {
				t.Fatalf("gate %q should not be OK without --probe-dataset/--probe-gateway", g.Name)
			}
			if !strings.Contains(g.Detail, "not probed") {
				t.Fatalf("gate %q detail = %q, want it to say not probed", g.Name, g.Detail)
			}
		}
	}
	for _, want := range []string{"uv_present", "python311_present", "hf_dataset_reachable", "fak_gateway_reachable", "sandbox_available"} {
		if !byName[want] {
			t.Fatalf("missing gate %q", want)
		}
	}
}
