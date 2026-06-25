package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// runCA drives runCallavoid with a stdin string and captured streams, returning
// (exit, stdout, stderr).
func runCA(t *testing.T, stdin string, args ...string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := runCallavoid(strings.NewReader(stdin), &out, &errb, args)
	return code, out.String(), errb.String()
}

func TestCallavoidProveMemoMemoizes(t *testing.T) {
	// A pure call seen many times in a near-static world memoizes (net-positive).
	code, out, errb := runCA(t, `{"accesses":20,"validate_cost":0.02,"mutation_rate":0.05,"capture_cost":0.1}`, "prove-memo")
	if code != 0 {
		t.Fatalf("prove-memo exit=%d, want 0\nstderr=%s", code, errb)
	}
	var proof struct {
		Status   string `json:"status"`
		Decision string `json:"decision"`
	}
	if err := json.Unmarshal([]byte(out), &proof); err != nil {
		t.Fatalf("prove-memo output is not valid JSON: %v\n%s", err, out)
	}
	if proof.Status != "PROVEN" || proof.Decision != "memoize" {
		t.Fatalf("prove-memo verdict = %s/%s, want PROVEN/memoize", proof.Status, proof.Decision)
	}
}

func TestCallavoidAccountAmplifies(t *testing.T) {
	code, out, errb := runCA(t, `{"execute":4,"memo_hit":6}`, "account")
	if code != 0 {
		t.Fatalf("account exit=%d, want 0\nstderr=%s", code, errb)
	}
	var rep struct {
		Status        string  `json:"status"`
		Grade         string  `json:"grade"`
		Amplification float64 `json:"amplification"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("account output is not valid JSON: %v\n%s", err, out)
	}
	if rep.Status != "amplifying" || rep.Grade != "B" {
		t.Fatalf("account verdict = %s/%s, want amplifying/B", rep.Status, rep.Grade)
	}
	if rep.Amplification <= 1 {
		t.Fatalf("amplification = %v, want > 1", rep.Amplification)
	}
}

func TestCallavoidAccountGateFailsOnRegression(t *testing.T) {
	// A stale-miss-only window is a net loss; --gate must exit 1.
	code, _, errb := runCA(t, `{"stale_miss":5}`, "account", "--gate")
	if code != 1 {
		t.Fatalf("regressing --gate exit=%d, want 1\nstderr=%s", code, errb)
	}
	if !strings.Contains(errb, "regressing") {
		t.Fatalf("gate failure should name the regression:\n%s", errb)
	}
	// Without --gate the same window exits 0 (the report is informational, not a gate).
	if code, _, _ := runCA(t, `{"stale_miss":5}`, "account"); code != 0 {
		t.Fatalf("regressing without --gate exit=%d, want 0", code)
	}
}

func TestCallavoidRejectsMalformedInput(t *testing.T) {
	// An unknown field (a typo like memohit for memo_hit) fails loudly, exit 2.
	if code, _, _ := runCA(t, `{"memohit":9}`, "account"); code != 2 {
		t.Fatalf("unknown-field input exit=%d, want 2", code)
	}
	// Non-JSON input is exit 2.
	if code, _, _ := runCA(t, `not json`, "prove-memo"); code != 2 {
		t.Fatalf("non-JSON input exit=%d, want 2", code)
	}
	// Empty stdin is exit 2 (no silent zero-value decision).
	if code, _, _ := runCA(t, ``, "account"); code != 2 {
		t.Fatalf("empty stdin exit=%d, want 2", code)
	}
}

func TestCallavoidUnknownSubcommand(t *testing.T) {
	if code, _, errb := runCA(t, ``, "frobnicate"); code != 2 || !strings.Contains(errb, "unknown subcommand") {
		t.Fatalf("unknown subcommand exit=%d stderr=%q, want 2 + a named error", code, errb)
	}
	// No subcommand at all is a usage error, exit 2.
	if code, _, _ := runCA(t, ``); code != 2 {
		t.Fatalf("no subcommand exit=%d, want 2", code)
	}
}
