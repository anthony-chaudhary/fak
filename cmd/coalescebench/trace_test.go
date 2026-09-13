// Package main tests for the coalescebench real-activation-trace witness (#1298).
// Everything here is resource-free and deterministic: no model file, no network,
// no clock — the point under test is the fail-closed no-trace verdict and the
// [SW-VERIFIED] receipt the witness path renders for one captured trace.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/deepseekv4moe"
)

func writeTrace(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write trace fixture: %v", err)
	}
	return path
}

// TestLoadActivationTraceFailsClosedNoTrace proves every load failure mode —
// empty path, missing file, malformed JSON, route-less JSON — surfaces as an
// error wrapping ErrNoTrace rather than a zero-value trace.
func TestLoadActivationTraceFailsClosedNoTrace(t *testing.T) {
	validJSON := `{"layers":2,"experts":4,"top_k":2,"group_bytes":1,"budget_bytes":4,"routes":[{"Layer":0,"Experts":[0,1]}]}`
	emptyRoutes := `{"layers":2,"experts":4,"top_k":2,"group_bytes":1,"budget_bytes":4,"routes":[]}`
	absentRoutes := `{"layers":2,"experts":4,"top_k":2,"group_bytes":1,"budget_bytes":4}`

	tests := []struct {
		name string
		path func(*testing.T) string
	}{
		{"empty path", func(t *testing.T) string { return "" }},
		{"missing file", func(t *testing.T) string { return filepath.Join(t.TempDir(), "does-not-exist.json") }},
		{"invalid json", func(t *testing.T) string { return writeTrace(t, "bad.json", "{invalid json") }},
		{"empty routes", func(t *testing.T) string { return writeTrace(t, "empty.json", emptyRoutes) }},
		{"absent routes", func(t *testing.T) string { return writeTrace(t, "absent.json", absentRoutes) }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := loadActivationTrace(tc.path(t)); !errors.Is(err, deepseekv4moe.ErrNoTrace) {
				t.Fatalf("error = %v, want ErrNoTrace", err)
			}
		})
	}

	if trace, err := loadActivationTrace(writeTrace(t, "valid.json", validJSON)); err != nil || len(trace.Routes) != 1 {
		t.Fatalf("valid trace load = (%+v, %v), want one route", trace, err)
	}
}

// TestRunTraceWitnessReceipt pins the [SW-VERIFIED] text receipt and the additive
// JSON artifact for a valid trace, cross-checking the artifact witness against a
// fresh load + witness call so the emitted numbers cannot drift.
func TestRunTraceWitnessReceipt(t *testing.T) {
	body := `{"schema":"fak.expert-activation-trace/v1","name":"unit-trace","source":"hand-authored","layers":2,"experts":4,"top_k":2,"group_bytes":1,"budget_bytes":4,"routes":[{"Layer":0,"Experts":[0,1]},{"Layer":0,"Experts":[2,3]},{"Layer":0,"Experts":[0,1]}]}`
	path := writeTrace(t, "trace.json", body)
	artifact := filepath.Join(t.TempDir(), "receipt.json")

	var buf bytes.Buffer
	if err := runTraceWitness(path, artifact, &buf); err != nil {
		t.Fatalf("runTraceWitness: %v", err)
	}
	out := buf.String()
	for _, want := range []string{
		"[SW-VERIFIED]",
		"schema=fak.expert-cache-hitrate-witness/v1",
		"policy=lru",
		"hit_rate=",
		"byte_weighted_hit_rate=",
		"belady_regret_hits=",
		"NOT a served measurement",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("receipt missing %q:\n%s", want, out)
		}
	}

	raw, err := os.ReadFile(artifact)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	var report struct {
		Schema  string                                  `json:"schema"`
		Witness deepseekv4moe.ExpertCacheHitRateWitness `json:"witness"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatalf("decode artifact: %v", err)
	}
	if report.Schema != deepseekv4moe.ExpertCacheHitRateWitnessSchema {
		t.Fatalf("artifact schema = %q, want %q", report.Schema, deepseekv4moe.ExpertCacheHitRateWitnessSchema)
	}

	loaded, err := loadActivationTrace(path)
	if err != nil {
		t.Fatalf("reload trace: %v", err)
	}
	want, err := deepseekv4moe.WitnessExpertCacheHitRate(loaded)
	if err != nil {
		t.Fatalf("recompute witness: %v", err)
	}
	if report.Witness != want {
		t.Fatalf("artifact witness = %+v, want %+v", report.Witness, want)
	}
	if report.Witness.Label != "SW-VERIFIED" || !report.Witness.HitRateKnown {
		t.Fatalf("artifact witness identity = %+v", report.Witness)
	}
}

// TestRunTraceWitnessNoTraceVerdict proves a missing trace is a typed no-trace
// verdict printed to out — no fabricated numbers — and returned as an error.
func TestRunTraceWitnessNoTraceVerdict(t *testing.T) {
	var buf bytes.Buffer
	err := runTraceWitness(filepath.Join(t.TempDir(), "missing.json"), "", &buf)
	if err == nil {
		t.Fatal("runTraceWitness on a missing path returned nil error")
	}
	if !errors.Is(err, deepseekv4moe.ErrNoTrace) {
		t.Fatalf("error = %v, want ErrNoTrace", err)
	}
	if out := buf.String(); !strings.Contains(out, "no-trace verdict") {
		t.Fatalf("out missing no-trace verdict:\n%s", out)
	}
}
