package main

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func runPerfRSI(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var out, err bytes.Buffer
	c := runPerformanceRSIScorecard(&out, &err, args)
	return c, out.String(), err.String()
}
func fixturePath() string {
	return filepath.Join("..", "..", "internal", "perfrsiscore", "testdata", "complete.json")
}

func TestPerformanceRSIRenderers(t *testing.T) {
	for _, tc := range [][]string{{"--input", fixturePath()}, {"--input", fixturePath(), "--json"}, {"--input", fixturePath(), "--markdown"}} {
		code, out, err := runPerfRSI(t, tc...)
		if code != 0 {
			t.Fatalf("%v code=%d err=%s", tc, code, err)
		}
		if !strings.Contains(out, "cycle_time") {
			t.Fatalf("%v missing dimension", tc)
		}
	}
}
func TestPerformanceRSIComparison(t *testing.T) {
	code, out, err := runPerfRSI(t, "--input", fixturePath(), "--json")
	if code != 0 {
		t.Fatal(err)
	}
	var p map[string]any
	if json.Unmarshal([]byte(out), &p) != nil {
		t.Fatal("bad json")
	}
	p["snapshot"] = "prior"
	b, _ := json.Marshal(p)
	path := filepath.Join(t.TempDir(), "prior.json")
	if os.WriteFile(path, b, 0600) != nil {
		t.Fatal("write")
	}
	code, out, err = runPerfRSI(t, "--input", fixturePath(), "--prior", path, "--json")
	if code != 0 || !strings.Contains(out, `"prior_snapshot": "prior"`) {
		t.Fatalf("code=%d out=%s err=%s", code, out, err)
	}
}
func TestPerformanceRSIRequiresInput(t *testing.T) {
	code, _, _ := runPerfRSI(t)
	if code != 2 {
		t.Fatal(code)
	}
}

func TestPerformanceRSICycleWitness(t *testing.T) {
	witness := filepath.Join("..", "..", "docs", "_witnesses", "issue-9780-performance-rsi-cycle.json")
	code, out, err := runPerfRSI(t, "--input", witness, "--json")
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, err)
	}
	var report struct {
		Dimensions []struct {
			ID      string   `json:"id"`
			Status  string   `json:"status"`
			Current *float64 `json:"current"`
		} `json:"dimensions"`
	}
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{
		"cycle_time": false, "evaluation_latency": false,
		"experiment_throughput": false, "automation_coverage": false,
	}
	for _, d := range report.Dimensions {
		if _, ok := want[d.ID]; ok {
			if d.Status == "UNKNOWN" || d.Current == nil {
				t.Errorf("%s remained UNKNOWN", d.ID)
			}
			want[d.ID] = true
		}
	}
	for id, found := range want {
		if !found {
			t.Errorf("missing derived dimension %s", id)
		}
	}
}

func TestPerformanceRSIImprovementReceiptAcceptance(t *testing.T) {
	witness := filepath.Join("..", "..", "docs", "_witnesses", "issue-9781-performance-rsi-improvement.json")
	code, out, errText := runPerfRSI(t, "--input", witness, "--json")
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errText)
	}
	for _, id := range []string{"improvement_yield", "receipt_coverage", "quality_gate_coverage", "attribution_quality"} {
		if !strings.Contains(out, `"id": "`+id+`"`) {
			t.Errorf("missing %s", id)
		}
	}
	if strings.Count(out, `"status": "UNKNOWN"`) > 8 {
		t.Fatalf("receipt dimensions remained unknown: %s", out)
	}
}

func TestPerformanceRSIImprovementReceiptRefusals(t *testing.T) {
	witness := filepath.Join("..", "..", "docs", "_witnesses", "issue-9781-performance-rsi-improvement.json")
	original, err := os.ReadFile(witness)
	if err != nil {
		t.Fatal(err)
	}
	var base map[string]any
	if err := json.Unmarshal(original, &base); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(map[string]any)
	}{
		{"schema", func(r map[string]any) { r["schema"] = "unsupported" }},
		{"units", func(r map[string]any) { r["baseline"].(map[string]any)["unit"] = "seconds" }},
		{"quality", func(r map[string]any) { delete(r["quality"].(map[string]any), "parity") }},
		{"envelope", func(r map[string]any) { r["candidate_envelope"].(map[string]any)["batch_size"] = 2 }},
		{"overhead", func(r map[string]any) { r["net_true_gain"].(map[string]any)["includes_overhead"] = false }},
		{"causal", func(r map[string]any) { delete(r["causal"].(map[string]any), "isolates_change") }},
		{"engine", func(r map[string]any) { r["engine"] = "llama.cpp" }},
		{"strict", func(r map[string]any) { r["unexpected"] = true }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var doc map[string]any
			b, _ := json.Marshal(base)
			_ = json.Unmarshal(b, &doc)
			tc.edit(doc["improvement"].(map[string]any))
			b, _ = json.Marshal(doc)
			path := filepath.Join(t.TempDir(), "receipt.json")
			if err := os.WriteFile(path, b, 0600); err != nil {
				t.Fatal(err)
			}
			code, _, errText := runPerfRSI(t, "--input", path, "--json")
			if code == 0 || strings.TrimSpace(errText) == "" {
				t.Fatalf("expected refusal: code=%d err=%q", code, errText)
			}
		})
	}
}

func TestPerformanceRSIProvenanceAcceptance(t *testing.T) {
	witness := filepath.Join("..", "..", "docs", "_witnesses", "issue-9782-performance-rsi-provenance.json")
	code, out, errText := runPerfRSI(t, "--input", witness, "--json")
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errText)
	}
	var report struct {
		Dimensions []struct {
			ID           string   `json:"id"`
			Current      *float64 `json:"current"`
			EvidenceKind string   `json:"evidence_kind"`
		} `json:"dimensions"`
	}
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatal(err)
	}
	want := map[string]float64{
		"discovery_freshness": 16.051666666666666,
		"adaptation_speed":    0.8580555555555556,
		"reuse_ratio":         100,
		"production_transfer": 82.59222222222222,
	}
	found := 0
	for _, d := range report.Dimensions {
		wantCurrent, ok := want[d.ID]
		if !ok {
			continue
		}
		found++
		if d.Current == nil || math.Abs(*d.Current-wantCurrent) > 1e-9 {
			t.Errorf("%s current=%v want %.12g", d.ID, d.Current, wantCurrent)
		}
		if d.EvidenceKind != "research_transfer_receipt" {
			t.Errorf("%s evidence_kind=%q", d.ID, d.EvidenceKind)
		}
	}
	if found != len(want) {
		t.Fatalf("found %d provenance-derived dimensions, want %d: %s", found, len(want), out)
	}
}

func TestPerformanceRSIProvenanceRefusals(t *testing.T) {
	witness := filepath.Join("..", "..", "docs", "_witnesses", "issue-9782-performance-rsi-provenance.json")
	original, err := os.ReadFile(witness)
	if err != nil {
		t.Fatal(err)
	}
	var base map[string]any
	if err := json.Unmarshal(original, &base); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(map[string]any)
	}{
		{"schema", func(r map[string]any) { r["schema"] = "unsupported" }},
		{"revision", func(r map[string]any) { r["source"].(map[string]any)["revision"] = "main" }},
		{"timeline", func(r map[string]any) { r["discovery_at"] = "2026-08-24T00:00:00Z" }},
		{"explicit", func(r map[string]any) { r["adaptation_start_explicit"] = false }},
		{"experiment", func(r map[string]any) { r["experiment"].(map[string]any)["linked"] = false }},
		{"classification", func(r map[string]any) { r["reuse"].(map[string]any)["classification"] = "invented_here" }},
		{"counts", func(r map[string]any) { r["reuse"].(map[string]any)["reused_mechanisms"] = -1 }},
		{"commit", func(r map[string]any) { r["production"].(map[string]any)["commit_sha"] = "5e0db65c5" }},
		{"module", func(r map[string]any) {
			r["production"].(map[string]any)["module_at_rev"] = "internal/qwen38quantrun@r17+gdeadbee"
		}},
		{"engine", func(r map[string]any) { r["production"].(map[string]any)["engine"] = "llama.cpp" }},
		{"unit", func(r map[string]any) { r["unit"] = "days" }},
		{"strict", func(r map[string]any) { r["unexpected"] = true }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var doc map[string]any
			b, _ := json.Marshal(base)
			_ = json.Unmarshal(b, &doc)
			tc.edit(doc["provenance"].(map[string]any))
			b, _ = json.Marshal(doc)
			path := filepath.Join(t.TempDir(), "receipt.json")
			if err := os.WriteFile(path, b, 0600); err != nil {
				t.Fatal(err)
			}
			code, _, errText := runPerfRSI(t, "--input", path, "--json")
			if code == 0 || strings.TrimSpace(errText) == "" {
				t.Fatalf("expected refusal: code=%d err=%q", code, errText)
			}
		})
	}
}
