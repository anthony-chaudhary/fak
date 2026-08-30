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

func performanceRSIDogfoodPath() string {
	return filepath.Join("..", "..", "docs", "_witnesses", "issue-9768-performance-rsi-dogfood", "input.json")
}

func performanceRSICommittedReceiptPaths() []string {
	base := filepath.Join("..", "..", "docs", "_witnesses")
	return []string{
		filepath.Join(base, "issue-9780-performance-rsi-cycle.json"),
		filepath.Join(base, "issue-9781-performance-rsi-improvement.json"),
		filepath.Join(base, "issue-9782-performance-rsi-provenance.json"),
		filepath.Join(base, "issue-9783-performance-rsi-learning.json"),
	}
}

func performanceRSILearningTestReceipt(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile(fixturePath())
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	doc["learning"] = map[string]any{
		"schema": "fak-performance-rsi-learning/1",
		"rows": []any{
			map[string]any{"cycle_id": "c1", "hypothesis_id": "h1", "recurrence_key": "parser", "predicted_improvement_percent": 20.0, "confidence_percent": 50.0, "observed_improvement_percent": 10.0, "learning_id": "l1", "learning_recorded": true, "learning_reused": false, "prior_learning_id": "", "repeated_failure": false, "cycle_time_hours": 10.0, "engine": "fak-native", "artifact": "synthetic-test-c1"},
			map[string]any{"cycle_id": "c2", "hypothesis_id": "h2", "recurrence_key": "parser", "predicted_improvement_percent": 12.0, "confidence_percent": 30.0, "observed_improvement_percent": 0.0, "learning_id": "", "learning_recorded": false, "learning_reused": true, "prior_learning_id": "l1", "repeated_failure": false, "cycle_time_hours": 8.0, "engine": "fak-native", "artifact": "synthetic-test-c2"},
			map[string]any{"cycle_id": "c3", "hypothesis_id": "h3", "recurrence_key": "parser", "predicted_improvement_percent": 8.0, "confidence_percent": 20.0, "observed_improvement_percent": 0.0, "learning_id": "", "learning_recorded": false, "learning_reused": true, "prior_learning_id": "l1", "repeated_failure": true, "cycle_time_hours": 6.0, "engine": "fak-native", "artifact": "synthetic-test-c3"},
		},
	}
	for _, raw := range doc["dimensions"].([]any) {
		d := raw.(map[string]any)
		switch d["id"] {
		case "hypothesis_calibration", "learning_retention", "compounding_rate":
			d["direction"] = "higher"
			d["unit"] = "percent"
		}
	}
	b, err = json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func writePerformanceRSILearningTestReceipt(t *testing.T, b []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestPerformanceRSILearningAcceptance(t *testing.T) {
	path := writePerformanceRSILearningTestReceipt(t, performanceRSILearningTestReceipt(t))
	code, out, errText := runPerfRSI(t, "--input", path, "--json")
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errText)
	}
	var report struct {
		Dimensions []struct {
			ID      string   `json:"id"`
			Current *float64 `json:"current"`
		} `json:"dimensions"`
	}
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatal(err)
	}
	want := map[string]float64{"hypothesis_calibration": 89.8, "learning_retention": 100, "compounding_rate": 40}
	for _, d := range report.Dimensions {
		if value, ok := want[d.ID]; ok {
			if d.Current == nil || math.Abs(*d.Current-value) > 1e-9 {
				t.Errorf("%s=%v want %.1f", d.ID, d.Current, value)
			}
			delete(want, d.ID)
		}
	}
	if len(want) != 0 {
		t.Fatalf("missing derived dimensions: %v", want)
	}
}

func TestPerformanceRSICommittedLearningReceiptSourceLabels(t *testing.T) {
	witness := filepath.Join("..", "..", "docs", "_witnesses", "issue-9783-performance-rsi-learning.json")
	code, out, errText := runPerfRSI(t, "--input", witness, "--json")
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errText)
	}
	var report struct {
		Dimensions []struct {
			ID           string `json:"id"`
			Source       string `json:"source"`
			EvidenceKind string `json:"evidence_kind"`
			Engine       string `json:"engine"`
		} `json:"dimensions"`
	}
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{
		"hypothesis_calibration": false,
		"learning_retention":     false,
		"compounding_rate":       false,
	}
	for _, d := range report.Dimensions {
		if _, ok := want[d.ID]; !ok {
			continue
		}
		if d.Source != "learning:fak-performance-rsi-learning/1" ||
			d.EvidenceKind != "performance_rsi_learning_receipt" ||
			d.Engine != "fak-native" {
			t.Errorf("%s provenance=%+v", d.ID, d)
		}
		want[d.ID] = true
	}
	for id, found := range want {
		if !found {
			t.Errorf("missing learning-derived dimension %s", id)
		}
	}
}

func TestPerformanceRSILearningRefusals(t *testing.T) {
	original := performanceRSILearningTestReceipt(t)
	var base map[string]any
	if err := json.Unmarshal(original, &base); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(map[string]any)
	}{
		{"insufficient history", func(doc map[string]any) { l := doc["learning"].(map[string]any); l["rows"] = l["rows"].([]any)[:1] }},
		{"false compounding", func(doc map[string]any) {
			doc["learning"].(map[string]any)["rows"].([]any)[2].(map[string]any)["cycle_time_hours"] = float64(10)
		}},
		{"negative compounding", func(doc map[string]any) {
			doc["learning"].(map[string]any)["rows"].([]any)[2].(map[string]any)["cycle_time_hours"] = float64(11)
		}},
		{"strict", func(doc map[string]any) { doc["learning"].(map[string]any)["unexpected"] = true }},
		{"strict top level", func(doc map[string]any) { doc["unexpected_receipt_field"] = true }},
		{"engine", func(doc map[string]any) {
			doc["learning"].(map[string]any)["rows"].([]any)[1].(map[string]any)["engine"] = "fak-native/qwen"
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, _ := json.Marshal(base)
			var doc map[string]any
			_ = json.Unmarshal(b, &doc)
			tc.edit(doc)
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

func TestPerformanceRSIRenderers(t *testing.T) {
	for _, tc := range [][]string{{"--input", fixturePath()}, {"--input", fixturePath(), "--json"}, {"--input", fixturePath(), "--markdown"}} {
		code, out, err := runPerfRSI(t, tc...)
		if code != 0 {
			t.Fatalf("%v code=%d err=%s", tc, code, err)
		}
		if !strings.Contains(out, "cycle_time") {
			t.Fatalf("%v missing dimension", tc)
		}
		if !strings.Contains(out, "loop health") && !strings.Contains(out, "loop_health") && !strings.Contains(out, "Loop-health grade") {
			t.Fatalf("%v missing loop-health grade", tc)
		}
	}
}

func TestPerformanceRSIDogfoodJSONReportsPoorNamedLoopHealthDebt(t *testing.T) {
	code, out, errText := runPerfRSI(t, "--input", performanceRSIDogfoodPath(), "--json")
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errText)
	}
	var report struct {
		LoopHealth struct {
			Score          float64 `json:"score"`
			Grade          string  `json:"grade"`
			Clean          bool    `json:"clean"`
			Interpretation string  `json:"interpretation"`
		} `json:"loop_health"`
		DebtSummary struct {
			PerformanceRSIDebt int `json:"performance_rsi_debt"`
			Total              int `json:"total"`
			DimensionsMeasured int `json:"dimensions_measured"`
			DimensionsTotal    int `json:"dimensions_total"`
			Behind             int `json:"behind"`
			Unknown            int `json:"unknown"`
			Evidence           []struct {
				Dimension       string   `json:"dimension"`
				Status          string   `json:"status"`
				NormalizedRatio *float64 `json:"normalized_ratio"`
				Source          string   `json:"source"`
			} `json:"evidence"`
		} `json:"debt_summary"`
	}
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatal(err)
	}
	if report.LoopHealth.Score != 62.6 || report.LoopHealth.Grade != "D" || report.LoopHealth.Clean ||
		!strings.Contains(report.LoopHealth.Interpretation, "does not prove") {
		t.Fatalf("loop health=%+v", report.LoopHealth)
	}
	if report.DebtSummary.PerformanceRSIDebt != 9 || report.DebtSummary.Total != 9 ||
		report.DebtSummary.DimensionsMeasured != 15 || report.DebtSummary.DimensionsTotal != 16 ||
		report.DebtSummary.Behind != 8 || report.DebtSummary.Unknown != 1 ||
		len(report.DebtSummary.Evidence) != report.DebtSummary.PerformanceRSIDebt {
		t.Fatalf("debt summary=%+v", report.DebtSummary)
	}
	if got := report.DebtSummary.Evidence[0]; got.Dimension != "cycle_time" ||
		got.Status != "BEHIND" || got.NormalizedRatio == nil || *got.NormalizedRatio != 0.01 ||
		got.Source != "cycle:fak-performance-rsi-cycle/1" {
		t.Fatalf("first named evidence=%+v", got)
	}
	if got := report.DebtSummary.Evidence[6]; got.Dimension != "hardware_utilization" ||
		got.Status != "UNKNOWN" || got.NormalizedRatio != nil || got.Source != "fixture/hardware" {
		t.Fatalf("UNKNOWN named evidence=%+v", got)
	}
}

func TestPerformanceRSIComposeFeedsExistingScorecard(t *testing.T) {
	args := append([]string{"compose", "--snapshot", "issue-9823-cli-regression"}, performanceRSICommittedReceiptPaths()...)
	code, composed, errText := runPerfRSI(t, args...)
	if code != 0 {
		t.Fatalf("compose code=%d err=%s", code, errText)
	}
	var evidence struct {
		Schema   string `json:"schema"`
		Snapshot string `json:"snapshot"`
	}
	if err := json.Unmarshal([]byte(composed), &evidence); err != nil {
		t.Fatal(err)
	}
	if evidence.Schema != "fak-performance-rsi-evidence/1" || evidence.Snapshot != "issue-9823-cli-regression" {
		t.Fatalf("composed evidence header=%+v", evidence)
	}

	input := filepath.Join(t.TempDir(), "composed.json")
	if err := os.WriteFile(input, []byte(composed), 0o600); err != nil {
		t.Fatal(err)
	}
	code, out, errText := runPerfRSI(t, "--input", input, "--json")
	if code != 0 {
		t.Fatalf("scorecard code=%d err=%s", code, errText)
	}
	var report struct {
		UnknownDebt int `json:"unknown_debt"`
		Dimensions  []struct {
			ID     string `json:"id"`
			Status string `json:"status"`
		} `json:"dimensions"`
	}
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatal(err)
	}
	if report.UnknownDebt != 1 {
		t.Fatalf("UNKNOWN debt=%d, want 1", report.UnknownDebt)
	}
	var unknown []string
	for _, d := range report.Dimensions {
		if d.Status == "UNKNOWN" {
			unknown = append(unknown, d.ID)
		}
	}
	if strings.Join(unknown, ",") != "hardware_utilization" {
		t.Fatalf("UNKNOWN dimensions=%v, want only hardware_utilization", unknown)
	}

	reversed := performanceRSICommittedReceiptPaths()
	for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
		reversed[i], reversed[j] = reversed[j], reversed[i]
	}
	reverseArgs := append([]string{"compose", "--snapshot", "issue-9823-cli-regression"}, reversed...)
	code, reordered, errText := runPerfRSI(t, reverseArgs...)
	if code != 0 || reordered != composed {
		t.Fatalf("reordered compose code=%d err=%s deterministic=%v", code, errText, reordered == composed)
	}
}

func TestPerformanceRSIComposeRequiresSnapshotAndReceipts(t *testing.T) {
	for _, args := range [][]string{
		{"compose"},
		{"compose", "--snapshot", "missing-receipts"},
	} {
		code, _, errText := runPerfRSI(t, args...)
		if code != 2 || strings.TrimSpace(errText) == "" {
			t.Fatalf("%v code=%d err=%q", args, code, errText)
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

func performanceRSIHardwareTestReceipt(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile(fixturePath())
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	doc["hardware"] = map[string]any{
		"schema": "fak-performance-rsi-hardware/1",
		"runs": []any{
			map[string]any{
				"enqueued_at": "2026-08-28T10:00:00Z", "started_at": "2026-08-28T10:10:00Z",
				"ended_at": "2026-08-28T11:10:00Z", "requested_device_class": "cuda-l4",
				"active_utilization": 50.0, "utilization_unit": "percent", "workload_id": "workload-a",
				"engine": "fak-native",
			},
			map[string]any{
				"enqueued_at": "2026-08-28T12:00:00Z", "started_at": "2026-08-28T12:20:00Z",
				"ended_at": "2026-08-28T15:20:00Z", "requested_device_class": "cuda-h100",
				"active_utilization": 90.0, "utilization_unit": "percent", "workload_id": "workload-b",
				"engine": "fak-native",
			},
		},
	}
	for _, raw := range doc["dimensions"].([]any) {
		d := raw.(map[string]any)
		if d["id"] == "hardware_utilization" {
			d["direction"] = "higher"
			d["unit"] = "percent"
		}
	}
	b, err = json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestPerformanceRSIHardwareAcceptance(t *testing.T) {
	path := writePerformanceRSILearningTestReceipt(t, performanceRSIHardwareTestReceipt(t))
	code, out, errText := runPerfRSI(t, "--input", path, "--json")
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errText)
	}
	var report struct {
		Dimensions []struct {
			ID           string   `json:"id"`
			Source       string   `json:"source"`
			EvidenceKind string   `json:"evidence_kind"`
			Engine       string   `json:"engine"`
			Current      *float64 `json:"current"`
		} `json:"dimensions"`
	}
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatal(err)
	}
	for _, d := range report.Dimensions {
		if d.ID != "hardware_utilization" {
			continue
		}
		if d.Current == nil || *d.Current != 80 {
			t.Fatalf("hardware_utilization=%v, want 80", d.Current)
		}
		if d.Source != "hardware:fak-performance-rsi-hardware/1;queue_delay_seconds_total=1800;queue_delay_seconds_mean=900" ||
			d.EvidenceKind != "hardware_utilization_receipt" || d.Engine != "fak-native" {
			t.Fatalf("hardware metadata: %+v", d)
		}
		return
	}
	t.Fatal("missing hardware_utilization")
}

func TestPerformanceRSIHardwareAllowsBenignLocalNoGPUDeviceClassText(t *testing.T) {
	var doc map[string]any
	if err := json.Unmarshal(performanceRSIHardwareTestReceipt(t), &doc); err != nil {
		t.Fatal(err)
	}
	doc["hardware"].(map[string]any)["runs"].([]any)[0].(map[string]any)["requested_device_class"] =
		"simulator-local-no-gpu-compatible"
	b, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	path := writePerformanceRSILearningTestReceipt(t, b)
	code, out, errText := runPerfRSI(t, "--input", path, "--json")
	if code != 0 || strings.TrimSpace(out) == "" || errText != "" {
		t.Fatalf("benign device class rejected: code=%d out=%q err=%q", code, out, errText)
	}
}

func TestPerformanceRSIHardwareRejectsTypedLocalNoGPUEvidencePrecisely(t *testing.T) {
	var doc map[string]any
	if err := json.Unmarshal(performanceRSIHardwareTestReceipt(t), &doc); err != nil {
		t.Fatal(err)
	}
	doc["hardware"].(map[string]any)["runs"].([]any)[0].(map[string]any)["terminal_evidence"] =
		map[string]any{"type": "local-no-gpu"}
	b, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	path := writePerformanceRSILearningTestReceipt(t, b)
	code, out, errText := runPerfRSI(t, "--input", path, "--json")
	wantErr := "fak performance-rsi-scorecard: hardware run 0 terminal_evidence type " +
		"\"local-no-gpu\": local-no-GPU is a terminal blocker, not a hardware utilization measurement\n"
	if code == 0 || out != "" || errText != wantErr {
		t.Fatalf("typed terminal evidence result: code=%d out=%q err=%q, want err=%q", code, out, errText, wantErr)
	}
}

func TestPerformanceRSIHardwareRefusals(t *testing.T) {
	original := performanceRSIHardwareTestReceipt(t)
	var base map[string]any
	if err := json.Unmarshal(original, &base); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(map[string]any)
	}{
		{"schema", func(r map[string]any) { r["schema"] = "unsupported" }},
		{"missing timestamp", func(r map[string]any) { delete(r["runs"].([]any)[0].(map[string]any), "started_at") }},
		{"non UTC", func(r map[string]any) {
			r["runs"].([]any)[0].(map[string]any)["started_at"] = "2026-08-28T10:10:00-07:00"
		}},
		{"order", func(r map[string]any) { r["runs"].([]any)[0].(map[string]any)["ended_at"] = "2026-08-28T10:10:00Z" }},
		{"unit", func(r map[string]any) { r["runs"].([]any)[0].(map[string]any)["utilization_unit"] = "%" }},
		{"device", func(r map[string]any) { r["runs"].([]any)[0].(map[string]any)["requested_device_class"] = "" }},
		{"workload", func(r map[string]any) { r["runs"].([]any)[0].(map[string]any)["workload_id"] = "" }},
		{"engine", func(r map[string]any) { r["runs"].([]any)[0].(map[string]any)["engine"] = "llama.cpp" }},
		{"range", func(r map[string]any) { r["runs"].([]any)[0].(map[string]any)["active_utilization"] = 101.0 }},
		{"invalid second run", func(r map[string]any) { r["runs"].([]any)[1].(map[string]any)["active_utilization"] = -1.0 }},
		{"private host", func(r map[string]any) { r["runs"].([]any)[0].(map[string]any)["host"] = "private-node" }},
		{"private path", func(r map[string]any) { r["runs"].([]any)[0].(map[string]any)["path"] = "/private/lab" }},
		{"typed local no GPU", func(r map[string]any) {
			r["runs"].([]any)[0].(map[string]any)["terminal_evidence"] = map[string]any{"type": "local-no-gpu"}
		}},
		{"unsupported terminal evidence", func(r map[string]any) {
			r["runs"].([]any)[0].(map[string]any)["terminal_evidence"] = map[string]any{"type": "gpu-driver-missing"}
		}},
		{"unknown terminal evidence field", func(r map[string]any) {
			r["runs"].([]any)[0].(map[string]any)["terminal_evidence"] =
				map[string]any{"type": "local-no-gpu", "host": "private-node"}
		}},
		{"strict receipt", func(r map[string]any) { r["private_node"] = "private-node" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			copyJSON, _ := json.Marshal(base)
			var doc map[string]any
			_ = json.Unmarshal(copyJSON, &doc)
			tc.edit(doc["hardware"].(map[string]any))
			b, _ := json.Marshal(doc)
			path := writePerformanceRSILearningTestReceipt(t, b)
			code, out, errText := runPerfRSI(t, "--input", path, "--json")
			if code == 0 || out != "" || strings.TrimSpace(errText) == "" {
				t.Fatalf("expected refusal without report: code=%d out=%q err=%q", code, out, errText)
			}
		})
	}
}
