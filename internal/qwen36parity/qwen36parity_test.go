package qwen36parity

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func witness(fakIDs []int) map[string]any {
	return witnessWith(fakIDs, nil, nil, "Darwin 24.5.0 arm64 M3Pro-Metal", false, 0, 1.84)
}

func witnessWith(fakIDs, oracle []int, div *int, host string, metalPass bool, prefill, decode float64) map[string]any {
	if oracle == nil {
		oracle = append([]int(nil), OracleIDs...)
	}
	d := FirstDivergence(fakIDs, oracle)
	if div != nil {
		d = *div
	}
	return map[string]any{
		"host":                   host,
		"commit":                 "deadbeef",
		"captured_at":            "20260628T120000Z",
		"prompt_ids":             []any{248045, 8678, 198},
		"llamacpp_token_ids":     intAny(oracle),
		"fak_token_ids":          intAny(fakIDs),
		"first_divergence_index": float64(d),
		"correctness_parity":     d == -1,
		"metal_gate_pass":        metalPass,
		"fak_prefill_tok_s":      prefill,
		"fak_decode_tok_s":       decode,
		"bar_prefill_tok_s":      BarPrefill,
		"bar_decode_tok_s":       BarDecode,
	}
}

func TestQwen36ParityGate(t *testing.T) {
	r := GradeWitness(witness(KnownFakIDs), "<inline>", 0)
	if r.Correctness.Verdict != "KNOWN_DRIFT" || r.Correctness.Regression || !r.Passed {
		t.Fatalf("known drift report = %+v", r)
	}
	r = GradeWitness(witness(OracleIDs), "<inline>", 0)
	if r.Correctness.Verdict != "PARITY" || *r.Correctness.FirstDivergenceIndex != -1 || !strings.Contains(r.Correctness.Note, "CLOSED") || !r.Passed {
		t.Fatalf("parity report = %+v", r)
	}
	r = GradeWitness(witness([]int{248068, 999, 8160}), "<inline>", 0)
	if r.Correctness.Verdict != "REGRESSION" || !r.Correctness.Regression || r.Passed {
		t.Fatalf("regression report = %+v", r)
	}
	r = GradeWitness(witnessWith([]int{248068, 198, 90700, 111}, []int{248068, 198, 90700, 222}, nil, "Darwin 24.5.0 arm64 M3Pro-Metal", false, 0, 1.84), "<inline>", 0)
	if r.Correctness.Verdict != "PROGRESS" || r.Correctness.Regression || !r.Passed {
		t.Fatalf("progress report = %+v", r)
	}
	r = GradeWitness(witness([]int{248068, 198, 12345}), "<inline>", 0)
	if r.Correctness.Verdict != "DRIFT_CHANGED" || r.Correctness.Regression || !r.Passed {
		t.Fatalf("changed drift report = %+v", r)
	}
	r = GradeWitness(witnessWith([]int{248068, 198, 8160}, []int{248068, 198, 99999}, ptr(2), "Darwin 24.5.0 arm64 M3Pro-Metal", false, 0, 1.84), "<inline>", 0)
	if r.Correctness.Verdict != "ORACLE_DRIFT" || r.Passed {
		t.Fatalf("oracle drift report = %+v", r)
	}
	w := witness(KnownFakIDs)
	w["first_divergence_index"] = float64(0)
	if r := GradeWitness(w, "<inline>", 0); r.Correctness.Verdict != "MALFORMED" || r.Passed {
		t.Fatalf("malformed index report = %+v", r)
	}
	w = witness(KnownFakIDs)
	delete(w, "fak_token_ids")
	if r := GradeWitness(w, "<inline>", 0); r.Correctness.Verdict != "MALFORMED" || r.Passed {
		t.Fatalf("missing ids report = %+v", r)
	}
	for _, host := range []string{"my-macbook.tail9abc.ts.net", "192.168.1.42", "user@studio.local"} {
		w := witness(KnownFakIDs)
		w["host"] = host
		r := GradeWitness(w, "<inline>", 0)
		if clean, _ := r.Scrub["clean"].(bool); clean || r.Passed {
			t.Fatalf("leak not caught for %s: %+v", host, r)
		}
	}
	for _, host := range []string{"Darwin 24.5.0 arm64 M3Pro-Metal", "node-macos-a.local"} {
		r := GradeWitness(witnessWith(KnownFakIDs, nil, nil, host, false, 0, 1.84), "<inline>", 0)
		if clean, _ := r.Scrub["clean"].(bool); !clean || !r.Passed {
			t.Fatalf("scrubbed host not allowed for %s: %+v", host, r)
		}
	}
	r = GradeWitness(witnessWith(KnownFakIDs, nil, nil, "Darwin 24.5.0 arm64 M3Pro-Metal", false, 0, 1.2), "<inline>", 0)
	if r.Speed.Gated || r.Speed.DecodeRatio == 0 || !r.Passed {
		t.Fatalf("speed should be report-only by default: %+v", r)
	}
	r = GradeWitness(witnessWith(KnownFakIDs, nil, nil, "Darwin 24.5.0 arm64 M3Pro-Metal", false, 0, 1.2), "<inline>", 0.5)
	if !r.Speed.Gated || r.Passed || len(r.Speed.Failures) == 0 {
		t.Fatalf("speed gate did not fail: %+v", r)
	}
	if got := GradeWitness(witness(KnownFakIDs), "<inline>", 0).Issues["metal_gate"]; got[0] != 71 || got[1] != 1458 {
		t.Fatalf("issue links = %v", got)
	}
	md := RenderMarkdown(GradeWitness(witness(KnownFakIDs), "<inline>", 0))
	if !strings.Contains(md, "Metal hybrid-prefill gate (#71)") || strings.Contains(md, "#116") {
		t.Fatalf("bad markdown: %s", md)
	}
	if NoWitnessReport(false).Passed != true || NoWitnessReport(true).Passed != false {
		t.Fatalf("bad no-witness contract")
	}
}

func TestFindLatestAndLoad(t *testing.T) {
	dir := t.TempDir()
	old := filepath.Join(dir, "qwen36-mac-parity-gate-20260628T100000Z.json")
	newest := filepath.Join(dir, "qwen36-mac-parity-gate-20260628T180000Z.json")
	if err := os.WriteFile(old, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(newest, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "unrelated.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, ok := FindLatestWitness(dir); !ok || got != newest {
		t.Fatalf("latest = %q %v, want %q", got, ok, newest)
	}
	p := filepath.Join(t.TempDir(), "qwen36-mac-parity-gate-20260628T120000Z.json")
	data, _ := json.Marshal(witness(KnownFakIDs))
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := LoadAndGrade("/", p, false, 0)
	if err != nil || !report.Passed {
		t.Fatalf("LoadAndGrade = %+v, %v", report, err)
	}
}

func intAny(v []int) []any {
	out := make([]any, len(v))
	for i, n := range v {
		out[i] = float64(n)
	}
	return out
}

func ptr[T any](v T) *T { return &v }
