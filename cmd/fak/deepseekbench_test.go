package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// runDSB drives the testable core with captured streams.
func runDSB(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := runDeepSeekBench(&out, &errb, args)
	return code, out.String(), errb.String()
}

// TestDeepSeekBenchRequiredFields is the FIELD LOCK: every required JSON key from the
// issue must be present on every emitted row. If a field is renamed/dropped without
// updating requiredBenchFields, this fails.
func TestDeepSeekBenchRequiredFields(t *testing.T) {
	code, out, _ := runDSB()
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) == 0 {
		t.Fatal("no JSONL rows emitted")
	}
	for i, ln := range lines {
		var m map[string]json.RawMessage
		if err := json.Unmarshal([]byte(ln), &m); err != nil {
			t.Fatalf("row %d not valid JSON: %v\n%s", i, err, ln)
		}
		for _, want := range requiredBenchFields() {
			if _, ok := m[want]; !ok {
				t.Fatalf("row %d missing required field %q:\n%s", i, want, ln)
			}
		}
		// No stray fields beyond the locked schema — the row struct is the schema.
		if len(m) != len(requiredBenchFields()) {
			t.Fatalf("row %d has %d fields, schema locks %d:\n%s", i, len(m), len(requiredBenchFields()), ln)
		}
	}
}

// TestDeepSeekBenchDryRunHonesty pins the no-key fixture invariants: every DeepSeek row
// is labelled a fixture placeholder (never a measurement), and both V4 Pro and V4 Flash
// appear side by side.
func TestDeepSeekBenchDryRunHonesty(t *testing.T) {
	rows := dryRunRows()
	sawPro, sawFlash := false, false
	for _, r := range rows {
		if r.ProviderRoute == "deepseek" {
			if r.Measurement != "dry-run-fixture" {
				t.Fatalf("dry-run row not labelled a fixture: %+v", r)
			}
			if r.SpeedProvenance == "provider-observed" {
				t.Fatalf("dry-run row claims provider-observed speed: %+v", r)
			}
		}
		switch r.ModelID {
		case "deepseek-v4-pro":
			sawPro = true
		case "deepseek-v4-flash":
			sawFlash = true
		}
	}
	if !sawPro || !sawFlash {
		t.Fatalf("scorecard must carry BOTH V4 Pro and V4 Flash (pro=%v flash=%v)", sawPro, sawFlash)
	}
}

// TestDeepSeekBenchAxisCoverage confirms every locked axis value appears in the fixture.
func TestDeepSeekBenchAxisCoverage(t *testing.T) {
	rows := dryRunRows()
	buckets, targets, modes := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, r := range rows {
		if r.ProviderRoute != "deepseek" {
			continue
		}
		buckets[r.ContextBucket] = true
		targets[r.OutputTarget] = true
		modes[r.ReasoningMode] = true
	}
	for _, b := range benchContextBuckets {
		if !buckets[b] {
			t.Fatalf("context bucket %q never appears", b)
		}
	}
	for _, tg := range benchOutputTargets {
		if !targets[tg] {
			t.Fatalf("output target %q never appears", tg)
		}
	}
	for _, md := range benchReasoningModes {
		if !modes[md] {
			t.Fatalf("reasoning mode %q never appears", md)
		}
	}
}

// TestDeepSeekBenchSpeedupRefusal is the honesty gate: the scorecard must NOT print a
// speedup for a dry-run fixture, a shape mismatch, or an unverified parity — and MUST
// print one only when shape + parity + live all line up.
func TestDeepSeekBenchSpeedupRefusal(t *testing.T) {
	base := DeepSeekBenchRow{Measurement: "live", QualityParity: "verified", E2EMillis: 100, PromptShape: "4K|short|non-thinking|stream=true", ModelID: "baseline"}
	subj := DeepSeekBenchRow{Measurement: "live", QualityParity: "verified", E2EMillis: 50, PromptShape: "4K|short|non-thinking|stream=true", ModelID: "deepseek-v4-flash"}

	// (a) dry-run fixture -> refuse.
	if line, printed := compareSpeedup(DeepSeekBenchRow{Measurement: "dry-run-fixture"}, base); printed || !strings.Contains(line, "NOT COMPARABLE") {
		t.Fatalf("dry-run must refuse a speedup, got printed=%v line=%q", printed, line)
	}
	// (b) shape mismatch -> refuse.
	mismatch := subj
	mismatch.PromptShape = "1M|8K|max|stream=false"
	if line, printed := compareSpeedup(mismatch, base); printed || !strings.Contains(line, "prompt shape differs") {
		t.Fatalf("shape mismatch must refuse, got printed=%v line=%q", printed, line)
	}
	// (c) parity not verified -> refuse.
	noparity := subj
	noparity.QualityParity = "unknown"
	if line, printed := compareSpeedup(noparity, base); printed || !strings.Contains(line, "quality parity") {
		t.Fatalf("unverified parity must refuse, got printed=%v line=%q", printed, line)
	}
	// (d) all aligned -> a labelled OBSERVED delta.
	line, printed := compareSpeedup(subj, base)
	if !printed || !strings.Contains(line, "OBSERVED provider speed") || !strings.Contains(line, "not a fak-authored saving") {
		t.Fatalf("aligned rows must print an OBSERVED delta, got printed=%v line=%q", printed, line)
	}
}

// TestDeepSeekBenchLiveGate confirms the live arm refuses BEFORE any network when the
// key or the spend flag is missing.
func TestDeepSeekBenchLiveGate(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")
	if code, _, errb := runDSB("--live", "--spend"); code != 2 || !strings.Contains(errb, "DEEPSEEK_API_KEY") {
		t.Fatalf("missing key must refuse (code=2), got code=%d err=%q", code, errb)
	}
	t.Setenv("DEEPSEEK_API_KEY", "sk-test-not-used")
	if code, _, errb := runDSB("--live"); code != 2 || !strings.Contains(errb, "--spend") {
		t.Fatalf("missing --spend must refuse (code=2), got code=%d err=%q", code, errb)
	}
}
