package servereceipt

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestKimiFixtureVerdicts pins the two verdict-level facts the fixture exists
// for, under BOTH readings of the profile's "96": 96 GiB (2^30 B) and 96
// decimal GB (10^9 B, expressed in the profile's GiB field). The verdicts
// must hold under either reading.
func TestKimiFixtureVerdicts(t *testing.T) {
	facts := loadKimiFacts(t)
	cfg := func(tp, pp int) Config {
		return Config{TP: tp, PP: pp, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2}
	}
	readings := []struct {
		name string
		gpu  GPUProfile
	}{
		{"vram_gb_as_gib", testGPU(t, "rtx-pro-6000-blackwell-96gb")},
		{"vram_gb_as_decimal_gb", GPUProfile{Key: "rtx-pro-6000-96gb-decimal", VRAMGB: 96e9 / float64(GibBytes), SM: "12.0", FP4Native: true}},
	}
	for _, reading := range readings {
		t.Run("tp16_infeasible/"+reading.name, func(t *testing.T) {
			receipt, err := Plan(facts, reading.gpu, cfg(16, 1))
			if err != nil {
				t.Fatalf("Plan: %v", err)
			}
			if receipt.Verdict != VerdictInfeasible {
				t.Fatalf("verdict = %s (%s), want INFEASIBLE", receipt.Verdict, receipt.Reason)
			}
			if !strings.Contains(receipt.Reason, "usable_per_gpu ≤ 0 after weights") {
				t.Fatalf("reason = %q, want the weights-exceed-budget cause", receipt.Reason)
			}
		})
		t.Run("tp8_pp3_ok/"+reading.name, func(t *testing.T) {
			receipt, err := Plan(facts, reading.gpu, cfg(8, 3))
			if err != nil {
				t.Fatalf("Plan: %v", err)
			}
			if receipt.Verdict != VerdictOK {
				t.Fatalf("verdict = %s (%s), want OK", receipt.Verdict, receipt.Reason)
			}
			if receipt.Arithmetic.MaxCachedTokens <= 0 {
				t.Fatalf("max_cached_tokens = %d, want > 0", receipt.Arithmetic.MaxCachedTokens)
			}
		})
	}

	// The tp8 PCIe-wide-TP warning fires on the NVLink-less card (non-fatal).
	receipt, err := Plan(facts, testGPU(t, "rtx-pro-6000-blackwell-96gb"), cfg(8, 3))
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	found := false
	for _, w := range receipt.Warnings {
		if strings.Contains(w, "wide TP all-reduce over PCIe") {
			found = true
		}
	}
	if !found {
		t.Fatalf("tp8 warnings %v: want the PCIe wide-TP warning", receipt.Warnings)
	}
}

// TestValidateRefusesMutation: validate recomputes every number from the
// embedded facts+config; any perturbed value must fail with its field named.
func TestValidateRefusesMutation(t *testing.T) {
	facts := loadTinyFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")
	receipt, err := Plan(facts, gpu, Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}

	// Unperturbed receipt reconciles (JSON round-trip, the CLI's storage path).
	raw, err := RenderJSON(receipt)
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}
	var clean Receipt
	if err := json.Unmarshal(raw, &clean); err != nil {
		t.Fatalf("round-trip unmarshal: %v", err)
	}
	if doc := Validate(clean); !doc.OK {
		t.Fatalf("clean receipt failed validate: %+v", doc.Diffs)
	}

	ada, err := Plan(facts, testGPU(t, "rtx-6000-ada-48gb"), Config{TP: 8, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan ada: %v", err)
	}

	tests := []struct {
		name       string
		base       func() Receipt
		mutate     func(*Receipt)
		wantFields []string // every named field must appear in the diff
	}{
		{
			name:       "perturbed max_cached_tokens",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Arithmetic.MaxCachedTokens++ },
			wantFields: []string{"arithmetic.max_cached_tokens"},
		},
		{
			name:       "perturbed weights_bytes",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Facts.WeightsBytes *= 2 },
			wantFields: []string{"arithmetic.weights_per_gpu_bytes", "arithmetic.usable_per_gpu_bytes"},
		},
		{
			name:       "perturbed tp",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Config.TP = 8 },
			wantFields: []string{"arithmetic.gpus_total", "arithmetic.kv_bytes_per_token_per_gpu", "arithmetic.kv_formula"},
		},
		{
			name:       "perturbed built-in gpu vram",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.GPU.VRAMGB = 95 },
			wantFields: []string{"gpu.vram_gb", "arithmetic.vram_bytes_per_gpu"},
		},
		{
			name:       "perturbed verdict",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Verdict = VerdictTight },
			wantFields: []string{"verdict"},
		},
		{
			name:       "dropped warning",
			base:       func() Receipt { return ada },
			mutate:     func(r *Receipt) { r.Warnings = r.Warnings[1:] },
			wantFields: []string{"warnings"},
		},
		{
			name:       "perturbed kv rank",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Facts.KV.KVLoraRank = 65 },
			wantFields: []string{"arithmetic.kv_bytes_per_token_per_gpu", "arithmetic.kv_formula"},
		},
		{
			name:       "perturbed max_num_seqs",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Config.MaxNumSeqs = 64 },
			wantFields: []string{"arithmetic.reserved_state_bytes"},
		},
		{
			name:       "dropped checklist entry",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.WitnessChecklist = r.WitnessChecklist[:3] },
			wantFields: []string{"witness_checklist"},
		},
		{
			name:       "perturbed observed_at",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.ObservedAt = "2026-01-01" },
			wantFields: []string{"observed_at"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mutated := tc.base()
			tc.mutate(&mutated)
			doc := Validate(mutated)
			if doc.OK {
				t.Fatal("validate = OK, want a field-by-field mismatch")
			}
			for _, want := range tc.wantFields {
				found := false
				for _, d := range doc.Diffs {
					if d.Field == want {
						found = true
					}
				}
				if !found {
					t.Fatalf("diffs %+v: no field %q named", doc.Diffs, want)
				}
			}
		})
	}
}

// TestValidateMDRendersDiff checks the md validate output names fields.
func TestValidateMDRendersDiff(t *testing.T) {
	facts := loadTinyFacts(t)
	receipt, err := Plan(facts, testGPU(t, "rtx-pro-6000-blackwell-96gb"),
		Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	receipt.Arithmetic.MaxCachedTokens += 7
	md := RenderValidateMD(Validate(receipt))
	if !strings.Contains(md, "**MISMATCH**") || !strings.Contains(md, "arithmetic.max_cached_tokens") {
		t.Fatalf("validate md missing mismatch markers:\n%s", md)
	}

	okDoc := Validate(receipt)
	okDoc.OK = true
	okDoc.Diffs = nil
	if md := RenderValidateMD(okDoc); !strings.Contains(md, "**OK**") {
		t.Fatalf("validate md missing OK marker:\n%s", md)
	}
}

// TestLoadErrors: missing file, bad JSON, wrong schema, unknown profile key,
// and a custom @file profile all behave as documented.
func TestLoadErrors(t *testing.T) {
	if _, err := LoadFacts(filepath.Join("testdata", "does-not-exist.json")); err == nil {
		t.Fatal("LoadFacts(missing) = nil error")
	}
	if _, err := LoadReceipt(filepath.Join("testdata", "does-not-exist.json")); err == nil {
		t.Fatal("LoadReceipt(missing) = nil error")
	}

	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte("{not json"), 0o644); err != nil {
		t.Fatalf("write bad.json: %v", err)
	}
	if _, err := LoadFacts(bad); err == nil {
		t.Fatal("LoadFacts(bad json) = nil error")
	}

	wrongSchema := filepath.Join(dir, "wrong-schema.json")
	if err := os.WriteFile(wrongSchema, []byte(`{"schema":"fak.serve-receipt.facts/0"}`), 0o644); err != nil {
		t.Fatalf("write wrong-schema.json: %v", err)
	}
	if _, err := LoadFacts(wrongSchema); err == nil || !strings.Contains(err.Error(), FactsSchema) {
		t.Fatalf("LoadFacts(wrong schema) = %v, want an error naming %q", err, FactsSchema)
	}

	if _, err := LoadGPUProfile("no-such-gpu"); err == nil || !strings.Contains(err.Error(), "@file.json") {
		t.Fatalf("LoadGPUProfile(unknown) = %v, want an error pointing at @file.json", err)
	}
	if _, err := LoadGPUProfile("@does-not-exist.json"); err == nil {
		t.Fatal("LoadGPUProfile(@missing) = nil error")
	}

	custom := filepath.Join(dir, "my-box.json")
	if err := os.WriteFile(custom, []byte(`{"key":"","vram_gb":64,"sm":"12.0","fp4_native":true}`), 0o644); err != nil {
		t.Fatalf("write custom profile: %v", err)
	}
	p, err := LoadGPUProfile("@" + custom)
	if err != nil {
		t.Fatalf("LoadGPUProfile(@custom): %v", err)
	}
	if p.Key != "my-box" { // key defaulted from the filename
		t.Fatalf("custom profile key = %q, want my-box", p.Key)
	}
	noVram := filepath.Join(dir, "no-vram.json")
	if err := os.WriteFile(noVram, []byte(`{"key":"broken","vram_gb":0,"sm":"12.0"}`), 0o644); err != nil {
		t.Fatalf("write no-vram.json: %v", err)
	}
	if _, err := LoadGPUProfile("@" + noVram); err == nil {
		t.Fatal("LoadGPUProfile(vram_gb=0) = nil error, want a shape error")
	}
}

// TestRenderGolden pins the tiny-fixture receipt's md and json rendering.
// Regenerate with SERVERECEIPT_UPDATE_GOLDENS=1 (deterministic: no wall-clock
// in the receipt).
func TestRenderGolden(t *testing.T) {
	facts := loadTinyFacts(t)
	receipt, err := Plan(facts, testGPU(t, "rtx-pro-6000-blackwell-96gb"),
		Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	goldens := []struct {
		path    string
		content []byte
	}{
		{filepath.Join("testdata", "tiny.plan.md.golden"), []byte(RenderMD(receipt))},
		{filepath.Join("testdata", "tiny.plan.json.golden"), mustRenderJSON(t, receipt)},
	}
	if os.Getenv("SERVERECEIPT_UPDATE_GOLDENS") == "1" {
		for _, g := range goldens {
			if err := os.WriteFile(g.path, g.content, 0o644); err != nil {
				t.Fatalf("update golden %s: %v", g.path, err)
			}
		}
		t.Log("goldens regenerated")
		return
	}
	for _, g := range goldens {
		want, err := os.ReadFile(g.path)
		if err != nil {
			t.Fatalf("read golden %s (regenerate with SERVERECEIPT_UPDATE_GOLDENS=1): %v", g.path, err)
		}
		if string(want) != string(g.content) {
			t.Fatalf("golden %s drifted:\n--- want ---\n%s\n--- got ---\n%s", g.path, want, g.content)
		}
	}
}

func mustRenderJSON(t *testing.T, r Receipt) []byte {
	t.Helper()
	raw, err := RenderJSON(r)
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}
	return raw
}
