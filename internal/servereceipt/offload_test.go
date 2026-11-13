package servereceipt

import (
	"encoding/json"
	"math"
	"path/filepath"
	"strings"
	"testing"
)

// loadKimiOffloadFacts loads the committed Kimi offload-model fixture (issue
// #14438): the examples/ facts plus kv_variants and the routed-expert fields.
func loadKimiOffloadFacts(t *testing.T) Facts {
	t.Helper()
	f, err := LoadFacts(filepath.Join("testdata", "kimi-k31-pro.facts.json"))
	if err != nil {
		t.Fatalf("LoadFacts(testdata/kimi-k31-pro.facts.json): %v", err)
	}
	return f
}

// TestKVVariantBytes pins all three built-in KV variants on the Kimi shape,
// via BOTH resolution paths: the facts.kv_variants map and the derived
// fallback (map stripped). The two paths must agree byte-for-byte.
func TestKVVariantBytes(t *testing.T) {
	facts := loadKimiOffloadFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")

	want := map[string]int64{
		KVVariantFP8:            16_992, // 24 × (576 + 128 + 4) — the v1 constant
		KVVariantNVFP4:          11_616, // 24 × (352 + 132 + 0)
		KVVariantNVFP4MXFP4Keys: 10_080, // 24 × (352 + 68 + 0)
	}
	for _, name := range KnownKVVariantNames {
		t.Run("facts_map/"+name, func(t *testing.T) {
			receipt, err := Plan(facts, gpu, Config{TP: 8, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2, KVVariant: name})
			if err != nil {
				t.Fatalf("Plan(%s): %v", name, err)
			}
			if got := receipt.Arithmetic.KVBytesPerTokenPerGPU; got != want[name] {
				t.Fatalf("kv/token[%s] = %d, want %d", name, got, want[name])
			}
			if got := receipt.Arithmetic.KVVariant; got != name {
				t.Fatalf("kv_variant = %q, want %q", got, name)
			}
			if receipt.Arithmetic.KVVariantFormula == "" {
				t.Fatalf("kv_variant_formula[%s] empty, want the per-layer derivation", name)
			}
		})
	}

	// Derived fallback: same bytes with the facts map stripped — the built-ins
	// must reproduce the fixture's numbers from the base kv fields alone.
	derived := facts
	derived.KVVariants = nil
	for _, name := range KnownKVVariantNames {
		t.Run("derived/"+name, func(t *testing.T) {
			receipt, err := Plan(derived, gpu, Config{TP: 8, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2, KVVariant: name})
			if err != nil {
				t.Fatalf("Plan(%s): %v", name, err)
			}
			if got := receipt.Arithmetic.KVBytesPerTokenPerGPU; got != want[name] {
				t.Fatalf("derived kv/token[%s] = %d, want %d", name, got, want[name])
			}
			// fp8 keeps the v1 KVFormula rendering byte-identical; the other
			// variants carry their per-layer derivation.
			if name == KVVariantFP8 && receipt.Arithmetic.KVVariantFormula != "" {
				t.Fatalf("derived kv_variant_formula[fp8] = %q, want empty (v1 rendering)", receipt.Arithmetic.KVVariantFormula)
			}
			if name != KVVariantFP8 && receipt.Arithmetic.KVVariantFormula == "" {
				t.Fatalf("derived kv_variant_formula[%s] empty, want the per-layer derivation", name)
			}
		})
	}

	// Default variant is fp8: an unset KVVariant must plan identically to the
	// explicit fp8 selection (this is what keeps v1 receipts recomputing).
	def, err := Plan(facts, gpu, Config{TP: 8, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan(default variant): %v", err)
	}
	if def.Arithmetic.KVVariant != KVVariantFP8 || def.Arithmetic.KVBytesPerTokenPerGPU != want[KVVariantFP8] {
		t.Fatalf("default variant = %s/%d, want fp8/%d", def.Arithmetic.KVVariant, def.Arithmetic.KVBytesPerTokenPerGPU, want[KVVariantFP8])
	}
}

// TestOffloadS8Anchor pins the receipt against the S8 production-runtime
// anchor: 16×RTX PRO 6000 at util 0.974, 32 GiB host pool at usability 0.5,
// dcp 16 → ~1.02M physical FP8 KV tokens observed (AnchorCalibration). The
// plan's floor((device_pool + usable_host) / (kv/dcp)) must land within 5%.
func TestOffloadS8Anchor(t *testing.T) {
	facts := loadKimiOffloadFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")
	receipt, err := Plan(facts, gpu, Config{
		TP: 16, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.974, ActivationReserveGB: 2,
		HostKVGiB: 32, HostPoolUsability: 0.5, DCP: 16,
	})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	ar := receipt.Arithmetic

	// Exact intermediates (hand-checked): weights/gpu, the 0.974 budget, the
	// 78.8 MB usable slice, a clamped-to-zero device pool, and the per-rank
	// host share.
	if ar.WeightsPerGPUBytes != 100_320_352_852 {
		t.Fatalf("weights_per_gpu = %d, want 100320352852", ar.WeightsPerGPUBytes)
	}
	if ar.UtilBudgetBytes != 100_399_155_511 {
		t.Fatalf("util_budget = %d, want 100399155511", ar.UtilBudgetBytes)
	}
	if ar.UsablePerGPUBytes != 78_802_659 {
		t.Fatalf("usable_per_gpu = %d, want 78802659", ar.UsablePerGPUBytes)
	}
	if ar.DevicePoolBytes != 0 {
		t.Fatalf("device_pool = %d, want 0 (weights+state+reserve fill the card)", ar.DevicePoolBytes)
	}
	if ar.UsableHostBytes != 1_073_741_824 { // 32 GiB × 0.5 ÷ dcp 16
		t.Fatalf("usable_host = %d, want 1073741824", ar.UsableHostBytes)
	}
	if ar.KVEffectivePerTokenGPU != 1_062 { // 16,992 ÷ dcp 16
		t.Fatalf("kv_effective = %d, want 1062", ar.KVEffectivePerTokenGPU)
	}

	// The anchor band: ±5% of the observed ~1,010,000 physical tokens.
	const anchorTokens = 1_010_000
	if math.Abs(float64(ar.PhysicalTokens-anchorTokens))/float64(anchorTokens) > 0.05 {
		t.Fatalf("physical_tokens = %d, want within ±5%% of %d (anchor: %s)", ar.PhysicalTokens, anchorTokens, AnchorCalibration)
	}
	if ar.PhysicalTokens != 1_011_056 { // floor(1,073,741,824 / 1,062)
		t.Fatalf("physical_tokens = %d, want exactly 1011056", ar.PhysicalTokens)
	}

	if receipt.Verdict != VerdictOffloadRequired {
		t.Fatalf("verdict = %s (%s), want OFFLOAD_REQUIRED", receipt.Verdict, receipt.Reason)
	}
	if !strings.Contains(receipt.Reason, "host pool carries the cache") {
		t.Fatalf("reason = %q, want the host-carries-cache cause", receipt.Reason)
	}
}

// TestExpertOffloadRejection: on a weights-INFEASIBLE layout the routed-expert
// fields turn "can't we just stream experts from host?" into arithmetic — the
// per-forward union read is ~17.5 s at 50 GB/s, so the warning rejects expert
// offload and keeps KV offload as the only viable surface.
func TestExpertOffloadRejection(t *testing.T) {
	facts := loadKimiOffloadFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")
	receipt, err := Plan(facts, gpu, Config{TP: 8, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if receipt.Verdict != VerdictInfeasible {
		t.Fatalf("verdict = %s, want INFEASIBLE (weights gate dominates)", receipt.Verdict)
	}
	ar := receipt.Arithmetic
	// bytes/expert = 1,531,000,000,000 ÷ 896 (floor)
	if ar.ExpertBytesPerExpert != 1_708_705_357 {
		t.Fatalf("expert_bytes_per_expert = %d, want 1708705357", ar.ExpertBytesPerExpert)
	}
	// union = min(896 routed, 32 seqs × 16 per token) = 512
	if ar.ExpertUnionExperts != 512 {
		t.Fatalf("expert_union_experts = %d, want 512", ar.ExpertUnionExperts)
	}
	if ar.ExpertUnionBytes != 874_857_142_784 {
		t.Fatalf("expert_union_bytes = %d, want 874857142784", ar.ExpertUnionBytes)
	}
	// 874,857,142,784 B ÷ 50 GB/s ≈ 17.5 s/forward (±10% band around 17.5)
	if ar.ExpertSecondsPerForward < 15.75 || ar.ExpertSecondsPerForward > 19.25 {
		t.Fatalf("expert_seconds_per_forward = %g, want in [15.75, 19.25]", ar.ExpertSecondsPerForward)
	}
	if ar.ExpertEstimated {
		t.Fatal("expert_estimated = true, want false (facts carry expert_bytes_packed)")
	}
	wantWarn := "weight/expert host offload rejected: ~17.5 s/forward (union of ~512 experts × 1.71 GB) — KV offload only"
	found := false
	for _, w := range receipt.Warnings {
		if w == wantWarn {
			found = true
		}
	}
	if !found {
		t.Fatalf("warnings %v: want the exact expert-rejection warning %q", receipt.Warnings, wantWarn)
	}

	// Estimated path: strip the measured expert bytes → bytes/expert falls
	// back to the weights share and the receipt says so.
	est := facts
	est.ExpertBytesPacked = 0
	est.ExpertScaleBytes = 0
	estReceipt, err := Plan(est, gpu, Config{TP: 8, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan(estimated): %v", err)
	}
	if got := estReceipt.Arithmetic.ExpertBytesPerExpert; got != 1_791_434_872 { // floor(1,605,125,645,632 / 896)
		t.Fatalf("estimated bytes/expert = %d, want 1791434872", got)
	}
	if !estReceipt.Arithmetic.ExpertEstimated {
		t.Fatal("expert_estimated = false, want true when facts carry no expert bytes")
	}

	// No expert fields in facts → no rejection warning (v1 behavior).
	v1 := facts
	v1.RoutedExperts, v1.ExpertsPerToken = 0, 0
	v1Receipt, err := Plan(v1, gpu, Config{TP: 8, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan(v1 facts): %v", err)
	}
	for _, w := range v1Receipt.Warnings {
		if strings.Contains(w, "expert host offload rejected") {
			t.Fatalf("v1 facts produced the expert warning %q; want none", w)
		}
	}
}

// TestOffloadVerdictTransitions: the verdict ladder with the host pool in
// play — weights-INFEASIBLE still dominates; a device shortfall the host
// covers becomes OFFLOAD_REQUIRED (exit 0, marked); TIGHT/OK otherwise.
func TestOffloadVerdictTransitions(t *testing.T) {
	kimi := loadKimiOffloadFacts(t)
	tiny := loadTinyFacts(t)
	blackwell := testGPU(t, "rtx-pro-6000-blackwell-96gb")

	tests := []struct {
		name string
		fact Facts
		cfg  Config
		want Verdict
	}{
		{
			name: "weights_infeasible_dominates_even_with_host",
			fact: kimi,
			cfg:  Config{TP: 16, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2, HostKVGiB: 128},
			want: VerdictInfeasible,
		},
		{
			name: "no_len_device_exhausted_host_covers",
			fact: kimi,
			cfg:  Config{TP: 16, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.974, ActivationReserveGB: 2, HostKVGiB: 32, DCP: 16},
			want: VerdictOffloadRequired,
		},
		{
			name: "len_short_device_host_covers",
			fact: tiny,
			cfg:  Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2, MaxModelLen: 200_000_000, HostKVGiB: 128},
			want: VerdictOffloadRequired,
		},
		{
			name: "len_short_device_host_absent",
			fact: tiny,
			cfg:  Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2, MaxModelLen: 200_000_000},
			want: VerdictTight,
		},
		{
			name: "host_present_but_not_needed",
			fact: tiny,
			cfg:  Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2, HostKVGiB: 128},
			want: VerdictOK,
		},
		{
			name: "tp8_pp3_ok_unchanged",
			fact: kimi,
			cfg:  Config{TP: 8, PP: 3, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2},
			want: VerdictOK,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			receipt, err := Plan(tc.fact, blackwell, tc.cfg)
			if err != nil {
				t.Fatalf("Plan: %v", err)
			}
			if receipt.Verdict != tc.want {
				t.Fatalf("verdict = %s (%s), want %s", receipt.Verdict, receipt.Reason, tc.want)
			}
		})
	}

	// The len-short OFFLOAD_REQUIRED reason names the covered share.
	receipt, err := Plan(tiny, blackwell, Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2, MaxModelLen: 200_000_000, HostKVGiB: 128})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if !strings.Contains(receipt.Reason, "device KV budget covers 158,697,006 of the requested max-model-len 200,000,000") {
		t.Fatalf("reason = %q, want the covered-share wording", receipt.Reason)
	}
}

// TestValidateBackCompatV1: a receipt produced by the v1 code (no offload
// fields anywhere) must still validate OK — the new arithmetic checks are
// gated on the physical_formula marker the old receipts never carried.
func TestValidateBackCompatV1(t *testing.T) {
	facts := loadTinyFacts(t)
	receipt, err := Plan(facts, testGPU(t, "rtx-pro-6000-blackwell-96gb"),
		Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}

	// Strip every new arithmetic field via a JSON map round-trip — exactly
	// what a v1 receipt on disk looks like.
	raw, err := RenderJSON(receipt)
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("map round-trip: %v", err)
	}
	ar := doc["arithmetic"].(map[string]any)
	for _, k := range []string{
		"kv_variant", "kv_variant_formula", "dcp", "kv_effective_per_token_per_gpu",
		"device_pool_bytes", "host_pool_bytes", "host_pool_usability", "usable_host_bytes",
		"host_bandwidth_gbs", "physical_tokens", "physical_formula",
	} {
		delete(ar, k)
	}
	v1Raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("re-marshal: %v", err)
	}
	var v1 Receipt
	if err := json.Unmarshal(v1Raw, &v1); err != nil {
		t.Fatalf("v1 unmarshal: %v", err)
	}
	if v1.Arithmetic.PhysicalFormula != "" {
		t.Fatalf("strip failed: physical_formula = %q", v1.Arithmetic.PhysicalFormula)
	}
	docv := Validate(v1)
	if !docv.OK {
		t.Fatalf("v1-shaped receipt failed validate: %+v", docv.Diffs)
	}
}

// TestValidateOffloadFields: a new receipt round-trips clean, and mutating
// any offload field (or the curve block) is refused with the field named.
func TestValidateOffloadFields(t *testing.T) {
	facts := loadKimiOffloadFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")
	receipt, err := Plan(facts, gpu, Config{
		TP: 16, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.974, ActivationReserveGB: 2,
		HostKVGiB: 32, HostPoolUsability: 0.5, DCP: 16, Curve: true,
	})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	raw, err := RenderJSON(receipt)
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}
	var clean Receipt
	if err := json.Unmarshal(raw, &clean); err != nil {
		t.Fatalf("round-trip: %v", err)
	}
	if doc := Validate(clean); !doc.OK {
		t.Fatalf("clean offload receipt failed validate: %+v", doc.Diffs)
	}
	if clean.Curve == nil {
		t.Fatal("curve missing from the round-tripped receipt")
	}

	// A weights-INFEASIBLE receipt carries the expert fields; expert-field
	// mutations run against this one.
	infeasible, err := Plan(facts, gpu, Config{TP: 8, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan(infeasible): %v", err)
	}
	if infeasible.Arithmetic.ExpertUnionBytes == 0 {
		t.Fatal("infeasible receipt carries no expert arithmetic; fixture/config broken")
	}

	tests := []struct {
		name       string
		base       func() Receipt
		mutate     func(*Receipt)
		wantFields []string
	}{
		{
			name:       "perturbed physical_tokens",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Arithmetic.PhysicalTokens++ },
			wantFields: []string{"arithmetic.physical_tokens"},
		},
		{
			name:       "perturbed physical_formula",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Arithmetic.PhysicalFormula = "floor(something else)" },
			wantFields: []string{"arithmetic.physical_formula"},
		},
		{
			name:       "perturbed config host pool",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Config.HostKVGiB += 16 },
			wantFields: []string{"arithmetic.host_pool_bytes", "arithmetic.usable_host_bytes", "arithmetic.physical_tokens"},
		},
		{
			name:       "perturbed config usability",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Config.HostPoolUsability = 0.9 },
			wantFields: []string{"arithmetic.host_pool_usability", "arithmetic.usable_host_bytes"},
		},
		{
			name:       "perturbed config kv_variant",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Config.KVVariant = KVVariantNVFP4 },
			wantFields: []string{"arithmetic.kv_variant", "arithmetic.kv_bytes_per_token_per_gpu"},
		},
		{
			name:       "perturbed config dcp",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Config.DCP = 8 },
			wantFields: []string{"arithmetic.dcp", "arithmetic.kv_effective_per_token_per_gpu"},
		},
		{
			name:       "perturbed expert_union_bytes",
			base:       func() Receipt { return infeasible },
			mutate:     func(r *Receipt) { r.Arithmetic.ExpertUnionBytes *= 2 },
			wantFields: []string{"arithmetic.expert_union_bytes"},
		},
		{
			name:       "dropped curve block",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Curve = nil },
			wantFields: []string{"curve.present"},
		},
		{
			name:       "perturbed curve capacity row",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Curve.Capacity[2].PhysicalTokens++ },
			wantFields: []string{"curve.capacity[2].physical_tokens"},
		},
		{
			name:       "perturbed curve miss row",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Curve.MissLatency[1].ExtraLatencyUS *= 3 },
			wantFields: []string{"curve.miss_latency[1].extra_latency_us"},
		},
		{
			name:       "forged curve anchor",
			base:       func() Receipt { return clean },
			mutate:     func(r *Receipt) { r.Curve.Anchor = "made up" },
			wantFields: []string{"curve.anchor"},
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

// TestCurveBlock: the --curve sweep — capacity rows monotone in the host
// pool, the 32 GiB row reproducing the receipt's own physical_tokens, the
// miss rows scaling linearly in the miss rate, and the anchor cited verbatim
// in the md rendering.
func TestCurveBlock(t *testing.T) {
	facts := loadKimiOffloadFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")
	receipt, err := Plan(facts, gpu, Config{
		TP: 16, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.974, ActivationReserveGB: 2,
		HostKVGiB: 32, HostPoolUsability: 0.5, DCP: 16, Curve: true,
	})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	c := receipt.Curve
	if c == nil {
		t.Fatal("curve missing with Curve: true")
	}
	if c.Anchor != AnchorCalibration || c.PenaltyNote != CurvePenaltyNote {
		t.Fatalf("anchor/penalty note not the fixed literals: %q / %q", c.Anchor, c.PenaltyNote)
	}
	if len(c.Capacity) != 5 {
		t.Fatalf("capacity rows = %d, want 5", len(c.Capacity))
	}
	prev := int64(-1)
	for i, row := range c.Capacity {
		if row.PhysicalTokens < prev {
			t.Fatalf("capacity row %d (%g GiB → %d) below previous %d: sweep must be monotone",
				i, row.HostGiB, row.PhysicalTokens, prev)
		}
		prev = row.PhysicalTokens
	}
	if c.Capacity[0].PhysicalTokens != 0 {
		t.Fatalf("host 0 row = %d, want 0 (device pool is clamped to 0 on this layout)", c.Capacity[0].PhysicalTokens)
	}
	if got := c.Capacity[2]; got.HostGiB != 32 || got.PhysicalTokens != receipt.Arithmetic.PhysicalTokens {
		t.Fatalf("32 GiB row = %+v, want the receipt's own physical_tokens %d", got, receipt.Arithmetic.PhysicalTokens)
	}
	if c.Capacity[4].HostGiB != 128 || c.Capacity[4].PhysicalTokens != 4_044_225 { // floor((0 + 128×0.5×2^30/16) / 1062)
		t.Fatalf("128 GiB row = %+v, want 4044225", c.Capacity[4])
	}
	if len(c.MissLatency) != 3 {
		t.Fatalf("miss rows = %d, want 3", len(c.MissLatency))
	}
	// m × (kv/dcp) bytes ÷ (50 GB/s) → µs: 0.25 × 1062 / 5e10 × 1e6 = 0.00531
	wantMiss := []float64{0.001062, 0.002124, 0.00531}
	for i, row := range c.MissLatency {
		if math.Abs(row.ExtraLatencyUS-wantMiss[i]) > 1e-9 {
			t.Fatalf("miss row %d (m=%g) = %g µs, want %g", i, row.MissRate, row.ExtraLatencyUS, wantMiss[i])
		}
	}

	// No --curve → no block.
	plain, err := Plan(facts, gpu, Config{TP: 16, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.974, ActivationReserveGB: 2, HostKVGiB: 32, DCP: 16})
	if err != nil {
		t.Fatalf("Plan(plain): %v", err)
	}
	if plain.Curve != nil {
		t.Fatal("curve attached without Curve: true")
	}

	// The md rendering carries the section header and the anchor verbatim.
	md := RenderMD(receipt)
	if !strings.Contains(md, "## Performance curve (ESTIMATE)") {
		t.Fatalf("md missing the curve section header:\n%s", md)
	}
	if !strings.Contains(md, AnchorCalibration) || !strings.Contains(md, CurvePenaltyNote) {
		t.Fatalf("md missing the verbatim anchor/penalty note:\n%s", md)
	}
	if !strings.Contains(md, "OFFLOAD_REQUIRED: the device KV pool alone does not cover the request") {
		t.Fatalf("md missing the OFFLOAD_REQUIRED note:\n%s", md)
	}
}

// TestDCPWarning: the measured DCP-slower warning fires from dcp 8 up.
func TestDCPWarning(t *testing.T) {
	facts := loadTinyFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")
	hasDCPWarn := func(dcp int) bool {
		receipt, err := Plan(facts, gpu, Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2, DCP: dcp})
		if err != nil {
			t.Fatalf("Plan(dcp %d): %v", dcp, err)
		}
		for _, w := range receipt.Warnings {
			if w == WarnDCPSlow {
				return true
			}
		}
		return false
	}
	if !hasDCPWarn(8) {
		t.Fatal("dcp 8: want the DCP-slower warning")
	}
	if !hasDCPWarn(16) {
		t.Fatal("dcp 16: want the DCP-slower warning")
	}
	if hasDCPWarn(4) {
		t.Fatal("dcp 4: want no DCP warning (fires from 8 up)")
	}
}

// TestOffloadInputErrors: negative/out-of-range/unknown offload inputs refuse
// with usage-class errors (exit 2), distinct from a computed verdict.
func TestOffloadInputErrors(t *testing.T) {
	facts := loadTinyFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")
	base := Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2}
	tests := []struct {
		name   string
		mutate func(*Config)
		want   string
	}{
		{"negative host pool", func(c *Config) { c.HostKVGiB = -1 }, "host_kv_gib"},
		{"usability above 1", func(c *Config) { c.HostPoolUsability = 1.5 }, "host_pool_usability"},
		{"negative bandwidth", func(c *Config) { c.HostBandwidthGBs = -5 }, "host_bandwidth_gbs"},
		{"negative dcp", func(c *Config) { c.DCP = -2 }, "dcp"},
		{"unknown kv variant", func(c *Config) { c.KVVariant = "q4_kl" }, "unknown"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			tc.mutate(&cfg)
			_, err := Plan(facts, gpu, cfg)
			if err == nil {
				t.Fatal("Plan = nil error, want a usage-class input error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %q, want it to name %q", err, tc.want)
			}
		})
	}
}
