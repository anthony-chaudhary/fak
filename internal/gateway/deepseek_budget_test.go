package gateway

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestDeepSeekBudgetConstantsPinned pins the #3015 acceptance that Pro/Flash
// configured constants come from the official sources and the table FAILS CLOSED for
// any other model id — the same discipline the price table holds.
func TestDeepSeekBudgetConstantsPinned(t *testing.T) {
	want := map[string]DeepSeekModelSpec{
		"deepseek-v4-pro": {
			Model:             "deepseek-v4-pro",
			TotalParams:       1.6e12,
			ActiveParams:      49e9,
			MaxContextTokens:  1_000_000,
			KVCacheRatioVsV32: 0.10,
			FLOPRatioVsV32:    0.27,
		},
		"deepseek-v4-flash": {
			Model:             "deepseek-v4-flash",
			TotalParams:       284e9,
			ActiveParams:      13e9,
			MaxContextTokens:  1_000_000,
			KVCacheRatioVsV32: 0.07,
			FLOPRatioVsV32:    0.10,
		},
	}
	for model, w := range want {
		spec, ok := deepSeekBudgetSpecFor(model)
		if !ok {
			t.Fatalf("no budget spec for %q — the table must carry both V4 tiers", model)
		}
		if spec.TotalParams != w.TotalParams || spec.ActiveParams != w.ActiveParams {
			t.Errorf("%s params = (%.0f total, %.0f active), want (%.0f, %.0f)",
				model, spec.TotalParams, spec.ActiveParams, w.TotalParams, w.ActiveParams)
		}
		if spec.MaxContextTokens != w.MaxContextTokens {
			t.Errorf("%s max context = %d, want %d", model, spec.MaxContextTokens, w.MaxContextTokens)
		}
		if spec.KVCacheRatioVsV32 != w.KVCacheRatioVsV32 || spec.FLOPRatioVsV32 != w.FLOPRatioVsV32 {
			t.Errorf("%s ratios = (kv %.2f, flop %.2f), want (%.2f, %.2f)",
				model, spec.KVCacheRatioVsV32, spec.FLOPRatioVsV32, w.KVCacheRatioVsV32, w.FLOPRatioVsV32)
		}
		if spec.Provenance != ProvenanceSourceDocumented {
			t.Errorf("%s spec provenance = %q, want SOURCE_DOCUMENTED", model, spec.Provenance)
		}
		if spec.RatioProvenance != ProvenancePaperClaimed {
			t.Errorf("%s ratio provenance = %q, want PAPER_CLAIMED — a paper claim is never SOURCE_DOCUMENTED", model, spec.RatioProvenance)
		}
		if len(spec.Sources) == 0 {
			t.Errorf("%s spec carries no source URL — a pinned constant must cite its source", model)
		}
	}
	// The pre-V4 aliases resolve to the flash tier.
	for _, alias := range []string{"deepseek-chat", "deepseek-reasoner"} {
		if spec, ok := deepSeekBudgetSpecFor(alias); !ok || spec.Model != "deepseek-v4-flash" {
			t.Errorf("alias %q resolved to (%q, %v), want the flash tier", alias, spec.Model, ok)
		}
	}
	// Fail closed on anything else.
	if _, ok := deepSeekBudgetSpecFor("gpt-4o"); ok {
		t.Error("deepSeekBudgetSpecFor resolved a non-DeepSeek model — the table must fail closed")
	}
	if _, ok := DeepSeekBudget("not-a-model", BudgetInputs{}); ok {
		t.Error("DeepSeekBudget resolved an unknown model — it must fail closed")
	}
}

// TestDeepSeekBudgetEveryRowLabeled is the acceptance witness that the report FAILS
// (here: is caught) if any emitted row lacks provenance. Every flattened row must
// carry a non-empty label from the closed vocabulary, and the deterministic report
// must never emit WITNESSED (there is no fak measurement behind it).
func TestDeepSeekBudgetEveryRowLabeled(t *testing.T) {
	for _, model := range []string{"deepseek-v4-pro", "deepseek-v4-flash"} {
		rep, ok := DeepSeekBudget(model, BudgetInputs{})
		if !ok {
			t.Fatalf("%s: DeepSeekBudget failed closed unexpectedly", model)
		}
		rows := rep.Rows()
		if len(rows) == 0 {
			t.Fatalf("%s: report emitted zero rows", model)
		}
		for _, row := range rows {
			if !DeepSeekBudgetProvenanceValid(row.Provenance) {
				t.Errorf("%s: row %q carries an off-vocabulary/empty provenance %q", model, row.Name, row.Provenance)
			}
			if row.Provenance == ProvenanceWitnessed {
				t.Errorf("%s: row %q claims WITNESSED, but the deterministic calculator measures nothing", model, row.Name)
			}
		}
	}
	// The closed-vocabulary predicate itself: empty and junk are rejected, known labels accepted.
	if DeepSeekBudgetProvenanceValid("") {
		t.Error("empty provenance was accepted — an unlabeled row must be rejected")
	}
	if DeepSeekBudgetProvenanceValid("GUESSED") {
		t.Error("off-vocabulary provenance was accepted — the vocabulary must be closed")
	}
	for _, ok := range []DeepSeekBudgetProvenance{ProvenanceSourceDocumented, ProvenancePaperClaimed, ProvenanceModeled, ProvenanceWitnessed} {
		if !DeepSeekBudgetProvenanceValid(ok) {
			t.Errorf("known provenance %q was rejected", ok)
		}
	}
}

// TestDeepSeekBudgetRefusesNativeSupport witnesses the honest-refusal condition: a
// deterministic report NEVER grants a "1M native support" claim, and it names the
// three missing measured witnesses instead of laundering MODELED numbers into a yes.
func TestDeepSeekBudgetRefusesNativeSupport(t *testing.T) {
	rep, ok := DeepSeekBudget("deepseek-v4-pro", BudgetInputs{V32KVBytesPerToken: 128})
	if !ok {
		t.Fatal("DeepSeekBudget(deepseek-v4-pro) failed closed")
	}
	claim := rep.NativeSupport
	if claim.Claim != "1M native support" {
		t.Errorf("claim subject = %q, want \"1M native support\"", claim.Claim)
	}
	if claim.Granted {
		t.Errorf("native-support claim GRANTED off a MODELED/PAPER_CLAIMED report — it must be refused")
	}
	if len(claim.MissingWitnesses) < 3 {
		t.Errorf("missing witnesses = %v, want at least memory/cache-layout/throughput", claim.MissingWitnesses)
	}
	if !strings.Contains(claim.Reason, "REFUSED") {
		t.Errorf("refusal reason = %q, want it to say REFUSED", claim.Reason)
	}
}

// TestDeepSeekBudgetWeightStorageBounds witnesses the FP4/FP8 mixed-precision bounds:
// the floor is the all-FP4 lower bound, strictly below the all-FP8 ceiling, MODELED.
func TestDeepSeekBudgetWeightStorageBounds(t *testing.T) {
	rep, _ := DeepSeekBudget("deepseek-v4-pro", BudgetInputs{})
	ws := rep.WeightStorage
	if ws.FloorBytes != DeepSeekV4ProTotalParams*bytesPerParamFP4 {
		t.Errorf("floor = %.3g, want total×0.5 = %.3g", ws.FloorBytes, DeepSeekV4ProTotalParams*bytesPerParamFP4)
	}
	if ws.CeilingBytes != DeepSeekV4ProTotalParams*bytesPerParamFP8 {
		t.Errorf("ceiling = %.3g, want total×1.0 = %.3g", ws.CeilingBytes, DeepSeekV4ProTotalParams*bytesPerParamFP8)
	}
	if !(ws.FloorBytes < ws.CeilingBytes) {
		t.Errorf("floor %.3g not below ceiling %.3g — the mixed layout must be bounded", ws.FloorBytes, ws.CeilingBytes)
	}
	if ws.Provenance != ProvenanceModeled {
		t.Errorf("weight storage provenance = %q, want MODELED", ws.Provenance)
	}
}

// TestDeepSeekBudgetKVBaselineSeam witnesses the seam where a witnessed baseline plugs
// in: with no baseline the KV rows are ratio-only (PAPER_CLAIMED, zero absolute); with
// a baseline supplied the absolute KV grows linearly with context and turns MODELED —
// but never WITNESSED, because ratio×baseline is still a model.
func TestDeepSeekBudgetKVBaselineSeam(t *testing.T) {
	// Default: ratio-only.
	def, _ := DeepSeekBudget("deepseek-v4-flash", BudgetInputs{})
	for _, c := range def.Contexts {
		if c.AbsoluteKVBytes != 0 {
			t.Errorf("default report emitted an absolute KV (%.3g) at %d tokens without a baseline", c.AbsoluteKVBytes, c.ContextTokens)
		}
		if c.Provenance != ProvenancePaperClaimed {
			t.Errorf("ratio-only KV row provenance = %q, want PAPER_CLAIMED", c.Provenance)
		}
		if c.KVRatioVsV32 != DeepSeekV4FlashKVRatioVsV32 {
			t.Errorf("KV ratio = %.2f, want %.2f", c.KVRatioVsV32, DeepSeekV4FlashKVRatioVsV32)
		}
	}
	// With a baseline: absolute KV, MODELED, monotonically increasing in context.
	withBase, _ := DeepSeekBudget("deepseek-v4-flash", BudgetInputs{V32KVBytesPerToken: 100})
	var prev float64 = -1
	sawAbsolute := false
	for _, c := range withBase.Contexts {
		// Compute the expected value through the same runtime float64 path (not
		// constant folding) so the check is exact rather than off by ulps.
		ratio := DeepSeekV4FlashKVRatioVsV32
		baseline := 100.0
		want := ratio * baseline * float64(c.ContextTokens)
		if rel := (c.AbsoluteKVBytes - want) / want; rel > 1e-9 || rel < -1e-9 {
			t.Errorf("absolute KV at %d = %.6g, want ratio×baseline×tokens = %.6g", c.ContextTokens, c.AbsoluteKVBytes, want)
		}
		if c.Provenance != ProvenanceModeled {
			t.Errorf("baselined KV row provenance = %q, want MODELED", c.Provenance)
		}
		if c.AbsoluteKVBytes <= prev {
			t.Errorf("absolute KV not increasing with context: %.3g then %.3g", prev, c.AbsoluteKVBytes)
		}
		prev = c.AbsoluteKVBytes
		sawAbsolute = true
	}
	if !sawAbsolute {
		t.Fatal("no context rows emitted with a baseline supplied")
	}
	// A supplied baseline must not manufacture a WITNESSED grant.
	if withBase.NativeSupport.Granted {
		t.Error("a MODELED absolute KV granted native support — a model is not a witness")
	}
}

// TestDeepSeekBudgetContextClamp witnesses that the default 4K..1M buckets are emitted
// in ascending order and no bucket exceeds the documented context ceiling.
func TestDeepSeekBudgetContextClamp(t *testing.T) {
	// 2_000_000 is above the 1,000,000 ceiling and must be dropped; the exact-ceiling
	// bucket must be kept.
	rep, _ := DeepSeekBudget("deepseek-v4-pro", BudgetInputs{Contexts: []int{4096, DeepSeekV4MaxContextTokens, 2_000_000, 32768}})
	var last int
	for _, c := range rep.Contexts {
		if c.ContextTokens > DeepSeekV4MaxContextTokens {
			t.Errorf("emitted a %d-token row above the %d ceiling", c.ContextTokens, DeepSeekV4MaxContextTokens)
		}
		if c.ContextTokens <= last {
			t.Errorf("context rows not ascending: %d then %d", last, c.ContextTokens)
		}
		last = c.ContextTokens
	}
	if len(rep.Contexts) != 3 { // 2_000_000 is dropped (> 1M ceiling)
		t.Errorf("emitted %d rows, want 3 (the over-ceiling bucket dropped)", len(rep.Contexts))
	}
	if last != DeepSeekV4MaxContextTokens {
		t.Errorf("top emitted bucket = %d, want the ceiling %d kept", last, DeepSeekV4MaxContextTokens)
	}
}

// TestDeepSeekBudgetRenderJSONMarkdown witnesses acceptance row #1: the report emits
// JSON + markdown, each carrying per-row provenance labels.
func TestDeepSeekBudgetRenderJSONMarkdown(t *testing.T) {
	rep, _ := DeepSeekBudget("deepseek-v4-pro", BudgetInputs{})

	buf, err := rep.RenderJSON()
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}
	var back DeepSeekBudgetReport
	if err := json.Unmarshal(buf, &back); err != nil {
		t.Fatalf("RenderJSON produced invalid JSON: %v", err)
	}
	if back.Model != "deepseek-v4-pro" {
		t.Errorf("round-tripped model = %q, want deepseek-v4-pro", back.Model)
	}
	js := string(buf)
	for _, label := range []string{"SOURCE_DOCUMENTED", "PAPER_CLAIMED", "MODELED"} {
		if !strings.Contains(js, label) {
			t.Errorf("JSON is missing provenance label %q", label)
		}
	}

	md := rep.RenderMarkdown()
	for _, want := range []string{"PAPER_CLAIMED", "MODELED", "REFUSED", "context tokens", deepSeekBudgetNote[:20]} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown is missing %q", want)
		}
	}
}
