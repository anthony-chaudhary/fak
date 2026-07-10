package gateway

import (
	"encoding/json"
	"testing"
)

// TestDeepSeekBudgetJSONPerRowProvenance pins the #3015 acceptance gate literally:
// the `fak deepseek budget --json` equivalent — RenderJSON — must emit the required
// per-row `provenance` FIELD on every context row, for BOTH V4 tiers built from the
// same deterministic calculator (the issue's "include Pro and Flash separately").
//
// This is deliberately distinct from TestDeepSeekBudgetRenderJSONMarkdown, which only
// greps for label substrings anywhere in the blob: this test parses the emitted JSON
// and asserts each contexts[] element carries a non-empty, in-vocabulary `provenance`
// KEY (plus the section-level provenance fields), so a row that silently dropped its
// label — the exact failure the acceptance gate names — can never pass unnoticed.
func TestDeepSeekBudgetJSONPerRowProvenance(t *testing.T) {
	// The two V4 tiers, side by side, with the pinned constants the JSON must carry so
	// Flash is never inferred from Pro (#3015 grounding: 1.6T/49B Pro, 284B/13B Flash;
	// paper KV/FLOP ratios 0.10/0.27 Pro, 0.07/0.10 Flash).
	tiers := map[string]struct {
		total, active, kvRatio, flopRatio float64
	}{
		"deepseek-v4-pro":   {1.6e12, 49e9, 0.10, 0.27},
		"deepseek-v4-flash": {284e9, 13e9, 0.07, 0.10},
	}
	for model, want := range tiers {
		rep, ok := DeepSeekBudget(model, BudgetInputs{})
		if !ok {
			t.Fatalf("%s: DeepSeekBudget failed closed unexpectedly", model)
		}
		buf, err := rep.RenderJSON()
		if err != nil {
			t.Fatalf("%s: RenderJSON: %v", model, err)
		}
		var doc map[string]any
		if err := json.Unmarshal(buf, &doc); err != nil {
			t.Fatalf("%s: RenderJSON emitted invalid JSON: %v", model, err)
		}

		// The per-row provenance field the acceptance gate names: every contexts[] element
		// must carry a non-empty, in-vocabulary `provenance` key, and never WITNESSED.
		ctxs, isArr := doc["contexts"].([]any)
		if !isArr || len(ctxs) == 0 {
			t.Fatalf("%s: JSON has no contexts[] rows to label", model)
		}
		for i, raw := range ctxs {
			row, isObj := raw.(map[string]any)
			if !isObj {
				t.Fatalf("%s: contexts[%d] is not a JSON object", model, i)
			}
			label, has := row["provenance"].(string)
			if !has || label == "" {
				t.Errorf("%s: contexts[%d] (%v tokens) is missing the required per-row provenance field",
					model, i, row["context_tokens"])
				continue
			}
			if !DeepSeekBudgetProvenanceValid(DeepSeekBudgetProvenance(label)) {
				t.Errorf("%s: contexts[%d] provenance %q is outside the closed vocabulary", model, i, label)
			}
			if label == string(ProvenanceWitnessed) {
				t.Errorf("%s: contexts[%d] claims WITNESSED, but the deterministic calculator measures nothing", model, i)
			}
		}

		// The section-level provenance fields must also be present, so no whole section
		// (spec / weight_storage / compute) is emitted unlabeled.
		for _, section := range []string{"spec", "weight_storage", "compute"} {
			obj, isObj := doc[section].(map[string]any)
			if !isObj {
				t.Fatalf("%s: JSON section %q missing", model, section)
			}
			if label, _ := obj["provenance"].(string); label == "" {
				t.Errorf("%s: section %q emitted without a provenance field", model, section)
			}
		}

		// Pro and Flash are pinned to their OWN documented constants — read straight off
		// the emitted spec so a JSON that drifts from the pinned source constants reds.
		spec, _ := doc["spec"].(map[string]any)
		if got, _ := spec["total_params"].(float64); got != want.total {
			t.Errorf("%s: JSON total_params = %.0f, want %.0f", model, got, want.total)
		}
		if got, _ := spec["active_params"].(float64); got != want.active {
			t.Errorf("%s: JSON active_params = %.0f, want %.0f", model, got, want.active)
		}
		if got, _ := spec["kv_cache_ratio_vs_v32"].(float64); got != want.kvRatio {
			t.Errorf("%s: JSON kv_cache_ratio_vs_v32 = %.2f, want %.2f", model, got, want.kvRatio)
		}
		if got, _ := spec["flop_ratio_vs_v32"].(float64); got != want.flopRatio {
			t.Errorf("%s: JSON flop_ratio_vs_v32 = %.2f, want %.2f", model, got, want.flopRatio)
		}

		// The honest-refusal verdict must ride in the emitted JSON too: a deterministic
		// report never grants "1M native support" (#3015 refusal condition).
		ns, _ := doc["native_support"].(map[string]any)
		if granted, _ := ns["granted"].(bool); granted {
			t.Errorf("%s: JSON native_support.granted = true off a MODELED report — must refuse", model)
		}
	}
}
