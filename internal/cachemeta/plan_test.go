package cachemeta

import "testing"

func samplePlanTemplate() PlanTemplate {
	return PlanTemplate{
		TaskClass:          "book_flight",
		ParamsDigest:       "params-1",
		PlanSchemaDigest:   "schema-1",
		ToolManifestDigest: "tools-1",
		PolicyVersion:      "policy-v1",
		Steps:              []string{"search_flights", "book"},
	}
}

func TestPlanTemplateExactMatchWithWitnessIsAdvisoryHit(t *testing.T) {
	tpl := samplePlanTemplate()
	req := PlanCacheRequest{
		TaskClass: tpl.TaskClass, ParamsDigest: tpl.ParamsDigest,
		PlanSchemaDigest: tpl.PlanSchemaDigest, ToolManifestDigest: tpl.ToolManifestDigest,
		PolicyVersion: tpl.PolicyVersion, StateWitness: "state:ok",
	}
	v := PlanTemplateLookup(req, tpl)
	if v.Kind != LookupHit {
		t.Fatalf("exact match + witness should HIT, got %s", v.Kind)
	}
	// Refusal rule 5: a cached plan is never an execution permit.
	if v.Meta["must_reenter_plancfi"] != "true" {
		t.Fatalf("plan HIT must demand re-entry to plancfi")
	}
	if v.Entry.Security.AdmissionVerdict != AdmissionDefer {
		t.Fatalf("plan-template hit must defer admission (not execute), got %q", v.Entry.Security.AdmissionVerdict)
	}
	if v.Entry.Plane != PlanePlanTemplate || v.Entry.ID.MediaType != MediaPlanTemplate {
		t.Fatalf("bad plan-template identity: %+v", v.Entry)
	}
}

func TestPlanTemplateExactMatchWithoutWitnessRevalidates(t *testing.T) {
	tpl := samplePlanTemplate()
	req := PlanCacheRequest{
		TaskClass: tpl.TaskClass, ParamsDigest: tpl.ParamsDigest,
		PlanSchemaDigest: tpl.PlanSchemaDigest, ToolManifestDigest: tpl.ToolManifestDigest,
		PolicyVersion: tpl.PolicyVersion, StateWitness: "", // state/witness missing
	}
	v := PlanTemplateLookup(req, tpl)
	if v.Kind != LookupRevalidate || v.Reason != ReasonStale {
		t.Fatalf("missing witness should REVALIDATE(stale), got %+v", v)
	}
}

func TestPlanTemplateBindingMismatchMisses(t *testing.T) {
	tpl := samplePlanTemplate()
	req := PlanCacheRequest{
		TaskClass: tpl.TaskClass, ParamsDigest: tpl.ParamsDigest,
		PlanSchemaDigest: tpl.PlanSchemaDigest, ToolManifestDigest: "DIFFERENT", // mismatch
		PolicyVersion: tpl.PolicyVersion, StateWitness: "state:ok",
	}
	v := PlanTemplateLookup(req, tpl)
	if v.Kind != LookupMiss {
		t.Fatalf("tool-manifest mismatch should MISS, got %s", v.Kind)
	}
}

func TestIntentKeyExactHighPrecisionIsAdvisoryHit(t *testing.T) {
	k := IntentKey{IntentDigest: "intent-1", PrecisionThreshold: 0.95, Precision: 0.99}
	v := IntentKeyLookup(IntentCacheRequest{IntentDigest: "intent-1"}, k)
	if v.Kind != LookupHit {
		t.Fatalf("exact digest + sufficient precision should HIT, got %s", v.Kind)
	}
	// Refusal rule 4: a semantic cache may suggest; it may not bypass adjudication.
	if v.Meta["advisory_only"] != "true" || v.Entry.Security.AdmissionVerdict != AdmissionDefer {
		t.Fatalf("intent HIT must be advisory-only (no direct effect): %+v", v)
	}
	if v.Entry.Plane != PlaneSemanticIntent {
		t.Fatalf("intent entry must be on semantic_intent plane: %s", v.Entry.Plane)
	}
}

func TestIntentKeyAbstainsOnLowPrecision(t *testing.T) {
	k := IntentKey{IntentDigest: "intent-1", PrecisionThreshold: 0.95, Precision: 0.80}
	v := IntentKeyLookup(IntentCacheRequest{IntentDigest: "intent-1"}, k)
	if v.Kind != LookupMiss || v.Reason != ReasonApproxFault {
		t.Fatalf("sub-threshold precision must abstain MISS(approximate_fault), got %+v", v)
	}
}

func TestIntentKeyDigestMismatchMisses(t *testing.T) {
	k := IntentKey{IntentDigest: "intent-1", PrecisionThreshold: 0.0, Precision: 0.0}
	v := IntentKeyLookup(IntentCacheRequest{IntentDigest: "intent-2"}, k)
	if v.Kind != LookupMiss || v.Reason != ReasonAbsent {
		t.Fatalf("digest mismatch should MISS(absent), got %+v", v)
	}
}
