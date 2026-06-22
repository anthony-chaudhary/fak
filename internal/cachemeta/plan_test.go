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

func TestClusterPrecisionIsMeasurable(t *testing.T) {
	// 3 of 4 members share the canonical intent => precision 0.75.
	c := IntentCluster{
		CanonicalIntent: "book_flight",
		Members: []ClusterMember{
			{IntentDigest: "a", TrueIntent: "book_flight"},
			{IntentDigest: "b", TrueIntent: "book_flight"},
			{IntentDigest: "c", TrueIntent: "book_hotel"}, // false grouping
			{IntentDigest: "d", TrueIntent: "book_flight"},
		},
	}
	if got := ClusterPrecision(c); got != 0.75 {
		t.Fatalf("ClusterPrecision = %v, want 0.75", got)
	}
	// An empty cluster scores 0 (never act on an unobserved key).
	if got := ClusterPrecision(IntentCluster{CanonicalIntent: "x"}); got != 0 {
		t.Fatalf("empty cluster precision = %v, want 0", got)
	}
}

func TestIntentKeyFromLooseClusterAbstains(t *testing.T) {
	// A loose cluster (precision 0.75) below a 0.95 threshold must abstain, even
	// on an exact digest match — the metric, not an asserted float, drives it.
	loose := IntentCluster{
		CanonicalIntent: "book_flight",
		Members: []ClusterMember{
			{IntentDigest: "a", TrueIntent: "book_flight"},
			{IntentDigest: "b", TrueIntent: "book_flight"},
			{IntentDigest: "c", TrueIntent: "book_hotel"},
			{IntentDigest: "d", TrueIntent: "book_flight"},
		},
	}
	k := IntentKeyFromCluster("intent-1", loose, 0.95, IntentKey{})
	if k.Precision != 0.75 {
		t.Fatalf("minted key precision = %v, want measured 0.75", k.Precision)
	}
	v := IntentKeyLookup(IntentCacheRequest{IntentDigest: "intent-1"}, k)
	if v.Kind != LookupMiss || v.Reason != ReasonApproxFault {
		t.Fatalf("loose cluster must abstain MISS(approximate_fault), got %+v", v)
	}
}

func TestIntentKeyFromTightClusterIsAdvisoryHit(t *testing.T) {
	// A tight cluster (precision 1.0) at/above threshold HITs — still advisory.
	tight := IntentCluster{
		CanonicalIntent: "book_flight",
		Members: []ClusterMember{
			{IntentDigest: "a", TrueIntent: "book_flight"},
			{IntentDigest: "b", TrueIntent: "book_flight"},
		},
	}
	k := IntentKeyFromCluster("intent-2", tight, 0.95, IntentKey{})
	v := IntentKeyLookup(IntentCacheRequest{IntentDigest: "intent-2"}, k)
	if v.Kind != LookupHit {
		t.Fatalf("tight cluster exact match should HIT, got %s", v.Kind)
	}
	if v.Meta["advisory_only"] != "true" || v.Entry.Security.AdmissionVerdict != AdmissionDefer {
		t.Fatalf("intent HIT must stay advisory-only (no direct effect): %+v", v)
	}
}
